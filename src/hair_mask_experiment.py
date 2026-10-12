"""Student-selected hair occlusion with a fixed exported model; no training.

First run --make-only to inspect every mask and freeze annotations before inference.
Canonical 256 RGB copies reproduce each source's original 224 model input exactly.
Only annotated hair pixels become RGB(128,128,128); all other pixels stay exact.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import unicodedata

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F

ROOT = Path.cwd().resolve()
sys.path.insert(0, str(ROOT / 'src'))
from data import CROP, MEAN, RESIZE, STD, prepare_image


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def record_path(path):
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)


def normalized(path):
    return unicodedata.normalize('NFC', str(path))


def read_config(path):
    config = json.loads(path.read_text())
    assert config['method'] == 'annotated_hair_rgb128_on_exact_input_canonical256'
    assert config['fill_rgb'] == [128,128,128]
    assert len(config['cases']) == 24
    sources = {normalized(p.relative_to(ROOT)) for p in (ROOT/'data/new_images').rglob('*')
               if p.is_file() and p.suffix.lower() in {'.jpg','.jpeg','.png'}}
    assert sources == {normalized(c['source']) for c in config['cases']}, 'Must use all 24, without selection by outcome.'
    assert len({c['id'] for c in config['cases']}) == 24
    for case in config['cases']:
        assert (ROOT/case['source']).is_file()
        assert unicodedata.normalize('NFC', (ROOT/case['source']).parent.name) == case['label']
        assert sha(ROOT/case['source']) == case['source_sha256']
        for poly in case['polygons'] + case.get('exclude_polygons', []):
            assert len(poly) >= 3
            assert all(len(p)==2 and all(type(v)==int and 0<=v<=255 for v in p) for p in poly)
    return config


def canonical(source):
    with Image.open(source) as im:
        rgb = im.convert('RGB')
        square = F.center_crop(F.resize(rgb,256,interpolation=InterpolationMode.BILINEAR),[256,256])
        assert np.array_equal(prepare_image(rgb), prepare_image(square))
    return square


def mask_from(case):
    im = Image.new('L',(256,256),0)
    d = ImageDraw.Draw(im)
    for poly in case['polygons']:
        d.polygon([tuple(p) for p in poly],fill=255)
    for poly in case.get('exclude_polygons',[]):
        d.polygon([tuple(p) for p in poly],fill=0)
    return im


def make_cases(config, folder):
    records = []
    folder.mkdir(parents=True,exist_ok=True)
    for case in config['cases']:
        directory = folder / case['id']
        directory.mkdir(exist_ok=True)
        before = canonical(ROOT/case['source'])
        mask = mask_from(case)
        binary = np.asarray(mask)>0
        arr = np.array(before)
        changed = arr.copy()
        changed[binary] = [128,128,128]
        assert np.array_equal(changed[~binary], arr[~binary])
        after = Image.fromarray(changed)
        for name, im in [('original.png',before),('changed.png',after),('mask.png',mask),
                         ('before_input.png',F.center_crop(before,[224,224])),
                         ('after_input.png',F.center_crop(after,[224,224]))]:
            im.save(directory/name,format='PNG')
        with Image.open(directory/'changed.png') as im:
            assert np.array_equal(np.asarray(im),changed)
        actual_mask = binary[16:240,16:240]
        actual_changes = np.any(arr[16:240,16:240]!=changed[16:240,16:240],axis=2)
        assert not np.any(actual_changes & ~actual_mask)
        records.append({**case,'canonical_path':record_path(directory/'original.png'),
                        'changed_path':record_path(directory/'changed.png'),'mask_path':record_path(directory/'mask.png'),
                        'before_input_path':record_path(directory/'before_input.png'),'after_input_path':record_path(directory/'after_input.png'),
                        'mask_pixels':int(binary.sum()),'mask_pixels_in_actual224':int(actual_mask.sum()),
                        'changed_pixels_in_actual224':int(actual_changes.sum()),
                        'outside_mask_changed_pixels':int(np.any(arr!=changed,axis=2)[~binary].sum()),
                        'original_input_exact':True})
    return records


def draw_sheets(records, folder):
    # Exact 256 images at native scale; labels in margins, never over photo pixels.
    for cls in ['yangyang','camellya','lynae','aemeath']:
        cases = [c for c in records if c['id'].startswith(cls+'_')]
        assert len(cases)==6
        sheet = Image.new('RGB',(768,6*282),'white')
        draw = ImageDraw.Draw(sheet)
        for row,c in enumerate(cases):
            draw.text((4,row*282+4),c['source_name']+' | original / hair mask / changed',fill='black')
            for col,name in enumerate(['original.png','mask.png','changed.png']):
                with Image.open(folder/c['id']/name) as im:
                    sheet.paste(im.convert('RGB'),(col*256,row*282+24))
        sheet.save(folder/(cls+'_mask_review.png'))


def predict(session, meta, im, label):
    prepared = prepare_image(im)
    logits = np.asarray(session.run([meta['output']['name']],{meta['input']['name']:prepared[None]})[0][0],dtype=np.float32)
    assert logits.shape==(len(meta['labels']),) and np.isfinite(logits).all()
    weights=np.exp(logits.astype(np.float64)-float(logits.max()))
    probs=weights/weights.sum()
    pred=int(logits.argmax()); true=meta['labels'].index(label)
    return {'pred_label':meta['labels'][pred],'pred_probability':float(probs[pred]),
            'true_probability':float(probs[true]),'correct':meta['labels'][pred]==label,
            'logits':logits.tolist(),'probabilities':dict(zip(meta['labels'],probs.tolist()))}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True)
    p.add_argument('--folder',default='data/changed/student_hair_masks')
    p.add_argument('--output',default='results/student_hair_masks.json')
    p.add_argument('--make-only',action='store_true')
    args=p.parse_args()
    config_path=Path(args.config).resolve()
    config=read_config(config_path)
    folder=Path(args.folder).resolve()
    assert folder.is_relative_to(ROOT/'data/changed') or folder.is_relative_to(Path('/private/tmp'))
    model=ROOT/'docs/model.onnx'; meta_path=ROOT/'docs/model.json'
    model_sha=sha(model); config_sha=sha(config_path); meta_sha=sha(meta_path)
    assert model_sha==config['model_sha256']
    records=make_cases(config,folder)
    draw_sheets(records,folder)
    proof={'config_sha256':config_sha,'model_sha256':model_sha,'cases':records,
           'inference_done':False,'all_original_inputs_equal':True,
           'all_outside_mask_pixels_equal':all(c['outside_mask_changed_pixels']==0 for c in records)}
    (folder/'mask_preflight.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n')
    if args.make_only:
        print('Mask preflight: 24 cases; original input equality 24/24; zero changed pixels outside annotated hair masks.')
        print('Masks are fixed before inference. Review the four *_mask_review.png sheets.')
        return
    assert config.get('mask_visual_review_passed') is True
    meta=json.loads(meta_path.read_text())
    expected={'resize':RESIZE,'crop':CROP,'mean':list(MEAN),'std':list(STD)}
    assert all(meta['input'].get(k)==v for k,v in expected.items())
    options=ort.SessionOptions(); options.intra_op_num_threads=1; options.inter_op_num_threads=1
    session=ort.InferenceSession(str(model),sess_options=options,providers=['CPUExecutionProvider'])
    print('Student-selected hair mask experiment: all 24 personal images, existing expanded ONNX model, no training')
    print('Student hypothesis: '+config['student_hypothesis_verbatim'])
    print('Student selection: '+config['student_scope_and_change_verbatim'])
    print('Pixel method: RGB(128,128,128) on annotated hair only; preserve the other canonical pixels exactly')
    print('Canonical method: original RGB, short-side resize256 bilinear, center256; exact original224 input')
    print('Hair ornaments/uncertain edges preserved; hair silhouette can remain; no claim of complete hairstyle removal')
    print('Fixed configuration SHA256: '+config_sha)
    print('Fixed model SHA256: '+model_sha)
    for rec in records:
        directory=folder/rec['id']
        with Image.open(directory/'original.png') as im:
            rec['before']=predict(session,meta,im,rec['label'])
        with Image.open(directory/'changed.png') as im:
            rec['after']=predict(session,meta,im,rec['label'])
        print('\n'+rec['id']+' | '+rec['source_name']+' | true '+rec['label'])
        print(f"Hair mask: {rec['mask_pixels']} pixels; actual224 mask {rec['mask_pixels_in_actual224']} pixels; changed224 {rec['changed_pixels_in_actual224']} pixels")
        print('Coverage note: '+rec['coverage_note'])
        for stage in ['before','after']:
            ans=rec[stage]
            print(f"{stage}: {ans['pred_label']} {ans['pred_probability']:.1%}; true-label {ans['true_probability']:.1%}; correct {ans['correct']}")
            print('  '+'; '.join(f"{label} {ans['probabilities'][label]:.1%}" for label in meta['labels']))
        print('Outside-mask changed pixels: 0; original224 exact: True')
    summary={}
    for label in [*meta['labels'],'ALL']:
        part=records if label=='ALL' else [c for c in records if c['label']==label]
        s={'total':len(part)}
        for stage in ['before','after']:
            s[stage+'_correct']=sum(c[stage]['correct'] for c in part)
            s[stage+'_percent']=100*s[stage+'_correct']/len(part)
        s['prediction_changes']=sum(c['before']['pred_label']!=c['after']['pred_label'] for c in part)
        s['score_rises']=sum(c['after']['true_probability']>c['before']['true_probability'] for c in part)
        s['score_falls']=sum(c['after']['true_probability']<c['before']['true_probability'] for c in part)
        s['score_same']=len(part)-s['score_rises']-s['score_falls']
        s['zero_visible_mask']=sum(c['mask_pixels_in_actual224']==0 for c in part)
        summary[label]=s
        print(f"\n{label}: before {s['before_correct']}/{s['total']} = {s['before_percent']:.1f}%; after {s['after_correct']}/{s['total']} = {s['after_percent']:.1f}%; prediction changes {s['prediction_changes']}")
        print(f"True-label score rose {s['score_rises']}; fell {s['score_falls']}; unchanged {s['score_same']}; zero-visible-mask cases {s['zero_visible_mask']}")
    assert sha(model)==model_sha and sha(config_path)==config_sha and sha(meta_path)==meta_sha
    assert all(sha(ROOT/c['source'])==c['source_sha256'] for c in config['cases'])
    assert summary['ALL']['before_correct']==22, 'Original evaluation must match the existing 22/24 result.'
    print('\nOriginal photos, model, metadata and fixed config SHA256 unchanged: True')
    result={'config_sha256':config_sha,'model_sha256':model_sha,'metadata_sha256':meta_sha,
            'inferences_per_case':{'before':1,'after':1},'no_training':True,
            'original_inputs_equal':True,'outside_masks_unchanged':True,
            'summary':summary,'cases':records,'student_reflection_received':False}
    out=Path(args.output).resolve();out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('Saved '+record_path(out))


if __name__=='__main__':
    main()
