"""Measure existing runs and produce report evidence without retraining."""
import csv
import hashlib
import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from data import CROP, MEAN, STD, load_folder, prepare_image, split_train_test
from evaluate import accuracy, confusion_matrix, predict_logits, worst_examples
from train import load
import run_experiment  # registers a Korean font and readable plot labels
import evaluate

def nfc(text):
    return unicodedata.normalize('NFC', str(text))

def probabilities(logits):
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

def describe_prediction(model, picture, names):
    x = prepare_image(picture)
    logits = predict_logits(model, x[None])[0]
    p = probabilities(logits)
    i = int(p.argmax())
    return {'predicted': i, 'label': nfc(names[i]), 'confidence': float(p[i])}

def main():
    out = ROOT / 'results'
    final_tag = sys.argv[1] if len(sys.argv) > 1 else 'final'
    names = json.loads((out / f'{final_tag}_settings.json').read_text())['class_names']
    records = {}
    Xnew, ynew, newnames, newpaths = load_folder(ROOT / 'data/new_images')
    assert newnames == names
    for tag in ['run', 'scratch', 'frozen', 'lr3e4', 'clean', final_tag]:
        settings = json.loads((out / f'{tag}_settings.json').read_text())
        model = load(out / f'{tag}_model.pt')
        # Restore each run's unchanged test images from raw, including those
        # deliberately removed later. The saved paths define the actual split.
        paths = [ROOT / 'data/raw' / Path(p).parent.name / Path(p).name
                 for p in settings['test_paths']]
        images = []
        for p in paths:
            with Image.open(p) as im:
                images.append(prepare_image(im))
        Xtest = np.stack(images)
        ytest = np.array([names.index(p.parent.name) for p in paths])
        matrix = confusion_matrix(model, Xtest, ytest, len(names))
        train_pct = f"{settings['train_accuracy']:.1%}"
        test_pct = f"{settings['test_accuracy']:.1%}"
        gap = float(train_pct.rstrip('%')) - float(test_pct.rstrip('%'))
        new_logits = predict_logits(model, Xnew)
        new_accuracy = float((new_logits.argmax(1) == ynew).mean())
        pair = matrix + matrix.T
        np.fill_diagonal(pair, 0)
        a, b = np.unravel_index(pair.argmax(), pair.shape)
        print(f'{tag}: train {train_pct}; downloaded test {test_pct}; new images {new_accuracy:.1%} '
              f'({int((new_logits.argmax(1) == ynew).sum())}/{len(ynew)}); gap {gap:.1f} percentage points')
        print(f'{tag} test confusion matrix (true rows, predicted columns):\n{matrix}')
        print(f'{tag} largest confusion pair: {nfc(names[a])} / {nfc(names[b])}: '
              f'{int(matrix[a,b])} + {int(matrix[b,a])} = {int(pair[a,b])}')
        records[tag] = {'train_pct': train_pct, 'test_pct': test_pct,
            'new_pct': f'{new_accuracy:.1%}', 'new_correct': int((new_logits.argmax(1) == ynew).sum()),
            'gap': f'{gap:.1f}', 'matrix': matrix.tolist(), 'settings': settings}
        del model
    print('ImageNet minus scratch test:',
          f"{float(records['run']['test_pct'].rstrip('%')) - float(records['scratch']['test_pct'].rstrip('%')):.1f}",
          'percentage points')
    print('All layers minus frozen test:',
          f"{float(records['run']['test_pct'].rstrip('%')) - float(records['frozen']['test_pct'].rstrip('%')):.1f}",
          'percentage points')
    print('lr3e4 minus default test:',
          f"{float(records['lr3e4']['test_pct'].rstrip('%')) - float(records['run']['test_pct'].rstrip('%')):.1f}",
          'percentage points')

    model = load(out / f'{final_tag}_model.pt')
    new_mistakes = worst_examples(model, Xnew, ynew, newpaths, k=5)
    evaluate.plot_worst(new_mistakes, [nfc(n) for n in names], out / 'new_worst.png')
    serial_mistakes = []
    for rank, item in enumerate(new_mistakes, 1):
        row = dict(item)
        row['path'] = str(item['path'].relative_to(ROOT))
        row['true_label'] = nfc(names[item['true']])
        row['predicted_label'] = nfc(names[item['predicted']])
        row['confidence_pct'] = f"{item['confidence']:.1%}"
        serial_mistakes.append(row)
        print(f"Confident new-image mistake {rank}: {nfc(row['path'])}: "
              f"true {row['true_label']}, predicted {row['predicted_label']} {row['confidence_pct']}")

    decisions = json.loads((out / 'cleaning_decisions.json').read_text())
    supplement = json.loads((out / 'supplement_decisions.json').read_text())
    decisions.update(supplement)
    suspect_rows = list(csv.DictReader((out / 'cleaning/suspects/suspects.csv').open()))
    copy_pairs = [r for r in suspect_rows if r['label'] == 'copy of']
    print('Initial near-copy candidate pairs:', len(copy_pairs))
    pair_retained = 0
    for r in copy_pairs:
        if decisions[nfc(r['file'])]['keep'] and decisions[nfc(r['rank_in_class'])]['keep']:
            pair_retained += 1
    print('Initial candidate pairs with both kept after review:', pair_retained)
    first20 = {}
    for label in names:
        top = [r for r in suspect_rows if r['label'] == label and int(r['rank_in_class']) <= 20]
        removed = sum(not decisions[nfc(r['file'])]['keep'] for r in top)
        first20[nfc(label)] = removed
        print(f'{nfc(label)} first 20 suspects removed: {removed}/{len(top)}')
    with (ROOT / 'data/removed/log.csv').open() as handle:
        log = list(csv.DictReader(handle))
    ranks = {nfc(r['file']): int(r['rank_in_class']) for r in suspect_rows if r['label'] != 'copy of'}
    outside_top20 = [r for r in log if nfc(r['file']) in ranks and ranks[nfc(r['file'])] > 20]
    print('Removed initial images outside top20:', len(outside_top20))
    raw_hashes = json.loads((out / 'raw_hashes_before_cleaning.json').read_text())
    unchanged = all(hashlib.sha256((ROOT / 'data/raw' / key).read_bytes()).hexdigest() == digest
                    for key, digest in raw_hashes.items())
    print('Initial 600 raw originals unchanged:', unchanged)
    assert unchanged

    # Preparation example: real downloaded screenshot, not a fabricated image.
    original_path = ROOT / 'data/raw' / next(n for n in names if '에이메스' in nfc(n)) / '0033.jpg'
    with Image.open(original_path) as im:
        rgb = im.convert('RGB')
        rgb.save(out / 'prepare_original.jpg', quality=95)
        x = prepare_image(rgb)
        restored = ((x.transpose(1,2,0) * np.array(STD) + np.array(MEAN)) * 255).round().clip(0,255).astype('uint8')
        Image.fromarray(restored).save(out / 'prepare_result.png')
        print('Preparation source:', nfc(original_path.relative_to(ROOT)), 'original dimensions', rgb.size,
              'tensor shape', x.shape, 'dtype', x.dtype)

    # OOD sample was excluded during cleaning, so it did not train the published model.
    ood = ROOT / 'data/raw' / next(n for n in names if '양양' in nfc(n)) / '0089.jpg'
    with Image.open(ood) as im:
        ood_answer = describe_prediction(model, im, names)
    print('Weapons-only OOD image:', nfc(ood.relative_to(ROOT)), '=>', ood_answer['label'],
          f"{ood_answer['confidence']:.1%}", '(no unknown class)')

    chosen = [
        ('양양','IMG_1390.JPG',(107,20,208,110),(89,135,197,239)),
        ('양양','IMG_1392.JPG',(42,22,130,93),(62,133,152,239)),
        ('카멜리아','IMG_0584.jpg',(45,16,196,118),(67,127,182,238)),
        ('카멜리아','IMG_2555.jpg',(16,16,230,140),(17,145,239,239)),
        ('린네','IMG_1417.JPG',(95,25,171,89),(85,120,174,230)),
        ('린네','IMG_0850.jpg',(80,56,193,150),(90,148,189,239)),
        ('에이메스','IMG_0215.PNG',(98,16,212,115),(56,133,213,239)),
        ('에이메스','IMG_0553.jpg',(16,16,239,192),None),
    ]
    from PIL import ImageDraw, ImageOps
    changed_root = ROOT / 'data/changed'
    ablations = []
    for label, filename, headbox, clothesbox in chosen:
        directory = next(d for d in (ROOT/'data/new_images').iterdir() if nfc(d.name) == '명조_'+label)
        source = directory / filename
        with Image.open(source) as im:
            square = F.center_crop(F.resize(im.convert('RGB'), 256, interpolation=InterpolationMode.BILINEAR),256)
            reference_x = prepare_image(im)
        assert np.array_equal(prepare_image(square), reference_x), 'Reference square must preserve all model input pixels'
        stem = label+'_'+Path(filename).stem
        original_target = changed_root/'original'/f'{stem}.png'
        original_target.parent.mkdir(parents=True,exist_ok=True)
        square.save(original_target)
        baseline = describe_prediction(model,square,names)
        for kind in ['gray','head_mask','clothes_mask']:
            if kind == 'clothes_mask' and clothesbox is None:
                continue  # face-only photo has no clothing to cover
            changed = square.copy()
            box = None
            if kind == 'gray':
                changed = ImageOps.grayscale(square).convert('RGB')
            else:
                box = headbox if kind=='head_mask' else clothesbox
                ImageDraw.Draw(changed).rectangle(box,fill=(128,128,128))
            target = changed_root/kind/f'{stem}.png'
            target.parent.mkdir(parents=True,exist_ok=True)
            changed.save(target)
            answer = describe_prediction(model,changed,names)
            row = {'kind':kind,'true_label':'명조_'+label,'source':str(source.relative_to(ROOT)),
                   'original':str(original_target.relative_to(ROOT)),'changed':str(target.relative_to(ROOT)),
                   'box':box,'baseline':baseline,'answer':answer}
            ablations.append(row)
            print(f"{kind}: {label}/{filename}: {baseline['label']} {baseline['confidence']:.1%} => "
                  f"{answer['label']} {answer['confidence']:.1%}; rectangle {box}")
    for kind in ['gray','head_mask','clothes_mask']:
        selected = [r for r in ablations if r['kind']==kind]
        before = sum(r['baseline']['label']==r['true_label'] for r in selected)
        after = sum(r['answer']['label']==r['true_label'] for r in selected)
        flips = sum(r['baseline']['label']!=r['answer']['label'] for r in selected)
        print(f'{kind} summary: original {before}/{len(selected)}; changed {after}/{len(selected)}; label flips {flips}')

    output = {'runs':records,'names':[nfc(n) for n in names],'first20':first20,
              'copy_pairs':len(copy_pairs),'pair_retained':pair_retained,
              'removed_outside_top20':len(outside_top20), 'new_mistakes':serial_mistakes,
              'ood':ood_answer,'ablations':ablations,'final_tag':final_tag}
    (out/'analysis.json').write_text(json.dumps(output,ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
