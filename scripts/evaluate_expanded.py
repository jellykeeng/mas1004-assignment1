"""Measure the expanded model on fixed evaluation photos and existing changed copies."""
import hashlib,json,sys,unicodedata
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts'))
import run_experiment
from data import load_folder,prepare_image
from evaluate import predict_logits,confusion_matrix,worst_examples,plot_worst
from train import load

def nfc(text):return unicodedata.normalize('NFC',str(text))
def answer(model,path,names):
    with Image.open(path) as im:x=prepare_image(im)
    logits=predict_logits(model,x[None])[0]
    p=np.exp(logits-logits.max());p/=p.sum()
    i=int(p.argmax())
    return {'predicted':i,'label':nfc(names[i]),'confidence':float(p[i])}

def main():
    out=ROOT/'results';tag='expanded'
    s=json.loads((out/f'{tag}_settings.json').read_text())
    names=s['class_names'];model=load(out/f'{tag}_model.pt')
    X,y,actual_names,paths=load_folder(ROOT/'data/new_images');assert actual_names==names
    logits=predict_logits(model,X);correct=int((logits.argmax(1)==y).sum())
    matrix=confusion_matrix(model,X,y,len(names))
    train_pct=f"{s['train_accuracy']:.1%}";test_pct=f"{s['test_accuracy']:.1%}";new_pct=f'{correct/len(y):.1%}'
    print(f'{tag}: train {train_pct}; downloaded test {test_pct}; fixed personal evaluation {new_pct} ({correct}/{len(y)})')
    print('Fixed personal confusion matrix (true rows, predicted columns):\n'+str(matrix))
    mistakes=worst_examples(model,X,y,paths,k=5)
    if mistakes:plot_worst(mistakes,[nfc(n) for n in names],out/'expanded_new_worst.png')
    serial=[]
    for rank,m in enumerate(mistakes,1):
        row=dict(m,path=nfc(m['path'].relative_to(ROOT)),true_label=nfc(names[m['true']]),predicted_label=nfc(names[m['predicted']]),confidence_pct=f"{m['confidence']:.1%}")
        serial.append(row)
        print(f"Confident new-image mistake {rank}: {row['path']}; true {row['true_label']}; predicted {row['predicted_label']} {row['confidence_pct']}")
    old=json.loads((out/'analysis.json').read_text())
    ablations=[];hashes={}
    for row in old['ablations']:
        item=dict(row)
        for kind in ['original','changed']:
            path=ROOT/row[kind];hashes[row[kind]]=hashlib.sha256(path.read_bytes()).hexdigest()
        with Image.open(ROOT/row['source']) as source,Image.open(ROOT/row['original']) as original:
            assert np.array_equal(prepare_image(source),prepare_image(original)),row['source']
        item['baseline']=answer(model,ROOT/row['original'],names)
        item['answer']=answer(model,ROOT/row['changed'],names)
        ablations.append(item)
        before=item['baseline'];after=item['answer']
        print(f"{row['kind']}: {nfc(row['source'])}: {before['label']} {before['confidence']:.1%} => {after['label']} {after['confidence']:.1%}; rectangle {row['box']}")
    sums={}
    for kind in ['gray','head_mask','clothes_mask']:
        group=[r for r in ablations if r['kind']==kind]
        a=sum(r['baseline']['label']==r['true_label'] for r in group)
        b=sum(r['answer']['label']==r['true_label'] for r in group)
        flips=sum(r['baseline']['label']!=r['answer']['label'] for r in group)
        sums[kind]={'before':a,'after':b,'total':len(group),'flips':flips}
        print(f'{kind} summary: original {a}/{len(group)}; changed {b}/{len(group)}; label flips {flips}')
    ood_path=ROOT/'data/raw'/next(d.name for d in (ROOT/'data/raw').iterdir() if nfc(d.name)=='명조_양양')/'0089.jpg'
    ood=answer(model,ood_path,names)
    print('Weapons-only OOD image:',nfc(ood_path.relative_to(ROOT)),'=>',ood['label'],f"{ood['confidence']:.1%}",'(no unknown class)')
    for relative,expected in hashes.items():assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==expected
    print('Existing changed image files unchanged:',len(hashes))
    result={'tag':tag,'settings':s,'train_pct':train_pct,'test_pct':test_pct,'new_pct':new_pct,'new_correct':correct,'new_total':len(y),'new_matrix':matrix.tolist(),'new_mistakes':serial,'ablations':ablations,'ablation_summaries':sums,'ood':ood,'changed_hashes':hashes}
    (out/'expanded_analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
