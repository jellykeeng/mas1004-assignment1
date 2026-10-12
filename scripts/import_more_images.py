"""Import reviewed web supplements, preserving all previous evidence and originals."""
import argparse, hashlib, json, shutil, subprocess, unicodedata
from collections import Counter, defaultdict
from pathlib import Path

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def nfc(text):
    return unicodedata.normalize('NFC',str(text))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',required=True,type=Path)
    args=parser.parse_args()
    root=Path.cwd(); out=root/'results'
    if (out/'expanded_import.json').exists():
        raise RuntimeError('Expansion already imported; refusing to rerun or overwrite evidence')
    bundle=json.loads(args.bundle.read_text())
    categories=[nfc(spec['category']) for spec in bundle['classes']]
    assert len(categories)==len(set(categories)),'Duplicate category in bundle'
    for spec in bundle['classes']:
        for field in ['manifest','download_output','sources']:
            if spec.get(field): assert Path(spec[field]).is_file(),spec[field]
    raw_folders={nfc(d.name):d for d in (root/'data/raw').iterdir() if d.is_dir()}
    protected=[p for base in ['data/raw','data/new_images','data/clean','short_report'] for p in (root/base).rglob('*') if p.is_file()]
    protected += [root/'long_report/mas1004.sty']
    before={str(p.relative_to(root)):digest(p) for p in protected}
    summary={}; decisions={}; path_map={}; planned=[]
    for spec in bundle['classes']:
        folder=raw_folders[nfc(spec['category'])]
        review=json.loads(Path(spec['manifest']).read_text())
        rows=review['decisions'] if 'decisions' in review else review
        if isinstance(rows,dict): rows=list(rows.values())
        assert all(isinstance(x.get('keep'),bool) for x in rows)
        start=max(int(p.stem) for p in folder.glob('*.jpg'))+1
        rows=sorted(rows,key=lambda x:x['source_path'])
        counts=Counter()
        for number,row in enumerate(rows,start):
            source=Path(row['source_path']); assert source.is_file(),source
            if row.get('sha256'): assert digest(source)==row['sha256'],source
            raw=folder/f'{number:04d}.jpg'
            clean=root/'data/clean'/folder.name/raw.name
            removed=root/'data/removed'/folder.name/raw.name
            assert not raw.exists() and not clean.exists() and not removed.exists(),raw
            key=nfc(raw.relative_to(root/'data/raw'))
            newrow=dict(row,path=key,category=nfc(folder.name),sha256=digest(source),imported_raw='data/raw/'+key)
            code=row.get('code',row.get('reason_code',row.get('cause_code')))
            if row['keep']: code='keep'
            if code not in ['keep','duplicate','wrong_character','not_main','ui','severe_crop']:
                raise ValueError(f'Explicit removal code required for {source}: {code}')
            newrow['code']=code
            decisions[key]=newrow
            path_map[str(source)]=key
            planned.append((newrow,source,raw,clean))
            counts[code]+=1
        summary[nfc(folder.name)]={'downloaded':len(rows),'added':counts['keep'],'removed':len(rows)-counts['keep'],'reasons':dict(counts)}
    for row in decisions.values():
        ref=row.get('duplicate_of')
        if ref and str(ref) in path_map:row['duplicate_of']=path_map[str(ref)]
    assert len(planned)==len({str(raw) for _,_,raw,_ in planned}),'Duplicate planned destination'
    # Validate all sources and destinations before changing the dataset.
    reasons={
        'duplicate':'동일 원본·자세의 크기/압축/자막/UI 변형인 근접 중복',
        'wrong_character':'다른 캐릭터이거나 대상 캐릭터를 식별할 수 없음',
        'not_main':'대상 캐릭터가 주된 대상이 아니거나 여러 인물이 대등하게 등장',
        'ui':'글·표·UI·상품 설명이 주된 이미지',
        'severe_crop':'심하게 가리거나 잘려 캐릭터를 식별하기 어려움',
    }
    groups=defaultdict(list)
    for row,source,raw,clean in planned:
        shutil.copy2(source,raw);shutil.copy2(raw,clean)
        if not row['keep']:groups[row['code']].append(str(clean.relative_to(root/'data/clean')))
    for code,paths in groups.items():
        for start in range(0,len(paths),100):
            subprocess.run(['uv','run','--frozen','python','src/clean.py','remove',*paths[start:start+100],'--reason',reasons[code]],check=True)
    for relative,expected in before.items():
        assert digest(root/relative)==expected,f'Existing protected file changed: {relative}'
    for row,source,raw,clean in planned:
        assert digest(raw)==row['sha256']
        assert clean.exists()==row['keep']
    (out/'expanded_decisions.json').write_text(json.dumps(decisions,ensure_ascii=False,indent=2))
    for spec in bundle['classes']:
        short=spec['short']
        shutil.copy2(spec['download_output'],out/f'expanded_collect_{short}.txt')
        if spec.get('sources'):
            shutil.copy2(spec['sources'],out/f'expanded_sources_{short}.json')
    details={'classes':summary,'protected_files':len(before),'all_previous_files_unchanged':True,'before_hashes':before}
    (out/'expanded_import.json').write_text(json.dumps(details,ensure_ascii=False,indent=2))
    for category,counts in summary.items():
        print(category+': additional downloaded '+str(counts['downloaded'])+'; useful added '+str(counts['added'])+'; removed '+str(counts['removed']))
        print('Reasons:',counts['reasons'])
    print('All previous raw/clean/personal photos, short report and style unchanged:',len(before),'files')
    print('Additional candidates:',len(planned),'useful:',sum(x['keep'] for x in decisions.values()))

if __name__=='__main__': main()
