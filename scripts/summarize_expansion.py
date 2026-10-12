import csv,hashlib,json,unicodedata
from pathlib import Path
from collections import Counter
root=Path.cwd();out=root/'results'
nfc=lambda s:unicodedata.normalize('NFC',str(s))
imported=json.loads((out/'expanded_import.json').read_text())
review=json.loads((out/'expanded_decisions.json').read_text())
counts={}
for folder in sorted((root/'data/raw').iterdir()):
    if not folder.is_dir():continue
    raw=list(folder.glob('*.jpg'));clean=list((root/'data/clean'/folder.name).glob('*.jpg'))
    name=nfc(folder.name)
    counts[name]={'raw':len(raw),'clean':len(clean),'removed':len(raw)-len(clean)}
    before=counts[name]['clean']-imported['classes'][name]['added']
    print(name+': previously clean '+str(before)+'; useful added '+str(imported['classes'][name]['added'])+'; current clean '+str(len(clean))+'; current raw '+str(len(raw))+'; current removed '+str(len(raw)-len(clean)))
    queries=Counter(row['query'] for row in review.values() if row['category']==name)
    for q,n in queries.items():print('  additional query: '+q+'; downloaded '+str(n))
print('Current raw images:',sum(c['raw'] for c in counts.values()))
print('Current clean images:',sum(c['clean'] for c in counts.values()))
print('Current removed images:',sum(c['removed'] for c in counts.values()))
print('Personal evaluation images unchanged:',sum(1 for p in (root/'data/new_images').glob('*/*') if p.is_file()))
print('All previous files preserved:',imported['all_previous_files_unchanged'],'files checked:',imported['protected_files'])
reasons=Counter(r['reason'] for r in csv.DictReader((root/'data/removed/log.csv').open()))
print('All logged removal reasons:')
for reason,n in reasons.items():print('  '+reason+': '+str(n))
(out/'expanded_summary.json').write_text(json.dumps({'counts':counts,'removal_reasons':dict(reasons),'additional_classes':imported['classes']},ensure_ascii=False,indent=2))
