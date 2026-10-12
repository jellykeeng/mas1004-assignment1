import argparse,csv,hashlib,json,subprocess,sys,unicodedata
from collections import Counter,defaultdict
from pathlib import Path
root=Path.cwd()
def nfc(s):return unicodedata.normalize('NFC',s)
parser=argparse.ArgumentParser();parser.add_argument('--manifest',required=True);args=parser.parse_args()
decisions=json.loads(Path(args.manifest).read_text())
folders={nfc(d.name):d for d in (root/'data/clean').iterdir() if d.is_dir()}
raw_hashes={nfc(str(p.relative_to(root/'data/raw'))):hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'data/raw').rglob('*.jpg')}
assert len(decisions)==600
assert sum(len(list(d.glob('*.jpg'))) for d in folders.values())==600,'Initial cleaning already applied; refuse rerun'
hash_record=root/'results/raw_hashes_before_cleaning.json'
if hash_record.exists():
 assert json.loads(hash_record.read_text())==raw_hashes,'Refuse to replace historical raw-image hash evidence'
else:
 hash_record.write_text(json.dumps(raw_hashes,ensure_ascii=False,indent=2))
groups=defaultdict(list)
for key,d in decisions.items():
 if d['keep']:continue
 label,filename=key.split('/')
 reason=d.get('reason','')
 category=d.get('category','')
 if d.get('duplicate_of') or '중복' in reason or '같은 원본' in reason or '동일 원본' in reason or '가까운 복사' in reason:
  common='동일 원본·자세의 크기/압축/자막/UI 변형인 근접 중복'
 elif category=='multiple_subjects' or '대등한' in reason or '주된 대상' in reason or '여러 캐릭터' in reason:
  common='대상 캐릭터가 주된 대상이 아니거나 여러 인물이 대등하게 등장'
 elif category=='text_ui_or_unrelated' or any(w in reason for w in ['표','아이콘','설명 글','공지','글·','본문','캐릭터없는','없는 무기','인터페이스','음식']):
  common='글·표·무기·UI 또는 무관한 대상이 주된 이미지'
 elif category=='occluded_or_cropped' or any(w in reason for w in ['가려','심하게','식별불가','잘린']):
  common='심한 가림·크롭으로 대상 캐릭터를 식별하기 어려움'
 else:
  common='다른 캐릭터이거나 대상 캐릭터를 식별할 수 없음'
 rel=str((folders[label]/filename).relative_to(root/'data/clean'))
 groups[common].append(rel)
for reason,files in groups.items():
 subprocess.run(['uv','run','--frozen','python','src/clean.py','remove',*files,'--reason',reason],check=True)
current_hashes={nfc(str(p.relative_to(root/'data/raw'))):hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'data/raw').rglob('*.jpg')}
assert current_hashes==raw_hashes,'Raw data changed'
print('Raw originals unchanged:',len(raw_hashes),'SHA-256 checks passed')
print('Decision total:',len(decisions),'kept:',sum(d['keep'] for d in decisions.values()),'removed:',sum(not d['keep'] for d in decisions.values()))
for label,directory in sorted(folders.items()):print(label,'kept',len(list(directory.glob('*.jpg'))))
