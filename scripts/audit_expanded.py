"""Verify the new split, image contents, and preservation of the fixed evaluation inputs."""
import hashlib,json,unicodedata
from collections import defaultdict
from pathlib import Path
from PIL import Image
root=Path.cwd();out=root/'results'
nfc=lambda s:unicodedata.normalize('NFC',str(s))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
s=json.loads((out/'expanded_settings.json').read_text())
members={nfc(p.relative_to(root/'data/clean')):p for p in (root/'data/clean').rglob('*.jpg')}
test={nfc(Path(p).parent.name+'/'+Path(p).name) for p in s['test_paths']}
train=set(members)-test
assert len(members)==s['n_images'] and test<=set(members)
path_overlap=train&test
trainhash={sha(members[k]) for k in train};testhash={sha(members[k]) for k in test}
personal=[p for p in (root/'data/new_images').glob('*/*') if p.is_file()]
personalhash={sha(p) for p in personal}
print('expanded training images:',len(train),'test images:',len(test),'total:',len(members))
print('Train/test shared paths:',len(path_overlap))
print('Train/test shared exact image contents:',len(trainhash&testhash))
print('Personal evaluation images:',len(personal))
print('Training/downloaded-test content shared with personal evaluation:',len((trainhash|testhash)&personalhash))
assert not path_overlap and not trainhash&testhash and not (trainhash|testhash)&personalhash
pixel_groups=defaultdict(list)
for key,path in members.items():
    with Image.open(path) as im:
        im=im.convert('RGB')
        h=hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest()
    pixel_groups[h].append(key)
duplicates=[v for v in pixel_groups.values() if len(v)>1]
print('Clean exact pixel duplicate groups:',len(duplicates))
assert not duplicates,duplicates
personalpixels=set()
for path in personal:
    with Image.open(path) as im:
        im=im.convert('RGB')
        personalpixels.add(hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest())
pixel_overlap=set(pixel_groups)&personalpixels
print('Clean/personal evaluation exact pixel overlap:',len(pixel_overlap))
assert not pixel_overlap
protected=json.loads((out/'expanded_import.json').read_text())['before_hashes']
assert all(sha(root/key)==value for key,value in protected.items())
print('Previous raw/clean and all personal photos, short report and style unchanged:',len(protected),'files')
old_analysis=json.loads((out/'analysis.json').read_text())
new_analysis=json.loads((out/'expanded_analysis.json').read_text())
assert [(r['kind'],r['source'],r['original'],r['changed'],r['box']) for r in old_analysis['ablations']]==[(r['kind'],r['source'],r['original'],r['changed'],r['box']) for r in new_analysis['ablations']]
assert all(sha(root/key)==value for key,value in new_analysis['changed_hashes'].items())
print('Identical existing image-change definitions:',len(new_analysis['ablations']))
print('Existing original/changed image files preserved:',len(new_analysis['changed_hashes']))
