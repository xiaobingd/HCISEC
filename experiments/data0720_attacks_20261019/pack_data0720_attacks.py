"""Package labelled ACC sources, auditing duplicate collection directories."""
import json,hashlib,zipfile
from pathlib import Path
import numpy as np

root=Path('E:/desk/Data0720')
out=Path(__file__).resolve().parent.parent/'outputs/data0720_attacks_20261019'
out.mkdir(exist_ok=True)
rows=[];canonical={};unlabelled=[];invalid=[]
users=['fanweibo','fanweiqi','fanweiqisis']
for p in sorted(root.rglob('accel_data*.npy')):
    rel=p.relative_to(root);b=p.read_bytes();h=hashlib.sha256(b).hexdigest()
    if rel.parts[1] not in users:
        unlabelled.append(dict(path=rel.as_posix(),sha256=h));continue
    a=np.load(p);row=dict(task=rel.parts[0],user=rel.parts[1],key=p.stem,path=rel.as_posix(),sha256=h,shape=list(a.shape),finite=bool(np.isfinite(a).all()))
    if not (row['finite'] and a.ndim==2 and a.shape[1]==4 and len(a)>1):
        invalid.append(row);continue
    rows.append(row);canonical.setdefault((row['user'],row['key']),row)
selected=[];seen={};duplicates=[]
for r in canonical.values():
    if r['sha256'] in seen:
        assert seen[r['sha256']]['user']==r['user'],('Cross-user duplicate',r)
        duplicates.append(r);continue
    seen[r['sha256']]=r;selected.append(r)
with zipfile.ZipFile(out/'acc_sources.zip','w',zipfile.ZIP_DEFLATED) as z:
    for r in selected:z.write(root/r['path'],r['path'])
    z.writestr('manifest.json',json.dumps(dict(records=selected),indent=2))
audit=dict(labelled_files=len(rows),unique_user_keys=len(canonical),unique_content=len(selected),counts={u:sum(r['user']==u for r in selected) for u in users},shapes=sorted({str(r['shape']) for r in rows}),unlabelled_files=len(unlabelled),unlabelled_matched=sum(r['sha256'] in seen for r in unlabelled),unlabelled_unmatched=[r['path'] for r in unlabelled if r['sha256'] not in seen],content_duplicates_removed=len(duplicates),source='E:/desk/Data0720',selection='All labelled users; canonical task order then exact raw-ACC content dedup, independent of authentication scores.')
audit['invalid_files']=invalid
(out/'source_audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit));print('ZIP_BYTES',(out/'acc_sources.zip').stat().st_size)
