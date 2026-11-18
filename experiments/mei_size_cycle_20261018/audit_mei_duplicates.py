"""Audit existing TRAIN/VAL source content only; do not read old test rows."""
import json,hashlib
from pathlib import Path
from collections import defaultdict,Counter

root=Path('outputs/mei_size_cycle_20261018');root.mkdir(exist_ok=True)
manifest=json.loads(Path('outputs/data0723_identity_cyclegan/split_manifest.json').read_text())
rows=[r for r in manifest['records'] if r['user']=='wangjingmei' and r['split'] in ['train','validation']]
groups=defaultdict(list);axis=defaultdict(list);audio=defaultdict(list)
for row in rows:
    a=Path(row['acc']).read_bytes();y=Path(row['audio']).read_bytes()
    entry={k:row[k] for k in ['task','key','split']}
    groups[hashlib.sha256(a+y).hexdigest()].append(entry)
    axis[hashlib.sha256(a).hexdigest()].append(entry);audio[hashlib.sha256(y).hexdigest()].append(entry)
cross=[dict(sha256=h,records=rr) for h,rr in groups.items() if len({r['split'] for r in rr})>1]
report=dict(nominal_pairs=len(rows),unique_pairs=len(groups),unique_ACC=len(axis),unique_audio=len(audio),per_split={s:dict(nominal=sum(r['split']==s for r in rows),unique_pairs=sum(any(r['split']==s for r in rr) for rr in groups.values())) for s in ['train','validation']},cross_split_duplicate_groups=len(cross),cross_split_examples=cross[:5],all_pair_groups=[dict(sha256=h,records=rr) for h,rr in groups.items()],test='not read')
all_rows=[r for r in manifest['records'] if r['user']=='wangjingmei']
protected={r['key'] for r in all_rows if r['split']!='train'}
eligible=[r for r in rows if r['split']=='train' and r['key'] not in protected]
report.update(unique_source_keys_metadata=len({r['key'] for r in all_rows}),clean_train_source_keys_after_protecting_val_and_test=len({r['key'] for r in eligible}),protected_rule='All VAL/test source keys blocked from TRAIN based on metadata; test arrays not read')
(root/'duplicate_audit_private.json').write_text(json.dumps(report,indent=2))
public={k:v for k,v in report.items() if k not in ['all_pair_groups','cross_split_examples']}
(root/'duplicate_audit.json').write_text(json.dumps(public,indent=2))
print(json.dumps({**public,'examples':cross[:2]},indent=2))
