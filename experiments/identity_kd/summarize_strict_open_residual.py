"""Summarize completed outer folds, without selecting methods on final scores."""
import argparse
import json
from pathlib import Path
import numpy as np

ap=argparse.ArgumentParser()
ap.add_argument('root',type=Path)
a=ap.parse_args()
rows=[json.loads(p.read_text()) for p in sorted(a.root.glob('target*_seed*/results_private.json'))]
summary={'completed_folds':len(rows),'expected_folds':24,'seed_macro':{},'paired_deltas':{}}
for scope in ['unseen_attacker','all_seven_sources']:
    summary['seed_macro'][scope]={}
    for mode in ['raw_acc','identity_only','uniform_audio','targeted_audio']:
        summary['seed_macro'][scope][mode]={}
        for cond in ['sameword','crossword']:
            perseed={}
            for seed in [42,43,44]:
                selected=[r for r in rows if r['seed']==seed]
                if selected:
                    perseed[str(seed)]={'n_targets':len(selected),**{m:float(np.mean([r['methods'][mode][cond][scope][m] for r in selected])) for m in ['tar','far','hter','auc']}}
            complete=[v for v in perseed.values() if v['n_targets']==8]
            summary['seed_macro'][scope][mode][cond]={'by_seed':perseed,'complete_seeds':len(complete)}
            if complete:
                summary['seed_macro'][scope][mode][cond]['mean']={m:float(np.mean([v[m] for v in complete])) for m in ['tar','far','hter','auc']}
                if len(complete)>1:summary['seed_macro'][scope][mode][cond]['seed_std']={m:float(np.std([v[m] for v in complete],ddof=1)) for m in ['tar','far','hter','auc']}
        summary['paired_deltas'][scope+'_'+mode]=[dict(target=r['fold']['target'],seed=r['seed'],cross_minus_same_hter=r['methods'][mode]['crossword'][scope]['hter']-r['methods'][mode]['sameword'][scope]['hter']) for r in rows]
(a.root/'summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
