"""Independent split/count/pairwise-AUC verification of completed strict targets."""
import json
from pathlib import Path
import numpy as np
ROOT=Path('/autodl-fs/data/mei_strict_baseline_20261029')
def main():
 m=json.loads((ROOT/'manifest_private.json').read_text());groups=m['groups'];assert sorted(sum(groups.values(),[]))==list(range(30))
 for u,s in m['sources'].items():
  roles=s['roles'];content=np.asarray(s['content_groups']);assert sorted(sum(roles.values(),[]))==list(range(s['rows']))
  for k,idx in roles.items():assert set(content[idx])==set(groups[k])
 rows={}
 for u in m['queue_order']:
  folder=ROOT/('U%d'%u)/'auth'
  if not (folder/'COMPLETE.json').exists():continue
  r=json.loads((folder/'results_public.json').read_text());d=np.load(str(folder/'scores_private.npz'));g=d['final_genuine'];n=d['final_impostor'];labels=d['attack_labels'];a=r['final_attack'];t=a['threshold'];auc=float(((g[:,None]>n[None,:])+.5*(g[:,None]==n[None,:])).mean())
  assert len(g)==len(m['sources'][str(u)]['roles']['test'])
  assert len(n)==sum(len(s['roles']['test']) for v,s in m['sources'].items() if int(v)!=u)
  assert int((g>=t).sum())==a['genuine_accepted'];assert int((n>=t).sum())==a['attack_accepted'];assert abs(auc-a['auc'])<1e-12
  assert abs(np.mean(n>=t)-a['far'])<1e-12 and abs(np.mean(g<t)-a['frr'])<1e-12
  for label in range(9):
   if label==u-1:continue
   name='bo' if label==0 else 'user%d'%(label+1);z=n[labels==label];v=a['attack_by_user'][name];assert len(z)==v['n'] and int((z>=t).sum())==v['accepted']
  rows[str(u)]=dict(selected_epoch=r['best_epoch'],genuine_n=len(g),genuine_accepted=int((g>=t).sum()),attack_n=len(n),attack_accepted=int((n>=t).sum()),TAR=float(np.mean(g>=t)),FAR=float(np.mean(n>=t)),AUC=auc,by_attacker=a['attack_by_user'])
 out=dict(status='PASS',completed_targets=list(rows),target_count=len(rows),all_nine_complete=len(rows)==9,content_role_disjoint=True,source_row_complete=True,counts_and_pairwise_auc=True,scope=m['scope'],limitations=m['limitations'],results=rows)
 (ROOT/'verified_summary_public.json').write_text(json.dumps(out,indent=2))
 text=['# Mei mixed-content held-out baseline','', 'Status: %d/9 personal targets verified. All eight attack identities retained.'%len(rows),'','Content roles are disjoint under the inferred digit10/command20 layout. Semantic per-row transcript mapping remains unverified. Closed-set same-session historical recordings; fresh personal CycleGAN200 + original full772 trainb SupCon/BCE.','', '|Target|Selected head epoch|Genuine accepted|Attack accepted|TAR|FAR|AUC|','|---|---:|---|---|---:|---:|---:|']
 for u,a in rows.items():text.append('|U%s|%d|%d/%d|%d/%d|%.2f%%|%.2f%%|%.4f|'%(u,a['selected_epoch'],a['genuine_accepted'],a['genuine_n'],a['attack_accepted'],a['attack_n'],a['TAR']*100,a['FAR']*100,a['AUC']))
 (ROOT/'results_report_public.md').write_text('\n'.join(text)+'\n');print('INDEPENDENT_STRICT_VERIFICATION_PASS',len(rows),flush=True)
if __name__=='__main__':main()
