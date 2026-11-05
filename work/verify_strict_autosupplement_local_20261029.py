"""Independent local result and development-only selection checks."""
import sys,json,zipfile,hashlib
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'work/public_signal_runtime_20261028'))
import numpy as np
R=Path('outputs/mei_strict_autosupplement_20261029')
def auc(y,s):
 p=s[y];n=s[~y];return float(((p[:,None]>n[None,:])+.5*(p[:,None]==n[None,:])).mean())
def check_select(oof,y,words,start,end,allowed,max_add):
 masks=[words==w for w in np.unique(words)]
 def values(s):return np.asarray([auc(y[m],s[m]) for m in masks])
 current=dict(start);extra=[k for k in end if k not in current]
 assert len(extra)<=max_add
 for step in range(max_add):
  score=sum(w*oof[k] for k,w in current.items());before=values(score);candidates=[]
  for name in allowed:
   if name in current:continue
   for w in (.25,.5,1.):
    after=values(score+w*oof[name]);gains=after-before;mean=float(gains.mean());se=float(gains.std(ddof=1)/np.sqrt(5));ok=int(np.sum(gains>1e-6))>=4 and mean>max(.0025,se)
    if ok:candidates.append((float(after.mean()),name,w))
  if step>=len(extra):assert not candidates;break
  name=extra[step];w=end[name];matching=[a for a in candidates if a[1:]==(name,w)]
  assert matching and matching[0][0]>=max(a[0] for a in candidates)-1e-12
  current[name]=w
 assert current==end
def main():
 with zipfile.ZipFile(R/'review_private.zip') as z:
  assert z.testzip() is None;assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in z.namelist());z.extractall(R/'review_private')
 P=R/'review_private';m=json.loads((P/'strict_manifest_private.json').read_text());r=json.loads((R/'verified_results_public.json').read_text());assert len(r['results'])==9
 words=np.concatenate([np.asarray(m['sources'][str(v)]['content_groups'])[m['sources'][str(v)]['roles']['auth_train']] for v in range(1,10)]);uid=np.repeat(np.arange(1,10),50);count=0
 for u,row in r['results'].items():
  folder=P/('U%s'%u);lock=json.loads((folder/'locked_before_final.json').read_text());s=np.load(folder/'scores_private.npz');oof=dict(np.load(folder/'oof_private.npz'));y=uid==int(u)
  allowed=[k for k in sorted(oof) if k!='base' and not k.startswith('diff_')];ww=lock['weights']
  assert ww['baseline_rbf']=={'base':1.}
  check_select(oof,y,words,{'base':1.},ww['auto_without_diffusion'],allowed,2)
  check_select(oof,y,words,ww['auto_without_diffusion'],ww['auto_with_diffusion'],[k for k in sorted(oof) if k.startswith('diff_')],1)
  meta=json.loads((folder/('U%s_diffusion_training.json'%u)).read_text());roles=m['sources'][u]['roles'];assert meta['fit_ids']==roles['gan_fit'] and meta['selection_ids']==roles['gan_val'] and meta['epochs']==50
  for route,a in row['routes'].items():
   g=s[route+'__final_genuine'];n=s[route+'__final_impostor'];t=lock['thresholds'][route];ys=np.r_[np.ones(len(g),bool),np.zeros(len(n),bool)]
   assert len(g)==a['genuine_n']==len(roles['test']);assert len(n)==a['impostor_n']==sum(len(v['roles']['test']) for k,v in m['sources'].items() if k!=u)
   assert abs(auc(ys,np.r_[g,n])-a['auc'])<1e-12 and abs(np.mean(g>=t)-a['tar'])<1e-12 and abs(np.mean(n>=t)-a['far'])<1e-12
   for v,b in a['attack_by_user'].items():z=n[s['attack_labels']==int(v)-1];assert len(z)==b['n'] and int((z>=t).sum())==b['accepted']
   assert a['weights']==ww[route];count+=1
 out=dict(status='PASS',targets=9,routes=count,zip_crc=True,pairwise_auc=True,all_source_counts=True,upstream_diffusion_groups=True,training_only_selection_replayed=True)
 (R/'local_verification_public.json').write_text(json.dumps(out,indent=2));print(out)
if __name__=='__main__':main()
