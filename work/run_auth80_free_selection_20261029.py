"""Greedy train-only selection of zero to all available supplementary branches."""
import sys,json,os,time,subprocess
from pathlib import Path
import numpy as np
sys.path.insert(0,'/root')
import run_strict_autosupplement_20261029 as old
FIXED=Path('/autodl-fs/data/mei_auth80_diffusion_20261029')
TEACHER=Path('/autodl-fs/data/mei_auth80_20261029')
ROOT=Path('/autodl-fs/data/mei_auth80_free_selection_20261029')
def save(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2));q.replace(p)
def target(u):
 folder=ROOT/('U%d'%u);folder.mkdir(parents=True,exist_ok=True)
 src=old.ROOT/('U%d'%u);fixed=FIXED/('U%d'%u);p=old.protocol(u)
 features=dict(np.load(str(src/'development_features_private.npz')))
 models=old.sel.joblib.load(str(fixed/'models_private.joblib'));oof=dict(np.load(str(fixed/'oof_private.npz')))
 y=np.concatenate([p.labels[k] for k in ('auth_train','model_validation')])==u-1
 words=np.concatenate([np.asarray(old.M['sources'][str(v)]['content_groups'])[old.M['sources'][str(v)]['roles'][role]] for role in ('auth_train','model_validation') for v in old.USERS])
 assert len(y)==720 and y.sum()==80 and len(np.unique(words))==8
 ids=sum([p.sample_ids[k] for k in ('auth_train','model_validation')],[])
 blocked={str(v)+':'+str(i) for v in old.USERS for k in ('gan_fit','gan_val','calibration','test') for i in old.M['sources'][str(v)]['roles'][k]}
 assert len(set(ids))==720 and not set(ids)&blocked
 folds=list(old.sel.StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42).split(np.zeros(len(y)),y,words))
 allowed=sorted(k for k in oof if k!='base');assert len(allowed)==11
 # No family quotas: each extra source must improve current train-fold combination.
 weights,selection=old.sel.forward_select(oof,y,folds,allowed,{'base':1.},len(allowed))
 def scores(role,name):
  m=models[name];return (old.sel.rbf.score(m['model'],[features[role+'__'+name]])-m['mean'])/m['std']
 cal={role:sum(w*scores(role,k) for k,w in weights.items()) for role in ('target_dev_pool','impostor_calibration')}
 threshold=float(old.base.original.choose_threshold(cal['target_dev_pool'],cal['impostor_calibration']))
 locked=dict(target=u,weights=weights,threshold=threshold,selection=selection,final_seen=False,eligible_sources=allowed,quotas='none;0 to11 optional sources;772-dimensional base retained',model_reference_sha256=old.sha(fixed/'models_private.joblib'))
 save(folder/'locked_before_queries.json',locked);np.savez_compressed(str(folder/'oof_private.npz'),**oof)
 print('LOCKED',u,weights,flush=True)
 gan=old.load_gan(u);teachers={}
 for family,name in [('old','teacher'),('identity','identity_preserving_teacher'),('robust','domain_robust_teacher')]:
  saved=old.torch.load(str(src/('U%d_%s.pt'%(u,name))),map_location=old.D,weights_only=False)
  net=old.base.Teacher().to(old.D);net.load_state_dict(saved['state']);net.eval()
  for z in net.parameters():z.requires_grad_(False)
  teachers[family]=net
 saved=old.torch.load(str(src/('U%d_diffusion.pt'%u)),map_location=old.D,weights_only=False)
 meta=json.loads((src/('U%d_diffusion_training.json'%u)).read_text());r=old.M['sources'][str(u)]['roles']
 assert meta['fit_ids']==r['gan_fit'] and meta['selection_ids']==r['gan_val']
 nets=[old.dp.BiometricUNet64().to(old.D),old.dp.BiometricUNet64().to(old.D)]
 for net,key in zip(nets,('a2u','u2a')):
  net.load_state_dict(saved['model_'+key+'_state_dict']);net.eval()
  for z in net.parameters():z.requires_grad_(False)
 schedule=old.dp.DiffusionSchedule(1000,device=old.D)
 features.update(old.extract(gan,teachers,nets,schedule,p,('final_genuine','final_impostor')))
 # Cache all final feature branches after lock to allow independent score replay.
 np.savez_compressed(str(folder/'final_features_private.npz'),**{k:v for k,v in features.items() if k.startswith('final_')})
 ss={role:sum(w*scores(role,k) for k,w in weights.items()) for role in ('final_genuine','final_impostor')}
 result=old.sel.evaluate_route(ss,p,u,threshold);result['weights']=weights
 np.savez_compressed(str(folder/'scores_private.npz'),**ss,attack_labels=p.labels['final_impostor'])
 previous=json.loads((fixed/'results_public.json').read_text());ref=json.loads((TEACHER/('U%d'%u)/'results_public.json').read_text())
 save(folder/'results_public.json',dict(target=u,route=result,selected_count=len(weights)-1,reference=ref['routes'],fixed_quota_reference=previous['route'],weights=weights))
 print('TARGET_DONE',u,result,flush=True)
def main():
 ROOT.mkdir(exist_ok=True)
 with (ROOT/'QUEUE_LOCK').open('x') as f:f.write(str(os.getpid()))
 save(ROOT/'protocol_public.json',dict(fit_positive=80,fit_negative=640,base=772,optional_sources=11,selection='Greedy forward selection; no modality quota;0 through11 supplementary branches; weights .25/.5/1; nested5outer/3inner;each addition wins4/5folds and gain>max(.0025,pairedSE)',upstream_and_queries_unchanged=True,all_eight_attacks=True,seed=42,limitations=['Existing final observations; exploratory not blind','Single seed','Greedy heuristic not global optimum or significance test','Content mapping inferred from10/20period and missing-tail assumption'],source_sha256=old.sha(Path(__file__))))
 save(ROOT/'status_public.json',dict(state='waiting_for_fixed_quota_control'))
 while True:
  status=FIXED/'status_public.json'
  if status.exists() and json.loads(status.read_text()).get('state')=='complete':break
  time.sleep(5)
 rows={}
 for u in old.ORDER:
  save(ROOT/'status_public.json',dict(state='running',target=u,completed=list(rows)))
  folder=ROOT/('U%d'%u);folder.mkdir(exist_ok=True)
  with (folder/'run.log').open('w') as log:subprocess.run([sys.executable,'-u',__file__,str(u)],stdout=log,stderr=subprocess.STDOUT,check=True)
  rows[str(u)]=json.loads((folder/'results_public.json').read_text());save(ROOT/'results_public.json',rows)
 save(ROOT/'status_public.json',dict(state='complete',completed=list(rows)))
if __name__=='__main__':
 if len(sys.argv)==2:target(int(sys.argv[1]))
 else:main()

