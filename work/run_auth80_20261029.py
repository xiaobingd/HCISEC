"""Increase authentication fit50 to fit80; fixed upstream, calibration and test."""
import sys,json,os,hashlib,subprocess
from pathlib import Path
import numpy as np
sys.path.insert(0,'/root')
import run_strict_autosupplement_20261029 as old
ROOT=Path('/autodl-fs/data/mei_auth80_20261029')
def save(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2));q.replace(p)
def target(u):
 folder=ROOT/('U%d'%u);folder.mkdir(parents=True,exist_ok=True)
 p=old.protocol(u);src=old.ROOT/('U%d'%u)
 features=dict(np.load(str(src/'development_features_private.npz')))
 names=sorted(k.split('__',1)[1] for k in features if k.startswith('auth_train__') and not k.split('__',1)[1].startswith('diff_'))
 y=np.concatenate([p.labels[k] for k in ('auth_train','model_validation')])==u-1
 words=np.concatenate([np.asarray(old.M['sources'][str(v)]['content_groups'])[old.M['sources'][str(v)]['roles'][role]] for role in ('auth_train','model_validation') for v in old.USERS])
 ids=sum([p.sample_ids[k] for k in ('auth_train','model_validation')],[])
 upstream={str(v)+':'+str(i) for v in old.USERS for k in ('gan_fit','gan_val') for i in old.M['sources'][str(v)]['roles'][k]}
 final=set(p.sample_ids['final_genuine']+p.sample_ids['final_impostor'])
 cal=set(p.sample_ids['target_dev_pool']+p.sample_ids['impostor_calibration'])
 assert len(ids)==720 and len(set(ids))==720 and y.sum()==80 and (~y).sum()==640
 assert not set(ids)&(upstream|final|cal) and len(np.unique(words))==8
 folds=list(old.sel.StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42).split(np.zeros(len(y)),y,words))
 oof={};models={}
 for name in names:
  x=np.concatenate([features[k+'__'+name] for k in ('auth_train','model_validation')])
  oof[name],models[name]=old.sel.nested_branch(x,y,words,folds)
  print('BRANCH_DONE',u,name,flush=True)
 weights,selection=old.sel.forward_select(oof,y,folds,[k for k in names if k!='base'],{'base':1.},2)
 routes={'baseline_rbf':{'base':1.},'auto_without_diffusion':weights}
 def score(role,name):
  m=models[name];return (old.sel.rbf.score(m['model'],[features[role+'__'+name]])-m['mean'])/m['std']
 thresholds={k:float(old.base.original.choose_threshold(sum(w*score('target_dev_pool',n) for n,w in ww.items()),sum(w*score('impostor_calibration',n) for n,w in ww.items()))) for k,ww in routes.items()}
 save(folder/'locked_before_queries.json',dict(target=u,weights=routes,thresholds=thresholds,selection=selection,final_seen=False,fit_positive=80,fit_negative=640,content_groups=sorted(np.unique(words).tolist())))
 old.sel.joblib.dump(models,str(folder/'models_private.joblib'));np.savez_compressed(str(folder/'oof_private.npz'),**oof)
 # Upstream weights unchanged, extract final ACC only after classifiers and thresholds lock.
 model=old.load_gan(u);teachers={}
 for family,name in [('old','teacher'),('identity','identity_preserving_teacher'),('robust','domain_robust_teacher')]:
  saved=old.torch.load(str(src/('U%d_%s.pt'%(u,name))),map_location=old.D,weights_only=False)
  t=old.base.Teacher().to(old.D);t.load_state_dict(saved['state']);t.eval()
  for z in t.parameters():z.requires_grad_(False)
  teachers[family]=t
 with old.torch.no_grad():
  for role in ('final_genuine','final_impostor'):
   parts={k:[] for k in names}
   for i in range(0,len(p.arrays[role]),16):
    raw=p.arrays[role][i:i+16];x=old.te.tensor(raw);fake=model.g_acc2audio(x)
    z={'base':old.base.fusion_inputs(model,x),'raw':old.torch.as_tensor(np.concatenate([raw.mean(2),raw.std(2),raw.mean(1),raw.std(1)],1),device=old.D)}
    for family,t in teachers.items():z[family+'_acc']=t.acc(x);z[family+'_audio']=t.audio(fake)
    for k in names:parts[k].append(z[k].cpu().numpy())
   for k in names:features[role+'__'+k]=np.concatenate(parts[k])
 results={};dump={}
 for k,ww in routes.items():
  ss={role:sum(w*score(role,n) for n,w in ww.items()) for role in ('final_genuine','final_impostor')}
  r=old.sel.evaluate_route(ss,p,u,thresholds[k]);r['weights']=ww;results[k]=r
  g=ss['final_genuine'];n=ss['final_impostor'];t=thresholds[k]
  assert abs(r['auc']-float(((g[:,None]>n)+.5*(g[:,None]==n)).mean()))<1e-12
  assert abs(r['tar']-np.mean(g>=t))<1e-12 and abs(r['far']-np.mean(n>=t))<1e-12 and len(r['attack_by_user'])==8
  for role,s in ss.items():dump[k+'__'+role]=s
 np.savez_compressed(str(folder/'scores_private.npz'),**dump,attack_labels=p.labels['final_impostor'])
 reference=json.loads((src/'results_public.json').read_text())['routes']
 save(folder/'results_public.json',dict(target=u,routes=results,fit50_reference={k:reference[k] for k in routes},verification='PASS',fit_positive=80,fit_negative=640))
 print('TARGET_DONE',u,results,flush=True)
def main():
 ROOT.mkdir(exist_ok=True)
 with (ROOT/'QUEUE_LOCK').open('x') as f:f.write(str(os.getpid()))
 save(ROOT/'protocol_public.json',dict(fit_positive=80,fit_negative=640,fit_groups=[5,6,7,20,21,22,23,24],calibration_groups=[8,25],test_groups=[9,26,27,28,29],upstream_unchanged=True,closed_set=True,all_eight_attack_sources=True,seed=42,limitation='Existing test observations; exploratory repeated evaluation, not a fresh blind holdout',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
 results={}
 for u in old.ORDER:
  save(ROOT/'status_public.json',dict(state='running',target=u,completed=list(results)))
  folder=ROOT/('U%d'%u);folder.mkdir(exist_ok=True)
  with (folder/'run.log').open('w') as log:subprocess.run([sys.executable,'-u',__file__,str(u)],stdout=log,stderr=subprocess.STDOUT,check=True)
  results[str(u)]=json.loads((folder/'results_public.json').read_text());save(ROOT/'results_public.json',results)
 save(ROOT/'status_public.json',dict(state='complete',completed=list(results)))
if __name__=='__main__':
 if len(sys.argv)==2:target(int(sys.argv[1]))
 else:main()

