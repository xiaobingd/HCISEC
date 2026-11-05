"""Optional diffusion supplement on locked fit80 teacher authentication models."""
import sys,json,hashlib,subprocess,os
from pathlib import Path
import numpy as np
sys.path.insert(0,'/root')
import run_strict_autosupplement_20261029 as old
PRIOR=Path('/autodl-fs/data/mei_auth80_20261029')
ROOT=Path('/autodl-fs/data/mei_auth80_diffusion_20261029')
def save(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2));q.replace(p)
def target(u):
 folder=ROOT/('U%d'%u);folder.mkdir(parents=True,exist_ok=True)
 src=old.ROOT/('U%d'%u);previous=PRIOR/('U%d'%u);p=old.protocol(u)
 features=dict(np.load(str(src/'development_features_private.npz')))
 models=old.sel.joblib.load(str(previous/'models_private.joblib'))
 oof=dict(np.load(str(previous/'oof_private.npz')))
 locked0=json.loads((previous/'locked_before_queries.json').read_text())
 weights0=locked0['weights']['auto_without_diffusion']
 y=np.concatenate([p.labels[k] for k in ('auth_train','model_validation')])==u-1
 words=np.concatenate([np.asarray(old.M['sources'][str(v)]['content_groups'])[old.M['sources'][str(v)]['roles'][role]] for role in ('auth_train','model_validation') for v in old.USERS])
 ids=sum([p.sample_ids[k] for k in ('auth_train','model_validation')],[])
 blocked={str(v)+':'+str(i) for v in old.USERS for k in ('gan_fit','gan_val','calibration','test') for i in old.M['sources'][str(v)]['roles'][k]}
 assert len(set(ids))==720 and len(ids)==720 and not set(ids)&blocked and y.sum()==80
 assert set(np.unique(words))==set([5,6,7,20,21,22,23,24])
 folds=list(old.sel.StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42).split(np.zeros(len(y)),y,words))
 names=sorted(k.split('__',1)[1] for k in features if k.startswith('auth_train__diff_'))
 assert set(names)==set(['diff_latent','diff_guidance','diff_cycle','diff_bundle'])
 for name in names:
  x=np.concatenate([features[k+'__'+name] for k in ('auth_train','model_validation')])
  oof[name],models[name]=old.sel.nested_branch(x,y,words,folds)
  print('DIFF_BRANCH_DONE',u,name,flush=True)
 weights,selection=old.sel.forward_select(oof,y,folds,names,weights0,1)
 selected=[k for k in weights if k.startswith('diff_')]
 assert len(selected)<=1
 def score(role,name):
  m=models[name];return (old.sel.rbf.score(m['model'],[features[role+'__'+name]])-m['mean'])/m['std']
 cal={role:sum(w*score(role,k) for k,w in weights.items()) for role in ('target_dev_pool','impostor_calibration')}
 threshold=float(old.base.original.choose_threshold(cal['target_dev_pool'],cal['impostor_calibration']))
 save(folder/'locked_before_queries.json',dict(target=u,weights=weights,threshold=threshold,selection=selection,final_seen=False,teacher_reference_sha256=old.sha(previous/'locked_before_queries.json'),selected_diffusion=selected))
 old.sel.joblib.dump(models,str(folder/'models_private.joblib'));np.savez_compressed(str(folder/'oof_private.npz'),**oof)
 ref=json.loads((previous/'results_public.json').read_text());ss0=np.load(str(previous/'scores_private.npz'))
 scores={role:ss0['auto_without_diffusion__'+role].copy() for role in ('final_genuine','final_impostor')}
 if selected:
  path=src/('U%d_diffusion.pt'%u);saved=old.torch.load(str(path),map_location=old.D,weights_only=False)
  meta=json.loads((src/('U%d_diffusion_training.json'%u)).read_text());r=old.M['sources'][str(u)]['roles']
  assert meta['fit_ids']==r['gan_fit'] and meta['selection_ids']==r['gan_val']
  nets=[old.dp.BiometricUNet64().to(old.D),old.dp.BiometricUNet64().to(old.D)]
  for net,key in zip(nets,('a2u','u2a')):
   net.load_state_dict(saved['model_'+key+'_state_dict']);net.eval()
   for z in net.parameters():z.requires_grad_(False)
  schedule=old.dp.DiffusionSchedule(1000,device=old.D);gan=old.load_gan(u)
  with old.torch.no_grad():
   for role in scores:
    parts=[]
    for i in range(0,len(p.arrays[role]),8):
     x=old.te.tensor(p.arrays[role][i:i+8]);fake=gan.g_acc2audio(x)
     df=old.dp.diffusion_features(x,fake,nets,schedule)
     df['diff_bundle']=old.torch.cat([df[k] for k in ('diff_latent','diff_guidance','diff_cycle')],dim=1)
     parts.append(df[selected[0]].cpu().numpy())
    features[role+'__'+selected[0]]=np.concatenate(parts)
    scores[role]+=weights[selected[0]]*score(role,selected[0])
 else:assert threshold==locked0['thresholds']['auto_without_diffusion']
 result=old.sel.evaluate_route(scores,p,u,threshold);result['weights']=weights
 g=scores['final_genuine'];n=scores['final_impostor']
 assert abs(float(((g[:,None]>n)+.5*(g[:,None]==n)).mean())-result['auc'])<1e-12
 assert len(result['attack_by_user'])==8
 for v,b in result['attack_by_user'].items():
  s=n[p.labels['final_impostor']==int(v)-1];assert len(s)==b['n'] and int((s>=threshold).sum())==b['accepted']
 np.savez_compressed(str(folder/'scores_private.npz'),**scores,attack_labels=p.labels['final_impostor'])
 save(folder/'results_public.json',dict(target=u,route=result,reference=ref['routes'],selected_diffusion=selected,verification='PASS',diffusion_checkpoint_sha256=old.sha(src/('U%d_diffusion.pt'%u))))
 print('TARGET_DONE',u,result,flush=True)
def main():
 ROOT.mkdir(exist_ok=True)
 with (ROOT/'QUEUE_LOCK').open('x') as f:f.write(str(os.getpid()))
 save(ROOT/'protocol_public.json',dict(fit_positive=80,fit_negative=640,calibration_per_user=20,final_genuine='47-50 unchanged',closed_set=True,all_eight_attacks=True,upstream='Reused frozen strict personal generators, teachers and personal50epoch diffusion initialized from external ESMB0001; no upstream refit',selector='Original nested5outer/3inner content folds, max1 optional diffusion source on fixed fit80 teacher weights',seed=42,limitation='Existing final tests observed previously; exploratory, not blind',source_sha256=old.sha(Path(__file__))))
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

