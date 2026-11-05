"""Replay upstream selection, training OOF fusion, calibration and all final scores."""
import sys,json
import numpy as np
import torch
sys.path.insert(0,'/root')
import run_personalized_teacher_20261029 as run
old=run.old
for u in [4,5]:
 folder=run.ROOT/('U%d'%u);p=old.protocol(u);src=run.AUTH80/('U%d'%u)
 result=json.loads((folder/'results_public.json').read_text());lock=json.loads((folder/'locked_before_final.json').read_text());saved=np.load(str(folder/'scores_private.npz'));candidate_oof=np.load(str(folder/'oof_private.npz'));base_oof=np.load(str(src/'oof_private.npz'))
 cached=old.sel.joblib.load(str(src/'models_private.joblib'));features=np.load(str(old.ROOT/('U%d'%u)/'development_features_private.npz'));basefinal=np.load(str(src/'scores_private.npz'))
 y=np.concatenate([p.labels[r] for r in ['auth_train','model_validation']])==u-1;words=np.concatenate([run.first.words(p,r) for r in ['auth_train','model_validation']]);folds=list(old.sel.StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42).split(np.zeros(len(y)),y,words))
 assert np.array_equal(saved['attack_labels'],p.labels['final_impostor'])
 assert old.sha(old.UP/('U%d'%u)/'cyclegan/models/checkpoint_epoch_200.pth')==result['gan_sha256_unchanged']
 def standardized(m,x):return (old.sel.rbf.score(m['model'],[x])-m['mean'])/m['std']
 for key,r in result['routes'].items():
  spec=lock['routes'][key];ck=torch.load(str(folder/(key+'_teacher_private.pt')),map_location=run.D,weights_only=False);meta=ck['metadata'];best=min(meta['history'],key=lambda h:(-h['word_auc'],-h['auc'],h['bce'],h['step']));assert best['step']==meta['selected_step']==r['teacher_selected_step']
  assert meta['config']==run.settings(u,key) and meta['teacher_fit']==1080 and meta['teacher_val']==270
  t=old.base.Teacher().to(run.D);t.load_state_dict(ck['state']);t.eval();model=old.sel.joblib.load(str(folder/(key+'_rbf_private.joblib')))
  weights,selection=old.sel.forward_select({'base':base_oof['base'],'raw':base_oof['raw'],'teacher':candidate_oof[key]},y,folds,['raw','teacher'],{'base':1.},2)
  assert weights==spec['weights'] and abs(selection['mean_fold_auc']-spec['selection']['mean_fold_auc'])<1e-12
  def score(role):
   if role.startswith('final_'):s=basefinal['baseline_rbf__'+role].copy()
   else:s=standardized(cached['base'],features[role+'__base'])
   if 'raw' in weights:
    x=p.arrays[role];stats=np.concatenate([x.mean(2),x.std(2),x.mean(1),x.std(1)],1);s+=weights['raw']*standardized(cached['raw'],stats)
   if 'teacher' in weights:s+=weights['teacher']*standardized(model,run.embed(t,p.arrays[role]))
   return s
  threshold=old.base.original.choose_threshold(score('target_dev_pool'),score('impostor_calibration'));assert abs(threshold-spec['threshold'])<1e-12
  for role in ['final_genuine','final_impostor']:assert np.allclose(score(role),saved[key+'__'+role],atol=1e-7,rtol=1e-6)
  g=saved[key+'__final_genuine'];n=saved[key+'__final_impostor'];assert len(g)==r['genuine_n'] and len(n)==r['impostor_n']
  assert abs(((g[:,None]>n)+.5*(g[:,None]==n)).mean()-r['auc'])<1e-12
  assert abs((g>=threshold).mean()-r['tar'])<1e-12 and abs((n>=threshold).mean()-r['far'])<1e-12 and len(r['attack_by_user'])==8
  for v,a in r['attack_by_user'].items():
   s=n[saved['attack_labels']==int(v)-1];assert len(s)==a['n'] and int((s>=threshold).sum())==a['accepted']
 selected=min(['personal_mild','personal_strong'],key=lambda k:(-lock['routes'][k]['selection']['mean_fold_auc'],k));assert selected==result['personalized_choice']
 print('VERIFIED',u,selected,flush=True)
old.save(run.ROOT/'verification_public.json',dict(status='PASS',targets=[4,5],routes=6,checks=['upstream checkpoints replay','training OOF forward selection replay','personal parameter choice OOF replay','calibration threshold','saved teachers and classifiers final inference replay','AUC and acceptance counts all8attackers','unchanged G checkpoint SHA']))

