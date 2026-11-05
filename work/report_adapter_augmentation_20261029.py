"""Verify training-only augmentation banks, checkpoint replay and unchanged clean test controls."""
import sys,json,hashlib
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
sys.path.insert(0,'/root')
import run_adapter_augmentation_20261029 as run
trial=run.trial;old=run.old;ROOT=run.ROOT
def checks():
 x=np.tile(np.arange(8,dtype='float32'),(2,3,1));z=run.shifted(x,np.asarray([2,-2]))
 assert np.array_equal(z[0,:,:2],np.zeros((3,2))) and np.array_equal(z[0,:,2:],x[0,:,:-2])
 assert np.array_equal(z[1,:,-2:],np.full((3,2),7)) and np.array_equal(z[1,:,:-2],x[1,:,2:])
 return 'PASS non-circular time-axis shift and edge fill'
def metrics(g,n,t,r):
 assert len(g)==r['genuine_n'] and len(n)==r['impostor_n']
 assert abs(((g[:,None]>n)+.5*(g[:,None]==n)).mean()-r['auc'])<1e-12
 assert abs((g>=t).mean()-r['tar'])<1e-12 and abs((n>=t).mean()-r['far'])<1e-12
 assert abs((1-r['tar']+r['far'])/2-r['hter'])<1e-12
 r['genuine_accepted']=int((g>=t).sum());r['attack_accepted']=int((n>=t).sum())
augment_check=checks();rows={}
for u in [1,4,5]:
 folder=ROOT/('U%d'%u);src=trial.ROOT/('U%d'%u);p=old.protocol(u)
 a=json.loads((folder/'results_public.json').read_text());lock=json.loads((folder/'locked_before_final.json').read_text())
 meta=json.loads((folder/'augmentation_public.json').read_text());raw=p.arrays['auth_train']
 assert meta['source_ids_sha256']==hashlib.sha256(json.dumps(p.sample_ids['auth_train']).encode()).hexdigest()
 assert len(raw)==meta['source_recordings']==450 and meta['noise_sigma']==float(.02*np.median(raw.std(axis=(1,2))))
 blocked=set(sum([p.sample_ids[k] for k in ['model_validation','target_dev_pool','impostor_calibration','final_genuine','final_impostor']],[]))
 assert not set(p.sample_ids['auth_train'])&blocked
 dev=np.load(str(old.ROOT/('U%d'%u)/'development_features_private.npz'));final=np.load(str(trial.FINAL/('U%d'%u)/'final_features_private.npz'))
 scaler=old.sel.joblib.load(str(folder/'scaler_private.joblib'));original_scaler=old.sel.joblib.load(str(src/'scaler_private.joblib'))
 assert int(scaler.n_samples_seen_)==450 and np.array_equal(scaler.mean_,original_scaler.mean_) and np.array_equal(scaler.scale_,original_scaler.scale_)
 banks=np.load(str(folder/'augmented_train_features_private.npz'));gan=old.load_gan(u);drift={}
 sample=np.asarray([0,49,50,249,449])
 with torch.no_grad():
  for kind in run.AUGS:
   assert banks[kind].shape==(2,450,772)
   drift[kind]=[]
   for view in range(2):
    perturbed=run.augment(raw,view,kind,meta['noise_sigma'],meta['train_min'],meta['train_max'])
    regenerated=old.base.fusion_inputs(gan,old.te.tensor(perturbed[sample])).cpu().numpy()
    assert np.allclose(regenerated,banks[kind][view,sample],atol=2e-4,rtol=2e-4)
    clean=scaler.transform(dev['auth_train__base']);aug=scaler.transform(banks[kind][view]);cos=(clean*aug).sum(1)/(np.linalg.norm(clean,axis=1)*np.linalg.norm(aug,axis=1)+1e-12)
    drift[kind].append(float(np.median(cos)))
 del gan
 x=trial.tensor(scaler.transform(dev['auth_train__base']));labels=p.labels['auth_train']
 scores=np.load(str(folder/'scores_private.npz'));reference=np.load(str(src/'scores_private.npz'));reference_meta=json.loads((src/'results_public.json').read_text())
 assert np.array_equal(scores['attack_labels'],p.labels['final_impostor']) and np.array_equal(reference['attack_labels'],p.labels['final_impostor'])
 def wordids(role):return np.asarray([old.M['sources'][s.split(':')[0]]['content_groups'][int(s.split(':')[1])] for s in p.sample_ids[role]])
 wg=wordids('final_genuine');wn=wordids('final_impostor')
 for key,r in a['routes'].items():
  if r['augmentation']=='none':
   original_key=r['mode']+'_seed%d'%r['seed'];g=reference[original_key+'__final_genuine'];n=reference[original_key+'__final_impostor']
   for m in ['tar','far','auc','hter','threshold']:assert r[m]==reference_meta['routes'][original_key][m]
  else:
   hist=lock['history'][key];best=min(hist,key=lambda row:(-row['validation_sameword_auc'],-row['validation_auc'],row['validation_multiclass_ce'],row['step']));assert best['step']==r['selected_step']
   model=trial.Adapter().to(trial.DEVICE);ck=torch.load(str(folder/(key+'_private.pt')),map_location=trial.DEVICE,weights_only=False);model.load_state_dict(ck['state']);model.eval()
   def predict(raw_features):
    with torch.no_grad():cc=trial.centers(model(x),labels);return (model(trial.tensor(scaler.transform(raw_features)))@cc[u-1]).cpu().numpy()
   val=predict(dev['model_validation__base']);q=trial.quality(val,p.labels['model_validation']==u-1,wordids('model_validation'))
   assert abs(q[0]-r['validation_sameword_auc'])<1e-6 and abs(q[1]-r['validation_auc'])<1e-6
   cg=predict(dev['target_dev_pool__base']);cn=predict(dev['impostor_calibration__base']);assert abs(old.base.original.choose_threshold(cg,cn)-r['threshold'])<2e-6
   for role in ['final_genuine','final_impostor']:assert np.allclose(predict(final[role+'__base']),scores[key+'__'+role],atol=2e-6,rtol=1e-6)
   g=scores[key+'__final_genuine'];n=scores[key+'__final_impostor']
  metrics(g,n,r['threshold'],r);assert len(r['attack_by_user'])==8
  for v,b in r['attack_by_user'].items():
   s=n[p.labels['final_impostor']==int(v)-1];assert len(s)==b['n'] and int((s>=r['threshold']).sum())==b['accepted']
  for w,rr in r['per_word'].items():metrics(g[wg==int(w)],n[wn==int(w)],r['threshold'],rr)
 summary={}
 for mode in run.MODES:
  for aug in ['none']+run.AUGS:
   key=mode+'__'+aug
   summary[key]={m:dict(mean=float(np.mean([a['routes'][key+'_seed%d'%s][m] for s in trial.SEEDS])),std=float(np.std([a['routes'][key+'_seed%d'%s][m] for s in trial.SEEDS],ddof=1))) for m in ['tar','far','auc','hter','same_word_mean_auc']}
 a['seed_summary']=summary;a['training_feature_median_cosine_drift']=drift;rows[str(u)]=a
macro={key:{m:float(np.mean([a['seed_summary'][key][m]['mean'] for a in rows.values()])) for m in ['tar','far','auc','hter','same_word_mean_auc']} for key in rows['1']['seed_summary']}
out=dict(status='PASS',targets=3,routes_verified=72,results=rows,macro=macro,augmentation_checks=augment_check,verification='Regenerated sampled augmentation features through frozen G/F;row hashes and split disjointness;clean scaler equality;checkpoint/validation/calibration score replay;unchanged controls;all eight attacks/per-word AUC/counts;3seeds averaged')
(ROOT/'verified_summary_public.json').write_text(json.dumps(out,indent=2));print(json.dumps(dict(status='PASS',macro=macro)))

