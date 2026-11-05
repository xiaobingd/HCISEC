"""Verify encoder-only changes, gradient routes, data isolation and locked authentication scores."""
import sys,json
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,'/root')
import run_encoder_identity_feedback_20261029 as run
trial=run.trial;old=run.old;ROOT=run.ROOT
def check(g,n,t,r):
 assert len(g)==r['genuine_n'] and len(n)==r['impostor_n']
 assert abs(((g[:,None]>n)+.5*(g[:,None]==n)).mean()-r['auc'])<1e-12
 assert abs((g>=t).mean()-r['tar'])<1e-12 and abs((n>=t).mean()-r['far'])<1e-12
 assert abs((1-r['tar']+r['far'])/2-r['hter'])<1e-12
 r['genuine_accepted']=int((g>=t).sum());r['attack_accepted']=int((n>=t).sum())
for v,source in old.M['sources'].items():
 up=set(source['roles']['gan_fit']+source['roles']['gan_val']);auth=set(sum([source['roles'][k] for k in ['auth_train','model_validation','calibration','test']],[]))
 assert not up&auth and len(source['roles']['gan_fit'])==120
rows={}
for u in [4,1,5]:
 folder=ROOT/('U%d'%u);a=json.loads((folder/'results_public.json').read_text());p=old.protocol(u);up=run.up_data(u)
 scaler=old.sel.joblib.load(str(folder/'upstream_scaler_private.joblib'));assert int(scaler.n_samples_seen_)==1080;stats=run.moments(scaler)
 original=old.load_gan(u);_,allowed=run.enc_setup(original,False);initial=run.enc_state(original,allowed);frozen_sha=run.frozen_hash(original,allowed)
 baseline_cv,baseline_fake=run.conversion(original,up['own_val_acc'],up['own_val_audio'])
 for key,value in baseline_cv.items():assert abs(value-a['baseline_upstream_conversion'][key])<2e-6
 base_features=run.feature_array(original,up['a']);assert np.allclose(base_features.mean(0),scaler.mean_,atol=2e-5,rtol=2e-5)
 del original
 for variant,r in a['routes'].items():
  sub=folder/variant;ck=torch.load(str(sub/'upstream_private.pt'),map_location='cpu',weights_only=False)
  assert ck['encoder_names']==allowed and set(ck['encoder_updates'])==set(allowed)
  assert ck['frozen_decoder_reverse_sha256']==frozen_sha
  delta=run.norm_grad([ck['encoder_updates'][k]-initial[k] for k in allowed]);assert abs(delta-r['encoder_delta_l2'])<1e-12
  if variant=='frozen':assert delta==0 and ck['gradient_audit'] is None
  elif variant=='conversion_only':assert ck['gradient_audit']['identity_encoder_grad_norm']==ck['gradient_audit']['identity_generated_audio_grad_norm']==0
  else:assert ck['gradient_audit']['identity_encoder_grad_norm']>0 and ck['gradient_audit']['identity_generated_audio_grad_norm']>0
  passing=[q for q in ck['history'] if q['passes_conversion_guard']]
  selected=min(passing,key=lambda q:(-q['identity_sameword_auc'],-q['identity_auc'],q['conversion']['paired_l1']+q['conversion']['cycle_l1'],q['step']))
  assert selected['step']==r['selected_upstream_step']==ck['selected_upstream_step']
  m=old.load_gan(u);run.apply_enc(m,ck['encoder_updates']);assert run.frozen_hash(m,allowed)==frozen_sha
  netA=trial.Adapter().to(trial.DEVICE);netA.load_state_dict(ck['upstream_adapter']);netA.eval()
  q=run.quality_up(m,netA,stats,up,u,baseline_cv,baseline_fake)
  assert q['passes_conversion_guard']
  for k in ['identity_auc','identity_sameword_auc']:assert abs(q[k]-ck['upstream_quality'][k])<1e-6
  if variant!='frozen':
   enc,_=run.enc_setup(m,True);ix,_=run.sample_batch(up['labels'],up['words'],np.random.RandomState(142))
   f,fake,_=run.differentiable_features(m,run.spectrogram(up['a'][ix]));loss=run.identity_loss(netA,f.detach() if variant=='conversion_only' else f,stats,up['labels'][ix],up['words'][ix])
   gg=torch.autograd.grad(loss,enc+[fake],allow_unused=True)
   if variant=='conversion_only':assert run.norm_grad(gg)==0
   else:assert run.norm_grad(gg[:-1])>0 and run.norm_grad(gg[-1:])>0
   run.enc_setup(m,False)
  dev=np.load(str(sub/'development_features_private.npz'));final=np.load(str(sub/'final_features_private.npz'));saved=np.load(str(sub/'final_scores_private.npz'))
  assert np.array_equal(saved['attack_labels'],p.labels['final_impostor'])
  # Regenerate samples spanning all stages to check checkpoint/feature provenance.
  for role,source in [('auth_train',dev),('model_validation',dev),('target_dev_pool',dev),('final_genuine',final),('final_impostor',final)]:
   ix=np.unique(np.linspace(0,len(p.arrays[role])-1,min(4,len(p.arrays[role]))).astype(int));f=run.feature_array(m,p.arrays[role][ix]);assert np.allclose(f,source[role][ix],atol=2e-4,rtol=2e-4)
  head=torch.load(str(sub/'auth_head_private.pt'),map_location=trial.DEVICE,weights_only=False);lock=json.loads((sub/'auth_locked_before_final.json').read_text())
  best=min(lock['history'],key=lambda row:(-row['validation_sameword_auc'],-row['validation_auc'],row['validation_ce'],row['step']));assert best['step']==head['selected_step']==r['selected_auth_step']
  net=trial.Adapter().to(trial.DEVICE);net.load_state_dict(head['state']);net.eval()
  with torch.no_grad():center=trial.centers(net(run.normalized_features(run.tensor(dev['auth_train']),stats)),p.labels['auth_train'])[u-1]
  def predict(features):
   with torch.no_grad():return (net(run.normalized_features(run.tensor(features),stats))@center).cpu().numpy()
  cg=predict(dev['target_dev_pool']);cn=predict(dev['impostor_calibration']);assert abs(old.base.original.choose_threshold(cg,cn)-r['threshold'])<2e-6
  for role in ['final_genuine','final_impostor']:assert np.allclose(predict(final[role]),saved[role],atol=2e-6,rtol=1e-6)
  g=saved['final_genuine'];n=saved['final_impostor'];check(g,n,r['threshold'],r);assert len(r['attack_by_user'])==8
  for v,b in r['attack_by_user'].items():
   s=n[saved['attack_labels']==int(v)-1];assert len(s)==b['n'] and int((s>=r['threshold']).sum())==b['accepted']
  def words(role):return np.asarray([old.M['sources'][s.split(':')[0]]['content_groups'][int(s.split(':')[1])] for s in p.sample_ids[role]])
  wg=words('final_genuine');wn=words('final_impostor')
  with torch.no_grad():zg=net(run.normalized_features(run.tensor(final['final_genuine']),stats)).cpu().numpy();zn=net(run.normalized_features(run.tensor(final['final_impostor']),stats)).cpu().numpy()
  intra=(zg@zg.T)[wg[:,None]!=wg[None,:]];inter=(zg@zn.T)[wg[:,None]==wn[None,:]]
  r['target_crossword_intra_cos']=float(intra.mean());r['target_sameword_inter_cos']=float(inter.mean());r['identity_cosine_gap']=float(intra.mean()-inter.mean())
  audio=np.load(str(old.DATA/('audio.%d.npy'%u))).astype('float32')[:,:80,:80];ix=[int(s.split(':')[1]) for s in p.sample_ids['final_genuine']]
  cv,_=run.conversion(m,p.arrays['final_genuine'],audio[ix])
  for k,value in cv.items():assert abs(value-r['final_conversion_diagnostic'][k])<2e-6
  del m,net,netA
 rows[str(u)]=a
macro={variant:{k:float(np.mean([a['routes'][variant][k] for a in rows.values()])) for k in ['tar','far','auc','hter','identity_cosine_gap']} for variant in run.VARIANTS}
out=dict(status='PASS',targets=3,seed=42,results=rows,macro=macro,verification='Upstream/auth disjointness;scaler1080 upstream stats;only6 allowed G tensors change;decoder and F hashes fixed;live identity encoder/fake gradients;upstream guarded and auth selection replay;checkpoint features,calibration,AUC/counts/eight attackers and paired conversion reproduced')
(ROOT/'verified_summary_public.json').write_text(json.dumps(out,indent=2));print(json.dumps(dict(status='PASS',macro=macro)))

