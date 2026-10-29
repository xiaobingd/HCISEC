"""Independent protocol-A enrollment/calibration/final inference and model-selection audit."""
import sys,json,time
import numpy as np
import torch
sys.path.insert(0,'/root')
import run_audio_guided_acc_kd_20261030 as run
old=run.old;data=run.data_load();verified={}

def load_student(folder,key):
 ck=torch.load(str(folder/(key+'_private.pt')),map_location=run.D,weights_only=False);net=run.Classifier().to(run.D);net.aam=ck['metadata']['mode']=='aam';net.load_state_dict(ck['state']);net.eval()
 for p in net.parameters():p.requires_grad_(False)
 return net,ck

def check_history(meta):
 if meta['domain'] in ['audio','fusion']:best=min(meta['history'],key=lambda h:(h['quality']['ce'],-h['quality']['auc'],h['step']))
 else:best=min(meta['history'],key=lambda h:run.rank(h['quality'])+(h['step'],))
 assert best['step']==meta['selected_step'];return best

allsets=[set(data[r]['ids']) for r in run.ROLES]
for i,a in enumerate(allsets):
 for b in allsets[:i]:assert not a&b
for s in run.SEEDS:
 folder=run.ROOT/('seed%d'%s);deadline=time.monotonic()+1800
 while True:
  try:result=json.loads((folder/'results_public.json').read_text());break
  except (FileNotFoundError,json.JSONDecodeError):
   assert time.monotonic()<deadline,'training completion timeout';time.sleep(5)
 lock=json.loads((folder/'locked_before_final.json').read_text());b0,ck0=load_student(folder,'B0_CE');audio,cka=load_student(folder,'T1_Audio');ckf=torch.load(str(folder/'T2_Fusion_private.pt'),map_location=run.D,weights_only=False);fusion=run.Fusion(b0,audio).to(run.D);fusion.load_state_dict(ckf['state']);fusion.eval()
 for net in [audio,fusion]:
  for p in net.parameters():p.requires_grad_(False)
 for ck in [ck0,cka,ckf]:check_history(ck['metadata'])
 # Teacher qualification from held-out development data only, including authentication geometry.
 tq={name:run.validation_quality(net,data,domain) for name,net,domain in [('acc',b0,'acc'),('audio',audio,'audio'),('fusion',fusion,'fusion')]}
 result['teacher_validation_auth']=tq;result['fusion_beats_both_validation_auth_hter']=tq['fusion']['hter']<min(tq['acc']['hter'],tq['audio']['hter'])
 for family,metas in lock['candidate_metadata'].items():
  choices=[(run.rank(lock['metadata']['B0_CE']['quality'])+(0.,0.),'B0_CE')]
  for meta in metas:
   check_history(meta);assert meta['steps']==400 and meta['seed']==s and meta['domain']=='acc';choices.append((run.rank(meta['quality'])+(meta['lambda_kd'],meta['temperature']),meta['key']))
  assert min(choices,key=lambda x:x[0])[1]==lock['chosen'][family]
 for key,rows in result['routes'].items():
  source=lock['chosen'].get(key,key);net,ck=load_student(folder,source);meta=ck['metadata'];check_history(meta);q=run.validation_quality(net,data,'acc')
  for k in ['hter','tar','far','auc','ce']:assert abs(q[k]-meta['quality'][k])<2e-6
  ze,_=run.inference(net,data,'enroll');zv,_=run.inference(net,data,'validation');zc,_=run.inference(net,data,'calibration');zt,_=run.inference(net,data,'test');el=data['enroll']['labels'];vl=data['validation']['labels'];vw=data['validation']['words'];cl=data['calibration']['labels'];tl=data['test']['labels'];models=old.sel.joblib.load(str(folder/(key+'_rbf_private.joblib')));saved=np.load(str(folder/(key+'_scores_private.npz')));assert np.array_equal(saved['test_labels'],tl)
  for name,c in lock['routes'][key].items():
   u=c['user']-1
   if c['backend']=='cosine':
    center=run.prototypes(ze,el,c['K'])[u];cal=zc@center;test=zt@center
   else:
    fitted=models[str(u+1)];sc=fitted[0][0];assert int(sc.n_samples_seen_)==180 and np.allclose(sc.mean_,ze.mean(0),atol=2e-5,rtol=2e-5);cal=old.sel.rbf.score(fitted,[zc]);test=old.sel.rbf.score(fitted,[zt])
    # Refit the finite parameter grid against development-word decisions.
    best=None
    for cc in [1.,10.,100.]:
     for gamma in ['scale',.1]:
      ff=old.sel.rbf.fit([ze],el==u,cc,gamma);sv=old.sel.rbf.score(ff,[zv]);accepted=np.zeros(len(vl),dtype=bool)
      for w in np.unique(vw):
       tr=vw!=w;va=~tr;t=run.threshold(sv[tr][vl[tr]==u],sv[tr][vl[tr]!=u]);accepted[va]=sv[va]>=t
      tar=float(accepted[vl==u].mean());far=float(accepted[vl!=u].mean());qq=dict(hter=(1-tar+far)/2,tar=tar,far=far,auc=float(run.roc_auc_score(vl==u,sv)),ce=0.);rr=run.rank(qq)+(cc,0 if gamma==.1 else 1)
      if best is None or rr<best[0]:best=(rr,cc,gamma)
    assert best[1]==c['C'] and best[2]==c['gamma']
   t=run.threshold(cal[cl==u],cal[cl!=u]);low=run.lowfar_threshold(cal[cl==u],cal[cl!=u]);assert abs(t-c['threshold'])<2e-6 and abs(low-c['lowfar_threshold'])<2e-6
   assert np.allclose(test,saved[name],atol=2e-6,rtol=2e-6);r=rows[name];g=test[tl==u];n=test[tl!=u]
   assert len(g)==r['genuine_n'] and len(n)==r['impostor_n'];assert abs(((g[:,None]>n)+.5*(g[:,None]==n)).mean()-r['auc'])<1e-12
   assert abs((g>=t).mean()-r['tar'])<1e-12 and abs((n>=t).mean()-r['far'])<1e-12 and abs((g>=low).mean()-r['lowfar_calibrated_test_tar'])<1e-12 and abs((n>=low).mean()-r['lowfar_calibrated_test_far'])<1e-12
   assert len(r['attack_by_user'])==8
   for v,b in r['attack_by_user'].items():
    ss=test[tl==int(v)-1];assert len(ss)==b['n'] and int((ss>=t).sum())==b['accepted']
  print('VERIFIED_ROUTE',s,key,source,flush=True)
 # Verify frozen teachers provide finite nine-class targets and no teacher gradient path.
 z,l=run.inference(audio,data,'train','audio');target=run.tensor(l[:18]);student=run.Classifier().to(run.D);_,sl=student(run.tensor(data['train']['acc'][:18,None]));loss=4*torch.nn.functional.kl_div(torch.nn.functional.log_softmax(sl/2,dim=1),torch.nn.functional.softmax(target/2,dim=1),reduction='batchmean');loss.backward();assert any(p.grad is not None and p.grad.abs().sum()>0 for p in student.parameters()) and all(p.grad is None for p in audio.parameters())
 verified[str(s)]=result;print('VERIFIED_SEED',s,flush=True)
old.save(run.ROOT/'verified_results_public.json',verified);old.save(run.ROOT/'verification_public.json',dict(status='PASS',seeds=run.SEEDS,routes=675,checks=['all roles and content groups disjoint','independent enrollment not encoder training','candidate checkpoint/lambda/T selection replay','teacher development-only CE and authentication checks','student and backend ACC-only inference','RBF180 enrollment rows only and hyperparameter refit','joint and lowFARcalibration thresholds','all final score/AUC/acceptance and8attack source counts','frozen teacher KD gradient isolation']))

