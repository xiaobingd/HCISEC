import sys,json,time
import numpy as np
import torch
sys.path.insert(0,'/root')
import run_u4_normalization_20261030 as run
prev=run.p;old=prev.old;data=prev.data_load();verified={}
for seed in prev.SEEDS:
 folder=run.ROOT/('seed%d'%seed);deadline=time.monotonic()+1800
 while True:
  try:r=json.loads((folder/'results_public.json').read_text());break
  except (FileNotFoundError,json.JSONDecodeError):assert time.monotonic()<deadline;time.sleep(5)
 lock=json.loads((folder/'locked_before_final.json').read_text());stats=lock['stats'];x=data['train']['acc'];assert abs(float(x.mean())-stats['mean'])<1e-7 and abs(float(x.std())-stats['std'])<1e-7 and stats['fit_n']==1620
 for key,rows in r['routes'].items():
  ck=torch.load(str(folder/(key+'_private.pt')),map_location=prev.D,weights_only=False);net=run.Model(key,stats).to(prev.D);net.load_state_dict(ck['state']);net.eval();meta=ck['metadata'];q=prev.validation_quality(net,data,'acc');assert min(meta['history'],key=lambda h:prev.rank(h['quality'])+(h['step'],))['step']==meta['selected']['step']
  for k in ['hter','tar','far','auc','ce']:assert abs(q[k]-meta['selected']['quality'][k])<2e-6
  ze,_=prev.inference(net,data,'enroll');zc,_=prev.inference(net,data,'calibration');zt,_=prev.inference(net,data,'test');el=data['enroll']['labels'];cl=data['calibration']['labels'];tl=data['test']['labels'];models=old.sel.joblib.load(str(folder/(key+'_rbf_private.joblib')));saved=np.load(str(folder/(key+'_scores_private.npz')));assert np.array_equal(saved['test_labels'],tl)
  for name,c in lock['routes'][key].items():
   u=c['user']-1
   if c['backend']=='cosine':center=prev.prototypes(ze,el,c['K'])[u];cal=zc@center;test=zt@center
   else:
    fitted=models[str(u+1)];sc=fitted[0][0];assert int(sc.n_samples_seen_)==180 and np.allclose(sc.mean_,ze.mean(0),atol=2e-5,rtol=2e-5);cal=old.sel.rbf.score(fitted,[zc]);test=old.sel.rbf.score(fitted,[zt])
   t=prev.threshold(cal[cl==u],cal[cl!=u]);low=prev.lowfar_threshold(cal[cl==u],cal[cl!=u]);assert abs(t-c['threshold'])<2e-6 and abs(low-c['lowfar_threshold'])<2e-6;assert np.allclose(test,saved[name],atol=2e-6,rtol=2e-6);row=rows[name];g=test[tl==u];n=test[tl!=u]
   assert len(g)==row['genuine_n'] and len(n)==row['impostor_n'];assert abs(((g[:,None]>n)+.5*(g[:,None]==n)).mean()-row['auc'])<1e-12
   assert abs((g>=t).mean()-row['tar'])<1e-12 and abs((n>=t).mean()-row['far'])<1e-12 and abs((g>=low).mean()-row['lowfar_calibrated_test_tar'])<1e-12 and abs((n>=low).mean()-row['lowfar_calibrated_test_far'])<1e-12
   assert len(row['attack_by_user'])==8
   for v,b in row['attack_by_user'].items():
    ss=test[tl==int(v)-1];assert len(ss)==b['n'] and int((ss>=t).sum())==b['accepted']
  print('VERIFIED',seed,key,flush=True)
 verified[str(seed)]=r
old.save(run.ROOT/'verified_results_public.json',verified);old.save(run.ROOT/'verification_public.json',dict(status='PASS',routes=540,checks=['train-only global scalar statistics','validation checkpoint audit','student-only score replay','enrollment scaler','calibration thresholds','AUC FAR FRR all8attacks']))
