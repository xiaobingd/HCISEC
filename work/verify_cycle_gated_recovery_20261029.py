"""Independent saved-head/calibration/score count replay (no test fitting)."""
import sys,json
import numpy as np
import torch
sys.path.insert(0,'/root')
import run_cycle_gated_recovery_20261029 as run
old=run.old
for u in [1,4,5]:
 folder=run.ROOT/('U%d'%u);p=old.protocol(u)
 result=json.loads((folder/'results_public.json').read_text());lock=json.loads((folder/'locked_before_final.json').read_text())
 saved=np.load(str(folder/'scores_private.npz'));dev=np.load(str(run.first.ROOT/('U%d'%u)/'development_private.npz'))
 scaler=old.sel.joblib.load(str(folder/'scaler_private.joblib'));assert int(scaler.n_samples_seen_)==450
 assert np.allclose(scaler.mean_,dev['auth_train__raw640'].mean(0),atol=1e-5)
 for key,r in result['routes'].items():
  spec=lock['routes'][key];best=min(spec['history'],key=lambda h:(-h['word_auc'],-h['auc'],h['bce'],h['step']));assert best['step']==r['selected_step']
  reference=run.nn.Sequential(run.nn.Linear(640,128),run.nn.GELU(),run.nn.Linear(128,64)).to(run.D)
  net=run.Recovery(reference,r['mode']).to(run.D);net.load_state_dict(torch.load(str(folder/(key+'_private.pt')),map_location=run.D,weights_only=False));net.eval()
  clf=old.sel.joblib.load(str(folder/(key+'_rbf_private.joblib')))
  def score(role):
   a=[run.tensor(scaler.transform(dev[role+'__'+k])) for k in ['raw640','cycle640']];z,_=run.get(net,a);return old.sel.rbf.score(clf,[z])
  threshold=old.base.original.choose_threshold(score('target_dev_pool'),score('impostor_calibration'));assert abs(threshold-r['threshold'])<1e-12
  g=saved[key+'__final_genuine'];n=saved[key+'__final_impostor'];assert len(g)==r['genuine_n'] and len(n)==r['impostor_n']
  assert abs(((g[:,None]>n)+.5*(g[:,None]==n)).mean()-r['auc'])<1e-12
  assert abs((g>=threshold).mean()-r['tar'])<1e-12 and abs((n>=threshold).mean()-r['far'])<1e-12
  assert len(r['attack_by_user'])==8
  for v,a in r['attack_by_user'].items():
   z=n[saved['attack_labels']==int(v)-1];assert len(z)==a['n'] and int((z>=threshold).sum())==a['accepted']
 print('VERIFIED',u,flush=True)
old.save(run.ROOT/'verification_public.json',dict(status='PASS',targets=[1,4,5],routes=36,checks=['scaler authtrain450 only','validation selected checkpoint replay','saved classifier calibration threshold replay','AUC/TAR/FAR pairwise/count replay','all eight attack sources']))

