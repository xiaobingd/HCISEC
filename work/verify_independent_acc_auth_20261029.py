"""Replay ACC-only checkpoints, preprocessing roles, calibration and final inference."""
import sys,json
import numpy as np
import torch
sys.path.insert(0,'/root')
import run_independent_acc_auth_20261029 as run
old=run.old
for u in [1,4,5]:
 folder=run.ROOT/('U%d'%u);p=old.protocol(u);r=json.loads((folder/'results_public.json').read_text());lock=json.loads((folder/'locked_before_final.json').read_text());saved=np.load(str(folder/'scores_private.npz'))
 assert r['no_generator_or_audio'] and np.array_equal(saved['attack_labels'],p.labels['final_impostor'])
 wordsets=[set(run.first.words(p,role)) for role in ['auth_train','model_validation','target_dev_pool','final_genuine']]
 for i,a in enumerate(wordsets):
  for b in wordsets[:i]:assert not a&b
 for key,result in r['routes'].items():
  spec=lock['routes'][key];model=old.sel.joblib.load(str(folder/(key+'_rbf_private.joblib')))
  if key.startswith('cnn'):
   best=min(spec['history'],key=lambda h:(-h['word_auc'],-h['auc'],h['bce'],h['step']));assert best['step']==spec['selected_step']
   net=run.Encoder().to(run.D);net.load_state_dict(torch.load(str(folder/(key+'_private.pt')),map_location=run.D,weights_only=False));net.eval();fn=lambda x:run.embedding(net,x)
  elif key=='direct_stats320':fn=run.statistics
  else:fn=lambda x:x.reshape(len(x),-1)
  def score(role):return old.sel.rbf.score(model,[fn(p.arrays[role])])
  t=old.base.original.choose_threshold(score('target_dev_pool'),score('impostor_calibration'));assert abs(t-result['threshold'])<1e-12
  for role in ['final_genuine','final_impostor']:assert np.allclose(score(role),saved[key+'__'+role],atol=1e-7,rtol=1e-6)
  g=saved[key+'__final_genuine'];n=saved[key+'__final_impostor'];assert len(g)==result['genuine_n'] and len(n)==result['impostor_n']
  assert abs(((g[:,None]>n)+.5*(g[:,None]==n)).mean()-result['auc'])<1e-12
  assert abs((g>=t).mean()-result['tar'])<1e-12 and abs((n>=t).mean()-result['far'])<1e-12
  assert len(result['attack_by_user'])==8
  for v,a in result['attack_by_user'].items():
   s=n[saved['attack_labels']==int(v)-1];assert len(s)==a['n'] and int((s>=t).sum())==a['accepted']
 print('VERIFIED',u,flush=True)
old.save(run.ROOT/'verification_public.json',dict(status='PASS',targets=[1,4,5],routes=15,checks=['crossword split disjointness','CNN checkpoint selection replay','saved model independent raw ACC inference','calibration thresholds replay','pairwise AUC and acceptance counts','all8attackers']))

