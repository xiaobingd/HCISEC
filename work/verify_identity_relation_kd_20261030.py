"""Replay new student inference, locked thresholds and all attacker counts."""
import sys,json
import numpy as np
import torch
sys.path.insert(0,'/root')
import run_identity_relation_kd_20261030 as r
p=r.p;data=p.data_load();verified={}
roles=list(p.ROLES)
for i,a in enumerate(roles):
 for b in roles[:i]:assert not set(data[a]['ids'])&set(data[b]['ids']) and not set(data[a]['words'])&set(data[b]['words'])
for s in p.SEEDS:
 folder=r.ROOT/('seed%d'%s);lock=json.loads((folder/'locked_before_final.json').read_text());out=json.loads((folder/'results_public.json').read_text());assert lock['final_seen'] is False
 teachers=list(lock['teachers'].values());assert len(set(m['parameters'] for m in teachers))==1 and len(set(m['trainable_parameters'] for m in teachers))==1
 for meta in list(lock['metadata'].values())+teachers:
  best=min(meta['history'],key=lambda h:p.rank(h['quality'])+(h['step'],));assert best['step']==meta['selected_step'] and meta['steps']==400
 mapping=np.load(folder/'shuffle_map_private.npy');assert len(set(mapping.tolist()))==len(mapping) and np.all(data['train']['labels']==data['train']['labels'][mapping]) and np.all(data['train']['words']!=data['train']['words'][mapping])
 source=p.ROOT/('seed%d'%s);counts=0
 for key,rows in out['routes'].items():
  path=source/('B0_CE_private.pt' if key=='E0_ACC_CE' else 'B1_AAM_private.pt') if key.startswith('E0') else folder/(key+'_private.pt')
  ck=torch.load(path,map_location=p.D,weights_only=False);net=p.Classifier().to(p.D);net.aam=key=='E0_ACC_AAM';net.load_state_dict(ck['state']);net.eval()
  z={role:p.inference(net,data,role)[0] for role in ['enroll','calibration','test']};el=data['enroll']['labels'];cl=data['calibration']['labels'];tl=data['test']['labels'];backends=p.old.sel.joblib.load(str(folder/(key+'_rbf_private.joblib')));saved=np.load(folder/(key+'_scores_private.npz'))
  assert sum(q.numel() for q in net.parameters())==sum(q.numel() for q in p.Classifier().parameters())
  if not key.startswith('E0'):
   q=p.validation_quality(net,data,'acc')
   for metric in ['tar','far','hter','auc']:assert abs(q[metric]-ck['metadata']['quality'][metric])<2e-6
  for name,c in lock['routes'][key].items():
   u=c['user']-1
   if c['backend']=='cosine':
    center=p.prototypes(z['enroll'],el,c['K'])[u];cal=z['calibration']@center;score=z['test']@center
   else:
    model=backends[str(u+1)];sc=model[0][0];assert int(sc.n_samples_seen_)==180 and np.allclose(sc.mean_,z['enroll'].mean(0),atol=2e-5,rtol=2e-5)
    cal=p.old.sel.rbf.score(model,[z['calibration']]);score=p.old.sel.rbf.score(model,[z['test']])
   threshold=p.threshold(cal[cl==u],cal[cl!=u]);assert abs(threshold-c['threshold'])<2e-6 and np.allclose(score,saved[name],atol=2e-6,rtol=2e-6)
   row=rows[name];g=score[tl==u];n=score[tl!=u];assert len(g)==row['genuine_n'] and len(n)==row['impostor_n']
   assert abs((g>=threshold).mean()-row['tar'])<1e-12 and abs((n>=threshold).mean()-row['far'])<1e-12
   assert abs(((g[:,None]>n)+.5*(g[:,None]==n)).mean()-row['auc'])<1e-12
   assert len(row['attack_by_user'])==8
   for v,aa in row['attack_by_user'].items():assert int((score[tl==int(v)-1]>=threshold).sum())==aa['accepted']
   counts+=1
  print('VERIFIED',s,key,flush=True)
 # Teacher targets cannot receive gradients; relational loss reaches student features.
 zz=torch.randn(72,128,device=p.D,requires_grad=True);tt=torch.randn(72,128,device=p.D);yy=torch.arange(9,device=p.D).repeat_interleave(8);ww=torch.arange(4,device=p.D).repeat_interleave(2).repeat(9)
 loss=r.relation(torch.nn.functional.normalize(zz,dim=1),torch.nn.functional.normalize(tt,dim=1),yy,ww);loss.backward();assert zz.grad.abs().sum()>0 and not tt.requires_grad
 verified[str(s)]=dict(routes=counts,baseline_replayed=out['baseline_replayed'],teacher_parameters=teachers[0]['parameters'],teacher_trainable_parameters=teachers[0]['trainable_parameters'])
p.old.save(r.ROOT/'verification_public.json',dict(status='PASS',seeds=verified,checks=['record and word disjoint roles','400updates and validation-only checkpoint selection','equal teacher architecture parameter counts','unchanged student parameter count','train-only sameID differentword shuffle bijection','independent enrollment RBF scaler','ACC-only inference replay','calibration-only threshold replay','all final AUC/TAR/FAR/eightattack sources','nonzero relation gradient to student']))

