"""Independent train-OOF refits, calibration search and all nine final-score checks."""
import sys,json
import numpy as np
sys.path.insert(0,'/root')
import run_nine_sameword_balanced_20261030 as run
old=run.old
M=run.same.manifest();all_results={}
for u in range(1,10):
 folder=run.ROOT/('U%d'%u);data=run.dataset(u,M);r=json.loads((folder/'results_public.json').read_text());lock=json.loads((folder/'locked_before_final.json').read_text());models=old.sel.joblib.load(str(folder/'models_private.joblib'));oof=np.load(str(folder/'oof_private.npz'));scores=np.load(str(folder/'scores_private.npz'));y=data['auth_train']['labels']==u-1;groups=data['auth_train']['repeats'];folds=list(old.sel.StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42).split(np.zeros(len(y)),y,groups))
 assert np.array_equal(scores['labels'],data['test']['labels']) and np.array_equal(scores['words'],data['test']['words'])
 for k,m in models.items():
  x=data['auth_train']['features'][k];sc=m['model'][0][0];assert int(sc.n_samples_seen_)==675 and np.allclose(sc.mean_,x.mean(0),atol=2e-5,rtol=2e-5)
  recreated=np.full(len(y),np.nan)
  for i,(tr,va) in enumerate(folds):
   assert not set(groups[tr])&set(groups[va]);assert set(data['auth_train']['words'][tr])==set(data['auth_train']['words'][va])==set(run.same.GROUPS)
   c=m['outer_configs'][i];f=old.sel.rbf.fit([x[tr]],y[tr],c['C'],c['gamma']);ss=old.sel.rbf.score(f,[x[tr]]);recreated[va]=(old.sel.rbf.score(f,[x[va]])-ss.mean())/(ss.std()+1e-6)
  assert np.allclose(recreated,oof[k],atol=1e-7,rtol=1e-6)
 for c in lock['candidates']:
  s=oof['base']+c['raw_weight']*oof['raw']+(c['teacher_weight']*oof[c['teacher']] if c['teacher_weight'] else 0.)
  q=run.fold_quality(s,y,groups,folds)
  for k in ['tar','far','hter','worst','auc']:assert abs(q[k]-c['quality'][k])<1e-12
 assert min(lock['candidates'],key=run.route_rank)==lock['chosen']==r['chosen']
 for key,spec in lock['routes'].items():
  cal=run.score(models,data,spec,'calibration');cp=data['calibration']['labels']==u-1;g=cal[cp];n=cal[~cp];values=np.unique(np.r_[g,n]);tt=np.r_[values,np.nextafter(values[-1],np.inf)]
  # Brute-force independent threshold objective, including reject-all endpoint.
  rows=[((float((g<t).mean())+float((n>=t).mean()))/2,max(float((g<t).mean()),float((n>=t).mean())),float(t)) for t in tt]
  best=min(rows,key=lambda x:(round(x[0],12),round(x[1],12),-x[2]));assert best[2]==spec['threshold']
  final=run.score(models,data,spec,'test');assert np.allclose(final,scores[key],atol=1e-7,rtol=1e-6);pos=data['test']['labels']==u-1;g=final[pos];n=final[~pos];row=r['routes'][key];t=spec['threshold']
  assert len(g)==30 and len(n)==240 and abs(((g[:,None]>n)+.5*(g[:,None]==n)).mean()-row['auc'])<1e-12
  assert abs((g>=t).mean()-row['tar'])<1e-12 and abs((n>=t).mean()-row['far'])<1e-12 and abs((1-row['tar']+row['far'])/2-row['hter'])<1e-12
  assert len(row['attack_by_user'])==8
  for v,a in row['attack_by_user'].items():
   s=final[data['test']['labels']==int(v)-1];assert len(s)==a['n']==30 and int((s>=t).sum())==a['accepted']
  for w in run.same.GROUPS:
   mask=data['test']['words']==w;wp=pos[mask];s=final[mask];assert wp.sum()==2 and (~wp).sum()==16;p=row['per_word'][run.same.WORDS[w]];assert abs(((s[wp,None]>s[~wp])+.5*(s[wp,None]==s[~wp])).mean()-p['auc'])<1e-12
 all_results[str(u)]=r;print('VERIFIED',u,flush=True)
old.save(run.ROOT/'verified_results_public.json',all_results)
old.save(run.ROOT/'verification_public.json',dict(status='PASS',targets=list(range(1,10)),routes=27,checks=['identical word/repeat roles allusers','all auth indices upstream-disjoint','675train-only scalers','225 outermodel refits regenerate OOF','40fusion choices peruser replay','bruteforce joint FAR FRR calibration threshold','saved models final score replay','AUC and perword2positive16negative','8attack sources30each']))

