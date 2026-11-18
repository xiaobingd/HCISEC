"""Verify generated-only Audio provenance, OOF refits, fusion decisions and final counts."""
import sys,json
import numpy as np
import torch
sys.path.insert(0,'/root')
import run_nine_audio_supplement_20261030 as run
base=run.base;old=run.old

def independent_threshold(g,n):
 g=np.sort(g);n=np.sort(n);v=np.unique(np.r_[g,n]);v=np.r_[v,np.nextafter(v[-1],np.inf)];frr=np.searchsorted(g,v,side='left')/len(g);far=1-np.searchsorted(n,v,side='left')/len(n)
 i=np.lexsort((-v,np.round(np.maximum(far,frr),12),np.round((far+frr)/2,12)))[0];return float(v[i])

base.balanced_threshold=independent_threshold
M=base.same.manifest();verified={}
for u in range(1,10):
 folder=run.ROOT/('U%d'%u);data=run.dataset(u,M);r=json.loads((folder/'results_public.json').read_text());lock=json.loads((folder/'locked_before_final.json').read_text());models=old.sel.joblib.load(str(folder/'models_private.joblib'));oof=np.load(str(folder/'oof_private.npz'));saved=np.load(str(folder/'scores_private.npz'));y=data['auth_train']['labels']==u-1;groups=data['auth_train']['repeats'];folds=list(old.sel.StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42).split(np.zeros(len(y)),y,groups))
 assert np.array_equal(saved['labels'],data['test']['labels']) and np.array_equal(saved['words'],data['test']['words'])
 for key in run.AUDIO:
  m=models[key];x=data['auth_train']['features'][key];sc=m['model'][0][0];assert int(sc.n_samples_seen_)==675 and np.allclose(sc.mean_,x.mean(0),atol=2e-5,rtol=2e-5);pred=np.full(len(y),np.nan)
  for i,(tr,va) in enumerate(folds):
   c=m['outer_configs'][i];f=old.sel.rbf.fit([x[tr]],y[tr],c['C'],c['gamma']);s=old.sel.rbf.score(f,[x[tr]]);pred[va]=(old.sel.rbf.score(f,[x[va]])-s.mean())/(s.std()+1e-6)
  assert np.allclose(pred,oof[key],atol=1e-7,rtol=1e-6)
 for mode,rows in lock['searches'].items():
  for c in rows:
   s=oof['base']+c['raw_weight']*oof['raw']
   if c['acc_weight']:s+=c['acc_weight']*oof[c['acc']]
   if c['audio_weight']:s+=c['audio_weight']*oof[c['audio']]
   q=base.fold_quality(s,y,groups,folds)
   for k in ['tar','far','hter','worst','auc']:assert abs(q[k]-c['quality'][k])<1e-12
  assert min(rows,key=run.rank)==lock['selected'][mode]
 a=lock['routes']['acc'];f=lock['routes']['fixed_acc_audio'];assert all(a[k]==f[k] for k in ['acc','raw_weight','acc_weight'])
 for mode,c in lock['routes'].items():
  s=run.score(models,data,c,'calibration');pos=data['calibration']['labels']==u-1;t=independent_threshold(s[pos],s[~pos]);assert t==c['threshold']
  s=run.score(models,data,c,'test');assert np.allclose(s,saved[mode],atol=1e-7,rtol=1e-6);pos=data['test']['labels']==u-1;g=s[pos];n=s[~pos];row=r['routes'][mode]
  assert len(g)==30 and len(n)==240 and abs(((g[:,None]>n)+.5*(g[:,None]==n)).mean()-row['auc'])<1e-12
  assert abs((g>=t).mean()-row['tar'])<1e-12 and abs((n>=t).mean()-row['far'])<1e-12 and len(row['attack_by_user'])==8
  for v,b in row['attack_by_user'].items():
   z=s[data['test']['labels']==int(v)-1];assert len(z)==b['n']==30 and int((z>=t).sum())==b['accepted']
  for w in base.same.GROUPS:
   mask=data['test']['words']==w;wp=pos[mask];ws=s[mask];assert wp.sum()==2 and (~wp).sum()==16;pr=row['per_word'][base.same.WORDS[w]];assert abs(((ws[wp,None]>ws[~wp])+.5*(ws[wp,None]==ws[~wp])).mean()-pr['auc'])<1e-12
 # Recompute selected cache rows directly from raw ACC -> frozen G -> Audio teacher.
 gan=old.load_gan(u)
 for family,name in [('old','teacher'),('identity','identity_preserving_teacher'),('robust','domain_robust_teacher')]:
  ck=torch.load(str(old.ROOT/('U%d'%u)/('U%d_%s.pt'%(u,name))),map_location=old.D,weights_only=False);teacher=old.base.Teacher().to(old.D);teacher.load_state_dict(ck['state']);teacher.eval()
  for role in ['calibration','test']:
   records=[(v,j) for v in old.USERS for j in M['sources'][str(v)]['roles'][role]];ix=np.unique([0,(u-1)*len(M['sources'][str(u)]['roles'][role]),len(records)-1]);raw=[]
   for i in ix:
    v,j=records[i];raw.append(np.load(str(old.DATA/('acc.%d.npy'%v)))[j,:80,:80])
   with torch.no_grad():z=teacher.audio(gan.g_acc2audio(old.te.tensor(np.stack(raw).astype('float32')))).cpu().numpy()
   assert np.allclose(z,data[role]['features'][family+'_audio'][ix],atol=2e-4,rtol=2e-4)
 del gan,teacher;torch.cuda.empty_cache();verified[str(u)]=r;print('VERIFIED',u,flush=True)
old.save(run.ROOT/'verified_results_public.json',verified);old.save(run.ROOT/'verification_public.json',dict(status='PASS',targets=list(range(1,10)),routes=45,checks=['135 Audio outer-model refits reproduce OOF','490candidates pertarget replay','ACC-only previous control exact replay','fixedACC anchor unchanged','independent threshold search','all final scores counts and perword2/16','all8sources30each','rawACC->G->AudioTeacher cache provenance on cal/test;no realAudio']))

