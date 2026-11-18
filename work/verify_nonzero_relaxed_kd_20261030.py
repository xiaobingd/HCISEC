"""Audit MKD controls, identity-preserving derangement, frozen targets and all auth scores."""
import sys,json,time
import numpy as np
import torch
sys.path.insert(0,'/root')
import run_relaxed_audio_feature_kd_20261030 as run
prev=run.prev;old=run.old;data=prev.data_load();verified={}

def load_choice(folder,source,choice):
 path=(folder if choice['source']=='current' else source)/(choice['key']+'_private.pt');ck=torch.load(str(path),map_location=run.D,weights_only=False);net=prev.Classifier().to(run.D);net.aam=ck['metadata'].get('mode','ce')=='aam';net.load_state_dict(ck['state']);net.eval()
 for p in net.parameters():p.requires_grad_(False)
 return net,ck

for s in prev.SEEDS:
 folder=run.ROOT/('seed%d'%s);source=prev.ROOT/('seed%d'%s)
 r=json.loads((folder/'nonzero_results_public.json').read_text());lock=json.loads((folder/'nonzero_locked_before_final.json').read_text());main=json.loads((folder/'locked_before_final.json').read_text())
 for family,metas in main['selection'].items():
  selected=min(metas,key=lambda m:prev.rank(m['quality'])+(m['lambda_mkd'],m['margin']));assert selected['key']==r['chosen'][family]['key'] and selected['lambda_mkd']>0
 for key,info in r['sources'].items():
  net,ck=load_choice(folder,source,dict(source='current',key=info['key']));meta=ck['metadata'];best=min(meta['history'],key=lambda h:prev.rank(h['quality'])+(h['step'],));assert best['step']==meta['selected_step'] and meta['steps']==400
  if key=='E2_nonzero_Shuffled':
   pair=r['chosen']['E2_AudioMKD'];assert meta['shuffle'] and meta['lambda_mkd']==pair['lambda_mkd'] and meta['margin']==pair['margin']
 mapping=np.load(str(folder/'shuffle_map_private.npy'));assert np.array_equal(mapping,run.shuffle_map(data,s))
 for key,rows in r['routes'].items():
  net,ck=load_choice(folder,source,dict(source='current',key=r['sources'][key]['key']));meta=ck['metadata'];q=prev.validation_quality(net,data,'acc')
  for k in ['hter','far','tar','auc','ce']:assert abs(q[k]-meta['quality'][k])<2e-6
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
  print('VERIFIED_ROUTE',s,key,r['sources'][key]['key'],flush=True)
 verified[str(s)]=r
old.save(run.ROOT/'nonzero_verified_results_public.json',verified);old.save(run.ROOT/'nonzero_verification_public.json',dict(status='PASS',seeds=prev.SEEDS,routes=405,checks=['validation strictlynonzero selection','checkpoint selection and fixed shuffle parameters','student-only independent model replay','enrollment scalers calibration scores AUC all8sources']))
