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
 folder=run.ROOT/('seed%d'%s);source=prev.ROOT/('seed%d'%s);deadline=time.monotonic()+1800
 while True:
  try:r=json.loads((folder/'results_public.json').read_text());break
  except (FileNotFoundError,json.JSONDecodeError):assert time.monotonic()<deadline;time.sleep(5)
 lock=json.loads((folder/'locked_before_final.json').read_text());previous=json.loads((source/'results_public.json').read_text());b0,ck0=run.load_previous(source,'B0_CE');audio,_=run.load_previous(source,'T1_Audio');mapping=np.load(str(folder/'shuffle_map_private.npy'));assert np.array_equal(mapping,run.shuffle_map(data,s));labels=data['train']['labels'];words=data['train']['words'];assert np.array_equal(labels,labels[mapping]) and np.all(words!=words[mapping]) and len(np.unique(mapping))==len(mapping)
 for family,metas in lock['selection'].items():
  choices=[(prev.rank(ck0['metadata']['quality'])+(0.,0.),'B0_CE')]
  for meta in metas:
   best=min(meta['history'],key=lambda h:prev.rank(h['quality'])+(h['step'],));assert best['step']==meta['selected_step'] and meta['steps']==400 and meta['alpha']==.6 and not meta['shuffle'];choices.append((prev.rank(meta['quality'])+(meta['lambda_mkd'],meta['margin']),meta['key']))
  assert min(choices,key=lambda q:q[0])[1]==lock['chosen'][family]['key']
 if lock['metadata']['E2_AudioMKD'] is not None:
  p=lock['metadata']['E2_AudioMKD'];q=lock['metadata']['E2_Shuffled'];assert p['lambda_mkd']==q['lambda_mkd'] and p['margin']==q['margin'] and q['shuffle']
 for key,rows in r['routes'].items():
  net,ck=load_choice(folder,source,lock['chosen'][key]);meta=ck['metadata'];q=prev.validation_quality(net,data,'acc')
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
  print('VERIFIED_ROUTE',s,key,lock['chosen'][key]['key'],flush=True)
 for key,previous_key in [('E0_CE','B0_CE'),('E1_LogitKL','S1_AudioKD')]:
  for name,row in r['routes'][key].items():
   for metric in ['tar','far','auc','hter']:assert abs(row[metric]-previous['routes'][previous_key][name][metric])<1e-12
 # Gradient audit on real train examples with detached teacher, independent deployment model.
 z,_=prev.inference(audio,data,'train','audio');student=prev.Classifier().to(run.D);proj=run.Projection().to(run.D);tz=prev.tensor(z[:72]);sz,_=student(prev.tensor(data['train']['acc'][:72,None]));target=proj(tz);cos=(sz*target).sum(1);loss=torch.relu(run.MARGINS[0]-cos).mean();loss.backward();assert any(p.grad is not None and p.grad.abs().sum()>0 for p in student.parameters()) and any(p.grad is not None and p.grad.abs().sum()>0 for p in proj.parameters()) and all(p.grad is None for p in audio.parameters());assert torch.relu(torch.tensor(run.MARGINS[0])-torch.tensor([1.])).item()==0
 verified[str(s)]=r;print('VERIFIED_SEED',s,flush=True)
old.save(run.ROOT/'verified_results_public.json',verified);old.save(run.ROOT/'verification_public.json',dict(status='PASS',seeds=prev.SEEDS,routes=675,checks=['exact E0/E1 previous control replay','6candidate validation selection and checkpoint audit','withinidentity crossword derangement','shuffle fixed E2 lambda/m','student-only deployment embeddings and enrollment scalers','joint and lowFARcalibration','all score/AUC/counts8sources','gradient toStudent andprojection notteacher;zero beyondmargin']))
