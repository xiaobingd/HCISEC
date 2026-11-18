"""Independent utterance isolation, OOF/model replay and generated-Audio provenance."""
import sys,json,importlib.util
import numpy as np
import torch
sys.path.insert(0,'/root')
import run_vibravox_current_baseline_20261030 as run
old=run.old;p,d,speakers=run.configuration();verified={}
for owner in p['owners']:
 folder=run.ROOT/owner;source=run.SOURCE/owner;r=json.loads((folder/'results_public.json').read_text());lock=json.loads((folder/'locked_before_final.json').read_text());models=old.sel.joblib.load(str(folder/'models_private.joblib'));oof=np.load(str(folder/'oof_private.npz'));scores=np.load(str(folder/'scores_private.npz'));cache=dict(np.load(str(source/'frozen_features_private.npz')));features_saved=np.load(str(folder/'teacher_features_private.npz'))
 assert run.sha(run.SOURCE/'inputs.npz')==r['input_sha256'] and run.sha(source/'models/checkpoint_epoch_200.pth')==r['checkpoint_sha256']
 ck=torch.load(str(folder/'teacher_private.pt'),map_location=run.D,weights_only=False);meta=ck['metadata'];best=min(meta['history'],key=lambda h:(-h['mean_real_modality_val_auc'],h['val_bce'],h['step']));assert best['step']==meta['selected_step']==r['teacher_selected_step']
 fit_teacher=set(sum(meta['fit_units'].values(),[]));val_teacher=set(meta['val_units']);auth=set(sum([p['roles'][who][role] for who in p['owners'] for role in ['auth_train','calibration','final']],[]));unknown=set(sum([p['roles'][who]['final'] for who in p['unknown']],[]));assert not fit_teacher&val_teacher and not (fit_teacher|val_teacher)&(auth|unknown)
 assert fit_teacher==set(sum([p['roles'][who]['gan_train'] for who in p['owners']],[]))
 teacher=old.base.Teacher().to(run.D);teacher.load_state_dict(ck['state']);teacher.eval();cache['teacher_acc']=run.teacher_embed(teacher,d['acc'],'acc');cache['teacher_audio']=run.teacher_embed(teacher,cache['generated'],'audio')
 for key in features_saved.files:assert np.allclose(cache[key],features_saved[key],atol=2e-5,rtol=2e-5)
 names=['full772','raw320','input_encoder640','cycle_encoder640','teacher_acc','teacher_audio'];units=np.unique(d['unit_id']);features={k:{int(u):cache[k][d['unit_id']==u].mean(0) for u in units} for k in names};known=r['known_negative'];fit=p['roles'][owner]['auth_train']+p['roles'][known]['auth_train'];y=np.asarray([speakers[u]==owner for u in fit]);folds=list(run.StratifiedKFold(5,shuffle=True,random_state=42).split(np.zeros(len(y)),y))
 for key,m in models.items():
  x=np.stack([features[key][u] for u in fit]);sc=m['model'][0][0];assert int(sc.n_samples_seen_)==20 and np.allclose(sc.mean_,x.mean(0),atol=2e-5,rtol=2e-5);pred=np.full(len(y),np.nan)
  for i,(tr,va) in enumerate(folds):
   assert not set(np.asarray(fit)[tr])&set(np.asarray(fit)[va]);c=m['outer_configs'][i];f=old.sel.rbf.fit([x[tr]],y[tr],c['C'],c['gamma']);s=old.sel.rbf.score(f,[x[tr]]);pred[va]=(old.sel.rbf.score(f,[x[va]])-s.mean())/(s.std()+1e-6)
  assert np.allclose(pred,oof[key],atol=1e-7,rtol=1e-6)
 for mode,rows in lock['searches'].items():
  for c in rows:
   s=oof['full772']+c['raw']*oof['raw320']+c['acc']*oof['teacher_acc']+c['audio']*oof['teacher_audio'];q=run.fold_quality(s,y,folds)
   for k in ['tar','far','auc','hter','worst']:assert abs(q[k]-c['quality'][k])<1e-12
  assert min(rows,key=run.rank)==lock['selected'][mode]
 a=lock['routes']['acc']['weights'];b=lock['routes']['fixed_acc_audio']['weights'];assert all(a[k]==b[k] for k in ['full772','raw320','teacher_acc'])
 for mode,c in lock['routes'].items():
  g=run.score(models,features,c,p['roles'][owner]['calibration']);n=run.score(models,features,c,p['roles'][known]['calibration']);tt=np.unique(np.r_[g,n]);tt=np.r_[tt,np.nextafter(tt[-1],np.inf)];t=min(tt,key=lambda v:(round(float(((g<v).mean()+(n>=v).mean())/2),12),round(float(max((g<v).mean(),(n>=v).mean())),12),-float(v)));assert float(t)==c['threshold']
  g=run.score(models,features,c,p['roles'][owner]['final']);assert len(g)==8 and np.allclose(g,scores[mode+'__genuine'],atol=1e-7,rtol=1e-6);neg={who:run.score(models,features,c,p['roles'][who]['final']) for who in [known]+p['unknown']}
  for who,s in neg.items():assert np.allclose(s,scores[mode+'__'+who],atol=1e-7,rtol=1e-6) and len(s)==r['routes'][mode]['per_attacker'][who]['n'] and int((s>=t).sum())==r['routes'][mode]['per_attacker'][who]['accepted']
  for scope,n in [('known',neg[known]),('unseen',np.concatenate([neg[who] for who in p['unknown']])),('all_attackers',np.concatenate(list(neg.values())))]:
   rr=r['routes'][mode][scope];assert len(g)==rr['genuine_n'] and len(n)==rr['impostor_n'];assert abs(((g[:,None]>n)+.5*(g[:,None]==n)).mean()-rr['auc'])<1e-12 and abs((g>=t).mean()-rr['tar'])<1e-12 and abs((n>=t).mean()-rr['far'])<1e-12
 # Recreate full772 definition and generated spectrum from original checkpoint.
 spec=importlib.util.spec_from_file_location('vibravox_original_model',str(run.SOURCE.parent/'source/model_drop.py'));mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);ckg=torch.load(str(source/'models/checkpoint_epoch_200.pth'),map_location=run.D,weights_only=False);gnet=mod.GeneratorAccToAudio(input_dim=1,hidden_dim=64,output_dim=1).to(run.D);fnet=mod.GeneratorAudioToAcc(input_dim=1,hidden_dim=64,output_dim=1).to(run.D);gnet.load_state_dict(ckg['G_acc2audio_state_dict']);fnet.load_state_dict(ckg['G_audio2acc_state_dict']);gnet.eval();fnet.eval()
 unit_choices=[p['roles'][owner]['gan_train'][0],p['roles'][owner]['auth_train'][0],p['roles'][owner]['final'][0],p['roles'][p['unknown'][0]]['final'][0]];ix=np.asarray([np.flatnonzero(d['unit_id']==u)[0] for u in unit_choices])
 with torch.no_grad():
  x=old.te.tensor(d['acc'][ix]);generated=gnet(x);cycle=fnet(generated);diff=x-cycle;h=gnet.model[:14](x);flat=diff.flatten(1);full=torch.cat([gnet.model[:7](x).mean((2,3)),h.mean((2,3)),h.flatten(2).std(2),gnet.model[:7](diff).mean((2,3)),torch.stack([flat.mean(1),flat.std(1),flat.abs().max(1).values,flat.square().mean(1).sqrt()],1)],1).cpu().numpy();audio_teacher=teacher.audio(generated).cpu().numpy()
 assert np.allclose(full,cache['full772'][ix],atol=2e-4,rtol=2e-4) and np.allclose(generated[:,0].cpu().numpy(),cache['generated'][ix],atol=2e-4,rtol=2e-4) and np.allclose(audio_teacher,cache['teacher_audio'][ix],atol=2e-4,rtol=2e-4)
 del gnet,fnet,teacher,ckg;torch.cuda.empty_cache();verified[owner]=r;print('VERIFIED',owner,flush=True)
old.save(run.ROOT/'verified_results_public.json',verified);old.save(run.ROOT/'verification_public.json',dict(status='PASS',owners=p['owners'],routes=16,checks=['wholeutterance role and exact text isolation','normalization known GANtrain only','teacher upstream units no auth/unseen','teacher checkpoint selection replay','60outer RBF refits OOF replay','fusion candidates and fixedACC anchor replay','joint calibration threshold brute force','known/unseen separate pairwiseAUC acceptancecounts','canonical full772 regenerated from original G/F','generatedAudio teacher provenance from ACC only']))

