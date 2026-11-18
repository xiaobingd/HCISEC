"""ACC-spectrum augmentation before frozen G/F; paired fixed-budget adapter training."""
import sys,json,os,copy,hashlib
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.preprocessing import StandardScaler
sys.path.insert(0,'/root')
import run_crosscontent_adapter_20261029 as trial
old=trial.old;ROOT=Path('/autodl-fs/data/mei_adapter_augmentation_20261029')
AUGS=['noise','time_shift','noise_shift'];MODES=['combined','episode_only'];VIEWS=2
def save(p,x):old.save(p,x)
def shifted(x,shifts):
 out=np.empty_like(x)
 for i,k in enumerate(shifts):
  if k>0:out[i,:,k:]=x[i,:,:-k];out[i,:,:k]=x[i,:,:1]
  else:out[i,:,:k]=x[i,:,-k:];out[i,:,k:]=x[i,:,-1:]
 return out
def augment(raw,view,kind,sigma,lo,hi):
 rng=np.random.RandomState(51000+view);noise=rng.normal(0,sigma,raw.shape).astype('float32')
 shifts=np.random.RandomState(52000+view).choice([-2,-1,1,2],len(raw))
 x=raw.copy()
 if kind in ['time_shift','noise_shift']:x=shifted(x,shifts)
 if kind in ['noise','noise_shift']:x=np.clip(x+noise,lo,hi)
 return x
@torch.no_grad()
def augmented_features(u,p,x,folder):
 raw=p.arrays['auth_train'];sigma=float(.02*np.median(raw.std(axis=(1,2))));lo=float(raw.min());hi=float(raw.max())
 assert len(raw)==len(x)==450
 gan=old.load_gan(u)
 fresh=old.base.fusion_inputs(gan,old.te.tensor(raw[:16])).cpu().numpy()
 assert np.allclose(fresh,x[:16],atol=2e-4,rtol=2e-4),'Frozen checkpoint/cache mismatch'
 features={};meta=dict(noise_sigma=sigma,train_min=lo,train_max=hi,source_recordings=450,view_count=2,source_ids_sha256=hashlib.sha256(json.dumps(p.sample_ids['auth_train']).encode()).hexdigest(),gan_checkpoint_sha256=old.sha(old.UP/('U%d'%u)/'cyclegan/models/checkpoint_epoch_200.pth'))
 for kind in AUGS:
  views=[]
  for view in range(VIEWS):
   perturbed=augment(raw,view,kind,sigma,lo,hi);parts=[]
   assert np.isfinite(perturbed).all()
   for start in range(0,len(raw),8):parts.append(old.base.fusion_inputs(gan,old.te.tensor(perturbed[start:start+8])).cpu().numpy())
   z=np.concatenate(parts);assert z.shape==(450,772) and np.isfinite(z).all();views.append(z)
   print('AUG_FEATURES_DONE',u,kind,view,flush=True)
  features[kind]=np.stack(views)
 np.savez_compressed(str(folder/'augmented_train_features_private.npz'),**features);save(folder/'augmentation_public.json',meta)
 del gan
 if torch.cuda.is_available():torch.cuda.empty_cache()
 return features
def target(u):
 folder=ROOT/('U%d'%u);folder.mkdir(exist_ok=True);p=old.protocol(u);src=trial.ROOT/('U%d'%u)
 dev=np.load(str(old.ROOT/('U%d'%u)/'development_features_private.npz'));x=dev['auth_train__base'];val=dev['model_validation__base'];labels=p.labels['auth_train']
 def wordids(role):return np.asarray([old.M['sources'][s.split(':')[0]]['content_groups'][int(s.split(':')[1])] for s in p.sample_ids[role]])
 words=wordids('auth_train');valwords=wordids('model_validation');scaler=StandardScaler().fit(x);old.sel.joblib.dump(scaler,str(folder/'scaler_private.joblib'))
 original=old.sel.joblib.load(str(src/'scaler_private.joblib'));assert np.array_equal(scaler.mean_,original.mean_) and np.array_equal(scaler.scale_,original.scale_)
 blocked=set(sum([p.sample_ids[k] for k in ['model_validation','target_dev_pool','impostor_calibration','final_genuine','final_impostor']],[]))
 assert len(set(p.sample_ids['auth_train']))==450 and not set(p.sample_ids['auth_train'])&blocked
 features=augmented_features(u,p,x,folder);xx=trial.tensor(scaler.transform(x));vv=trial.tensor(scaler.transform(val));tl=torch.as_tensor(labels,device=trial.DEVICE);tw=torch.as_tensor(words,device=trial.DEVICE)
 models={};locks={};history={}
 for mode in MODES:
  for kind in AUGS:
   bank=torch.stack([trial.tensor(scaler.transform(view)) for view in features[kind]])
   for ss in trial.SEEDS:
    trial.seed(ss);net=trial.Adapter().to(trial.DEVICE);opt=torch.optim.AdamW(net.parameters(),lr=.0003,weight_decay=.001)
    erng=np.random.RandomState(ss);arng=np.random.RandomState(61000+ss);best=None;hist=[]
    for step in range(1,trial.STEPS+1):
     # Independent augmentation RNG preserves paired episode draws across conditions.
     choice=arng.randint(0,4,len(x));view=np.maximum(choice-2,0)
     aug=bank[torch.as_tensor(view,device=trial.DEVICE),torch.arange(len(x),device=trial.DEVICE)]
     train_x=torch.where(torch.as_tensor(choice>=2,device=trial.DEVICE)[:,None],aug,xx)
     net.train();z=net(train_x);loss=trial.episode(z,labels,words,erng)
     if mode=='combined':loss=loss+trial.contrast(z,tl,tw,True)
     assert torch.isfinite(loss);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
     if step%25==0:
      net.eval()
      with torch.no_grad():
       centers=trial.centers(net(xx),labels);zv=net(vv);s=(zv@centers[u-1]).cpu().numpy();q=trial.quality(s,p.labels['model_validation']==u-1,valwords)
       ce=float(F.cross_entropy(zv@centers.T/.1,torch.as_tensor(p.labels['model_validation'],device=trial.DEVICE)))
      hist.append(dict(step=step,training_loss=float(loss.detach()),validation_sameword_auc=q[0],validation_auc=q[1],validation_multiclass_ce=ce))
      rank=(-q[0],-q[1],ce,step)
      if best is None or rank<best[0]:best=(rank,copy.deepcopy(net.state_dict()),step,q)
    net.load_state_dict(best[1]);net.eval()
    for par in net.parameters():par.requires_grad_(False)
    key='%s__%s_seed%d'%(mode,kind,ss);models[key]=net;history[key]=hist;locks[key]=dict(mode=mode,augmentation=kind,seed=ss,selected_step=best[2],validation_sameword_auc=best[3][0],validation_auc=best[3][1])
    torch.save(dict(state=best[1],metadata=locks[key]),str(folder/(key+'_private.pt')));print('MODEL_DONE',u,key,best[2],flush=True)
 def score(key,raw):
  with torch.no_grad():centers=trial.centers(models[key](xx),labels);return (models[key](trial.tensor(scaler.transform(raw)))@centers[u-1]).cpu().numpy()
 for key in locks:
  g=score(key,dev['target_dev_pool__base']);n=score(key,dev['impostor_calibration__base']);locks[key]['threshold']=float(old.base.original.choose_threshold(g,n))
 save(folder/'locked_before_final.json',dict(target=u,routes=locks,history=history,final_scores_seen=False))
 final=np.load(str(trial.FINAL/('U%d'%u)/'final_features_private.npz'));rows={};dump={}
 for key,spec in locks.items():
  ss={role:score(key,final[role+'__base']) for role in ['final_genuine','final_impostor']};r=old.sel.evaluate_route(ss,p,u,spec['threshold']);r.update(spec)
  wg=wordids('final_genuine');wn=wordids('final_impostor');r['per_word']={str(w):old.sel.exp.metrics(ss['final_genuine'][wg==w],ss['final_impostor'][wn==w],spec['threshold']) for w in sorted(np.unique(wg))}
  r['same_word_mean_auc']=float(np.mean([v['auc'] for v in r['per_word'].values()]));rows[key]=r
  for role,s in ss.items():dump[key+'__'+role]=s
 reference=json.loads((src/'results_public.json').read_text())
 for mode in MODES:
  for ss in trial.SEEDS:
   r=reference['routes'][mode+'_seed%d'%ss];rows['%s__none_seed%d'%(mode,ss)]=dict(r,augmentation='none',reused_control=True)
 np.savez_compressed(str(folder/'scores_private.npz'),**dump,attack_labels=p.labels['final_impostor']);save(folder/'results_public.json',dict(target=u,routes=rows));print('TARGET_COMPLETE',u,flush=True)
def main():
 ROOT.mkdir(exist_ok=True)
 with (ROOT/'QUEUE_LOCK').open('x') as f:f.write(str(os.getpid()))
 old.records();save(ROOT/'protocol_public.json',dict(targets=[1,4,5],modes=MODES,augmentations=['none']+AUGS,seeds=trial.SEEDS,views=VIEWS,noise='Spectrum-domain Gaussian sigma=.02*median per-sample training-ACC std;global training min/max clamp;NOT physical waveform-noise simulation',shift='+-1/2 time frames,edge-filled,non-circular',view_sampling='Each step each recording50percent clean,25percent each fixed augmented view;batch450,400steps,episode RNG paired with original controls',unchanged=['G/F','teacher','train50/val30/cal20/test content splits','all eight attackers','adapter772-128-64','optimizer','clean-only scaler fit','clean enrollment centers','clean validation/calibration/final'],limitations=['Two fixed views,not more independent recordings','Historical final set reused,exploratory','No time-domain physical noise model','Known closed-set identities','Content10/20period and missing-tail assumptions'],source_sha256=old.sha(Path(__file__))))
 results={}
 for u in [1,4,5]:
  save(ROOT/'status_public.json',dict(state='running',target=u,completed=list(results)));target(u)
  results[str(u)]=json.loads((ROOT/('U%d'%u)/'results_public.json').read_text());save(ROOT/'results_public.json',results)
 save(ROOT/'status_public.json',dict(state='complete',completed=list(results)))
if __name__=='__main__':main()

