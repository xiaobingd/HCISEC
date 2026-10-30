"""Two-stage upstream-only identity gradients to G encoder; auth records never update G."""
import sys,json,os,copy,hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.preprocessing import StandardScaler
sys.path.insert(0,'/root')
import run_crosscontent_adapter_20261029 as trial
old=trial.old;ROOT=Path('/autodl-fs/data/mei_encoder_identity_feedback_20261029')
VARIANTS=['frozen','conversion_only','identity_conversion'];WARM=50;UPSTEPS=100;AUTHSTEPS=400;SEED=42
def save(p,x):old.save(p,x)
def tensor(x):return trial.tensor(x)
def spectrogram(x):return old.te.tensor(x)
def differentiable_features(m,x):
 cur=x;shallow=None;deep=None
 for i,layer in enumerate(m.g_acc2audio.model):
  cur=layer(cur)
  if i==6:shallow=cur
  if i==13:deep=cur
 fake=cur;recovered=m.g_audio2acc(fake);diff=x-recovered;flat=diff.flatten(1)
 stats=torch.stack([flat.mean(1),flat.std(1),flat.abs().max(1).values,flat.square().mean(1).sqrt()],1)
 f=torch.cat([m.shallow_agg(shallow),m.bottleneck_agg(deep),m.diff_agg(m.diff_encoder(diff)),stats],1)
 assert f.shape[1]==772
 return f,fake,recovered
def frozen_hash(m,allowed):
 h=hashlib.sha256()
 for prefix,net in [('G',m.g_acc2audio),('F',m.g_audio2acc)]:
  for name,t in net.state_dict().items():
   if prefix=='G' and name in allowed:continue
   h.update((prefix+name).encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def enc_setup(m,enabled):
 for p in m.parameters():p.requires_grad_(False)
 enc=list(m.g_acc2audio.model[:10].parameters());ids={id(p) for p in enc}
 names=[n for n,p in m.g_acc2audio.named_parameters() if id(p) in ids]
 for p in enc:p.requires_grad_(enabled)
 assert sum(p.numel() for p in enc)==372224
 m.eval();return enc,names
def enc_state(m,names):return {k:v.detach().cpu().clone() for k,v in m.g_acc2audio.state_dict().items() if k in names}
def apply_enc(m,state):
 result=m.g_acc2audio.load_state_dict(state,strict=False);assert not result.unexpected_keys
def feature_array(m,raw):
 out=[]
 with torch.no_grad():
  for i in range(0,len(raw),8):out.append(differentiable_features(m,spectrogram(raw[i:i+8]))[0].cpu().numpy())
 return np.concatenate(out)
def moments(scaler):return tensor(scaler.mean_),tensor(np.maximum(scaler.scale_,.001))
def normalized_features(f,stats):return (f-stats[0])/stats[1]
def sample_batch(labels,words,rng):
 ws=rng.choice(np.unique(words),2,replace=False)
 ix=np.asarray([rng.choice(np.flatnonzero((labels==uid)&(words==w))) for uid in range(9) for w in ws])
 return ix,ws
def identity_loss(net,features,stats,labels,words):
 z=net(normalized_features(features,stats));tl=torch.as_tensor(labels,device=trial.DEVICE);tw=torch.as_tensor(words,device=trial.DEVICE)
 contrast=trial.contrast(z,tl,tw,True);first=np.unique(words)[0]
 support=z[torch.as_tensor(words==first,device=trial.DEVICE)];query=z[torch.as_tensor(words!=first,device=trial.DEVICE)]
 assert len(support)==len(query)==9
 episodic=F.cross_entropy(query@support.T/.1,torch.arange(9,device=trial.DEVICE))
 return contrast+episodic
def norm_grad(gg):return float(np.sqrt(sum(float(g.detach().double().square().sum()) for g in gg if g is not None)))
def conversion(m,a,b,reference=None):
 fake=[];rec=[]
 with torch.no_grad():
  for i in range(0,len(a),8):
   x=spectrogram(a[i:i+8]);y=m.g_acc2audio(x);r=m.g_audio2acc(y);fake.append(y[:,0].cpu().numpy());rec.append(r[:,0].cpu().numpy())
 f=np.concatenate(fake);r=np.concatenate(rec);fc=f.reshape(len(f),-1);bc=b.reshape(len(b),-1);fc=fc-fc.mean(1,keepdims=True);bc=bc-bc.mean(1,keepdims=True)
 out=dict(paired_l1=float(np.abs(f-b).mean()),cycle_l1=float(np.abs(r-a).mean()),paired_corr=float(np.mean((fc*bc).sum(1)/(np.linalg.norm(fc,axis=1)*np.linalg.norm(bc,axis=1)+1e-12))))
 if reference is not None:out['output_drift_l1']=float(np.abs(f-reference).mean())
 return out,f
def quality_up(m,net,stats,up,u,baseconv,basefake):
 reg=feature_array(m,up['a'][up['registration']]);val=feature_array(m,up['va'])
 with torch.no_grad():
  c=trial.centers(net(normalized_features(tensor(reg),stats)),up['labels'][up['registration']]);score=(net(normalized_features(tensor(val),stats))@c[u-1]).cpu().numpy()
 q=trial.quality(score,up['vlabels']==u-1,up['vwords']);cv,_=conversion(m,up['own_val_acc'],up['own_val_audio'],basefake)
 safe=cv['paired_l1']<=baseconv['paired_l1']*1.10+1e-6 and cv['cycle_l1']<=baseconv['cycle_l1']*1.10+1e-6 and cv['paired_corr']>=baseconv['paired_corr']-.02 and cv['output_drift_l1']<=.05
 return dict(identity_sameword_auc=q[0],identity_auc=q[1],conversion=cv,passes_conversion_guard=bool(safe))
def up_data(u):
 d={k:[] for k in ['a','labels','words','va','vlabels','vwords']}
 for v in old.USERS:
  source=old.M['sources'][str(v)];a=np.load(str(old.DATA/('acc.%d.npy'%v))).astype('float32')[:,:80,:80]
  for role,prefix in [('gan_fit',''),('gan_val','v')]:
   ix=source['roles'][role];d[prefix+'a'].append(a[ix]);d[prefix+'labels'].append(np.full(len(ix),v-1));d[prefix+'words'].append(np.asarray(source['content_groups'])[ix])
  if v==u:
   audio=np.load(str(old.DATA/('audio.%d.npy'%v))).astype('float32')[:,:80,:80]
   d['own_fit_audio']=audio[source['roles']['gan_fit']];d['own_val_acc']=a[source['roles']['gan_val']];d['own_val_audio']=audio[source['roles']['gan_val']]
 for k in ['a','labels','words','va','vlabels','vwords']:d[k]=np.concatenate(d[k])
 d['registration']=np.asarray([np.flatnonzero((d['labels']==uid)&(d['words']==w))[0] for uid in range(9) for w in sorted(np.unique(d['words']))])
 assert d['a'].shape==(1080,80,80) and len(d['va'])==270
 return d
def upstream(u,variant,up,scaler,warmstate,original_features,original_own_fake,baseline_conv,baseline_val_fake,folder):
 m=old.load_gan(u);enc,names=enc_setup(m,variant!='frozen');before=frozen_hash(m,names);initial=enc_state(m,names)
 net=trial.Adapter().to(trial.DEVICE);net.load_state_dict(warmstate);stats=moments(scaler)
 opta=torch.optim.AdamW(net.parameters(),lr=.0003,weight_decay=.001)
 optg=torch.optim.Adam(enc,lr=1e-5) if variant!='frozen' else None
 rng=np.random.RandomState(142);history=[];audit=None
 best=None
 for step in range(UPSTEPS+1):
  if step in [0,50,100]:
   net.eval();q=quality_up(m,net,stats,up,u,baseline_conv,baseline_val_fake);history.append(dict(step=step,**q))
   rank=(-q['identity_sameword_auc'],-q['identity_auc'],q['conversion']['paired_l1']+q['conversion']['cycle_l1'],step)
   if q['passes_conversion_guard'] and (best is None or rank<best[0]):best=(rank,enc_state(m,names),copy.deepcopy(net.state_dict()),step,q)
   print('UP_VALIDATION',u,variant,step,q,flush=True)
  if step==UPSTEPS:break
  net.train();ix,ws=sample_batch(up['labels'],up['words'],rng);labs=up['labels'][ix];ww=up['words'][ix]
  if variant=='frozen':
   lid=identity_loss(net,tensor(original_features[ix]),stats,labs,ww);loss=lid
  else:
   x=spectrogram(up['a'][ix]);f,fake,recovered=differentiable_features(m,x)
   lid=identity_loss(net,f.detach() if variant=='conversion_only' else f,stats,labs,ww)
   owner=labs==u-1;own_local=ix[owner]-(u-1)*120
   pair=F.l1_loss(fake[torch.as_tensor(owner,device=trial.DEVICE),0],tensor(up['own_fit_audio'][own_local]))
   cyc=F.l1_loss(recovered[torch.as_tensor(owner,device=trial.DEVICE)],x[torch.as_tensor(owner,device=trial.DEVICE)])
   distill=F.l1_loss(fake[torch.as_tensor(owner,device=trial.DEVICE),0],tensor(original_own_fake[own_local]))
   loss=lid+10*(pair+cyc+distill)
   if step==0:
    gg=torch.autograd.grad(lid,enc+[fake],retain_graph=True,allow_unused=True)
    audit=dict(identity_encoder_grad_norm=norm_grad(gg[:-1]),identity_generated_audio_grad_norm=norm_grad(gg[-1:]))
    if variant=='identity_conversion':assert audit['identity_encoder_grad_norm']>0 and audit['identity_generated_audio_grad_norm']>0
    else:assert audit['identity_encoder_grad_norm']==0 and audit['identity_generated_audio_grad_norm']==0
  assert torch.isfinite(loss);opta.zero_grad()
  if optg:optg.zero_grad()
  loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1.)
  if optg:nn.utils.clip_grad_norm_(enc,1.);optg.step()
  opta.step()
 assert best is not None and frozen_hash(m,names)==before
 apply_enc(m,best[1]);net.load_state_dict(best[2]);delta=norm_grad([best[1][k]-initial[k] for k in names])
 checkpoint=dict(encoder_updates=best[1],upstream_adapter=best[2],selected_upstream_step=best[3],upstream_quality=best[4],history=history,gradient_audit=audit,encoder_delta_l2=delta,frozen_decoder_reverse_sha256=before,encoder_names=names)
 torch.save(checkpoint,str(folder/'upstream_private.pt'));save(folder/'upstream_public.json',{k:v for k,v in checkpoint.items() if k not in ['encoder_updates','upstream_adapter']})
 for p in m.parameters():p.requires_grad_(False)
 assert frozen_hash(m,names)==before
 return m,net,stats,checkpoint
def downstream(u,m,net,stats,p,folder):
 roles=['auth_train','model_validation','target_dev_pool','impostor_calibration'];features={role:feature_array(m,p.arrays[role]) for role in roles}
 np.savez_compressed(str(folder/'development_features_private.npz'),**features)
 xx=normalized_features(tensor(features['auth_train']),stats);vv=normalized_features(tensor(features['model_validation']),stats);labels=p.labels['auth_train']
 def words(role):return np.asarray([old.M['sources'][s.split(':')[0]]['content_groups'][int(s.split(':')[1])] for s in p.sample_ids[role]])
 w=words('auth_train');tl=torch.as_tensor(labels,device=trial.DEVICE);tw=torch.as_tensor(w,device=trial.DEVICE)
 opt=torch.optim.AdamW(net.parameters(),lr=.0003,weight_decay=.001);rng=np.random.RandomState(42);best=None;history=[]
 for step in range(1,AUTHSTEPS+1):
  net.train();z=net(xx);loss=trial.contrast(z,tl,tw,True)+trial.episode(z,labels,w,rng)
  opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
  if step%25==0:
   net.eval()
   with torch.no_grad():
    centers=trial.centers(net(xx),labels);zv=net(vv);s=(zv@centers[u-1]).cpu().numpy();q=trial.quality(s,p.labels['model_validation']==u-1,words('model_validation'));ce=float(F.cross_entropy(zv@centers.T/.1,torch.as_tensor(p.labels['model_validation'],device=trial.DEVICE)))
   history.append(dict(step=step,validation_sameword_auc=q[0],validation_auc=q[1],validation_ce=ce));rank=(-q[0],-q[1],ce,step)
   if best is None or rank<best[0]:best=(rank,copy.deepcopy(net.state_dict()),step)
 net.load_state_dict(best[1]);net.eval()
 with torch.no_grad():
  c=trial.centers(net(xx),labels);cal={role:(net(normalized_features(tensor(features[role]),stats))@c[u-1]).cpu().numpy() for role in ['target_dev_pool','impostor_calibration']}
 threshold=float(old.base.original.choose_threshold(cal['target_dev_pool'],cal['impostor_calibration']))
 torch.save(dict(state=best[1],selected_step=best[2],threshold=threshold),str(folder/'auth_head_private.pt'))
 save(folder/'auth_locked_before_final.json',dict(selected_step=best[2],threshold=threshold,history=history,generator_frozen_during_auth=True,new_final_scores_seen=False))
 return net,threshold
def target(u):
 folder=ROOT/('U%d'%u);folder.mkdir(exist_ok=True);up=up_data(u);p=old.protocol(u);original=old.load_gan(u)
 # A differentiable implementation must reproduce the original feature values before any update.
 with torch.no_grad():
  x=spectrogram(up['a'][:4]);assert torch.allclose(differentiable_features(original,x)[0],old.base.fusion_inputs(original,x),atol=2e-4,rtol=2e-4)
 base_features=feature_array(original,up['a']);scaler=StandardScaler().fit(base_features);old.sel.joblib.dump(scaler,str(folder/'upstream_scaler_private.joblib'));stats=moments(scaler)
 baseline_conv,baseline_fake=conversion(original,up['own_val_acc'],up['own_val_audio'])
 own=up['labels']==u-1
 with torch.no_grad():ownfake=np.concatenate([original.g_acc2audio(spectrogram(up['a'][own][i:i+8]))[:,0].cpu().numpy() for i in range(0,120,8)])
 trial.seed(SEED);warm=trial.Adapter().to(trial.DEVICE);opt=torch.optim.AdamW(warm.parameters(),lr=.0003,weight_decay=.001);rng=np.random.RandomState(42)
 for step in range(WARM):
  ix,_=sample_batch(up['labels'],up['words'],rng);loss=identity_loss(warm,tensor(base_features[ix]),stats,up['labels'][ix],up['words'][ix]);opt.zero_grad();loss.backward();opt.step()
 warmstate=copy.deepcopy(warm.state_dict());metadata={}
 del original
 for variant in VARIANTS:
  sub=folder/variant;sub.mkdir(exist_ok=True);save(ROOT/'status_public.json',dict(state='running',target=u,variant=variant,stage='upstream'))
  m,adapter,stats,ck=upstream(u,variant,up,scaler,warmstate,base_features,ownfake,baseline_conv,baseline_fake,sub)
  before=frozen_hash(m,[]);save(ROOT/'status_public.json',dict(state='running',target=u,variant=variant,stage='auth_head'))
  adapter,t=downstream(u,m,adapter,stats,p,sub);assert frozen_hash(m,[])==before
  metadata[variant]=dict(threshold=t,encoder_updates_sha256=old.sha(sub/'upstream_private.pt'),auth_generator_state_unchanged=True)
  del m,adapter
 save(folder/'all_routes_locked_before_final.json',metadata);rows={}
 # All three variants lock before any new final score or paired-audio diagnostic is read.
 for variant in VARIANTS:
  sub=folder/variant;m=old.load_gan(u);ck=torch.load(str(sub/'upstream_private.pt'),map_location='cpu',weights_only=False);apply_enc(m,ck['encoder_updates'])
  assert frozen_hash(m,ck['encoder_names'])==ck['frozen_decoder_reverse_sha256']
  head=torch.load(str(sub/'auth_head_private.pt'),map_location=trial.DEVICE,weights_only=False);net=trial.Adapter().to(trial.DEVICE);net.load_state_dict(head['state']);net.eval()
  dev=np.load(str(sub/'development_features_private.npz'));stats=moments(scaler);scores={};final_features={}
  with torch.no_grad():
   c=trial.centers(net(normalized_features(tensor(dev['auth_train']),stats)),p.labels['auth_train'])
   for role in ['final_genuine','final_impostor']:
    f=feature_array(m,p.arrays[role]);final_features[role]=f;scores[role]=(net(normalized_features(tensor(f),stats))@c[u-1]).cpu().numpy()
  r=old.sel.evaluate_route(scores,p,u,head['threshold']);r.update(selected_upstream_step=ck['selected_upstream_step'],selected_auth_step=head['selected_step'],gradient_audit=ck['gradient_audit'],encoder_delta_l2=ck['encoder_delta_l2'],upstream_conversion=ck['upstream_quality']['conversion'])
  audio=np.load(str(old.DATA/('audio.%d.npy'%u))).astype('float32')[:,:80,:80];ix=[int(s.split(':')[1]) for s in p.sample_ids['final_genuine']]
  cv,_=conversion(m,p.arrays['final_genuine'],audio[ix]);r['final_conversion_diagnostic']=cv
  np.savez_compressed(str(sub/'final_scores_private.npz'),**scores,attack_labels=p.labels['final_impostor']);np.savez_compressed(str(sub/'final_features_private.npz'),**final_features)
  save(sub/'results_public.json',r);rows[variant]=r;print('ROUTE_COMPLETE',u,variant,r,flush=True);del m,net
 save(folder/'results_public.json',dict(target=u,routes=rows,baseline_upstream_conversion=baseline_conv));print('TARGET_COMPLETE',u,flush=True)
def main():
 ROOT.mkdir(exist_ok=True)
 with (ROOT/'QUEUE_LOCK').open('x') as f:f.write(str(os.getpid()))
 old.records();save(ROOT/'protocol_public.json',dict(targets=[4,1,5],seed=SEED,variants=VARIANTS,encoder='G.model[:10]: initial convolution and two downsamples;6 tensors372224 params;all bottleneck/residual/attention/decoder/F frozen',feature_path='New differentiable772;no no_grad inside cycle path',upstream='Nine identities original gan_fit120 each only;conversion losses only target owner;gan_val30 peridentity used for selection;NO auth rows update G',warm_adapter_steps=WARM,upstream_steps=UPSTEPS,encoder_lr=1e-5,adapter_lr=.0003,upstream_loss='identity cross-word+1shot-support/query CE; plus10*(owner pairedL1+owner cycleL1+original owner fakeAudio driftL1)',validation_guard='owner gan_val pairedL1 and cycleL1 <=1.10baseline;pairedcorr>=baseline-.02;output drift<=.05;best passing upstream identity-wordAUC at0/50/100',standardization='Fixed original G upstream1080 mean/std;scale floor.001;same all variants;NO auth or test stats',downstream='Same upstream-trained adapter fine-tuned400 steps with G locked;auth50/val30/cal20;all eight attackers;final test unchanged',limitations=['Single-seed pilot','Frozen original generator contrast differs from previous adapter experiment because of upstream adapter pretraining and upstream scaler','Known identities not open-set','Historical final data reused','Content10/20period and tail-missing mapping','Identity gradients see other users upstream data;personal G is now identity-supervised,not purely owner-only unsupervised conversion'],source_sha256=old.sha(Path(__file__))))
 rows={}
 for u in [4,1,5]:target(u);rows[str(u)]=json.loads((ROOT/('U%d'%u)/'results_public.json').read_text());save(ROOT/'results_public.json',rows)
 save(ROOT/'status_public.json',dict(state='complete',completed=list(rows)))
def smoke(u):
 m=old.load_gan(u);enc,names=enc_setup(m,True);up=up_data(u);ix,_=sample_batch(up['labels'],up['words'],np.random.RandomState(42));f,fake,rec=differentiable_features(m,spectrogram(up['a'][ix]));net=trial.Adapter().to(trial.DEVICE)
 loss=identity_loss(net,f,(torch.zeros(772,device=trial.DEVICE),torch.ones(772,device=trial.DEVICE)),up['labels'][ix],up['words'][ix]);gg=torch.autograd.grad(loss,enc+[fake],allow_unused=True)
 print(dict(shape=list(f.shape),encoder_gradient=norm_grad(gg[:-1]),fake_audio_gradient=norm_grad(gg[-1:]),peak_cuda_mb=torch.cuda.max_memory_allocated()/1024**2,allowed=names),flush=True)
if __name__=='__main__':
 if len(sys.argv)>1:smoke(int(sys.argv[1]))
 else:main()

