"""Feature MKD adaptation of AAAI2023 Eq2/Eq4; same protocolA matched controls."""
import sys,json,copy
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
sys.path.insert(0,'/root')
import run_audio_guided_acc_kd_20261030 as prev
old=prev.old;D=prev.D;ROOT=Path('/autodl-fs/data/mei_relaxed_audio_feature_kd_20261030');ALPHA=.6;MARGINS=[float(np.cos(np.pi/6)),float(np.cos(np.pi/4))];LAMBDAS=[.1,.5,1.]

class Projection(nn.Module):
 def __init__(self):super().__init__();self.net=nn.Sequential(nn.Linear(128,128),nn.GELU(),nn.Linear(128,128),nn.GELU(),nn.Linear(128,128))
 def forward(self,z):return F.normalize(ALPHA*z+(1-ALPHA)*self.net(z),dim=1)

def load_previous(folder,key):
 ck=torch.load(str(folder/(key+'_private.pt')),map_location=D,weights_only=False);net=prev.Classifier().to(D);net.aam=ck['metadata']['mode']=='aam';net.load_state_dict(ck['state']);net.eval()
 for p in net.parameters():p.requires_grad_(False)
 return net,ck

def shuffle_map(data,s):
 # Same identity, different word and different recording; permutation of train only.
 labels=data['train']['labels'];words=data['train']['words'];rng=np.random.RandomState(s+9000);mapping=np.arange(len(labels))
 for u in range(9):
  shift=int(rng.randint(1,len(prev.ROLES['train'])));ww=np.asarray(prev.ROLES['train']);dest=np.roll(ww,shift)
  for w,v in zip(ww,dest):
   a=np.flatnonzero((labels==u)&(words==w));b=np.flatnonzero((labels==u)&(words==v));mapping[a]=rng.permutation(b)
 assert len(np.unique(mapping))==len(mapping) and np.array_equal(labels,labels[mapping]) and np.all(words!=words[mapping]) and np.all(mapping!=np.arange(len(mapping)))
 return mapping

def train(data,s,folder,key,teacher_z,lam,margin,mapping=None):
 prev.seed(s);net=prev.Classifier().to(D);projection=Projection().to(D);params=list(net.parameters())+list(projection.parameters());opt=torch.optim.AdamW(params,lr=3e-4,weight_decay=.001);rng=np.random.RandomState(s);best=None;history=[];lookup={(u,w):np.flatnonzero((data['train']['labels']==u)&(data['train']['words']==w)) for u in range(9) for w in prev.ROLES['train']}
 for step in range(1,prev.STEPS+1):
  net.train();projection.train();ww=rng.choice(prev.ROLES['train'],4,replace=False);ix=np.concatenate([rng.choice(lookup[u,w],2,replace=False) for u in range(9) for w in ww]);y=torch.as_tensor(data['train']['labels'][ix],device=D);z,logits=net(prev.tensor(data['train']['acc'][ix,None]));tz=teacher_z[ix if mapping is None else mapping[ix]];target=projection(tz);cos=(z*target).sum(1);mkd=F.relu(margin-cos).mean();ce=F.cross_entropy(logits,y);loss=ce+lam*mkd
  opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(params,1.);opt.step()
  if step%25==0:
   q=prev.validation_quality(net,data,'acc');row=dict(step=step,loss=float(loss.detach()),ce=float(ce.detach()),mkd=float(mkd.detach()),mean_cosine=float(cos.detach().mean()),active_fraction=float((cos.detach()<margin).float().mean()),quality=q);history.append(row);rank=prev.rank(q)+(step,)
   if best is None or rank<best[0]:best=(rank,copy.deepcopy(net.state_dict()),copy.deepcopy(projection.state_dict()),step,q)
 net.load_state_dict(best[1]);projection.load_state_dict(best[2]);net.eval();projection.eval()
 for p in list(net.parameters())+list(projection.parameters()):p.requires_grad_(False)
 meta=dict(key=key,seed=s,lambda_mkd=lam,margin=margin,alpha=ALPHA,steps=prev.STEPS,selected_step=best[3],quality=best[4],history=history,shuffle=mapping is not None)
 torch.save(dict(state=best[1],projection=best[2],metadata=meta),str(folder/(key+'_private.pt')));print('TRAINED',s,key,best[3],flush=True);return net,projection,meta

def run_seed(s,data):
 folder=ROOT/('seed%d'%s);folder.mkdir(exist_ok=True);source=prev.ROOT/('seed%d'%s);oldlock=json.loads((source/'locked_before_final.json').read_text());e0,ck0=load_previous(source,'B0_CE');e1_source=oldlock['chosen']['S1_AudioKD'];e1,ck1=load_previous(source,e1_source);audio,_=load_previous(source,'T1_Audio')
 models={'E0_CE':e0,'E1_LogitKL':e1};chosen={'E0_CE':dict(source='previous',key='B0_CE'),'E1_LogitKL':dict(source='previous',key=e1_source)};selection={};metadata={};_,dummy=prev.inference(e0,data,'train');del dummy
 az,_=prev.inference(audio,data,'train','audio');cz,_=prev.inference(e0,data,'train','acc');targets={'E2_AudioMKD':prev.tensor(az),'E3_SelfMKD':prev.tensor(cz)}
 mapping=shuffle_map(data,s);np.save(str(folder/'shuffle_map_private.npy'),mapping)
 for family in ['E2_AudioMKD','E3_SelfMKD']:
  candidates=[(prev.rank(ck0['metadata']['quality'])+(0.,0.),e0,None,None,'E0_CE')];selection[family]=[]
  for lam in LAMBDAS:
   for mi,m in enumerate(MARGINS):
    key=family+'_L%s_M%d'%(str(lam).replace('.','p'),mi);net,proj,meta=train(data,s,folder,key,targets[family],lam,m);selection[family].append(meta);candidates.append((prev.rank(meta['quality'])+(lam,m),net,proj,meta,key))
  best=min(candidates,key=lambda q:q[0]);models[family]=best[1];metadata[family]=best[3];chosen[family]=dict(source='current' if best[3] else 'previous',key=best[4] if best[3] else 'B0_CE',lambda_mkd=best[3]['lambda_mkd'] if best[3] else 0.,margin=best[3]['margin'] if best[3] else None)
 selected=metadata['E2_AudioMKD']
 if selected is not None:
  net,proj,meta=train(data,s,folder,'E2_Shuffled',targets['E2_AudioMKD'],selected['lambda_mkd'],selected['margin'],mapping);models['E2_Shuffled']=net;metadata['E2_Shuffled']=meta;chosen['E2_Shuffled']=dict(source='current',key='E2_Shuffled',lambda_mkd=meta['lambda_mkd'],margin=meta['margin'])
 else:models['E2_Shuffled']=e0;metadata['E2_Shuffled']=None;chosen['E2_Shuffled']=dict(source='previous',key='B0_CE',lambda_mkd=0.,diagnostic_informative=False)
 # Projected teacher identity drift is development-only, never final Audio.
 diagnostic={}
 for family,teacher,domain in [('E2_AudioMKD',audio,'audio'),('E3_SelfMKD',e0,'acc')]:
  meta=metadata[family]
  if meta is None:diagnostic[family]=dict(selected_lambda=0.);continue
  ck=torch.load(str(folder/(meta['key']+'_private.pt')),map_location=D,weights_only=False);proj=Projection().to(D);proj.load_state_dict(ck['projection']);proj.eval();z,l=prev.inference(teacher,data,'validation',domain)
  with torch.no_grad():projected=proj(prev.tensor(z));pl=teacher.head(projected).cpu().numpy();norm=float(torch.linalg.vector_norm(proj.net(prev.tensor(z)),dim=1).mean())
  diagnostic[family]=dict(raw_teacher_top1=float((l.argmax(1)==data['validation']['labels']).mean()),projected_teacher_top1=float((pl.argmax(1)==data['validation']['labels']).mean()),projection_component_norm=norm)
 locks={}
 for key,net in models.items():
  z={r:prev.inference(net,data,r)[0] for r in ['enroll','validation','calibration']};locks[key]=prev.backend_lock(z,data,folder,key)
 old.save(folder/'locked_before_final.json',dict(seed=s,chosen=chosen,selection=selection,metadata=metadata,projection_diagnostic=diagnostic,routes=locks,final_seen=False,shuffle_rule='within identity different word bijection;trainonly;fixed paired lambda/m'))
 rows={k:prev.evaluate_final(net,data,folder,k,locks[k]) for k,net in models.items()};previous=json.loads((source/'results_public.json').read_text())
 for key,pk in [('E0_CE','B0_CE'),('E1_LogitKL','S1_AudioKD')]:
  for name,r in rows[key].items():
   for metric in ['tar','far','auc','hter']:assert abs(r[metric]-previous['routes'][pk][name][metric])<1e-12
 out=dict(seed=s,routes=rows,chosen=chosen,projection_diagnostic=diagnostic,controls_replayed=True);old.save(folder/'results_public.json',out);print('SEED_DONE',s,flush=True);return out

def main():
 ROOT.mkdir(exist_ok=True);data=prev.data_load();old.save(ROOT/'protocol_public.json',dict(seeds=prev.SEEDS,groups=prev.ROLES,counts='180train/20independentenroll/30validation/20cal/47-50test peridentity',student='sameCNN128 and init/batches400updates as previous;E0/E1 checkpoints reused and replayed',feature_mkd='mean relu(m-cos(zS,normalize(.6*zT+.4*MLP(zT))));MLP3linear128 with GELU,trainable;teacher frozen',reference='AAAI2023 Eq2/Eq4 https://ojs.aaai.org/index.php/AAAI/article/view/26525/26297 ;adaptation not fullVGSR replication',search='lambda0/.1/.5/1; margin cos30/cos45;6nonzero candidates vs6oldKL;validation-only;selfMKD same grid',shuffle='same identity crossword bijection, no exact pair; matched selected E2 lambda/m/init/batches/steps, own validation checkpoint; no shufflehyper search',deployment='only ACC Student plus cosineK3/5/10/20 orRBF20;no teacher/projection/Audio inference',limitations=['known9identities protocolA,not unseenidentity','historical final reused exploratory single-session','noQAW,nofusion teacher thisround','learned projection extra trainable parameters but removed at inference','equal permodel updates and6candidates forcore distillers,not equalteacher pretrainingresources','shuffle tests recording/content pairing dependence,not physical timing alone','equal shuffle performance may indicate useful identity-level transfer,not automatic failure']))
 rows={}
 for s in prev.SEEDS:
  old.save(ROOT/'status_public.json',dict(state='running',seed=s,completed=list(rows)));rows[str(s)]=run_seed(s,data);old.save(ROOT/'results_public.json',rows)
 old.save(ROOT/'status_public.json',dict(state='complete',completed=list(rows)))
if __name__=='__main__':
 try:main()
 except Exception as e:
  ROOT.mkdir(exist_ok=True);old.save(ROOT/'status_public.json',dict(state='failed',error=repr(e)));raise
