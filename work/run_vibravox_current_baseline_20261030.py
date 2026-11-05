"""Current full772 balanced RBF + personal ACC/generated-Audio teacher, VibraVox pilot."""
import sys,json,copy,hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.model_selection import StratifiedKFold
sys.path.insert(0,'/root')
import run_nine_sameword_balanced_20261030 as shared
old=shared.old;D=old.D
SOURCE=Path('/autodl-fs/data/public_baseline_compare_20261028/vibravox');ROOT=Path('/autodl-fs/data/vibravox_current_baseline_20261030');GRID=[0.,.25,.5,1.]

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def configuration():
 p=json.loads((SOURCE/'protocol_private.json').read_text());d=dict(np.load(str(SOURCE/'inputs.npz')));units=np.unique(d['unit_id']);speaker={int(u):str(d['speaker_id'][d['unit_id']==u][0]) for u in units};roles={}
 for who,rr in p['roles'].items():
  values=sum(rr.values(),[]);assert len(values)==len(set(values))
  for role,ids in rr.items():
   for u in ids:assert speaker[u]==who and (d['unit_id']==u).sum()==2
  roles[who]=rr
 normalized=set(p['normalization_fit_units']);expected=set(sum([rr['gan_train'] for who,rr in roles.items() if who in p['owners']],[]));assert normalized==expected
 assert not expected&set(sum([roles[who]['final'] for who in p['unknown']],[]))
 texts={};text_roles={}
 for s in p['source']:
  texts[int(s['unit'])]=s['text']
 for who,rr in roles.items():
  for role,ids in rr.items():
   for u in ids:
    if texts[u] in text_roles:assert text_roles[texts[u]]==role
    text_roles[texts[u]]=role
 return p,d,speaker

def fold_quality(s,y,folds):return shared.fold_quality(s,y,np.arange(len(y)),folds)
def choose_params(x,y,seed):
 folds=list(StratifiedKFold(3,shuffle=True,random_state=seed).split(x,y));best=None
 for c in [1.,10.,100.]:
  for gamma in ['scale',.1]:
   pred=np.full(len(y),np.nan)
   for tr,va in folds:
    m=old.sel.rbf.fit([x[tr]],y[tr],c,gamma);s=old.sel.rbf.score(m,[x[tr]]);pred[va]=(old.sel.rbf.score(m,[x[va]])-s.mean())/(s.std()+1e-6)
   q=fold_quality(pred,y,folds);rank=shared.quality_rank(q)+(c,0 if gamma==.1 else 1)
   if best is None or rank<best[0]:best=(rank,c,gamma,q)
 return best[1:]
def branch(x,y):
 folds=list(StratifiedKFold(5,shuffle=True,random_state=42).split(x,y));oof=np.full(len(y),np.nan);cfg=[]
 for i,(tr,va) in enumerate(folds):
  c,gamma,q=choose_params(x[tr],y[tr],71+i);m=old.sel.rbf.fit([x[tr]],y[tr],c,gamma);s=old.sel.rbf.score(m,[x[tr]]);oof[va]=(old.sel.rbf.score(m,[x[va]])-s.mean())/(s.std()+1e-6);cfg.append(dict(C=c,gamma=gamma))
 c,gamma,q=choose_params(x,y,71);m=old.sel.rbf.fit([x],y,c,gamma);s=old.sel.rbf.score(m,[x]);return oof,dict(model=m,mean=float(s.mean()),std=float(s.std()+1e-6),C=c,gamma=gamma,outer_configs=cfg)

def unit_pool(z,d,ids):return np.stack([z[d['unit_id']==u].mean(0) for u in ids])
@torch.no_grad()
def teacher_embed(t,x,domain):
 return np.concatenate([getattr(t,domain)(old.te.tensor(x[i:i+32])).cpu().numpy() for i in range(0,len(x),32)])

def train_teacher(owner,p,d,folder):
 owners=p['owners'];target=owners.index(owner);old.base.seed(4300+target);rng=np.random.RandomState(4300+target);t=old.base.Teacher().to(D);identity=nn.Linear(64,2).to(D);params=list(t.parameters())+list(identity.parameters());opt=torch.optim.AdamW(params,lr=.0005,weight_decay=.003)
 fit={who:p['roles'][who]['gan_train'] for who in owners};val=sum([p['roles'][who]['gan_val'] for who in owners],[]);best=None;history=[]
 valmask=np.isin(d['unit_id'],val)
 def pooldomain(domain,x):return unit_pool(teacher_embed(t,x[valmask],domain),{'unit_id':d['unit_id'][valmask]},val)
 vl=np.asarray([int(u in p['roles'][owner]['gan_val']) for u in val])
 for step in range(1,601):
  t.train();uids=sum([rng.choice(fit[who],8,replace=False).tolist() for who in owners],[]);ix=np.asarray([rng.choice(np.flatnonzero(d['unit_id']==u)) for u in uids]);labels=torch.as_tensor(np.repeat(np.arange(2),8),device=D);words=torch.as_tensor(uids,device=D)
  za=t.acc(old.te.tensor(d['acc'][ix])+.003*torch.randn((len(ix),1,80,80),device=D));zm=t.audio(old.te.tensor(d['audio'][ix])+.003*torch.randn((len(ix),1,80,80),device=D));z=torch.cat([za,zm]);labs=torch.cat([labels,labels]);unitlabels=torch.cat([words,words]);binary=(labs==target).float()
  loss=F.binary_cross_entropy_with_logits(t.binary(z).flatten(),binary)+.3*F.cross_entropy(identity(z),labs)+.1*shared_contrast(z,labs,unitlabels)+.1*(1-(za*zm).sum(1)).mean()
  opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(params,1.);opt.step()
  if step%25==0:
   t.eval();values=[];bces=[]
   for domain,x in [('acc',d['acc']),('audio',d['audio'])]:
    e=pooldomain(domain,x)
    with torch.no_grad():s=t.binary(torch.as_tensor(e,device=D)).flatten().cpu().numpy()
    values.append(old.sel.auc(vl,s));bces.append(float(F.binary_cross_entropy_with_logits(torch.as_tensor(s,device=D),torch.as_tensor(vl,dtype=torch.float32,device=D))))
   row=dict(step=step,mean_real_modality_val_auc=float(np.mean(values)),val_bce=float(np.mean(bces)),training_loss=float(loss.detach()));history.append(row);rank=(-row['mean_real_modality_val_auc'],row['val_bce'],step)
   if best is None or rank<best[0]:best=(rank,step,copy.deepcopy(t.state_dict()))
 t.load_state_dict(best[2]);t.eval();meta=dict(selected_step=best[1],seed=4300+target,fit_units=fit,val_units=val,steps=600,history=history,known_identities=owners,unseen_excluded=p['unknown'])
 torch.save(dict(state=best[2],metadata=meta),str(folder/'teacher_private.pt'));old.save(folder/'teacher_training_public.json',{k:v for k,v in meta.items() if k not in ['fit_units','val_units']});return t,meta

def shared_contrast(z,labs,units):
 same=labs[:,None]==labs[None,:];different_unit=units[:,None]!=units[None,:];eye=torch.eye(len(z),device=D,dtype=torch.bool);positive=same&different_unit;valid=~same|positive;logits=z@z.T/.1;logp=logits-torch.logsumexp(logits.masked_fill(~valid,-1e9),dim=1,keepdim=True);assert (positive.sum(1)>0).all();return -((logp*positive).sum(1)/positive.sum(1)).mean()

def candidates(oof,y,mode,anchor=None):
 folds=list(StratifiedKFold(5,shuffle=True,random_state=42).split(np.zeros(len(y)),y));rows=[]
 for rw in (GRID if anchor is None else [anchor['raw']]):
  for aw in (GRID if mode in ['acc','both'] else [anchor['acc']] if anchor is not None else [0.]):
   for mw in (GRID if mode in ['audio','both','fixed_acc_audio'] else [0.]):
    s=oof['full772']+rw*oof['raw320']+aw*oof['teacher_acc']+mw*oof['teacher_audio'];q=fold_quality(s,y,folds);rows.append(dict(raw=rw,acc=aw,audio=mw,quality=q))
 return rows
def rank(c):return shared.quality_rank(c['quality'])+(c['raw']+c['acc']+c['audio'],c['raw'],c['acc'],c['audio'])
def score(models,features,c,ids):
 out=None
 for k,w in c['weights'].items():
  if not w:continue
  m=models[k];x=np.stack([features[k][int(u)] for u in ids]);v=(old.sel.rbf.score(m['model'],[x])-m['mean'])/m['std'];out=w*v if out is None else out+w*v
 return out

def target(owner,p,d,speakers):
 folder=ROOT/owner;folder.mkdir(parents=True,exist_ok=True);source=SOURCE/owner;cache=dict(np.load(str(source/'frozen_features_private.npz')));assert str(cache['input_sha'])==sha(SOURCE/'inputs.npz') and str(cache['checkpoint_sha'])==sha(source/'models/checkpoint_epoch_200.pth')
 t,meta=train_teacher(owner,p,d,folder);cache['teacher_acc']=teacher_embed(t,d['acc'],'acc');cache['teacher_audio']=teacher_embed(t,cache['generated'],'audio')
 names=['full772','raw320','input_encoder640','cycle_encoder640','teacher_acc','teacher_audio'];units=np.unique(d['unit_id']);features={k:{int(u):cache[k][d['unit_id']==u].mean(0) for u in units} for k in names};known=next(w for w in p['owners'] if w!=owner);fit=p['roles'][owner]['auth_train']+p['roles'][known]['auth_train'];y=np.asarray([speakers[u]==owner for u in fit]);models={};oof={}
 assert len(fit)==20 and y.sum()==10
 for k in names:
  x=np.stack([features[k][u] for u in fit]);oof[k],models[k]=branch(x,y);print('BRANCH_DONE',owner,k,flush=True)
 searches={};selected={}
 for mode in ['acc','audio','both']:
  searches[mode]=candidates(oof,y,mode);selected[mode]=min(searches[mode],key=rank)
 searches['fixed_acc_audio']=candidates(oof,y,'fixed_acc_audio',selected['acc']);selected['fixed_acc_audio']=min(searches['fixed_acc_audio'],key=rank)
 specs={k:dict(weights={k:1.}) for k in ['full772','raw320','input_encoder640','cycle_encoder640']}
 for mode,c in selected.items():specs[mode]=dict(weights=dict(full772=1.,raw320=c['raw'],teacher_acc=c['acc'],teacher_audio=c['audio']))
 for mode,c in specs.items():
  g=score(models,features,c,p['roles'][owner]['calibration']);n=score(models,features,c,p['roles'][known]['calibration']);c['threshold']=shared.balanced_threshold(g,n);c['calibration_tar']=float((g>=c['threshold']).mean());c['calibration_far']=float((n>=c['threshold']).mean())
 old.sel.joblib.dump(models,str(folder/'models_private.joblib'));np.savez_compressed(str(folder/'oof_private.npz'),**oof);np.savez_compressed(str(folder/'teacher_features_private.npz'),teacher_acc=cache['teacher_acc'],teacher_audio=cache['teacher_audio'])
 old.save(folder/'locked_before_final.json',dict(owner=owner,known_negative=known,routes=specs,searches=searches,selected=selected,final_scores_seen=False))
 rows={};dump={}
 for mode,c in specs.items():
  g=score(models,features,c,p['roles'][owner]['final']);neg={w:score(models,features,c,p['roles'][w]['final']) for w in [known]+p['unknown']};unseen=np.concatenate([neg[w] for w in p['unknown']]);allneg=np.concatenate(list(neg.values()));threshold=c['threshold']
  row=dict(known=old.sel.exp.metrics(g,neg[known],threshold),unseen=old.sel.exp.metrics(g,unseen,threshold),all_attackers=old.sel.exp.metrics(g,allneg,threshold),weights=c['weights'],threshold=threshold,per_attacker={w:dict(n=len(s),accepted=int((s>=threshold).sum())) for w,s in neg.items()})
  for metric,n in [(row['known'],neg[known]),(row['unseen'],unseen),(row['all_attackers'],allneg)]:assert abs(((g[:,None]>n)+.5*(g[:,None]==n)).mean()-metric['auc'])<1e-12
  rows[mode]=row;dump[mode+'__genuine']=g
  for w,s in neg.items():dump[mode+'__'+w]=s
 np.savez_compressed(str(folder/'scores_private.npz'),**dump);out=dict(owner=owner,known_negative=known,unseen=p['unknown'],routes=rows,teacher_selected_step=meta['selected_step'],checkpoint_sha256=str(cache['checkpoint_sha']),input_sha256=str(cache['input_sha']),fit_positive=10,fit_negative=10,final_positive=8);old.save(folder/'results_public.json',out);print('TARGET_DONE',owner,flush=True);return out

def main():
 ROOT.mkdir(exist_ok=True);p,d,speakers=configuration();old.save(ROOT/'protocol_public.json',dict(dataset='VibraVox speech_clean foreheadACC+headsetmic',owners=p['owners'],unseen=p['unknown'],source='existing200epoch personal canonical models;not retrained',teacher='new personal ACC/realAudio teacher perowner on both known owners GANtrain only;600steps;valmean realmodalityAUC;unknown excluded',data='30GANtrain/6GANval/10authtrain/6cal/8final whole utterances perowner;2unseen24final each;2central0.8s windows perutterance pooled beforeauth',normalization='existing shared GANtrain-only scales applied to all identities/roles;no peruser scale',content='whole sentence split with no exact text crossing roles;different speakers not same prompt;not same-content experiment',authentication='same full772 definition and current balanced RBF recipe;5outer3inner at utterance rows;equalHTER threshold calibration;no TAR priority',selection='optionalraw+one personalACCteacher+one generatedAudio teacher;16ACC/16Audio/64joint/4fixedACC candidates;allroutes locked before final',limitations=['tiny historical pilot,only8independent genuine finals and48unseen attacks','one known negative identity','cohort previously explored,not blind','reusedOOF metaselection not independent nesting','forehead dedicated vibration sensor differs from VR IMU','no sameprompt content matching','old200epoch generator used60windows=30utterances only,not fullVibraVox training']))
 rows={}
 for owner in p['owners']:
  old.save(ROOT/'status_public.json',dict(state='running',owner=owner,completed=list(rows)));rows[owner]=target(owner,p,d,speakers);old.save(ROOT/'results_public.json',rows)
 old.save(ROOT/'status_public.json',dict(state='complete',completed=list(rows)))
if __name__=='__main__':
 try:main()
 except Exception as e:
  ROOT.mkdir(exist_ok=True);old.save(ROOT/'status_public.json',dict(state='failed',error=repr(e)));raise

