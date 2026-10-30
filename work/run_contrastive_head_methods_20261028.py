"""Controlled, personal contrastive-head ablations on frozen complete baseline features.

Every CV fold refits projection and scalers. Final queries never choose a model.
CHNS-inspired full-batch reweighting is not a reproduction of CHNS batch sampling.
"""
import copy,json,sys,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,roc_curve,accuracy_score
import joblib
sys.path.insert(0,'/root')
import personal_rbf_auth_20261028 as rbf
exp=rbf.exp;base=exp.base
FEATURE_ROOT=Path('/autodl-fs/data/autoselect_diffusion_supplement_20261028')
OUT=Path('/autodl-fs/data/contrastive_head_methods_20261028')
D=exp.D
SEEDS=(42,43,44);EPOCHS=(50,100,200)
VARIANTS=('compact64','compact128','crossword64','semi_hard64','factorized64_16')
PARAMS=[(c,g) for c in (1.,10.,100.) for g in ('scale',.1)]

def protocol_module():
 scope={};prefix=Path('/root/run_expanded_auth_test_20261028.py').read_text().split("source=Path('/root/refined_teacher_rbf_auth_20261028.py').read_text()",1)[0]
 exec(compile(prefix,'contrastive_protocol','exec'),scope);return scope['expanded']

class Head(nn.Module):
 def __init__(self,variant):
  super().__init__();dim=128 if variant=='compact128' else 64
  self.identity=nn.Linear(772,dim);self.binary=nn.Linear(dim,1)
  self.factorized=variant=='factorized64_16'
  if self.factorized:self.content=nn.Linear(772,16);self.word_classifier=nn.Linear(16,30)
 def forward(self,x):return F.normalize(self.identity(x),dim=1)

@torch.no_grad()
def hard_weights(z,uid):
 identities=torch.unique(uid);centers=torch.stack([F.normalize(z[uid==u].mean(0),dim=0) for u in identities])
 sim=centers@centers.T;sim.fill_diagonal_(-1e4);nearest=sim.topk(min(2,len(identities)-1),dim=1).indices
 hard=torch.zeros((len(uid),len(uid)),dtype=torch.bool,device=D)
 for i,u in enumerate(identities):
  negative_ids=identities[nearest[i]]
  hard|=(uid[:,None]==u)&torch.isin(uid[None,:],negative_ids)
 pair=z@z.T;positive=(uid[:,None]==uid[None,:]);positive.fill_diagonal_(False)
 ceiling=pair.masked_fill(~positive,-1e4).max(1).values
 hard&=pair<ceiling[:,None]
 return hard

def contrastive(z,uid,word,variant,strength):
 n=len(z);eye=torch.eye(n,dtype=torch.bool,device=D);same=uid[:,None]==uid[None,:]
 pos=same&~eye;valid=~eye
 if variant=='crossword64':
  pos&=word[:,None]!=word[None,:]
  valid=pos|~same
 logits=z@z.T/.07
 if variant=='semi_hard64' and strength>0:
  neg=~same;weights=torch.ones_like(logits)+strength*hard_weights(z.detach(),uid).float()
  # Preserve each anchor's total negative weight; no sample or user is removed.
  normalizer=neg.sum(1,keepdim=True)/(weights*neg).sum(1,keepdim=True).clamp_min(1)
  weights=torch.where(neg,weights*normalizer,torch.ones_like(weights))
  denominator=logits+weights.log()
 else:denominator=logits
 denominator=denominator.masked_fill(~valid,-1e4)
 logp=logits-torch.logsumexp(denominator,dim=1,keepdim=True)
 keep=pos.sum(1)>0;assert keep.all()
 return -((logp*pos).sum(1)/pos.sum(1)).mean()

def fit_projection(x,uid,word,target,variant,seed,epochs,collect=()):
 base.seed(seed);scaler=StandardScaler().fit(x)
 tx=torch.as_tensor(scaler.transform(x)/np.sqrt(x.shape[1]),dtype=torch.float32,device=D)
 ids=torch.as_tensor(uid,dtype=torch.long,device=D);words=torch.as_tensor(word,dtype=torch.long,device=D)
 yy=(ids==target-1).float();assert yy.sum()>0 and yy.sum()<len(yy)
 model=Head(variant).to(D);optimizer=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.01)
 pos_weight=(yy==0).sum()/(yy==1).sum();states={};losses=[]
 for epoch in range(1,epochs+1):
  model.train();z=model(tx)
  strength=max(0.,min(1.,(epoch-20)/80.))
  loss=contrastive(z,ids,words,variant,strength)+F.binary_cross_entropy_with_logits(model.binary(z).squeeze(1),yy,pos_weight=pos_weight)
  if model.factorized:
   content=F.normalize(model.content(tx),dim=1)
   covariance=(z-z.mean(0)).T@(content-content.mean(0))/len(z)
   loss=loss+.2*F.cross_entropy(model.word_classifier(content),words)+covariance.square().sum()
  optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
  losses.append(float(loss.detach()))
  if epoch in collect:states[epoch]={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
 model.eval()
 return scaler,model,states,{'first_loss':losses[0],'last_loss':losses[-1],'epochs':epochs,'seed':seed}

@torch.no_grad()
def embed(model,scaler,x):
 model.eval();tx=torch.as_tensor(scaler.transform(x)/np.sqrt(x.shape[1]),dtype=torch.float32,device=D)
 return model(tx).cpu().numpy()

def cv_variant(x,uid,words,target,variant,folds):
 y=uid==target-1;table={}
 for epoch in EPOCHS:
  for c,g in PARAMS:table[str((epoch,c,g))]={'epoch':epoch,'C':c,'gamma':g,'by_seed':{str(seed):[] for seed in SEEDS}}
 for seed in SEEDS:
  for fold,(tr,va) in enumerate(folds):
   assert not set(words[tr])&set(words[va])
   scaler,model,states,_=fit_projection(x[tr],uid[tr],words[tr],target,variant,seed+1000*fold,max(EPOCHS),EPOCHS)
   for epoch,state in states.items():
    model.load_state_dict(state);ztr=embed(model,scaler,x[tr]);zva=embed(model,scaler,x[va])
    for c,g in PARAMS:
     fitted=rbf.fit([ztr],y[tr],c,g);s=rbf.score(fitted,[zva])
     table[str((epoch,c,g))]['by_seed'][str(seed)].append(float(roc_auc_score(y[va],s)))
  print('CV_SEED_DONE',target,variant,seed,flush=True)
 rows=[]
 for row in table.values():
  row['seed_mean_aucs']=[float(np.mean(v)) for v in row['by_seed'].values()]
  row['mean_fold_auc']=float(np.mean(row['seed_mean_aucs']));row['std_seed_auc']=float(np.std(row['seed_mean_aucs']))
  rows.append(row)
 selected=min(rows,key=lambda r:(-r['mean_fold_auc'],r['epoch'],r['C'],0 if r['gamma']==.1 else 1))
 return {'selected':selected,'candidates':rows}

def baseline_cv(x,y,words,folds):
 rows=[]
 for c,g in PARAMS:
  aucs=[]
  for tr,va in folds:
   fitted=rbf.fit([x[tr]],y[tr],c,g);aucs.append(float(roc_auc_score(y[va],rbf.score(fitted,[x[va]]))))
  rows.append({'C':c,'gamma':g,'mean_fold_auc':float(np.mean(aucs)),'fold_aucs':aucs})
 return min(rows,key=lambda r:(-r['mean_fold_auc'],r['C'],0 if r['gamma']==.1 else 1))

def metrics(sg,sn,t):
 result=exp.metrics(sg,sn,t);y=np.r_[np.ones(len(sg)),np.zeros(len(sn))];s=np.r_[sg,sn]
 fpr,tpr,_=roc_curve(y,s);fnr=1-tpr;delta=fpr-fnr;i=int(np.flatnonzero(delta>=0)[0])
 if i==0:eer=float(fpr[0])
 else:
  weight=-delta[i-1]/(delta[i]-delta[i-1]);eer=float(fpr[i-1]+weight*(fpr[i]-fpr[i-1]))
 result['eer']=eer;return result

def word_probe(train,test,words_train,words_test):
 scaler=StandardScaler().fit(train);model=LogisticRegression(C=.1,max_iter=1500,solver='lbfgs').fit(scaler.transform(train),words_train)
 return float(accuracy_score(words_test,model.predict(scaler.transform(test))))

def main():
 OUT.mkdir(parents=True,exist_ok=True);pm=protocol_module();results={}
 for target in (1,4,5):
  print('TARGET_START',target,flush=True);p=pm.build_target_protocol(target,42,42)
  manifest=json.loads((FEATURE_ROOT/('U%d_split_manifest.json'%target)).read_text());assert p.manifest==manifest
  pm.save_manifest(p,str(OUT/('U%d_split_manifest.json'%target)))
  archive=np.load(str(FEATURE_ROOT/('U%d_features.npz'%target)))
  arrays={k:archive[k+'__base'] for k,v in p.arrays.items() if len(v)}
  assert all(x.shape==(len(p.arrays[k]),772) and np.isfinite(x).all() for k,x in arrays.items())
  uid=p.labels['auth_train'];y=uid==target-1
  words={k:np.asarray([int(str(s).split(':')[1])//10 for s in ids]) for k,ids in p.sample_ids.items() if len(ids)}
  folds=list(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42).split(np.zeros(len(y)),y,words['auth_train']))
  cv={'baseline':baseline_cv(arrays['auth_train'],y,words['auth_train'],folds)}
  for variant in VARIANTS:
   cache=OUT/('U%d_%s_cv.json'%(target,variant))
   if cache.exists():cv[variant]=json.loads(cache.read_text())
   else:
    cv[variant]=cv_variant(arrays['auth_train'],uid,words['auth_train'],target,variant,folds)
    cache.write_text(json.dumps(cv[variant],indent=2))
   print('VARIANT_LOCKED',target,variant,cv[variant]['selected'],flush=True)
  ranks={'baseline':cv['baseline']['mean_fold_auc']}
  ranks.update({v:cv[v]['selected']['mean_fold_auc'] for v in VARIANTS})
  selected=max(ranks,key=lambda k:(ranks[k],k=='baseline'))
  (OUT/('U%d_locked.json'%target)).write_text(json.dumps({'cv':cv,'selected':selected,'selection':'mean word-grouped fold AUC; projection and scaler refitted in every fitting fold; final labels not used'},indent=2))
  rows={};final_dump={}
  # All variants are locked before ANY final query score is evaluated.
  for variant in ('baseline',)+VARIANTS:
   per_seed=[];seeds=(42,) if variant=='baseline' else SEEDS
   for seed in seeds:
    if variant=='baseline':
     config=cv['baseline'];embeddings=arrays;projection=None;trace={}
    else:
     config=cv[variant]['selected'];scaler,head,_,trace=fit_projection(arrays['auth_train'],uid,words['auth_train'],target,variant,seed,config['epoch'])
     embeddings={k:embed(head,scaler,x) for k,x in arrays.items()}
     projection={'state':{k:v.detach().cpu() for k,v in head.state_dict().items()},'scaler':scaler,'variant':variant,'trace':trace}
    fitted=rbf.fit([embeddings['auth_train']],y,config['C'],config['gamma'])
    cal_g=rbf.score(fitted,[embeddings['target_dev_pool']]);cal_n=rbf.score(fitted,[embeddings['impostor_calibration']])
    threshold=float(exp.base.original.choose_threshold(cal_g,cal_n))
    joblib.dump({'classifier':fitted,'projection':projection,'threshold':threshold,'config':config},str(OUT/('U%d_%s_seed%d.joblib'%(target,variant,seed))))
    sg=rbf.score(fitted,[embeddings['final_genuine']]);sn=rbf.score(fitted,[embeddings['final_impostor']])
    row=metrics(sg,sn,threshold);row['seed']=seed;row['attack_by_user']={}
    for label in np.unique(p.labels['final_impostor']):
     s=sn[p.labels['final_impostor']==label];row['attack_by_user'][str(int(label)+1)]={'n':len(s),'accepted':int(np.sum(s>=threshold))}
    row['word_probe_acc']=word_probe(embeddings['auth_train'],np.concatenate([embeddings['final_genuine'],embeddings['final_impostor']]),words['auth_train'],np.r_[words['final_genuine'],words['final_impostor']])
    per_seed.append(row);final_dump[variant+'_genuine_'+str(seed)]=sg;final_dump[variant+'_impostor_'+str(seed)]=sn
   summary={k:{'mean':float(np.mean([r[k] for r in per_seed])),'std':float(np.std([r[k] for r in per_seed]))} for k in ('tar','far','auc','hter','eer','word_probe_acc')}
   rows[variant]={'config':config,'per_seed':per_seed,'summary':summary}
   print('FINAL_VARIANT_DONE',target,variant,summary,flush=True)
  results[str(target)]={'selected_on_training_cv':selected,'variants':rows,'counts':p.manifest['counts'],'training_seeds':list(SEEDS),
      'feature_source':'Identical complete 772-D frozen CycleGAN baseline; no teacher/diffusion features used in this isolated loss experiment',
      'method_status':'Compact and factorized adaptations; semi-hard is CHNS-inspired reweighting, not original CHNS sampling',
      'test_status':'Existing recordings and repeated final observations; exploratory, not a new independent confirmation set'}
  np.savez_compressed(str(OUT/('U%d_final_scores.npz'%target)),**final_dump)
  tmp=OUT/'results.tmp.json';tmp.write_text(json.dumps(results,indent=2));tmp.replace(OUT/'results.json')
  print('TARGET_DONE',target,'SELECTED',selected,flush=True)
 print('ALL_COMPLETE',flush=True)

if __name__=='__main__':main()
