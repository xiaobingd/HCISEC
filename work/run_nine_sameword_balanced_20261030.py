"""Nine-user content-matched authentication; joint FAR/FRR optimization, no TAR constraint."""
import sys,json
from pathlib import Path
import numpy as np
sys.path.insert(0,'/root')
import run_sameword_teacher_baseline_20261029 as same
old=same.old;ROOT=Path('/autodl-fs/data/mei_nine_sameword_balanced_20261030');GRID=[0.,.25,.5,1.];FAMILIES=['old_acc','identity_acc','robust_acc']

def balanced_threshold(g,n):
 g=np.sort(np.asarray(g));n=np.sort(np.asarray(n));values=np.unique(np.r_[g,n]);thresholds=np.r_[values,np.nextafter(values[-1],np.inf)]
 frr=np.searchsorted(g,thresholds,side='left')/len(g);far=1-np.searchsorted(n,thresholds,side='left')/len(n);hter=(far+frr)/2
 k=min(range(len(thresholds)),key=lambda i:(round(float(hter[i]),12),round(float(max(far[i],frr[i])),12),-float(thresholds[i])))
 return float(thresholds[k])

def fold_quality(scores,y,groups,folds):
 accepted=np.zeros(len(y),dtype=bool);rows=[]
 for tr,va in folds:
  assert not set(groups[tr])&set(groups[va]);t=balanced_threshold(scores[tr][y[tr]],scores[tr][~y[tr]]);accepted[va]=scores[va]>=t
  rows.append(dict(threshold=t,tar=float(accepted[va][y[va]].mean()),far=float(accepted[va][~y[va]].mean())))
 tar=float(accepted[y].mean());far=float(accepted[~y].mean());return dict(tar=tar,far=far,hter=(1-tar+far)/2,worst=max(1-tar,far),auc=float(old.sel.auc(y,scores)),folds=rows)

def quality_rank(q):return (round(q['hter'],12),round(q['worst'],12),-q['auc'])
def route_rank(c):return quality_rank(c['quality'])+(c['raw_weight']+c['teacher_weight'],c['teacher'],c['raw_weight'],c['teacher_weight'])

def choose_params(x,y,groups,seed):
 folds=list(old.sel.StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=seed).split(np.zeros(len(y)),y,groups));best=None
 for c in [1.,10.,100.]:
  for gamma in ['scale',.1]:
   pred=np.full(len(y),np.nan)
   for tr,va in folds:
    fitted=old.sel.rbf.fit([x[tr]],y[tr],c,gamma);s=old.sel.rbf.score(fitted,[x[tr]]);pred[va]=(old.sel.rbf.score(fitted,[x[va]])-s.mean())/(s.std()+1e-6)
   q=fold_quality(pred,y,groups,folds);rank=quality_rank(q)+(c,0 if gamma==.1 else 1)
   if best is None or rank<best[0]:best=(rank,c,gamma,q)
 return best[1:]

def nested_branch(x,y,groups,folds):
 oof=np.full(len(y),np.nan);configs=[]
 for i,(tr,va) in enumerate(folds):
  c,gamma,q=choose_params(x[tr],y[tr],groups[tr],71+i);fitted=old.sel.rbf.fit([x[tr]],y[tr],c,gamma);s=old.sel.rbf.score(fitted,[x[tr]]);oof[va]=(old.sel.rbf.score(fitted,[x[va]])-s.mean())/(s.std()+1e-6);configs.append(dict(C=c,gamma=gamma,selection=q))
 assert np.isfinite(oof).all();c,gamma,q=choose_params(x,y,groups,71);fitted=old.sel.rbf.fit([x],y,c,gamma);s=old.sel.rbf.score(fitted,[x]);return oof,dict(model=fitted,mean=float(s.mean()),std=float(s.std()+1e-6),C=c,gamma=gamma,selection=q,outer_configs=configs)

def dataset(u,M):
 p=old.protocol(u);dev=np.load(str(old.ROOT/('U%d'%u)/'development_features_private.npz'));final=np.load(str(same.FINAL/('U%d'%u)/'final_features_private.npz'));names=['base']+FAMILIES;ids=[];blocks={k:[] for k in names}
 for role in ['auth_train','model_validation','target_dev_pool','impostor_calibration','final_genuine','final_impostor']:
  ids+=p.sample_ids[role];f=final if role.startswith('final_') else dev
  for k in names:blocks[k].append(f[role+'__'+k])
 assert len(ids)==len(set(ids));lookup={s:i for i,s in enumerate(ids)};blocks={k:np.concatenate(v) for k,v in blocks.items()};acc={v:np.load(str(old.DATA/('acc.%d.npy'%v))).astype('float32')[:,:80,:80] for v in old.USERS};out={}
 for role in same.REPEATS:
  rr=[(v,j) for v in old.USERS for j in M['sources'][str(v)]['roles'][role]];ix=[lookup[str(v)+':'+str(j)] for v,j in rr];x=np.stack([acc[v][j] for v,j in rr]);f={k:z[ix] for k,z in blocks.items()};f['raw']=np.concatenate([x.mean(2),x.std(2),x.mean(1),x.std(1)],1)
  out[role]=dict(features=f,labels=np.asarray([v-1 for v,j in rr]),words=np.asarray([M['sources'][str(v)]['groups'][j] for v,j in rr]),repeats=np.asarray([M['sources'][str(v)]['repeats'][j] for v,j in rr]))
 return out

def score(models,data,spec,role):
 s=None
 for name,w in [('base',spec.get('base_weight',1.)),('raw',spec['raw_weight']),(spec['teacher'],spec['teacher_weight'])]:
  if not w:continue
  m=models[name];z=(old.sel.rbf.score(m['model'],[data[role]['features'][name]])-m['mean'])/m['std'];s=w*z if s is None else s+w*z
 return s

def target(u,M):
 folder=ROOT/('U%d'%u);folder.mkdir(parents=True,exist_ok=True);data=dataset(u,M);y=data['auth_train']['labels']==u-1;groups=data['auth_train']['repeats'];folds=list(old.sel.StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42).split(np.zeros(len(y)),y,groups));oof={};models={}
 for tr,va in folds:assert set(data['auth_train']['words'][tr])==set(data['auth_train']['words'][va])==set(same.GROUPS)
 for k,x in data['auth_train']['features'].items():
  oof[k],models[k]=nested_branch(x,y,groups,folds);print('BRANCH_DONE',u,k,flush=True)
 candidates=[]
 for teacher in ['none']+FAMILIES:
  for rw in GRID:
   for tw in ([0.] if teacher=='none' else GRID[1:]):
    s=oof['base']+rw*oof['raw']+(tw*oof[teacher] if tw else 0.);candidates.append(dict(teacher=teacher,raw_weight=rw,teacher_weight=tw,quality=fold_quality(s,y,groups,folds)))
 chosen=min(candidates,key=route_rank);assert len(candidates)==40
 specs={'baseline772':dict(teacher='none',raw_weight=0.,teacher_weight=0.),'raw_acc_only':dict(base_weight=0.,teacher='none',raw_weight=1.,teacher_weight=0.),'balanced_selected':{k:chosen[k] for k in ['teacher','raw_weight','teacher_weight']}}
 for key,spec in specs.items():
  s=score(models,data,spec,'calibration');pos=data['calibration']['labels']==u-1;t=balanced_threshold(s[pos],s[~pos]);spec.update(threshold=t,calibration_tar=float((s[pos]>=t).mean()),calibration_far=float((s[~pos]>=t).mean()))
 old.sel.joblib.dump(models,str(folder/'models_private.joblib'));np.savez_compressed(str(folder/'oof_private.npz'),**oof)
 old.save(folder/'locked_before_final.json',dict(target=u,routes=specs,candidates=candidates,chosen=chosen,final_scores_seen=False))
 rows={};dump={};labels=data['test']['labels'];words=data['test']['words'];pos=labels==u-1
 for key,spec in specs.items():
  s=score(models,data,spec,'test');g=s[pos];n=s[~pos];t=spec['threshold'];r=old.sel.exp.metrics(g,n,t);r.update(spec);r['attack_by_user']={str(v):dict(n=int((labels==v-1).sum()),accepted=int((s[labels==v-1]>=t).sum())) for v in old.USERS if v!=u};r['per_word']={}
  for w in same.GROUPS:
   mask=words==w;wp=pos[mask];assert wp.sum()==2 and (~wp).sum()==16;r['per_word'][same.WORDS[w]]=old.sel.exp.metrics(s[mask][wp],s[mask][~wp],t)
  r['same_word_mean_auc']=float(np.mean([v['auc'] for v in r['per_word'].values()]));assert len(g)==30 and len(n)==240 and len(r['attack_by_user'])==8
  assert abs(((g[:,None]>n)+.5*(g[:,None]==n)).mean()-r['auc'])<1e-12;rows[key]=r;dump[key]=s
 np.savez_compressed(str(folder/'scores_private.npz'),**dump,labels=labels,words=words);out=dict(target=u,routes=rows,chosen=chosen,counts=dict(train_positive=75,train_negative=600,cal_positive=30,cal_negative=240,test_positive=30,test_negative=240));old.save(folder/'results_public.json',out);print('TARGET_DONE',u,flush=True);return out

def main():
 ROOT.mkdir(exist_ok=True);M=same.manifest();M['targets']=list(range(1,10));old.save(ROOT/'manifest_private.json',M)
 old.save(ROOT/'protocol_public.json',dict(targets=M['targets'],auth_words=M['auth_words'],repeat_roles=same.REPEATS,counts_per_identity=dict(train=75,calibration=30,test=30),objective='equal-weightHTER=(FAR+FRR)/2;tie minimize max(FAR,FRR),then largerAUC;no TAR constraint',branch_models='RBF base772/rawACCstats320/oldACCteacher64/identityACCteacher64/robustACCteacher64;same existing frozen teachers allnine;no G/F/teacher training',selection='branch hyperparameters3inner repeatOOF balancedHTER;5outerOOF for40fusion search;base772 retained,optionalraw andone teacher,zeroallowed;calibration thresholdminHTER using bothgenuine andattacks',all8attackers=True,limitations=['historical final reused exploratory knownusers','wordlabels10/20period andphase conditional not independent transcript','nested branch fits but reused OOF meta-selection and internal threshold estimates not fully unbiased','40candidate selection may overfit','sameword vs crossword vocab/samplecounts differ','allnine existing teacherfamilies differ from last U4/U5 newpersonalparameter family pool','ACCteacher gain not proof generatedAudioidentity contribution']))
 rows={}
 for u in M['targets']:
  old.save(ROOT/'status_public.json',dict(state='running',target=u,completed=list(rows)));rows[str(u)]=target(u,M);old.save(ROOT/'results_public.json',rows)
 old.save(ROOT/'status_public.json',dict(state='complete',completed=list(rows)))
if __name__=='__main__':
 try:main()
 except Exception as e:
  ROOT.mkdir(exist_ok=True);old.save(ROOT/'status_public.json',dict(state='failed',error=repr(e)));raise

