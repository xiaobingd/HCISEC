"""Same-word U4 counterfactuals on cached authenticated branches."""
import sys,json
from pathlib import Path
import numpy as np
sys.path.insert(0,'/root')
import run_nine_sameword_balanced_20261030 as b
old=b.old;ROOT=Path('/autodl-fs/data/u4_sameword_raw_increment_20261030')

def main():
 ROOT.mkdir(exist_ok=True);M=b.same.manifest();data=b.dataset(4,M);source=b.ROOT/'U4';models=old.sel.joblib.load(str(source/'models_private.joblib'));prior=json.loads((source/'results_public.json').read_text());names={'baseline772':{'base':1.},'raw_only':{'raw':1.},'baseline_raw_fixed1':{'base':1.,'raw':1.},'baseline_teacher':{'base':1.,'old_acc':1.},'baseline_teacher_raw_fixed1':{'base':1.,'old_acc':1.,'raw':1.}};routes={};scores={}
 oof=np.load(str(source/'oof_private.npz'));y=data['auth_train']['labels']==3;groups=data['auth_train']['repeats'];folds=list(old.sel.StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42).split(np.zeros(len(y)),y,groups));selection={}
 for key,start in [('baseline_raw_oof',{'base':1.}),('baseline_teacher_raw_oof',{'base':1.,'old_acc':1.})]:
  candidates=[]
  for weight in [0.,.25,.5,1.]:
   spec=dict(start);spec['raw']=weight
   q=b.fold_quality(sum(v*oof[k] for k,v in spec.items() if v),y,groups,folds);c=dict(raw_weight=weight,teacher='old_acc' if 'old_acc' in start else 'none',teacher_weight=start.get('old_acc',0.),quality=q);candidates.append(c)
  best=min(candidates,key=b.route_rank);selection[key]=dict(selected_weight=best['raw_weight'],quality=best['quality'],candidates=candidates);names[key]={**start,**({'raw':best['raw_weight']} if best['raw_weight'] else {})}
 for name,weights in names.items():
  cal=b.score(models,data,dict(teacher='none',raw_weight=weights.get('raw',0.),teacher_weight=0.,**({'base_weight':weights['base']} if 'base' in weights else {})), 'calibration') if len(weights)==1 else sum(w*(old.sel.rbf.score(models[k]['model'],[data['calibration']['features'][k]])-models[k]['mean'])/models[k]['std'] for k,w in weights.items())
  labels=data['calibration']['labels'];pos=labels==3;t=b.balanced_threshold(cal[pos],cal[~pos]);test=sum(w*(old.sel.rbf.score(models[k]['model'],[data['test']['features'][k]])-models[k]['mean'])/models[k]['std'] for k,w in weights.items());tl=data['test']['labels'];g=test[tl==3];n=test[tl!=3];r=old.sel.exp.metrics(g,n,t);r.update(weights=weights,threshold=t);r['attack_by_user']={str(v+1):dict(n=int((tl==v).sum()),accepted=int((test[tl==v]>=t).sum())) for v in range(9) if v!=3};routes[name]=r;scores[name+'__test']=test
  assert len(g)==30 and len(n)==240 and abs(((g[:,None]>n)+.5*(g[:,None]==n)).mean()-r['auc'])<1e-12
  for w in np.unique(data['test']['words']):
   mask=data['test']['words']==w;assert (tl[mask]==3).sum()==2 and (tl[mask]!=3).sum()==16
 # Exact replay of the original baseline and its selected full combination.
 for key,prior_key in [('baseline772','baseline772'),('baseline_teacher_raw_fixed1','balanced_selected')]:
  oldrow=prior['routes'][prior_key];newrow=routes[key]
  for m in ['tar','far','auc','hter']:assert abs(oldrow[m]-newrow[m])<1e-12,(key,m,oldrow[m],newrow[m])
 np.savez_compressed(str(ROOT/'scores_private.npz'),**scores);old.save(ROOT/'results_public.json',dict(target=4,routes=routes,selection=selection,counts=dict(train_positive=75,train_negative=600,cal_positive=30,cal_negative=240,test_positive=30,test_negative=240),selection_rule='training-only nested same-word OOF search over raw weight 0/.25/.5/1; fixed-weight1 counterfactuals are also shown; no final selection; independent balance thresholds on calibration',scope='known nine identities, same 15 words, U4, historical final previously observed exploratory replay'));old.save(ROOT/'verification_public.json',dict(status='PASS',checks=['reuse U4 same-word fitted branch models','recomputed independent balanced thresholds','recomputed all route scores AUC and 8 attacker counts','exact replay prior baseline and selected fused route','raw weight independently selected from training-only OOF']));print(json.dumps(routes),flush=True)
if __name__=='__main__':main()

