"""Controlled raw-input ACC increment on frozen full772 RBF, strict cross-word U4."""
import sys,json
from pathlib import Path
import numpy as np
sys.path.insert(0,'/root')
import run_strict_autosupplement_20261029 as s
import run_nine_sameword_balanced_20261030 as b
ROOT=Path('/autodl-fs/data/mei_u4_baseline_raw_increment_20261030')

def prepare():
 p=s.protocol(4);source=s.ROOT/'U4';dev=np.load(str(source/'development_features_private.npz'));final=np.load(str(b.same.FINAL/'U4/final_features_private.npz'));models=s.sel.joblib.load(str(source/'branch_classifiers.joblib'));oof=np.load(str(source/'oof_private.npz'));lock=json.loads((source/'locked_before_final.json').read_text());anchor={k:v for k,v in lock['weights']['auto_without_diffusion'].items() if k!='raw'}
 assert anchor.get('base')==1 and not any(k.startswith('diff') for k in anchor)
 # Explicitly check the raw branch cache and full baseline dimensions against actual protocol input.
 for role in ['auth_train','target_dev_pool','impostor_calibration','final_genuine','final_impostor']:
  src=final if role.startswith('final') else dev;x=p.arrays[role];raw=np.concatenate([x.mean(2),x.std(2),x.mean(1),x.std(1)],1)
  assert np.allclose(src[role+'__raw'],raw,atol=2e-6) and src[role+'__base'].shape==(len(x),772)
 return p,source,dev,final,models,oof,anchor

def scoring(models,src,role,weights):
 return sum(w*(s.sel.rbf.score(models[k]['model'],[src[role+'__'+k]])-models[k]['mean'])/models[k]['std'] for k,w in weights.items())

def main():
 ROOT.mkdir(exist_ok=True);p,source,dev,final,models,oof,anchor=prepare();y=p.labels['auth_train']==3;folds=list(s.sel.StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42).split(np.zeros(len(y)),y,p.words))
 for tr,va in folds:assert not set(p.words[tr])&set(p.words[va])
 weights={'baseline772':{'base':1.},'raw_only':{'raw':1.},'baseline_teacher':anchor,'baseline_raw_fixed1':{'base':1.,'raw':1.}};selection={}
 for key,start in [('baseline_raw_optional',{'base':1.}),('baseline_teacher_raw_optional',anchor)]:
  weights[key],selection[key]=s.sel.forward_select(oof,y,folds,['raw'],start,1)
 thresholds={}
 for key,ww in weights.items():
  g=scoring(models,dev,'target_dev_pool',ww);n=scoring(models,dev,'impostor_calibration',ww);thresholds[key]=b.balanced_threshold(g,n)
 s.save(ROOT/'locked_before_final.json',dict(weights=weights,thresholds=thresholds,selection=selection,source=str(source),teacher_anchor='prior training-OOF selected nonDiff route, removing raw; no reselection',calibration='joint minHTER with max error tie; differs from old acceptance-priority threshold',final_scores_seen=False,limitation='historical final already seen in prior experiments; exploratory replay'))
 rows={};scores={}
 for key,ww in weights.items():
  g=scoring(models,final,'final_genuine',ww);n=scoring(models,final,'final_impostor',ww);t=thresholds[key];r=s.sel.exp.metrics(g,n,t);r.update(weights=ww,threshold=t);r['attack_by_user']={str(v):dict(n=int((p.labels['final_impostor']==v-1).sum()),accepted=int((n[p.labels['final_impostor']==v-1]>=t).sum())) for v in range(1,10) if v!=4};rows[key]=r;scores[key+'__genuine']=g;scores[key+'__impostor']=n
  assert abs(((g[:,None]>n)+.5*(g[:,None]==n)).mean()-r['auc'])<1e-12
 np.savez_compressed(str(ROOT/'scores_private.npz'),**scores)
 s.save(ROOT/'results_public.json',dict(target=4,routes=rows,selection=selection,counts=dict(train_positive=50,train_negative=400,calibration_positive=20,calibration_negative=160,test_positive=47,test_negative=394),feature_definition='full772 unchanged; raw320=time/frequency mean/std of original input stored ACC spectrum; not unnormalized physical energy',scope='strict inferred crossword known9users; frozen prior personalG/F and teachers, cached RBF; only score weights/calibration replay'))
 print(json.dumps(rows),flush=True)
if __name__=='__main__':main()
