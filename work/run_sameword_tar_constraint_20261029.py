"""Content matched repeats; TAR-constrained optional personal teacher score fusion."""
import sys,json
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,'/root')
import run_sameword_teacher_baseline_20261029 as same
import run_tar_constrained_selection_20261029 as rule
old=same.old;prior=rule.prior;ROOT=Path('/autodl-fs/data/mei_sameword_tar_constraint_20261029')

def dataset(u,M):
 p=old.protocol(u);src=np.load(str(old.ROOT/('U%d'%u)/'development_features_private.npz'));final=np.load(str(same.FINAL/('U%d'%u)/'final_features_private.npz'))
 ids=[];base=[]
 for role in ['auth_train','model_validation','target_dev_pool','impostor_calibration','final_genuine','final_impostor']:
  ids+=p.sample_ids[role];f=final if role.startswith('final_') else src;base.append(f[role+'__base'])
 assert len(ids)==len(set(ids));lookup={sid:i for i,sid in enumerate(ids)};base=np.concatenate(base);acc={v:np.load(str(old.DATA/('acc.%d.npy'%v))).astype('float32')[:,:80,:80] for v in old.USERS};out={}
 for role in same.REPEATS:
  records=[(v,j) for v in old.USERS for j in M['sources'][str(v)]['roles'][role]]
  out[role]=dict(a=np.stack([acc[v][j] for v,j in records]),base=base[[lookup[str(v)+':'+str(j)] for v,j in records]],labels=np.asarray([v-1 for v,j in records]),words=np.asarray([M['sources'][str(v)]['groups'][j] for v,j in records]),repeats=np.asarray([M['sources'][str(v)]['repeats'][j] for v,j in records]))
 return out

def stats(x):return np.concatenate([x.mean(2),x.std(2),x.mean(1),x.std(1)],1)
def standardized(m,x):return (old.sel.rbf.score(m['model'],[x])-m['mean'])/m['std']

def target(u,M):
 folder=ROOT/('U%d'%u);folder.mkdir(parents=True,exist_ok=True);data=dataset(u,M);source=same.ROOT/('U%d'%u);personal=prior.ROOT/('U%d'%u)
 y=data['auth_train']['labels']==u-1;reps=data['auth_train']['repeats'];folds=list(old.sel.StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42).split(np.zeros(len(y)),y,reps))
 for tr,va in folds:assert not set(reps[tr])&set(reps[va]) and set(data['auth_train']['words'][tr])==set(data['auth_train']['words'][va])==set(same.GROUPS)
 existing=old.sel.joblib.load(str(source/'models_private.joblib'));existing_oof=np.load(str(source/'oof_private.npz'));models={'base':existing['baseline772']};oof={'base':existing_oof['baseline772']};teachers={}
 oof['raw'],models['raw']=old.sel.nested_branch(stats(data['auth_train']['a']),y,reps,folds)
 for key in prior.CONFIGS:
  ck=torch.load(str(personal/(key+'_teacher_private.pt')),map_location=prior.D,weights_only=False);t=old.base.Teacher().to(prior.D);t.load_state_dict(ck['state']);t.eval();teachers[key]=t
  f=prior.embed(t,data['auth_train']['a']);oof[key],models[key]=old.sel.nested_branch(f,y,reps,folds);print('BRANCH_DONE',u,key,flush=True)
 candidates=[]
 for key in ['none']+prior.CONFIGS:
  for rw in rule.GRID:
   for tw in ([0.] if key=='none' else rule.GRID[1:]):
    s=oof['base']+rw*oof['raw']+(tw*oof[key] if tw else 0.);candidates.append(dict(teacher_key=key,raw_weight=rw,teacher_weight=tw,quality=rule.oof_quality(s,y,reps,folds)))
 chosen=min(candidates,key=rule.rank);assert len(candidates)==40
 def score(spec,role):
  d=data[role];s=standardized(models['base'],d['base'])
  if spec['raw_weight']:s+=spec['raw_weight']*standardized(models['raw'],stats(d['a']))
  if spec['teacher_weight']:s+=spec['teacher_weight']*standardized(models[spec['teacher_key']],prior.embed(teachers[spec['teacher_key']],d['a']))
  return s
 bw=dict(teacher_key='none',raw_weight=0.,teacher_weight=0.);pw=json.loads((personal/'locked_before_final.json').read_text())['routes']['control']['weights'];cw=dict(teacher_key='control',raw_weight=pw.get('raw',0.),teacher_weight=pw.get('teacher',0.))
 specs={}
 for name,spec in [('baseline_original',bw),('control_original',cw),('baseline_tar90',bw),('control_tar90',cw),('selected_tar90',{k:chosen[k] for k in ['teacher_key','raw_weight','teacher_weight']})]:
  s=score(spec,'calibration');mask=data['calibration']['labels']==u-1;g=s[mask];n=s[~mask]
  t=rule.threshold(g) if name.endswith('tar90') else float(old.base.original.choose_threshold(g,n))
  specs[name]=dict(spec,threshold=t,calibration_tar=float((g>=t).mean()),calibration_far=float((n>=t).mean()))
 old.sel.joblib.dump(models,str(folder/'models_private.joblib'));np.savez_compressed(str(folder/'oof_private.npz'),**oof)
 old.save(folder/'locked_before_final.json',dict(routes=specs,chosen=chosen,candidates=candidates,final_scores_seen=False,repeat_groups=sorted(np.unique(reps).tolist())))
 labels=data['test']['labels'];words=data['test']['words'];pos=labels==u-1;rows={};dump={}
 for name,spec in specs.items():
  s=score(spec,'test');g=s[pos];n=s[~pos];t=spec['threshold'];r=old.sel.exp.metrics(g,n,t);r.update(spec);r['attack_by_user']={str(v):dict(n=int((labels==v-1).sum()),accepted=int((s[labels==v-1]>=t).sum())) for v in old.USERS if v!=u};r['per_word']={}
  for w in same.GROUPS:
   mask=words==w;mp=pos[mask];assert mp.sum()==2 and (~mp).sum()==16;r['per_word'][same.WORDS[w]]=old.sel.exp.metrics(s[mask][mp],s[mask][~mp],t)
  r['same_word_mean_auc']=float(np.mean([a['auc'] for a in r['per_word'].values()]));assert len(g)==30 and len(n)==240
  assert abs(((g[:,None]>n)+.5*(g[:,None]==n)).mean()-r['auc'])<1e-12;assert len(r['attack_by_user'])==8
  rows[name]=r;dump[name]=s
 reference=json.loads((source/'results_public.json').read_text())['routes']
 for k in ['tar','far','auc']:assert abs(rows['baseline_original'][k]-reference['baseline772'][k])<1e-12
 np.savez_compressed(str(folder/'scores_private.npz'),**dump,labels=labels,words=words)
 out=dict(target=u,routes=rows,chosen=chosen,old_sameword_reference={k:reference[k] for k in ['baseline772','train_selected_teacher']},baseline_replayed=True,counts=dict(train_positive=75,train_negative=600,cal_positive=30,cal_negative=240,test_positive=30,test_negative=240))
 old.save(folder/'results_public.json',out);print('COMPLETE',u,flush=True);return out

def main():
 ROOT.mkdir(exist_ok=True);M=same.manifest();old.save(ROOT/'manifest_private.json',M)
 old.save(ROOT/'protocol_public.json',dict(targets=[4,5],auth_words=[same.WORDS[w] for w in same.GROUPS],repeat_roles=same.REPEATS,counts_per_user=dict(train=75,calibration=30,test=30),model='frozen original772 baseline,originalACC statistics,new previous personalACCteachers;no teacher orG retrain',selection='40fusions,training OOF grouped by repetition not word;each fold same15words across allidentities;TAR90priority thenFAR;quantile thresholdcal30',all8attackers=True,limitations=['word labels conditional on10/20period and commonphase;not independent transcript','different samplecounts/vocabulary from crosscontent experiment;not onefactor causal comparison','historical features reused including already observed tests;exploratory closedset','reused OOF meta-validation not fully nested selection','personal teacher settings were explored under previous crosscontent final;not independent of experiment history','900D previous teacher concatenation differs from new ACCteacher score supplement','cal90TAR does not guarantee test90TAR']))
 rows={}
 for u in [4,5]:rows[str(u)]=target(u,M);old.save(ROOT/'results_public.json',rows)
 old.save(ROOT/'COMPLETE.json',dict(status='PASS'))
if __name__=='__main__':main()

