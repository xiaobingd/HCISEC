"""Nine-user matched ACC / generated-Audio / both teacher score ablation."""
import sys,json
from pathlib import Path
import numpy as np
sys.path.insert(0,'/root')
import run_nine_sameword_balanced_20261030 as base
old=base.old;ROOT=Path('/autodl-fs/data/mei_nine_audio_supplement_20261030');AUDIO=['old_audio','identity_audio','robust_audio']

def dataset(u,M):
 data=base.dataset(u,M);p=old.protocol(u);dev=np.load(str(old.ROOT/('U%d'%u)/'development_features_private.npz'));final=np.load(str(base.same.FINAL/('U%d'%u)/'final_features_private.npz'));ids=[];parts={k:[] for k in AUDIO}
 for role in ['auth_train','model_validation','target_dev_pool','impostor_calibration','final_genuine','final_impostor']:
  ids+=p.sample_ids[role];src=final if role.startswith('final_') else dev
  for k in AUDIO:parts[k].append(src[role+'__'+k])
 assert len(ids)==len(set(ids));index={s:i for i,s in enumerate(ids)};parts={k:np.concatenate(v) for k,v in parts.items()}
 for role in base.same.REPEATS:
  rr=[str(v)+':'+str(j) for v in old.USERS for j in M['sources'][str(v)]['roles'][role]];ix=[index[s] for s in rr]
  for k,v in parts.items():data[role]['features'][k]=v[ix]
 return data

def candidates(oof,y,groups,folds,mode,anchor=None):
 accs=['none']+base.FAMILIES if mode in ['acc','both'] else ['none'];auds=['none']+AUDIO if mode in ['audio','both','fixed_acc_audio'] else ['none'];rows=[]
 for acc in accs:
  for aud in auds:
   for rw in (base.GRID if anchor is None else [anchor['raw_weight']]):
    for aw in ([0.] if acc=='none' else base.GRID[1:]):
     for mw in ([0.] if aud=='none' else base.GRID[1:]):
      c=dict(acc=acc,audio=aud,raw_weight=rw,acc_weight=aw,audio_weight=mw)
      if anchor is not None:c.update(acc=anchor['acc'],acc_weight=anchor['acc_weight'])
      s=oof['base']+rw*oof['raw']
      if c['acc_weight']:s=s+c['acc_weight']*oof[c['acc']]
      if mw:s=s+mw*oof[aud]
      c['quality']=base.fold_quality(s,y,groups,folds);rows.append(c)
 return rows

def rank(c):return base.quality_rank(c['quality'])+(c['raw_weight']+c['acc_weight']+c['audio_weight'],c['acc'],c['audio'],c['raw_weight'],c['acc_weight'],c['audio_weight'])

def score(models,data,c,role):
 s=None
 for k,w in [('base',1.),('raw',c['raw_weight']),(c['acc'],c['acc_weight']),(c['audio'],c['audio_weight'])]:
  if not w:continue
  m=models[k];v=(old.sel.rbf.score(m['model'],[data[role]['features'][k]])-m['mean'])/m['std'];s=w*v if s is None else s+w*v
 return s

def target(u,M):
 folder=ROOT/('U%d'%u);folder.mkdir(parents=True,exist_ok=True);data=dataset(u,M);source=base.ROOT/('U%d'%u);models=old.sel.joblib.load(str(source/'models_private.joblib'));existing=np.load(str(source/'oof_private.npz'));oof={k:existing[k] for k in existing.files}
 y=data['auth_train']['labels']==u-1;groups=data['auth_train']['repeats'];folds=list(old.sel.StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42).split(np.zeros(len(y)),y,groups))
 for k in AUDIO:
  oof[k],models[k]=base.nested_branch(data['auth_train']['features'][k],y,groups,folds);print('AUDIO_BRANCH_DONE',u,k,flush=True)
 searches={};selected={}
 for mode in ['acc','audio','both']:
  searches[mode]=candidates(oof,y,groups,folds,mode);selected[mode]=min(searches[mode],key=rank)
 searches['fixed_acc_audio']=candidates(oof,y,groups,folds,'fixed_acc_audio',selected['acc']);selected['fixed_acc_audio']=min(searches['fixed_acc_audio'],key=rank)
 assert [len(searches[k]) for k in ['acc','audio','both','fixed_acc_audio']]==[40,40,400,10]
 previous=json.loads((source/'results_public.json').read_text());a=selected['acc'];p=previous['chosen'];assert (a['acc'],a['raw_weight'],a['acc_weight'])==(p['teacher'],p['raw_weight'],p['teacher_weight'])
 selected['baseline772']=dict(acc='none',audio='none',raw_weight=0.,acc_weight=0.,audio_weight=0.)
 specs={}
 for mode,c in selected.items():
  spec={k:c[k] for k in ['acc','audio','raw_weight','acc_weight','audio_weight']};s=score(models,data,spec,'calibration');pos=data['calibration']['labels']==u-1;t=base.balanced_threshold(s[pos],s[~pos]);spec.update(threshold=t,calibration_tar=float((s[pos]>=t).mean()),calibration_far=float((s[~pos]>=t).mean()));specs[mode]=spec
 old.sel.joblib.dump(models,str(folder/'models_private.joblib'));np.savez_compressed(str(folder/'oof_private.npz'),**oof)
 old.save(folder/'locked_before_final.json',dict(routes=specs,selected=selected,searches=searches,final_scores_seen=False))
 labels=data['test']['labels'];words=data['test']['words'];pos=labels==u-1;rows={};dump={}
 for mode,c in specs.items():
  s=score(models,data,c,'test');g=s[pos];n=s[~pos];t=c['threshold'];r=old.sel.exp.metrics(g,n,t);r.update(c);r['attack_by_user']={str(v):dict(n=int((labels==v-1).sum()),accepted=int((s[labels==v-1]>=t).sum())) for v in old.USERS if v!=u};r['per_word']={}
  for w in base.same.GROUPS:
   mask=words==w;wp=pos[mask];assert wp.sum()==2 and (~wp).sum()==16;r['per_word'][base.same.WORDS[w]]=old.sel.exp.metrics(s[mask][wp],s[mask][~wp],t)
  r['same_word_mean_auc']=float(np.mean([a['auc'] for a in r['per_word'].values()]));assert len(g)==30 and len(n)==240 and len(r['attack_by_user'])==8
  assert abs(((g[:,None]>n)+.5*(g[:,None]==n)).mean()-r['auc'])<1e-12;rows[mode]=r;dump[mode]=s
 for k in ['tar','far','auc','hter']:assert abs(rows['acc'][k]-previous['routes']['balanced_selected'][k])<1e-12
 np.savez_compressed(str(folder/'scores_private.npz'),**dump,labels=labels,words=words);out=dict(target=u,routes=rows,selected={k:{j:v for j,v in c.items() if j!='quality'} for k,c in selected.items()},acc_control_replayed=True);old.save(folder/'results_public.json',out);print('TARGET_DONE',u,flush=True);return out

def main():
 ROOT.mkdir(exist_ok=True);M=base.same.manifest();old.save(ROOT/'manifest_private.json',M)
 old.save(ROOT/'protocol_public.json',dict(targets=list(range(1,10)),split='same15words train75/cal30/test30 peridentity;all8attackers',definition='Audio features E_M(G_ACCtoAudio(x));real authAudio never supplied;G/F internal coding already in772base',model='reuse base/raw/ACCteacher frozen RBF;new generatedAudio64branch RBF same nested balanced recipe',routes={'acc':40,'audio':40,'both':400,'fixed_acc_audio':10},both='independent ACC/Audio teacher family and weight choices;0allowed;rawoptional;full772always',fixed='lock original OOFselected ACC/raw and only add optional Audio',objective='joint equalHTER;no acceptancepriority;calibration bothpositive andnegative;allroutes lockbeforefinal',limitations=['historical final reused exploratory closedset','sameperiod mapping and sharedphase assumptions','400 vs40 search spaces differ;fixedACCcontrol helps attribution','reused OOF meta-selection not full independent nesting','no newGor teacher training','generatedAudio branch can encode nuisance ormodelstyle,not proof physiology']))
 rows={}
 for u in range(1,10):
  old.save(ROOT/'status_public.json',dict(state='running',target=u,completed=list(rows)));rows[str(u)]=target(u,M);old.save(ROOT/'results_public.json',rows)
 old.save(ROOT/'status_public.json',dict(state='complete',completed=list(rows)))
if __name__=='__main__':
 try:main()
 except Exception as e:
  ROOT.mkdir(exist_ok=True);old.save(ROOT/'status_public.json',dict(state='failed',error=repr(e)));raise

