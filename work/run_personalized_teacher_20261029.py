"""Personal teacher adaptation; upstream-only fit; intact772 + optional ACC teacher."""
import sys,copy,json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
sys.path.insert(0,'/root')
import run_frozen_cycle_identity_compare_20261029 as first
old=first.old;trial=first.trial;D=old.D
ROOT=Path('/autodl-fs/data/mei_personalized_teacher_20261029');AUTH80=Path('/autodl-fs/data/mei_auth80_20261029')
CONFIGS=['control','personal_mild','personal_strong'];STEPS=600

def settings(u,key):
 cfg=dict(lr=.0005,wd=.003,crossword=False,contrast=.1,negative_weight=1.,patience=0)
 if key!='control':
  strong=key=='personal_strong'
  if u==4:cfg['negative_weight']=2. if strong else 1.5
  else:cfg.update(lr=.0001 if strong else .0003,wd=.01 if strong else .005,crossword=True,contrast=.3 if strong else .2,patience=4)
 return cfg

def upstream():
 out={k:[] for k in ['a','m','uid','word','fit']}
 for v in old.USERS:
  src=old.M['sources'][str(v)];ix=src['roles']['gan_fit']+src['roles']['gan_val']
  for domain,k in [('acc','a'),('audio','m')]:out[k].append(np.load(str(old.DATA/('%s.%d.npy'%(domain,v)))).astype('float32')[ix,:80,:80])
  out['uid'].append(np.full(150,v-1));out['word'].append(np.asarray(src['content_groups'])[ix]);out['fit'].append(np.arange(150)<120)
 return {k:np.concatenate(v) for k,v in out.items()}

@torch.no_grad()
def embed(t,x,domain='acc'):
 return np.concatenate([getattr(t,domain)(old.te.tensor(x[i:i+48])).cpu().numpy() for i in range(0,len(x),48)])

def train_teacher(u,key,dev,folder):
 cfg=settings(u,key);trial.seed(3800+u);rng=np.random.RandomState(3800+u)
 t=old.base.Teacher().to(D);identity=nn.Linear(64,9).to(D);params=list(t.parameters())+list(identity.parameters());opt=torch.optim.AdamW(params,lr=cfg['lr'],weight_decay=cfg['wd'])
 tr=np.flatnonzero(dev['fit']);va=np.flatnonzero(~dev['fit']);common=np.unique(dev['word'][tr]);lookup={(v,w):tr[(dev['uid'][tr]==v)&(dev['word'][tr]==w)] for v in range(9) for w in common};best=None;history=[];stale=0
 for step in range(1,STEPS+1):
  t.train();ww=rng.choice(common,8,replace=False);ix=np.asarray([rng.choice(lookup[v,w]) for v in range(9) for w in ww]);a=old.te.tensor(dev['a'][ix]);m=old.te.tensor(dev['m'][ix])
  za=t.acc(a+.003*torch.randn_like(a));zm=t.audio(m+.003*torch.randn_like(m));z=torch.cat([za,zm]);labels=torch.as_tensor(np.tile(dev['uid'][ix],2),device=D);words=torch.as_tensor(np.tile(dev['word'][ix],2),device=D);binary=(labels==u-1).float();logits=t.binary(z).flatten()
  losses=F.binary_cross_entropy_with_logits(logits,binary,pos_weight=trial.tensor(8.),reduction='none');weight=torch.ones_like(losses)
  if cfg['negative_weight']>1:
   # Moderately high scoring training negatives only, excluding top10% extremes.
   mask=binary==0;s=logits.detach()[mask];lo=torch.quantile(s,.5);hi=torch.quantile(s,.9);semi=mask&(logits.detach()>=lo)&(logits.detach()<=hi);weight[semi]=cfg['negative_weight'];weight[mask]/=weight[mask].mean()
  bce=(losses*weight).mean();pair=(1-(za*zm).sum(1)).mean();con=trial.contrast(z,labels,words,True) if cfg['crossword'] else old.te.supcon(z,labels)
  loss=bce+.3*F.cross_entropy(identity(z),labels)+cfg['contrast']*con+.1*pair
  opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(params,1.);opt.step()
  if step%25==0:
   t.eval();qualities=[];pooled=[];vl=[]
   for domain,k in [('acc','a'),('audio','m')]:
    zz=embed(t,dev[k][va],domain)
    with torch.no_grad():s=t.binary(trial.tensor(zz)).flatten().cpu().numpy()
    q=trial.quality(s,dev['uid'][va]==u-1,dev['word'][va]);qualities.append(q[0]);pooled.append(q[1]);vl.append(float(F.binary_cross_entropy_with_logits(trial.tensor(s),trial.tensor((dev['uid'][va]==u-1).astype('float32')),pos_weight=trial.tensor(8.))))
   row=dict(step=step,word_auc=float(np.mean(qualities)),auc=float(np.mean(pooled)),bce=float(np.mean(vl)),training_loss=float(loss.detach()));history.append(row);rank=(-row['word_auc'],-row['auc'],row['bce'],step)
   if best is None or rank<best[0]:best=(rank,step,copy.deepcopy(t.state_dict()));stale=0
   else:stale+=1
   if cfg['patience'] and stale>=cfg['patience']:break
 t.load_state_dict(best[2]);t.eval()
 for par in t.parameters():par.requires_grad_(False)
 meta=dict(config=cfg,selected_step=best[1],actual_steps=step,history=history,seed=3800+u,teacher_fit=1080,teacher_val=270)
 torch.save(dict(state=best[2],metadata=meta),str(folder/(key+'_teacher_private.pt')));old.save(folder/(key+'_teacher_public.json'),meta);print('TEACHER_DONE',u,key,best[1],step,flush=True);return t,meta

def target(u,dev):
 folder=ROOT/('U%d'%u);folder.mkdir(parents=True,exist_ok=True);p=old.protocol(u);src=AUTH80/('U%d'%u)
 fitroles=['auth_train','model_validation'];xx=np.concatenate([p.arrays[r] for r in fitroles]);labs=np.concatenate([p.labels[r] for r in fitroles]);y=labs==u-1;words=np.concatenate([first.words(p,r) for r in fitroles]);assert len(y)==720 and y.sum()==80
 upstream_ids={str(v)+':'+str(i) for v in old.USERS for role in ['gan_fit','gan_val'] for i in old.M['sources'][str(v)]['roles'][role]}
 auth_ids=sum([p.sample_ids[r] for r in first.ROLES],[]);assert not upstream_ids&set(auth_ids)
 ganpath=old.UP/('U%d'%u)/'cyclegan/models/checkpoint_epoch_200.pth';gansha=old.sha(ganpath)
 cached=old.sel.joblib.load(str(src/'models_private.joblib'));oofcache=np.load(str(src/'oof_private.npz'));folds=list(old.sel.StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42).split(np.zeros(len(y)),y,words))
 oldfeatures=np.load(str(old.ROOT/('U%d'%u)/'development_features_private.npz'))
 teachers={};models={};locks={};metas={};all_oof={}
 def standardized(m,x):return (old.sel.rbf.score(m['model'],[x])-m['mean'])/m['std']
 for key in CONFIGS:
  t,meta=train_teacher(u,key,dev,folder);teachers[key]=t;metas[key]=meta
  features=embed(t,xx);oof,model=old.sel.nested_branch(features,y,words,folds);models[key]=model;all_oof[key]=oof
  weights,selection=old.sel.forward_select({'base':oofcache['base'],'raw':oofcache['raw'],'teacher':oof},y,folds,['raw','teacher'],{'base':1.},2)
  def score_cal(role):
   result=standardized(cached['base'],oldfeatures[role+'__base'])
   if 'raw' in weights:result=result+weights['raw']*standardized(cached['raw'],oldfeatures[role+'__raw'])
   if 'teacher' in weights:result=result+weights['teacher']*standardized(model,embed(t,p.arrays[role]))
   return result
  threshold=float(old.base.original.choose_threshold(score_cal('target_dev_pool'),score_cal('impostor_calibration')))
  locks[key]=dict(weights=weights,threshold=threshold,selection=selection,teacher_selected_step=meta['selected_step'],teacher_actual_steps=meta['actual_steps'],teacher_config=meta['config'])
  old.sel.joblib.dump(model,str(folder/(key+'_rbf_private.joblib')));print('ROUTE_LOCKED',u,key,weights,flush=True)
 # Predefined personalized family: choose mild/strong by training OOF only.
 chosen=min(['personal_mild','personal_strong'],key=lambda k:(-locks[k]['selection']['mean_fold_auc'],k))
 old.save(folder/'locked_before_final.json',dict(routes=locks,personalized_choice=chosen,final_seen=False));np.savez_compressed(str(folder/'oof_private.npz'),**all_oof)
 finalcache=np.load(str(src/'scores_private.npz'));rows={};dump={}
 for key,spec in locks.items():
  scores={}
  for role in ['final_genuine','final_impostor']:
   s=finalcache['baseline_rbf__'+role].copy()
   if 'raw' in spec['weights']:
    raw=p.arrays[role];stats=np.concatenate([raw.mean(2),raw.std(2),raw.mean(1),raw.std(1)],1);s+=spec['weights']['raw']*standardized(cached['raw'],stats)
   if 'teacher' in spec['weights']:s+=spec['weights']['teacher']*standardized(models[key],embed(teachers[key],p.arrays[role]))
   scores[role]=s;dump[key+'__'+role]=s
  r=old.sel.evaluate_route(scores,p,u,spec['threshold']);r.update(weights=spec['weights'],teacher_selected_step=spec['teacher_selected_step']);g=scores['final_genuine'];n=scores['final_impostor'];threshold=spec['threshold']
  assert abs(((g[:,None]>n)+.5*(g[:,None]==n)).mean()-r['auc'])<1e-12
  assert abs((g>=threshold).mean()-r['tar'])<1e-12 and abs((n>=threshold).mean()-r['far'])<1e-12 and len(r['attack_by_user'])==8
  rows[key]=r
 assert old.sha(ganpath)==gansha
 np.savez_compressed(str(folder/'scores_private.npz'),**dump,attack_labels=p.labels['final_impostor'])
 out=dict(target=u,routes=rows,personalized_choice=chosen,gan_sha256_unchanged=gansha,baseline80_reference=json.loads((src/'results_public.json').read_text())['routes'],teacher_metadata=metas)
 old.save(folder/'results_public.json',out);return out

def main():
 ROOT.mkdir(exist_ok=True);old.records();dev=upstream()
 old.save(ROOT/'protocol_public.json',dict(targets=[4,5],configs={str(u):{k:settings(u,k) for k in CONFIGS} for u in [4,5]},seed='one matched initialization3800+target;pilot',teacher='same ACC/Audio teacher architecture;upstream120fit30val per9identities;no auth rows train teacher;max600steps',control='ordinary SupCon .1 +BCE8 +identityCE.3 +real crossmodal cosine.1',personal4='semihard training negatives logits quantile50-90%, weights1.5/2 normalized,all sources retained',personal5='crossword contrast.2/.3, lr3e-4/1e-4,wd.005/.01, patience4 checks',selection='upstream teacher samewordAUC then pooledAUC then BCE; auth80 nested5outer/3inner contentCV; same baseline/raw frozen OOF; candidate teacher/raw weights0/.25/.5/1 stability gate; choose mild/strong by OOF only; independent cal20 threshold',baseline='intact original772RBF cached from auth80;G/F frozen and checkpoint SHA verified',limitations=['pilot single matched seed','historical final reused exploratory closed-set8attackers','current candidates raw and one retrained ACCteacher,not historical allteacher family search','personalized choice uses OOF already used for forwardselection;needs new data','content mappings period10/20 tailmissing assumed','ACCteacher inclusion not proof of useful generated Audio','mild/strong changes multiple training knobs inU5;not singlefactor attribution']))
 rows={}
 for u in [4,5]:
  old.save(ROOT/'status_public.json',dict(state='running',target=u,completed=list(rows)));rows[str(u)]=target(u,dev);old.save(ROOT/'results_public.json',rows)
 old.save(ROOT/'status_public.json',dict(state='complete',completed=list(rows)))
if __name__=='__main__':
 try:main()
 except Exception as e:
  ROOT.mkdir(exist_ok=True);old.save(ROOT/'status_public.json',dict(state='failed',error=repr(e)));raise

