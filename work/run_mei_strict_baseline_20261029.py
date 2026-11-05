"""Fresh personal CycleGAN + original full trainb, mixed10/20 content-group holdout.

Grouping follows Audio periodicity and user-confirmed digit/command composition;
semantic position and missing-tail assumptions are not a transcript manifest.
"""
import os,sys,json,re,time,hashlib,subprocess,types,copy,random
from pathlib import Path
import numpy as np
ROOT=Path('/autodl-fs/data/mei_strict_baseline_20261029');SRC=ROOT/'source';DATA=Path('/root/autodl-tmp/train_core/data')
GROUPS={'gan_fit':[0,1,2,3]+list(range(10,18)),'gan_val':[4,18,19],'auth_train':[5,6,20,21,22],'model_validation':[7,23,24],'calibration':[8,25],'test':[9,26,27,28,29]}
USERS=list(range(1,10));ORDER=[1,4,5,2,6,7,9,3,8]
def save(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,indent=2));q.replace(p)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def content(n):
 i=np.arange(n);return np.where(i<100,i%10,10+(i-100)%20)
def repeats(n):
 i=np.arange(n);return np.where(i<100,i//10,(i-100)//20)
def prepare():
 ROOT.mkdir(exist_ok=True);assert sorted(sum(GROUPS.values(),[]))==list(range(30))
 sources={}
 for u in USERS:
  paths={k:DATA/('%s.%d.npy'%(k,u)) for k in ('acc','audio')};a=np.load(str(paths['acc']));m=np.load(str(paths['audio']))
  assert a.shape==m.shape and a.ndim==3 and a.shape[1:]==(80,81) and 297<=len(a)<=300
  assert np.isfinite(a).all() and np.isfinite(m).all();g=content(len(a));r=repeats(len(a));roles={k:np.flatnonzero(np.isin(g,v)).tolist() for k,v in GROUPS.items()}
  assert sorted(sum(roles.values(),[]))==list(range(len(a)))
  assert len(roles['gan_fit'])==120 and len(roles['gan_val'])==30
  for k in roles:assert set(g[roles[k]])==set(GROUPS[k])
  sources[str(u)]=dict(rows=len(a),shape=list(a.shape),sha256={k:sha(p) for k,p in paths.items()},roles=roles,content_groups=g.tolist(),repeat_ids=r.tolist())
 manifest=dict(users=USERS,queue_order=ORDER,groups=GROUPS,sources=sources,seed=42,gan_epochs=200,auth_epochs=100,features=772,head='Original trainb fusion512/SupCon128 plus target BCE',preprocessing='Original arrays unchanged, float32 crop [:,:80,:80] identically for everyone; no per-user filtering/scaling',scope='Closed-set known negative identities; disjoint content groups at all upstream/authentication stages under mixed10/20 mapping',limitations=['Content grouping inferred from real Audio and user-confirmed two-part composition, not per-record transcript verification','Missing rows assumed at tail per user; actual source names/phase not individually verified','Existing spectrogram normalization provenance unknown; no new all-data normalizer fitted','Same-session, historical recordings reused; newly initialized weights do not make data blind'],all_sources_retained=True,no_teacher=True,no_diffusion=True)
 p=ROOT/'manifest_private.json'
 if p.exists():assert json.loads(p.read_text())==manifest
 else:save(p,manifest)
 public={k:v for k,v in manifest.items() if k!='sources'}
 public.update(counts={u:{k:len(v) for k,v in s['roles'].items()} for u,s in sources.items()},source_sha256={u:s['sha256'] for u,s in sources.items()})
 save(ROOT/'protocol_public.json',public)
 print('PREPARED', {u:{k:len(v) for k,v in s['roles'].items()} for u,s in sources.items()},flush=True)
 return manifest
def train_gan(u):
 import torch
 torch.set_num_threads(2);sys.path.insert(0,str(SRC));os.chdir(str(SRC));os.environ['MPLBACKEND']='Agg'
 p=json.loads((ROOT/'manifest_private.json').read_text());folder=ROOT/('U%d'%u)/'cyclegan';folder.mkdir(parents=True,exist_ok=True)
 source=(SRC/'train_cyclegan_user1.py').read_text();original_sha=hashlib.sha256(source.encode()).hexdigest()
 assert source.count('model_dir = "/root/autodl-fs/auth_full_pipeline_150x150_20260924/user1/cyclegan/model"')==1
 source=source.replace('model_dir = "/root/autodl-fs/auth_full_pipeline_150x150_20260924/user1/cyclegan/model"','model_dir = '+repr(str(folder/'models')))
 source=source.replace('log_dir   = "/root/autodl-fs/auth_full_pipeline_150x150_20260924/user1/cyclegan/logs"','log_dir = '+repr(str(folder/'logs')))
 start=source.index('data_loader = DataLoader(');end=source.index('# =====================\n# 3.',start)
 adapter='''data_loader=DataLoader.__new__(DataLoader)
__a=np.load(str(DATA/('acc.%d.npy'%OWNER))).astype('float32')[:,:80,:80]
__m=np.load(str(DATA/('audio.%d.npy'%OWNER))).astype('float32')[:,:80,:80]
for __role,__prefix in [('gan_fit','train'),('gan_val','val')]:
 __idx=np.asarray(MANIFEST['sources'][str(OWNER)]['roles'][__role])
 setattr(data_loader,__prefix+'_acc',__a[__idx]);setattr(data_loader,__prefix+'_audio',__m[__idx])
data_loader.acc_data=data_loader.train_acc;data_loader.audio_data=data_loader.train_audio
data_loader.test_acc=data_loader.val_acc[:0];data_loader.test_audio=data_loader.val_audio[:0]
assert len(data_loader.train_acc)==120 and len(data_loader.val_acc)==30
'''
 source=source[:start]+adapter+'\n'+source[end:]
 source,n=re.subn(r'^for epoch in range\(total_epochs\):$',lambda m:m.group(0)+'\n    print("STRICT_EPOCH",OWNER,epoch+1,time.time(),flush=True)',source,flags=re.M);assert n==1
 source,n=re.subn(r'^        g_total\.backward\(\)$',lambda m:'        assert torch.isfinite(g_total).item(), "nonfinite generator loss"\n'+m.group(0),source,flags=re.M);assert n==1
 compile(source,str(folder/'executed_source.py'),'exec');(folder/'executed_source.py').write_text(source)
 save(folder/'training_protocol_public.json',dict(owner=u,seed=42,epochs=200,train_groups=GROUPS['gan_fit'],validation_groups=GROUPS['gan_val'],train_rows=120,validation_rows=30,source_sha256=original_sha,changes='Data adapter, output paths, timestamp logging and finite loss guard only; canonical architecture/optimizer/losses/schedules unchanged',no_resume=True,source_bundle_sha256={p.name:sha(p) for p in SRC.glob('*.py')}))
 exec(compile(source,str(folder/'executed_source.py'),'exec'),dict(__name__='__main__',__file__=str(folder/'executed_source.py'),DATA=DATA,OWNER=u,MANIFEST=p,time=time))
 checkpoint=folder/'models/checkpoint_epoch_200.pth';assert checkpoint.exists();save(folder/'COMPLETE.json',dict(epoch=200,checkpoint_sha256=sha(checkpoint)))
 print('GAN_COMPLETE',u,flush=True)
def auth_protocol(u,manifest):
 arrays={};labels={};ids={};raw={v:np.load(str(DATA/('acc.%d.npy'%v))).astype('float32')[:,:80,:80] for v in USERS}
 for k in ('auth_train','model_validation','calibration','test'):
  arrays[k]=np.concatenate([raw[v][manifest['sources'][str(v)]['roles'][k]] for v in USERS]);labels[k]=np.concatenate([np.full(len(manifest['sources'][str(v)]['roles'][k]),v-1) for v in USERS]);ids[k]=[str(v)+':'+str(i) for v in USERS for i in manifest['sources'][str(v)]['roles'][k]]
 for src,dst in [('calibration','target_dev_pool'),('calibration','impostor_calibration'),('test','final_genuine'),('test','final_impostor')]:
  mask=labels[src]==u-1 if dst in ('target_dev_pool','final_genuine') else labels[src]!=u-1
  arrays[dst]=arrays[src][mask];labels[dst]=labels[src][mask];ids[dst]=np.asarray(ids[src])[mask].tolist()
 counts={k:dict(total=len(a),positive=int((labels[k]==u-1).sum()),negative=int((labels[k]!=u-1).sum())) for k,a in arrays.items()}
 return types.SimpleNamespace(arrays=arrays,labels=labels,sample_ids=ids,manifest=dict(target=u,counts=counts,groups=GROUPS,source_rows={v:s['roles'] for v,s in manifest['sources'].items()},mapping_scope=manifest['limitations'][0]))
def train_auth(u):
 import torch
 torch.set_num_threads(2);sys.path.insert(0,str(SRC));os.chdir(str(SRC));import trainb_target_binary as original
 from sklearn.metrics import roc_auc_score
 manifest=json.loads((ROOT/'manifest_private.json').read_text());p=auth_protocol(u,manifest);folder=ROOT/('U%d'%u)/'auth';folder.mkdir(exist_ok=True)
 source=(SRC/'trainb_target_binary.py').read_text();assert 'choices=(1, 3, 4, 5)' in source;source=source.replace('choices=(1, 3, 4, 5)','choices=tuple(range(1,10))')
 scope=dict(original.__dict__);scope['__name__']='strict_trainb';exec(compile(source,'strict_trainb','exec'),scope)
 precompute=scope['precompute_features'];evaluate=scope['evaluate'];final_ids={id(p.arrays[k]) for k in ('final_genuine','final_impostor')}
 def cached(model,specs,labels,device,batch_size,augment=None):
  if id(specs) in final_ids:return torch.empty((0,772)),torch.empty(0,dtype=torch.long)
  return precompute(model,specs,labels,device,batch_size,augment)
 def evaluated(model,binary,cache,protocol,target_user,final=False):
  cc={k:v for k,v in cache.items() if k not in ('final_genuine','final_impostor')}
  if not final:return evaluate(model,binary,cc,protocol,target_user,False)
  device=next(model.parameters()).device;cal_g=scope['binary_values'](model,binary,cc['target_dev_pool'],device);cal_n=scope['binary_values'](model,binary,cc['impostor_calibration'],device);t=float(scope['choose_threshold'](cal_g,cal_n))
  torch.save(dict(model_state_dict=model.state_dict(),binary_state_dict=binary.state_dict(),threshold=t),folder/'locked_before_queries.pth')
  for k in ('final_genuine','final_impostor'):
   x,_=precompute(model,p.arrays[k],p.labels[k],device,16);cc[k]=x.to(device)
  row=evaluate(model,binary,cc,protocol,target_user,True);assert row['threshold']==t
  ss={k:scope['binary_values'](model,binary,cc[k],device) for k in ('final_genuine','final_impostor')}
  np.savez_compressed(str(folder/'scores_private.npz'),**ss,attack_labels=p.labels['final_impostor'])
  row.update(genuine_n=len(ss['final_genuine']),genuine_accepted=int(np.sum(ss['final_genuine']>=t)),attack_n=len(ss['final_impostor']),attack_accepted=int(np.sum(ss['final_impostor']>=t)))
  g=ss['final_genuine'];n=ss['final_impostor'];assert abs(roc_auc_score(np.r_[np.ones(len(g)),np.zeros(len(n))],np.r_[g,n])-row['auc'])<1e-12
  assert sum(x['accepted'] for x in row['attack_by_user'].values())==row['attack_accepted'];assert len(row['attack_by_user'])==8
  save(folder/'verification_public.json',dict(status='PASS',counts=True,auc=True,all_eight_attack_sources=True,locked_before_queries=True));return row
 scope.update(build_target_protocol=lambda *a,**kw:p,save_manifest=lambda p,path:save(Path(path),p.manifest),precompute_features=cached,evaluate=evaluated)
 random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
 sys.argv=['trainb_target_binary.py','--target_user',str(u),'--cyclegan_ckpt',str(ROOT/('U%d'%u)/'cyclegan/models/checkpoint_epoch_200.pth'),'--save_dir',str(folder),'--log_dir',str(folder/'logs'),'--feature_batch_size','16']
 scope['main']();r=json.loads((folder/'trainb_binary_results.json').read_text());r.update(features=772,generator='Fresh personal200epoch',content_groups=GROUPS,scope=manifest['scope'],limitations=manifest['limitations'])
 save(folder/'results_public.json',r);save(folder/'COMPLETE.json',dict(status='PASS'));print('AUTH_COMPLETE',u,r['final_attack'],flush=True)
def queue():
 prepare();lock=ROOT/'QUEUE_LOCK'
 # Exclusive marker prevents accidental duplicate queue launch. Do not remove automatically.
 with lock.open('x') as f:f.write(str(os.getpid()))
 results={}
 for u in ORDER:
  (ROOT/('U%d'%u)).mkdir(exist_ok=True)
  for stage in ('gan','auth'):
   folder=ROOT/('U%d'%u)/('cyclegan' if stage=='gan' else 'auth');folder.mkdir(exist_ok=True)
   if (folder/'COMPLETE.json').exists():continue
   save(ROOT/'status_public.json',dict(state='running',user=u,stage=stage,completed_targets=list(results),time_utc=time.time()))
   with (folder/'run.log').open('w') as log:subprocess.run([sys.executable,'-u',__file__,stage,str(u)],stdout=log,stderr=subprocess.STDOUT,check=True,env=dict(os.environ,OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',MPLBACKEND='Agg'))
  results[str(u)]=json.loads((ROOT/('U%d'%u)/'auth/results_public.json').read_text());save(ROOT/'results_public.json',results)
  subprocess.run([sys.executable,'/root/report_mei_strict_baseline_20261029.py'],check=True)
 save(ROOT/'status_public.json',dict(state='complete',completed_targets=list(results)));save(ROOT/'COMPLETE.json',dict(targets=9,verification='Each target counts/AUC/eight sources checked before completion'));print('ALL_STRICT_BASELINES_COMPLETE',flush=True)
if __name__=='__main__':
 if len(sys.argv)==2 and sys.argv[1]=='prepare':prepare()
 elif len(sys.argv)==3:
  if sys.argv[1]=='gan':train_gan(int(sys.argv[2]))
  elif sys.argv[1]=='auth':train_auth(int(sys.argv[2]))
  else:raise ValueError(sys.argv)
 else:queue()
