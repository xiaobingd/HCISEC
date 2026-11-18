"""Strict-content rerun of optional teacher/diffusion score fusion."""
import os,sys,json,inspect,hashlib,types,subprocess,importlib.util
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,'/root')
import run_complete_baseline_teacher_20261028 as base
import optimize_personal_teacher_auth_20261028 as te
import personal_diffusion_supplement_20261028 as dp
import select_personal_diffusion_supplement_20261028 as sel
UP=Path('/autodl-fs/data/mei_strict_baseline_20261029')
ROOT=Path('/autodl-fs/data/mei_strict_autosupplement_20261029')
DATA=Path('/root/autodl-tmp/train_core/data')
USERS=list(range(1,10));ORDER=[1,4,5,2,6,7,9,3,8];D=te.D
M=json.loads((UP/'manifest_private.json').read_text());torch.set_num_threads(2)
def save(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def changed(s,a,b):
 assert a in s,a
 return s.replace(a,b)
def records():
 out={}
 for u in USERS:
  s=M['sources'][str(u)];r=s['roles']
  for k in ('acc','audio'):assert sha(DATA/('%s.%d.npy'%(k,u)))==s['sha256'][k]
  fit=r['gan_fit']+r['gan_val'];auth=sum((r[k] for k in ('auth_train','model_validation','calibration','test')),[]);assert not set(fit)&set(auth)
  out['bo' if u==1 else 'user%d'%u]=dict(cyclegan_indices=fit,auth_indices=auth,sha256=s['sha256']['acc'])
 return out
def fit_diffusion(u,sources,folder):
 src=changed(inspect.getsource(dp.train_personal),"idx=np.random.RandomState(4200+u).permutation(np.asarray(sources[name]['cyclegan_indices']))","idx=np.asarray(sources[name]['cyclegan_indices']).copy()")
 for k in ('acc','audio'):src=changed(src,"base.PROJECT/'meiy'/str(u)/'%s.npy'"%k,"DATA/('%s.%%d.npy'%%u)"%k)
 ns=dict(dp.__dict__);ns.update(OUT=folder,DATA=DATA);exec(compile(src,'strict_diffusion','exec'),ns);(folder/'executed_diffusion.py').write_text(src);path=ns['train_personal'](u,sources)
 meta=json.loads((folder/('U%d_diffusion_training.json'%u)).read_text());r=M['sources'][str(u)]['roles'];assert meta['fit_ids']==r['gan_fit'] and meta['selection_ids']==r['gan_val']
 saved=torch.load(str(path),map_location=D,weights_only=False);nets=[dp.BiometricUNet64().to(D),dp.BiometricUNet64().to(D)]
 for net,key in zip(nets,('a2u','u2a')):
  net.load_state_dict(saved['model_'+key+'_state_dict']);net.eval()
  for p in net.parameters():p.requires_grad_(False)
 return nets,dp.DiffusionSchedule(1000,device=D)
@torch.no_grad()
def extract(model,teachers,nets,schedule,p,roles):
 out={};checked=False
 for role in roles:
  parts={};xx=p.arrays[role]
  for i in range(0,len(xx),8):
   raw=xx[i:i+8];x=te.tensor(raw);fake=model.g_acc2audio(x)
   features={'base':base.fusion_inputs(model,x),'raw':torch.as_tensor(np.concatenate([raw.mean(2),raw.std(2),raw.mean(1),raw.std(1)],1),device=D)}
   for family,t in teachers.items():features[family+'_acc']=t.acc(x);features[family+'_audio']=t.audio(fake)
   df=dp.diffusion_features(x,fake,nets,schedule);features.update(df)
   if not checked:
    again=dp.diffusion_features(x[:2],fake[:2],nets,schedule);assert all(torch.allclose(df[k][:2],v,atol=2e-4,rtol=2e-4) for k,v in again.items());checked=True
   for name,z in features.items():assert torch.isfinite(z).all();parts.setdefault(name,[]).append(z.cpu().numpy())
  for name,z in parts.items():out[role+'__'+name]=np.concatenate(z)
  out[role+'__diff_bundle']=np.concatenate([out[role+'__'+k] for k in ('diff_latent','diff_guidance','diff_cycle')],1)
  print('EXTRACTED',role,len(xx),flush=True)
 return out
def protocol(u):
 import run_mei_strict_baseline_20261029 as strict
 p=strict.auth_protocol(u,M)
 p.words=np.concatenate([np.asarray(M['sources'][str(v)]['content_groups'])[M['sources'][str(v)]['roles']['auth_train']] for v in USERS])
 return p
def load_gan(u):
 spec=importlib.util.spec_from_file_location('model_drop',str(UP/'source/model_drop.py'));mod=importlib.util.module_from_spec(spec);sys.modules['model_drop']=mod;spec.loader.exec_module(mod)
 spec=importlib.util.spec_from_file_location('strict_autosupp_modelb',str(UP/'source/modelb.py'));mb=importlib.util.module_from_spec(spec);spec.loader.exec_module(mb)
 path=UP/('U%d'%u)/'cyclegan/models/checkpoint_epoch_200.pth';assert sha(path)==json.loads((path.parent.parent/'COMPLETE.json').read_text())['checkpoint_sha256']
 model,_=mb.create_multi_source_model(str(path),freeze_generators=True,proj_dim=128,device=D);model.eval()
 for p in model.parameters():p.requires_grad_(False)
 return model
def fit_teachers(u,model,sources,folder):
 src=changed(inspect.getsource(base.fit_teacher),'users=[1,2,4,5,6,7,9]','users=list(range(1,10))')
 src=changed(src,'idx=np.random.RandomState(4200+u).permutation(idx)','idx=idx.copy() # fit120 followed by val30')
 for k in ('acc','audio'):src=changed(src,"PROJECT/'meiy'/str(u)/'%s.npy'"%k,"DATA/('%s.%%d.npy'%%u)"%k)
 src=changed(src,'torch.tensor(6.,device=DEVICE)','torch.tensor(8.,device=DEVICE)')
 ns=dict(base.__dict__);ns.update(OUT=folder,DATA=DATA);exec(compile(src,'strict_old_teacher','exec'),ns)
 if not (folder/('U%d_teacher.pt'%u)).exists():ns['fit_teacher'](u,sources)
 dev={k:[] for k in ('a','m','uid','word','isfit')}
 for v in USERS:
  r=M['sources'][str(v)]['roles'];ix=np.asarray(r['gan_fit']+r['gan_val'])
  for domain,key in [('acc','a'),('audio','m')]:dev[key].append(np.load(str(DATA/('%s.%d.npy'%(domain,v)))).astype('float32')[ix,:80,:80])
  dev['uid'].append(np.full(150,v-1));dev['word'].append(np.asarray(M['sources'][str(v)]['content_groups'])[ix]);dev['isfit'].append(np.arange(150)<120)
 dev={k:np.concatenate(v) for k,v in dev.items()};dev['g']=te.generate(model.g_acc2audio,dev['a'])
 src2=changed(inspect.getsource(te.fit_teacher),'torch.tensor(6.,device=D)','torch.tensor(8.,device=D)');ns2=dict(te.__dict__);ns2.update(OUT=folder,DEVUSERS=USERS);exec(compile(src2,'strict_identity_teacher','exec'),ns2)
 for robust in (False,True):
  if not (folder/('U%d_%s_teacher.pt'%(u,'domain_robust' if robust else 'identity_preserving'))).exists():ns2['fit_teacher'](u,dev,robust)
 (folder/'executed_teachers.py').write_text(src+'\n\n'+src2);out={}
 for family,name in [('old','teacher'),('identity','identity_preserving_teacher'),('robust','domain_robust_teacher')]:
  saved=torch.load(str(folder/('U%d_%s.pt'%(u,name))),map_location=D,weights_only=False);t=base.Teacher().to(D);t.load_state_dict(saved['state']);t.eval()
  for p in t.parameters():p.requires_grad_(False)
  out[family]=t
 return out
def target(u):
 folder=ROOT/('U%d'%u);folder.mkdir(parents=True,exist_ok=True);sources=records();p=protocol(u);model=load_gan(u)
 save(folder/'lineage_public.json',dict(target=u,gan_sha256=sha(UP/('U%d'%u)/'cyclegan/models/checkpoint_epoch_200.pth'),strict_manifest_sha256=sha(UP/'manifest_private.json'),external_pretrain_sha256=sha(dp.PRETRAIN),teacher_users=USERS,fit_groups=M['groups']['gan_fit'],selection_groups=M['groups']['gan_val'],changes='Old three personal teacher families and personal50epoch diffusion; users7->9/posweight6->8 for closed set; corrected strict groups, no old Mei teacher/DDPM weights'))
 print('TEACHERS_START',u,flush=True);save(folder/'stage_public.json',{'stage':'teachers'})
 tt=fit_teachers(u,model,sources,folder);print('TEACHERS_COMPLETE',u,flush=True)
 save(folder/'stage_public.json',{'stage':'diffusion'});nets,schedule=fit_diffusion(u,sources,folder)
 save(folder/'stage_public.json',{'stage':'features_and_selection'})
 features=extract(model,tt,nets,schedule,p,('auth_train','model_validation','target_dev_pool','impostor_calibration'));np.savez_compressed(str(folder/'development_features_private.npz'),**features)
 names=sorted(k.split('__',1)[1] for k in features if k.startswith('auth_train__'));y=p.labels['auth_train']==u-1;words=p.words;assert len(np.unique(words))==5
 folds=list(sel.StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42).split(np.zeros(len(y)),y,words));oof={};classifiers={}
 for name in names:
  oof[name],classifiers[name]=sel.nested_branch(features['auth_train__'+name],y,words,folds);print('BRANCH_COMPLETE',u,name,flush=True)
 allowed=[k for k in names if k!='base' and not k.startswith('diff_')]
 w0,meta0=sel.forward_select(oof,y,folds,allowed,{'base':1.},2);w1,meta1=sel.forward_select(oof,y,folds,[k for k in names if k.startswith('diff_')],w0,1)
 weights={'baseline_rbf':{'base':1.},'auto_without_diffusion':w0,'auto_with_diffusion':w1}
 sel.joblib.dump(classifiers,str(folder/'branch_classifiers.joblib'));np.savez_compressed(str(folder/'oof_private.npz'),**oof)
 def scores(role,name):
  m=classifiers[name];return (sel.rbf.score(m['model'],[features[role+'__'+name]])-m['mean'])/m['std']
 thresholds={}
 for route,ww in weights.items():
  g=sum(w*scores('target_dev_pool',k) for k,w in ww.items());n=sum(w*scores('impostor_calibration',k) for k,w in ww.items());thresholds[route]=float(base.original.choose_threshold(g,n))
 locked=dict(target=u,weights=weights,thresholds=thresholds,non_diffusion_selection=meta0,diffusion_selection=meta1,rule='Original nested5word/inner3 CV, gain>=4folds and mean>max(.0025,pairedSE), weights0/.25/.5/1, max2nonDiff+1Diff; zero supplement allowed',final_seen=False)
 save(folder/'locked_before_final.json',locked)
 # No final feature extraction until models, supplementary weights and thresholds lock.
 features.update(extract(model,tt,nets,schedule,p,('final_genuine','final_impostor')));rows={};dump={}
 for route,ww in weights.items():
  ss={role:sum(w*scores(role,k) for k,w in ww.items()) for role in ('final_genuine','final_impostor')};rows[route]=sel.evaluate_route(ss,p,u,thresholds[route]);rows[route]['weights']=ww
  for role,value in ss.items():dump[route+'__'+role]=value
 np.savez_compressed(str(folder/'scores_private.npz'),**dump,attack_labels=p.labels['final_impostor'])
 reference=json.loads((UP/('U%d'%u)/'auth/results_public.json').read_text())['final_attack']
 save(folder/'results_public.json',dict(target=u,routes=rows,trainb_reference=reference,selection=locked,scope='Closed-set strict mixed-content group holdout under inferred ordering; no post-test route selection',diffusion_training=json.loads((folder/('U%d_diffusion_training.json'%u)).read_text())))
 verify_target(u);print('TARGET_COMPLETE',u,rows,flush=True)
def verify_target(u):
 folder=ROOT/('U%d'%u);r=json.loads((folder/'results_public.json').read_text());d=np.load(str(folder/'scores_private.npz'));checked=[]
 for route,a in r['routes'].items():
  g=d[route+'__final_genuine'];n=d[route+'__final_impostor'];t=r['selection']['thresholds'][route];auc=float(((g[:,None]>n[None,:])+.5*(g[:,None]==n[None,:])).mean());assert abs(auc-a['auc'])<1e-12
  assert abs(np.mean(g>=t)-a['tar'])<1e-12 and abs(np.mean(n>=t)-a['far'])<1e-12
  assert len(g)==a['genuine_n'] and len(n)==a['impostor_n'];assert len(a['attack_by_user'])==8
  for v,b in a['attack_by_user'].items():
   z=n[d['attack_labels']==int(v)-1];assert len(z)==b['n'] and int((z>=t).sum())==b['accepted']
  checked.append(route)
 save(folder/'COMPLETE.json',dict(status='PASS',routes=checked))
def queue():
 ROOT.mkdir(exist_ok=True)
 with (ROOT/'QUEUE_LOCK').open('x') as f:f.write(str(os.getpid()))
 save(ROOT/'protocol_public.json',dict(targets=ORDER,strict_protocol_sha256=sha(UP/'protocol_public.json'),groups=M['groups'],base_features=772,teacher_families=['old','identity','robust'],diffusion='Independent external ESMB0001 EMA initialization; new personal50epoch fine-tune strict fit120/val30; no old Mei diffusion',selector='Original RBF optional score fusion;5outer content folds/3inner folds within authfit50 peridentity, zero supplements allowed',closed_set=True,all_eight_attacks=True,limitations=M['limitations'],source_sha256={name:sha(Path('/root')/name) for name in ['personal_diffusion_supplement_20261028.py','select_personal_diffusion_supplement_20261028.py','optimize_personal_teacher_auth_20261028.py','run_complete_baseline_teacher_20261028.py']}))
 results={}
 for u in ORDER:
  folder=ROOT/('U%d'%u);folder.mkdir(exist_ok=True);save(ROOT/'status_public.json',dict(state='running',target=u,completed=list(results)))
  with (folder/'run.log').open('w') as f:subprocess.run([sys.executable,'-u',__file__,'target',str(u)],stdout=f,stderr=subprocess.STDOUT,check=True,env=dict(os.environ,OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'))
  results[str(u)]=json.loads((folder/'results_public.json').read_text());save(ROOT/'results_public.json',results)
 save(ROOT/'status_public.json',dict(state='complete',completed=list(results)));print('ALL_STRICT_AUTOSUPPLEMENT_COMPLETE',flush=True)
if __name__=='__main__':
 if len(sys.argv)==3 and sys.argv[1]=='target':target(int(sys.argv[2]))
 else:
  try:queue()
  except Exception as e:
   save(ROOT/'status_public.json',{'state':'failed','error':str(e)});raise
