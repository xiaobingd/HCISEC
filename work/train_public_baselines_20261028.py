"""Run fixed200-epoch canonical CycleGAN, change only the dataset adapter and output paths."""
import os,sys,json,time,hashlib,subprocess,re
from pathlib import Path
import numpy as np
ROOT=Path('/autodl-fs/data/public_baseline_compare_20261028');SRC=ROOT/'source';sys.path.insert(0,str(SRC));os.chdir(str(SRC))
os.environ['MPLBACKEND']='Agg'
import torch
torch.set_num_threads(2)

def train(dataset,owner):
 root=ROOT/dataset/owner;root.mkdir(exist_ok=True);data=ROOT/dataset
 p=json.loads((data/'protocol_private.json').read_text());fit=p['roles'][owner]['gan_train'];val=p['roles'][owner]['gan_val']
 other=set(sum((ids for role,ids in p['roles'][owner].items() if role not in ('gan_train','gan_val')),[]));assert not set(fit+val)&other
 source=(SRC/'train_cyclegan_user1.py').read_text();original_sha=hashlib.sha256(source.encode()).hexdigest()
 source=source.replace('model_dir = "/root/autodl-fs/auth_full_pipeline_150x150_20260924/user1/cyclegan/model"','model_dir = '+repr(str(root/'models')))
 source=source.replace('log_dir   = "/root/autodl-fs/auth_full_pipeline_150x150_20260924/user1/cyclegan/logs"','log_dir = '+repr(str(root/'logs')))
 a=source.index('data_loader = DataLoader(');b=source.index('# =====================\n# 3.',a)
 adapter="""data_loader=DataLoader.__new__(DataLoader)
__d=dict(np.load(DATA/'inputs.npz'));__p=json.loads((DATA/'protocol_private.json').read_text())
for __role,__prefix in [('gan_train','train'),('gan_val','val')]:
 __mask=np.isin(__d['unit_id'],__p['roles'][OWNER][__role])
 setattr(data_loader,__prefix+'_acc',__d['acc'][__mask]);setattr(data_loader,__prefix+'_audio',__d['audio'][__mask])
data_loader.acc_data=data_loader.train_acc;data_loader.audio_data=data_loader.train_audio
data_loader.test_acc=data_loader.val_acc[:0];data_loader.test_audio=data_loader.val_audio[:0]
assert len(data_loader.train_acc)==60 and len(data_loader.val_acc)==12
"""
 source=source[:a]+adapter+'\n'+source[b:]
 source,n=re.subn(r'^for epoch in range\(total_epochs\):$',lambda m:m.group(0)+'\n    print("PILOT_EPOCH",DATASET,OWNER,epoch+1,time.time(),flush=True)',source,flags=re.M);assert n==1
 source,n=re.subn(r'^        g_total\.backward\(\)$',lambda m:'        assert torch.isfinite(g_total).item(), "nonfinite generator loss"\n'+m.group(0),source,flags=re.M);assert n==1
 compile(source,str(root/'executed_training_source.py'),'exec')
 (root/'executed_training_source.py').write_text(source)
 (root/'training_protocol.json').write_text(json.dumps({'dataset':dataset,'owner':owner,'seed':42,'endpoint_epoch':200,'canonical_source_sha256':original_sha,
 'source_changes':'DataLoader adapter, output paths, epoch timestamp logging and finite-loss guard only. Architecture, losses, optimizer, schedules, batch16 unchanged.',
 'independent_fit_units':len(fit),'correlated_train_windows':60,'fit_units':fit,'validation_units':val,'authentication_overlap':0,
 'updates':'3 full batches/epoch x200=600 generator updates. Canonical loader drops last12windows each shuffled epoch; no extra independent recordings implied.'},indent=2))
 print('MODEL_START',dataset,owner,flush=True)
 exec(compile(source,str(root/'executed_training_source.py'),'exec'),{'__name__':'__main__','__file__':str(root/'executed_training_source.py'),'DATA':data,'OWNER':owner,'DATASET':dataset,'json':json,'time':time})
 assert (root/'models/checkpoint_epoch_200.pth').exists();print('MODEL_COMPLETE',dataset,owner,flush=True)

if __name__=='__main__':
 if len(sys.argv)==3:train(sys.argv[1],sys.argv[2])
 else:
  plans={d:json.loads((ROOT/d/'protocol_private.json').read_text())['owners'] for d in ('esmb','vibravox')}
  for j in range(2):
   for d in ('esmb','vibravox'):
    u=plans[d][j];p=ROOT/d/u;p.mkdir(exist_ok=True)
    if (p/'models/checkpoint_epoch_200.pth').exists():continue
    with (p/'training.log').open('w') as log:subprocess.run([sys.executable,'-u',__file__,d,u],stdout=log,stderr=subprocess.STDOUT,check=True,env=dict(os.environ,OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'))
  print('ALL_FOUR_GENERATORS_COMPLETE',flush=True)
  subprocess.run([sys.executable,'/root/evaluate_public_baselines_20261028.py'],check=True)
