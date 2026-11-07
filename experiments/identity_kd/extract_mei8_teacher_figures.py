import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import sys,json,time
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
sys.path.insert(0,'/root');import run_raw9_300 as r
torch.set_num_threads(1)
ROOT=Path('/autodl-fs/data');SRC=ROOT/'identity_kd_all9_raw_full_multisource_20261103';OUT=ROOT/'mei8_teacher_figures_20261108';OUT.mkdir(exist_ok=True)
a=np.load(SRC/'raw9_full_features_private.npz');keep=a['user']!=9
ix=np.concatenate([np.flatnonzero((a['user']==u)&(a['word']==w)&(a['role']=='final'))[:2] for u in range(1,9) for w in range(24,30)])
for n in ['model','deps']:
 if not (OUT/n).exists():(OUT/n).symlink_to(SRC/n,target_is_directory=True)
model=r.teacher_load(OUT);ck=torch.load(ROOT/'vibravox_dual_teacher_20261105/audio_teacher/best_backbone_private.pt',map_location='cpu',weights_only=False);model.mods.embedding_model.load_state_dict(ck['embedding_model']);del ck;model.eval()
z=np.zeros((len(ix),192),dtype='float32');start=time.time()
for j,i in enumerate(ix):
 with torch.no_grad():z[j]=F.normalize(model.encode_batch(torch.tensor(a['wave'][i:i+1].copy())).squeeze(1),dim=1).numpy()[0]
 if j%20==0:print(j,len(ix),round(time.time()-start,1),flush=True)
za=np.load(ROOT/'mei8_exclude_yue_20261108/acc_embeddings_private.npy');lookup=np.full(len(keep),-1);lookup[np.flatnonzero(keep)]=np.arange(keep.sum())
np.savez(OUT/'heldout_embeddings_private.npz',audio=z,acc=za[lookup[ix]],user=a['user'][ix],word=a['word'][ix],sample_id=a['sample_id'][ix])
print('COMPLETE',flush=True)
