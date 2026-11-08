"""Audio inference ONLY on first four final-word repeats used for enrollment."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import json
from pathlib import Path
import sys
import numpy as np
import torch
from torch.nn import functional as F
sys.path.insert(0,'/root')
from run_ecapa_pilot import teacher_load
ROOT=Path('/autodl-fs/data');SRC=ROOT/'identity_kd_all9_raw_full_multisource_20261103';OUT=ROOT/'p1_multimodal_enrollment_20261108';OUT.mkdir(exist_ok=True)
for name in ['model','deps']:
    if not(OUT/name).exists():(OUT/name).symlink_to(SRC/name,target_is_directory=True)
torch.set_num_threads(1);device='cuda' if torch.cuda.is_available() else 'cpu'
a=np.load(SRC/'raw9_full_features_private.npz');waves=np.load(ROOT/'mei8_audio_teacher_localft_20261108/waves_private.npy',mmap_mode='r');assert len(waves)==len(a['user']);ix=np.concatenate([np.flatnonzero((a['user']==u)&(a['word']==w)&(a['role']=='final'))[:4] for u in range(1,9) for w in range(24,30)])
assert len(ix)==192 and len(np.unique(ix))==192
for domain in ['ecapa','eres']:
    path=OUT/(domain+'_enrollment_private.npz')
    if path.exists():continue
    result=np.zeros((len(ix),192),dtype='float32');done=np.zeros(len(ix),bool)
    if domain=='ecapa':
        model=teacher_load(OUT);state=torch.load(ROOT/'vibravox_dual_teacher_20261105/audio_teacher/best_backbone_private.pt',map_location='cpu',weights_only=False);model.mods.embedding_model.load_state_dict(state['embedding_model']);del state;model.eval()
        old=np.load(ROOT/'mei8_teacher_figures_20261108/heldout_embeddings_private.npz');lookup={str(s):j for j,s in enumerate(old['sample_id'])}
        for j,i in enumerate(ix):
            if str(a['sample_id'][i]) in lookup:result[j]=old['audio'][lookup[str(a['sample_id'][i])]];done[j]=True
        def encode(w):return F.normalize(model.encode_batch(w).squeeze(1),dim=1)
    else:
        sys.path.insert(0,str(ROOT/'eres2netv2_control_20261108'))
        from speakerlab.models.eres2net.ERes2NetV2 import ERes2NetV2
        import torchaudio.compliance.kaldi as kaldi
        model=ERes2NetV2(feat_dim=80,embedding_size=192).to(device);model.load_state_dict(torch.load(ROOT/'eres2netv2_control_20261108/pretrained_eres2netv2.ckpt',map_location='cpu'));model.eval()
        def encode(w):
            f=kaldi.fbank(w,num_mel_bins=80,sample_frequency=16000,dither=0);f=f-f.mean(0,keepdim=True);return F.normalize(model(f[None].to(device)),dim=1)
    for j,i in enumerate(ix):
        if not done[j]:
            w=torch.tensor(waves[i:i+1].astype('float32'),device=device if domain=='ecapa' else 'cpu')
            with torch.no_grad():result[j]=encode(w).cpu().numpy()[0]
        if j%20==0:(OUT/'status.json').write_text(json.dumps(dict(state='extracting_registration_audio',teacher=domain,done=j+1,total=len(ix))));print(domain,j+1,len(ix),flush=True)
    np.savez(path,audio=result,sample_id=a['sample_id'][ix],source_indices=ix)
    del model
(OUT/'status.json').write_text(json.dumps(dict(state='registration_audio_complete',n=192,query_audio_read=False)));print('COMPLETE',flush=True)
