"""Extend audited eight-fold identity-disjoint caches to include unseen ninth user."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import json,sys,zipfile,time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,'/root')
import run_raw9_300 as r
ROOT=Path('/autodl-fs/data');OUT=ROOT/'mei9_open_raw_fusion_20261109';OLD=ROOT/'mei8_strict_open_residual_20261108'
def main():
    OUT.mkdir(exist_ok=True);torch.set_num_threads(1);r.p.DEVICE='cpu';a=np.load(ROOT/'identity_kd_all9_raw_full_multisource_20261103/raw9_full_features_private.npz');d={k:a[k] for k in ['user','word','role','sample_id']};np.savez(OUT/'metadata_private.npz',**d);ninth=d['user']==9;keep=~ninth;raw=a['acc_vel_300'][ninth].astype('float32');audit={}
    for target in range(1,9):
        key='target'+str(target)+'_seed42';p=OLD/key;info=json.loads((p/'backbone_complete.json').read_text());order=[1+(target-1+j)%8 for j in range(8)];assert info['base_users']==order[4:] and target not in info['base_users']
        dest=OUT/(key+'_full_embeddings_private.npy')
        if not dest.exists():
            scale=json.loads((p/'scaling.json').read_text());assert scale['users']==info['base_users'];x=np.clip(2*(raw-scale['low'])/(scale['high']-scale['low'])-1,-1,1).astype('float32');net=r.Student();ck=torch.load(p/'best_backbone_private.pt',map_location='cpu');net.load_state_dict(ck['state']);net.eval();vals=[]
            with torch.no_grad():
                for i in range(0,len(x),8):vals.append(net(torch.tensor(x[i:i+8])).numpy())
            z=np.zeros((len(ninth),192),dtype='float32');z[keep]=np.load(p/'backbone_embeddings_private.npy');z[ninth]=np.concatenate(vals);assert np.isfinite(z).all() and np.all(np.linalg.norm(z,axis=1)>.9);np.save(dest,z);del net,x
        audit[key]=dict(base_users=info['base_users'],seed=42,steps=info['steps'],new_inference_only='user9 never in eight original model training/dev/scales',original_users=8,full_users=9)
        (OUT/'export_status_public.json').write_text(json.dumps(dict(state='exporting',completed=target,total=8)));print('EXPORTED',target,flush=True)
    teacher=ROOT/'mei9_vibravox_teacher_transfer_20261107/vibravox_audio_embeddings_private.npy';t=np.load(teacher);assert t.shape==(len(ninth),192)
    with zipfile.ZipFile(OUT/'bundle_private.zip','w',compression=zipfile.ZIP_STORED) as z:
        z.write(OUT/'metadata_private.npz','metadata_private.npz');z.write(teacher,'audio_private.npy')
        for target in range(1,9):
            name='target'+str(target)+'_seed42_full_embeddings_private.npy';z.write(OUT/name,name)
        z.writestr('provenance_private.json',json.dumps(audit))
    (OUT/'export_status_public.json').write_text(json.dumps(dict(state='complete',completed=8,total=8)));print('BUNDLE_COMPLETE',flush=True)
if __name__=='__main__':main()
