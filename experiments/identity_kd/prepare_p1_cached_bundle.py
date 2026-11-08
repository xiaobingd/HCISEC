"""Pack existing frozen, identity-disjoint embeddings; no model training."""
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np
ROOT=Path('/autodl-fs/data');BASE=ROOT/'mei8_strict_open_residual_20261108';OUT=ROOT/'p1_multimodal_enrollment_20261108';OUT.mkdir(exist_ok=True)
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
a=np.load(ROOT/'identity_kd_all9_raw_full_multisource_20261103/raw9_full_features_private.npz');keep=a['user']!=9
np.savez(OUT/'metadata_private.npz',**{k:a[k][keep] for k in ['user','word','role','sample_id']})
provenance={}
with zipfile.ZipFile(OUT/'cached_bundle_private.zip','w',compression=zipfile.ZIP_STORED) as bundle:
    bundle.write(OUT/'metadata_private.npz','metadata_private.npz')
    for seed in [42,43,44]:
        for target in range(1,9):
            key=f'target{target}_seed{seed}';p=BASE/key
            info=json.loads((p/'backbone_complete.json').read_text());order=[1+(target-1+j)%8 for j in range(8)];assert info['base_users']==order[4:]
            provenance[key]=dict(**info,embedding_sha256=digest(p/'backbone_embeddings_private.npy'),model_sha256=digest(p/'best_backbone_private.pt'))
            bundle.write(p/'backbone_embeddings_private.npy',key+'/acc_private.npy')
    for name,p in [('ecapa_fit_private.npy',ROOT/'mei9_vibravox_teacher_transfer_20261107/vibravox_audio_embeddings_private.npy'),('eres_fit_private.npy',ROOT/'eres2netv2_control_20261108/audio_embedding_private.npy')]:
        bundle.write(p,name)
    bundle.writestr('provenance.json',json.dumps(provenance,indent=2))
print('BUNDLE_READY',OUT/'cached_bundle_private.zip',flush=True)
