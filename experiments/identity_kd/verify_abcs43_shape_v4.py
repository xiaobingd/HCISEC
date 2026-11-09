"""Independent saved-model inference and split audit; no Audio at query time."""
import json,sys
from pathlib import Path
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from run_rkd01_vibravox import Student,encode,norm

root=Path(sys.argv[1]);torch.set_num_threads(1)
d=dict(np.load(root/'inputs_private.npz'));a=dict(np.load(root/'audio_fit_register_private.npz'))
fit=d['role']=='fit';x=StandardScaler().fit(d['x'][fit]).transform(d['x']).astype('float32')
shape=norm(x[:,:40]);stats=norm(StandardScaler().fit(d['statistics'][fit]).transform(d['statistics']))
roles=['fit','develop','registration','calibration','query']
sets=[set(d['text_index'][d['role']==r]) for r in roles]
assert all(not sets[i]&sets[j] for i in range(5) for j in range(i))
assert not np.isin(d['role'][a['indices']],['query','calibration','develop']).any()
public=json.loads((root/'per_user_public.json').read_text());thresholds=json.loads((root/'thresholds_private.json').read_text())
checks=0;maxerror=0.;reset=0.;reference=None
for u in range(23,43):
    folder=root/f'owner{u}';t=dict(np.load(folder/'teacher_targets_private.npz'))
    expected=set(np.flatnonzero(((d['user']<23)&d['replay'])|((d['user']==u)&(d['role']=='registration'))))
    assert set(t['indices'])==expected and len(expected)==104
    audit=json.loads((folder/'teacher_audit_private.json').read_text());assert audit['backbone_update']>0 and audit['query_audio_model_forwards']==0
    replay=t['frozen'][:92]
    if reference is None:reference=replay
    reset=max(reset,float(np.max(np.abs(reference-replay))));assert np.allclose(reference,replay,atol=1e-6)
    audit=json.loads((folder/'student_audit_private.json').read_text());assert set(audit['training_indices'])==expected
    group=range(23,31) if u<31 else range(31,43);role='calibration' if u<31 else 'query'
    en=np.flatnonzero((d['user']==u)&(d['role']=='registration'));q=np.flatnonzero(np.isin(d['user'],list(group))&(d['role']==role))
    assert len(en)==12 and len(set(d['text_index'][en]))==12
    for v in group:assert len(set(d['text_index'][q[d['user'][q]==v]]))==24
    net=Student();net.load_state_dict(torch.load(folder/'student_private.pt',map_location='cpu',weights_only=False))
    z=encode(net,x);joint=np.concatenate([shape,z],1)/np.sqrt(2)
    saved=json.loads((folder/'scores_private.json').read_text())
    uid=f"ABCS43-{'Cal' if u<31 else 'Test'}-{u-22 if u<31 else u-30:02}"
    for method,features in [('shape40',shape),('statistics240',stats),('student192',z),('supplement232',joint)]:
        s=features[q]@norm(features[en].mean(0));y=d['user'][q]==u
        error=float(np.max(np.abs(s-np.array(saved['methods'][method]['scores']))));maxerror=max(maxerror,error);assert error<1e-5
        for point,tau in thresholds[method].items():
            row=next(r for r in public if r['user']==uid and r['method']==method and r['point']==point)
            tar=float((s[y]>=tau).mean());far=float((s[~y]>=tau).mean())
            assert abs(row['tar']-tar)<1e-7 and abs(row['far']-far)<1e-7 and abs(row['auc']-roc_auc_score(y,s))<1e-7
            checks+=1
result=dict(state='PASS',owners=20,per_user_workpoint_checks=checks,max_saved_score_error=maxerror,teacher_reset_max_error=reset,text_roles_disjoint=True,query_audio_forwards=0,student_inference='vibration features only')
(root/'verification_public.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
