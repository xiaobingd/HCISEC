"""Independent scalar score recount plus Base-dev auxiliary diagnostics."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import json,gc
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score,r2_score
import run_esmb_t3_joint as r

torch.set_num_threads(1);root=r.ROOT;d={k:v for k,v in np.load(root/'inputs_private.npz').items()};stored=json.loads((root/'results_private.json').read_text());checks={};hashes=[]
dev=np.flatnonzero((d['user']<=4)&(d['index']>=80)&(d['index']<100))
for method in r.METHODS:
    z=np.load(root/method/'embeddings_private.npy');info=json.loads((root/method/'history.json').read_text());hashes.append(info['init_sha256']);ck=torch.load(root/method/'best_private.pt',map_location='cpu',weights_only=True);net=r.Model();net.load_state_dict(ck['state']);net.eval();pred=[]
    with torch.no_grad():
        for j in range(0,len(dev),8):pred.append(net(torch.tensor(d['x'][dev[j:j+8]]))[2].numpy())
    pred=np.concatenate(pred);truth=d['audio'][dev];checks[method]=dict(base_dev_spectrum_mse=float(np.mean((pred[:,:17]-truth[:,:17])**2)),base_dev_envelope_mse=float(np.mean((pred[:,17:]-truth[:,17:])**2)),base_dev_spectrum_r2=float(r2_score(truth[:,:17].transpose(0,2,1).reshape(-1,17),pred[:,:17].transpose(0,2,1).reshape(-1,17))),base_dev_envelope_r2=float(r2_score(truth[:,17:].ravel(),pred[:,17:].ravel())),selected_step=ck['step'])
    for rule,entry in stored[method].items():
        tau=entry['threshold']
        if rule=='cal_far05':assert entry['cal_far']<=.05+1e-12
        for u in [9,10]:
            en=np.flatnonzero((d['user']==u)&(d['index']<120));v=z[en].mean(0);proto=v/np.linalg.norm(v)
            for scope,allowed in [('unseen_attacker',[9,10]),('all_nine_sources',list(range(1,11)))]:
                ix=np.flatnonzero(np.isin(d['user'],allowed)&(d['index']>=160));assert not set(en)&set(ix);labels=d['user'][ix]==u;scores=np.array([float(np.dot(z[i],proto)) for i in ix]);old=entry['targets'][r.USERS[u-1]][scope]
                tar=float((scores[labels]>=tau).mean());far=float((scores[~labels]>=tau).mean());auc=float(roc_auc_score(labels,scores))
                for k,value in [('tar',tar),('far',far),('hter',(1-tar+far)/2),('auc',auc)]:assert abs(old[k]-value)<1e-7,(method,rule,u,k)
                assert old['genuine_n']==40 and old['impostor_n']==(40 if scope=='unseen_attacker' else 360)
    del net,ck,pred;gc.collect()
assert len(set(hashes))==1
assert np.all(d['audio'][d['user']>4]==0)
r.write(root/'verification_public.json',dict(status='pass',score_recount='all4methods bothrules bothtargets bothscopes',init_sha256=hashes[0],target_audio_targets_zero=True,checks=checks))
print(json.dumps(checks),flush=True)