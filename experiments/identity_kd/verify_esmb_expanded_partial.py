"""Recount locked T0/T3 before remaining controls finish; never change training."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
root=Path('/autodl-fs/data/esmb_t3_expanded_20261109');p=json.loads((root/'protocol.json').read_text());users=p['users'];nb=len(p['base']);nc=len(p['cal']);targets=range(nb+nc+1,len(users)+1);cal=range(nb+1,nb+nc+1);a=np.load(root/'inputs_private.npz');u=a['user'];j=a['index'];results=json.loads((root/'partial_T0_T3_private.json').read_text());hashes=[];completion={}
for method,entries in results.items():
    z=np.load(root/method/'embeddings_private.npy');hist=json.loads((root/method/'history.json').read_text());done=json.loads((root/method/'complete.json').read_text());best=max(hist['history'],key=lambda x:(x['base_dev_auc'],-x['step']));assert done['steps']==400 and best['step']==done['selected_step'];hashes.append(hist['init_sha256']);completion[method]=done
    for rule,entry in entries.items():
        tau=entry['threshold'];cs=[];cy=[]
        for cu in cal:
            en=np.flatnonzero((u==cu)&(j<120));v=z[en].mean(0);v/=np.linalg.norm(v);ix=np.flatnonzero(np.isin(u,list(cal))&(j>=200));ss=z[ix]@v;cs.extend(ss);cy.extend(u[ix]==cu)
        cs=np.array(cs);cy=np.array(cy)
        assert abs(entry['cal_tar']-(cs[cy]>=tau).mean())<1e-9 and abs(entry['cal_far']-(cs[~cy]>=tau).mean())<1e-9
        if rule=='cal_far05':assert entry['cal_far']<=.05+1e-12
        for who in targets:
            en=np.flatnonzero((u==who)&(j<120));assert len(en)==120;v=z[en].mean(0);v/=np.linalg.norm(v)
            for scope,ids in [('unseen_attackers',list(targets)),('all_other_sources',range(1,len(users)+1))]:
                ix=np.flatnonzero(np.isin(u,list(ids))&(j>=200));assert not set(ix)&set(en);truth=u[ix]==who;ss=np.array([float(np.dot(z[k],v)) for k in ix]);m=entry['targets'][users[who-1]][scope];tar=float((ss[truth]>=tau).mean());far=float((ss[~truth]>=tau).mean());auc=float(roc_auc_score(truth,ss))
                for k,value in [('tar',tar),('far',far),('hter',(1-tar+far)/2),('auc',auc)]:assert abs(m[k]-value)<1e-7,(method,rule,who,scope,k)
                assert m['genuine_n']==40 and m['impostor_n']==(160 if scope=='unseen_attackers' else 520)
assert len(set(hashes))==1 and np.all(a['audio'][u>nb]==0)
out=dict(state='pass_partial_T0_T3_only',completed=completion,init_sha256=hashes[0],rules='both predefined rules;Calstats andscalarqueries recounted',pending=['T3_wrong_time','T3_wrong_user'],target_audio_zero=True)
(root/'partial_verification_public.json').write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True)