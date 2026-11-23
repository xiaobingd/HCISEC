"""Independent global-index replay of the first Chinese fold/owner/seed."""
import itertools
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from calibrate_threeword import cutoff,measure

root=Path(__file__).resolve().parent
p=root/'strict_within_content';base=root/'phrase_auth'
d=np.load(base/'frontend_private.npz');u,w,r=d['user'],d['word'],d['repeat']
torch.set_num_threads(2)
x=torch.tensor(StandardScaler().fit(d['features'][d['fit']]).transform(d['features']),dtype=torch.float32)
net=nn.Sequential(nn.Linear(762,256),nn.LayerNorm(256),nn.GELU(),nn.Linear(256,192))
net.load_state_dict(torch.load(base/'AudioKD_42_private.pt',map_location='cpu',weights_only=True))
with torch.inference_mode():v=F.normalize(net.eval()(x),dim=1).numpy()
reg=w==35
head=SVC(C=10,class_weight='balanced').fit(v[reg],u[reg]==1)
s=head.decision_function(v)
rows=json.loads((p/'fold_user_results_private.json').read_text())
locks=json.loads((p/'thresholds_private.json').read_text());errors=[]
for k in (1,3):
    cal=[];query=[];cy=[];qy=[]
    for us in range(1,11):
        c=sorted(np.flatnonzero((u==us)&(w==36)),key=lambda i:r[i])
        if k==1:
            cal.extend(s[c]);cy.extend([us==1]*8)
            q=np.flatnonzero((u==us)&np.isin(w,[37,38,39]))
            query.extend(s[q]);qy.extend([us==1]*24)
        else:
            cs=[float(np.mean(s[list(combo)])) for combo in itertools.combinations(c,3)]
            cal.extend(cs);cy.extend([us==1]*56)
            for rep in range(8):
                q=np.flatnonzero((u==us)&np.isin(w,[37,38,39])&(r==rep))
                assert len(q)==3
                query.append(float(s[q].mean()));qy.append(us==1)
    tau=cutoff(np.array(cal),np.array(cy),'quantile_gap')
    result=measure(np.array(query),np.array(qy),tau)
    prior=next(a for a in rows if a['seed']==42 and a['method']=='AudioKD' and a['family']=='chinese_long'
               and a['fold']==0 and a['user']==1 and a['words']==k)
    priorlock=next(a for a in locks if a['seed']==42 and a['method']=='AudioKD' and a['family']=='chinese_long'
               and a['fold']==0 and a['user']==1 and a['words']==k)
    errors.append(abs(tau-priorlock['threshold']))
    errors += [abs(result[z]-prior[z]) for z in ['far','frr','hter','eer','auc']]
assert max(errors)<1e-10
verification=json.loads((p/'verification_public.json').read_text())
verification.update(independent_first_chinese_fold_global_index_replay_max_error=max(errors),
                    replay_case=dict(seed=42,method='AudioKD',user=1,fold=0,challenge_lengths=[1,3]))
(p/'verification_public.json').write_text(json.dumps(verification,indent=2))
print('INDEPENDENT_REPLAY_PASS',max(errors))
