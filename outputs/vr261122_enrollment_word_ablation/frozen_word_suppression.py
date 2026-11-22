"""Paired intervention in the existing ACC encoder's embedding space."""
import json
import torch
from torch import nn
from enrollment_word_ablation import features, root, user, word, repeat, np, normalize, SVC, roc_curve

torch.set_num_threads(2)
ck = torch.load(root/"adapted_acc_private.pt",map_location="cpu",weights_only=True)
encoder = nn.Sequential(nn.Linear(features.shape[1],256),nn.LayerNorm(256),nn.GELU(),nn.Linear(256,192))
encoder.load_state_dict(ck["encoder"])
with torch.inference_mode():
    x = torch.tensor((features-ck["mean"].numpy())/ck["scale"].numpy(),dtype=torch.float32)
    z = normalize(encoder.eval()(x).numpy())
with np.load(root/"paired_raw_acc_private.npz") as data:
    fit = data["role"]=="fit"
    dev = data["role"]=="develop"
enrolled = np.isin(word,range(20,24))
reg = enrolled & (repeat<3)
samecal = enrolled & np.isin(repeat,[3,4])
cal = samecal | dev
query = np.isin(word,range(24,30)) & (repeat<4)
assert [m.sum() for m in (fit,dev,reg,samecal,query)]==[2080,320,120,80,240]
assert (fit.astype(int)+reg+cal+query).max()==1
centers = np.stack([z[word==w].mean(0) for w in np.unique(word[fit])])
basis = np.linalg.svd(centers-centers.mean(0),full_matrices=False)[2][:3]
rows = []
for shrink in [False,True]:
    view = normalize(z-.5*(z@basis.T)@basis) if shrink else z
    for u in range(1,10):
        y = user==u
        model = SVC(C=10,class_weight="balanced").fit(view[reg],y[reg])
        f,t,h = roc_curve(y[cal],model.decision_function(view[cal]),
                          sample_weight=np.where(samecal[cal],4.,1.),drop_intermediate=False)
        tau = h[np.argmin(abs(f+t-1))]
        s = model.decision_function(view[query])
        labels = y[query]
        far,frr = np.mean(s[~labels]>=tau),np.mean(s[labels]<tau)
        f,t,_ = roc_curve(labels,s,drop_intermediate=False)
        j = np.argmin(abs(f+t-1))
        rows.append(dict(user=u,shrink=shrink,eer=(f[j]+1-t[j])/2,far=far,frr=frr,hter=(far+frr)/2))
    print(shrink, [round(100*np.mean([r[k] for r in rows if r["shrink"]==shrink]),2)
                   for k in ("eer","hter","far","frr")])
(root/"frozen_word_suppression_public.json").write_text(json.dumps(rows,indent=2))

