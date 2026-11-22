"""Frozen Audio-guided student, fixed 24-shot cross-word enrollment."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import normalize
from sklearn.svm import SVC
from sklearn.metrics import roc_curve

parser=argparse.ArgumentParser()
parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parent)
args=parser.parse_args()
root=args.root
torch.set_num_threads(2)
with np.load(root/"paired_raw_acc_private.npz") as d:
    ids=d["sample_id"]
    u,w,r=np.array([s.split("_") for s in ids],dtype=int).T
    spec=d["spectrum"].reshape(len(ids),120,-1)
    amplitude=d["amplitude"]
    fit=d["role"]=="fit"
with np.load(root/"stats27_private.npz") as d:
    assert np.array_equal(ids,d["sample_id"])
    features=np.column_stack([spec.mean(-1),spec.std(-1),
                             *(s.mean(-1) for s in np.array_split(spec,4,axis=-1)),
                             amplitude,d["features"]])
ck=torch.load(root/"adapted_acc_private.pt",map_location="cpu",weights_only=True)
net=nn.Sequential(nn.Linear(features.shape[1],256),nn.LayerNorm(256),nn.GELU(),nn.Linear(256,192))
x=torch.tensor((features-ck["mean"].numpy())/ck["scale"].numpy(),dtype=torch.float32)
base_words=[20,21,22,23]
extra_words=[8,9,18,19]
cal=np.isin(w,base_words+extra_words)&(r>=6)
query=np.isin(w,range(24,30))&(r<4)
assert len(set(ids))==3200 and cal.sum()==160 and query.sum()==240
rows=[]
for model in ["acc","previous_audio"]:
    model_root=root/"full_run_10x40_raw_acc"/"hard_relation_teacher"
    net.load_state_dict(torch.load(model_root/f"{model}_student_private.pt",
                                   map_location="cpu",weights_only=True))
    with torch.inference_mode():
        view=normalize(net.eval()(x).numpy())
    for width in [4,8]:
        reg=np.isin(w,base_words if width==4 else base_words+extra_words)&(r<(6 if width==4 else 3))
        assert reg.sum()==240
        assert (fit.astype(int)+reg+cal+query).max()==1
        for target in range(1,11):
            y=u==target
            svm=SVC(C=10,class_weight="balanced").fit(view[reg],y[reg])
            f,t,h=roc_curve(y[cal],svm.decision_function(view[cal]),drop_intermediate=False)
            j=min(range(len(h)),key=lambda i:(abs(f[i]+t[i]-1),f[i]))
            tau=h[j]
            score=svm.decision_function(view[query])
            labels=y[query]
            far=np.mean(score[~labels]>=tau)
            frr=np.mean(score[labels]<tau)
            f,t,_=roc_curve(labels,score,drop_intermediate=False)
            j=np.argmin(abs(f+t-1))
            rows.append(dict(model=model,words=width,user=target,eer=(f[j]+1-t[j])/2,
                             far=far,frr=frr,hter=(far+frr)/2))
        rr=[z for z in rows if z["model"]==model and z["words"]==width and z["user"]!=10]
        print(model,width,[round(100*np.mean([z[k] for z in rr]),2) for k in ["eer","hter","far","frr"]],flush=True)
(root/"teacher_enrollment_coverage_public.json").write_text(json.dumps(rows,indent=2))

