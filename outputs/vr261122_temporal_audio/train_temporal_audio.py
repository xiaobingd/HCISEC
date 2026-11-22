"""Controlled CPU pilot. Private arrays stay on the experiment instance."""
import json
from pathlib import Path
import numpy as np
from scipy import signal
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import roc_curve

root=Path("/root")
out=root/"full_run_10x40_raw_acc"/"temporal_audio"
out.mkdir(exist_ok=True)
torch.set_num_threads(4)
d=np.load(root/"paired_raw_acc_private.npz")
u,w,r=np.array([s.split("_") for s in d["sample_id"]],dtype=int).T
fit=d["role"]=="fit"
reg=np.isin(w,[20,21,22,23,8,9,18,19])&(r<3)
cal=np.isin(w,[20,21,22,23,8,9,18,19])&(r>=6)
query=np.isin(w,range(24,30))&(r<4)
assert (fit.astype(int)+reg+cal+query).max()==1
raw=np.load(root/"stats27_private.npz")
assert np.array_equal(raw["sample_id"],d["sample_id"])
spec=d["spectrum"].reshape(len(u),120,28)
stats=np.c_[d["amplitude"],raw["features"]]
sc=StandardScaler().fit(stats[fit])
side=torch.tensor(sc.transform(stats),dtype=torch.float32)
mean=spec[fit].mean((0,2),keepdims=True)
std=spec[fit].std((0,2),keepdims=True)
x=torch.tensor((spec-mean)/np.maximum(std,1e-5),dtype=torch.float32)
flat=np.column_stack([spec.mean(-1),spec.std(-1),*(v.mean(-1) for v in np.array_split(spec,4,axis=-1)),stats])
flat=torch.tensor(StandardScaler().fit(flat[fit]).transform(flat),dtype=torch.float32)
y=torch.tensor(u-1)
cache=np.load(root/"full_run_10x40_raw_acc"/"adapted_audio"/"teacher_targets_private.npz")
assert len(set(cache["indices"]))==len(cache["indices"])
assert np.all(fit[cache["indices"]]|reg[cache["indices"]])
teacher=torch.zeros(len(u),192)
available=np.zeros(len(u),bool)
teacher[cache["indices"]]=torch.tensor(cache["targets"])
available[cache["indices"]]=True
available &= fit
# Access training Audio only for auxiliary targets.
waves=d["wave"][fit].astype("float32")
targets=np.zeros((len(u),41,28),dtype="float32")
freqs=np.geomspace(10,300,40)
for index,wave in zip(np.flatnonzero(fit),waves):
    f,t,z=signal.stft(wave,fs=16000,nperseg=4096,noverlap=3072,boundary=None,padded=False)
    power=abs(z)**2
    bands=np.stack([np.interp(freqs,f,power[:,i]) for i in range(len(t))],axis=1)
    bands=np.log(bands+1e-12)
    env=np.log(power.sum(0,keepdims=True)+1e-12)
    targets[index]=np.stack([np.interp(np.linspace(0,len(t)-1,28),np.arange(len(t)),v)
                              for v in np.r_[bands,env]])
am=targets[fit].mean((0,2),keepdims=True)
asd=targets[fit].std((0,2),keepdims=True)
targets=torch.tensor((targets-am)/np.maximum(asd,1e-5))
groups=np.stack([[np.flatnonzero(fit&(u==owner)&(w==word))
                  for word in sorted(set(w[fit]))] for owner in range(1,11)])

class Student(nn.Module):
    def __init__(self,temporal):
        super().__init__()
        self.temporal=temporal
        if temporal:
            self.body=nn.Sequential(nn.Conv1d(120,64,3,padding=1),nn.GroupNorm(8,64),nn.GELU(),
                                    nn.Conv1d(64,64,3,padding=1),nn.GroupNorm(8,64),nn.GELU())
            self.project=nn.Linear(128+side.shape[1],192)
            self.aux=nn.Conv1d(64,41,1)
        else:
            self.body=nn.Sequential(nn.Linear(flat.shape[1],256),nn.LayerNorm(256),nn.GELU(),nn.Linear(256,192))
    def forward(self,indices):
        if not self.temporal:
            return F.normalize(self.body(flat[indices]),dim=1),None
        h=self.body(x[indices])
        z=self.project(torch.cat([h.mean(-1),h.std(-1),side[indices]],dim=1))
        return F.normalize(z,dim=1),self.aux(h)

rows=[]
for seed in (42,43,44):
    for method in ("A_stats","B_temporal","C_audio_temporal"):
        torch.manual_seed(seed)
        rng=np.random.default_rng(seed)
        net=Student(method!="A_stats")
        head=nn.Linear(192,10,bias=False)
        opt=torch.optim.AdamW([*net.parameters(),*head.parameters()],lr=0.001)
        for step in range(400):
            words=rng.choice(groups.shape[1],4,replace=False)
            batch=groups[:,words,:]
            idx=np.take_along_axis(batch,rng.integers(8,size=(10,4,1)),axis=2).ravel()
            z,pred=net(idx)
            labels=y[idx]
            loss=F.cross_entropy(16*F.linear(z,F.normalize(head.weight,dim=1)),labels)
            if method=="C_audio_temporal":
                # Coarse time pooling tolerates approximate alignment.
                aux=F.smooth_l1_loss(F.adaptive_avg_pool1d(pred,7),
                                     F.adaptive_avg_pool1d(targets[idx],7))
                keep=available[idx]
                kd=z.sum()*0
                if keep.sum()>=3:
                    a=z[keep]@z[keep].T
                    b=teacher[idx[keep]]@teacher[idx[keep]].T
                    mask=~torch.eye(len(a),dtype=torch.bool)
                    kd=F.mse_loss(a[mask],b[mask])
                loss=loss+.1*aux+.1*kd
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_([*net.parameters(),*head.parameters()],1)
            opt.step()
        with torch.inference_mode():
            emb=net.eval()(np.arange(len(u)))[0].numpy()
        for target in range(1,11):
            labels=u==target
            svm=SVC(C=10,class_weight="balanced").fit(emb[reg],labels[reg])
            f,t,h=roc_curve(labels[cal],svm.decision_function(emb[cal]),drop_intermediate=False)
            tau=h[min(range(len(h)),key=lambda i:(abs(f[i]+t[i]-1),f[i]))]
            s=svm.decision_function(emb[query])
            fq,tq,_=roc_curve(labels[query],s,drop_intermediate=False)
            j=np.argmin(abs(fq+tq-1))
            far=np.mean(s[~labels[query]]>=tau)
            frr=np.mean(s[labels[query]]<tau)
            rows.append(dict(seed=seed,method=method,user=target,eer=(fq[j]+1-tq[j])/2,
                             hter=(far+frr)/2,far=far,frr=frr))
        torch.save(net.state_dict(),out/f"{method}_{seed}_private.pt")
        (out/"results_public.json").write_text(json.dumps(rows,indent=2))
        rr=[v for v in rows if v["seed"]==seed and v["method"]==method and v["user"]!=10]
        print(seed,method,[round(100*np.mean([v[k] for v in rr]),2) for k in ("eer","hter","far","frr")],flush=True)

