"""Strictly isolated hardware ACC / Audio front end for the V1 CPU pilot."""
from pathlib import Path
import numpy as np
import torch
from scipy import signal
from torch.nn import functional as F

root=Path("/root")
out=root/"full_run_10x40_raw_acc"/"diffusion_baseline"
out.mkdir(exist_ok=True)
torch.set_num_threads(2)
d=np.load(root/"paired_raw_acc_private.npz")
u,w,r=np.array([s.split("_") for s in d["sample_id"]],dtype=int).T
fit=d["role"]=="fit"
held=np.random.default_rng(1123).choice(np.unique(w[fit]),6,replace=False)
train=fit&~np.isin(w,held)
valid=fit&np.isin(w,held)
reg=np.isin(w,[20,21,22,23,8,9,18,19])&(r<3)
cal=np.isin(w,[20,21,22,23,8,9,18,19])&(r>=6)
same=np.isin(w,[20,21,22,23,8,9,18,19])&np.isin(r,[3,4,5])
cross=np.isin(w,range(24,30))&(r<4)
assert (train.astype(int)+valid+reg+cal+same+cross).max()==1
spec=torch.tensor(d["spectrum"])
acc=torch.logsumexp(spec,dim=1,keepdim=True)
acc=F.interpolate(acc,size=(80,80),mode="bilinear",align_corners=False).numpy()
audio=np.zeros_like(acc)
wave=d["wave"][fit].astype("float32")
for i,x in zip(np.flatnonzero(fit),wave):
    f,_,z=signal.stft(x-x.mean(),fs=16000,nperseg=4096,noverlap=3072,nfft=16384,boundary=None,padded=False)
    lp=np.log(abs(z)**2+1e-20)
    b=np.stack([np.interp(np.geomspace(50,7600,80),f,col) for col in lp.T],axis=1)
    audio[i,0]=np.stack([np.interp(np.linspace(0,b.shape[1]-1,80),np.arange(b.shape[1]),v) for v in b])
am,asd=float(acc[train].mean()),float(acc[train].std())
bm,bsd=float(audio[train].mean()),float(audio[train].std())
acc=np.clip((acc-am)/(3*asd),-1,1)
audio[fit]=np.clip((audio[fit]-bm)/(3*bsd),-1,1)
np.savez_compressed(out/"frontend_private.npz",acc=acc,audio=audio,sample_id=d["sample_id"],
                    user=u,word=w,repeat=r,train=train,valid=valid,register=reg,
                    calibration=cal,sameword=same,crossword=cross)

