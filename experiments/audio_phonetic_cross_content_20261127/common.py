"""Frozen hierarchical phonetic experiment primitives."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class Reverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,coefficient):ctx.coefficient=coefficient;return x.view_as(x)
    @staticmethod
    def backward(ctx,gradient):return -ctx.coefficient*gradient,None

class Student(nn.Module):
    def __init__(self,vocab_size=392,words=16):
        super().__init__()
        self.backbone=nn.Sequential(nn.Conv1d(147,64,5,padding=2),nn.GroupNorm(8,64),nn.GELU(),
          nn.Conv1d(64,64,3,padding=2,dilation=2),nn.GELU(),nn.Conv1d(64,64,3,padding=3,dilation=3),nn.GELU())
        self.phone_head=nn.Linear(64,vocab_size)
        self.attention=nn.Conv1d(64,1,1)
        self.identity=nn.Linear(128,192)
        self.classifier=nn.Linear(192,10,bias=False)
        self.word_head=nn.Sequential(nn.Linear(192,64),nn.GELU(),nn.Linear(64,words))
    def forward(self,x,valid,auxiliary=True):
        h=self.backbone(x);logit=self.attention(h).squeeze(1);weight=logit.masked_fill(~valid,-1e4).softmax(1)*valid.float();weight/=weight.sum(1,keepdim=True).clamp_min(1e-12)
        mean=(h*weight[:,None,:]).sum(2);std=(((h-mean[:,:,None]).square()*weight[:,None,:]).sum(2)+1e-6).sqrt()
        z=F.normalize(self.identity(torch.cat([mean,std],1)),dim=1)
        if not auxiliary:return z,None,None
        return z,self.phone_head(h.transpose(1,2)),self.word_head(Reverse.apply(z,1.))

def crossword_supcon(z,user,word):
    same=user[:,None]==user[None,:];positive=same&(word[:,None]!=word[None,:]);allowed=(~same)|positive
    good=positive.any(1)
    if not good.any():return z.sum()*0
    logits=z@z.T/.1;logp=logits-torch.logsumexp(logits.masked_fill(~allowed,-1e4),1,keepdim=True)
    return -((logp*positive).sum(1)/positive.sum(1).clamp_min(1))[good].mean()

def phone_kl(logits,target,valid,target_logsum=None):
    logp=F.log_softmax(logits,dim=-1)
    # Frozen target entropy may be cached once; exactly the same full-vocabulary KL.
    # This avoids recomputing millions of constant xlogy terms on CPU every step.
    per=F.kl_div(logp,target,reduction='none').sum(-1) if target_logsum is None else target_logsum-(target*logp).sum(-1)
    return (per*valid).sum()/valid.sum().clamp_min(1)

def ctc_posterior(logprob,tokens,blank=0):
    """Exact CTC forward/backward conditioned on an explicit phoneme token sequence."""
    logprob=np.asarray(logprob,dtype='float64');tokens=np.asarray(tokens,dtype=int)
    T,V=logprob.shape;ext=np.full(2*len(tokens)+1,blank,dtype=int);ext[1::2]=tokens;S=len(ext)
    assert T>0 and np.all(tokens!=blank) and np.all((tokens>=0)&(tokens<V))
    emit=logprob[:,ext];alpha=np.full((T,S),-np.inf);beta=alpha.copy();alpha[0,0]=emit[0,0]
    if S>1:alpha[0,1]=emit[0,1]
    skip=np.zeros(S,bool)
    if S>2:skip[2:]=(ext[2:]!=blank)&(ext[2:]!=ext[:-2])
    for t in range(1,T):
        a=alpha[t-1];previous=np.r_[-np.inf,a[:-1]];jump=np.full(S,-np.inf)
        if S>2:jump[2:]=a[:-2]
        alpha[t]=np.logaddexp(np.logaddexp(a,previous),np.where(skip,jump,-np.inf))+emit[t]
    logZ=alpha[-1,-1] if S==1 else np.logaddexp(alpha[-1,-1],alpha[-1,-2])
    if not np.isfinite(logZ):raise ValueError('CTC alignment impossible')
    beta[-1,-1]=0
    if S>1:beta[-1,-2]=0
    source_skip=np.zeros(S,bool)
    if S>2:source_skip[:-2]=skip[2:]
    for t in range(T-2,-1,-1):
        stay=beta[t+1]+emit[t+1];nxt=np.r_[stay[1:],-np.inf];jump=np.full(S,-np.inf)
        if S>2:jump[:-2]=stay[2:]
        beta[t]=np.logaddexp(np.logaddexp(stay,nxt),np.where(source_skip,jump,-np.inf))
    state=np.exp(alpha+beta-logZ);posterior=np.zeros((T,V),dtype='float64')
    for s,label in enumerate(ext):posterior[:,label]+=state[:,s]
    posterior/=np.maximum(posterior.sum(1,keepdims=True),1e-300)
    assert np.isfinite(posterior).all() and np.allclose(posterior.sum(1),1,atol=1e-8)
    return posterior.astype('float32'),float(logZ)

def smooth_to_acc(audio_time,posterior,acc_time,sigma=.1):
    audio_time=np.asarray(audio_time);acc_time=np.asarray(acc_time);difference=acc_time[:,None]-audio_time[None,:]
    weights=np.exp(-.5*(difference/sigma)**2)*(abs(difference)<=3*sigma)
    valid=(acc_time>=audio_time[0])&(acc_time<=audio_time[-1])&(weights.sum(1)>1e-12)
    q=weights@posterior/np.maximum(weights.sum(1,keepdims=True),1e-20)
    q[~valid]=0;q[~valid,0]=1
    return q.astype('float32'),valid

def shifted_posteriors(target,valid,seed):
    out=target.copy();permutation=np.tile(np.arange(target.shape[1]),(target.shape[0],1));rng=np.random.default_rng(seed+700)
    changed=[]
    for i in range(len(target)):
        idx=np.flatnonzero(valid[i])
        if len(idx)>1:
            shift=int(rng.integers(max(1,len(idx)//4),max(2,3*len(idx)//4)))
            donor=np.roll(idx,shift);out[i,idx]=target[i,donor];permutation[i,idx]=donor
            changed.append(float(np.mean(abs(out[i,idx]-target[i,idx]).sum(1))))
        else:changed.append(0.)
    assert np.array_equal(valid,valid.copy())
    return out,permutation,changed

def template(tokens,vocab_size,blank=0):
    """Dictionary-only fixed-window pseudo sequence; no Audio timing or output."""
    time=.05+.02*np.arange(146);q=np.zeros((146,vocab_size),dtype='float32');q[:,blank]=1
    if not tokens:return q,np.zeros(146,bool)
    active=(time>=.5)&(time<2.5);position=np.minimum(((time[active]-.5)/2*len(tokens)).astype(int),len(tokens)-1)
    q[active]=0;q[np.flatnonzero(active),np.asarray(tokens)[position]]=1
    q,valid=smooth_to_acc(time,q,time,.1)
    return q,valid
