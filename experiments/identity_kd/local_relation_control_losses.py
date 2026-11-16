"""Train-only stable-neighbor graphs and matched local relation objectives."""
import numpy as np
import torch
from torch.nn import functional as F

def norm(x):return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)

def graph_similarity(z):
    # user,word,repeat,dim; aggregate only different-word comparisons.
    p=norm(norm(z).mean(2));w=p.shape[1]
    s=np.einsum('uwd,vkd->wkuv',p,p)
    return s[~np.eye(w,dtype=bool)].mean(0)

def top2(s):
    s=s.copy();np.fill_diagonal(s,-np.inf);g=np.zeros_like(s,dtype=bool)
    g[np.arange(len(s))[:,None],np.argsort(-s,axis=1)[:,:2]]=True
    return g

def stable_graph(z):
    assert z.ndim==4 and z.shape[2]==6
    variants=[z,z[:,:,:3],z[:,:,3:],z[:,::2],z[:,1::2]]
    graphs=[top2(graph_similarity(a)) for a in variants]
    return np.logical_and.reduce(graphs)

def local_relation_kl(z,teacher,user,word,graph,temperature=.2):
    same=user[:,None]==user[None,:];sw=word[:,None]==word[None,:]
    positive=same&~sw
    negative=~same&sw&graph[user[:,None],user[None,:]]
    active=negative.any(1)&positive.any(1)
    if not active.any():return z.sum()*0,0
    valid=(positive|negative)[active]
    z=F.normalize(z,dim=1);t=F.normalize(teacher.detach(),dim=1)
    a=(z@z.T/temperature)[active].masked_fill(~valid,-1e4)
    b=(t@t.T/temperature)[active].masked_fill(~valid,-1e4)
    return F.kl_div(F.log_softmax(a,1),F.softmax(b,1),reduction='batchmean'),int(active.sum())

def combine_gradients(ce,kd,encoder,head,rho=.03):
    # Exact current-step encoder norm matching; no third backward pass.
    parameters=list(encoder.parameters());heads=list(head.parameters())
    a=torch.autograd.grad(ce,parameters+heads,retain_graph=True)
    b=torch.autograd.grad(kd,parameters)
    ni=torch.sqrt(sum(g.square().sum() for g in a[:len(parameters)]))
    nk=torch.sqrt(sum(g.square().sum() for g in b))
    lam=min(1.,rho*float(ni)/max(float(nk),1e-8)) if float(nk)>1e-8 else 0.
    for p,g,h in zip(parameters,a,b):p.grad=g+lam*h
    for p,g in zip(heads,a[len(parameters):]):p.grad=g
    cosine=float(sum((g*h).sum() for g,h in zip(a,b))/(ni*nk)) if float(ni*nk)>1e-12 else None
    return dict(lambda_kd=lam,identity_gradient_norm=float(ni),relation_gradient_norm=float(nk),
                weighted_gradient_ratio=lam*float(nk)/max(float(ni),1e-12),gradient_cosine=cosine)
