"""Conditional nonblank KL: both modalities exclude blank; detached confidence/mass weights."""
import numpy as np
import torch
from torch.nn import functional as F
from scipy.special import xlogy

MASS_FLOOR=.001

def targets(posterior,valid):
    q=posterior[...,1:].copy();mass=q.sum(-1)
    np.divide(q,mass[...,None],out=q,where=mass[...,None]>0)
    confidence=q.max(-1)
    weight=mass*confidence*valid*(mass>=MASS_FLOOR)
    entropy_term=xlogy(q,q).sum(-1)
    assert np.isfinite(q).all() and np.isfinite(weight).all() and (weight>=0).all()
    return q.astype('float32'),weight.astype('float32'),entropy_term.astype('float32')

def loss(logits,target,weight,target_logsum):
    # softmax(nonblank logits) equals conditioning the student's full softmax on nonblank,
    # and stays stable even if its blank probability rounds to1.
    logp=F.log_softmax(logits[...,1:],dim=-1)
    per=target_logsum-(target*logp).sum(-1)
    return (per*weight).sum()/weight.sum().clamp_min(1e-12)
