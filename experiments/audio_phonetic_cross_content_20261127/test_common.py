import itertools
import numpy as np
import torch
from torch.nn import functional as F
from common import Reverse,Student,ctc_posterior,smooth_to_acc,shifted_posteriors,template,phone_kl
torch.set_num_threads(2);torch.manual_seed(8)
for seq in [[1,2],[1,1],[]]:
    logits=torch.randn(4,3,dtype=torch.float64);lp=logits.log_softmax(-1).numpy();got,logZ=ctc_posterior(lp,seq)
    exact=np.zeros((4,3));prob=0.
    for path in itertools.product(range(3),repeat=4):
        collapsed=[v for i,v in enumerate(path) if (i==0 or v!=path[i-1]) and v!=0]
        if collapsed==seq:
            p=float(np.exp(sum(lp[i,v] for i,v in enumerate(path))));prob+=p
            for i,v in enumerate(path):exact[i,v]+=p
    exact/=prob
    assert np.allclose(got,exact,atol=1e-6) and np.isclose(logZ,np.log(prob))
z=torch.randn(6,8,requires_grad=True);head=torch.nn.Linear(8,4);label=torch.arange(6)%4
plain=F.cross_entropy(head(z),label);gz,gw=torch.autograd.grad(plain,[z,head.weight],retain_graph=True)
rev=F.cross_entropy(head(Reverse.apply(z,1.)),label);rz,rw=torch.autograd.grad(rev,[z,head.weight])
assert torch.allclose(rz,-gz) and torch.allclose(rw,gw),'GRL sign must affectencoderonly'
q=np.random.default_rng(42).dirichlet(np.ones(4),size=(3,10)).astype('float32');valid=np.ones((3,10),bool);valid[:,-2:]=False
wrong,perm,changed=shifted_posteriors(q,valid,42)
assert np.allclose(q[:,:8].sum(1),wrong[:,:8].sum(1)) and np.array_equal(q[:,8:],wrong[:,8:]) and max(changed)>0
t,m=template([1,2,1],4);assert t.shape==(146,4) and np.allclose(t.sum(1),1)
model=Student(392,16);model.eval();x=torch.randn(2,147,146);mask=torch.ones(2,146,dtype=torch.bool);emb,phone,word=model(x,mask)
assert emb.shape==(2,192) and phone.shape==(2,146,392) and word.shape==(2,16)
assert torch.allclose(emb,model(x,mask,auxiliary=False)[0])
target=torch.rand(3,9,392);target[:,:,20:]=0;target/=target.sum(-1,keepdim=True)
logits=torch.randn_like(target,requires_grad=True);mask=torch.rand(3,9)>.2
a=phone_kl(logits,target,mask);b=phone_kl(logits,target,mask,torch.special.xlogy(target,target).sum(-1))
ga=torch.autograd.grad(a,logits,retain_graph=True)[0];gb=torch.autograd.grad(b,logits)[0]
assert torch.allclose(a,b,atol=1e-6) and torch.allclose(ga,gb,atol=1e-8),'Cached frozen entropy must retain exact loss and gradients'
print('PASS: exactCTCposterior,GRLencoder/headsign,valid-onlyshuffle,pseudosequence,studenttensor/inference;parameters',sum(p.numel() for p in model.parameters()))
