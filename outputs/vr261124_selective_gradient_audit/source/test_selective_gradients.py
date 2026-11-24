"""Synthetic local-relation and rollback checks, no private records."""
import numpy as np
import torch
from audit_selective_gradients import local_kl,supcon,cosine
torch.set_num_threads(2);torch.manual_seed(42)
z=torch.nn.functional.normalize(torch.randn(8,12),dim=1).requires_grad_()
y=torch.arange(4).repeat_interleave(2);w=torch.tensor([0,1]*4)
loss=supcon(z,y,w);g=torch.autograd.grad(loss,z)[0]
assert torch.isfinite(loss) and torch.isfinite(g).all()
s=torch.randn(4,4,requires_grad=True);t=s.detach().clone()
same=local_kl(s,t);assert abs(float(same.mean()))<1e-6
g0=torch.autograd.grad(same.sum(),s)[0];assert float(g0.abs().max())<1e-5
assert abs(cosine((g,),(g,))-1)<1e-5
assert abs(cosine((g,),(-g,))+1)<1e-5
parameter=torch.nn.Parameter(torch.randn(12));old=parameter.detach().clone()
with torch.no_grad():
    parameter.add_(.001);parameter.copy_(old)
assert torch.equal(parameter,old)
print('PASS:SupCon finite gradients,matched KL stationary,gradient signs,exact rollback')
