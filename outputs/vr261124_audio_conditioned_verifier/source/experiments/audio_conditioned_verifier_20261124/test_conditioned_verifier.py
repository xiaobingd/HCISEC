import importlib.util
from pathlib import Path
import numpy as np
import torch

spec=importlib.util.spec_from_file_location('gate',Path(__file__).with_name('run_conditioned_verifier.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
torch.set_num_threads(2);torch.manual_seed(42)
p=torch.nn.functional.normalize(torch.randn(9,192),dim=1)
a=torch.nn.functional.normalize(torch.randn(9,192),dim=1);d=torch.rand(9,192)*.001
q=torch.nn.functional.normalize(torch.randn(9,8,192),dim=2)
net=m.Gate();w=net(p,a,d)
assert torch.equal(w,torch.ones_like(w))
assert torch.allclose(m.weighted_cosine(q,p,w),(q*p[:,None,:]).sum(2),atol=1e-6)
loss=m.weighted_cosine(q,p,w).square().mean();loss.backward()
assert net.net[-1].weight.grad.abs().sum()>0
assert p.grad is None and a.grad is None and q.grad is None
for parameter in net.parameters():
    if parameter.grad is not None:parameter.data-=.001*parameter.grad
assert net(p,a,d).min()>=.75 and net(p,a,d).max()<=1.25
permutation=np.roll(np.arange(9),1)
assert np.all(permutation!=np.arange(9))
mu=np.arange(192)/192;sd=np.ones(192)*.05
noise=m.matched_noise((40,9,192),mu,sd,np.random.default_rng(42))
assert np.allclose(noise.mean((0,1)),mu,atol=1e-6)
assert np.allclose(noise.std((0,1)),sd,atol=1e-6)
print('PASS:identity gate,weightedcos,gate-only gradients,bounds,other-user derangement')
