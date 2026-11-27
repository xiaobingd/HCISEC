import numpy as np
import torch
from conditional_loss import targets,loss

torch.set_num_threads(2)
q=np.array([[[.8,.2,0.],[.2,0.,.8],[1.,0.,0.],[.9999,.0001,0.]]],dtype='float32')
valid=np.ones((1,4),bool);t,w,h=targets(q,valid)
assert np.allclose(t[0,:2],[[1,0],[0,1]]) and np.allclose(w[0,:2],[.2,.8]) and np.array_equal(w[0,2:],[0,0])
z=torch.tensor([[[0.,2.,-1.],[0.,1.,3.],[0.,8.,1.],[0.,-3.,9.]]],requires_grad=True)
actual=loss(z,torch.tensor(t),torch.tensor(w),torch.tensor(h));lp=z[...,1:].log_softmax(-1)
expected=-.2*lp[0,0,0]-.8*lp[0,1,1]
assert torch.allclose(actual,expected,atol=1e-6)
g=torch.autograd.grad(actual,z)[0];assert torch.equal(g[...,0],torch.zeros_like(g[...,0])) and torch.equal(g[0,2:],torch.zeros_like(g[0,2:]))
changed=z.detach().clone();changed[...,0]=1000
assert torch.allclose(actual,loss(changed,torch.tensor(t),torch.tensor(w),torch.tensor(h)))
blank=np.zeros((2,5,4),dtype='float32');blank[...,0]=1;tt,ww,hh=targets(blank,np.ones((2,5),bool));zz=torch.randn(2,5,4,requires_grad=True)
zero=loss(zz,torch.tensor(tt),torch.tensor(ww),torch.tensor(hh));assert zero.item()==0 and torch.isfinite(zero)
zero.backward();assert torch.equal(zz.grad,torch.zeros_like(zz.grad))
permuted=q[:,[1,0,3,2]];tp,wp,hp=targets(permuted,valid)
assert np.allclose(tp,t[:,[1,0,3,2]]) and np.allclose(wp,w[:,[1,0,3,2]]) and np.isclose(w.sum(),wp.sum())
print('PASS:analyticweightedKL,teacherandstudentconditioning,blanklogitinvariance,zeroblankgradient,lowmassmask,allblankfinitezero,jointposteriorweightshuffle')
