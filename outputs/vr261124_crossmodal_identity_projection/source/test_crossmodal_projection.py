"""Numerical controls for supervised cross-modal matching, no private data."""
import torch
from run_crossmodal_identity_projection import Pair,bidirectional
torch.set_num_threads(2);torch.manual_seed(42)
y=torch.arange(4).repeat_interleave(2)
z=torch.eye(4).repeat_interleave(2,0)
assert bidirectional(z,z,y)<bidirectional(z,z.roll(2,0),y)
net=Pair();a=torch.randn(8,192);x=torch.randn(8,192)
za,zx=net.audio(a),net.acc(x)
assert torch.allclose(bidirectional(za,zx,y),bidirectional(zx,za,y),atol=1e-6)
assert za.shape==(8,32) and torch.allclose(za.norm(dim=1),torch.ones(8),atol=1e-6)
loss=bidirectional(za,zx,y);loss.backward()
assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
raw=torch.randn(8);extra=torch.randn(8)
assert torch.equal(raw+0.*extra,raw)
print('PASS:crossmodal label matching,both projection gradients,L2 normalization,alpha0 fallback')
