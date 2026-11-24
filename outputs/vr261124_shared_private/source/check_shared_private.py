"""Synthetic checks: loss gradient routing, initialization, reconstruction, role isolation."""
import torch
from torch.nn import functional as F
from train_shared_private import Split,encoder,decorrelation
from train_audio_multiword_fitonly import relation
from run_strict_within_content import role_check,splits

torch.set_num_threads(2);torch.manual_seed(42)
model=Split(encoder().state_dict());x=torch.randn(16,762)
h,s,p,f,recon=model(x)
assert s.shape==p.shape==(16,96) and f.shape==(16,192)
assert torch.allclose(recon,h,atol=1e-6)
y=torch.arange(4).repeat_interleave(4);w=torch.tensor([0,0,1,1]*4)
t=F.normalize(torch.randn(16,192),dim=1)
loss=relation(s,t,y,w);loss.backward()
assert sum(v.grad.square().sum() for v in model.encoder.parameters())>0
assert model.shared.weight.grad.abs().sum()>0
assert model.private.weight.grad is None and model.decoder.weight.grad is None
assert torch.isfinite(decorrelation(s,p))
assert len(splits([1,2,3,4,5]))==20
role_check([9],[1],[2],[3,4,5])
try:role_check([9],[1],[1],[3,4,5]);raise AssertionError('leak accepted')
except ValueError:pass
print('PASS: Audio reaches ACC encoder/shared head; no direct private/decoder gradient; disjoint roles enforced')
