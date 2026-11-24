import torch
from train_crossword_mechanism import Network,crossword_supcon

torch.set_num_threads(2);torch.manual_seed(42)
y=torch.arange(4).repeat_interleave(4);w=torch.tensor([0,0,1,1]*4)
z=torch.nn.functional.normalize(torch.randn(16,192),dim=1).requires_grad_()
loss=crossword_supcon(z,y,w);loss.backward();assert torch.isfinite(loss) and torch.isfinite(z.grad).all()
perm=torch.randperm(16)
assert torch.allclose(loss,crossword_supcon(z[perm],y[perm],w[perm]),atol=1e-5)
# A batch lacking cross-word identity positives must fail instead of treating repeats as negatives.
try:crossword_supcon(z,y,torch.zeros_like(w));raise RuntimeError('Invalid batch accepted')
except AssertionError:pass
model=Network();identity,local=model(torch.randn(8,147,146));local.square().mean().backward()
assert model.identity.weight.grad is None
assert sum(p.grad.square().sum() for p in model.body.parameters())>0
print('PASS: finite/permutation invariant crossword loss;missingpositive rejection;local gradient separation')
