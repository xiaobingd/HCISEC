import torch
from train_selected_mechanism import Network,crossword_supcon

torch.set_num_threads(2);torch.manual_seed(42)
n=Network();z,p=n(torch.randn(8,147,146));loss=torch.nn.functional.smooth_l1_loss(p[:,2:],torch.randn_like(p[:,2:]));loss.backward()
assert n.identity.weight.grad is None
assert torch.equal(n.local.weight.grad[:2],torch.zeros_like(n.local.weight.grad[:2]))
assert n.local.weight.grad[2:].abs().sum()>0
assert sum(v.grad.square().sum() for v in n.body.parameters())>0
print('PASS:selected24bins supervise sharedbody;excluded50/60receivezero localgradient;identityprojection unaffecteddirectly')
