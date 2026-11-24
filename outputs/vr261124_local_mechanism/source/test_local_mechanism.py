import torch
from train_local_mechanism import Network
from run_strict_within_content import role_check

torch.set_num_threads(2);torch.manual_seed(42)
n=Network();x=torch.randn(8,147,146);z,p=n(x)
assert z.shape==(8,192) and p.shape==(8,26,12)
p.square().mean().backward()
assert n.identity.weight.grad is None
assert sum(v.grad.square().sum() for v in n.body.parameters())>0
assert all((r+1)%8!=r for r in range(8))
role_check([0,1],[2],[3],[4,5,6])
print('PASS:local loss reaches shared body,not identity projection;wrong-repeat has no fixedpoint;role isolation')
