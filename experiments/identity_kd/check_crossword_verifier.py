"""Meaningful invariance, gradient and role isolation checks before training."""
import json
import torch
import numpy as np
from crossword_neural_verifier import ACCEncoder, Verifier, objective
from run_crossword_neural_verifier import Tasks

torch.set_num_threads(2);torch.manual_seed(42)
encoder=ACCEncoder().eval();net=Verifier().eval()
with torch.no_grad():
    q=encoder(torch.randn(2,3,40,28))
    r=encoder(torch.randn(8,3,40,28)).reshape(2,4,28,128)
base=torch.tensor([-.3,.4]);labels=torch.tensor([0.,1.])
s,delta=net(q,r,base)
assert torch.equal(s,base), 'zero-init must preserve baseline exactly'
opt=torch.optim.Adam(net.parameters(),lr=.001)
loss=objective(s,delta,labels,torch.tensor([-2.,2.]))
opt.zero_grad();loss.backward();opt.step()
assert net.head[-1].weight.abs().sum()>0
opt.zero_grad()
s,delta=net(q,r,base);objective(s,delta,labels).backward()
assert net.attention.in_proj_weight.grad.abs().sum()>0
with torch.no_grad():
    original=net(q,r,base)[0]
    shuffled=net(q,r[:,[2,0,3,1]],base)[0]
assert torch.allclose(original,shuffled,atol=1e-6)
d=np.load('outputs/mei_common_frontend_private.npz');tasks=Tasks(d['user'],d['word'])
t=tasks.sample(list(range(18)),360,42)
assert np.all(d['word'][t['q']]!=d['word'][t['r'][:,0]])
assert set(t['owner'])==set(range(1,10))
assert all(set(d['user'][t['q'][(t['owner']==u)&~t['y']]])==set(range(1,10))-{u} for u in range(1,10))
print(json.dumps(dict(state='PASS',baseline_preserved=True,enrollment_permutation_invariant=True,
    attention_gradient_nonzero=True,crossword_tasks=True,all_attack_sources=True)))
