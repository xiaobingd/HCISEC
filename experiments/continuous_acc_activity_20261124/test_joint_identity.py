import numpy as np
import torch
from run_joint_identity import JointNet,crossword_supcon
torch.set_num_threads(2);torch.manual_seed(42)
net=JointNet();x=torch.randn(32,15,146);raw=torch.randn(32,16);valid=torch.ones(32,146,dtype=torch.bool)
y=torch.arange(8).repeat_interleave(4);words=torch.tensor([0,0,1,1]*8)
z,logit,weight=net(x,raw,valid)
assert z.shape==(32,32) and torch.allclose(z.norm(dim=1),torch.ones(32),atol=1e-6)
assert torch.allclose(weight.sum(1),torch.ones(32),atol=1e-6) and (weight>0).all()
identity=torch.nn.functional.cross_entropy(16*torch.nn.functional.linear(z,torch.nn.functional.normalize(net.classifier.weight,dim=1)),y)+.1*crossword_supcon(z,y,words)
aux=torch.nn.functional.binary_cross_entropy_with_logits(logit,torch.randint(0,2,logit.shape).float())
gi=torch.autograd.grad(identity,[*net.body.parameters(),net.time.weight],retain_graph=True)
ga=torch.autograd.grad(aux,[*net.body.parameters(),net.time.weight],retain_graph=True)
assert all(torch.isfinite(v).all() for v in gi+ga)
assert sum(float(v.abs().sum()) for v in gi[:-1])>0
# Uniform pooling starts with a zero time kernel. Audio first updates that
# kernel, and reaches body features on subsequent joint steps.
assert sum(float(v.abs().sum()) for v in ga[:-1])==0
assert float(gi[-1].abs().sum())>0 and float(ga[-1].abs().sum())>0
projection_aux=torch.autograd.grad(aux,net.project.weight,allow_unused=True,retain_graph=True)[0];assert projection_aux is None
optimizer=torch.optim.SGD(net.parameters(),lr=.01);optimizer.zero_grad();(identity+.1*aux).backward();optimizer.step()
_,logit2,_=net(x,raw,valid);aux2=torch.nn.functional.binary_cross_entropy_with_logits(logit2,torch.randint(0,2,logit2.shape).float())
later=torch.autograd.grad(aux2,net.body.parameters());assert sum(float(v.abs().sum()) for v in later)>0
print('PASS:identity+Audio reachsharedencoder andtimeweights;L2embedding;softfullsupport;auxnotembeddingregression')
