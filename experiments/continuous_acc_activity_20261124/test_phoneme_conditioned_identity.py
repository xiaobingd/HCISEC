import numpy as np
import torch
from train_phoneme_conditioned_identity import Net,contrast

torch.set_num_threads(2);torch.manual_seed(42);net=Net();x=torch.randn(24,41,24)
y=torch.arange(4).repeat_interleave(6);word=torch.arange(6).repeat(4);phone=torch.tensor([0,0,1,1,1,2]).repeat(4)
weight=torch.randn(24,41).softmax(1);z,local=net(x,weight);z2,_=net(x,weight.roll(5,1))
assert torch.equal(z,z2),'Inference embedding must not depend on teacherweights'
loss=contrast(local,y,word,phone)
grad=torch.autograd.grad(loss,[net.encoder[0].weight,net.project.weight],retain_graph=True)
assert all(g.abs().sum()>0 for g in grad),'Local identity loss must updateencoder/projector'
attn_grad=torch.autograd.grad(loss,net.attention.weight,allow_unused=True,retain_graph=True)[0]
assert attn_grad is None or attn_grad.abs().sum()==0,'This version transfers throughsharedencoder,notdirectattention supervision'
renamed=torch.tensor([2,0,1])[phone]
assert torch.allclose(loss,contrast(local,y,word,renamed)),'Global class renaming is NOT a valid negative control'
shuffle=phone.reshape(4,6).clone()
for row in range(4):shuffle[row]=shuffle[row].roll(row+1)
assert not torch.allclose(loss,contrast(local,y,word,shuffle.ravel()))
assert torch.isfinite(contrast(local,y,word,torch.arange(24))).item()
assert sum(p.numel() for p in net.parameters())==5057
print('PASS: teacher-free inference, identitygradients reachencoder, classrenaming invariance, nontrivial shuffle, singletonhandling,5057parameters')
