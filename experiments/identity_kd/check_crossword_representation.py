"""Check real backbone gradients and a genuinely destructive control."""
import json
import torch
from torch.nn import functional as F
from crossword_representation_losses import IdentityEncoder,crossword_supcon,relation_kl

torch.set_num_threads(2);torch.manual_seed(42)
y=torch.arange(4).repeat_interleave(4)
w=torch.tensor([0,0,1,1]*4)
model=IdentityEncoder();x=torch.randn(16,3,40,28)
teacher=torch.randn(16,192,requires_grad=True)
z=model(x);loss=crossword_supcon(z,y,w)+relation_kl(z,teacher,y,w)
loss.backward()
assert model.backbone.network[0].weight.grad.abs().sum()>0
assert teacher.grad is None
perfect=F.one_hot(y,4).float()
wrong=perfect.clone()
wrong[w==0]=F.one_hot((y[w==0]+1)%4,4).float()
wrong[w==1]=F.one_hot((y[w==1]+2)%4,4).float()
assert relation_kl(perfect,perfect,y,w).abs()<1e-6
assert relation_kl(perfect,wrong,y,w)>1e-3
before=model.backbone.network[0].weight.detach().clone()
optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4);optimizer.step()
assert not torch.equal(before,model.backbone.network[0].weight)
print(json.dumps(dict(state='PASS',encoder_gradient_nonzero=True,encoder_parameters_update=True,
    teacher_detached=True,wrong_identity_relation_changes=True,matched_relation_zero=True)))
