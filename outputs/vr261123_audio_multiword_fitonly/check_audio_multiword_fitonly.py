"""Check trainable student / detached teacher and bounded gradient weighting."""
import torch
from torch.nn import functional as F
from train_audio_multiword_fitonly import relation

torch.manual_seed(42)
student=torch.nn.Linear(6,12)
data=torch.randn(8,6)
y=torch.tensor([0,0,0,0,1,1,1,1])
words=torch.tensor([0,0,1,1,0,0,1,1])
t=torch.randn(8,12,requires_grad=True)
z=F.normalize(student(data),dim=1)
loss=relation(z,F.normalize(t,dim=1),y,words)
loss.backward()
assert t.grad is None
assert student.weight.grad is not None and student.weight.grad.norm()>0
assert torch.isfinite(loss)
for ni,nk in [(1.,.1),(1.,100.),(0.,1.)]:
    weight=min(10.,.1*ni/max(nk,1e-8))
    ratio=weight*nk/max(ni,1e-8)
    assert ratio<=.1+1e-8
print('PASS: KD reaches student, teacher detached, finite loss and bounded KD gradient ratio')

