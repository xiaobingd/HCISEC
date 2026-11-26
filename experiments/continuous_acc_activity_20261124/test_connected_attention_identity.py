import torch
from train_connected_attention_identity import Net,OldNet,contrast
torch.set_num_threads(2);torch.manual_seed(42)
x=torch.randn(24,41,24);teacher=torch.randn(24,41).softmax(1);y=torch.arange(4).repeat_interleave(6);word=torch.arange(6).repeat(4);phone=torch.tensor([0,0,1,1,1,2]).repeat(4)
fixed=Net(False);connected=Net(True);connected.load_state_dict(fixed.state_dict());old=OldNet();old.load_state_dict(fixed.state_dict())
zf,lf=fixed(x,teacher);zc,lc=connected(x,teacher);zo,lo=old(x,teacher)
assert torch.equal(zf,zc) and torch.equal(zo,zf) and torch.equal(lf,lo)
assert torch.allclose(lf,lc,atol=1e-6),'At initialuniform attention,local views mustmatch'
gf=torch.autograd.grad(contrast(lf,y,word,phone),fixed.attention.weight,allow_unused=True)[0]
gc=torch.autograd.grad(contrast(lc,y,word,phone),connected.attention.weight,retain_graph=True)[0]
assert gf is None and gc.abs().sum()>0,'Onlyconnectedlocalidentityloss mustupdatequeryattention'
assert torch.equal(connected(x)[0],connected(x,teacher.roll(7,1))[0]),'Queryembedding mustnot useprior'
before=connected.attention.weight.detach().clone();opt=torch.optim.SGD(connected.parameters(),lr=.1);opt.zero_grad();contrast(lc,y,word,phone).backward();opt.step()
assert not torch.equal(before,connected.attention.weight)
assert sum(p.numel() for p in connected.parameters())==5057
print('PASS: matched5057parameters/initialviews;localgradientandupdatereachqueryattention;detachedcontrolzerograd;querypriorindependent')
