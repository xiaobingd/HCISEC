"""Synthetic graph, label-mismatch and live-gradient checks."""
import numpy as np
import torch
from torch.nn import functional as F
from local_relation_control_losses import stable_graph,local_relation_kl,combine_gradients

def main():
    torch.set_num_threads(2);rng=np.random.RandomState(9)
    latent=rng.randn(4,1,1,8);data=latent+.001*rng.randn(4,12,6,8)
    graph=stable_graph(data);assert graph.shape==(4,4) and (graph.sum(1)==2).all()
    torch.manual_seed(9);net=torch.nn.Linear(5,8);head=torch.nn.Linear(8,4)
    user=torch.tensor(np.repeat(np.arange(4),4),dtype=torch.long);word=torch.tensor(np.tile(np.repeat([0,1],2),4),dtype=torch.long)
    x=torch.randn(16,5);z=F.normalize(net(x),dim=1);teacher=z.detach().clone().requires_grad_(True)
    same,_=local_relation_kl(z,teacher,user,word,torch.tensor(graph));assert abs(float(same))<1e-6
    permutation=np.arange(16)
    for i in range(16):permutation[i]=((int(user[i])+(1 if int(word[i])==0 else 2))%4)*4+i%4
    wrong,_=local_relation_kl(z,teacher[permutation],user,word,torch.tensor(graph));assert float(wrong)>1e-3
    ce=F.cross_entropy(head(z),user);before=net.weight.detach().clone()
    opt=torch.optim.SGD(list(net.parameters())+list(head.parameters()),lr=.1)
    enc=list(net.parameters());heads=list(head.parameters())
    ga=torch.autograd.grad(ce,enc+heads,retain_graph=True)
    gb=torch.autograd.grad(wrong,enc,retain_graph=True)
    stats=combine_gradients(ce,wrong,net,head)
    assert 0<stats['weighted_gradient_ratio']<=.030001 and teacher.grad is None
    assert net.weight.grad.norm()>0 and head.weight.grad.norm()>0
    for p,a,b in zip(enc,ga,gb):assert torch.allclose(p.grad,a+stats['lambda_kd']*b,atol=1e-7)
    for p,a in zip(heads,ga[len(enc):]):assert torch.allclose(p.grad,a,atol=1e-7)
    opt.step();assert not torch.equal(before,net.weight)
    print('PASS: stable train-only graph,self match,wrong relation,teacher detach,encoder update,matched gradient cap')

if __name__=='__main__':main()
