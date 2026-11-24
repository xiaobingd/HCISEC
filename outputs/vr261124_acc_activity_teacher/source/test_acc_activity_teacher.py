import numpy as np
import torch
from run_acc_activity_teacher import ActivityNet,acc_input,infer_acc,choose_threshold
torch.set_num_threads(2);torch.manual_seed(42);rng=np.random.default_rng(42)
x=rng.normal(size=(5,146,7));m=rng.normal(size=(5,146));v=acc_input(x,m)
assert v.shape==(5,146,15)
net=ActivityNet().eval();mean=np.zeros(15);std=np.ones(15)
allp=infer_acc(net,v,mean,std);separate=np.concatenate([infer_acc(net,v[i:i+1],mean,std) for i in range(5)])
assert np.allclose(allp,separate,atol=1e-6)
logit=net(torch.tensor(v.astype('float32').transpose(0,2,1)));torch.nn.functional.binary_cross_entropy_with_logits(logit,torch.ones_like(logit)).backward()
assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
best,_=choose_threshold(np.array([[.1,.2,.8,.9]]),np.array([[False,False,True,True]]))
assert best['threshold']==.5 and best['f1']==1
print('PASS:ACC-only15channel input,sample-independent inference,finite predictor gradients,threshold tie rule')
