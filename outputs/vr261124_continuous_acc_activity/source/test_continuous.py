import numpy as np
import torch
from prepare_continuous import fixed_starts
from run_continuous import ActivityNet,acc_input,infer,stats,choose_auth,choose_probability
torch.set_num_threads(2);rng=np.random.default_rng(42);torch.manual_seed(42)
starts=fixed_starts(100.,118.2);assert np.array_equal(starts,np.array([100.5,103.5,106.5,109.5,112.5]))
assert all(starts[i]+3<=starts[i+1] for i in range(len(starts)-1)) and starts[-1]+3<=117.7
x=rng.normal(size=(4,146,7));m=rng.normal(size=(4,146));v=acc_input(x,m);net=ActivityNet().eval()
p=infer(net,v,np.zeros(15),np.ones(15));solo=np.concatenate([infer(net,v[i:i+1],np.zeros(15),np.ones(15)) for i in range(4)])
assert np.allclose(p,solo,atol=1e-6)
empty=np.zeros((4,146),bool);z,missing=stats(x,empty);assert missing.all() and np.isnan(z).all()
score=np.array([.9,.7,-1.1,.1,-.1,-1.1]);label=np.array([1,1,1,0,0,0],bool);tau=choose_auth(score,label)
assert tau>=-1 and not (score[score==-1.1]>=tau).any()
truth=np.array([[0,0,1,1.]])
best,_=choose_probability(np.array([[.1,.2,.8,.9]]),truth,np.ones_like(truth,bool));assert best['threshold']==.5
print('PASS:IMU-only fixed boundaries,nonoverlap,ACC-only batch-invariant inference,missing activity,abstention cannot accept,threshold ties')
