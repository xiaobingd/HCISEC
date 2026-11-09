"""Synthetic check of detached gradient balancing, without real query data."""
import numpy as np
import torch
from run_registration_rkd02 import adapt_student
from run_rkd01_vibravox import Student,norm
def main():
    torch.set_num_threads(1);rng=np.random.RandomState(42);u=np.r_[np.repeat(np.arange(7),4),np.repeat(7,12)];d=dict(user=u);x=rng.randn(40,246).astype('float32');t=norm(rng.randn(40,192).astype('float32'));torch.manual_seed(42);initial=Student();net,trace=adapt_student(d,x,initial,7,list(range(7)),np.arange(40),t,'V2',.1)
    assert abs(trace[0]['gradient_ratio']-.1)<1e-5
    assert all(np.isfinite(v['gradient_ratio']) and 0<=v['lambda_rkd']<=100 for v in trace)
    assert all(torch.isfinite(p).all() for p in net.parameters())
    assert any(not torch.equal(a,b) for a,b in zip(initial.parameters(),net.parameters()))
    print('PASS: first-step target gradient ratio, bounded detached weight, finite updates, input model unchanged')
if __name__=='__main__':main()
