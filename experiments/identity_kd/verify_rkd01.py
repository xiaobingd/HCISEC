"""Small loss/protocol checks using synthetic data only, never Test selection."""
import numpy as np
import torch
from run_rkd01_vibravox import rkd,Student,thresholds
def main():
    torch.set_num_threads(1);torch.manual_seed(42);y=torch.arange(8).repeat_interleave(4);centers=torch.randn(8,192);t=centers[y]+.1*torch.randn(32,192);z=t.clone().requires_grad_(True)
    assert float(rkd(z,t,y))<1e-10
    # Euclidean geometry is invariant to common scale, translation and rotation.
    q,_=torch.linalg.qr(torch.randn(192,192));other=3*t@q+torch.randn(1,192);assert float(rkd(other,t,y))<1e-8
    wrong=t[torch.randperm(32)];loss=rkd(z,wrong,y);assert float(loss)>.001;loss.backward();assert torch.isfinite(z.grad).all() and float(z.grad.abs().sum())>0
    # Whole-class renaming does not change geometry and is not a negative control.
    assert float(rkd(t,t,(y+1)%8))<1e-10
    net=Student();o=net(torch.randn(32,246));assert o.shape==(32,192);assert torch.allclose(o.norm(dim=1),torch.ones(32),atol=1e-6)
    labels=np.r_[np.ones(30,dtype=bool),np.zeros(100,dtype=bool)];s=np.r_[np.linspace(.2,1,30),np.linspace(-1,.7,100)];tau=thresholds(s,labels)['Cal_FAR5'];assert float((s[~labels]>=tau).mean())<=.05
    print('PASS: RKD invariance, nontrivial incorrect correspondence, finite gradient, output normalization, Cal-FAR5 bound')
if __name__=='__main__':main()
