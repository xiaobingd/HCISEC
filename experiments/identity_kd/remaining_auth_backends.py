"""Closed-identity training-only AS-Norm, Gaussian PLDA and pair neural scores."""
import numpy as np
import torch
from torch import nn
from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf

def norm(x):return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-8)

class Backends:
 def __init__(self,v,u,w,fit,users):
  self.v=norm(v);self.users=users
  self.cohort=norm(np.stack([self.v[fit][u[fit]==us].mean(0) for us in users]))
  self.pca=PCA(n_components=32,svd_solver='full').fit(v[fit]);self.x=self.pca.transform(v)
  x=self.x[fit];means=np.stack([x[u[fit]==us].mean(0) for us in users]);self.center=x.mean(0)
  residual=x.copy()
  for i,us in enumerate(users):residual[u[fit]==us]-=means[i]
  self.W=LedoitWolf().fit(residual).covariance_
  b=np.cov(means,rowvar=False)-self.W/np.mean([sum(u[fit]==us) for us in users])
  eig,vec=np.linalg.eigh((b+b.T)/2);self.B=(vec*np.maximum(eig,1e-5))@vec.T
  self.scale=np.maximum(x.std(0),1e-4);self.nx=(self.x-self.center)/self.scale
  torch.set_num_threads(2);torch.manual_seed(42)
  self.net=nn.Sequential(nn.Linear(64,32),nn.GELU(),nn.Dropout(.1),nn.Linear(32,1))
  opt=torch.optim.AdamW(self.net.parameters(),lr=1e-3,weight_decay=.01);rng=np.random.RandomState(42)
  for step in range(400):
   aa=[];bb=[];yy=[]
   for j in range(64):
    us=int(rng.choice(users));a=int(rng.choice(fit[u[fit]==us]));positive=j%2==0
    mask=(u[fit]==us)&(w[fit]!=w[a]) if positive else u[fit]!=us
    b=int(rng.choice(fit[mask]));aa.append(a);bb.append(b);yy.append(float(positive))
   features=self.pairs(self.nx[aa],self.nx[bb]);loss=nn.functional.binary_cross_entropy_with_logits(self.net(torch.tensor(features,dtype=torch.float32)).flatten(),torch.tensor(yy))
   opt.zero_grad();loss.backward();opt.step()
  self.net.eval();self.plda_cache={}
 def pairs(self,a,b):return np.concatenate([np.abs(a-b),a*b],axis=-1)
 def score(self,backend,owner,rr,qq):
  if backend=='asnorm':
   p=norm(self.v[rr].mean(0));q=self.v[qq];co=self.cohort[np.array(self.users)!=owner]
   es=np.sort(co@p)[-3:];qs=np.sort(q@co.T,axis=1)[:,-3:]
   raw=q@p
   return .5*((raw-es.mean())/max(es.std(),1e-3)+(raw-qs.mean(1))/np.maximum(qs.std(1),1e-3))
  if backend=='neural':
   with torch.no_grad():
    f=self.pairs(self.nx[qq,None,:],np.broadcast_to(self.nx[rr],(len(qq),len(rr),32)))
    return self.net(torch.tensor(f,dtype=torch.float32)).squeeze(-1).mean(1).numpy()
  n=len(rr)
  if n not in self.plda_cache:
   A=self.B+self.W/n;D=self.B+self.W
   same=np.block([[A,self.B],[self.B,D]]);diff=np.block([[A,np.zeros_like(A)],[np.zeros_like(A),D]])
   self.plda_cache[n]=(np.linalg.inv(same)-np.linalg.inv(diff),np.linalg.slogdet(same)[1]-np.linalg.slogdet(diff)[1])
  inverse,logdet=self.plda_cache[n];p=self.x[rr].mean(0)-self.center;q=self.x[qq]-self.center
  z=np.c_[np.broadcast_to(p,q.shape),q]
  return -.5*(np.einsum('bi,ij,bj->b',z,inverse,z)+logdet)
