"""Identical physical-signal frontend for both public-data baseline pilots."""
import numpy as np,struct,hashlib
from pathlib import Path
FS=16000;WIN=12800;FFT=512;HOP=160

def wav(path):
 raw=Path(path).read_bytes();assert raw[:4]==b'RIFF' and raw[8:12]==b'WAVE';pos=12;fmt=None;payload=None
 while pos+8<=len(raw):
  key=raw[pos:pos+4];n=struct.unpack_from('<I',raw,pos+4)[0];chunk=raw[pos+8:pos+8+n]
  if key==b'fmt ':fmt=chunk
  if key==b'data':payload=chunk
  pos+=8+n+(n%2)
 assert fmt is not None and payload is not None
 kind,ch,sr=struct.unpack_from('<HHI',fmt);bits=struct.unpack_from('<H',fmt,14)[0]
 if kind==65534:kind=struct.unpack_from('<H',fmt,24)[0]
 if kind==3:assert bits in (32,64);x=np.frombuffer(payload,dtype='<f%d'%(bits//8)).astype(np.float64)
 elif kind==1:
  assert bits in (16,32),(kind,bits);x=np.frombuffer(payload,dtype='<i%d'%(bits//8)).astype(np.float64)/(2.**(bits-1))
 else:raise ValueError((kind,bits))
 return sr,x.reshape(-1,ch),{'sha256':hashlib.sha256(raw).hexdigest(),'format':kind,'bits':bits,'channels':ch,'sample_rate':sr}

def resample(x,sr):
 if sr==FS:return x
 assert sr==48000
 n=np.arange(-48,49);h=np.sinc(n/3.)*np.kaiser(len(n),5.);h/=h.sum()
 return np.convolve(x,h,mode='same')[::3]

def resize(z):
 f=np.linspace(0,z.shape[0]-1,80);i=np.floor(f).astype(int);a=f-i;j=np.minimum(i+1,z.shape[0]-1)
 v=z[i]*(1-a[:,None])+z[j]*a[:,None]
 t=np.linspace(0,z.shape[1]-1,80);i=np.floor(t).astype(int);a=t-i;j=np.minimum(i+1,z.shape[1]-1)
 return (v[:,i]*(1-a[None,:])+v[:,j]*a[None,:]).astype('float32')

def pair(acc,mic):
 assert len(acc)==len(mic)==WIN
 def magnitude(x):
  x=x-x.mean();frames=np.lib.stride_tricks.sliding_window_view(x,FFT)[::HOP];w=.5-.5*np.cos(2*np.pi*np.arange(FFT)/FFT)
  return (np.abs(np.fft.rfft(frames*w,axis=1))/w.sum()).T
 a=magnitude(acc);m=magnitude(mic)
 # Explicit HTK-spaced triangular Mel bank; same fixed bank for all identities.
 hz=700*(10**(np.linspace(0,2595*np.log10(1+8000/700),82)/2595)-1);freq=np.arange(FFT//2+1)*FS/FFT
 bank=np.maximum(0,np.minimum((freq[None,:]-hz[:-2,None])/(hz[1:-1,None]-hz[:-2,None]),(hz[2:,None]-freq[None,:])/(hz[2:,None]-hz[1:-1,None])))
 return resize(20*np.log10(a+1e-8)),resize(10*np.log10(bank@(m*m)+1e-16))
