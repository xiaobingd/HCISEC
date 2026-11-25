import numpy as np
from extract_raw_periodicity import track,summarize
r=np.random.default_rng(8)
for rate in [1000,44100]:
  for f in [80,120,200,300]:
    time=np.arange(rate*3)/rate;wave=np.sin(2*np.pi*f*time)+.2*np.sin(4*np.pi*f*time)
    a=track(wave,rate);b=track(wave*7,rate)
    assert abs(np.median(a[5:-5,0])-f)<f*.04,(rate,f,np.median(a[:,0]))
    assert a[:,2].mean()>.9 and np.allclose(a,b,atol=1e-5)
    p,h=summarize(a);assert p.shape==(6,) and h.shape==(4,)
silent=track(np.zeros(12000),4000);assert not silent[:,2].any()
time=np.arange(12000)/4000
mains=track(np.sin(2*np.pi*50*time),4000);assert mains[5:-5,2].mean()<.05
noise=track(r.normal(size=12000),4000)
print('PASS: 8 harmonic-tone F0 checks, gain invariance and silence; bandlimited-noise voiced fraction',noise[:,2].mean())
