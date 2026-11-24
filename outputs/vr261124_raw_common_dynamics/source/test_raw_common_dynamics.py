"""Check event correspondence beyond a fixed word template, and quality handling."""
import numpy as np
from audit_raw_common_dynamics import band_correlation,identity_features,compare
from prepare_raw_common_dynamics import local_power
from raw_common_feature_identity import batch_features
rng=np.random.default_rng(42);time=np.arange(146);word=np.sin(time/15)[:,None]*np.ones((1,7))
audio=[];acc=[]
for repeat in range(8):
    transient=np.cumsum(rng.normal(size=(146,7)),axis=0)*.1
    audio.append(word+transient);acc.append(word+transient+.05*rng.normal(size=(146,7)))
audio=np.array(audio);acc=np.array(acc);ta=audio[:4].mean(0);tx=acc[:4].mean(0)
s=np.array([[band_correlation(audio[j]-ta,acc[i]-tx,np.ones(146,bool))[0].mean() for j in range(4,8)] for i in range(4,8)])
assert np.mean(s.argmax(1)==np.arange(4))==1
assert np.diag(s).mean()>s[~np.eye(4,dtype=bool)].mean()+.1
zeros=np.ones((146,7));value,bad=band_correlation(zeros,zeros,np.ones(146,bool));assert np.array_equal(value,np.zeros(7)) and not bad
bad=identity_features(zeros,zeros,np.full(146,-100.),np.zeros(146),1.)
assert np.isnan(bad[0]).all() and bad[3]==0
rate=1000;t=np.arange(3000)/rate;wave=np.sin(2*np.pi*140*t)
b,p,f,tm=local_power(wave,rate);assert b.shape==(146,7) and np.argmax(b.mean(0))==3
aa=rng.normal(size=(8,146,7));xx=rng.normal(size=(8,146,7));mask=rng.random((8,146))>.4;mask[0]=False
batch=batch_features(aa,xx,mask)
for i in range(8):
    single=identity_features(aa[i],xx[i],np.zeros(146),np.zeros(146),1.,mask[i])
    for j in range(3):assert np.allclose(single[j],batch[j][i],rtol=1e-10,atol=1e-10,equal_nan=True)
print('PASS:repeat-specific shared dynamics survive template control,constant-correlation guard,silence proxy missingness,local frequency bins')
