"""Synthetic checks for calibration logic; no private data required."""
import numpy as np
from calibrate_threeword import cutoff, normalization, measure

labels=np.array([False,False,True,True])
score=np.array([-3.,-2.,2.,3.])
strict=cutoff(score,labels,'strict5')
middle=cutoff(score,labels,'midpoint5')
assert strict==2. and middle==0.
assert np.array_equal(score>=strict,score>=middle)
assert 1.<strict and 1.>=middle
assert measure(score,labels,middle)['hter']==0.
for rule in ['strict5','midpoint5','quantile_gap']:
    a=cutoff(score,labels,rule)
    b=cutoff(4*score+7,labels,rule)
    assert np.isclose(b,4*a+7)
    assert np.isfinite(a)

rng=np.random.default_rng(42)
s=rng.normal(size=(30,10));z=rng.normal(size=(10,10))
result=normalization(s,z,.05)
changed=z.copy();np.fill_diagonal(changed,1e9)
again=normalization(s,changed,.05)
for name in result:
    np.testing.assert_allclose(result[name],again[name])
np.testing.assert_array_equal(result['raw'],s)
owner=2;other=np.arange(10)!=owner
zs=z[other,owner];ts=s[:,other]
np.testing.assert_allclose(result['s'][:,owner],.5*((s[:,owner]-zs.mean())/max(zs.std(),.05)+
    (s[:,owner]-ts.mean(1))/np.maximum(ts.std(1),.05)))
az=np.sort(zs)[-3:];at=np.sort(ts,axis=1)[:,-3:]
np.testing.assert_allclose(result['as3'][:,owner],.5*((s[:,owner]-az.mean())/max(az.std(),.05)+
    (s[:,owner]-at.mean(1))/np.maximum(at.std(1),.05)))
constant=normalization(np.zeros((5,10)),np.zeros((10,10)),.05)
assert all(np.isfinite(a).all() for a in constant.values())
print('PASS: midpoint decisions, margin, affine calibration, owner exclusion, S/AS formulas, degenerate variance')

