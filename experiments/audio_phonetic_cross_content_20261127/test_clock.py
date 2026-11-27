"""Known-rate, known-jump synthetic clocks; prohibit smoothing across missing intervals."""
import numpy as np
from clock_mapping import clock_regions,map_receptive_centers,regional_targets

rng=np.random.default_rng(44100)
for rate in [44100,48000]:
    end=480*np.arange(1,1801);host=100+end/rate+rng.uniform(-.0003,.0003,len(end));host[900:]+=.3
    nominal,segments,audit=clock_regions(end,host)
    assert nominal==rate and audit['callback_discontinuities']==1 and len(segments)==2
    centers=np.arange(600,end[-1]-600,900);h,regions=map_receptive_centers(centers,550,segments)
    assert len(regions)==2
    q=np.zeros((len(centers),3),dtype='float32');q[:,1]=1
    lo=h[regions[0,1]-1];hi=h[regions[1,0]];acc=np.array([h[regions[0,0]+5],(lo+hi)/2,h[regions[1,0]+5]])
    target,valid=regional_targets(h,q,acc,regions)
    assert valid.tolist()==[True,False,True] and target[1,0]==1 and np.allclose(target.sum(1),1)
    for first,last in regions:
        idx=np.arange(first,last);expected=100+centers[idx]/rate+(idx>=regions[1,0])*.3
        assert abs(h[idx]-expected).max()<.001
print('PASS:44.1/48k block-rate recovery,300ms jump split,RF boundary exclusion,no smoothing across missing host interval')
