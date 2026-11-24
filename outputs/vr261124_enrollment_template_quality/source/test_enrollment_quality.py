import numpy as np
from evaluate_enrollment_quality import top3,centrality,audio_quality
x=np.eye(8)
assert np.array_equal(top3(np.ones(8)),np.arange(3))
assert len(set(top3(np.arange(8))))==3
assert np.isfinite(centrality(x)).all()
assert np.isfinite(audio_quality(np.zeros(4800,dtype=np.int16),48000)).all()
try:audio_quality(np.zeros(4800,dtype=np.int16),44100);raise RuntimeError('Rate mismatch accepted')
except AssertionError:pass
print('PASS:ties deterministic;fixed count;silent quality finite;rate interpretation explicit')
