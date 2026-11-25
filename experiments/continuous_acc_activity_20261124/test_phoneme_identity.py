import numpy as np
from audit_phoneme_identity import descriptor,window,GROUPS,Decoder

wave=np.random.default_rng(8).normal(size=(3,3000))
assert descriptor(wave[:,:120]).shape==(24,)
assert np.isfinite(descriptor(np.zeros((3,120)))).all()
assert np.allclose(window(wave,1.5),descriptor(wave[:,1440:1560]))
assert window(wave,0) is None and window(wave,3) is None
decoder=Decoder(loglevel='FATAL')
assert 'AY' in decoder.lookup_word('five') and 'AY' in decoder.lookup_word('nine')
assert all(len(words)>=2 for words in GROUPS.values())
print('PASS: fixed crop, finite descriptors, out-of-range rejection and dictionary AY coverage')
