"""Synthetic known-signal and affine-input invariance checks."""
import numpy as np
from audit_conditional_residual import fit_predict, unit

rng=np.random.default_rng(3)
x=rng.normal(size=(100,3));y=2*x[:,0]-x[:,1]
p=fit_predict(x[:80],y[:80],x[80:])
q=fit_predict(7*x[:80]+5,y[:80],7*x[80:]+5)
assert np.max(abs(p-q))<1e-10
assert np.mean((p-y[80:])**2)<np.var(y[80:])*.1
assert np.allclose(np.linalg.norm(unit(x),axis=1),1)
print('PASS: affine invariance, known predictable relation, normalized cosine input')
