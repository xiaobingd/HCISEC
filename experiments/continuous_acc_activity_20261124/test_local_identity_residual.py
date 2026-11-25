import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from audit_local_identity_residual import descriptors,predict,holm

r=np.random.default_rng(44);a=r.normal(size=(30,146,7));e=r.normal(size=(30,146))
d=descriptors(a,e);shift=r.normal(size=(30,146,1));other=descriptors(a+shift,4*e+9)
assert d['shape'].shape==(30,14) and d['envelope'].shape==(30,15)
for k in d:assert np.allclose(d[k],other[k],atol=1e-10)
assert np.allclose(holm([.01,.04,.03]),[.03,.06,.06])
x=r.normal(size=(120,4));y=2*x[:,0]+r.normal(size=120)*.05
sc=StandardScaler().fit(x[:90]);expected=Ridge(alpha=10).fit(sc.transform(x[:90]),y[:90]).predict(sc.transform(x[90:]))
assert np.allclose(predict(x[:90],y[:90],x[90:]),expected)
assert np.mean((predict(x[:90],y[:90],x[90:])-y[90:])**2)<np.var(y[90:])*.1
print('PASS: shape common-gain removal, envelope affine invariance, Holm, ridge equivalence and known signal')
