import numpy as np
from screen_shared_bands import auc_scores
s=np.tile(np.eye(10),(26,12,1,1))
assert np.allclose(auc_scores(s,np.arange(10)),1)
assert np.allclose(auc_scores(np.zeros_like(s),np.arange(10)),.5)
assert np.all(auc_scores(s,np.roll(np.arange(10),1))<.5)
print('PASS: identity AUC perfect/tied/deranged cases;word-pair labels consistent')
