"""Recover a known shared latent space on genuinely held synthetic identities."""
import numpy as np
from threadpoolctl import threadpool_limits
from audit_linear_shared_identity import fit_mapping,similarity,target_metrics
with threadpool_limits(limits=1):
    rng=np.random.default_rng(42);s=rng.normal(size=(80,3));a=s@rng.normal(size=(3,32))+.01*rng.normal(size=(80,32));x=s@rng.normal(size=(3,32))+.01*rng.normal(size=(80,32))
    for method in ['rcca','lowrank_ridge']:
        model=fit_mapping(a[:40],x[:40],method,3,.1)
        score=similarity(model,a[40:],x[40:]);correct=np.mean(score.argmax(1)==np.arange(40))
        wrong=fit_mapping(a[rng.permutation(40)],x[:40],method,3,.1)
        bad=similarity(wrong,a[40:],x[40:]);incorrect=np.mean(bad.argmax(1)==np.arange(40))
        assert correct>.8 and correct>incorrect+.5,(method,correct,incorrect)
        assert np.array_equal(model['ma'],a[:40].mean(0)) and np.array_equal(model['mx'],x[:40].mean(0))
    tie=target_metrics(np.ones((2,2)),1,[1,2]);assert tie['top1']==tie['auc']==.5
    near=target_metrics(np.array([[1.,1.+1e-12],[0.,1.]]),1,[1,2]);assert near['auc']==.5
print('PASS:known shared latent recovery,identity-mismatch negative control,train-only centering,tie-aware chance')
