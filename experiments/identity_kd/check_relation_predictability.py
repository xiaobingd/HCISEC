"""Synthetic predictive, memorization, axes and fit-scope checks."""
import numpy as np
from evaluate_relation_predictability import relation,features,fit_predict,metrics,cross_prior,rank_arrays,identity_permutation_concordance

def main():
    rng=np.random.RandomState(5)
    z=rng.normal(size=(5,4,8,12));s=relation(z)
    assert s.shape==(4,4,5,5)
    x=features(s,cross_prior(s),range(4));assert x.shape==(4,4,5,5,14)
    # Affine Audio relation genuinely predictable from the ACC pair score.
    target=.7*s+.1
    p=fit_predict(x,target,x)
    m=metrics(p,target)
    assert m['crossword']['task_spearman_macro']>.95
    assert np.allclose(rank_arrays(s,s),1)
    control=identity_permutation_concordance(s,s,count=39)
    assert abs(control['task_spearman']-1)<1e-12 and control['p_one_sided']<=.1
    memory=np.broadcast_to(cross_prior(s),s.shape)
    mm=metrics(memory,s)
    assert mm['crossword']['residual_spearman_macro'] is None
    # Removed endpoint target labels must not influence a held-out-endpoint fit.
    mask=np.ones((5,5),bool);mask[0,:]=mask[:,0]=False
    modified=target.copy();modified[:,:,0,:]+=100;modified[:,:,:,0]-=30
    a=fit_predict(x,target,x,edgemask=mask);b=fit_predict(x,modified,x,edgemask=mask)
    assert np.allclose(a,b)
    # Held-out word labels have zero influence when wordmask restricts fit.
    mask=np.zeros((4,4),bool);mask[:2,:2]=True
    modified=target.copy();modified[2:]+=100;modified[:,2:]-=30
    a=fit_predict(x,target,x,wordmask=mask);b=fit_predict(x,modified,x,wordmask=mask)
    assert np.allclose(a,b)
    print('PASS: relation axes,affine prediction,no spurious static residual,heldout endpoint and word isolation')

if __name__=='__main__':main()
