"""Hand-computable authentication cases, independent replay, and no-positive contrastive case."""
import numpy as np
import torch
from evaluate import metrics,comparison_stats
from verify import independent_metrics
from common import crossword_supcon

for labels,scores,expected in [([1,1,0,0],[.9,.8,.2,.1],(0.,1.)),([1,1,0,0],[.1,.2,.8,.9],(1.,0.)),([1,1,0,0],[.5,.5,.5,.5],(.5,.5))]:
    got=metrics(labels,scores,.5);check=independent_metrics(labels,scores,.5)
    assert np.isclose(got['EER'],expected[0]) and np.isclose(got['AUC'],expected[1])
    assert all(np.isclose(got[k],v) for k,v in check.items())
rng=np.random.default_rng(27)
for _ in range(100):
    y=rng.integers(0,2,50);s=np.round(rng.normal(size=50),1);a=metrics(y,s,.1);b=independent_metrics(y,s,.1)
    assert max(abs(a[k]-b[k]) for k in b)<1e-12
z=torch.nn.functional.normalize(torch.randn(6,8,requires_grad=True),dim=1)
no_positive=crossword_supcon(z,torch.arange(6),torch.arange(6));assert no_positive.item()==0 and no_positive.requires_grad
loss=crossword_supcon(z,torch.tensor([0,0,0,1,1,1]),torch.tensor([0,0,1,0,0,1]));assert torch.isfinite(loss)
for improvement in [.01,0.]:
    rows=[dict(seed=seed,user=user,arm=arm,family=family,EER=.1-(improvement if arm=='T3' else 0.)) for seed in [42,43,44] for user in range(1,11) for arm in ['T0','T2','T3','T4'] for family in ['english_short','english_long']]
    comparisons=comparison_stats(rows)
    for r in comparisons:
        assert np.isclose(r['mean_difference'],-improvement) and np.allclose(r['user_cluster_bootstrap_95CI'],-improvement)
        assert np.isclose(r['exact_user_signflip_p'],2/1024 if improvement else 1.)
        assert np.isclose(r['Holm_corrected_p'],6/1024 if improvement else 1.)
        assert r['seeds_improved']==(3 if improvement else 0) and r['users_improved']==(10 if improvement else 0)
print('PASS: hand-computed perfect/reversed/tied ROC,100 independent rank-metric replays,contrastive valid/no-positive anchors')
print('PASS: exact1024signflip,Holm3,user-cluster bootstrap,seed directions;constant improvement and all-tie controls')
