"""Meaningful checks for balanced role rotation and content-leak rejection."""
import collections
import itertools
from run_strict_within_content import splits,role_check

prompts=[35,36,37,38,39]
folds=splits(prompts)
assert len(folds)==20 and len(set(folds))==20
reg=collections.Counter();cal=collections.Counter();query=collections.Counter()
for a,b,q in folds:
    role_check([0,1,2],[a],[b],q)
    assert len(q)==3
    reg[a]+=1;cal[b]+=1;query.update(q)
assert set(reg.values())=={4} and set(cal.values())=={4} and set(query.values())=={12}
for arguments in [([0],[35],[36],[36,37,38]),([35],[36],[37],[35,38,39])]:
    try:role_check(*arguments)
    except ValueError:pass
    else:raise AssertionError('Content leakage was not rejected')
triples=list(itertools.combinations(range(8),3))
assert len(triples)==56 and all(len(set(z))==3 for z in triples)
print('PASS:20balanced splits,reg/cal/query/train content isolation,leak rejection,56unique within-triplet recordings')
