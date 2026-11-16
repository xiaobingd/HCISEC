"""Synthetic invariance, hard-rank and shape checks; no private data."""
import numpy as np
from audit_audio_acc_relations import analyze,rho,collapse,ranking,pair_vector,validate_alignment,rank_auc,pair_residual
from sklearn.metrics import roc_auc_score

def main():
    rng=np.random.RandomState(7)
    z=rng.normal(size=(5,1,1,12))+rng.normal(size=(1,6,1,12))*.15+rng.normal(size=(5,6,8,12))*.3
    result=analyze(z,z,bootstraps=20,permutations=39,seed=17)
    for r in result['relation_conditions'].values():
        assert abs(r['spearman']-1)<1e-10
        assert r['identity_permutation_p_one_sided']<=.1
    assert result['hard_impostor_ranking']['spearman_macro']>.9999
    assert result['hard_impostor_ranking']['topk_overlap']['1']==1
    assert result['authentication'][0]['macro_auc']==result['authentication'][1]['macro_auc']
    assert result['authentication'][0]['macro_auc']['crossword']>.95
    assert rho(np.ones(5),np.arange(5)) is None
    for _ in range(20):
        labels=np.arange(36)<4;scores=rng.randint(0,8,36)
        assert abs(rank_auc(labels,scores)-roc_auc_score(labels,scores))<1e-12
    additive=np.arange(5)[:,None]+np.arange(5)[None,:]
    assert np.max(np.abs(pair_residual(additive)))<1e-12
    a=rng.normal(size=(9,9));b=-a
    assert ranking(a,b)['spearman_macro']<-.999
    # Word matching is controlled before averaging; diagonal/offdiag differ.
    s=np.zeros((6,6,5,5));s[np.arange(6),np.arange(6)]=7
    assert np.allclose(collapse(s,'different_user_same_word'),7)
    assert np.allclose(collapse(s,'different_user_cross_word'),0)
    s[np.arange(6),np.arange(6)]=np.arange(6)[:,None,None]
    weights=np.array([2,1,0,0,0,0])
    assert np.allclose(collapse(s,'different_user_same_word',weights),1/3)
    # Global renaming preserves within-modality geometry; audit compares
    # corresponding identities and therefore must detect the mismatch.
    p=np.array([2,4,1,0,3]);m=rng.normal(size=(5,5));m=(m+m.T)/2
    assert rho(pair_vector(m,'different_user_cross_word'),pair_vector(m[np.ix_(p,p)],'different_user_cross_word'))<.95
    front=dict(sample_id=np.arange(3),user=np.array([1,2,3]),word=np.array([0,1,24]))
    encoded={k:v.copy() for k,v in front.items()}
    teacher=dict(sample_id=np.arange(3),indices=np.array([0,1]),done=np.array(2))
    assert np.array_equal(validate_alignment(front,teacher,encoded),[0,1])
    for change in ['ids','labels','scope','incomplete']:
        e={k:v.copy() for k,v in encoded.items()};t={k:v.copy() for k,v in teacher.items()}
        if change=='ids':t['sample_id']=t['sample_id'][::-1]
        if change=='labels':e['word'][0]=2
        if change=='scope':t['indices']=np.array([0,1,2])
        if change=='incomplete':t['done']=np.array(1)
        try:validate_alignment(front,t,e)
        except AssertionError:pass
        else:raise AssertionError('failed to reject '+change)
    print('PASS: paired shape,perfect match,identity permutation,word condition,hard ranks,ties')

if __name__=='__main__':main()
