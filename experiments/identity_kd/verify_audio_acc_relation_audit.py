"""Verify public aggregates, source immutability and repeated-run core parity."""
import hashlib
import json
from pathlib import Path

def main():
    out=Path('outputs/mei_audio_acc_relation_audit_20261116')
    r=json.loads((out/'results_public.json').read_text(encoding='utf-8'))
    d=json.loads((out/'data_checks_public.json').read_text(encoding='utf-8'))
    paths=['outputs/mei_common_frontend_private.npz',
           'outputs/mei_neural_verifier_20261116/redimnet2_train_embeddings_private.npz',
           'outputs/mei_crossword_representation_20261116/vectors_private.npz']
    for path,record in zip(paths,d['source_sha256']):assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==record['sha256']
    assert d['available_fit_records']==sum(sum(x['word_counts']) for x in d['per_user_word_counts'])
    assert d['balanced_used_records']==9*18*8
    assert d['available_fit_records']-d['balanced_used_records']==d['excluded_fit_records']
    assert len(r['all_directed_relations'])==72
    assert len({(x['target'],x['impostor']) for x in r['all_directed_relations']})==72
    for modality in r['authentication']:
        assert [x['user'] for x in modality['per_user']]==[f'U{i}' for i in range(1,10)]
        for condition in ['sameword','crossword']:
            assert abs(sum(x[condition] for x in modality['per_user'])/9-modality['macro_auc'][condition])<1e-12
    for c,v in r['relation_conditions'].items():
        assert -1<=v['spearman']<=1 and 0<v['identity_permutation_p_one_sided']<=1
        assert v['identity_permutation_p_holm_3']>=v['identity_permutation_p_one_sided']
        assert v['units']==(9 if c.startswith('same_user') else 36)
        assert len(v['leave_one_user_out_sensitivity'])==9
    assert all(float(x['audio_difficulty_rank']).is_integer() and float(x['acc_difficulty_rank']).is_integer() for x in r['all_directed_relations']),'ties require fractional top-k review'
    oldpath=out/'pre_nuisance_results_private.json';parity=None
    if oldpath.exists():
        old=json.loads(oldpath.read_text(encoding='utf-8'))
        assert old['authentication']==r['authentication'] and old['margin']==r['margin']
        for c in r['relation_conditions']:assert old['relation_conditions'][c]['spearman']==r['relation_conditions'][c]['spearman']
        assert old['hard_impostor_ranking']['per_user']==r['hard_impostor_ranking']['per_user']
        parity=True
    v=dict(state='PASS',source_caches_unchanged=True,all_nine_users_and_72_directed_pairs=True,
        aggregate_recalculation=True,no_tied_aggregate_hard_ranks=True,repeated_execution_core_parity=parity,
        synthetic_tests='run check_audio_acc_relation_audit.py separately;PASS recorded in work log',
        limits='aggregate/source checks;not independent waveform or embedding re-extraction')
    (out/'verification_public.json').write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8');print('VERIFICATION_PASS')

if __name__=='__main__':main()
