"""Lock every threshold before the one-shot historical-query review; independent deployment replay."""
import json,time
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from train_select import ROOT,OLD,SOURCE,PRIV,OUT,PREVIOUS,SEEDS,inputs,model_from,embed,sha,write
from common import Student
from train import batch_indices
from evaluate import metrics,normalize,comparison_stats
from verify import independent_metrics
from prepare_acc import OPT,DEV,CAL,SHORT,LONG,CHINESE

FAMILIES=dict(english_short=SHORT,english_long=LONG,chinese=CHINESE)
CODES=['BASE','ORIGINAL','MATCHED','TUNED','SHUFFLED']

def main():
    torch.set_num_threads(4);lock=json.loads((OUT/'selection_lock_public.json').read_text(encoding='utf-8'))
    catalog=json.loads((PRIV/'catalog_private.json').read_text(encoding='utf-8'));assert len(catalog)==15
    data,x,valid,u,w,words=inputs('final');norm=np.load(PRIV/'normalization_final_private.npz');oldnorm=np.load(SOURCE/'normalization_final_private.npz')
    assert np.array_equal(norm['mean'],oldnorm['mean']) and np.array_equal(norm['std'],oldnorm['std'])
    reg=data['registration_indices'];cal=np.flatnonzero(np.isin(w,CAL));query=np.flatnonzero(np.isin(w,SHORT+LONG+CHINESE));assert reg.shape==(10,12)
    assert not set(reg.ravel())&set(cal) and not set(reg.ravel())&set(query) and not set(cal)&set(query)
    tau=[];prototypes=[];audits=[];initial={}
    for row in catalog:
        path=__import__('pathlib').Path(row['path']);assert sha(path)==row['checkpoint_sha256'];model,saved=model_from(path,16)
        assert saved['steps']==1000 and row['seed'] in SEEDS
        batches=batch_indices(u,w,words,row['seed'],1000);bh=__import__('hashlib').sha256(batches.tobytes()).hexdigest()
        assert bh==saved['batch_sha256'] and np.isin(w[batches],OPT+DEV).all()
        initial.setdefault(row['seed'],saved['initial_sha256']);assert initial[row['seed']]==saved['initial_sha256']
        if not row['reused_reference']:assert path.stat().st_mtime>=lock['locked_at_unix']
        proto=normalize(embed(model,x,valid,reg.ravel()).reshape(10,12,192).mean(1));cs=embed(model,x,valid,cal)@proto.T
        tt=[(np.quantile(cs[u[cal]==user,user-1],.1)+np.quantile(cs[u[cal]!=user,user-1],.95))/2 for user in range(1,11)]
        tau.append(tt);prototypes.append(proto);audits.append(dict(code=row['code'],seed=row['seed'],checkpoint_sha256=row['checkpoint_sha256'],initial_sha256=saved['initial_sha256'],batch_sha256=bh,reused_reference=row['reused_reference']))
        enc={k:v for k,v in saved['model'].items() if k.startswith(('backbone.','attention.','identity.'))}
        torch.save(dict(encoder=enc,templates=torch.tensor(proto),thresholds=torch.tensor(tt),mean=torch.tensor(norm['mean']),std=torch.tensor(norm['std'])),PRIV/f'deploy_{row["seed"]}_{row["code"]}_private.pt')
    tau=np.array(tau);prototypes=np.array(prototypes);path=PRIV/'threshold_lock_private.npz'
    if path.exists():
        previous=np.load(path);assert np.array_equal(previous['thresholds'],tau) and np.array_equal(previous['prototypes'],prototypes)
    else:
        np.savez_compressed(path,thresholds=tau,prototypes=prototypes)
        write('threshold_lock_public.json',dict(count=150,locked_at_unix=time.time(),sha256=sha(path),selection_lock_sha256=sha(OUT/'selection_lock_public.json'),models=audits,new_query_scores_computed=0))
    print('TUNING_ALL_150_THRESHOLDS_LOCKED',flush=True)
    threshold_public=json.loads((OUT/'threshold_lock_public.json').read_text());assert threshold_public['sha256']==sha(path)
    phase=OUT/'query_phase_public.json'
    if not phase.exists():write(phase.name,dict(started_at_unix=time.time(),threshold_lock_sha256=sha(path),Audio_reads=0))
    assert json.loads(phase.read_text())['started_at_unix']>=threshold_public['locked_at_unix']>=lock['locked_at_unix']
    users=[];wordrows=[]
    for index,row in enumerate(catalog):
        model,_=model_from(__import__('pathlib').Path(row['path']),16);score=embed(model,x,valid,query)@prototypes[index].T
        np.savez_compressed(PRIV/f'query_{row["seed"]}_{row["code"]}_private.npz',scores=score,indices=query)
        for family,familywords in FAMILIES.items():
            keep=np.isin(w[query],familywords)
            for user in range(1,11):users.append(dict(seed=row['seed'],arm=row['code'],family=family,user=user,**metrics(u[query][keep]==user,score[keep,user-1],tau[index,user-1])))
            for word in familywords:
                keepword=w[query]==word
                for user in range(1,11):wordrows.append(dict(seed=row['seed'],arm=row['code'],family=family,word=word,user=user,**metrics(u[query][keepword]==user,score[keepword,user-1],tau[index,user-1])))
        print('TUNING_QUERY_COMPLETE',row['seed'],row['code'],flush=True)
    summary=[]
    for code in CODES:
        for family in list(FAMILIES)+['english_primary']:
            rr=[r for r in users if r['arm']==code and (r['family']==family or (family=='english_primary' and r['family']!='chinese'))]
            summary.append(dict(arm=code,family=family,**{k:float(np.mean([r[k] for r in rr])) for k in ['EER','AUC','FAR','FRR','TAR','HTER']}))
    mapping=dict(TUNED='T3',BASE='T0',MATCHED='T2',SHUFFLED='T4',ORIGINAL='OLD')
    comparisons=comparison_stats([{**r,'arm':mapping[r['arm']]} for r in users])
    for c,name in zip(comparisons,['TUNED-BASE','TUNED-MATCHED','TUNED-SHUFFLED']):
        c['comparison']=name;c['interpretation']='negative favors TUNED'
    result=dict(status='COMPLETE',selected=lock['selected'],parameters=lock['config'],development_gate_passed=lock['development_dual_metric_and_FAR_FRR_gate_passed'],
      units='fractions',summary=summary,per_seed_user=users,per_seed_word_user=wordrows,comparisons=comparisons,
      newly_trained_final_models=sum(not r['reused_reference'] for r in catalog),reused_reference_evaluations=sum(r['reused_reference'] for r in catalog),
      historical_query_pool=True,Audio_query_reads=0,no_test_parameter_selection=True)
    write('results_public.json',result)
    # Replay exported encoder-only models, calibrations and every reported metric through independent rank-statistic code.
    errors=[];score_error=0.;threshold_error=0.;deployed=[]
    for index,row in enumerate(catalog):
        saved=torch.load(PRIV/f'deploy_{row["seed"]}_{row["code"]}_private.pt',weights_only=True);model=Student(392,16)
        missing=model.load_state_dict(saved['encoder'],strict=False);assert not missing.unexpected_keys and all(not k.startswith(('backbone.','attention.','identity.')) for k in missing.missing_keys);model.eval()
        proto=normalize(embed(model,x,valid,reg.ravel()).reshape(10,12,192).mean(1));assert np.allclose(proto,prototypes[index],atol=1e-6)
        cs=embed(model,x,valid,cal)@proto.T
        for user in range(1,11):
            tt=(np.quantile(cs[u[cal]==user,user-1],.1)+np.quantile(cs[u[cal]!=user,user-1],.95))/2;threshold_error=max(threshold_error,abs(float(tt-tau[index,user-1])))
        sq=embed(model,x,valid,query)@proto.T;stored=np.load(PRIV/f'query_{row["seed"]}_{row["code"]}_private.npz');assert np.array_equal(stored['indices'],query)
        score_error=max(score_error,float(abs(sq-stored['scores']).max()))
        for r in users+wordrows:
            if r['seed']!=row['seed'] or r['arm']!=row['code']:continue
            keep=w[query]==r['word'] if 'word' in r else np.isin(w[query],FAMILIES[r['family']]);user=r['user'];mm=independent_metrics(u[query][keep]==user,sq[keep,user-1],tau[index,user-1])
            errors.extend(abs(r[k]-mm[k]) for k in mm)
        deployed.append(dict(seed=row['seed'],arm=row['code'],encoder_only=True));print('TUNING_REPLAY_VERIFIED',row['seed'],row['code'],flush=True)
    assert score_error<1e-6 and threshold_error<1e-6 and max(errors)<1e-6
    previous=json.loads((PREVIOUS/'summary_public.json').read_text(encoding='utf-8'));reference_error=0.
    for code,oldcode in [('BASE','T0'),('ORIGINAL','T3')]:
        for r in [r for r in summary if r['arm']==code]:
            old=next(o for o in previous['summary'] if o['arm']==oldcode and o['family']==r['family'])
            reference_error=max(reference_error,max(abs(r[k]-old[k]) for k in ['EER','AUC','FAR','FRR','HTER']))
    assert reference_error<1e-9
    write('verification_public.json',dict(status='PASS',deployments=deployed,calibrated_thresholds=150,independent_metric_fields=len(errors),metric_max_absolute_error=max(errors),
      score_max_absolute_error=score_error,threshold_max_absolute_error=threshold_error,previous_reference_max_error=reference_error,
      same_initialization_and_batches=True,normalization_equals_frozen_reference=True,development_only_parameter_selection=True,
      all10identities_all15querywords_retained=True,query_Audio_reads=0,source_data_sha256=sha(SOURCE/'acc_private.npz'),source_teacher_sha256=sha(SOURCE/'teacher_private.npz')))
    print('TUNING_VERIFICATION_PASS',json.dumps([r for r in summary if r['family']=='english_primary']),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=4):main()
