"""Independent interval, role, target, threshold and all15model metric replay."""
import json,hashlib,time,argparse
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits
from common import Student,ctc_posterior,smooth_to_acc,shifted_posteriors
from clock_mapping import clock_regions,map_receptive_centers,regional_targets
from prepare_acc import PRIVATE,OUT,OPT,DEV,REG,CAL,SHORT,LONG,CHINESE,ROLES
from train import batch_indices,protocol,digest

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def independent_metrics(y,s,tau):
    y=np.asarray(y,dtype=bool);s=np.asarray(s);pos=int(y.sum());neg=len(y)-pos;assert pos and neg
    auc=(rankdata(s)[y].sum()-pos*(pos+1)/2)/(pos*neg)
    order=np.argsort(-s,kind='stable');sorted_s=s[order];ends=np.r_[np.flatnonzero(np.diff(sorted_s)!=0),len(y)-1]
    tp=np.r_[0,np.cumsum(y[order])[ends]];fp=np.r_[0,np.cumsum(~y[order])[ends]];x=fp/neg;f=1-tp/pos
    difference=x-f;k=np.flatnonzero(difference>=0)[0]
    if k==0:eer=float((x[0]+f[0])/2)
    else:
        a=-difference[k-1]/(difference[k]-difference[k-1]);eer=float(x[k-1]+a*(x[k]-x[k-1]))
    far=float(np.mean(s[~y]>=tau));frr=float(np.mean(s[y]<tau))
    return dict(EER=eer,AUC=float(auc),FAR=far,FRR=frr,TAR=1-frr,HTER=(far+frr)/2)

def run(inputs_only=False):
    torch.set_num_threads(4);began=time.time();checks={}
    data=np.load(PRIVATE/'acc_private.npz');u,w=data['user'],data['word'];manifest=json.loads((PRIVATE/'windows_private.json').read_text(encoding='utf-8'))
    assert len(manifest)==len(u)==2179 and len(set(data['sample_id']))==2179
    assert sorted(sum(ROLES.values(),[]))==list(range(40))
    groups={}
    for i,row in enumerate(manifest):
        assert row['sample_id']==data['sample_id'][i] and row['user']==u[i] and row['word']==w[i]
        assert row['word'] in ROLES[row['role']];groups.setdefault(row['record_group'],[]).append(row)
    for group,rows in groups.items():
        assert len(set(r['role'] for r in rows))==1
        intervals=sorted((r['window_start'],r['window_end']) for r in rows)
        assert all(a[1]<=b[0]+1e-8 for a,b in zip(intervals,intervals[1:]))
    for user in range(1,11):
        intervals=sorted((r['window_start'],r['window_end']) for r in manifest if r['user']==user)
        assert all(a[1]<=b[0]+1e-8 for a,b in zip(intervals,intervals[1:])),'Cross-recording overlap for same user'
    reg=data['registration_indices'];assert reg.shape==(10,12) and np.isin(w[reg],REG).all() and len(set(reg.ravel()))==120
    assert np.array_equal(u[reg],np.repeat(np.arange(1,11)[:,None],12,axis=1))
    checks['data']=dict(windows=2179,unique_record_groups=len(groups),nonoverlap=True,role_group_overlap=0,registration_windows=120,
      raw_acc_SHA256=sha(PRIVATE/'acc_private.npz'),sameword_K12_comparable=False,cross_recording_same_user_window_overlap=0)
    assert checks['data']['raw_acc_SHA256']==json.loads((OUT/'data_audit_public.json').read_text())['data_sha256']
    records=json.loads((PRIVATE/'recordings_private.json').read_text(encoding='utf-8'));source_hashes=[]
    vocabulary=json.loads((OUT/'role_split_public.json').read_text(encoding='utf-8'))['vocabulary']
    for rec in records:
        stem=Path(rec['stem']);path=stem.with_name(stem.name+'_full_imu.csv');h=hashlib.sha256()
        # Label metadata only; never open the recording's Audio or Audio timestamp CSV.
        metadata=json.loads(stem.with_name(stem.name+'_meta.json').read_text(encoding='utf-8-sig'))
        assert metadata['prompt_text'].strip().lower()==rec['prompt'].strip().lower()==vocabulary[str(rec['word'])].strip().lower()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(2**20),b''):h.update(block)
        source_hashes.append(h.hexdigest())
    assert len(set(source_hashes))==len(records)==400;checks['data']['independently_rehashed_unique_raw_ACC_sources']=400
    checks['data']['original_prompt_metadata_labels_checked']=400
    # Independent direct-Hann FFT frontend replay for a predeclared training window.
    rec=next(r for r in records if r['user']==1 and r['word']==0);stem=Path(rec['stem']);raw=pd.read_csv(stem.with_name(stem.name+'_full_imu.csv'))
    latest=raw.groupby('host_read_perf_counter_s').imu_sample_time.max();offset=float(np.median(latest.index.to_numpy()-latest.to_numpy()))
    raw=raw[raw.offscale_flags==0].drop_duplicates('imu_sample_time').sort_values('imu_sample_time');tt=raw.imu_sample_time.to_numpy()+offset
    j=int(np.flatnonzero((u==1)&(w==0))[0]);grid=data['starts'][j]+np.arange(3000)/1000
    wave=np.stack([np.interp(grid,tt,raw[c].to_numpy()) for c in ['accel_x','accel_y','accel_z']]);wave-=wave.mean(1,keepdims=True)
    frames=np.lib.stride_tricks.sliding_window_view(wave,100,axis=1)[:,::20,:];hann=.5-.5*np.cos(2*np.pi*np.arange(100)/100)
    fft=np.fft.rfft(frames*hann,axis=-1)/hann.sum();manual=np.log(np.maximum(abs(fft[:,:,1:50])**2,1e-20)).transpose(0,2,1).reshape(147,146)
    frontend_error=float(abs(manual-data['acc'][j]).max());assert frontend_error<5e-6
    checks['data']['independent_direct_FFT_max_absolute_error']=frontend_error
    teacher=np.load(PRIVATE/'teacher_private.npz');ti=teacher['indices'];q=teacher['posterior'];mask=teacher['valid']
    assert np.array_equal(ti,np.flatnonzero(np.isin(w,OPT+DEV))) and q.shape==(818,146,392)
    audit=json.loads((OUT/'teacher_audit_public.json').read_text(encoding='utf-8'))
    assert audit['query_Audio_reads']==audit['forbidden_role_Audio_reads']==0 and len(audit['records'])==160
    assert all(row['word'] in OPT+DEV for row in audit['records']) and np.allclose(q.sum(2),1,atol=1e-5)
    # Repeat exact CTC forward/backward from saved logits on3predeclared cases, without opening Audio.
    replay=[]
    for user,word in [(1,0),(2,16),(10,34)]:
        saved=np.load(PRIVATE/'teacher_records'/f'u{user:02}_w{word:02}_private.npz')
        got,logZ=ctc_posterior(saved['logprob'],saved['tokens']);e=float(abs(got-saved['posterior']).max());assert e<1e-6
        nominal,segments,_=clock_regions(saved['clock_frame_end'],saved['clock_host'])
        host,regions=map_receptive_centers((399/2+np.arange(len(got))*320)/16000*nominal,399/2/16000*nominal,segments)
        assert np.array_equal(regions,saved['audio_regions']) and np.allclose(host,saved['audio_host'],equal_nan=True)
        ids=np.flatnonzero((u==user)&(w==word));mapped_error=0.
        for i in ids:
            pp,mm=regional_targets(host,got,data['frame_times'][i],regions,.1);j=int(np.flatnonzero(ti==i)[0]);mm&=data['valid'][i]
            assert np.array_equal(mm,mask[j]);mapped_error=max(mapped_error,float(abs(pp-q[j]).max()))
        assert mapped_error<1e-6;replay.append(dict(user=user,word=word,posterior_max_error=e,mapped_max_error=mapped_error))
    checks['teacher']=dict(training_development_only=True,query_Audio_reads=0,full_vocabulary=392,replay=replay,
      mean_valid_blank_probability=float(q[:,:,0][mask].mean()),nonblank_majority_frames=int(((q[:,:,0]<.5)&mask).sum()),
      valid_frames=int(mask.sum()),windows_without_auxiliary=int((~mask.any(1)).sum()))
    faulty=np.load(PRIVATE/'teacher_records/u09_w33_private.npz');nominal,segments,clockaudit=clock_regions(faulty['clock_frame_end'],faulty['clock_host'])
    centers=(399/2+np.arange(len(faulty['posterior']))*320)/16000*nominal;host,regions=map_receptive_centers(centers,399/2/16000*nominal,segments)
    assert clockaudit['callback_discontinuities']>0 and len(regions)>0 and np.allclose(host,faulty['audio_host'],equal_nan=True)
    fault_error=0.
    for i in np.flatnonzero((u==9)&(w==33)):
        pp,mm=regional_targets(host,faulty['posterior'],data['frame_times'][i],regions,.1);j=int(np.flatnonzero(ti==i)[0]);mm&=data['valid'][i]
        assert np.array_equal(mm,mask[j]);fault_error=max(fault_error,float(abs(pp-q[j]).max()))
    assert fault_error<1e-6;checks['teacher']['discontinuous_clock_case_replay']=dict(user=9,word=33,mapped_max_error=fault_error,clock_audit=clockaudit)
    input_checks=dict(checks,status='PASS_INPUTS_ONLY',authentication_models_checked=0,query_Audio_reads=0)
    (OUT/'input_verification_public.json').write_text(json.dumps(input_checks,indent=2),encoding='utf-8')
    if inputs_only:
        print('INDEPENDENT_INPUT_AND_CLOCK_VERIFICATION_PASS',flush=True);return
    norm=np.load(PRIVATE/'normalization_final_private.npz');fit=np.flatnonzero(np.isin(w,OPT+DEV));assert np.array_equal(fit,norm['fit_indices'])
    values=data['acc'][fit].transpose(0,2,1)[data['valid'][fit]]
    assert np.allclose(values.mean(0),norm['mean']) and np.allclose(np.maximum(values.std(0),1e-6),norm['std']);del values
    x=torch.tensor((data['acc']-norm['mean'][None,:,None])/norm['std'][None,:,None]);valid=torch.tensor(data['valid'])
    lock=np.load(PRIVATE/'threshold_lock_private.npz');publiclock=json.loads((OUT/'threshold_lock_public.json').read_text())
    queryphase=json.loads((OUT/'query_phase_public.json').read_text());devlock=json.loads((OUT/'development_lock_public.json').read_text())
    assert devlock['teacher_cache_sha256']==sha(PRIVATE/'teacher_private.npz')==audit['teacher_cache_sha256']
    assert devlock['locked_at_unix']<=publiclock['locked_at_unix']<=queryphase['query_scoring_started_at_unix']
    assert publiclock['count']==150 and publiclock['lock_sha256']==sha(PRIVATE/'threshold_lock_private.npz')
    results=json.loads((OUT/'results_public.json').read_text(encoding='utf-8'));metric_errors=[];score_error=0.;threshold_error=0.;models=[];index=0
    cal=np.flatnonzero(np.isin(w,CAL));query=np.flatnonzero(np.isin(w,SHORT+LONG+CHINESE))
    assert not np.isin(query,cal).any() and not np.isin(query,reg).any()
    families=dict(english_short=SHORT,english_long=LONG,chinese=CHINESE)
    def encode(model,ii):
        out=[]
        with torch.inference_mode():
            for part in np.array_split(ii,max(1,int(np.ceil(len(ii)/64)))):out.append(model(x[part],valid[part],auxiliary=False)[0].numpy())
        return np.concatenate(out)
    for seed in [42,43,44]:
      batches=batch_indices(u,w,OPT+DEV,seed,1000);batch_sha=hashlib.sha256(batches.tobytes()).hexdigest()
      actual=np.load(PRIVATE/f'final_seed{seed}_batches_private.npz')['batches'];assert np.array_equal(actual,batches) and not np.isin(w[batches],REG+CAL+SHORT+LONG+CHINESE).any()
      for batch in batches:
        assert len(batch)==80 and len(set(batch))==80
        assert len(set(w[batch]))==4 and all(np.sum(u[batch]==user)==8 for user in range(1,11))
      torch.manual_seed(seed);initial=Student(392,16);initial_sha=hashlib.sha256(b''.join(v.detach().numpy().tobytes() for v in initial.state_dict().values())).hexdigest()
      wrong,perm,changed=shifted_posteriors(q,mask,seed);saved_perm=np.load(PRIVATE/f'final_seed{seed}_temporal_permutation_private.npz')['permutation']
      assert np.array_equal(perm,saved_perm)
      for j in range(len(q)):
        assert np.allclose(q[j,mask[j]].sum(0),wrong[j,mask[j]].sum(0),atol=2e-4)
        assert np.array_equal(q[j,~mask[j]],wrong[j,~mask[j]])
      for arm in ['T0','T1','T2','T3','T4']:
        path=PRIVATE/f'final_{seed}_{arm}_private.pt';saved=torch.load(path,weights_only=True)
        assert saved['steps']==1000 and saved['initial_sha256']==initial_sha and saved['batch_sha256']==batch_sha
        assert path.stat().st_mtime>=devlock['locked_at_unix'] and sha(path)==publiclock['checkpoint_sha256'][index]
        model=Student(392,16);model.load_state_dict(saved['model']);model.eval();zreg=encode(model,reg.ravel()).reshape(10,12,192)
        proto=zreg.mean(1);proto/=np.linalg.norm(proto,axis=1,keepdims=True);assert np.allclose(proto,lock['templates'][index])
        cs=encode(model,cal)@proto.T
        for user in range(1,11):
            tau=(np.quantile(cs[u[cal]==user,user-1],.1)+np.quantile(cs[u[cal]!=user,user-1],.95))/2
            threshold_error=max(threshold_error,abs(float(tau-lock['thresholds'][index,user-1])))
        zs=encode(model,query);score=zs@proto.T;old=np.load(PRIVATE/f'query_scores_{seed}_{arm}_private.npz')
        assert np.array_equal(old['indices'],query);score_error=max(score_error,float(abs(score-old['scores']).max()))
        for row in results['per_seed_user']+results['per_seed_word_user']:
            if row['seed']!=seed or row['arm']!=arm:continue
            keep=w[query]==row['word'] if 'word' in row else np.isin(w[query],families[row['family']]);user=row['user']
            m=independent_metrics(u[query][keep]==user,score[keep,user-1],lock['thresholds'][index,user-1])
            metric_errors.extend(abs(m[k]-row[k]) for k in m)
        deployed=torch.load(PRIVATE/f'deploy_{seed}_{arm}_private.pt',weights_only=True)
        assert all(k.startswith(('backbone.','attention.','identity.')) for k in deployed['encoder'])
        deploy_model=Student(392,16);missing=deploy_model.load_state_dict(deployed['encoder'],strict=False);assert not missing.unexpected_keys
        deploy_model.eval();assert np.allclose(encode(deploy_model,query[:20]),zs[:20],atol=1e-6)
        models.append(dict(seed=seed,arm=arm,steps=1000,checkpoint_sha256=sha(path),initial_sha256=initial_sha,batch_sha256=batch_sha));index+=1
        print('VERIFIED_MODEL',seed,arm,flush=True)
    assert score_error<1e-5 and threshold_error<1e-6 and max(metric_errors)<1e-6
    checks.update(status='PASS',models=models,model_count=15,all_thresholds=150,score_replay_max_absolute_error=score_error,
      threshold_replay_max_absolute_error=threshold_error,reported_metric_fields_replayed=len(metric_errors),metric_max_absolute_error=max(metric_errors),
      same_initialization_and_batches=True,fit_only_normalization=True,independent_threshold_lock_before_query=True,
      encoder_only_deployment_equivalent=True,seeded_T4_valid_mask_and_posterior_marginal_preserved=True,
      final_users=10,query_words=15,all_query_windows_retained=len(query),runtime_s=time.time()-began)
    (OUT/'verification_public.json').write_text(json.dumps(checks,indent=2),encoding='utf-8');print('INDEPENDENT_VERIFICATION_PASS',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--inputs-only',action='store_true');args=parser.parse_args()
    with threadpool_limits(limits=4):run(args.inputs_only)
