import argparse,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from prepare_continuous import fixed_starts,spectrum
from run_continuous import ActivityNet,acc_input,infer,stats,unit,measure

def main(root):
    torch.set_num_threads(2);base=root/'continuous_acc_activity';out=base/'results';read=lambda path:json.loads(path.read_text(encoding='utf-8'))
    split=read(base/'word_split_public.json');source=read(base/'source_audit_public.json');prep=read(base/'preparation_audit_public.json');p=read(out/'protocol_public.json')
    assert hashlib.sha256((base/'continuous_private.npz').read_bytes()).hexdigest()==p['dataset_sha256']
    protocol_hash=hashlib.sha256((out/'protocol_public.json').read_bytes()).hexdigest()
    assert hashlib.sha256((base/'word_split_public.json').read_bytes()).hexdigest()==source['word_split_sha256']==prep['word_split_sha256']
    c=np.load(base/'continuous_private.npz');u,w=c['user'],c['word'];test=np.isin(w,split['test']);x,m=c['acc'].astype('float64'),c['motion'].astype('float64')
    for us in range(1,11):
      starts=np.sort(c['window_start_host'][u==us]);assert np.all(np.diff(starts)>=3.-1e-8),'Cross-recording IMU window overlap'
    assert np.isnan(c['audio_proxy'][test]).all() and prep['test_Audio_recordings_read']==prep['test_Audio_timestamp_files_read']==0
    assert len(prep['recordings'])==400 and all(not r['Audio_opened'] and not r['Audio_timestamp_opened'] for r in prep['recordings'] if r['role']=='test')
    assert set(split['train']).isdisjoint(split['development']) and set(split['test']).isdisjoint(split['train']+split['development'])
    # Reconstruct target1's eight test-word records without reading any Audio or its clock.
    recordings=read(base/'recordings_private.json');replayed=0
    for rec in recordings:
      if rec['user']!=1 or rec['word'] not in split['test']:continue
      stem=Path(rec['stem']);raw=pd.read_csv(stem.with_name(stem.name+'_full_imu.csv'),usecols=['host_read_perf_counter_s','imu_sample_time','offscale_flags','accel_x','accel_y','accel_z'])
      latest=raw.groupby('host_read_perf_counter_s').imu_sample_time.max();shift=float(np.median(latest.index.to_numpy()-latest.to_numpy()))
      raw=raw[raw.offscale_flags==0].drop_duplicates('imu_sample_time').sort_values('imu_sample_time');t=raw.imu_sample_time.to_numpy()+shift;acc=raw[['accel_x','accel_y','accel_z']].to_numpy()
      ix=np.flatnonzero((u==1)&(w==rec['word']));ix=ix[np.argsort(c['ordinal'][ix])];starts=fixed_starts(t[0],t[-1]);assert len(ix)==len(starts)
      assert np.max(abs(c['window_start_host'][ix]-starts))<1e-9
      for i,start in zip(ix,starts):
        wave=np.stack([np.interp(start+np.arange(3000)/1000,t,v) for v in acc.T]);xx,mm,_=spectrum(wave)
        assert np.array_equal(xx.astype('float32'),c['acc'][i]) and np.array_equal(mm.astype('float32'),c['motion'][i]);replayed+=1
    audit=read(out/'exclusion_audit_public.json');selection=read(out/'mask_thresholds_private.json');locks=read(out/'identity_thresholds_private.json');rows=read(out/'per_seed_user_public.json')
    assert len(audit)==30 and len(selection)==120 and len(locks)==len(rows)==150
    for a in audit:
      f=np.load(out/f"fold_seed{a['seed']}_held{a['user']}_private.npz");b=f['batches'];ti=f['training_indices'];di=f['development_indices']
      assert b.shape==(400,32) and not np.isin(u[b],[a['user'],a['development_user']]).any() and np.isin(w[b],split['train']).all()
      assert np.all(u[di]==a['development_user']) and np.isin(w[di],split['development']).all()
      assert hashlib.sha256(b.tobytes()).hexdigest()==a['batch_sha256']
    for a in selection:
      best=min(a['candidates'],key=lambda r:(-round(r['f1'],12),-round(r['balanced_accuracy'],12),abs(r['threshold']-.5),r['threshold']))
      assert a['threshold']==best['threshold']
    for a in rows:
      assert a['genuine_trials']==int(((u==a['user'])&test).sum()) and a['attack_trials']==int(((u!=a['user'])&test).sum())
      assert abs(a['far']-a['false_accepts']/a['attack_trials'])<1e-12 and abs(a['frr']-a['false_rejects']/a['genuine_trials'])<1e-12
    assert all(a['threshold']>=-1 for a in locks)
    # ACC-only inference replay uses neither Audio labels nor Audio timestamps.
    fold=np.load(out/'fold_seed42_held1_private.npz');ti=fold['training_indices'];usable=(m<=np.quantile(m[ti],.95))&c['imu_valid'];saved=np.load(out/'masks_private.npz');maxerr=0.
    for method in ['ACC_self_trained','Audio_correct','Audio_wrongwindow']:
      ck=torch.load(out/f'{method}_seed42_held1_private.pt',map_location='cpu',weights_only=True);net=ActivityNet();net.load_state_dict(ck['model'])
      assert ck['protocol_sha256']==protocol_hash and ck['steps']==400
      prob=infer(net,acc_input(x,m),fold['mean'],fold['std']);tau=next(v['threshold'] for v in selection if (v['seed'],v['user'],v['method'])==(42,1,method));mask=(prob>=tau)&usable
      assert np.array_equal(mask,saved[f'42_1_{method}'])
      z0,missing=stats(x,mask);mean=np.nanmean(z0[ti],0);mean=np.where(np.isfinite(mean),mean,0);z=np.where(np.isfinite(z0),z0,mean);z=(z-mean)/np.maximum(z[ti].std(0),1e-5)
      reg=np.array([np.flatnonzero((u==1)&(w==word)&(c['ordinal']==0))[0] for word in p['registration_words']]);proto=unit(z[reg].mean(0));qi=np.flatnonzero(test);score=unit(z[qi])@proto;score[missing[qi]]=-1.1
      auth=next(v['threshold'] for v in locks if (v['seed'],v['user'],v['method'])==(42,1,method));metrics=measure(score,u[qi]==1,auth);ref=next(v for v in rows if (v['seed'],v['user'],v['method'])==(42,1,method))
      maxerr=max(maxerr,max(abs(metrics[k]-ref[k]) for k in ['eer','hter','far','frr','auc']))
    assert maxerr<1e-10 and len(list(out.glob('*_private.pt')))==90
    result=dict(status='PASS',test_word_IMU_windows_reconstructed_without_Audio=replayed,test_Audio_opened=0,test_Audio_clock_opened=0,test_Audio_labels_all_missing=True,word_partition_hash_verified=True,training_identity_exclusion=True,mask_threshold_choices_recomputed=120,models=90,identity_workpoints=150,ACC_inference_replayed=True,max_metric_replay_error=maxerr,all_test_windows_counted=True)
    (out/'independent_verification_public.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
