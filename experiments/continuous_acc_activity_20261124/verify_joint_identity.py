import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from run_continuous import acc_input,stats,unit,measure
from run_joint_identity import JointNet,embed

def main(root):
    torch.set_num_threads(2);base=root/'continuous_acc_activity';out=base/'joint_identity';read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    protocol=read(out/'protocol_public.json');ph=hashlib.sha256((out/'protocol_public.json').read_bytes()).hexdigest()
    assert hashlib.sha256((base/'continuous_private.npz').read_bytes()).hexdigest()==protocol['data_sha256']
    previous=read(base/'results/independent_verification_public.json');assert previous['status']=='PASS' and previous['test_Audio_opened']==0
    c=np.load(base/'continuous_private.npz');u,w,valid=c['user'],c['word'],c['imu_valid'];split=protocol['word_split'];test=np.isin(w,split['test']);assert np.isnan(c['audio_proxy'][test]).all()
    chosen=read(out/'selection_public.json');audits=read(out/'exclusion_audit_public.json');rows=read(out/'per_seed_user_public.json');locks=read(out/'identity_thresholds_private.json')
    assert len(chosen)==len(audits)==30 and len(rows)==len(locks)==90
    for a in audits:
      f=np.load(out/f"fold_seed{a['seed']}_held{a['user']}_private.npz");batch=f['batches'];assert batch.shape==(400,32)
      assert hashlib.sha256(batch.tobytes()).hexdigest()==a['batch_sha256'];assert not np.isin(u[batch],[a['user'],a['development_user']]).any();assert np.isin(w[batch],split['train']).all()
      for ix in batch:
        assert len(set(w[ix]))==2 and len(set(ix))==32
        for us in a['training_users']:
          assert np.sum(u[ix]==us)==4
          for word in set(w[ix]):assert np.sum((u[ix]==us)&(w[ix]==word))==2
    for a in chosen:
      assert len(a['candidates'])==4 and len(a['shared_scores'])==2
      for v in a['shared_scores']:
        cc=[r for r in a['candidates'] if r['lambda_value']==v['lambda_value']];assert len(cc)==2
        assert abs(v['mean_development_auc']-np.mean([r['development_auc'] for r in cc]))<1e-12
        assert abs(v['mean_development_eer']-np.mean([r['development_eer'] for r in cc]))<1e-12
      best=min(a['shared_scores'],key=lambda r:(-round(r['mean_development_auc'],12),round(r['mean_development_eer'],12),r['lambda_value']))
      assert best['lambda_value']==a['shared_lambda']
    files=list(out.glob('*_private.pt'));assert len(files)==150
    for path in files:
      ck=torch.load(path,map_location='cpu',weights_only=True);assert ck['protocol_sha256']==ph and ck['steps']==400
      assert [v['step'] for v in ck['trace']]==[100,200,300,400]
      assert all(torch.isfinite(v).all() for v in ck['model'].values())
    for a in rows:
      assert a['genuine_trials']==int(((u==a['user'])&test).sum()) and a['attack_trials']==int(((u!=a['user'])&test).sum())
      assert abs(a['far']-a['false_accepts']/a['attack_trials'])<1e-12 and abs(a['frr']-a['false_rejects']/a['genuine_trials'])<1e-12
    x,m=c['acc'].astype('float64'),c['motion'].astype('float64');inputs=acc_input(x,m);raw,missing=stats(x,valid);assert not missing.any();enroll=np.load(out/'enrollment_indices_private.npz')['indices'];assert enroll.shape==(10,12)
    maxerr=0.
    for seed,held in [(42,1),(43,10),(44,10)]:
      f=np.load(out/f'fold_seed{seed}_held{held}_private.npz');norm=tuple(f[k] for k in ['mi','si','mr','sr']);choice=next(a['shared_lambda'] for a in chosen if a['seed']==seed and a['user']==held)
      for arm in ['Identity_only','Joint_correct','Joint_wrongwindow']:
        lam=0. if arm=='Identity_only' else choice;ck=torch.load(out/f'{arm}_lambda{lam}_seed{seed}_held{held}_private.pt',map_location='cpu',weights_only=True);net=JointNet();net.load_state_dict(ck['model'])
        vi=np.flatnonzero(~test);zv=embed(net,inputs,raw,valid,vi,norm);mapping=np.full(len(u),-1,int);mapping[vi]=np.arange(len(vi));proto=unit(np.stack([zv[mapping[ix]].mean(0) for ix in enroll]))
        qi=np.flatnonzero(test);zq=embed(net,inputs,raw,valid,qi,norm);score=zq@proto.T;tau=next(a['threshold'] for a in locks if a['seed']==seed and a['user']==held and a['arm']==arm)
        result=measure(score[:,held-1],u[qi]==held,tau);ref=next(a for a in rows if a['seed']==seed and a['user']==held and a['arm']==arm)
        maxerr=max(maxerr,max(abs(result[k]-ref[k]) for k in ['eer','hter','far','frr','auc']))
    assert maxerr<1e-10
    result=dict(status='PASS',candidate_models=150,selected_models=90,identity_folds=30,batch_tasks_checked=12000,shared_lambdas_recomputed=30,identity_workpoints=90,ACC_only_replayed_models=9,max_metric_replay_error=maxerr,test_Audio_labels_missing=True,data_hash_unchanged=True,all_test_windows_counted=True)
    (out/'independent_verification_public.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
