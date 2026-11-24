import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from run_acc_activity_teacher import ActivityNet,acc_input,infer_acc,choose_threshold,measure
from audit_raw_common_dynamics import normalize
from raw_common_feature_identity import batch_features

def main(root):
    torch.set_num_threads(2);out=root/'conditional_correspondence/acc_activity_teacher';read=lambda n:json.loads((out/n).read_text())
    old=out/'probability_selection_public.json';private=out/'mask_threshold_selection_private.json'
    if old.exists():old.replace(private)
    select=read('mask_threshold_selection_private.json');rows=read('per_seed_user_public.json');audit=read('exclusion_audit_public.json');locks=read('thresholds_private.json')
    c=np.load(root/'conditional_correspondence/raw_common_dynamics/local_power_private.npz');u,w=c['user'],c['word'];words=sorted(map(int,set(w)));trainwords=words[:12];newwords=words[12:]
    assert len(rows)==300 and len(audit)==30 and len(select)==90 and len(locks)==150
    for a in audit:
      f=np.load(out/f"fold_seed{a['seed']}_held{a['target']}_private.npz");b=f['batches'];ti=f['training_indices'];di=f['development_indices']
      assert b.shape==(400,32) and len(ti)==768 and len(di)==32
      assert not np.isin(u[b],[a['target'],a['development_user']]).any() and np.isin(w[b],trainwords).all()
      assert np.all(u[di]==a['development_user']) and a['target']!=a['development_user'] and np.isin(w[di],newwords).all()
      assert hashlib.sha256(b.tobytes()).hexdigest()==a['episode_sha256']
    for a in select:
      best=min(a['candidates'],key=lambda v:(-round(v['f1'],12),-round(v['balanced_accuracy'],12),abs(v['threshold']-.5),v['threshold']))
      assert a['probability_threshold']==best['threshold']
    for a in rows:
      assert a['genuine_trials']==(32 if a['scope']=='new_words' else 24) and a['attack_trials']==9*a['genuine_trials']
      assert abs(a['far']-a['false_accepts']/a['attack_trials'])<1e-12 and abs(a['frr']-a['false_rejects']/a['genuine_trials'])<1e-12
    # Regression against the same heldpair/scaler/prototype baseline in preceding raw audit.
    legacy=json.loads((root/'conditional_correspondence/raw_common_dynamics/audit_metrics_private.json').read_text())['features'];error=0.
    for a in rows:
      if a['method'] not in ['ACC_self','Audio_proxy_upperbound']:continue
      pair=sorted([a['user'],a['user']%10+1]);name='ACC_only_stats' if a['method']=='ACC_self' else 'ACC_stats_Audiomask'
      b=next(v for v in legacy if v['held_pair']==pair and v['user']==a['user'] and v['method']==name and v['scope']==a['scope'])
      error=max(error,abs(a['auc']-b['prototype_verification_auc']),abs(a['top1_10way']-b['top1_10way']))
    assert error<1e-10
    # Replay ACC-only inference and identity metrics for one excluded target, without Audio arrays.
    seed,held=42,1;f=np.load(out/'fold_seed42_held1_private.npz');ti=f['training_indices'];x,m=c['acc'].astype('float64'),c['motion'].astype('float64');motion=m<=np.quantile(m[ti],.95)
    masks=np.load(out/'masks_private.npz');replay=0.
    for method in ['Predicted_correct','Predicted_wrongrepeat']:
      net=ActivityNet();net.load_state_dict(torch.load(out/f'{method}_seed42_held1_private.pt',map_location='cpu',weights_only=True));net.eval()
      prob=infer_acc(net,acc_input(x,m),f['mean'],f['std']);p=next(v['probability_threshold'] for v in select if v['seed']==seed and v['user']==held and v['method']==method)
      mask=(prob>=p)&motion;assert np.array_equal(mask,masks[f'{seed}_{held}_{method}'])
      z0=batch_features(x,x,mask)[2];mean=np.nanmean(z0[ti],0);mean=np.where(np.isfinite(mean),mean,0);z=np.where(np.isfinite(z0),z0,mean);z=(z-mean)/np.maximum(z[ti].std(0),1e-5)
      proto=normalize(np.stack([z[(u==us)&np.isin(w,trainwords[:6])&(c['repeat']<2)].mean(0) for us in range(1,11)]))
      qi=np.flatnonzero(np.isin(w,newwords));score=normalize(z[qi])@proto[0];tau=next(v['threshold'] for v in locks if v['seed']==seed and v['user']==held and v['method']==method)
      result=measure(score,u[qi]==held,tau);ref=next(v for v in rows if v['seed']==seed and v['user']==held and v['method']==method and v['scope']=='new_words')
      replay=max(replay,max(abs(ref[k]-result[k]) for k in ['eer','hter','far','frr','auc']))
    assert replay<1e-10 and len(list(out.glob('Predicted*_private.pt')))==60
    v=dict(status='PASS',models=60,identity_folds=30,probability_selections=90,identity_thresholds=150,identity_workpoints=300,target_and_dev_excluded_from_training=True,ACC_only_inference_replayed=True,legacy_baseline_max_error=error,replay_max_error=replay)
    protocol=read('protocol_public.json');protocol['input_crop']='historical3secondACC windows centeredusingAudio-VAD metadata;mask/identityinference hasnoAudioargument,butthisisNOTend-to-endcontinuousACC-onlycropping'
    (out/'protocol_public.json').write_text(json.dumps(protocol,indent=2))
    (out/'independent_verification_public.json').write_text(json.dumps(v,indent=2));print(json.dumps(v))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
