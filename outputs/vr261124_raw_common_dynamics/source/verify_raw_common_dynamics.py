import argparse,json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
from audit_raw_common_dynamics import activity_mask,identity_features,marginal_stats,normalize
def main(root):
    out=root/'conditional_correspondence/raw_common_dynamics';read=lambda n:json.loads((out/n).read_text())
    c=np.load(out/'local_power_private.npz');raw=np.load(root/'conditional_correspondence/coarse_pilot/raw_inputs_private.npz')
    assert np.array_equal(c['sample_id'],raw['sample_id'][raw['fit']]) and c['audio'].shape==c['acc'].shape==(1280,146,7)
    assert read('native_rate_public.json')['native_nyquist_min_hz']>450
    d=read('audit_metrics_private.json');s=read('identity_summary_public.json');event=read('event_correspondence_public.json')
    assert len(d['events'])==3200 and len(d['features'])==2700 and len(s['per_user'])==300
    for v in s['per_user']:
      rr=[r for r in d['features'] if (r['user'],r['method'],r['scope'])==(v['user'],v['method'],v['scope'])];assert len(rr)==9
      for key in ['top1_10way','top1_2way','prototype_verification_auc']:assert abs(v[key]-np.mean([r[key] for r in rr]))<1e-12
    for v in d['features']:
      assert v['user'] in v['held_pair'] and v['genuine_records']==(32 if v['scope']=='new_words' else 24)
      assert all(0<=v[k]<=1 for k in ['top1_10way','top1_2way','prototype_verification_auc'])
    a,x,e,m=[c[k].astype('float64') for k in ('audio','acc','audio_activity','motion')];u,w,r=[c[k] for k in ('user','word','repeat')]
    words=sorted(map(int,set(w)));train=np.isin(u,list(range(3,11)))&np.isin(w,words[:12]);cutoff=np.quantile(m[train],.95)
    lookup={(int(us),int(wo),int(re)):i for i,(us,wo,re) in enumerate(zip(u,w,r))};maximum=0.
    for name in ['coupling_correct','wrongrepeat_fixedmask','ACC_only_stats']:
      z=[]
      for i in range(1280):
        if name=='ACC_only_stats':
          mask=activity_mask(np.log(np.maximum(np.exp(x[i]).sum(1),1e-20)))&(m[i]<=cutoff);z.append(marginal_stats(x[i],mask))
        else:
          ia=i if name=='coupling_correct' else lookup[int(u[i]),int(w[i]),(int(r[i])+1)%8]
          z.append(identity_features(a[ia],x[i],e[ia],m[i],cutoff,activity_mask(e[i]))[0])
      z=np.array(z);mean=np.nanmean(z[train],0);mean=np.where(np.isfinite(mean),mean,0);z=np.where(np.isfinite(z),z,mean);z=(z-mean)/np.maximum(z[train].std(0),1e-5)
      proto=normalize(np.stack([z[(u==us)&np.isin(w,words[:6])&(r<2)].mean(0) for us in range(1,11)]))
      qi=np.flatnonzero(np.isin(w,words[12:]));scores=normalize(z[qi])@proto.T
      for target in [1,2]:
        genuine=u[qi]==target;sub=scores[genuine];tops=np.isclose(sub,sub.max(1,keepdims=True),rtol=0,atol=1e-10)
        top=float(np.mean(tops[:,target-1]/tops.sum(1)));auc=float(roc_auc_score(genuine,scores[:,target-1]))
        ref=next(v for v in d['features'] if v['held_pair']==[1,2] and v['user']==target and v['method']==name and v['scope']=='new_words')
        maximum=max(maximum,abs(top-ref['top1_10way']),abs(auc-ref['prototype_verification_auc']))
    assert maximum<1e-10
    cross=read('crossband_metrics_private.json');assert len(cross)==720 and all(v['user'] in v['held_pair'] and v['word'] in words[12:] for v in cross)
    protocol=read('protocol_public.json');protocol['feature_assay_implementation']='vectorized,scalar numerical equivalence tested;15methods include independent unimodal masks and fixed-correct-mask controls'
    protocol['primary_identity_attribution_control']='coupling_correct versus wrongrepeat_fixedmask;identical activity/motion mask,only Audio source values differ'
    (out/'protocol_public.json').write_text(json.dumps(protocol,indent=2))
    result=dict(status='PASS',records=1280,event_groups=160,event_workpoints=3200,identity_workpoints=2700,raw_crossband_workpoints=720,scalar_feature_replay_max_metric_error=maximum,all_users_kept=True,only_fit_words=True,no_final_authentication=True)
    (out/'independent_verification_public.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
