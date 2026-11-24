"""Secondary fixed-capacity cross-frequency dynamic control; no identity embedding."""
import argparse,itertools,json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from audit_linear_shared_identity import fit_mapping
from audit_raw_common_dynamics import activity_mask,band_correlation

def main(root):
    out=root/'conditional_correspondence/raw_common_dynamics';c=np.load(out/'local_power_private.npz')
    a,x,e,m=[c[k].astype('float64') for k in ('audio','acc','audio_activity','motion')];u,w,r=[c[k] for k in ('user','word','repeat')]
    words=sorted(map(int,set(w)));trainwords=words[:12];newwords=words[12:]
    lookup={(int(us),int(wo),int(re)):i for i,(us,wo,re) in enumerate(zip(u,w,r))}
    ra=a.copy();rx=x.copy()
    for us in range(1,11):
      for word in words:
        template=np.array([lookup[us,word,rep] for rep in range(4)])
        group=np.flatnonzero((u==us)&(w==word));ra[group]-=a[template].mean(0);rx[group]-=x[template].mean(0)
    rows=[];models={}
    for fold,held in enumerate(itertools.combinations(range(1,11),2)):
      base=[us for us in range(1,11) if us not in held];train=np.isin(u,base)&np.isin(w,trainwords)
      cutoff=float(np.quantile(m[train],.95));indices=np.flatnonzero(train&(r>=4))
      for variant in ['correct','wrongrepeat']:
        af=[];xf=[]
        for ix in indices:
          ia=int(ix) if variant=='correct' else lookup[int(u[ix]),int(w[ix]),4+(int(r[ix])-4+1)%4]
          mask=activity_mask(e[ia])&(m[ix]<=cutoff)
          if mask.sum()>=12:af.append(ra[ia][mask]);xf.append(rx[ix][mask])
        model=fit_mapping(np.concatenate(af),np.concatenate(xf),'rcca',2,1.)
        for key,value in model.items():models[f'{fold}_{variant}_{key}']=value
        for us in held:
          for word in newwords:
            evals=np.array([lookup[us,word,rep] for rep in range(4,8)]);scores=np.zeros((4,4));bad=0
            for ai,ia in enumerate(evals):
              pa=(ra[ia]-model['ma'])@model['pa']
              for xi,ix in enumerate(evals):
                px=(rx[ix]-model['mx'])@model['px'];mask=activity_mask(e[ia])&(m[ix]<=cutoff)
                corr,missing=band_correlation(pa,px,mask);scores[xi,ai]=corr.mean();bad+=missing
            tops=np.isclose(scores,scores.max(1,keepdims=True),rtol=0,atol=1e-10)
            rows.append(dict(held_pair=list(held),user=us,word=word,method=variant,top1=float(np.mean(np.diag(tops)/tops.sum(1))),
              correlation_advantage=float(np.diag(scores).mean()-scores[~np.eye(4,dtype=bool)].mean()),insufficient_pairs=bad))
    users=[]
    for variant in ['correct','wrongrepeat']:
      for us in range(1,11):
        rr=[v for v in rows if v['method']==variant and v['user']==us];assert len(rr)==36
        users.append(dict(method=variant,user=us,top1=float(np.mean([v['top1'] for v in rr])),correlation_advantage=float(np.mean([v['correlation_advantage'] for v in rr]))))
    signs=np.array(list(itertools.product([-1.,1.],repeat=10)));correct=np.array([v['correlation_advantage'] for v in users if v['method']=='correct']);wrong=np.array([v['correlation_advantage'] for v in users if v['method']=='wrongrepeat'])
    result=dict(protocol='secondary fixed2D,ridge1RCCA onraw7bandtemplate-residual dynamics;leave2identities out;fit12words/new4wordevaluation;first4template,last4pairing;correctvswrongrepeattraining;notphysicalsynchronizationproof',
      macro=[dict(method=variant,top1=float(np.mean([v['top1'] for v in users if v['method']==variant])),correlation_advantage=float(np.mean([v['correlation_advantage'] for v in users if v['method']==variant]))) for variant in ['correct','wrongrepeat']],
      per_user=users,correct_advantage_exact_user_p=float(np.mean(signs@correct/10>=correct.mean()-1e-12)),
      correct_minus_wrong_exact_user_p=float(np.mean(signs@(correct-wrong)/10>=(correct-wrong).mean()-1e-12)),no_tuning=True,held_identity_folds=45)
    (out/'crossband_public.json').write_text(json.dumps(result,indent=2));(out/'crossband_metrics_private.json').write_text(json.dumps(rows));np.savez_compressed(out/'crossband_models_private.npz',**models)
    print(json.dumps(result['macro']),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    with threadpool_limits(limits=1):main(p.parse_args().root)
