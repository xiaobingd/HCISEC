"""Correct vs same-person/same-word wrong-repeat waveform-period controls."""
import json,itertools
import numpy as np
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from extract_raw_periodicity import BASE,OUT,PRIVATE
from audit_conditional_residual import unit

def main():
    c=np.load(PRIVATE/'features_private.npz');u,w,r=[c[k] for k in ['user','word','repeat']]
    a,x=c['audio_track'],c['acc_track'];assert a.shape==x.shape and a.shape[0]==1280
    lookup={(int(us),int(wo),int(re)):i for i,(us,wo,re) in enumerate(zip(u,w,r))}
    wrong=np.arange(len(u))
    for us in range(1,11):
      for wo in sorted(set(w)):
        ix=np.flatnonzero((u==us)&(w==wo));ix=ix[np.argsort(r[ix])];wrong[ix]=np.roll(ix,-4)
    assert np.all(wrong!=np.arange(len(u))) and np.array_equal(u[wrong],u) and np.array_equal(w[wrong],w)
    def metric(aa,xx,pitchscale=1.):
      mask=(aa[:,:,2]>0)&(xx[:,:,2]>0)
      diff=abs(np.log2(aa[:,:,0]*pitchscale/xx[:,:,0]))
      return dict(joint_voiced_fraction=float(mask.mean()),pitch_within10pct=float((diff[mask]<=np.log2(1.1)).mean()) if mask.any() else None,
        median_abs_cents=float(np.median(diff[mask])*1200) if mask.any() else None,
        audio_voiced_fraction=float(aa[:,:,2].mean()),acc_voiced_fraction=float(xx[:,:,2].mean()))
    rows=[]
    for us in range(1,11):
      mask=u==us;right=metric(a[mask],x[mask]);bad=metric(a[wrong[mask]],x[mask])
      delta=None if right['pitch_within10pct'] is None or bad['pitch_within10pct'] is None else right['pitch_within10pct']-bad['pitch_within10pct']
      rows.append(dict(user=us,correct=right,wrongrepeat=bad,delta_match=delta))
    delta=np.array([v['delta_match'] for v in rows if v['delta_match'] is not None]);signs=np.array(list(itertools.product([-1,1],repeat=len(delta))))
    p=float(np.mean((signs*delta).mean(1)>=delta.mean()-1e-12)) if len(delta) else None
    # Descriptive cross-word identity geometry: same split, even vs odd repeats.
    words=sorted(set(map(int,w)));train=words[:12];val=words[12:]
    ap=c['audio_periodicity'];xp=c['acc_periodicity']
    ap=StandardScaler().fit(ap[np.isin(w,train)]).transform(ap);xp=StandardScaler().fit(xp[np.isin(w,train)]).transform(xp)
    def matrices(z):
      mats=[]
      for n,wo in enumerate(val):
        def proto(ww,parity):return unit(np.stack([z[(u==us)&(w==ww)&(r%2==parity)].mean(0) for us in range(1,11)]))
        mats.append(proto(wo,0)@proto(val[(n+1)%4],1).T)
      return np.array(mats)
    am,xm=matrices(ap),matrices(xp);off=~np.eye(10,dtype=bool)
    def relation(perm):return float(spearmanr(am[:,perm][:,:,perm][:,off].ravel(),xm[:,off].ravel()).statistic)
    rho=relation(np.arange(10));rng=np.random.default_rng(20261125);null=[relation(rng.permutation(10)) for _ in range(199)]
    summary=dict(correct=metric(a,x),same_person_word_wrongrepeat=metric(a[wrong],x),user_macro_match_advantage=float(delta.mean()) if len(delta) else None,users_with_comparable_voiced_frames=len(delta),paired_user_signflip_p=p,
      descriptive_periodicity_identity_relation_spearman=rho,identity_permutation_p=float((1+sum(v>=rho for v in null))/200),
      scope='frame comparison at legacy alignment, NO lag tuning;wrongrepeat retains user/word;notphysicalsync validation;voicing intersections maydiffer;sharedidentitydescriptors notalone proof ofidentitytransfer;allcohortscalers trainwordsonly inthisdescriptiveanalysis')
    summary['header_rate_pitchscale_sensitivity']=dict(correct=metric(a,x,48000/44100),wrongrepeat=metric(a[wrong],x,48000/44100),
      caveat='pitch-only rescaling sensitivity,notfull header48000 re-extraction or clock realignment;notusedtoselect results')
    (OUT/'mechanism_public.json').write_text(json.dumps(summary,indent=2));(OUT/'mechanism_per_user_public.json').write_text(json.dumps(rows,indent=2))
    print(json.dumps(summary))
if __name__=='__main__':main()
