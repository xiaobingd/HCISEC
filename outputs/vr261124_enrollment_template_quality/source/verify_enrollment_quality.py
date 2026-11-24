import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits
from calibrate_threeword import cutoff,measure
from prepare_phrase_auth import REG,CAL,SHORT

ROOT=Path(__file__).resolve().parent

def verify(folder,expected,method):
    out=ROOT/'conditional_correspondence'/folder
    rows=json.loads((out/'results_private.json').read_text());locks=json.loads((out/'thresholds_private.json').read_text())
    assert len(rows)==expected and len(locks)==expected//3
    keys=['seed','method','backend','family','words','user']
    assert len({tuple(r[k] for k in keys) for r in rows})==expected
    for a in rows:
        assert a['genuine_trials']==(40 if a['words']==1 else 13) and a['attack_trials']==9*a['genuine_trials']
        assert abs(a['far']-a['false_accepts']/a['attack_trials'])<1e-12
        assert abs(a['frr']-a['false_rejects']/a['genuine_trials'])<1e-12
        assert abs(a['hter']-(a['far']+a['frr'])/2)<1e-12
    d=np.load(ROOT/'phrase_auth/frontend_private.npz');u,w,r=d['user'],d['word'],d['repeat']
    choices=json.loads((out/'selection_private.json').read_text());selection=next(a['indices'] for a in choices if a['seed']==42 and a['method']==method)
    for a in choices:
        ix=np.array(a['indices']);assert len(np.unique(ix))==len(ix)
        for us in range(1,11):
            for word in REG:assert np.sum((u[ix]==us)&(w[ix]==word))==(8 if a['method']=='all32_reference' else 3)
    ix=np.array(selection);z=np.load(ROOT/'conditional_correspondence/stage3/views_private.npz')['42_A_ACC']
    pos=ix[u[ix]==1];bg=np.flatnonzero(np.isin(w,REG)&(u!=1));assert len(pos)==12 and len(bg)==288
    with threadpool_limits(limits=2):scores=SVC(C=10,class_weight='balanced').fit(z[np.r_[pos,bg]],u[np.r_[pos,bg]]==1).decision_function(z)
    templates=np.stack([z[ix[u[ix]==us]].mean(0) for us in range(1,11)])
    templates/=np.maximum(np.linalg.norm(templates,axis=1,keepdims=True),1e-8)
    cscores=(z@templates.T)[:,0]
    lookup={(int(us),int(word),int(rep)):i for i,(us,word,rep) in enumerate(zip(u,w,r))};order=np.random.default_rng(1123).permutation(5)
    def group(words,k):
        source=[(int(word),rep) for rep in range(8) for word in np.array(words)[order]]
        chunks=[source[i:i+k] for i in range(0,40-k+1,k)]
        indices=np.array([[lookup[us,word,rep] for word,rep in chunk] for us in range(1,11) for chunk in chunks])
        return indices,np.repeat(np.arange(1,11),len(chunks))
    errors=[]
    for backend,sc in [('RBF',scores),('cosine',cscores)]:
        for k in (1,3):
            ci,cu=group(CAL,k);qi,qu=group(SHORT,k);tau=cutoff(sc[ci].mean(1),cu==1,'quantile_gap')
            result=measure(sc[qi].mean(1),qu==1,tau)
            sel=lambda a:a['seed']==42 and a['method']==method and a['backend']==backend and a['words']==k and a['user']==1
            row=next(a for a in rows if sel(a) and a['family']=='english_word');lock=next(a for a in locks if sel(a))
            errors.append(abs(tau-lock['threshold']));errors.extend(abs(result[key]-row[key]) for key in result)
    assert max(errors)<1e-10
    v=json.loads((out/'verification_public.json').read_text());v.update(count_rate_checks=True,per_user_per_word_budget_checked=True,fixed_negative_background_per_target=288,independent_replay_max_error=max(errors))
    (out/'verification_public.json').write_text(json.dumps(v,indent=2));return rows

def main():
    a=verify('enrollment_quality_fixed_background',2520,'acc_audio12')
    b=verify('enrollment_identity_quality_fixed_background',2880,'acc_audio_identity12')
    fields=['far','frr','hter','eer','auc'];errors=[]
    for row in b:
        if row['method'] in ('fixed12','random12','acc12','all32_reference'):
            match=next(v for v in a if all(row[k]==v[k] for k in ['seed','method','backend','family','words','user']))
            errors.extend(abs(row[f]-match[f]) for f in fields)
    assert max(errors)==0
    out=ROOT/'conditional_correspondence/enrollment_template_report';out.mkdir(exist_ok=True)
    sources=['evaluate_enrollment_quality.py','evaluate_enrollment_quality_fixed.py','evaluate_enrollment_identity_quality.py','evaluate_enrollment_identity_quality_fixed.py','verify_enrollment_quality.py','test_enrollment_quality.py']
    sha={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in sources}
    sha['work/extract_registered_ecapa.py']=hashlib.sha256((ROOT.parents[1]/'work/extract_registered_ecapa.py').read_bytes()).hexdigest()
    (out/'verification_public.json').write_text(json.dumps(dict(status='PASS',primary_workpoints=5400,independent_replay_max_error=0,matched_baselines_across_phases_max_difference=0,registered_audio_only=True,global_models_frozen=True,source_sha256=sha),indent=2))
    print('ENROLLMENT_VERIFIED',max(errors),flush=True)

if __name__=='__main__':main()
