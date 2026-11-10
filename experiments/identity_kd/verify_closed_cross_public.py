"""Closed-set split/training audit, cached SVM refit, GPU repeatability diagnostic."""
import argparse,json
from pathlib import Path
import numpy as np
import torch,joblib
from sklearn.preprocessing import StandardScaler
from run_closed_cross_public import Encoder,encode,fast_features,view,head,BRANCHES
from abcs43_open_personal_cycle import Generator
from run_rkd01_vibravox import norm,metrics,thresholds

def validate_rows(root,rows,d,ix,dataset):
    public=json.loads((root/'per_user_public.json').read_text());tt=json.loads((root/'thresholds_private.json').read_text());users=sorted(rows);checks=0
    for j,u in enumerate(users,1):
        for method,r in rows[u].items():
            cs=np.array(r['calibration']['scores']);cy=np.array(r['calibration']['labels']);assert cy.sum()==8 and (~cy).sum()==8*(len(users)-1);ts=thresholds(cs,cy);q=r['query'];s=np.array(q['scores']);y=np.array(q['labels']);assert y.sum()==24 and (~y).sum()==24*(len(users)-1);pos=s[y];neg=s[~y];auc=float(((pos[:,None]>neg)+.5*(pos[:,None]==neg)).mean())
            for point,tau in ts.items():
                stored=tt[method][str(u)][point];assert stored==tau or abs(stored-tau)<1e-7
                if point=='Cal_FAR5':assert (cs[~cy]>=tau).mean()<=.05
                p=next(p for p in public if p['user']==f'{dataset}-C3-U{j:02}' and p['method']==method and p['point']==point);tar=float((s[y]>=tau).mean());far=float((s[~y]>=tau).mean());assert abs(p['tar']-tar)<1e-7 and abs(p['far']-far)<1e-7 and abs(p['hter']-(1-tar+far)/2)<1e-7 and abs(p['auc']-auc)<1e-7;checks+=1
    return checks

def legacy(a,root):
    prior=Path(a.legacy_mei);d=dict(np.load(a.frontend));data=dict(np.load(prior/'prepared_private.npz'));ix={k:np.array(v) for k,v in json.loads((prior/'split_private.json').read_text()).items()};ix['query']=ix['crossword'];rows={int(k):v for k,v in json.loads((root/'scores_private.json').read_text()).items()};audit=json.loads((root/'no_audio_training_private.json').read_text());ti=np.load(prior/'teacher_targets_private.npz')['indices'];assert set(audit['training_indices'])==set(ti) and audit['sampler']=='all9x4' and audit['users']==list(range(1,10));assert not set(ti)&set(ix['query']);net=Encoder(data['x'].shape[1]);net.load_state_dict(torch.load(root/'no_audio_student_private.pt',map_location='cpu',weights_only=False));emb=encode(net,data['x']);base=np.concatenate([data['shape'],emb],1)/np.sqrt(2);maxerr=0
    for u in range(1,10):
        models=joblib.load(root/f'owner{u}'/'new_control_heads_private.joblib')
        for name,x in [('no_audio_encoder',emb),('no_audio_baseline',base)]:
            refit=head(x,d,ix,u)
            for role,ii in [('calibration',np.r_[ix['calA'],ix['calB']]),('query',ix['query'])]:old=np.array(rows[u][name][role]['scores']);err=max(float(np.max(abs(models[name].decision_function(x[ii])-old))),float(np.max(abs(refit.decision_function(x[ii])-old))));maxerr=max(maxerr,err);assert err<1e-7
    prior_audit=json.loads((prior/'verification_public.json').read_text());assert prior_audit['state']=='PASS';checks=validate_rows(root,rows,d,ix,'Mei9');result=dict(state='PASS',dataset='Mei9',users=9,known_attackers_each=8,new_matched_identity_only_control_verified=True,archived_cycle_verification_reused=True,archived_personal_checkpoints=18,workpoint_checks=checks,max_cached_refit_error=maxerr,query_audio_model_forwards=0)
    (root/'verification_public.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

def main(a):
    root=Path(a.out)
    if a.legacy_mei:return legacy(a,root)
    d=dict(np.load(root/'inputs_private.npz'));data=dict(np.load(root/'spectra_private.npz'));ix={k:np.array(v) for k,v in json.loads((root/'split_private.json').read_text()).items()};users=sorted(np.unique(d['user']).tolist());n=len(users)
    for i,k in enumerate(ix):
        for other in list(ix)[:i]:assert not set(ix[k])&set(ix[other])
    for u in users:assert len(ix['register'][d['user'][ix['register']]==u])==12 and len(ix['calA'][d['user'][ix['calA']]==u])==4 and len(ix['calB'][d['user'][ix['calB']]==u])==4 and len(ix['query'][d['user'][ix['query']]==u])==24
    assert set(d['user'][ix['fit']])==set(users)
    for k in ['fit','register','dev','calA','calB']:assert not set(d['text_id'][ix[k]])&set(d['text_id'][ix['query']])
    allowed=set(np.r_[ix['fit'],ix['register'],ix['dev']]);assert set(data['audio_indices'])==allowed and not data['b'][np.r_[ix['calA'],ix['calB'],ix['query']]].any();ta=dict(np.load(root/'teacher_audio_private.npz'));ti=ta['indices'];expected=set(np.r_[np.concatenate([ix['fit'][d['user'][ix['fit']]==u][:4] for u in users]),ix['register']]);assert set(ti)==expected and len(ti)==16*n;assert np.all((ta['lengths']>0)&(ta['lengths']<=1));teacher=json.loads((root/'teacher_audit_public.json').read_text());assert teacher['original_loaded_state_verified'] and teacher['backbone_maxdelta']>0 and teacher['users']==n
    initial=json.loads((root/'initial_training_private.json').read_text());assert initial['users']==users and set(initial['fit_indices'])==set(ix['fit'])
    for mode in ['audio','no_audio']:
        audit=json.loads((root/(mode+'_training_private.json')).read_text());assert audit['users']==users and set(audit['training_indices'])==expected and audit['steps']==100 and audit['sampler']=='random8identitiesx4' and audit['query_audio_model_forwards']==0
    ck=torch.load(root/'shared_gan_private.pt',map_location='cpu',weights_only=False);assert set(ck['fit_indices'])==set(ix['fit']) and set(ck['develop_indices'])==set(ix['dev']);del ck
    x=StandardScaler().fit(d['x'][ix['fit']]).transform(d['x']).astype('float32');shape=norm(x[:,:40]);stats=norm(StandardScaler().fit(d['statistics'][ix['fit']]).transform(d['statistics']));nets={}
    for mode in ['audio','no_audio']:net=Encoder(x.shape[1]);net.load_state_dict(torch.load(root/(mode+'_student_private.pt'),map_location='cpu',weights_only=False));nets[mode]=encode(net,x)
    refs=dict(raw_shape=shape,raw_statistics=stats,no_audio_encoder=nets['no_audio'],no_audio_baseline=np.concatenate([shape,nets['no_audio']],1)/np.sqrt(2),audio_encoder=nets['audio'],audio_baseline=np.concatenate([shape,nets['audio']],1)/np.sqrt(2));base=refs['audio_baseline'];sc=joblib.load(root/'feature_scalers_private.joblib');assert all(v.n_samples_seen_==len(ix['fit']) for v in sc.values());rows={int(k):v for k,v in json.loads((root/'scores_private.json').read_text()).items()};tt=json.loads((root/'thresholds_private.json').read_text());lock=json.loads((root/'selection_public.json').read_text());choice=lock['selected'];pre=np.unique(np.r_[ix['register'],ix['calA'],ix['calB']]);allix=np.unique(np.r_[pre,ix['query']]);missing=np.setdiff1d(allix,pre);cache={};maxfeat=maxinfer=maxrefit=0.;mismatches=0;comparisons=0;checkpoints=0
    for u in users:
        folder=root/f'owner{u}';en=ix['register'][d['user'][ix['register']]==u];audit=json.loads((folder/'training_audit_private.json').read_text());assert set(audit['train_indices'])==set(en) and audit['other_user_train_records']==0 and audit['query_audio_model_forwards']==0 and audit['reset_shared'];z={};za={}
        for step in [25,100]:
            ck=torch.load(folder/f'gan{step}_private.pt',map_location='cpu',weights_only=False);assert ck['owner']==u and set(ck['train_indices'])==set(en);g=Generator().cuda();f=Generator().cuda();g.load_state_dict(ck['G']);f.load_state_dict(ck['F']);old=dict(np.load(folder/f'features{step}_{len(allix)}_private.npz'));assert np.array_equal(old.pop('indices'),allix);pa=fast_features(g,f,data['a'],pre);qa=fast_features(g,f,data['a'],missing);z[step]={};za[step]={}
            for k in BRANCHES:
                actual=np.empty_like(old[k]);actual[np.searchsorted(allix,pre)]=pa[k];actual[np.searchsorted(allix,missing)]=qa[k];err=float(np.max(abs(actual-old[k])));maxfeat=max(maxfeat,err);assert err<3e-4;v=np.zeros((len(x),old[k].shape[1]),dtype='float32');v[allix]=norm(sc[k].transform(old[k]));z[step][k]=v;v2=np.zeros_like(v);v2[allix]=norm(sc[k].transform(actual));za[step][k]=v2
            assert (root/'selection_public.json').stat().st_mtime_ns<=(folder/f'features{step}_{len(allix)}_private.npz').stat().st_mtime_ns;checkpoints+=1;del ck,g,f
        def variants(z):
            v=dict(refs)
            for step in [25,100]:
                for k in BRANCHES:v[f'personal{step}_{k}']=view(base,z,dict(step=step,branch=k,alpha=.5))
            v['selected_cycle']=view(base,z,choice);v['cycle_only100']=z[100]['cycle256'];return v
        models=joblib.load(folder/'rbf_heads_private.joblib');cached=variants(z);actual=variants(za)
        for method,v in cached.items():
            refit=head(v,d,ix,u);assert models[method].shape_fit_[0]==12*n
            for role,ii in [('calibration',np.r_[ix['calA'],ix['calB']]),('query',ix['query'])]:
                old=np.array(rows[u][method][role]['scores']);err=max(float(np.max(abs(models[method].decision_function(v[ii])-old))),float(np.max(abs(refit.decision_function(v[ii])-old))));maxrefit=max(maxrefit,err);assert err<1e-7;s=models[method].decision_function(actual[method][ii]);e=float(np.max(abs(s-old)));maxinfer=max(maxinfer,e);assert e<2e-3
                if role=='query':
                    for tau in tt[method][str(u)].values():mismatches+=int(np.sum((s>=tau)!=(old>=tau)));comparisons+=len(s)
        cache[u]=z;print('C3_VERIFY_OWNER',a.dataset,u,flush=True)
    for c in lock['choices']:
        pp=[]
        for u in users:
            v=view(base,cache[u],c);model=head(v,d,ix,u);cs=model.decision_function(v[ix['calA']]);s=model.decision_function(v[ix['calB']]);pp.append(metrics(s,d['user'][ix['calB']]==u,thresholds(cs,d['user'][ix['calA']]==u)['Cal_EER']))
        assert abs(c['hter']-np.mean([p['hter'] for p in pp]))<1e-7 and abs(c['auc']-np.mean([p['auc'] for p in pp]))<1e-7
    best=min(lock['choices'],key=lambda c:(c['hter'],-c['auc'],c['alpha'],c['step'],c['branch']));assert best==choice;checks=validate_rows(root,rows,d,ix,a.dataset);result=dict(state='PASS',dataset=a.dataset,users=n,known_attackers_each=n-1,personal_checkpoints=checkpoints,workpoint_checks=checks,selection_replayed=True,selection_precedes_query_features=True,all_users_identity_training=True,personal_only_own12=True,query_audio_model_forwards=0,query_text_training_overlap=0,max_feature_error=maxfeat,max_cached_refit_error=maxrefit,max_gpu_reextraction_score_error=maxinfer,gpu_query_decision_mismatches=mismatches,gpu_query_decisions_compared=comparisons,numerical_note='cached-feature SVM refits exact;GPU reextraction diagnostic reports any boundary flips,not silently assumed bit-exact')
    (root/'verification_public.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--out',required=True);p.add_argument('--legacy-mei');p.add_argument('--frontend');a=p.parse_args();torch.set_num_threads(2);torch.backends.cudnn.benchmark=True;main(a)
