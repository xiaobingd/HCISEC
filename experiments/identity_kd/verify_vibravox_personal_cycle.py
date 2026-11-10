"""Audit train identities, owner-only adaptation and saved vibration-only inference."""
import sys,json
from pathlib import Path
import numpy as np
import joblib,torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from vibravox_cycle_supplement import Generator,features,scores,BASE,CAL,TEST
from run_rkd01_vibravox import Student,encode,norm,metrics,thresholds

torch.set_num_threads(2);source=Path(sys.argv[1]);shared=Path(sys.argv[2]);personal=Path(sys.argv[3]);prior=Path(sys.argv[4]);d=dict(np.load(source/'inputs_private.npz'));data=dict(np.load(shared/'spectra_private.npz'));fit=data['fit'];assert np.all((d['user'][fit]<35)&(d['role'][fit]=='fit'));gan=torch.load(shared/'best_generators_private.pt',map_location='cpu',weights_only=False);assert set(gan['fit_indices'])==set(fit);assert np.all((d['user'][gan['validation_indices']]<35)&(d['role'][gan['validation_indices']]=='develop'));assert not np.isin(d['user'][data['audio_indices']],CAL+TEST).any()
reg=dict(np.load(personal/'registration_audio_spectra_private.npz'));assert np.all(np.isin(d['user'][reg['indices']],CAL+TEST)&(d['role'][reg['indices']]=='registration')) and len(reg['indices'])==288
sc=joblib.load(shared/'feature_scalers_private.joblib');raw=dict(np.load(shared/'gan_features_private.npz'));rz=dict(np.load(shared/'random_features_private.npz'));zshared={k:norm(sc[k].transform(v)) for k,v in raw.items()};zrandom={k:norm(StandardScaler().fit(v[fit]).transform(v)) for k,v in rz.items()};x=StandardScaler().fit(d['x'][fit]).transform(d['x']).astype('float32');shape=norm(x[:,:40]);rows=json.loads((personal/'scores_private.json').read_text());public=json.loads((personal/'per_user_public.json').read_text());tt=json.loads((personal/'thresholds_private.json').read_text());lock=json.loads((personal/'selection_public.json').read_text());chosen=lock['selected'];cache={};maxerror=0.;checks=0;modelcount=0
probe=np.r_[fit[:3],np.flatnonzero((d['group']=='cal')&(d['role']=='registration'))[:3],np.flatnonzero((d['group']=='test')&(d['role']=='query'))[:3]]
for path,reference in [(shared/'best_generators_private.pt',raw),(shared/'initial_generators_private.pt',rz)]:
    ck=torch.load(path,map_location='cpu',weights_only=False);g=Generator().cuda();f=Generator().cuda();g.load_state_dict(ck['G']);f.load_state_dict(ck['F']);actual=features(g,f,data['a'],probe)
    for branch,v in actual.items():assert np.max(np.abs(v-reference[branch][probe]))<2e-4
    del g,f,ck
for u in CAL+TEST:
    folder=personal/f'owner{u}';en=np.flatnonzero((d['user']==u)&(d['role']=='registration'));group=CAL if u in CAL else TEST;q=np.flatnonzero(np.isin(d['user'],group)&(d['role']=='query'));ii=np.r_[en,q];audit=json.loads((folder/'training_audit_private.json').read_text());assert set(audit['train_indices'])==set(en) and set(audit['query_indices'])==set(q);assert audit['other_user_train_records']==0 and audit['query_audio_reads']==0 and audit['reset_from_shared'] and audit['updated_g_maxdelta']>0
    student=Student();student.load_state_dict(torch.load(prior/'outer'/f'owner{u}'/'V4_rho0.3_student_private.pt',map_location='cpu',weights_only=False));b=np.zeros((len(x),232),dtype='float32');b[ii]=np.concatenate([shape[ii],encode(student,x[ii])],1)/np.sqrt(2);z={}
    for step in [25,100]:
        ck=torch.load(folder/f'generators{step}_private.pt',map_location='cpu',weights_only=False);assert ck['owner']==u and set(ck['registration_indices'])==set(en);g=Generator().cuda();f=Generator().cuda();g.load_state_dict(ck['G']);f.load_state_dict(ck['F']);actual=features(g,f,data['a'],ii);saved=dict(np.load(folder/f'features{step}_private.npz'));modelcount+=1;z[step]={}
        for branch,v in actual.items():
            error=float(np.max(np.abs(v-saved[branch])));maxerror=max(maxerror,error);assert error<2e-4;zz=np.zeros((len(x),v.shape[1]),dtype='float32');zz[ii]=norm(sc[branch].transform(v));z[step][branch]=zz
        del g,f,ck
    variants=dict(shape40=shape,baseline232=b,cycle_only100=z[100]['cycle_encoder256'],input_only100=z[100]['input_encoder256'])
    for step in [25,100]:
        for branch in ['input_encoder256','cycle_encoder256','cycle_statistics240']:variants[f'personal{step}_{branch}']=np.concatenate([b,.5*z[step][branch]],1)/np.sqrt(1.25)
    if chosen['step']==0:variants.update(personal_selected=b,matched_shared_branch=b,matched_random_branch=b)
    else:
        step=chosen['step'];branch=chosen['branch'];alpha=chosen['alpha'];variants.update(personal_selected=np.concatenate([b,alpha*z[step][branch]],1)/np.sqrt(1+alpha*alpha),matched_shared_branch=np.concatenate([b,alpha*zshared[branch]],1)/np.sqrt(1+alpha*alpha),matched_random_branch=np.concatenate([b,alpha*zrandom[branch]],1)/np.sqrt(1+alpha*alpha))
    for name,v in variants.items():
        actual=scores(v,d,u,group);s=np.array(actual['scores']);y=np.array(actual['labels']);error=float(np.max(np.abs(s-np.array(rows[str(u)][name]['scores']))));maxerror=max(maxerror,error);assert error<2e-5
    role='Cal' if u in CAL else 'Test';uid=f'Vib59-{role}-{u-34 if u<47 else u-46:02}'
    for name,points in tt.items():
        row=rows[str(u)][name];s=np.array(row['scores']);y=np.array(row['labels']);assert y.sum()==24 and (~y).sum()==264
        for point,tau in points.items():
            v=next(v for v in public if v['user']==uid and v['method']==name and v['point']==point);expected=metrics(s,y,tau)
            for k in ['tar','far','hter','auc']:assert abs(v[k]-expected[k])<1e-7
            checks+=1
    cache[u]=(b,z)
    print('VERIFY_PERSONAL_OWNER',u,flush=True)
# Selection re-evaluated using exactly the two Cal subcohorts.
for c in lock['choices']:
    def collect(group):
        rr=[]
        for u in group:
            b,z=cache[u];v=b if c['step']==0 else np.concatenate([b,c['alpha']*z[c['step']][c['branch']]],1)/np.sqrt(1+c['alpha']**2);rr.append(scores(v,d,u,group))
        return np.concatenate([v['scores'] for v in rr]),np.concatenate([v['labels'] for v in rr])
    cs,cy=collect(CAL[:6]);qs,qy=collect(CAL[6:]);r=metrics(qs,qy,thresholds(cs,cy)['Cal_EER']);assert abs(c['hter']-r['hter'])<1e-6 and abs(c['auc']-r['auc'])<1e-6
best=min(lock['choices'],key=lambda c:(c['hter'],-c['auc'],c['alpha'],c['step'],c['branch']));assert best==chosen
# Compare reused baseline with prior published reference; no silent pipeline drift.
new=json.loads((personal/'results_public.json').read_text())['results']['baseline232'];old=json.loads((Path('/autodl-fs/data/shape_supplement_rkd03_20261110')/'results_public.json').read_text())['supplement']
for point in new:
    for k in ['tar','far','hter']:assert abs(new[point][k]-old[point][k])<1e-7
out=dict(state='PASS',shared_base_users=35,personal_owners=24,personal_checkpoints_verified=modelcount,per_user_workpoint_checks=checks,max_inference_or_feature_error=maxerror,cal_selection_recomputed=True,prior_baseline_reproduced=True,query_audio_reads=0,owner_only_registration_training=True,all_attackers_retained=11)
(personal/'verification_public.json').write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True)
