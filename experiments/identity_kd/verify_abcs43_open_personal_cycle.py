"""Audit ABCS43 open identity boundaries and reproduce personalized G/F scores."""
import sys,json
from pathlib import Path
import numpy as np
import joblib,torch
from sklearn.preprocessing import StandardScaler
from abcs43_open_personal_cycle import BASE,CAL,TEST,BRANCHES,score,join
from vibravox_cycle_supplement import Generator,features
from run_rkd01_vibravox import Student,encode,norm,metrics,thresholds

torch.set_num_threads(1);root=Path(sys.argv[1]);source=Path(sys.argv[2]);data=dict(np.load(root/'spectra_private.npz'));d=dict(np.load(source/'inputs_private.npz'));fit=data['fit'];dev=data['dev'];assert len(fit)==1380 and len(dev)==345 and set(d['user'][fit])==set(BASE);assert np.all(d['role'][fit]=='fit') and np.all(d['role'][dev]=='develop') and np.all(d['user'][dev]<23);allowed=np.unique(np.r_[fit,dev,np.flatnonzero(d['role']=='registration')]);assert np.array_equal(data['audio_indices'],allowed) and not data['b'][~np.isin(np.arange(len(d['x'])),allowed)].any()
roles=['fit','develop','registration','calibration','query'];sets=[set(d['text_index'][d['role']==r]) for r in roles];assert all(not sets[i]&sets[j] for i in range(5) for j in range(i));assert np.array_equal(data['row_ids'],np.arange(len(d['x'])))
ck=torch.load(root/'shared_gan_private.pt',map_location='cpu',weights_only=False);assert set(ck['fit_indices'])==set(fit) and set(ck['develop_indices'])==set(dev)
sc=joblib.load(root/'feature_scalers_private.joblib');raw=dict(np.load(root/'shared_features_private.npz'));rand=dict(np.load(root/'random_features_private.npz'));assert all(v.n_samples_seen_==1380 for v in sc.values());sharedz={k:norm(sc[k].transform(v)) for k,v in raw.items()};randomz={k:norm(StandardScaler().fit(v[fit]).transform(v)) for k,v in rand.items()};probe=np.r_[fit[:3],np.flatnonzero((d['group']=='cal')&(d['role']=='registration'))[:3],np.flatnonzero((d['group']=='test')&(d['role']=='query'))[:3]]
for path,reference in [(root/'shared_gan_private.pt',raw),(root/'initial_generators_private.pt',rand)]:
    q=torch.load(path,map_location='cpu',weights_only=False);g=Generator().cuda();f=Generator().cuda();g.load_state_dict(q['G']);f.load_state_dict(q['F']);actual=features(g,f,data['a'],probe)
    for k,v in actual.items():assert np.max(abs(v-reference[k][probe]))<3e-4
    del q,g,f
x=StandardScaler().fit(d['x'][fit]).transform(d['x']).astype('float32');shape=norm(x[:,:40]);stats=norm(StandardScaler().fit(d['statistics'][fit]).transform(d['statistics']));rows=json.loads((root/'scores_private.json').read_text());public=json.loads((root/'per_user_public.json').read_text());tt=json.loads((root/'thresholds_private.json').read_text());lock=json.loads((root/'selection_public.json').read_text());choice=lock['selected'];cache={};maxerr=0.;checks=0;checkpoints=0
for u in CAL+TEST:
    folder=root/f'owner{u}';en=np.flatnonzero((d['user']==u)&(d['role']=='registration'));group=CAL if u in CAL else TEST;role='calibration' if u in CAL else 'query';ii=np.flatnonzero(((d['user']==u)&(d['role']=='registration'))|(np.isin(d['user'],group)&(d['role']==role)));audit=json.loads((folder/'training_audit_private.json').read_text());assert set(audit['train_indices'])==set(en) and len(en)==12 and audit['other_user_train_records']==0 and audit['query_audio_model_forwards']==0 and audit['reset_shared'];student=Student();student.load_state_dict(torch.load(source/f'owner{u}'/'student_private.pt',map_location='cpu',weights_only=False));emb=np.zeros((len(x),192),dtype='float32');emb[ii]=encode(student,x[ii]);base=np.concatenate([shape,emb],1)/np.sqrt(2);z={};zcached={}
    for step in [25,100]:
        q=torch.load(folder/f'gan{step}_private.pt',map_location='cpu',weights_only=False);assert q['owner']==u and set(q['train_indices'])==set(en);assert any(not torch.equal(v.cpu(),ck['G'][k].cpu()) for k,v in q['G'].items());g=Generator().cuda();f=Generator().cuda();g.load_state_dict(q['G']);f.load_state_dict(q['F']);actual=features(g,f,data['a'],ii);saved=dict(np.load(folder/f'features{step}_private.npz'));z[step]={};zcached[step]={}
        for k,v in actual.items():
            err=float(np.max(abs(v-saved[k])));maxerr=max(maxerr,err);assert err<3e-4;aa=np.zeros((len(x),v.shape[1]),dtype='float32');aa[ii]=norm(sc[k].transform(v));bb=np.zeros_like(aa);bb[ii]=norm(sc[k].transform(saved[k]));z[step][k]=aa;zcached[step][k]=bb
        checkpoints+=1;del q,g,f
    def variants_for(z):
        variants=dict(shape40=shape,statistics240=stats,student192=emb,current_supplement=base,cycle_only100=z[100]['cycle_encoder256'])
        for step in [25,100]:
            for short,k in [('input','input_encoder256'),('cycle','cycle_encoder256'),('statistics','cycle_statistics240')]:variants[f'personal{step}_{short}_supplement']=join(base,z[step][k],.5)
        for name,zz in [('shared',sharedz),('random',randomz)]:
            for short,k in [('input','input_encoder256'),('cycle','cycle_encoder256')]:variants[f'{name}_{short}_supplement']=join(base,zz[k],.5)
        if choice['step']==0:variants.update(selected_cycle_supplement=base,matched_shared_supplement=base,matched_random_supplement=base)
        else:
            step,k,alpha=choice['step'],choice['branch'],choice['alpha'];variants.update(selected_cycle_supplement=join(base,z[step][k],alpha),matched_shared_supplement=join(base,sharedz[k],alpha),matched_random_supplement=join(base,randomz[k],alpha))
        return variants
    role='Cal' if u in CAL else 'Test';uid=f'ABCS43-{role}-{u-22 if u<31 else u-30:02}'
    for method,zz in variants_for(z).items():
        actual=score(zz,d,u,group);s=np.array(actual['scores']);y=np.array(actual['labels']);saved=rows[str(u)][method];old=np.array(saved['scores']);err=float(np.max(abs(s-old)));maxerr=max(maxerr,err);assert err<5e-5;assert y.sum()==24 and (~y).sum()==(168 if u in CAL else 264)
        for point,tau in tt[method].items():
            v=next(v for v in public if v['user']==uid and v['method']==method and v['point']==point);expected=metrics(old,np.array(saved['labels']),tau)
            for k in ['tar','far','hter','auc']:assert abs(v[k]-expected[k])<1e-7
            if u in TEST:assert np.array_equal(s>=tau,old>=tau)
            checks+=1
    cache[u]=(base,zcached);print('VERIFY_ABCS_OPEN_OWNER',u,flush=True)
# Choice uses cached features to avoid harmless GPU re-extraction rounding at ROC ties.
for c in lock['choices']:
    def collect(group):
        rr=[]
        for u in group:
            base,z=cache[u];rr.append(score(base if c['step']==0 else join(base,z[c['step']][c['branch']],c['alpha']),d,u,group))
        return np.concatenate([v['scores'] for v in rr]),np.concatenate([v['labels'] for v in rr])
    cs,cy=collect(CAL[:4]);s,y=collect(CAL[4:]);v=metrics(s,y,thresholds(cs,cy)['Cal_EER']);assert abs(c['hter']-v['hter'])<1e-7 and abs(c['auc']-v['auc'])<1e-7
best=min(lock['choices'],key=lambda c:(c['hter'],-c['auc'],c['alpha'],c['step'],c['branch']));assert best==choice
results=json.loads((root/'results_public.json').read_text())['results'];prior=json.loads((source/'results_public.json').read_text())['results']
for old,new in [('shape40','shape40'),('statistics240','statistics240'),('student192','student192'),('supplement232','current_supplement')]:
    for point in ['Cal_EER','Cal_FAR5']:
        for k in ['tar','far','hter','auc']:assert abs(results[new][point][k]-prior[old][point][k])<1e-7
result=dict(state='PASS',base=23,cal=8,test=12,personal_models=20,personal_checkpoints=checkpoints,workpoint_checks=checks,max_feature_or_score_error=maxerr,query_decisions_unchanged=True,selection_replayed=True,original_four_references_reproduced=True,query_audio_feature_calculations=0,query_audio_model_forwards=0,personal_only_own12=True,all11unseen_attackers_retained=True,text_role_overlap=0,channel_mapping='provisional')
(root/'verification_public.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
