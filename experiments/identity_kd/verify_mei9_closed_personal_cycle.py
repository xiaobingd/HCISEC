"""Saved-model reproduction and sample/identity split audit for true Mei9 closed set."""
import sys,json
from pathlib import Path
import numpy as np
import torch,joblib
from sklearn.preprocessing import StandardScaler
from mei9_closed_personal_cycle import indices,Generator,gan_features,matrices,svm,USERS,BRANCHES
from run_mei9_shape_v4 import Student,encode,replay
from run_rkd01_vibravox import norm,metrics,thresholds

torch.set_num_threads(2);root=Path(sys.argv[1]);d=dict(np.load(sys.argv[2]));ix=indices(d);saved=json.loads((root/'split_private.json').read_text());assert all(saved[k]==v.tolist() for k,v in ix.items());assert len(set(d['sample_id']))==2695;data=dict(np.load(root/'prepared_private.npz'));assert set(d['user'][ix['fit']])==set(USERS);allowed=set(np.r_[ix['fit'],ix['dev'],ix['register']]);assert set(data['audio_indices'])==allowed
assert set(d['word'][ix['fit']])==set(list(range(8))+list(range(10,18)))
assert set(d['word'][ix['dev']])=={8,9,18,19}
assert all(set(d['word'][ix[k]])=={20,21,22,23} for k in ['register','calA','calB','sameword'])
assert set(d['word'][ix['crossword']])==set(range(24,30))
t=dict(np.load(root/'teacher_targets_private.npz'));expect=set(np.r_[np.concatenate([replay(d,u) for u in USERS]),ix['register']]);assert set(t['indices'])==expect and len(expect)==144
audit=json.loads((root/'student_training_private.json').read_text());assert set(audit['base_fit_indices'])==set(ix['fit']) and set(audit['distill_indices'])==expect and audit['all_training_identities']==USERS and audit['query_audio_forwards']==0
ck=torch.load(root/'shared_gan_private.pt',map_location='cpu',weights_only=False);assert set(ck['fit_indices'])==set(ix['fit']) and set(ck['dev_indices'])==set(ix['dev']);del ck
net=Student();net.load_state_dict(torch.load(root/'student_private.pt',map_location='cpu',weights_only=False));emb=encode(net,data['x']);base=np.concatenate([data['shape'],emb],1)/np.sqrt(2);sc=joblib.load(root/'gan_scalers_private.joblib');lock=json.loads((root/'selection_public.json').read_text());choice=lock['selected'];rows=json.loads((root/'scores_private.json').read_text());users=json.loads((root/'per_user_public.json').read_text());tt=json.loads((root/'thresholds_private.json').read_text());allix=np.unique(np.r_[ix['register'],ix['calA'],ix['calB'],ix['sameword'],ix['crossword']]);cal=np.r_[ix['calA'],ix['calB']];cache={};maxerr=0.;checks=0;checkpoints=0
for u in USERS:
    folder=root/f'owner{u}';audit=json.loads((folder/'gan_training_private.json').read_text());en=ix['register'][d['user'][ix['register']]==u];assert set(audit['train_indices'])==set(en) and len(en)==12 and audit['other_user_records']==0 and audit['query_audio_model_forwards']==0
    zz={};cachedz={}
    for step in [25,100]:
        ck=torch.load(folder/f'gan{step}_private.pt',map_location='cpu',weights_only=False);assert ck['owner']==u and set(ck['train_indices'])==set(en);g=Generator(3,1).cuda();f=Generator(1,3).cuda();g.load_state_dict(ck['G']);f.load_state_dict(ck['F']);raw=gan_features(g,f,data['a'],allix);saved=dict(np.load(folder/f'features{step}_{len(allix)}_private.npz'));zz[step]={};cachedz[step]={}
        for k,v in raw.items():
            err=float(np.max(np.abs(v-saved[k])));maxerr=max(maxerr,err);assert err<3e-4;a=np.zeros((len(base),v.shape[1]),dtype='float32');a[allix]=norm(sc[k].transform(v));zz[step][k]=a;cached=np.zeros_like(a);cached[allix]=norm(sc[k].transform(saved[k]));cachedz[step][k]=cached
        checkpoints+=1;del g,f,ck
    def variants_for(z):
        variants=dict(raw120=data['shape'],statistics720=data['stats'],student192=emb,baseline312=base)
        for step in [25,100]:
            for k in BRANCHES:variants[f'personal{step}_{k}']=matrices(base,z,dict(step=step,branch=k,alpha=.5))
        variants['selected_cycle']=matrices(base,z,choice);variants['cycle_only100']=z[100]['cycle256'];return variants
    cache[u]=cachedz;variants=variants_for(zz);cachedvariants=variants_for(cachedz);models=joblib.load(folder/'rbf_heads_private.joblib')
    for method,x in variants.items():
        # Refit same fixed classifier to independently validate saved head and known negatives.
        refit=svm(cachedvariants[method],d,ix,u);trainy=d['user'][ix['register']]==u;assert trainy.sum()==12 and (~trainy).sum()==96
        for cond,j in [('calibration',cal),('sameword',ix['sameword']),('crossword',ix['crossword'])]:
            s=models[method].decision_function(x[j]);s2=refit.decision_function(cachedvariants[method][j]);old=np.array(rows[str(u)][method][cond]['scores']);inference_error=float(np.max(abs(s-old)));refit_error=float(np.max(abs(s2-old)));maxerr=max(maxerr,inference_error);assert inference_error<2e-4 and refit_error<1e-7
            if cond!='calibration':
                for tau in tt[method][str(u)].values():assert np.array_equal(s>=tau,old>=tau)
        c=rows[str(u)][method]['calibration'];cs=np.array(c['scores']);cy=np.array(c['labels']);assert cy.sum()==8 and (~cy).sum()==64;threshold=thresholds(cs,cy)
        for point,tau in threshold.items():
            stored_tau=tt[method][str(u)][point]
            assert (stored_tau==tau) or abs(stored_tau-tau)<1e-7
            if point=='Cal_FAR5':assert (cs[~cy]>=tau).mean()<=.05
            for cond in ['sameword','crossword']:
                row=rows[str(u)][method][cond];s=np.array(row['scores']);y=np.array(row['labels']);assert int(y.sum())==(16 if cond=='sameword' else 24) and int((~y).sum())==(128 if cond=='sameword' else 192);v=metrics(s,y,tau);p=next(p for p in users if p['user']==f'Mei9-U{u:02}' and p['method']==method and p['condition']==cond and p['point']==point)
                for k in ['tar','far','hter','auc']:assert abs(v[k]-p[k])<1e-7
                checks+=1
    print('VERIFIED_MEI_CLOSED_USER',u,flush=True)
for c in lock['choices']:
    pp=[]
    for u in USERS:
        x=matrices(base,cache[u],c);model=svm(x,d,ix,u);a=ix['calA'];b=ix['calB'];cs=model.decision_function(x[a]);s=model.decision_function(x[b]);pp.append(metrics(s,d['user'][b]==u,thresholds(cs,d['user'][a]==u)['Cal_EER']))
    assert abs(c['hter']-np.mean([p['hter'] for p in pp]))<1e-6 and abs(c['auc']-np.mean([p['auc'] for p in pp]))<1e-6
best=min(lock['choices'],key=lambda c:(c['hter'],-c['auc'],c['alpha'],c['step'],c['branch']));assert best==choice
result=dict(state='PASS',identities=9,closed_all_nine_train=True,known_attackers_each=8,personal_checkpoints=checkpoints,workpoint_checks=checks,max_feature_or_score_error=maxerr,original_cached_feature_refit_tolerance=1e-7,query_decisions_unchanged_under_gpu_reextraction=True,selection_replayed=True,query_audio_model_forwards=0,personal_gan_only_own12=True,splits_disjoint=True)
(root/'verification_public.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
