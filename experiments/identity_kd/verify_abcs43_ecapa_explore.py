"""Recompute both rounds from saved vibration-only students and private scores."""
import sys,json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from explore_abcs43_ecapa import setup,infer,CAL,TEST,BASE
from run_rkd01_vibravox import Student

torch.set_num_threads(1);source=Path(sys.argv[1]);r1=Path(sys.argv[2]);r2=Path(sys.argv[3]);d,x,initial,shape=setup(source);maxerr=0.;workpoints=0;models=0
original=torch.load(source/'model'/'embedding_model.ckpt',map_location='cpu',weights_only=False)
for root,methods in [(r1,['no_audio','frozen','wrong_identity','same_identity_shuffle']),(r2,['projection','attentive'])]:
    public=json.loads((root/'per_user_public.json').read_text());tt=json.loads((root/'thresholds_private.json').read_text());cached={}
    for u in CAL+TEST:
        role='Cal' if u in CAL else 'Test';uid=f'ABCS43-{role}-{u-22 if u<31 else u-30:02}';expected=set(np.flatnonzero(((d['user']<23)&d['replay'])|((d['user']==u)&(d['role']=='registration'))))
        cached[u]=json.loads(((root/f'owner{u}'/'scores_private.json') if root==r1 else (r1/f'owner{u}'/'scores_private.json')).read_text())
        for method in methods:
            folder=root/f'owner{u}' if root==r1 else root/method/f'owner{u}';modelpath=folder/(method+'_private.pt') if root==r1 else folder/'student_private.pt';auditpath=folder/(method+'_audit_private.json') if root==r1 else folder/'student_audit_private.json';audit=json.loads(auditpath.read_text());assert set(audit['training_indices'])==expected and audit['query_audio_forwards']==0
            if root==r2:
                ta=json.loads((folder/'teacher_audit_private.json').read_text());assert set(ta['training_indices'])==expected and ta['frozen_delta']==0 and ta['trainable_delta']>0 and ta['query_audio_forwards']==0
                state=torch.load(folder/'teacher_private.pt',map_location='cpu',weights_only=False);prefix=('fc.',) if method=='projection' else ('asp.','asp_bn.','fc.')
                assert set(state)==set(original) and all(torch.equal(v,original[k]) for k,v in state.items() if not k.startswith(prefix))
                assert any(not torch.equal(v,original[k]) for k,v in state.items() if k.startswith(prefix));del state
            net=Student();net.load_state_dict(torch.load(modelpath,map_location='cpu',weights_only=False));actual=infer(net,x,d,shape,u);models+=1
            stored=cached[u] if root==r1 else json.loads((folder/'scores_private.json').read_text())
            for branch,v in actual.items():
                methodname=method+'_'+branch;saved=stored[methodname] if root==r1 else stored[branch];error=float(np.max(np.abs(np.array(v['scores'])-saved['scores'])));maxerr=max(maxerr,error);assert error<1e-6;cached[u][methodname]=v
        for method,taupoints in tt.items():
            q=cached[u][method];s=np.array(q['scores']);y=np.array(q['labels']);assert y.sum()==24 and (~y).sum()==(168 if u<31 else 264)
            for point,tau in taupoints.items():
                v=next(p for p in public if p['user']==uid and p['method']==method and p['point']==point);tar=float((s[y]>=tau).mean());far=float((s[~y]>=tau).mean());assert abs(v['tar']-tar)<1e-7 and abs(v['far']-far)<1e-7 and abs(v['hter']-(1-tar+far)/2)<1e-7 and abs(v['auc']-roc_auc_score(y,s))<1e-7;workpoints+=1
    # Independently recompute route choice from first4 vs last4 Cal identities.
    if root==r2:
        candidates=['current_supplement','projection_supplement','attentive_supplement'];values=[]
        from run_rkd01_vibravox import thresholds,metrics
        for method in candidates:
            def collect(group):
                scores=[];labels=[];mask=np.repeat(np.isin(CAL,group),24)
                for u in group:
                    q=cached[u][method];scores.extend(np.array(q['scores'])[mask]);labels.extend(np.array(q['labels'])[mask])
                return np.array(scores),np.array(labels)
            cs,cy=collect(CAL[:4]);qs,qy=collect(CAL[4:]);v=metrics(qs,qy,thresholds(cs,cy)['Cal_EER']);values.append((v['hter'],-v['auc'],candidates.index(method),method))
        lock=json.loads((r2/'selection_public.json').read_text());assert min(values)[3]==lock['selected_method']
    result=dict(state='PASS',round=1 if root==r1 else 2,verified_saved_models=80 if root==r1 else 40,per_user_workpoint_checks=len(public),max_saved_score_error=maxerr,query_audio_forwards=0,training_registration_query_isolation=True,all_predeclared_users_and_methods_reported=True,calibration_selection_recomputed=root==r2)
    (root/'verification_public.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
print('VERIFIED_ALL',models,workpoints,flush=True)
