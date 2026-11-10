"""Two predeclared exploratory rounds, fixed ABCS43 data and calibration-only selection.

R1: matched-budget no-Audio/frozen/personal-teacher/mismatch controls.
R2: regularized partial ECAPA fine-tuning; student budget unchanged.
All sample scores, model weights and numeric thresholds remain private.
"""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import argparse,json,sys,copy
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.preprocessing import StandardScaler
from run_rkd01_vibravox import Student,norm,supcon,encode,metrics,thresholds
from run_registration_rkd02 import adapt_student,dump
BASE=list(range(23));CAL=list(range(23,31));TEST=list(range(31,43))

def setup(source):
    d=dict(np.load(source/'inputs_private.npz'));fit=(d['user']<23)&(d['role']=='fit');x=StandardScaler().fit(d['x'][fit]).transform(d['x']).astype('float32');initial=Student();initial.load_state_dict(torch.load(source/'base_student_private.pt',map_location='cpu',weights_only=False)['state']);return d,x,initial,norm(x[:,:40])

def infer(net,x,d,shape,u):
    group=CAL if u in CAL else TEST;role='calibration' if u in CAL else 'query';en=np.flatnonzero((d['user']==u)&(d['role']=='registration'));q=np.flatnonzero(np.isin(d['user'],group)&(d['role']==role));ii=np.r_[en,q];z=encode(net,x[ii]);s=z[len(en):]@norm(z[:len(en)].mean(0));j=np.concatenate([shape[ii],z],1)/np.sqrt(2);joint=j[len(en):]@norm(j[:len(en)].mean(0));return dict(student=dict(scores=s.tolist(),labels=(d['user'][q]==u).tolist()),supplement=dict(scores=joint.tolist(),labels=(d['user'][q]==u).tolist()))

def source_rows(source):
    return {u:json.loads((source/f'owner{u}'/'scores_private.json').read_text()) for u in CAL+TEST}

def evaluate(rows,root,description):
    public={};per=[];private={};paired={};methods=list(rows[23])
    for m in methods:
        c=[rows[u][m] for u in CAL];cs=np.concatenate([v['scores'] for v in c]);cy=np.concatenate([v['labels'] for v in c]);ts=thresholds(cs,cy);assert np.mean(cs[~cy]>=ts['Cal_FAR5'])<=.05;private[m]=ts;public[m]={}
        for point,tau in ts.items():
            ss=np.concatenate([rows[u][m]['scores'] for u in TEST]);yy=np.concatenate([rows[u][m]['labels'] for u in TEST]);r=metrics(ss,yy,tau);rr=[]
            for role,uu in [('Cal',CAL),('Test',TEST)]:
                for j,u in enumerate(uu,1):
                    v=rows[u][m];p=metrics(np.array(v['scores']),np.array(v['labels']),tau);per.append(dict(user=f'ABCS43-{role}-{j:02}',role=role,method=m,point=point,**p))
                    if role=='Test':rr.append(p)
            r.update(worst_target_far=max(v['far'] for v in rr),lowest_target_tar=min(v['tar'] for v in rr),macro_auc=float(np.mean([v['auc'] for v in rr])));public[m][point]=r
    rng=np.random.RandomState(20261111)
    for m in methods:
        if not m.endswith('_supplement'):continue
        for control in ['shape40','no_audio_supplement','current_supplement','frozen_supplement']:
            if control not in methods or control==m:continue
            delta=np.array([next(v['hter'] for v in per if v['user']==f'ABCS43-Test-{j:02}' and v['method']==m and v['point']=='Cal_EER')-next(v['hter'] for v in per if v['user']==f'ABCS43-Test-{j:02}' and v['method']==control and v['point']=='Cal_EER') for j in range(1,13)])
            boot=np.array([delta[rng.choice(12,12,replace=True)].mean() for _ in range(5000)]);paired[m+'_minus_'+control]=dict(hter_delta=float(delta.mean()),identity_bootstrap95=np.quantile(boot,[.025,.975]).tolist(),improved_targets=int((delta<0).sum()))
    dump(root/'thresholds_private.json',private);dump(root/'per_user_public.json',per);dump(root/'results_public.json',dict(state='complete',description=description,base=23,cal=8,test=12,K=12,seed=42,results=public,paired=paired,limits=['single seed','same explored cohort as previous experiment;not independent replication','physical channel mapping provisional','no query Audio']))
    return public

def load_controls(source,r1):
    old=source_rows(source);rows={u:{'shape40':old[u]['methods']['shape40'],'current_student':old[u]['methods']['student192'],'current_supplement':old[u]['methods']['supplement232']} for u in CAL+TEST}
    if r1 is not None:
        for u in CAL+TEST:
            q=json.loads((r1/f'owner{u}'/'scores_private.json').read_text());rows[u].update({m:v for m,v in q.items() if m not in rows[u]})
    return rows

def round1(source,out):
    d,x,initial,shape=setup(source);rows=load_controls(source,None)
    methods=[('no_audio','V0'),('frozen','V2'),('wrong_identity','V5'),('same_identity_shuffle','V4')]
    for u in CAL+TEST:
        folder=out/f'owner{u}';folder.mkdir(parents=True,exist_ok=True);t=dict(np.load(source/f'owner{u}'/'teacher_targets_private.npz'));indices=t['indices'];row=rows[u]
        for name,method in methods:
            target=t['frozen'] if name=='frozen' else t['adapted'];target=target.copy()
            if name=='same_identity_shuffle':
                rng=np.random.RandomState(71300+u)
                for v in BASE+[u]:
                    j=np.flatnonzero(d['user'][indices]==v);target[j]=target[np.roll(j,1)]
            path=folder/(name+'_private.pt')
            if path.exists():net=Student();net.load_state_dict(torch.load(path,map_location='cpu',weights_only=False))
            else:
                net,trace=adapt_student(d,x,initial,u,BASE,indices,target,method,.3);torch.save(net.state_dict(),path);dump(folder/(name+'_audit_private.json'),dict(training_indices=indices.tolist(),trace=trace,steps=100,method=method,query_audio_forwards=0))
            scored=infer(net,x,d,shape,u);row.update({name+'_'+k:v for k,v in scored.items()})
        dump(folder/'scores_private.json',row);print('ROUND1_OWNER',u,flush=True)
    evaluate(rows,out,'Round1 matched-budget controls; all four new student methods100updates, same initialization/batches/head/optimizer; current teacher reference reused; no tuning on Test');print('ROUND1_COMPLETE',flush=True)

def partial_teacher(source,out,u,model,original,cache,slots,mode):
    folder=out/mode/f'owner{u}';folder.mkdir(parents=True,exist_ok=True);path=folder/'teacher_targets_private.npz'
    if path.exists():t=dict(np.load(path));return t['indices'],t['adapted']
    prior=dict(np.load(source/f'owner{u}'/'teacher_targets_private.npz'));ix=prior['indices'];emb=model.mods.embedding_model;emb.load_state_dict(original);emb.eval();torch.manual_seed(4242+u);rng=np.random.RandomState(4242+u);prefixes=('fc.',) if mode=='projection' else ('asp.','asp_bn.','fc.')
    trainable=[];names=[]
    for name,p in emb.named_parameters():
        p.requires_grad_(name.startswith(prefixes))
        if p.requires_grad:trainable.append(p);names.append(name)
    assert trainable;device=next(emb.parameters()).device;jj=torch.tensor([slots[int(i)] for i in ix]);feat=cache['features'][jj].to(device);lens=cache['lengths'][jj].to(device);labels=torch.tensor([int(v) if v<23 else 23 for v in d_global['user'][ix]],device=device)
    def forward(j,augment=False):
        f=feat[j]
        if augment:
            f=f.clone();tlen=f.shape[1];flen=f.shape[2]
            for b in range(len(f)):
                valid=max(1,int(float(lens[j[b]])*tlen));tw=max(1,int(valid*.05));ts=rng.randint(max(1,valid-tw+1));fs=rng.randint(max(1,flen-4+1));f[b,ts:ts+tw,:]=0;f[b,:,fs:fs+4]=0
        return F.normalize(emb(f,lens[j]).squeeze(1),dim=1)
    with torch.no_grad():frozen=forward(torch.arange(len(ix),device=device)).detach()
    assert np.allclose(frozen.cpu().numpy(),prior['frozen'],atol=1e-6)
    head=nn.Linear(192,24,bias=False).to(device)
    with torch.no_grad():head.weight.copy_(F.normalize(torch.stack([frozen[labels==j].mean(0) for j in range(24)]),dim=1))
    lr=1e-5 if mode=='projection' else 3e-6;steps=60 if mode=='projection' else 80
    opt=torch.optim.AdamW([dict(params=trainable,lr=lr),dict(params=head.parameters(),lr=1e-4)],weight_decay=1e-4);anchor=[p.detach().clone() for p in trainable];trace=[]
    for step in range(1,steps+1):
        uu=rng.choice(BASE,4,replace=False);batch=np.r_[rng.choice(np.flatnonzero(d_global['user'][ix]==u),4,replace=False),np.concatenate([rng.choice(np.flatnonzero(d_global['user'][ix]==v),2,replace=False) for v in uu])];j=torch.tensor(batch,device=device);z=forward(j,True);y=labels[j];cos=F.linear(z,F.normalize(head.weight,dim=1));margin=F.one_hot(y,24).to(cos.dtype)*.1;ce=F.cross_entropy(16*(cos-margin),y);con=supcon(z.cpu(),y.cpu()).to(device);preserve=(1-F.cosine_similarity(z,frozen[j],dim=1)).mean();l2=sum((p-a).square().mean() for p,a in zip(trainable,anchor));loss=ce+.1*con+.5*preserve+10*l2;opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(trainable+list(head.parameters()),1);opt.step()
        if step in [1,steps]:trace.append(dict(step=step,ce=float(ce.detach()),anchor=float(preserve.detach()),loss=float(loss.detach())))
    for p in emb.parameters():p.requires_grad_(False)
    with torch.no_grad():adapted=forward(torch.arange(len(ix),device=device)).cpu().numpy()
    state=emb.state_dict();frozen_delta=max(float((v.cpu()-original[k]).abs().max()) for k,v in state.items() if not k.startswith(prefixes));delta=max(float((v.cpu()-original[k]).abs().max()) for k,v in state.items() if k.startswith(prefixes));assert frozen_delta==0 and delta>0
    np.savez_compressed(path,indices=ix,frozen=frozen.cpu().numpy(),adapted=adapted);torch.save({k:v.cpu() for k,v in state.items()},folder/'teacher_private.pt');dump(folder/'teacher_audit_private.json',dict(training_indices=ix.tolist(),trainable_names=names,trainable_parameters=sum(p.numel() for p in trainable),steps=steps,lr=lr,trainable_delta=delta,frozen_delta=frozen_delta,query_audio_forwards=0,trace=trace));return ix,adapted

def choose_cal(rows,out):
    # Four calibration identities establish the workpoint; remaining four choose route.
    # Restrict query attackers to each sub-cohort, avoiding shared calibration query records.
    methods=['current_supplement','projection_supplement','attentive_supplement'];choices=[]
    for m in methods:
        def subset(owners):
            ss=[];yy=[]
            for u in owners:
                row=rows[u][m];mask=np.repeat(np.isin(CAL,owners),24);ss.extend(np.array(row['scores'])[mask]);yy.extend(np.array(row['labels'])[mask])
            return np.array(ss),np.array(yy)
        cs,cy=subset(CAL[:4]);s,y=subset(CAL[4:]);tau=thresholds(cs,cy)['Cal_EER'];r=metrics(s,y,tau);choices.append(dict(method=m,hter=r['hter'],auc=r['auc'],tar=r['tar'],far=r['far']))
    chosen=min(choices,key=lambda q:(q['hter'],-q['auc'],methods.index(q['method'])))['method'];dump(out/'selection_public.json',dict(selected_method=chosen,rule='Cal01-04 threshold;Cal05-08 route selection;subcohort query attackers only;minHTER then maxAUC then simpler;Test scores excluded from selection; prior scores preloaded as references',choices=choices));return chosen

def round2(source,r1,out):
    global d_global
    d,x,initial,shape=setup(source);d_global=d;rows=load_controls(source,r1);sys.path.insert(0,'/root');from run_ecapa_pilot import teacher_load
    model=teacher_load(source);original={k:v.cpu().clone() for k,v in model.mods.embedding_model.state_dict().items()};cache=torch.load(source/'audio_frontend_private.pt',map_location='cpu',weights_only=False);slots={int(i):j for j,i in enumerate(cache['indices'])}
    # Finish all candidate calibration fits and lock selection BEFORE any new Test fit.
    for u in CAL+TEST:
        for mode in ['projection','attentive']:
            folder=out/mode/f'owner{u}';folder.mkdir(parents=True,exist_ok=True);path=folder/'scores_private.json'
            if path.exists():scored=json.loads(path.read_text())
            else:
                ix,target=partial_teacher(source,out,u,model,original,cache,slots,mode);net,trace=adapt_student(d,x,initial,u,BASE,ix,target,'V4',.3);torch.save(net.state_dict(),folder/'student_private.pt');dump(folder/'student_audit_private.json',dict(training_indices=ix.tolist(),trace=trace,steps=100,query_audio_forwards=0));scored=infer(net,x,d,shape,u);dump(path,scored)
            rows[u].update({mode+'_'+k:v for k,v in scored.items()})
        print('ROUND2_OWNER',u,flush=True)
        if u==30:
            chosen=choose_cal(rows,out);print('CAL_ROUTE_LOCKED',chosen,flush=True)
    evaluate(rows,out,'Round2 fixed partial-ECAPA candidates; projection60updates vs attentive-pooling80updates,augmentation+margin+anchor; same100update students; candidate selected onlyCal before Test fits; all predeclared candidates reported');print('ROUND2_COMPLETE',flush=True)

if __name__=='__main__':
    torch.set_num_threads(1);p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--out',required=True);p.add_argument('--round',type=int,choices=[1,2],required=True);p.add_argument('--round1');a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    if a.round==1:round1(Path(a.source),out)
    else:round2(Path(a.source),Path(a.round1),out)

