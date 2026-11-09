"""RKD-02: registration Audio adaptation, query vibration only, fixed K=12."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import argparse,copy,hashlib,json,sys,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.preprocessing import StandardScaler
from run_rkd01_vibravox import Student,norm,supcon,rkd,train,encode,thresholds,metrics

def dump(p,x):
    p=Path(p);q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2));q.replace(p)
def pool_ids(d,base):
    return np.array([i for u in base for i in sorted(np.flatnonzero((d['user']==u)&(d['role']=='fit')),key=lambda i:str(d['sample_id'][i]))[:4]])
def registration(d,u):return np.array(sorted(np.flatnonzero((d['user']==u)&(d['role']=='registration')),key=lambda i:str(d['sample_id'][i])))
def teacher_targets(a,d,records,source,root,owner,base,external):
    folder=root/f'owner{owner}';folder.mkdir(exist_ok=True);path=folder/'teacher_targets_private.npz'
    if path.exists():
        q=dict(np.load(path));return q['indices'],q['frozen'],q['adapted']
    torch.manual_seed(4242+owner);rng=np.random.RandomState(4242+owner);pi=pool_ids(d,base);en=registration(d,owner);assert len(en)==12;indices=np.r_[pi,en];assert not np.isin(d['role'][indices],['query','develop']).any();assert all(d['user'][i] in base or (d['user'][i]==owner and d['role'][i]=='registration') for i in indices)
    model=external['model'];emb=model.mods.embedding_model;emb.load_state_dict(external['initial']);emb.eval()
    for p in emb.parameters():p.requires_grad_(True)
    device=next(emb.parameters()).device
    def read(i):
        r=records[int(i)]
        with np.load(source/'paired'/r['file']) as q:w=q['audio'].astype('float32')
        assert hashlib.sha256(w.tobytes()).hexdigest()==r['audio_sha256'];s=(len(w)-32000)//2;w=w[s:s+32000].copy();w-=w.mean();return w
    # Only Base-fit or this owner's registration Audio can enter this cache.
    with ThreadPoolExecutor(max_workers=8) as workers:waves=list(workers.map(read,indices))
    features=[]
    with torch.no_grad():
        for s in range(0,len(waves),16):
            w=torch.tensor(np.stack(waves[s:s+16]),device=device);features.append(model.mods.mean_var_norm(model.mods.compute_features(w),torch.ones(len(w),device=device)).detach())
    feat=torch.cat(features);lens=torch.ones(len(feat),device=device)
    def vectors(ii):return F.normalize(emb(feat[ii],lens[ii]).squeeze(1),dim=1)
    with torch.no_grad():frozen=torch.cat([vectors(np.arange(s,min(s+16,len(feat)))) for s in range(0,len(feat),16)]).detach()
    ids={u:j for j,u in enumerate(base+[owner])};targetlabel=len(base);labels=torch.tensor([ids[int(u)] for u in d['user'][indices]],device=device);protos=F.normalize(torch.stack([frozen[labels==j].mean(0) for j in range(len(ids))]),dim=1);head=nn.Linear(192,len(ids),bias=False).to(device)
    with torch.no_grad():head.weight.copy_(protos)
    opt=torch.optim.AdamW([dict(params=emb.parameters(),lr=1e-6),dict(params=head.parameters(),lr=1e-4)],weight_decay=1e-4);watch=next(p for p in emb.parameters() if p.ndim>1);initial=watch.detach().clone();trace=[]
    for step in range(1,31):
        chosen=rng.choice(base,4,replace=False);ix=np.concatenate([rng.choice(np.flatnonzero(d['user'][indices]==owner),4,replace=False)]+[rng.choice(np.flatnonzero(d['user'][indices]==u),2,replace=False) for u in chosen]);ii=torch.tensor(ix,device=device);z=vectors(ii);y=labels[ii];loss=F.cross_entropy(16*F.linear(z,F.normalize(head.weight,dim=1)),y)+.1*supcon(z.cpu(),y.cpu()).to(device)+.1*(1-F.cosine_similarity(z[4:],frozen[ii[4:]],dim=1)).mean();opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(list(emb.parameters())+list(head.parameters()),1);opt.step()
        if step in [1,10,30]:trace.append(dict(step=step,loss=float(loss.detach())))
    delta=float((watch-initial).abs().max());assert delta>0
    for p in emb.parameters():p.requires_grad_(False)
    with torch.no_grad():adapted=torch.cat([vectors(np.arange(s,min(s+16,len(feat)))) for s in range(0,len(feat),16)]).cpu().numpy()
    np.savez_compressed(path,indices=indices,frozen=frozen.cpu().numpy(),adapted=adapted);torch.save(dict(embedding_model=emb.state_dict(),owner=owner,base_users=base,steps=30),folder/'ecapa_adapted_private.pt');dump(folder/'teacher_audit_private.json',dict(owner=owner,steps=30,backbone_update=delta,query_audio_reads=0,base_users=base,registration_records=12,train_audio_records=len(indices),trace=trace));print('TEACHER_COMPLETE',owner,delta,flush=True);return indices,frozen.cpu().numpy(),adapted
def adapt_student(d,x,initial,owner,base,indices,targets,method,rho):
    torch.manual_seed(42);rng=np.random.RandomState(42000+owner);net=copy.deepcopy(initial);head=nn.Linear(192,len(base)+1,bias=False);labels={u:j for j,u in enumerate(base+[owner])};yall=np.array([labels[int(u)] for u in d['user'][indices]]);lookup={u:np.flatnonzero(d['user'][indices]==u) for u in base+[owner]};xx=x[indices];start=encode(net,xx)
    with torch.no_grad():head.weight.copy_(torch.tensor(norm(np.stack([start[yall==j].mean(0) for j in range(len(labels))]))))
    opt=torch.optim.AdamW(list(net.parameters())+list(head.parameters()),lr=1e-4,weight_decay=1e-4);ema_id=None;ema_kd=None;trace=[];params=list(net.parameters());control_rng=np.random.RandomState(142000+owner);rp=torch.randn(246,192,generator=torch.Generator().manual_seed(20261110))/np.sqrt(246)
    for step in range(1,101):
        net.train();uu=rng.choice(base,6,replace=False);ii=np.concatenate([rng.choice(lookup[owner],8,replace=False)]+[rng.choice(lookup[u],4,replace=False) for u in uu]);y=torch.tensor(yall[ii],dtype=torch.long);z=net(torch.tensor(xx[ii]));identity=F.cross_entropy(16*F.linear(z,F.normalize(head.weight,dim=1)),y)+.1*supcon(z,y)+.1*(1-F.cosine_similarity(z[8:],torch.tensor(start[ii[8:]]),dim=1)).mean();kd=z.new_zeros(());lam=0.;ratio=0.
        if method!='V0':
            target=torch.tensor(targets[ii])
            if method=='V5':target=torch.tensor(targets[[control_rng.choice(np.flatnonzero(yall!=label)) for label in yall[ii]]])
            elif method=='V6':target=F.normalize(torch.tensor(xx[ii])@rp,dim=1)
            kd=rkd(z,target,y);gi=torch.autograd.grad(identity,params,retain_graph=True,allow_unused=True);gk=torch.autograd.grad(kd,params,retain_graph=True,allow_unused=True);ni=torch.sqrt(sum(g.square().sum() for g in gi if g is not None));nk=torch.sqrt(sum(g.square().sum() for g in gk if g is not None));ema_id=float(ni) if ema_id is None else .9*ema_id+.1*float(ni);ema_kd=float(nk) if ema_kd is None else .9*ema_kd+.1*float(nk);lam=.3 if method in ['V1','V3'] else min(100.,rho*ema_id/max(ema_kd,1e-8));ratio=lam*float(nk)/max(float(ni),1e-8)
        loss=identity+lam*kd;opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(list(net.parameters())+list(head.parameters()),1);opt.step()
        if step in [1,25,50,100]:trace.append(dict(step=step,lambda_rkd=lam,gradient_ratio=ratio,identity_loss=float(identity.detach()),rkd_loss=float(kd.detach())))
    return net,trace
def score_owner(net,x,d,owner,attackers):
    en=registration(d,owner);q=np.flatnonzero(np.isin(d['user'],[owner]+attackers)&(d['role']=='query'));emb=encode(net,x[np.r_[en,q]]);s=emb[len(en):]@norm(emb[:len(en)].mean(0));return dict(scores=s.tolist(),labels=(d['user'][q]==owner).tolist())
def assess_rows(cal,test):
    cs=np.concatenate([v['scores'] for v in cal]);cy=np.concatenate([v['labels'] for v in cal]);threshold=thresholds(cs,cy);all_s=np.concatenate([v['scores'] for v in test]);all_y=np.concatenate([v['labels'] for v in test]);out={}
    for point,tau in threshold.items():
        r=metrics(all_s,all_y,tau);each=[metrics(np.array(v['scores']),np.array(v['labels']),tau) for v in test];r.update(macro_auc=float(np.mean([v['auc'] for v in each])),macro_eer=float(np.mean([v['eer'] for v in each])),worst_target_far=max(v['far'] for v in each));out[point]=r
    return out
def evaluate_owners(a,d,x,initial,base,owners,root,external,rho,methods):
    root.mkdir(parents=True,exist_ok=True);records=json.loads((Path(a.source)/'manifest_private.json').read_text())['records'];src=Path(a.paired);allrows={m:{} for m in methods}
    for u in owners:
        folder=root/f'owner{u}';folder.mkdir(exist_ok=True);indices,frozen,adapted=teacher_targets(a,d,records,src,root,u,base,external)
        for method in methods:
            dest=folder/f'{method}_rho{rho}_private.json'
            if dest.exists():result=json.loads(dest.read_text())
            else:
                net,trace=adapt_student(d,x,initial,u,base,indices,adapted if method in ['V3','V4','V5'] else frozen,method,rho);group=list(range(29,32)) if u in range(29,32) else list(range(32,35)) if u in range(32,35) else list(range(35,47)) if u<47 else list(range(47,59));result=score_owner(net,x,d,u,[v for v in group if v!=u]);result['trace']=trace;dump(dest,result);torch.save(net.state_dict(),folder/f'{method}_rho{rho}_student_private.pt')
            allrows[method][u]=result
        print('OWNER_COMPLETE',u,rho,methods,flush=True)
    return allrows
def main(a):
    torch.set_num_threads(1);source=Path(a.source);root=Path(a.out);root.mkdir(parents=True,exist_ok=True);d=dict(np.load(source/'inputs_private.npz'));teacher=np.load(source/'audio_targets_private.npy');sys.path.insert(0,'/root');from run_ecapa_pilot import teacher_load
    for n in ['model','deps']:
        p=root/n
        if not p.exists():p.symlink_to(source/n,target_is_directory=True)
    model=teacher_load(root);external=dict(model=model,initial={k:v.detach().cpu().clone() for k,v in model.mods.embedding_model.state_dict().items()});assert torch.cuda.is_available()
    lock=root/'selection_public.json'
    if not lock.exists():
        base=list(range(29));fit=np.isin(d['user'],base)&(d['role']=='fit');x=StandardScaler().fit(d['x'][fit]).transform(d['x']).astype('float32');states,_=train(d,teacher,base,'S1',0,42,700,x);initial=Student();initial.load_state_dict(states[700]);choices=[]
        for rho in [.03,.1,.3]:
            rows=evaluate_owners(a,d,x,initial,base,list(range(29,35)),root/'inner',external,rho,['V2']);r=assess_rows([rows['V2'][u] for u in range(29,32)],[rows['V2'][u] for u in range(32,35)]);choices.append((r['Cal_EER']['hter'],-r['Cal_EER']['macro_auc'],rho))
        best=min(choices);dump(lock,dict(rho=best[2],selection='Base29train,3innerCal,3innerDev;seed42,K12;V2 only',teacher_steps=30,student_steps=100,choices=[dict(hter=v[0],rho=v[2]) for v in choices]));print('RHO_LOCKED',best[2],flush=True)
    rho=json.loads(lock.read_text())['rho'];base=list(range(35));fit=np.isin(d['user'],base)&(d['role']=='fit');x=StandardScaler().fit(d['x'][fit]).transform(d['x']).astype('float32');initial=Student();initial.load_state_dict(torch.load(source/'training/S1_seed42_private.pt',map_location='cpu',weights_only=False)['state']);methods=['V0','V1','V2','V3','V4']+(['V5','V6'] if a.controls else []);rows=evaluate_owners(a,d,x,initial,base,list(range(35,59)),root/'outer',external,rho,methods);public={}
    for method in methods:public[method]=assess_rows([rows[method][u] for u in range(35,47)],[rows[method][u] for u in range(47,59)])
    for name in ['R_frozen','S0_shape']:
        ref={};shape=norm(x[:,:40])
        for u in range(35,59):
            group=list(range(35,47)) if u<47 else list(range(47,59))
            if name=='R_frozen':ref[u]=score_owner(initial,x,d,u,[v for v in group if v!=u])
            else:
                en=registration(d,u);q=np.flatnonzero(np.isin(d['user'],group)&(d['role']=='query'));ref[u]=dict(scores=(shape[q]@norm(shape[en].mean(0))).tolist(),labels=(d['user'][q]==u).tolist())
        public[name]=assess_rows([ref[u] for u in range(35,47)],[ref[u] for u in range(47,59)])
    dump(root/'results_public.json',dict(state='complete',protocol='registration Audio+vibration allowed;query vibration only;K12;35Base12Cal12Test;seed42;one fixed identity split',rho=rho,methods=dict(V0='vibration-only personal adaptation',V1='frozen ECAPA,lambda0.3',V2='frozen ECAPA,gradient balanced',V3='registrant Audio-finetuned ECAPA,lambda0.3',V4='registrant Audio-finetuned ECAPA,gradient balanced',V5='wrong identity correspondence,adapted teacher,gradient balanced',V6='vibration random-projection geometry,no Audio,gradient balanced'),results=public));print('COMPLETE',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--paired',required=True);p.add_argument('--out',required=True);p.add_argument('--controls',action='store_true');main(p.parse_args())
