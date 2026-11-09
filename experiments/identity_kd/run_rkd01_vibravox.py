"""KD-RKD-01 paired S0/S1/S2/S3, isolated-identity hyperparameter selection."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import argparse,copy,json,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score,roc_curve

def dump(p,x):Path(p).write_text(json.dumps(x,indent=2))
def norm(x):return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)
def supcon(z,y):
    sim=z@z.T/.1;eye=torch.eye(len(z),dtype=torch.bool);pos=(y[:,None]==y[None,:])&~eye;logp=sim-torch.logsumexp(sim.masked_fill(eye,-1e9),1,keepdim=True);return -((logp*pos).sum(1)/pos.sum(1).clamp_min(1)).mean()
def rkd(z,t,y):
    mask=~torch.eye(len(z),dtype=torch.bool);ds=torch.cdist(z,z);dt=torch.cdist(t,t);ds=ds/ds[mask].mean().clamp_min(1e-8);dt=dt/dt[mask].mean().clamp_min(1e-8);err=F.smooth_l1_loss(ds,dt,reduction='none');same=(y[:,None]==y[None,:])&mask;different=(y[:,None]!=y[None,:])&mask;return .5*(err[same].mean()+err[different].mean())
def angles(z,t):
    def calc(x):
        v=F.normalize(x[:,None]-x[None,:],dim=-1);return torch.bmm(v,v.transpose(1,2))
    n=len(z);i=torch.arange(n);valid=(i[:,None,None]!=i[None,:,None])&(i[:,None,None]!=i[None,None,:])&(i[None,:,None]!=i[None,None,:]);return F.smooth_l1_loss(calc(z)[valid],calc(t)[valid])
def mmd(z,t):
    def kernel(a,b):
        dd=torch.cdist(a,b).square();return sum(torch.exp(-dd/(2*s*s)) for s in [.25,.5,1.,2.])/4
    return kernel(z,z).mean()+kernel(t,t).mean()-2*kernel(z,t).mean()
class Student(nn.Module):
    def __init__(self):
        super().__init__();self.proj=nn.Linear(246,192);self.res=nn.Sequential(nn.Linear(192,64),nn.GELU(),nn.Linear(64,192));self.kdproj=nn.Linear(192,192)
        nn.init.zeros_(self.res[-1].weight);nn.init.zeros_(self.res[-1].bias)
    def forward(self,x):
        z=self.proj(x);return F.normalize(z+self.res(z),dim=1)
def enroll_order(d,u):
    ix=np.flatnonzero((d['user']==u)&(d['role']=='registration'));return sorted(ix,key=lambda i:str(d['sample_id'][i]))
def comparisons(z,d,users,k):
    scores=[];ys=[];targets=[]
    for u in users:
        en=enroll_order(d,u)[:k];assert len(en)==k
        q=np.flatnonzero(np.isin(d['user'],users)&(d['role']=='query'));s=z[q]@norm(z[en].mean(0));scores.extend(s);ys.extend(d['user'][q]==u);targets.extend([u]*len(q))
    return np.array(scores),np.array(ys),np.array(targets)
def thresholds(s,y):
    far,tar,t=roc_curve(y,s,drop_intermediate=False);err=1-tar;i=min(range(len(t)),key=lambda j:(abs(far[j]-err[j]),far[j]));valid=np.flatnonzero(far<=.05);j=max(valid,key=lambda j:(tar[j],-far[j],t[j]));return dict(Cal_EER=float(t[i]),Cal_FAR5=float(t[j]))
def metrics(s,y,t):
    tar=float((s[y]>=t).mean());far=float((s[~y]>=t).mean());fp,tp,_=roc_curve(y,s);i=np.argmin(abs(fp-(1-tp)));return dict(tar=tar,far=far,frr=1-tar,hter=(1-tar+far)/2,auc=float(roc_auc_score(y,s)),eer=float((fp[i]+1-tp[i])/2),genuine_n=int(y.sum()),impostor_n=int((~y).sum()))
def assess(cal,query):
    cs,cy,_=cal;s,y,users=query;ts=thresholds(cs,cy);r={name:metrics(s,y,t) for name,t in ts.items()};per={str(u):{name:metrics(s[users==u],y[users==u],t) for name,t in ts.items()} for u in np.unique(users)}
    for name in ts:
        r[name]['worst_target_far']=max(v[name]['far'] for v in per.values())
        r[name]['macro_target_auc']=float(np.mean([v[name]['auc'] for v in per.values()]));r[name]['macro_target_eer']=float(np.mean([v[name]['eer'] for v in per.values()]))
    return dict(metrics=r,targets=per,thresholds=ts)
def encode(net,x):
    net.eval()
    with torch.no_grad():return np.concatenate([net(torch.tensor(x[i:i+256])).numpy() for i in range(0,len(x),256)])
def encode_subset(net,x,indices):
    z=np.zeros((len(x),192),dtype='float32');z[indices]=encode(net,x[indices]);return z
def prepared(d,users):
    fit=np.isin(d['user'],users)&(d['role']=='fit');sc=StandardScaler().fit(d['x'][fit]);x=sc.transform(d['x']).astype('float32');return x,norm(x),norm(x[:,:40]),sc
def train(d,t,users,method,lam,seed,steps,x,callback=None,callback_indices=None):
    torch.manual_seed(seed);rng=np.random.RandomState(seed);net=Student();head=nn.Linear(192,len(users));opt=torch.optim.AdamW(list(net.parameters())+list(head.parameters()),lr=1e-4,weight_decay=1e-4);lookup={u:np.flatnonzero((d['user']==u)&(d['role']=='fit')) for u in users};labels={u:i for i,u in enumerate(users)};history=[];states={}
    assert all(len(v)>=4 for v in lookup.values());control_rng=np.random.RandomState(seed+100000);rp=torch.randn(246,192,generator=torch.Generator().manual_seed(20261109))/np.sqrt(246)
    if method!='S1':assert all(np.linalg.norm(t[ix],axis=1).min()>.99 for ix in lookup.values())
    for step in range(1,steps+1):
        net.train();uu=rng.choice(users,8,replace=False);ix=np.concatenate([rng.choice(lookup[u],4,replace=False) for u in uu]);y=torch.tensor([labels[u] for u in d['user'][ix]],dtype=torch.long);z=net(torch.tensor(x[ix]));loss=F.cross_entropy(16*F.linear(z,F.normalize(head.weight,dim=1)),y)+.1*supcon(z,y);kd=z.new_zeros(())
        if method=='S2':kd=(1-F.cosine_similarity(net.kdproj(z),torch.tensor(t[ix]),dim=1)).mean()
        elif method=='S3':kd=rkd(z,torch.tensor(t[ix]),y)
        elif method=='S4':kd=rkd(z,torch.tensor(t[ix]),y)+.5*angles(z,torch.tensor(t[ix]))
        elif method=='S5':kd=mmd(F.normalize(net.kdproj(z),dim=1),torch.tensor(t[ix]))
        elif method=='S6':
            wrong=np.array([control_rng.choice(lookup[control_rng.choice([v for v in users if v!=u])]) for u in d['user'][ix]]);kd=rkd(z,torch.tensor(t[wrong]),y)
        elif method=='S7':kd=rkd(z,F.normalize(torch.tensor(x[ix])@rp,dim=1),y)
        loss=loss+lam*kd;opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(list(net.parameters())+list(head.parameters()),1);opt.step()
        if step%100==0 or step==steps:
            if callback:history.append(dict(step=step,**callback(encode_subset(net,x,callback_indices))))
            states[step]=copy.deepcopy(net.state_dict());print('TRAIN',method,seed,lam,step,float(loss.detach()),flush=True)
    return states,history
def fit_development(d,t,out,steps,methods):
    path=out/'selection_private.json'
    existing=json.loads(path.read_text()) if path.exists() else None
    if existing and all(m in existing['selected'] for m in methods):return existing
    users=list(range(29));cal=[29,30,31];dev=[32,33,34];x,stat,shape,sc=prepared(d,users)
    # Strong frozen shape baseline and full-stat reference are both reported.
    stat=shape;cs=comparisons(stat,d,cal,12);qs=comparisons(stat,d,dev,12);fixed=assess(cs,qs);selected=existing['selected'] if existing else {};allhist=existing['development'] if existing else {};selection_ix=np.flatnonzero(np.isin(d['user'],cal+dev)&np.isin(d['role'],['registration','query']))
    for method in methods:
        if method in selected:continue
        if method=='S6':
            p=selected['S3'];states,_=train(d,t,users,method,p['lam'],42,steps,x);net=Student();net.load_state_dict(states[p['step']]);zz=encode_subset(net,x,selection_ix);zc=comparisons(zz,d,cal,12);selected[method]=dict(p,score_means=[float(cs[0].mean()),float(zc[0].mean())],score_stds=[float(cs[0].std()),float(zc[0].std())],control='independent wrong other-identity record per sample;S3 lambda/step/alpha reused;own Base-inner-Cal score scale');continue
        best=None
        for lam in ([0] if method=='S1' else [0,.03,.1,.3]):
            def callback(z):
                result=assess(comparisons(z,d,cal,12),comparisons(z,d,dev,12));r=result['metrics']['Cal_EER'];return dict(hter=r['hter'],auc=r['auc'])
            states,hist=train(d,t,users,method,lam,42,steps,x,callback,selection_ix);allhist[f'{method}_{lam}']=hist
            for row in hist:
                key=(row['hter'],-row['auc'],lam,row['step'])
                if best is None or key<best[0]:best=(key,lam,row['step'],states[row['step']])
        _,lam,step,state=best;net=Student();net.load_state_dict(state);z=encode_subset(net,x,selection_ix);zc=comparisons(z,d,cal,12);zq=comparisons(z,d,dev,12);mu=[float(cs[0].mean()),float(zc[0].mean())];sd=[float(cs[0].std()),float(zc[0].std())];choice=None
        for alpha in [0,.25,.5,.75,1]:
            fc=((1-alpha)*(cs[0]-mu[0])/max(sd[0],1e-8)+alpha*(zc[0]-mu[1])/max(sd[1],1e-8),cs[1],cs[2]);fq=((1-alpha)*(qs[0]-mu[0])/max(sd[0],1e-8)+alpha*(zq[0]-mu[1])/max(sd[1],1e-8),qs[1],qs[2]);r=assess(fc,fq)['metrics']['Cal_EER'];key=(r['hter'],-r['auc'],alpha)
            if choice is None or key<choice[0]:choice=(key,alpha)
        selected[method]=dict(lam=lam,step=step,alpha=choice[1],score_means=mu,score_stds=sd,inner_hter=best[0][0]);print('LOCKED',method,json.dumps(selected[method]),flush=True)
    result=dict(controls=dict(inner_train=users,inner_cal=cal,inner_dev=dev,selection_seed=42,max_steps=steps,stat_branch='mean-spectrum-shape40,all-stat246 also reported',student_input=246),selected=selected,development=allhist);dump(path,result);return result
def evaluate_final(d,t,out,selection,seeds,steps,methods):
    users=list(range(35));cal=list(range(35,47));test=list(range(47,59));x,full,shape,sc=prepared(d,users);rows={}
    # No teacher is consulted below: t is passed only to training users.
    for seed in seeds:
        dest=out/f'seed{seed}_results_private.json'
        result=json.loads(dest.read_text()) if dest.exists() else {}
        for name,z in [('S0_shape',shape),('S0_allstats',full)]:result[name]={str(k):assess(comparisons(z,d,cal,k),comparisons(z,d,test,k)) for k in [3,5,12]}
        for method in methods:
            if method+'_student' in result:continue
            p=selection['selected'][method];states,_=train(d,t,users,method,p['lam'],seed,steps,x);net=Student();net.load_state_dict(states[p['step']]);torch.save(dict(state=states[p['step']],selection=p,base_users=users),out/f'{method}_seed{seed}_private.pt');final_ix=np.flatnonzero(np.isin(d['user'],cal+test)&np.isin(d['role'],['registration','query']));z=encode_subset(net,x,final_ix);result[method+'_student']={};result[method+'_fusion']={}
            for k in [3,5,12]:
                cc=comparisons(z,d,cal,k);qq=comparisons(z,d,test,k);result[method+'_student'][str(k)]=assess(cc,qq);bc=comparisons(shape,d,cal,k);bq=comparisons(shape,d,test,k);mu=p['score_means'];sd=p['score_stds'];alpha=p['alpha']
                def mix(s,b):return ((1-alpha)*(b[0]-mu[0])/max(sd[0],1e-8)+alpha*(s[0]-mu[1])/max(sd[1],1e-8),s[1],s[2])
                result[method+'_fusion'][str(k)]=assess(mix(cc,bc),mix(qq,bq))
        dump(dest,result);rows[str(seed)]=result;print('FINAL_SEED_COMPLETE',seed,flush=True)
    public={}
    for method in next(iter(rows.values())):
        public[method]={str(k):{point:{metric:float(np.mean([r[method][str(k)]['metrics'][point][metric] for r in rows.values()])) for metric in ['tar','far','frr','hter','auc','eer','macro_target_auc','macro_target_eer']} for point in ['Cal_EER','Cal_FAR5']} for k in [3,5,12]}
        for k in [3,5,12]:
            for point in ['Cal_EER','Cal_FAR5']:public[method][str(k)][point]['worst_target_far']=max(float(np.mean([r[method][str(k)]['targets'][str(u)][point]['far'] for r in rows.values()])) for u in test)
    rng=np.random.RandomState(424344);paired={}
    for k in [3,5,12]:
        for control in ['S1_student','S2_student']:
            a=np.array([[r['S3_student'][str(k)]['targets'][str(u)]['Cal_EER']['hter']-r[control][str(k)]['targets'][str(u)]['Cal_EER']['hter'] for u in test] for r in rows.values()]);delta=a.mean(0);boots=np.array([delta[rng.choice(12,12,replace=True)].mean() for _ in range(5000)]);paired[f'K{k}_S3_minus_{control}']=dict(hter_delta=float(delta.mean()),identity_bootstrap95=np.quantile(boots,[.025,.975]).tolist(),improved_identities=int((delta<0).sum()),seed_mean_deltas=a.mean(1).tolist(),bootstrap_unit='target identity;average seeds before resampling;not per recording')
    dump(out/'results_public.json',dict(state='complete',seeds=list(map(int,rows)),test_identities=12,results=public,paired=paired,scope='P2/P3 completed methods identified by result keys;ABCS/HMD replication remains separate;no Audio at enrollment/query'))
def main(a):
    torch.set_num_threads(1);out=Path(a.out);out.mkdir(parents=True,exist_ok=True);d=dict(np.load(Path(a.data)/'inputs_private.npz'));diag=json.loads((Path(a.data)/'teacher_diagnostic_public.json').read_text());assert diag['teacher_gate']=='pass';t=np.load(Path(a.data)/'audio_targets_private.npy');assert np.all(t[d['group']!='base']==0);assert set(np.unique(d['user'][d['group']=='base']))==set(range(35));assert not set(d['text'][d['role']=='registration'])&set(d['text'][d['role']=='query']);methods=['S1','S2','S3'] if a.phase=='p2' else ['S1','S2','S3','S4','S5','S6','S7'];sel=fit_development(d,t,out,a.steps,methods);evaluate_final(d,t,out,sel,a.seeds,a.steps,methods)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--out',required=True);p.add_argument('--phase',choices=['p2','p3'],default='p2');p.add_argument('--steps',type=int,default=1000);p.add_argument('--seeds',nargs='+',type=int,default=[42,43,44]);main(p.parse_args())
