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
    for name in ts:r[name]['worst_target_far']=max(v[name]['far'] for v in per.values())
    return dict(metrics=r,targets=per,thresholds=ts)
def encode(net,x):
    net.eval()
    with torch.no_grad():return np.concatenate([net(torch.tensor(x[i:i+256])).numpy() for i in range(0,len(x),256)])
def prepared(d,users):
    fit=np.isin(d['user'],users)&(d['role']=='fit');sc=StandardScaler().fit(d['x'][fit]);x=sc.transform(d['x']).astype('float32');return x,norm(x),norm(x[:,:40]),sc
def train(d,t,users,method,lam,seed,steps,x,callback=None):
    torch.manual_seed(seed);rng=np.random.RandomState(seed);net=Student();head=nn.Linear(192,len(users));opt=torch.optim.AdamW(list(net.parameters())+list(head.parameters()),lr=1e-4,weight_decay=1e-4);lookup={u:np.flatnonzero((d['user']==u)&(d['role']=='fit')) for u in users};labels={u:i for i,u in enumerate(users)};history=[];states={}
    assert all(len(v)>=4 for v in lookup.values())
    if method!='S1':assert all(np.linalg.norm(t[ix],axis=1).min()>.99 for ix in lookup.values())
    for step in range(1,steps+1):
        net.train();uu=rng.choice(users,8,replace=False);ix=np.concatenate([rng.choice(lookup[u],4,replace=False) for u in uu]);y=torch.tensor([labels[u] for u in d['user'][ix]],dtype=torch.long);z=net(torch.tensor(x[ix]));loss=F.cross_entropy(16*F.linear(z,F.normalize(head.weight,dim=1)),y)+.1*supcon(z,y);kd=z.new_zeros(())
        if method=='S2':kd=(1-F.cosine_similarity(net.kdproj(z),torch.tensor(t[ix]),dim=1)).mean()
        elif method=='S3':kd=rkd(z,torch.tensor(t[ix]),y)
        loss=loss+lam*kd;opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(list(net.parameters())+list(head.parameters()),1);opt.step()
        if step%100==0 or step==steps:
            if callback:history.append(dict(step=step,**callback(encode(net,x))))
            states[step]=copy.deepcopy(net.state_dict());print('TRAIN',method,seed,lam,step,float(loss.detach()),flush=True)
    return states,history
def fit_development(d,t,out,steps):
    path=out/'selection_private.json'
    if path.exists():return json.loads(path.read_text())
    users=list(range(29));cal=[29,30,31];dev=[32,33,34];x,stat,shape,sc=prepared(d,users)
    # Strong frozen shape baseline and full-stat reference are both reported.
    stat=shape;cs=comparisons(stat,d,cal,12);qs=comparisons(stat,d,dev,12);fixed=assess(cs,qs);selected={};allhist={}
    for method in ['S1','S2','S3']:
        best=None
        for lam in ([0] if method=='S1' else [0,.03,.1,.3]):
            def callback(z):
                result=assess(comparisons(z,d,cal,12),comparisons(z,d,dev,12));r=result['metrics']['Cal_EER'];return dict(hter=r['hter'],auc=r['auc'])
            states,hist=train(d,t,users,method,lam,42,steps,x,callback);allhist[f'{method}_{lam}']=hist
            for row in hist:
                key=(row['hter'],-row['auc'],lam,row['step'])
                if best is None or key<best[0]:best=(key,lam,row['step'],states[row['step']])
        _,lam,step,state=best;net=Student();net.load_state_dict(state);z=encode(net,x);zc=comparisons(z,d,cal,12);zq=comparisons(z,d,dev,12);mu=[float(cs[0].mean()),float(zc[0].mean())];sd=[float(cs[0].std()),float(zc[0].std())];choice=None
        for alpha in [0,.25,.5,.75,1]:
            fc=((1-alpha)*(cs[0]-mu[0])/max(sd[0],1e-8)+alpha*(zc[0]-mu[1])/max(sd[1],1e-8),cs[1],cs[2]);fq=((1-alpha)*(qs[0]-mu[0])/max(sd[0],1e-8)+alpha*(zq[0]-mu[1])/max(sd[1],1e-8),qs[1],qs[2]);r=assess(fc,fq)['metrics']['Cal_EER'];key=(r['hter'],-r['auc'],alpha)
            if choice is None or key<choice[0]:choice=(key,alpha)
        selected[method]=dict(lam=lam,step=step,alpha=choice[1],score_means=mu,score_stds=sd,inner_hter=best[0][0]);print('LOCKED',method,json.dumps(selected[method]),flush=True)
    result=dict(controls=dict(inner_train=users,inner_cal=cal,inner_dev=dev,selection_seed=42,max_steps=steps,stat_branch='mean-spectrum-shape40,all-stat246 also reported',student_input=246),selected=selected,development=allhist);dump(path,result);return result
def evaluate_final(d,t,out,selection,seeds,steps):
    users=list(range(35));cal=list(range(35,47));test=list(range(47,59));x,full,shape,sc=prepared(d,users);rows={}
    # No teacher is consulted below: t is passed only to training users.
    for seed in seeds:
        dest=out/f'seed{seed}_results_private.json'
        if dest.exists():rows[str(seed)]=json.loads(dest.read_text());continue
        result={}
        for name,z in [('S0_shape',shape),('S0_allstats',full)]:result[name]={str(k):assess(comparisons(z,d,cal,k),comparisons(z,d,test,k)) for k in [3,5,12]}
        for method in ['S1','S2','S3']:
            p=selection['selected'][method];states,_=train(d,t,users,method,p['lam'],seed,steps,x);net=Student();net.load_state_dict(states[p['step']]);torch.save(dict(state=states[p['step']],selection=p,base_users=users),out/f'{method}_seed{seed}_private.pt');z=encode(net,x);result[method+'_student']={};result[method+'_fusion']={}
            for k in [3,5,12]:
                cc=comparisons(z,d,cal,k);qq=comparisons(z,d,test,k);result[method+'_student'][str(k)]=assess(cc,qq);bc=comparisons(shape,d,cal,k);bq=comparisons(shape,d,test,k);mu=p['score_means'];sd=p['score_stds'];alpha=p['alpha']
                def mix(s,b):return ((1-alpha)*(b[0]-mu[0])/max(sd[0],1e-8)+alpha*(s[0]-mu[1])/max(sd[1],1e-8),s[1],s[2])
                result[method+'_fusion'][str(k)]=assess(mix(cc,bc),mix(qq,bq))
        dump(dest,result);rows[str(seed)]=result;print('FINAL_SEED_COMPLETE',seed,flush=True)
    public={}
    for method in next(iter(rows.values())):
        public[method]={str(k):{point:{metric:float(np.mean([r[method][str(k)]['metrics'][point][metric] for r in rows.values()])) for metric in ['tar','far','frr','hter','auc','eer','worst_target_far']} for point in ['Cal_EER','Cal_FAR5']} for k in [3,5,12]}
    rng=np.random.RandomState(424344);paired={}
    for k in [3,5,12]:
        for control in ['S1_student','S2_student']:
            a=np.array([[r['S3_student'][str(k)]['targets'][str(u)]['Cal_EER']['hter']-r[control][str(k)]['targets'][str(u)]['Cal_EER']['hter'] for u in test] for r in rows.values()]);delta=a.mean(0);boots=np.array([delta[rng.choice(12,12,replace=True)].mean() for _ in range(5000)]);paired[f'K{k}_S3_minus_{control}']=dict(hter_delta=float(delta.mean()),identity_bootstrap95=np.quantile(boots,[.025,.975]).tolist(),improved_identities=int((delta<0).sum()),seed_mean_deltas=a.mean(1).tolist(),bootstrap_unit='target identity;average seeds before resampling;not per recording')
    dump(out/'results_public.json',dict(state='complete',seeds=list(map(int,rows)),test_identities=12,results=public,paired=paired,scope='P2 only,S0/S1/S2/S3;P3 wrong-identity and other controls not yet completed;do not claim Audio causal benefit'))
def main(a):
    torch.set_num_threads(1);out=Path(a.out);out.mkdir(parents=True,exist_ok=True);d=dict(np.load(Path(a.data)/'inputs_private.npz'));diag=json.loads((Path(a.data)/'teacher_diagnostic_public.json').read_text());assert diag['teacher_gate']=='pass';t=np.load(Path(a.data)/'audio_targets_private.npy');assert np.all(t[d['group']!='base']==0);assert set(np.unique(d['user'][d['group']=='base']))==set(range(35));assert not set(d['text'][d['role']=='registration'])&set(d['text'][d['role']=='query']);sel=fit_development(d,t,out,a.steps);evaluate_final(d,t,out,sel,a.seeds,a.steps)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--out',required=True);p.add_argument('--steps',type=int,default=1000);p.add_argument('--seeds',nargs='+',type=int,default=[42,43,44]);main(p.parse_args())
