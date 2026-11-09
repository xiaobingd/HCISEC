"""Existing ResNet18 T3, closed-set ABCS cross-sentence, BC-only RBF authentication."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import argparse,json,time,hashlib,gc
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
import run_esmb_t3_joint as r

ROOT=Path('/autodl-fs/data/abcs_t3_closed_20261109');METHODS=['T0_matched','T3_paired','T3_wrong_time','T3_wrong_user']
def dev_auc(z,u,j):
    sim=z@z.T;upper=np.triu(np.ones(sim.shape,bool),1);pos=upper&(u[:,None]==u[None,:])&(j[:,None]!=j[None,:]);neg=upper&(u[:,None]!=u[None,:])&(j[:,None]==j[None,:])
    return float(roc_auc_score(np.r_[np.ones(pos.sum()),np.zeros(neg.sum())],np.r_[sim[pos],sim[neg]]))
def evaluate(d):
    result={};private={};auth=d['index']<75;cal=d['role']=='cal';query=d['role']=='query';y=d['user']
    for method in METHODS:
        folder=ROOT/method
        if not (folder/'complete.json').exists():continue
        z=np.load(folder/'embeddings_private.npy');sc=StandardScaler().fit(z[auth]);x=sc.transform(z);rows=[]
        for u in range(8):
            model=SVC(C=1.,gamma='scale',class_weight='balanced').fit(x[auth],y[auth]==u);cs=model.decision_function(x[cal]);cy=y[cal]==u
            ts=np.r_[np.nextafter(cs.min(),-np.inf),np.unique(cs),np.nextafter(cs.max(),np.inf)];tau=min(ts,key=lambda t:(abs(float((cs[~cy]>=t).mean())-float((cs[cy]<t).mean())),float((cs[~cy]>=t).mean())))
            s=model.decision_function(x[query]);truth=y[query]==u;tar=float((s[truth]>=tau).mean());far=float((s[~truth]>=tau).mean());rows.append(dict(user=u,threshold=float(tau),tar=tar,far=far,hter=(1-tar+far)/2,auc=float(roc_auc_score(truth,s))))
        private[method]=rows;result[method]={k:float(np.mean([v[k] for v in rows])) for k in ['tar','far','hter','auc']}
    r.write(ROOT/'results_private.json',private);r.write(ROOT/'summary_public.json',dict(state='complete' if len(result)==4 else 'partial',methods=result,completed_methods=len(result),protocol='8seenidentities;cross-sentence globally locked;25query genuine175attacks pertarget;75authfit/25Cal;BCchannel0 provisional;noqueryAudio;not open-set'))
def train(d,method,steps):
    folder=ROOT/method;folder.mkdir(exist_ok=True)
    if (folder/'complete.json').exists():return
    torch.manual_seed(42);rng=np.random.RandomState(42);net=r.Model(8);opt=torch.optim.AdamW(net.parameters(),lr=1e-4,weight_decay=1e-4);initial=hashlib.sha256(b''.join(t.numpy().tobytes() for t in net.state_dict().values())).hexdigest();fit=np.flatnonzero(d['role']=='fit');dev=np.flatnonzero(d['role']=='dev');lookup={(int(u),int(j)):i for i,(u,j) in enumerate(zip(d['user'],d['index']))};best=-1;history=[];gradient=None;start=time.time()
    for step in range(1,steps+1):
        net.train();ix=np.concatenate([rng.choice(fit[d['user'][fit]==u],2,replace=False) for u in range(8)]);yy=torch.tensor(d['user'][ix],dtype=torch.long);target=torch.tensor(d['audio'][ix].copy())
        if method=='T3_wrong_time':target=torch.roll(target,14,-1)
        if method=='T3_wrong_user':target=torch.tensor(d['audio'][[lookup[(int(d['user'][i])+1)%8,int(d['index'][i])] for i in ix]])
        z,logits,pred=net(torch.tensor(d['x'][ix]));identity=F.cross_entropy(logits,yy)+.1*r.supcon(z,yy);spec=F.mse_loss(pred[:,:17],target[:,:17]);env=F.mse_loss(pred[:,17:],target[:,17:]);aux=.1*(spec+env);loss=identity if method=='T0_matched' else identity+aux
        if step==1 and method!='T0_matched':gradient=float(torch.autograd.grad(aux,net.body.conv1.weight,retain_graph=True)[0].norm());assert gradient>0
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
        if step==1 or step%5==0:
            r.write(ROOT/'status.json',dict(state='training',method=method,step=step,steps=steps,loss=float(loss.detach()),aux_to_encoder_gradient=gradient,elapsed_seconds=time.time()-start));print('TRAIN',method,step,float(loss.detach()),flush=True)
        if step%50==0 or step==steps:
            auc=dev_auc(r.encode(net,d['x'][dev]),d['user'][dev],d['index'][dev]);history.append(dict(step=step,dev_auc=auc))
            if auc>best:best=auc;torch.save(dict(state=net.state_dict(),step=step,dev_auc=auc),folder/'best_private.pt')
            r.write(folder/'history.json',dict(init_sha256=initial,history=history,aux_to_encoder_gradient=gradient));torch.save(dict(state=net.state_dict(),step=step),folder/'last_private.pt')
    del opt,net;gc.collect();net=r.Model(8);ck=torch.load(folder/'best_private.pt',map_location='cpu',weights_only=True);net.load_state_dict(ck['state']);np.save(folder/'embeddings_private.npy',r.encode(net,d['x']));r.write(folder/'complete.json',dict(steps=steps,selected_step=ck['step'],dev_auc=ck['dev_auc'],aux_to_encoder_gradient=gradient,elapsed_seconds=time.time()-start));del net;gc.collect()
def main(a):
    ROOT.mkdir(exist_ok=True);torch.set_num_threads(1);d={k:v for k,v in np.load(ROOT/'inputs_private.npz').items()};assert d['x'].shape==(1000,80,80) and d['audio'].shape==(1000,18,80);assert np.all(d['audio'][np.isin(d['role'],['cal','query'])]==0)
    r.write(ROOT/'protocol_public.json',dict(model='reuse existing run_esmb_t3_joint.Model(8),ResNet18fromscratch192dim/18channelaux',loss='CE+.1SupCon+.1spectrumMSE+.1envelopeMSE;T0aux0',steps=a.steps,seed=42,batch=16,methods=METHODS,selection='15devtexts/user identityAUC every50steps;noCal/query checkpoint selection',backend='same balanced RBF C1 gamma scale;75text/user authfit;25Calthreshold;25query;all7attackers',channel='0BC/1Audio provisional peruser instruction;not independently confirmed',no_generator=True))
    for method in METHODS:train(d,method,a.steps);evaluate(d)
    r.write(ROOT/'status.json',dict(state='complete',methods=METHODS));print('ALL_COMPLETE',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=200);main(p.parse_args())
