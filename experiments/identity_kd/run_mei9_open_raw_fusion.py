"""Identity-disjoint fusion training with raw HMD statistics, no final identity tuning."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import argparse,json,time,copy
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

def norm(x):return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)
def fold(target):
    owner=target if target<=8 else 1;order=[1+(owner-1+j)%8 for j in range(8)]
    return dict(target=target,base=order[4:],calibration=order[2:4],unseen_attackers=[order[1],9] if target<=8 else [1,2],owner=owner)
def auc(z,y,w):
    sim=z@z.T;upper=np.triu(np.ones(sim.shape,bool),1);pos=upper&(y[:,None]==y[None,:])&(w[:,None]!=w[None,:]);neg=upper&(y[:,None]!=y[None,:])&(w[:,None]==w[None,:]);return float(roc_auc_score(np.r_[np.ones(pos.sum()),np.zeros(neg.sum())],np.r_[sim[pos],sim[neg]]))
def supcon(z,y):
    sim=z@z.T/.1;eye=torch.eye(len(y),dtype=torch.bool);pos=(y[:,None]==y[None,:])&~eye;lp=sim-torch.logsumexp(sim.masked_fill(eye,-1e9),1,keepdim=True);return -(lp*pos).sum(1).div(pos.sum(1).clamp_min(1)).mean()
class Fusion(nn.Module):
    def __init__(self):
        super().__init__();self.a=nn.Linear(256,64);self.b=nn.Linear(64,192);nn.init.zeros_(self.b.weight);nn.init.zeros_(self.b.bias)
    def forward(self,z,x):
        delta=self.b(F.gelu(self.a(torch.cat([z,x],1))));delta=.1*delta/torch.linalg.vector_norm(delta,dim=1,keepdim=True).clamp_min(1);return F.normalize(z+delta,dim=1)
def train(z,x,t,d,f,mode,steps):
    torch.manual_seed(42);rng=np.random.RandomState(42);net=Fusion();head=nn.Linear(192,4,bias=False);fit=np.flatnonzero(np.isin(d['user'],f['base'])&(d['role']=='fit'));dev=np.flatnonzero(np.isin(d['user'],f['base'])&(d['role']=='develop'));words=np.unique(d['word'][fit]);lookup={(u,w):fit[(d['user'][fit]==u)&(d['word'][fit]==w)] for u in f['base'] for w in words};labels={u:j for j,u in enumerate(f['base'])}
    with torch.no_grad():head.weight.copy_(torch.tensor(np.stack([norm(z[fit[d['user'][fit]==u]].mean(0)) for u in f['base']])))
    opt=torch.optim.AdamW(list(net.parameters())+list(head.parameters()),lr=1e-3,weight_decay=1e-4);best=auc(z[dev],d['user'][dev],d['word'][dev]);state=copy.deepcopy(net.state_dict());selected=0;raw_grad=None;history=[]
    for step in range(1,steps+1):
        ww=rng.choice(words,2,replace=False);ix=np.array([i for u in rng.permutation(f['base']) for w in ww for i in rng.choice(lookup[u,w],2,replace=False)]);y=torch.tensor([labels[u] for u in d['user'][ix]],dtype=torch.long);zz=torch.tensor(z[ix]);xx=torch.tensor(x[ix],requires_grad=(step==2));v=net(zz,xx);loss=F.cross_entropy(16*F.linear(v,F.normalize(head.weight,dim=1)),y)+.1*supcon(v,y)+.1*(1-(v*zz).sum(1)).mean()
        if mode=='fusion_audio':
            tt=torch.tensor(t[ix]);err=(v@v.T-tt@tt.T).square();eye=torch.eye(len(y),dtype=torch.bool);same=(y[:,None]==y[None,:])&~eye;diff=y[:,None]!=y[None,:];loss=loss+.05*(err[same].mean()+err[diff].mean())
        opt.zero_grad();loss.backward()
        if step==2:raw_grad=float(xx.grad.norm());assert raw_grad>0,'Raw branch must participate in the identity loss'
        opt.step()
        if step%100==0 or step==steps:
            with torch.no_grad():q=auc(net(torch.tensor(z[dev]),torch.tensor(x[dev])).numpy(),d['user'][dev],d['word'][dev])
            history.append(dict(step=step,dev_auc=q))
            if q>best:best=q;state=copy.deepcopy(net.state_dict());selected=step
    net.load_state_dict(state)
    with torch.no_grad():v=net(torch.tensor(z),torch.tensor(x)).numpy()
    return v,dict(selected_step=selected,base_dev_auc=best,raw_input_gradient_step2=raw_grad,history=history),state
def scores(z,d,client,attackers,cond,direction):
    a=[24,25,26] if direction==0 else [27,28,29];b=[27,28,29] if direction==0 else [24,25,26];ew=b if cond=='sameword' else a
    def rows(u,w):
        ix=np.flatnonzero((d['user']==u)&(d['word']==w)&(d['role']=='final'))[:8];assert len(ix)==8,(u,w,len(ix));return ix
    en=np.concatenate([rows(client,w)[:4] for w in ew]);query=np.concatenate([rows(u,w)[4:8] for u in [client]+attackers for w in b]);assert not set(en)&set(query);return z[query]@norm(z[en].mean(0)),d['user'][query]==client
def evaluate(z,d,f):
    cs=[];cy=[]
    for cond in ['sameword','crossword']:
        for direction in [0,1]:
            for u in f['calibration']:
                s,y=scores(z,d,u,[v for v in f['calibration'] if v!=u],cond,direction);cs.extend(s);cy.extend(y)
    cs=np.array(cs);cy=np.array(cy);vals=np.r_[np.unique(cs),np.nextafter(cs.max(),np.inf)];tau=min(vals,key=lambda t:(abs(float((cs[cy]<t).mean())-float((cs[~cy]>=t).mean())),float((cs[~cy]>=t).mean())))
    result={}
    for cond in ['sameword','crossword']:
        result[cond]={}
        for scope,attacks in [('two_unseen_attackers',f['unseen_attackers']),('all_eight_sources',[u for u in range(1,10) if u!=f['target']])]:
            ss=[];yy=[]
            for direction in [0,1]:s,y=scores(z,d,f['target'],attacks,cond,direction);ss.extend(s);yy.extend(y)
            ss=np.array(ss);yy=np.array(yy);tar=float((ss[yy]>=tau).mean());far=float((ss[~yy]>=tau).mean());result[cond][scope]=dict(tar=tar,far=far,hter=(1-tar+far)/2,auc=float(roc_auc_score(yy,ss)),genuine_n=int(yy.sum()),impostor_n=int((~yy).sum()),threshold=float(tau))
    return result
def main(a):
    torch.set_num_threads(1);out=Path(a.out);out.mkdir(exist_ok=True,parents=True);bundle=Path(a.bundle);meta=np.load(bundle/'metadata_private.npz');raw=np.load(a.raw);index={s:i for i,s in enumerate(meta['sample_id'])};ix=np.array([index[s] for s in raw['sample_id']]);d={k:meta[k][ix] for k in ['user','word','role','sample_id']};assert np.array_equal(d['user'],raw['user']) and np.array_equal(d['word'],raw['word']);assert len(set(d['sample_id']))==2695;t=np.load(bundle/'audio_private.npy')[ix].astype('float32');feature=raw['velocity_XYZ'];results={};start=time.time();provenance=json.loads((bundle/'provenance_private.json').read_text())
    for target in range(1,10):
        f=fold(target);assert len(set(f['base']+f['calibration']+[target]+f['unseen_attackers']))==9;folder=out/('target'+str(target));folder.mkdir(exist_ok=True)
        if (folder/'results_private.json').exists():results[str(target)]=json.loads((folder/'results_private.json').read_text());continue
        key='target'+str(f['owner'])+'_seed42';assert provenance[key]['base_users']==f['base'];z=np.load(bundle/(key+'_full_embeddings_private.npy'))[ix].astype('float32');fit=np.isin(d['user'],f['base'])&(d['role']=='fit');assert not np.isin(d['user'][fit],[target]+f['calibration']+f['unseen_attackers']).any();sc=StandardScaler().fit(feature[fit]);scaled=sc.transform(feature).astype('float32');pc=PCA(n_components=64,svd_solver='randomized',random_state=42).fit(scaled[fit]);x=norm(pc.transform(scaled).astype('float32'));outputs={'raw_statistics':norm(scaled),'frozen_encoder':z,'fixed_concat':np.concatenate([z,norm(scaled)],axis=1)/np.sqrt(2)};selection={}
        for mode in ['fusion_identity','fusion_audio']:
            path=folder/(mode+'_embeddings_private.npy')
            if path.exists():outputs[mode]=np.load(path);selection[mode]=json.loads((folder/(mode+'_selection_private.json')).read_text());continue
            outputs[mode],sel,state=train(z,x,t,d,f,mode,a.steps);selection[mode]=sel;torch.save(state,folder/(mode+'_private.pt'));np.save(path,outputs[mode]);(folder/(mode+'_selection_private.json')).write_text(json.dumps(sel));print('TRAINED',target,mode,sel['selected_step'],flush=True)
        result=dict(fold=f,selection=selection,methods={name:evaluate(v,d,f) for name,v in outputs.items()});(folder/'results_private.json').write_text(json.dumps(result,indent=2));results[str(target)]=result;(out/'results_private.json').write_text(json.dumps(results,indent=2));(out/'status.json').write_text(json.dumps(dict(state='running',completed_targets=len(results),total=9)));print('TARGET_COMPLETE',target,flush=True)
    summary={}
    for cond in ['sameword','crossword']:
        summary[cond]={}
        for method in ['raw_statistics','frozen_encoder','fixed_concat','fusion_identity','fusion_audio']:
            summary[cond][method]={scope:{k:float(np.mean([v['methods'][method][cond][scope][k] for v in results.values()])) for k in ['tar','far','hter','auc']} for scope in ['two_unseen_attackers','all_eight_sources']}
    (out/'summary_public.json').write_text(json.dumps(dict(state='complete',users=9,seed=42,results=summary,steps=a.steps,elapsed_seconds=time.time()-start,protocol='4Base2Cal1target2unseenattacks;target9reusesfold1backbone;Base-fit-only StandardScaler/PCA64;frozenidentity-disjoint encoder;raw720dimbranch;12shot;24genuine48unseenqueries;Cal-only mutual attacks pooledsame/cross;all8sourceanalysis separate',limits=['inferred wordlabels not independently verified','eight original backbones reused,new ninth target fold is dependent on fold1','single seed;explored population','new calibration/attacker scope differs from previous eight-user results;all baselines rerun','small residual may select initial no-change model']),indent=2));(out/'status.json').write_text(json.dumps(dict(state='complete',completed_targets=9)));print('ALL_COMPLETE',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bundle',required=True);p.add_argument('--raw',required=True);p.add_argument('--out',required=True);p.add_argument('--steps',type=int,default=1000);main(p.parse_args())
