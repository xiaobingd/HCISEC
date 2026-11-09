"""Independent dataset check of factor dominance and learned identity geometry."""
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
import run_mei9_open_raw_fusion as p
from mei9_controlled_models import ResNet18

def stats(b):
    m=b.mean(-1);s=b.std(-1);seg=np.stack([b[:,:,:,q].mean(-1) for q in np.array_split(np.arange(28),4)],axis=2);return np.concatenate([m[:,:,None],s[:,:,None],seg],axis=2).reshape(len(b),-1)
def factors(b,a):
    m=b.mean(-1);level=m.mean(-1);s=b.std(-1);seg=np.stack([b[:,:,:,q].mean(-1) for q in np.array_split(np.arange(28),4)],axis=2)
    return dict(shape=(m-level[:,:,None]).reshape(len(b),-1),std=s.reshape(len(b),-1),time=(seg-m[:,:,None]).reshape(len(b),-1),amplitude=np.c_[level,a])
def pairauc(z,y):
    sim=z@z.T;upper=np.triu(np.ones(sim.shape,bool),1);pos=upper&(y[:,None]==y[None,:]);neg=upper&(y[:,None]!=y[None,:]);return float(roc_auc_score(np.r_[np.ones(pos.sum()),np.zeros(neg.sum())],np.r_[sim[pos],sim[neg]]))
def scores(z,d,client,attackers,query_z=None):
    en=np.flatnonzero((d['user']==client)&(d['role']=='registration'));q=np.flatnonzero(np.isin(d['user'],[client]+attackers)&(d['role']=='query'));assert len(en)==12 and np.sum(d['user'][q]==client)==24;assert not set(en)&set(q);return (z if query_z is None else query_z)[q]@p.norm(z[en].mean(0)),d['user'][q]==client
def evaluate(z,d,f,query_z=None):
    cs=[];cy=[]
    for u in f['calibration']:
        s,y=scores(z,d,u,[v for v in f['calibration'] if v!=u]);cs.extend(s);cy.extend(y)
    cs=np.array(cs);cy=np.array(cy);ts=np.r_[np.unique(cs),np.nextafter(cs.max(),np.inf)];tau=min(ts,key=lambda t:(abs(float((cs[cy]<t).mean())-float((cs[~cy]>=t).mean())),float((cs[~cy]>=t).mean())))
    result={}
    for scope,attacks in [('two_unseen_attackers',f['unseen_attackers']),('all_eight_sources',[u for u in range(1,10) if u!=f['target']])]:
        s,y=scores(z,d,f['target'],attacks,query_z);tar=float((s[y]>=tau).mean());far=float((s[~y]>=tau).mean());result[scope]=dict(tar=tar,far=far,hter=(1-tar+far)/2,auc=float(roc_auc_score(y,s)),genuine_n=int(y.sum()),impostor_n=int((~y).sum()),threshold=float(tau))
    return result
def aggregate(rows):return {method:{scope:{k:float(np.mean([v['methods'][method][scope][k] for v in rows.values()])) for k in ['tar','far','hter','auc']} for scope in ['two_unseen_attackers','all_eight_sources']} for method in next(iter(rows.values()))['methods']}
def ablate(d,out):
    fac=factors(d['spectrum'],d['amplitude']);definitions={'all_factors':['shape','std','time','amplitude'],'without_shape':['std','time','amplitude'],'without_std':['shape','time','amplitude'],'without_time':['shape','std','amplitude'],'without_amplitude':['shape','std','time'],'shape_only':['shape'],'std_only':['std'],'time_only':['time'],'amplitude_only':['amplitude']};v=stats(d['spectrum']);center=stats(d['spectrum']-d['spectrum'].mean((2,3),keepdims=True));rows={}
    for target in range(1,10):
        f=p.fold(target);fit=np.isin(d['user'],f['base'])&(d['role']=='fit');scaled={k:StandardScaler().fit(x[fit]).transform(x).astype('float32')/np.sqrt(x.shape[1]) for k,x in fac.items()};vectors={name:p.norm(np.concatenate([scaled[k] for k in ks],1)) for name,ks in definitions.items()};sc=StandardScaler().fit(v[fit]);vectors['original_statistics']=p.norm(sc.transform(v));vectors['no_axis_level']=p.norm(StandardScaler().fit(center[fit]).transform(center))
        for band in ['low','mid']:
            x=stats(d[band+'_spectrum']);vectors[band+'_band_statistics']=p.norm(StandardScaler().fit(x[fit]).transform(x))
        result=dict(fold=f,methods={name:evaluate(z,d,f) for name,z in vectors.items()},gain_stress={})
        shape_sc=StandardScaler().fit(fac['shape'][fit])
        for gain,label in [(.5,'gain_half'),(2.,'gain_double')]:
            bb=d['spectrum']+2*np.log(gain);result['gain_stress'][label]=evaluate(vectors['original_statistics'],d,f,p.norm(sc.transform(stats(bb))))
            shape=factors(bb,d['amplitude']+np.log(gain))['shape'];result['gain_stress']['shape_'+label]=evaluate(vectors['shape_only'],d,f,p.norm(shape_sc.transform(shape)))
        rows[str(target)]=result;print('ABLATION_TARGET',target,flush=True)
    (out/'ablation_results_private.json').write_text(json.dumps(rows));stress={name:{k:float(np.mean([v['gain_stress'][name]['two_unseen_attackers'][k] for v in rows.values()])) for k in ['tar','far','hter','auc']} for name in ['gain_half','gain_double','shape_gain_half','shape_gain_double']};(out/'ablation_summary_public.json').write_text(json.dumps(dict(state='complete',targets=9,results=aggregate(rows),simulated_gain_query_only=stress,protocol='real-text globally disjoint registration/query;no matched same-content claim;4Base2Cal1target2unseen;12shot24genuine48impostor;noAudio;gain linear,no ADC clipping'),indent=2))
class MLP(nn.Module):
    def __init__(self):super().__init__();self.net=nn.Sequential(nn.Linear(240,256),nn.GELU(),nn.Linear(256,192))
    def forward(self,x):return F.normalize(self.net(x),dim=1)
class AccResNet(ResNet18):
    def __init__(self):super().__init__();self.conv1=nn.Conv2d(1,64,7,2,3,bias=False);nn.init.kaiming_normal_(self.conv1.weight,mode='fan_out',nonlinearity='relu')
def encode(net,x):
    net.eval()
    with torch.no_grad():return np.concatenate([net(torch.tensor(x[i:i+32])).numpy() for i in range(0,len(x),32)])
def train(name,x,d,f,args,folder):
    path=folder/(name+'_embeddings_private.npy')
    if path.exists():return np.load(path),json.loads((folder/(name+'_training_private.json')).read_text())
    torch.manual_seed(42);rng=np.random.RandomState(42);net=MLP() if name=='stats_mlp' else AccResNet();head=nn.Linear(192,4);opt=torch.optim.AdamW(list(net.parameters())+list(head.parameters()),lr=1e-4,weight_decay=1e-4);fit=np.flatnonzero(np.isin(d['user'],f['base'])&(d['role']=='fit'));dev=np.flatnonzero(np.isin(d['user'],f['base'])&(d['role']=='develop'));labels={u:j for j,u in enumerate(f['base'])};lookup={u:fit[d['user'][fit]==u] for u in f['base']};best=-1;history=[];state=None;selected=0;started=time.time()
    for step in range(1,args.steps+1):
        net.train();ii=np.concatenate([rng.choice(lookup[u],4,replace=False) for u in rng.permutation(f['base'])]);yy=torch.tensor([labels[u] for u in d['user'][ii]],dtype=torch.long);z=net(torch.tensor(x[ii]));loss=F.cross_entropy(head(z),yy)+.1*p.supcon(z,yy);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(list(net.parameters())+list(head.parameters()),1.);opt.step()
        if step==1 or step%25==0:print('TRAIN',f['owner'],name,step,float(loss.detach()),flush=True)
        if step%50==0 or step==args.steps:
            q=pairauc(encode(net,x[dev]),d['user'][dev]);history.append(dict(step=step,dev_auc=q))
            if q>best:best=q;state=copy.deepcopy(net.state_dict());selected=step
    assert state is not None;net.load_state_dict(state);z=encode(net,x);np.save(path,z);torch.save(state,folder/(name+'_private.pt'));info=dict(selected_step=selected,history=history,fit_pair_auc=pairauc(z[fit],d['user'][fit]),dev_pair_auc=best,parameters=sum(v.numel() for v in net.parameters()),seconds=time.time()-started);(folder/(name+'_training_private.json')).write_text(json.dumps(info));return z,info
def neural(d,out,args):
    b=d['spectrum'];results={}
    for owner in args.owners:
        f=p.fold(owner);folder=out/('owner'+str(owner));folder.mkdir(exist_ok=True);fit=np.isin(d['user'],f['base'])&(d['role']=='fit');mu=b[fit].mean((0,2,3),keepdims=True);sd=np.maximum(b[fit].std((0,2,3),keepdims=True),1e-4);x=((b-mu)/sd).astype('float32');v=stats(x);sc=StandardScaler().fit(v[fit]);sv=sc.transform(v).astype('float32');vectors={'original_statistics':p.norm(sv)};pc=PCA(n_components=192,svd_solver='randomized',random_state=42).fit(sv[fit]);vectors['statistics_pca192']=p.norm(pc.transform(sv));infos={}
        for name,xx in [('stats_mlp',sv),('resnet18',x)]:vectors[name],infos[name]=train(name,xx,d,f,args,folder)
        for target in ([owner,9] if owner==1 else [owner]):
            result=dict(fold=p.fold(target),methods={method:evaluate(z,d,p.fold(target)) for method,z in vectors.items()},training=infos);(out/('target'+str(target)+'_neural_private.json')).write_text(json.dumps(result,indent=2));results[str(target)]=result
        print('OWNER_COMPLETE',owner,flush=True)
    for path in out.glob('target*_neural_private.json'):results[path.name.split('_')[0][6:]]=json.loads(path.read_text())
    (out/'neural_summary_public.json').write_text(json.dumps(dict(state='complete' if len(results)==9 else 'partial',completed_targets=len(results),results=aggregate(results),protocol='VibraVox forehead1axis;wide50..7600Hz1x40x28;4Base2Cal1target2unseen;12shot;true text isolation;no same-content claim;noAudio;CE+.1SupCon200steps'),indent=2))
def main(args):
    torch.set_num_threads(args.threads);out=Path(args.out);out.mkdir(exist_ok=True,parents=True);d=dict(np.load(args.data));assert d['spectrum'].shape[1:]==(1,40,28);assert np.isfinite(d['spectrum']).all();sets=[set(d['text'][d['role']==k]) for k in ['fit','develop','registration','query']];assert all(not sets[i]&sets[j] for i in range(4) for j in range(i))
    if args.phase in ['ablation','all']:ablate(d,out)
    if args.phase in ['neural','all']:neural(d,out,args)
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--data',required=True);a.add_argument('--out',required=True);a.add_argument('--phase',choices=['ablation','neural','all'],default='all');a.add_argument('--owners',nargs='+',type=int,default=list(range(1,9)));a.add_argument('--steps',type=int,default=200);a.add_argument('--threads',type=int,default=1);main(a.parse_args())
