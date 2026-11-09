"""Eight-identity ABCS replication; matched actual text and identity-disjoint evaluation."""
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
from run_mei9_open_raw_fusion import norm,supcon
from run_vibravox_factor_validation import MLP,AccResNet,encode

def fold(target):
    order=[1+(target-1+j)%8 for j in range(8)];return dict(target=target,owner=target,base=order[4:],calibration=order[2:4],unseen_attackers=[order[1]])
def stats(b):
    m=b.mean(-1);s=b.std(-1);seg=np.stack([b[:,:,:,q].mean(-1) for q in np.array_split(np.arange(b.shape[-1]),4)],axis=2);return np.concatenate([m[:,:,None],s[:,:,None],seg],2).reshape(len(b),-1)
def factors(b,a):
    m=b.mean(-1);level=m.mean(-1);seg=np.stack([b[:,:,:,q].mean(-1) for q in np.array_split(np.arange(b.shape[-1]),4)],2);return dict(shape=(m-level[:,:,None]).reshape(len(b),-1),std=b.std(-1).reshape(len(b),-1),time=(seg-m[:,:,None]).reshape(len(b),-1),amplitude=np.c_[level,a])
def pairauc(z,y,t):
    sim=z@z.T;upper=np.triu(np.ones(sim.shape,bool),1);pos=upper&(y[:,None]==y[None,:])&(t[:,None]!=t[None,:]);neg=upper&(y[:,None]!=y[None,:])&(t[:,None]==t[None,:]);assert pos.any() and neg.any();return float(roc_auc_score(np.r_[np.ones(pos.sum()),np.zeros(neg.sum())],np.r_[sim[pos],sim[neg]]))
def scores(z,d,client,attackers,query=None,query_role='query'):
    en=np.flatnonzero((d['user']==client)&(d['role']=='registration'));ix=np.flatnonzero(np.isin(d['user'],[client]+attackers)&(d['role']==query_role));assert len(en)==12 and np.sum(d['user'][ix]==client)==(13 if query_role=='calibration' else 24);return (z if query is None else query)[ix]@norm(z[en].mean(0)),d['user'][ix]==client
def evaluate(z,d,f,query=None):
    cs=[];cy=[]
    for u in f['calibration']:
        s,y=scores(z,d,u,[v for v in f['calibration'] if v!=u],query_role='calibration');cs.extend(s);cy.extend(y)
    cs=np.array(cs);cy=np.array(cy);ts=np.r_[np.unique(cs),np.nextafter(cs.max(),np.inf)];tau=min(ts,key=lambda t:(abs(float((cs[cy]<t).mean())-float((cs[~cy]>=t).mean())),float((cs[~cy]>=t).mean())))
    result={}
    for scope,attacks in [('unseen_attacker',f['unseen_attackers']),('all_seven_sources',[u for u in range(1,9) if u!=f['target']])]:
        s,y=scores(z,d,f['target'],attacks,query);tar=float((s[y]>=tau).mean());far=float((s[~y]>=tau).mean());result[scope]=dict(tar=tar,far=far,hter=(1-tar+far)/2,auc=float(roc_auc_score(y,s)),genuine_n=int(y.sum()),impostor_n=int((~y).sum()),threshold=float(tau))
    return result
def aggregate(rows):return {method:{scope:{k:float(np.mean([v['methods'][method][scope][k] for v in rows.values()])) for k in ['tar','far','hter','auc']} for scope in ['unseen_attacker','all_seven_sources']} for method in next(iter(rows.values()))['methods']}
def ablate(d,out):
    fac=factors(d['spectrum'],d['amplitude']);definitions={'all_factors':['shape','std','time','amplitude'],'without_shape':['std','time','amplitude'],'without_std':['shape','time','amplitude'],'without_time':['shape','std','amplitude'],'without_amplitude':['shape','std','time'],'shape_only':['shape'],'std_only':['std'],'time_only':['time'],'amplitude_only':['amplitude']};v=stats(d['spectrum']);center=stats(d['spectrum']-d['spectrum'].mean((2,3),keepdims=True));rows={}
    for target in range(1,9):
        f=fold(target);fit=np.isin(d['user'],f['base'])&(d['role']=='fit');scaled={k:StandardScaler().fit(x[fit]).transform(x).astype('float32')/np.sqrt(x.shape[1]) for k,x in fac.items()};vectors={name:norm(np.concatenate([scaled[k] for k in ks],1)) for name,ks in definitions.items()};sc=StandardScaler().fit(v[fit]);vectors['original_statistics']=norm(sc.transform(v));vectors['no_axis_level']=norm(StandardScaler().fit(center[fit]).transform(center))
        for name,key in [('low_band_statistics','low_spectrum'),('mid_band_statistics','mid_spectrum'),('other_channel_statistics','other_spectrum'),('center_window_statistics','center_spectrum')]:
            x=stats(d[key]);vectors[name]=norm(StandardScaler().fit(x[fit]).transform(x))
        result=dict(fold=f,methods={name:evaluate(z,d,f) for name,z in vectors.items()},gain_stress={});shape_sc=StandardScaler().fit(fac['shape'][fit])
        for gain,label in [(.5,'gain_half'),(2.,'gain_double')]:
            bb=d['spectrum']+2*np.log(gain);result['gain_stress'][label]=evaluate(vectors['original_statistics'],d,f,norm(sc.transform(stats(bb))));shape=factors(bb,d['amplitude']+np.log(gain))['shape'];result['gain_stress']['shape_'+label]=evaluate(vectors['shape_only'],d,f,norm(shape_sc.transform(shape)))
        rows[str(target)]=result;print('ABLATION_TARGET',target,flush=True)
    (out/'ablation_results_private.json').write_text(json.dumps(rows));stress={name:{k:float(np.mean([v['gain_stress'][name]['unseen_attacker'][k] for v in rows.values()])) for k in ['tar','far','hter','auc']} for name in ['gain_half','gain_double','shape_gain_half','shape_gain_double']};(out/'ablation_summary_public.json').write_text(json.dumps(dict(state='complete',targets=8,results=aggregate(rows),simulated_gain_query_only=stress,protocol='8actual shared-text users;4Base2Cal1target1unseen;12shot24genuine24unseenqueries;all query texts matched;channel0 provisional'),indent=2))
def train(name,x,d,f,args,folder):
    path=folder/(name+'_embeddings_private.npy')
    if path.exists():return np.load(path),json.loads((folder/(name+'_training_private.json')).read_text())
    torch.manual_seed(42);rng=np.random.RandomState(42);net=MLP() if name=='stats_mlp' else AccResNet();head=nn.Linear(192,4);opt=torch.optim.AdamW(list(net.parameters())+list(head.parameters()),lr=1e-4,weight_decay=1e-4);fit=np.flatnonzero(np.isin(d['user'],f['base'])&(d['role']=='fit'));dev=np.flatnonzero(np.isin(d['user'],f['base'])&(d['role']=='develop'));texts=np.unique(d['text_index'][fit]);lookup={(u,t):int(fit[(d['user'][fit]==u)&(d['text_index'][fit]==t)][0]) for u in f['base'] for t in texts};labels={u:j for j,u in enumerate(f['base'])};best=-1;state=None;history=[];selected=0;started=time.time()
    for step in range(1,args.steps+1):
        net.train();tt=rng.choice(texts,4,replace=False);ii=np.array([lookup[u,t] for u in rng.permutation(f['base']) for t in tt]);y=torch.tensor([labels[u] for u in d['user'][ii]],dtype=torch.long);z=net(torch.tensor(x[ii]));loss=F.cross_entropy(head(z),y)+.1*supcon(z,y);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(list(net.parameters())+list(head.parameters()),1.);opt.step()
        if step==1 or step%25==0:print('TRAIN',f['target'],name,step,float(loss.detach()),flush=True)
        if step%50==0 or step==args.steps:
            q=pairauc(encode(net,x[dev]),d['user'][dev],d['text_index'][dev]);history.append(dict(step=step,dev_auc=q))
            if q>best:best=q;state=copy.deepcopy(net.state_dict());selected=step
    assert state is not None;net.load_state_dict(state);z=encode(net,x);np.save(path,z);torch.save(state,folder/(name+'_private.pt'));info=dict(selected_step=selected,history=history,fit_pair_auc=pairauc(z[fit],d['user'][fit],d['text_index'][fit]),dev_pair_auc=best,parameters=sum(p.numel() for p in net.parameters()),seconds=time.time()-started);(folder/(name+'_training_private.json')).write_text(json.dumps(info));return z,info
def neural(d,out,args):
    b=d['spectrum'];rows={}
    for target in args.targets:
        f=fold(target);folder=out/('target'+str(target));folder.mkdir(exist_ok=True);fit=np.isin(d['user'],f['base'])&(d['role']=='fit');mu=b[fit].mean((0,2,3),keepdims=True);sd=np.maximum(b[fit].std((0,2,3),keepdims=True),1e-4);x=((b-mu)/sd).astype('float32');v=stats(x);sc=StandardScaler().fit(v[fit]);sv=sc.transform(v).astype('float32');vectors={'original_statistics':norm(sv)};pc=PCA(n_components=192,svd_solver='randomized',random_state=42).fit(sv[fit]);vectors['statistics_pca192']=norm(pc.transform(sv));infos={}
        for name,xx in [('stats_mlp',sv),('resnet18',x)]:vectors[name],infos[name]=train(name,xx,d,f,args,folder)
        result=dict(fold=f,methods={name:evaluate(z,d,f) for name,z in vectors.items()},training=infos);(out/('target'+str(target)+'_neural_private.json')).write_text(json.dumps(result,indent=2));print('TARGET_COMPLETE',target,flush=True)
    for path in out.glob('target*_neural_private.json'):rows[path.name.split('_')[0][6:]]=json.loads(path.read_text())
    (out/'neural_summary_public.json').write_text(json.dumps(dict(state='complete' if len(rows)==8 else 'partial',completed_targets=len(rows),results=aggregate(rows),protocol='ABCS actual matched text;8fold open-set;channel0 provisional;50..7600Hz1x40x16;whole-utterance time normalization,no waveform padding;CE+.1SupCon200steps;12shot;noAudio'),indent=2))
def main(args):
    torch.set_num_threads(args.threads);out=Path(args.out);out.mkdir(exist_ok=True,parents=True);d=dict(np.load(args.data));assert d['spectrum'].shape==(992,1,40,16);assert np.isfinite(d['spectrum']).all();groups=[set(d['text'][d['role']==k]) for k in ['fit','develop','registration','calibration','query']];assert all(not groups[i]&groups[j] for i in range(5) for j in range(i))
    for role in ['fit','develop','registration','calibration','query']:
        sets=[set(d['text'][(d['user']==u)&(d['role']==role)]) for u in range(1,9)];assert all(s==sets[0] for s in sets)
    if args.phase in ['ablation','all']:ablate(d,out)
    if args.phase in ['neural','all']:neural(d,out,args)
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--data',required=True);a.add_argument('--out',required=True);a.add_argument('--phase',choices=['ablation','neural','all'],default='all');a.add_argument('--targets',nargs='+',type=int,default=list(range(1,9)));a.add_argument('--steps',type=int,default=200);a.add_argument('--threads',type=int,default=1);main(a.parse_args())
