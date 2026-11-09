"""Controlled raw-factor study and neural encoders with a common HMD frontend."""
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
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import run_mei9_open_raw_fusion as p
from mei9_controlled_models import ResNet18,StatsMLP

def stats(b):
    mean=b.mean(-1);std=b.std(-1);seg=np.stack([b[:,:,:,q].mean(-1) for q in np.array_split(np.arange(b.shape[-1]),4)],axis=2)
    return np.concatenate([mean[:,:,None,:],std[:,:,None,:],seg],axis=2).reshape(len(b),720)
def factors(b,amplitude):
    m=b.mean(-1);s=b.std(-1);seg=np.stack([b[:,:,:,q].mean(-1) for q in np.array_split(np.arange(b.shape[-1]),4)],axis=2);level=m.mean(-1);shape=m-level[:,:,None];time=seg-m[:,:,None,:]
    return dict(shape=shape.reshape(len(b),-1),std=s.reshape(len(b),-1),time=time.reshape(len(b),-1),amplitude=np.concatenate([level,amplitude],axis=1))
def subset_frames(d):
    mask=np.zeros(len(d['user']),bool)
    for u in range(1,10):
        for w in range(24,30):
            ix=np.flatnonzero((d['user']==u)&(d['word']==w)&(d['role']=='final'))[:8];assert len(ix)==8;mask[ix[4:8]]=True
    return mask
def perturbed_eval(native,changed,d,f):
    baseline=p.evaluate(native,d,f);mask=subset_frames(d);z=native.copy();z[mask]=changed[mask];out={}
    for cond in ['sameword','crossword']:
        out[cond]={}
        for scope,attackers in [('two_unseen_attackers',f['unseen_attackers']),('all_eight_sources',[u for u in range(1,10) if u!=f['target']])]:
            ss=[];yy=[]
            for direction in [0,1]:s,y=p.scores(z,d,f['target'],attackers,cond,direction);ss.extend(s);yy.extend(y)
            ss=np.array(ss);yy=np.array(yy);tau=baseline[cond][scope]['threshold'];tar=float((ss[yy]>=tau).mean());far=float((ss[~yy]>=tau).mean());out[cond][scope]=dict(tar=tar,far=far,hter=(1-tar+far)/2,auc=float(p.roc_auc_score(yy,ss)))
    return out
def macro(rows):
    return {cond:{method:{scope:{k:float(np.mean([v['methods'][method][cond][scope][k] for v in rows.values()])) for k in ['tar','far','hter','auc']} for scope in ['two_unseen_attackers','all_eight_sources']} for method in next(iter(rows.values()))['methods']} for cond in ['sameword','crossword']}
def descriptives(a,out):
    groups=factors(a['spectrum'],a['amplitude']);groups['quality']=a['quality'];u=a['user'];w=a['word'];user=np.eye(9)[u-1];word=np.eye(30)[w];date=np.eye(len(np.unique(a['record_date'])))[np.searchsorted(np.unique(a['record_date']),a['record_date'])]
    result={}
    for name,v in groups.items():
        y=StandardScaler().fit_transform(v);total=np.sum((y-y.mean(0))**2)
        def r2(x):return float(1-np.sum((y-x@np.linalg.lstsq(x,y,rcond=None)[0])**2)/total)
        result[name]=dict(user_r2=r2(user),word_r2=r2(word),user_plus_word_r2=r2(np.c_[user,word]))
    (out/'descriptive_public.json').write_text(json.dumps(dict(component_variance=result,user_design_rank=int(np.linalg.matrix_rank(user)),user_plus_date_design_rank=int(np.linalg.matrix_rank(np.c_[user,date])),interpretation='all-record descriptive ANOVA only,not fitting authentication;date is collinear with identity when each user has one recording date;not evidence of biological origin',user_confirmed_same_headset=True,user_confirmed_no_redon_or_crossday_repeat=True),indent=2))
def ablations(a,out):
    b=a['spectrum'];amp=a['amplitude'];fac=factors(b,amp);base_stats=stats(b);centered_stats=stats(b-b.mean((2,3),keepdims=True));d={k:a[k] for k in ['user','word','role','sample_id']};results={}
    definitions={'all_factors':['shape','std','time','amplitude'],'without_mean_shape':['std','time','amplitude'],'without_std':['shape','time','amplitude'],'without_time':['shape','std','amplitude'],'without_amplitude':['shape','std','time'],'shape_only':['shape'],'std_only':['std'],'time_only':['time'],'amplitude_only':['amplitude']}
    descriptives(a,out)
    for target in range(1,10):
        f=p.fold(target);fit=np.isin(d['user'],f['base'])&(d['role']=='fit');scalers={k:StandardScaler().fit(v[fit]) for k,v in fac.items()};scaled={k:scalers[k].transform(v).astype('float32')/np.sqrt(v.shape[1]) for k,v in fac.items()};vectors={name:p.norm(np.concatenate([scaled[k] for k in keys],1)) for name,keys in definitions.items()}
        raw_sc=StandardScaler().fit(base_stats[fit]);vectors['original_statistics']=p.norm(raw_sc.transform(base_stats));center_sc=StandardScaler().fit(centered_stats[fit]);vectors['original_statistics_no_axis_level']=p.norm(center_sc.transform(centered_stats));qsc=StandardScaler().fit(a['quality'][fit]);vectors['collection_quality_only']=p.norm(qsc.transform(a['quality']))
        for name,sl in [('sampling_clock_only',slice(0,7)),('tracking_update_only',slice(7,9))]:
            vv=a['quality'][:,sl];ss=StandardScaler().fit(vv[fit]);vectors[name]=p.norm(ss.transform(vv))
        methods={name:p.evaluate(v,d,f) for name,v in vectors.items()};stress={}
        for name,gain,swap in [('gain_half',.5,False),('gain_double',2.,False),('axis_XY_swap',1.,True)]:
            bb=b+2*np.log(gain);aa=amp+np.log(gain)
            if swap:bb=bb[:,[1,0,2]];aa=aa.reshape(len(aa),3,5)[:,[1,0,2]].reshape(len(aa),15)
            transformed=p.norm(raw_sc.transform(stats(bb)));stress[name]=perturbed_eval(vectors['original_statistics'],transformed,d,f)
        results[str(target)]=dict(fold=f,methods=methods,stress=stress);(out/'ablation_results_private.json').write_text(json.dumps(results));print('ABLATION_TARGET',target,flush=True)
    summary=macro(results);stress={cond:{kind:{k:float(np.mean([v['stress'][kind][cond]['two_unseen_attackers'][k] for v in results.values()])) for k in ['tar','far','hter','auc']} for kind in ['gain_half','gain_double','axis_XY_swap']} for cond in ['sameword','crossword']};(out/'ablation_summary_public.json').write_text(json.dumps(dict(state='complete',results=summary,simulated_query_only_stress=stress,notes='factor blocks fit-only StandardScaler and 1/sqrt(blockdim);no real redon/device sessions;no mean leakage through raw segmentmeans;main protocol unchanged'),indent=2))
def encode(net,x):
    net.eval()
    with torch.no_grad():return np.concatenate([net(torch.tensor(x[i:i+32])).numpy() for i in range(0,len(x),32)])
def train_model(name,x,d,f,steps,folder):
    complete=folder/(name+'_embeddings_private.npy')
    if complete.exists():return np.load(complete),json.loads((folder/(name+'_training_private.json')).read_text())
    torch.manual_seed(42);rng=np.random.RandomState(42);net=ResNet18() if name=='resnet18' else StatsMLP();head=nn.Linear(192,4);opt=torch.optim.AdamW(list(net.parameters())+list(head.parameters()),lr=1e-4,weight_decay=1e-4);fit=np.flatnonzero(np.isin(d['user'],f['base'])&(d['role']=='fit'));dev=np.flatnonzero(np.isin(d['user'],f['base'])&(d['role']=='develop'));words=np.unique(d['word'][fit]);lookup={(u,w):fit[(d['user'][fit]==u)&(d['word'][fit]==w)] for u in f['base'] for w in words};labels={u:i for i,u in enumerate(f['base'])};best=-1;history=[];started=time.time();state=None;selected=0
    for step in range(1,steps+1):
        net.train();ww=rng.choice(words,2,replace=False);ix=np.array([i for u in rng.permutation(f['base']) for w in ww for i in rng.choice(lookup[u,w],2,replace=False)]);y=torch.tensor([labels[u] for u in d['user'][ix]],dtype=torch.long);z=net(torch.tensor(x[ix]));loss=F.cross_entropy(head(z),y)+.1*p.supcon(z,y);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(list(net.parameters())+list(head.parameters()),1.);opt.step()
        if step==1 or step%25==0:print('TRAIN',f['owner'],name,step,float(loss.detach()),flush=True)
        if step%50==0 or step==steps:
            q=p.auc(encode(net,x[dev]),d['user'][dev],d['word'][dev]);history.append(dict(step=step,dev_auc=q))
            if q>best:best=q;state=copy.deepcopy(net.state_dict());selected=step
    assert state is not None;net.load_state_dict(state);z=encode(net,x);np.save(complete,z);torch.save(state,folder/(name+'_private.pt'));info=dict(selected_step=selected,history=history,parameters=sum(v.numel() for v in net.parameters()),fit_pair_auc=p.auc(z[fit],d['user'][fit],d['word'][fit]),dev_pair_auc=best,seconds=time.time()-started);(folder/(name+'_training_private.json')).write_text(json.dumps(info));return z,info
def neural(a,out,args):
    d={k:a[k] for k in ['user','word','role','sample_id']};b=a['spectrum'];v=stats(b);fac=factors(b,a['amplitude']);results={}
    for owner in args.owners:
        f=p.fold(owner);folder=out/('owner'+str(owner));folder.mkdir(exist_ok=True);fit=np.isin(d['user'],f['base'])&(d['role']=='fit');mu=b[fit].mean((0,2,3),keepdims=True);sd=np.maximum(b[fit].std((0,2,3),keepdims=True),1e-4);x=((b-mu)/sd).astype('float32');pooled=stats(x);sc=StandardScaler().fit(pooled[fit]);sv=sc.transform(pooled).astype('float32');vectors={'original_statistics':p.norm(sv)};infos={}
        pc=PCA(n_components=192,svd_solver='randomized',random_state=42).fit(sv[fit]);vectors['statistics_pca192']=p.norm(pc.transform(sv));
        for name,xx in [('stats_mlp',sv),('resnet18',x)]:vectors[name],infos[name]=train_model(name,xx,d,f,args.steps,folder)
        for target in ([owner,9] if owner==1 else [owner]):
            ff=p.fold(target);te=subset_frames(d)&(d['user']==target);dev=np.isin(d['user'],ff['base'])&(d['role']=='develop');probe={}
            for method,z in vectors.items():
                probe[method]={}
                for key,truth in fac.items():
                    # Fixed linear readout learned only from Base fit, not Cal or final targets.
                    scaler=StandardScaler().fit(truth[fit]);tt=scaler.transform(truth);model=Ridge(alpha=10).fit(z[fit],tt[fit]);pred=model.predict(z[te]);probe[method][key]=float(r2_score(tt[te],pred,multioutput='variance_weighted'))
            result=dict(fold=ff,methods={name:p.evaluate(z,d,ff) for name,z in vectors.items()},probe_target_query_r2=probe,training=infos);(out/('target'+str(target)+'_neural_private.json')).write_text(json.dumps(result,indent=2));results[str(target)]=result
        (out/'neural_status_public.json').write_text(json.dumps(dict(state='partial',owners_completed=owner,requested_owners=args.owners)));print('OWNER_COMPLETE',owner,flush=True)
    for path in out.glob('target*_neural_private.json'):
        target=path.name.split('_')[0][6:];results[target]=json.loads(path.read_text())
    (out/'neural_summary_public.json').write_text(json.dumps(dict(state='complete' if len(results)==9 else 'partial',completed_targets=len(results),results=macro(results),protocol='same 3x40x28STFT frontend;4Base2Cal1target2unseen;12shot cosine;Basefit-only scales;BaseDev-only checkpoints;CE+.1SupCon200stepssamebatch;raw720/PCA192/MLP192/ResNet192;noAudio'),indent=2))
def main(args):
    torch.set_num_threads(args.threads);out=Path(args.out);out.mkdir(exist_ok=True,parents=True);a={k:v for k,v in np.load(args.data).items()};assert a['spectrum'].shape==(2695,3,40,28);assert np.isfinite(a['spectrum']).all()
    if args.phase in ['ablation','all']:ablations(a,out)
    if args.phase in ['neural','all']:neural(a,out,args)
if __name__=='__main__':
    p1=argparse.ArgumentParser();p1.add_argument('--data',required=True);p1.add_argument('--out',required=True);p1.add_argument('--phase',choices=['ablation','neural','all'],default='all');p1.add_argument('--owners',nargs='+',type=int,default=list(range(1,9)));p1.add_argument('--steps',type=int,default=200);p1.add_argument('--threads',type=int,default=1);main(p1.parse_args())
