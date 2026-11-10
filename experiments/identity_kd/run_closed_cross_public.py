"""Shared all-known identity learning, personal CycleGAN, closed cross-sentence RBF.

ABCS43/VibraVox59; same K12/Cal8/query24 and matched no-Audio control.
Also adds a matched identity-only control to the completed Mei9 cross-word run.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import argparse,json,copy,sys
from pathlib import Path
import numpy as np
import torch,joblib
from torch import nn
from torch.nn import functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from run_rkd01_vibravox import norm,supcon,rkd,metrics,thresholds
from abcs43_open_personal_cycle import train as train_cycle,personalize,Generator,dump

BRANCHES=['input256','cycle256','cycle_statistics']
class Encoder(nn.Module):
    def __init__(self,n):
        super().__init__();self.proj=nn.Linear(n,192);self.res=nn.Sequential(nn.Linear(192,64),nn.GELU(),nn.Linear(64,192));nn.init.zeros_(self.res[-1].weight);nn.init.zeros_(self.res[-1].bias)
    def forward(self,x):z=self.proj(x);return F.normalize(z+self.res(z),dim=1)
def encode(net,x):
    net.eval()
    with torch.no_grad():return np.concatenate([net(torch.tensor(x[j:j+256])).numpy() for j in range(0,len(x),256)])
def initial_encoder(d,ix,x,out):
    path=out/'initial_student_private.pt';users=sorted(np.unique(d['user']).tolist());labels={u:j for j,u in enumerate(users)};torch.manual_seed(42);net=Encoder(x.shape[1])
    if path.exists():net.load_state_dict(torch.load(path,map_location='cpu',weights_only=False));return net
    head=nn.Linear(192,len(users),bias=False);opt=torch.optim.AdamW(list(net.parameters())+list(head.parameters()),lr=1e-4,weight_decay=1e-4);rng=np.random.RandomState(42);lookup={u:ix['fit'][d['user'][ix['fit']]==u] for u in users}
    for step in range(1,701):
        chosen=rng.choice(users,min(8,len(users)),replace=False);j=np.concatenate([rng.choice(lookup[u],4,replace=False) for u in chosen]);y=torch.tensor([labels[int(u)] for u in d['user'][j]]);z=net(torch.tensor(x[j]));loss=F.cross_entropy(16*F.linear(z,F.normalize(head.weight,dim=1)),y)+.1*supcon(z,y);opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(list(net.parameters())+list(head.parameters()),1);opt.step()
    torch.save(net.state_dict(),path);dump(out/'initial_training_private.json',dict(users=users,fit_indices=ix['fit'].tolist(),steps=700,batch_users=min(8,len(users)),batch_peruser=4));print('C3_INITIAL_ENCODER',len(users),flush=True);return net
def audio_targets(data,d,assets,out):
    path=out/'teacher_targets_private.npz'
    if path.exists():q=dict(np.load(path));return q['indices'],q['targets']
    sys.path.insert(0,'/root');from run_ecapa_pilot import teacher_load
    for n in ['model','deps']:
        p=out/n
        if not p.exists():p.symlink_to(Path(assets)/n,target_is_directory=True)
    model=teacher_load(out);emb=model.mods.embedding_model;official=torch.load(out/'model'/'embedding_model.ckpt',map_location='cpu',weights_only=False);assert set(official)==set(emb.state_dict()) and all(torch.equal(v.cpu(),official[k]) for k,v in emb.state_dict().items());del official;device=next(emb.parameters()).device;ti=data['indices'];users=sorted(np.unique(d['user']).tolist());label={u:j for j,u in enumerate(users)};yy=torch.tensor([label[int(v)] for v in d['user'][ti]],device=device);ff=[];lens=torch.tensor(data['lengths'],device=device)
    with torch.no_grad():
        for s in range(0,len(ti),16):w=torch.tensor(data['wave'][s:s+16],device=device);ff.append(model.mods.mean_var_norm(model.mods.compute_features(w),lens[s:s+16]))
    feat=torch.cat(ff);emb.eval()
    for p in emb.parameters():p.requires_grad_(True)
    def forward(j):return F.normalize(emb(feat[j],lens[j]).squeeze(1),dim=1)
    with torch.no_grad():frozen=torch.cat([forward(torch.arange(s,min(s+16,len(ti)),device=device)) for s in range(0,len(ti),16)]).detach()
    h=nn.Linear(192,len(users),bias=False).to(device)
    with torch.no_grad():h.weight.copy_(F.normalize(torch.stack([frozen[yy==j].mean(0) for j in range(len(users))]),dim=1))
    opt=torch.optim.AdamW([dict(params=emb.parameters(),lr=1e-6),dict(params=h.parameters(),lr=1e-4)],weight_decay=1e-4);rng=np.random.RandomState(4242);watch=next(p for p in emb.parameters() if p.ndim>1);before=watch.detach().clone();lookup={u:np.flatnonzero(d['user'][ti]==u) for u in users}
    for step in range(100):
        uu=rng.choice(users,min(8,len(users)),replace=False);j=np.concatenate([rng.choice(lookup[u],4,replace=False) for u in uu]);jt=torch.tensor(j,device=device);z=forward(jt);y=yy[jt];loss=F.cross_entropy(16*F.linear(z,F.normalize(h.weight,dim=1)),y)+.1*supcon(z.cpu(),y.cpu()).to(device)+.1*(1-F.cosine_similarity(z,frozen[jt],dim=1)).mean();opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(list(emb.parameters())+list(h.parameters()),1);opt.step()
    delta=float((watch-before).abs().max());assert delta>0
    with torch.no_grad():targets=torch.cat([forward(torch.arange(s,min(s+16,len(ti)),device=device)).cpu() for s in range(0,len(ti),16)]).numpy()
    np.savez_compressed(path,indices=ti,targets=targets);torch.save({k:v.cpu() for k,v in emb.state_dict().items()},out/'audio_teacher_private.pt');dump(out/'teacher_audit_public.json',dict(users=len(users),records=len(ti),steps=100,body_lr=1e-6,head_lr=1e-4,backbone_maxdelta=delta,original_loaded_state_verified=True,query_audio_model_forwards=0));print('C3_AUDIO_TEACHER',len(users),flush=True);return ti,targets
def adapt(initial,d,x,ti,targets,mode,out,legacy=False):
    path=out/(mode+'_student_private.pt');net=copy.deepcopy(initial)
    if path.exists():net.load_state_dict(torch.load(path,map_location='cpu',weights_only=False));return net
    users=sorted(np.unique(d['user']).tolist());labels={u:j for j,u in enumerate(users)};yall=np.array([labels[int(u)] for u in d['user'][ti]]);start=encode(net,x[ti]);head=nn.Linear(192,len(users),bias=False)
    with torch.no_grad():head.weight.copy_(torch.tensor(norm(np.stack([start[yall==j].mean(0) for j in range(len(users))]))))
    opt=torch.optim.AdamW(list(net.parameters())+list(head.parameters()),lr=1e-4,weight_decay=1e-4);rng=np.random.RandomState(42000);params=list(net.parameters());ei=ek=None;trace=[];lookup={u:np.flatnonzero(d['user'][ti]==u) for u in users}
    for step in range(1,101):
        uu=users if legacy else rng.choice(users,min(8,len(users)),replace=False);j=np.concatenate([rng.choice(lookup[u],4,replace=False) for u in uu]);y=torch.tensor(yall[j]);z=net(torch.tensor(x[ti[j]]));identity=F.cross_entropy(16*F.linear(z,F.normalize(head.weight,dim=1)),y)+.1*supcon(z,y)+.1*(1-F.cosine_similarity(z,torch.tensor(start[j]),dim=1)).mean();lam=0.;kd=z.new_zeros(())
        if mode=='audio':
            kd=rkd(z,torch.tensor(targets[j]),y);gi=torch.autograd.grad(identity,params,retain_graph=True);gk=torch.autograd.grad(kd,params,retain_graph=True);ni=float(torch.sqrt(sum(g.square().sum() for g in gi)));nk=float(torch.sqrt(sum(g.square().sum() for g in gk)));ei=ni if ei is None else .9*ei+.1*ni;ek=nk if ek is None else .9*ek+.1*nk;lam=min(100.,.3*ei/max(ek,1e-8))
        loss=identity+lam*kd;opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(list(net.parameters())+list(head.parameters()),1);opt.step()
        if step in [1,25,50,100]:trace.append(dict(step=step,lambda_rkd=lam,identity=float(identity.detach()),rkd=float(kd.detach())))
    torch.save(net.state_dict(),path);dump(out/(mode+'_training_private.json'),dict(users=users,training_indices=ti.tolist(),steps=100,sampler='all9x4' if legacy else 'random8identitiesx4',trace=trace,query_audio_model_forwards=0));print('C3_STUDENT',mode,len(users),flush=True);return net

@torch.no_grad()
def fast_features(g,f,a,ii):
    out={k:[] for k in BRANCHES};g.eval();f.eval()
    for s in range(0,len(ii),128):
        x=torch.tensor(a[ii[s:s+128]],device='cuda');e=g.encoder(x);m=g.decoder(e);c=f(m);ec=g.encoder(c);avg=c[:,0].mean(-1);shape=avg-avg.mean(1,keepdim=True);sd=c[:,0].std(-1,unbiased=False);seg=torch.cat([c[:,0,:,j*8:(j+1)*8].mean(-1)-avg for j in range(4)],1);pool=lambda z:torch.cat([z.mean((-2,-1)),z.std((-2,-1),unbiased=False)],1)
        for k,v in zip(BRANCHES,[pool(e),pool(ec),torch.cat([shape,sd,seg],1)]):out[k].append(v.cpu().numpy())
    return {k:np.concatenate(v) for k,v in out.items()}
def view(base,z,c):return base if c['step']==0 else np.concatenate([base,c['alpha']*z[c['step']][c['branch']]],1)/np.sqrt(1+c['alpha']**2)
def head(x,d,ix,u):return SVC(C=10,gamma='scale',class_weight='balanced').fit(x[ix['register']],d['user'][ix['register']]==u)
def summarize(rows,out,dataset):
    users=sorted(rows);methods=list(rows[users[0]]);public={};per=[];tt={}
    for method in methods:
        pp={point:[] for point in ['Cal_EER','Cal_FAR5']};tt[method]={}
        for j,u in enumerate(users,1):
            cal=rows[u][method]['calibration'];ts=thresholds(np.array(cal['scores']),np.array(cal['labels']));tt[method][str(u)]=ts;q=rows[u][method]['query']
            for point,tau in ts.items():v=metrics(np.array(q['scores']),np.array(q['labels']),tau);per.append(dict(user=f'{dataset}-C3-U{j:02}',method=method,point=point,**v));pp[point].append(v)
        public[method]={point:{**{k:float(np.mean([v[k] for v in vv])) for k in ['tar','far','frr','hter','auc','eer']},'worst_target_far':max(v['far'] for v in vv),'lowest_target_tar':min(v['tar'] for v in vv)} for point,vv in pp.items()}
    paired={};rng=np.random.RandomState(20261110)
    for method in methods:
        if method=='audio_baseline':continue
        delta=np.array([next(v['hter'] for v in per if v['user']==f'{dataset}-C3-U{j:02}' and v['method']==method and v['point']=='Cal_EER')-next(v['hter'] for v in per if v['user']==f'{dataset}-C3-U{j:02}' and v['method']=='audio_baseline' and v['point']=='Cal_EER') for j in range(1,len(users)+1)]);boots=np.array([delta[rng.choice(len(users),len(users),replace=True)].mean() for _ in range(5000)]);paired[method]=dict(hter_delta=float(delta.mean()),identity_bootstrap95=np.quantile(boots,[.025,.975]).tolist(),improved_users=int((delta<0).sum()))
    dump(out/'thresholds_private.json',tt);dump(out/'per_user_public.json',per);dump(out/'results_public.json',dict(state='complete',dataset=dataset,users=len(users),known_attackers_each=len(users)-1,seed=42,K=12,calibration_each=8,query_each=24,backend='balanced RBF C10 gammaScale;all-known identities;per-target calibration',results=public,paired_vs_audio_baseline=paired,scope='closed cross-content;known identities,unseen query recordings and exact sentences;Audio only fit/register;own12 personalCycleGAN',limits=['single seed/explored corpus','cross-sentence does not mean disjoint vocabulary','public corpora different prompts across users','ABCS channel mapping provisional','Mei word order inferred','no random-network GAN control in closed three-dataset comparison']))

def authenticate(d,ix,data,refs,g,f,da,db,out,dataset):
    users=sorted(np.unique(d['user']).tolist());base=refs['audio_baseline'];scpath=out/'feature_scalers_private.joblib'
    if scpath.exists():sc=joblib.load(scpath)
    else:raw=fast_features(g,f,data['a'],ix['fit']);sc={k:StandardScaler().fit(v) for k,v in raw.items()};joblib.dump(sc,scpath)
    pre=np.unique(np.r_[ix['register'],ix['calA'],ix['calB']]);allix=np.unique(np.r_[pre,ix['query']]);cache={}
    def owner_raw(u,step,ii):
        folder=out/f'owner{u}';path=folder/f'features{step}_{len(ii)}_private.npz'
        if path.exists():q=dict(np.load(path));assert np.array_equal(q.pop('indices'),ii);return q
        ck=torch.load(folder/f'gan{step}_private.pt',map_location='cpu',weights_only=False);gg=Generator().cuda();ff=Generator().cuda();gg.load_state_dict(ck['G']);ff.load_state_dict(ck['F']);prepath=folder/f'features{step}_{len(pre)}_private.npz'
        if len(ii)>len(pre) and prepath.exists():
            old=dict(np.load(prepath));oldix=old.pop('indices');missing=np.setdiff1d(ii,oldix);new=fast_features(gg,ff,data['a'],missing);q={}
            for k in BRANCHES:v=np.empty((len(ii),old[k].shape[1]),dtype='float32');v[np.searchsorted(ii,oldix)]=old[k];v[np.searchsorted(ii,missing)]=new[k];q[k]=v
        else:q=fast_features(gg,ff,data['a'],ii)
        np.savez_compressed(path,indices=ii,**q);return q
    def owner_z(u,ii):
        z={}
        for step in [25,100]:
            q=owner_raw(u,step,ii);z[step]={}
            for k,v in q.items():a=np.zeros((len(base),v.shape[1]),dtype='float32');a[ii]=norm(sc[k].transform(v));z[step][k]=a
        return z
    for u in users:personalize(u,d,data,g,f,da,db,out);cache[u]=owner_z(u,pre);print('C3_PERSONAL_PRE',dataset,u,flush=True)
    lock=out/'selection_public.json'
    if not lock.exists():
        choices=[dict(step=0,branch='none',alpha=0.)]+[dict(step=s,branch=k,alpha=w) for s in [25,100] for k in BRANCHES for w in [.25,.5,1.]]
        for c in choices:
            pp=[]
            for u in users:
                x=view(base,cache[u],c);model=head(x,d,ix,u);a=ix['calA'];b=ix['calB'];cs=model.decision_function(x[a]);s=model.decision_function(x[b]);pp.append(metrics(s,d['user'][b]==u,thresholds(cs,d['user'][a]==u)['Cal_EER']))
            c.update(hter=float(np.mean([v['hter'] for v in pp])),auc=float(np.mean([v['auc'] for v in pp])))
        best=min(choices,key=lambda c:(c['hter'],-c['auc'],c['alpha'],c['step'],c['branch']));dump(lock,dict(selected=best,choices=choices,rule='reg12/person fit eachbinaryhead;CalA4/person thresholds CalB4/person globalfeatureselection,minHTER/AUC/smallerweight/fewersteps;zeroallowed;before queryG/Ffeatures'));print('C3_LOCKED',dataset,best,flush=True)
    choice=json.loads(lock.read_text())['selected'];rows={};cal=np.r_[ix['calA'],ix['calB']]
    for u in users:
        del cache[u];z=owner_z(u,allix);variants=dict(refs)
        for step in [25,100]:
            for k in BRANCHES:variants[f'personal{step}_{k}']=view(base,z,dict(step=step,branch=k,alpha=.5))
        variants['selected_cycle']=view(base,z,choice);variants['cycle_only100']=z[100]['cycle256'];models={};rows[u]={}
        for name,x in variants.items():
            model=head(x,d,ix,u);models[name]=model;rows[u][name]={role:dict(scores=model.decision_function(x[ii]).tolist(),labels=(d['user'][ii]==u).tolist()) for role,ii in [('calibration',cal),('query',ix['query'])]}
        joblib.dump(models,out/f'owner{u}'/'rbf_heads_private.joblib');dump(out/f'owner{u}'/'scores_private.json',rows[u]);print('C3_AUTH',dataset,u,flush=True)
    dump(out/'scores_private.json',{str(k):v for k,v in rows.items()});summarize(rows,out,dataset);print('C3_COMPLETE',dataset,flush=True)

def legacy_mei(a,out):
    prior=Path(a.legacy_mei);d=dict(np.load(a.frontend));data=dict(np.load(prior/'prepared_private.npz'));oldix=json.loads((prior/'split_private.json').read_text());ix={k:np.array(v) for k,v in oldix.items()};ix['query']=ix['crossword'];initial=Encoder(data['x'].shape[1]);initial.load_state_dict(torch.load(prior/'initial_student_private.pt',map_location='cpu',weights_only=False));t=dict(np.load(prior/'teacher_targets_private.npz'));net=adapt(initial,d,data['x'],t['indices'],None,'no_audio',out,True);emb=encode(net,data['x']);z=np.concatenate([data['shape'],emb],1)/np.sqrt(2);rows={};cal=np.r_[ix['calA'],ix['calB']];old=json.loads((prior/'scores_private.json').read_text());mapping={'raw120':'raw_shape','statistics720':'raw_statistics','student192':'audio_encoder','baseline312':'audio_baseline','selected_cycle':'selected_cycle','cycle_only100':'cycle_only100'}
    for step in [25,100]:
        for oldbranch,newbranch in [('input256','input256'),('cycle256','cycle256'),('cycle_stats720','cycle_statistics')]:mapping[f'personal{step}_{oldbranch}']=f'personal{step}_{newbranch}'
    for u in range(1,10):
        folder=out/f'owner{u}';folder.mkdir(exist_ok=True);rows[u]={new:{'calibration':old[str(u)][name]['calibration'],'query':old[str(u)][name]['crossword']} for name,new in mapping.items()};models={}
        for name,x in [('no_audio_encoder',emb),('no_audio_baseline',z)]:
            model=head(x,d,ix,u);models[name]=model;rows[u][name]={role:dict(scores=model.decision_function(x[ii]).tolist(),labels=(d['user'][ii]==u).tolist()) for role,ii in [('calibration',cal),('query',ix['query'])]}
        joblib.dump(models,folder/'new_control_heads_private.joblib');dump(folder/'scores_private.json',rows[u])
    dump(out/'scores_private.json',{str(k):v for k,v in rows.items()});summarize(rows,out,'Mei9');r=json.loads((out/'results_public.json').read_text());r['scope']='Mei9 known identities,nominal cross-word split inferred from capture order,not transcript-verified';dump(out/'results_public.json',r);dump(out/'selection_public.json',json.loads((prior/'selection_public.json').read_text()));dump(out/'preparation_public.json',dict(dataset='Mei9',users=9,source=str(prior.name),crossword_only=True,reused_existing_complete_cycle_experiment=True,new_training='matched100step identity-only control from same archived700step initialization;all9x4 same batches',word_labels='nominal inferred,not transcript-verified',query_each=24,known_attackers=8));print('C3_COMPLETE Mei9',flush=True)

def main(a):
    torch.set_num_threads(2);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    if a.legacy_mei:return legacy_mei(a,out)
    torch.backends.cudnn.benchmark=True;d=dict(np.load(out/'inputs_private.npz'));data=dict(np.load(out/'spectra_private.npz'));ix={k:np.array(v) for k,v in json.loads((out/'split_private.json').read_text()).items()};users=sorted(np.unique(d['user']).tolist());assert all(len(ix['register'][d['user'][ix['register']]==u])==12 and len(ix['query'][d['user'][ix['query']]==u])==24 for u in users);x=StandardScaler().fit(d['x'][ix['fit']]).transform(d['x']).astype('float32');shape=norm(x[:,:40]);stats=norm(StandardScaler().fit(d['statistics'][ix['fit']]).transform(d['statistics']));initial=initial_encoder(d,ix,x,out);ta=dict(np.load(out/'teacher_audio_private.npz'));ti,targets=audio_targets(ta,d,a.assets,out);v0=adapt(initial,d,x,ti,None,'no_audio',out);v1=adapt(initial,d,x,ti,targets,'audio',out);z0=encode(v0,x);z1=encode(v1,x);refs=dict(raw_shape=shape,raw_statistics=stats,no_audio_encoder=z0,no_audio_baseline=np.concatenate([shape,z0],1)/np.sqrt(2),audio_encoder=z1,audio_baseline=np.concatenate([shape,z1],1)/np.sqrt(2));data.update(fit=ix['fit'],dev=ix['dev']);g,f,da,db=train_cycle(data,out);meta=json.loads((out/'gan_complete_public.json').read_text());meta.pop('base_users',None);meta.update(known_training_users=len(users),closed_all_users=True);dump(out/'gan_complete_public.json',meta);authenticate(d,ix,data,refs,g,f,da,db,out,a.dataset)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--out',required=True);p.add_argument('--assets');p.add_argument('--legacy-mei');p.add_argument('--frontend');main(p.parse_args())
