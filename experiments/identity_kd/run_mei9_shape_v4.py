"""Mei9 transfer of shape-preserving, registration-Audio-adapted V4 only."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import argparse,copy,json,sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.preprocessing import StandardScaler
from run_rkd01_vibravox import norm,supcon,rkd,thresholds,metrics
from run_registration_rkd02 import dump

class Student(nn.Module):
    def __init__(self):
        super().__init__();self.proj=nn.Linear(738,192);self.res=nn.Sequential(nn.Linear(192,64),nn.GELU(),nn.Linear(64,192));nn.init.zeros_(self.res[-1].weight);nn.init.zeros_(self.res[-1].bias)
    def forward(self,x):
        z=self.proj(x);return F.normalize(z+self.res(z),dim=1)
def encode(net,x):
    net.eval()
    with torch.no_grad():return np.concatenate([net(torch.tensor(x[s:s+256])).numpy() for s in range(0,len(x),256)])
def fold(u):
    order=[1+(u-1+i)%9 for i in range(9)];return dict(target=u,attackers=order[1:3],calibration=order[3:5],base=order[5:])
def rows(d,u,w):return np.array(sorted(np.flatnonzero((d['user']==u)&(d['word']==w)),key=lambda i:str(d['sample_id'][i])))
def reg(d,u):return np.concatenate([rows(d,u,w)[:3] for w in range(20,24)])
def queries(d,u,condition):
    return np.concatenate([rows(d,u,w)[3:9] for w in range(20,24)]) if condition=='sameword' else np.concatenate([rows(d,u,w)[:4] for w in range(24,30)])
def replay(d,u):return np.array([rows(d,u,w)[0] for w in [0,5,10,15]])
def features(d):
    b=d['spectrum'];m=b.mean(-1);level=m.mean(-1);seg=np.stack([b[:,:,:,q].mean(-1) for q in np.array_split(np.arange(b.shape[-1]),4)],2);x=np.concatenate([(m-level[:,:,None]).reshape(len(b),120),b.std(-1).reshape(len(b),120),(seg-m[:,:,None,:]).reshape(len(b),480),level,d['amplitude']],1).astype('float32');v=np.concatenate([m[:,:,None],b.std(-1)[:,:,None],seg],2).reshape(len(b),720).astype('float32');assert x.shape[1]==738;return x,v
def initial_student(d,x,base,path):
    net=Student()
    if path.exists():net.load_state_dict(torch.load(path,map_location='cpu',weights_only=False));return net
    torch.manual_seed(42);net=Student();head=nn.Linear(192,4,bias=False);opt=torch.optim.AdamW(list(net.parameters())+list(head.parameters()),lr=1e-4,weight_decay=1e-4);rng=np.random.RandomState(42);words=sorted(set(d['word'][d['role']=='fit']));lookup={(u,w):rows(d,u,w) for u in base for w in words};labels={u:j for j,u in enumerate(base)}
    for step in range(1,701):
        ww=rng.choice(words,4,replace=False);ix=np.concatenate([rng.choice(lookup[u,w],2,replace=False) for u in base for w in ww]);y=torch.tensor([labels[int(u)] for u in d['user'][ix]]);z=net(torch.tensor(x[ix]));loss=F.cross_entropy(16*F.linear(z,F.normalize(head.weight,dim=1)),y)+.1*supcon(z,y);opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(list(net.parameters())+list(head.parameters()),1);opt.step()
    torch.save(net.state_dict(),path);print('BASE_INITIAL_COMPLETE',base,flush=True);return net
def teacher_personal(d,u,base,folder,model,original,cache,slots):
    path=folder/'teacher_targets_private.npz'
    if path.exists():q=dict(np.load(path));return q['indices'],q['adapted']
    torch.manual_seed(4242+u);rng=np.random.RandomState(4242+u);ix=np.r_[np.concatenate([replay(d,v) for v in base]),reg(d,u)];assert len(ix)==28;assert all((int(d['user'][i]) in base and d['role'][i]=='fit') or (d['user'][i]==u and i in reg(d,u)) for i in ix);emb=model.mods.embedding_model;emb.load_state_dict(original);emb.eval()
    for p in emb.parameters():p.requires_grad_(True)
    dev=next(emb.parameters()).device;feat=cache[torch.tensor([slots[int(i)] for i in ix])].to(dev);labels_map={v:j for j,v in enumerate(base+[u])};yall=torch.tensor([labels_map[int(v)] for v in d['user'][ix]],device=dev)
    def forward(j):return F.normalize(emb(feat[j],torch.ones(len(j),device=dev)).squeeze(1),dim=1)
    with torch.no_grad():frozen=forward(torch.arange(len(ix),device=dev)).detach();proto=F.normalize(torch.stack([frozen[yall==j].mean(0) for j in range(5)]),dim=1)
    head=nn.Linear(192,5,bias=False).to(dev)
    with torch.no_grad():head.weight.copy_(proto)
    opt=torch.optim.AdamW([dict(params=emb.parameters(),lr=1e-6),dict(params=head.parameters(),lr=1e-4)],weight_decay=1e-4);watch=next(p for p in emb.parameters() if p.ndim>1);before=watch.detach().clone()
    for step in range(30):
        batch=np.r_[rng.choice(np.flatnonzero(d['user'][ix]==u),4,replace=False),np.concatenate([rng.choice(np.flatnonzero(d['user'][ix]==v),2,replace=False) for v in base])];jj=torch.tensor(batch,device=dev);z=forward(jj);y=yall[jj];loss=F.cross_entropy(16*F.linear(z,F.normalize(head.weight,dim=1)),y)+.1*supcon(z.cpu(),y.cpu()).to(dev)+.1*(1-F.cosine_similarity(z[4:],frozen[jj[4:]],dim=1)).mean();opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(list(emb.parameters())+list(head.parameters()),1);opt.step()
    delta=float((watch-before).abs().max());assert delta>0
    for p in emb.parameters():p.requires_grad_(False)
    with torch.no_grad():t=forward(torch.arange(len(ix),device=dev)).cpu().numpy()
    np.savez_compressed(path,indices=ix,adapted=t,frozen=frozen.cpu().numpy());torch.save(emb.state_dict(),folder/'ecapa_private.pt');dump(folder/'teacher_audit_private.json',dict(backbone_update=delta,query_audio_forwards=0,base=base,owner=u,registration=12,base_replay=16,steps=30));return ix,t
def personal_student(d,x,initial,u,base,ix,t,folder):
    path=folder/'student_private.pt';net=copy.deepcopy(initial)
    if path.exists():net.load_state_dict(torch.load(path,map_location='cpu',weights_only=False));return net
    torch.manual_seed(42);rng=np.random.RandomState(42000+u);xx=x[ix];start=encode(initial,xx);labels={v:j for j,v in enumerate(base+[u])};yy=np.array([labels[int(v)] for v in d['user'][ix]]);lookup={v:np.flatnonzero(d['user'][ix]==v) for v in base+[u]};head=nn.Linear(192,5,bias=False)
    with torch.no_grad():head.weight.copy_(torch.tensor(norm(np.stack([start[yy==j].mean(0) for j in range(5)]))))
    opt=torch.optim.AdamW(list(net.parameters())+list(head.parameters()),lr=1e-4,weight_decay=1e-4);params=list(net.parameters());ei=None;ek=None;trace=[]
    for step in range(1,101):
        batch=np.r_[rng.choice(lookup[u],8,replace=False),np.concatenate([lookup[v] for v in base])];y=torch.tensor(yy[batch]);z=net(torch.tensor(xx[batch]));identity=F.cross_entropy(16*F.linear(z,F.normalize(head.weight,dim=1)),y)+.1*supcon(z,y)+.1*(1-F.cosine_similarity(z[8:],torch.tensor(start[batch[8:]]),dim=1)).mean();kd=rkd(z,torch.tensor(t[batch]),y);gi=torch.autograd.grad(identity,params,retain_graph=True);gk=torch.autograd.grad(kd,params,retain_graph=True);ni=float(torch.sqrt(sum(g.square().sum() for g in gi)));nk=float(torch.sqrt(sum(g.square().sum() for g in gk)));ei=ni if ei is None else .9*ei+.1*ni;ek=nk if ek is None else .9*ek+.1*nk;lam=min(100.,.3*ei/max(ek,1e-8));loss=identity+lam*kd;opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(list(net.parameters())+list(head.parameters()),1);opt.step()
        if step in [1,25,50,100]:trace.append(dict(step=step,lambda_rkd=lam,gradient_ratio=lam*nk/max(ni,1e-8)))
    torch.save(net.state_dict(),path);dump(folder/'student_audit_private.json',dict(steps=100,rho=.3,trace=trace,training_ids=d['sample_id'][ix].tolist()));return net
def score(z,d,u,attacks,condition):
    en=reg(d,u);q=np.concatenate([queries(d,v,condition) for v in [u]+attacks]);assert len(en)==12 and not set(en)&set(q);s=z[q]@norm(z[en].mean(0));return dict(scores=s.tolist(),labels=(d['user'][q]==u).tolist())
def evaluate_owner(z,d,u,attacks):return {c:score(z,d,u,attacks,c) for c in ['sameword','crossword']}
def evaluate_fold(calrows,targetrows):
    cs=np.concatenate([q[c]['scores'] for q in calrows for c in ['sameword','crossword']]);cy=np.concatenate([q[c]['labels'] for q in calrows for c in ['sameword','crossword']]);taus=thresholds(cs,cy);result={}
    for condition in ['sameword','crossword']:
        s=np.array(targetrows[condition]['scores']);y=np.array(targetrows[condition]['labels']);result[condition]={point:metrics(s,y,tau) for point,tau in taus.items()}
    assert float((cs[~cy]>=taus['Cal_FAR5']).mean())<=.05;return result,taus
def main(a):
    torch.set_num_threads(1);out=Path(a.out);out.mkdir(parents=True,exist_ok=True);d=dict(np.load(a.frontend));assert d['spectrum'].shape==(2695,3,40,28);raw=np.load(Path(a.raw)/'raw9_full_features_private.npz');mapping={s:i for i,s in enumerate(raw['sample_id'])};ri=np.array([mapping[s] for s in d['sample_id']]);assert np.array_equal(raw['user'][ri],d['user']) and np.array_equal(raw['word'][ri],d['word']) and np.array_equal(raw['role'][ri],d['role']);assert len(set(d['sample_id']))==2695;xraw,vraw=features(d);manifest=json.loads((Path(a.raw)/'raw9_manifest_private.json').read_text());byid={r['sample_id']:r for r in manifest['records']};assert all(byid[s]['pair_mode']=='exact_timestamp' for s in d['sample_id'])
    for u in range(1,10):
        assert len(reg(d,u))==12 and len(queries(d,u,'sameword'))==24 and len(queries(d,u,'crossword'))==24
    selected=np.unique(np.concatenate([np.r_[replay(d,u),reg(d,u)] for u in range(1,10)]));assert len(selected)==144;assert not np.isin(d['role'][selected],['develop','final']).any();sys.path.insert(0,'/root');from run_ecapa_pilot import teacher_load
    for name in ['model','deps']:
        p=out/name
        if not p.exists():p.symlink_to(Path(a.raw)/name,target_is_directory=True)
    model=teacher_load(out);original={k:v.detach().cpu().clone() for k,v in model.mods.embedding_model.state_dict().items()};official=torch.load(out/'model'/'embedding_model.ckpt',map_location='cpu',weights_only=False);assert set(original)==set(official) and all(torch.equal(original[k],official[k]) for k in original);del official;dev=next(model.parameters()).device;cachepath=out/'audio_frontend_private.pt'
    if cachepath.exists():cache=torch.load(cachepath,map_location='cpu',weights_only=False)
    else:
        # Existing archive stores all waveforms; only these preselected fit/reg
        # rows are converted to float32 or forwarded, never any query Audio.
        wave=raw['wave'][ri[selected]].astype('float32');wave-=wave.mean(1,keepdims=True);ff=[]
        with torch.no_grad():
            for s in range(0,len(wave),16):
                w=torch.tensor(wave[s:s+16],device=dev);ff.append(model.mods.mean_var_norm(model.mods.compute_features(w),torch.ones(len(w),device=dev)).cpu())
        cache=torch.cat(ff);torch.save(cache,cachepath)
    slots={int(i):j for j,i in enumerate(selected)};all_results={}
    for target in range(1,10):
        f=fold(target);assert len(set(f['base']+f['calibration']+f['attackers']+[target]))==9;folder=out/f'target{target}';folder.mkdir(exist_ok=True);completed=folder/'results_private.json'
        if completed.exists():all_results[str(target)]=json.loads(completed.read_text());continue
        fit=np.isin(d['user'],f['base'])&(d['role']=='fit');x=StandardScaler().fit(xraw[fit]).transform(xraw).astype('float32');shape=norm(x[:,:120]);stats=norm(StandardScaler().fit(vraw[fit]).transform(vraw));initial=initial_student(d,x,f['base'],folder/'base_student_private.pt');vectors={}
        for owner in f['calibration']+[target]:
            p=folder/f'owner{owner}';p.mkdir(exist_ok=True);ix,t=teacher_personal(d,owner,f['base'],p,model,original,cache,slots);net=personal_student(d,x,initial,owner,f['base'],ix,t,p);z=encode(net,x);joint=np.concatenate([shape,z],1)/np.sqrt(2);assert joint.shape[1]==312 and np.allclose(joint[:,:120],shape/np.sqrt(2),atol=1e-7);vectors[owner]=dict(shape120=shape,statistics720=stats,student192=z,supplement312=joint)
        results={};rawscores={};thresholds_private={}
        for method in ['shape120','statistics720','student192','supplement312']:
            cal=[evaluate_owner(vectors[u][method],d,u,[v for v in f['calibration'] if v!=u]) for u in f['calibration']];results[method]={};rawscores[method]=dict(calibration=cal,test={})
            for scope,attacks in [('two_unseen',f['attackers']),('all_eight_mixed',[u for u in range(1,10) if u!=target])]:
                q=evaluate_owner(vectors[target][method],d,target,attacks);r,tau=evaluate_fold(cal,q);results[method][scope]=r;rawscores[method]['test'][scope]=q;thresholds_private[method]=tau
        row=dict(fold=f,results=results);dump(folder/'scores_private.json',rawscores);dump(folder/'thresholds_private.json',thresholds_private);dump(completed,row);all_results[str(target)]=row;print('TARGET_COMPLETE',target,flush=True)
    public={}
    for method in ['shape120','statistics720','student192','supplement312']:
        public[method]={scope:{cond:{point:{metric:float(np.mean([r['results'][method][scope][cond][point][metric] for r in all_results.values()])) for metric in ['tar','far','frr','hter','auc','eer']} for point in ['Cal_EER','Cal_FAR5']} for cond in ['sameword','crossword']} for scope in ['two_unseen','all_eight_mixed']}
    paired={};rng=np.random.RandomState(20261110)
    for cond in ['sameword','crossword']:
        delta=np.array([r['results']['supplement312']['two_unseen'][cond]['Cal_EER']['hter']-r['results']['shape120']['two_unseen'][cond]['Cal_EER']['hter'] for r in all_results.values()]);boots=np.array([delta[rng.choice(9,9,replace=True)].mean() for _ in range(5000)]);paired[cond]=dict(hter_delta=float(delta.mean()),identity_bootstrap95=np.quantile(boots,[.025,.975]).tolist(),improved_targets=int((delta<0).sum()))
    dump(out/'results_public.json',dict(state='complete',users=9,seed=42,K=12,alpha=1,rho=.3,results=public,paired=paired,protocol='9balanced rotating folds;4Base2Cal1target2unseen;12register24g48unknownattacks;matched nominal words across users;common Cal threshold pooledsame/cross;V4 only;shape120+student192',limits=['word order inferred,not independently transcript-verified','OpenVR vVelocityXYZ,not raw accelerometer','cached microphone waveforms float16,left-channel16kHz;onlyfit/register fed toAudio models','no redon/crossday/device controlled repeats','single seed;alpha/rho transferred fromVibraVox,not retuned onMeiTest','all_eight_mixed includes Base/Cal sources,not a pure unseen-identity result']));print('ALL_COMPLETE',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--raw',required=True);p.add_argument('--frontend',required=True);p.add_argument('--out',required=True);main(p.parse_args())
