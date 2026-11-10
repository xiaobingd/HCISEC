"""Base-only shared paired CycleGAN, frozen feature supplementation on VibraVox.

Not the old personal two-user GAN or historical full772 implementation.
"""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import sys,json,argparse,hashlib,copy,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from scipy import signal
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.preprocessing import StandardScaler
from run_rkd01_vibravox import Student,encode,norm,metrics,thresholds

BASE=list(range(35));CAL=list(range(35,47));TEST=list(range(47,59))
def dump(p,x):
    p=Path(p);q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2));q.replace(p)

class Residual(nn.Module):
    def __init__(self,c):
        super().__init__();self.net=nn.Sequential(nn.Conv2d(c,c,3,padding=1),nn.InstanceNorm2d(c),nn.ReLU(),nn.Conv2d(c,c,3,padding=1),nn.InstanceNorm2d(c))
    def forward(self,x):return x+self.net(x)

class Generator(nn.Module):
    def __init__(self):
        super().__init__();self.encoder=nn.Sequential(nn.ReflectionPad2d(3),nn.Conv2d(1,32,7),nn.InstanceNorm2d(32),nn.ReLU(),nn.Conv2d(32,64,3,2,1),nn.InstanceNorm2d(64),nn.ReLU(),nn.Conv2d(64,128,3,2,1),nn.InstanceNorm2d(128),nn.ReLU(),*[Residual(128) for _ in range(4)]);self.decoder=nn.Sequential(nn.Upsample(scale_factor=2,mode='nearest'),nn.Conv2d(128,64,3,padding=1),nn.InstanceNorm2d(64),nn.ReLU(),nn.Upsample(scale_factor=2,mode='nearest'),nn.Conv2d(64,32,3,padding=1),nn.InstanceNorm2d(32),nn.ReLU(),nn.ReflectionPad2d(3),nn.Conv2d(32,1,7),nn.Tanh())
    def forward(self,x):return self.decoder(self.encoder(x))

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Conv2d(1,32,4,2,1),nn.LeakyReLU(.2),nn.Conv2d(32,64,4,2,1),nn.InstanceNorm2d(64),nn.LeakyReLU(.2),nn.Conv2d(64,128,3,1,1),nn.InstanceNorm2d(128),nn.LeakyReLU(.2),nn.Conv2d(128,1,3,1,1))
    def forward(self,x):return self.net(x)

def prepare(source,paired,out):
    path=out/'spectra_private.npz'
    if path.exists():return dict(np.load(path))
    d=dict(np.load(source/'inputs_private.npz'));records=json.loads((source/'manifest_private.json').read_text())['records'];assert len(records)==len(d['x']);fit=np.flatnonzero((d['group']=='base')&(d['role']=='fit'));dev=np.flatnonzero((d['group']=='base')&(d['role']=='develop'));allowed=set(np.r_[fit,dev].tolist());seen=set();audios=0
    def spec(w):
        w=w.astype('float64');i=(len(w)-32000)//2;w=w[i:i+32000].copy();assert len(w)==32000;w-=w.mean();freq,_,z=signal.stft(w,16000,nperseg=4096,noverlap=3072,nfft=16384,boundary=None,padded=False);lp=np.log(abs(z)**2+1e-20);b=np.stack([np.interp(np.geomspace(50,7600,40),freq,lp[:,j]) for j in range(lp.shape[1])],1);return np.stack([np.interp(np.linspace(0,b.shape[1]-1,32),np.arange(b.shape[1]),row) for row in b]).astype('float32')
    def read(item):
        i,r=item
        with np.load(paired/'paired'/r['file']) as z:
            acc=z['acc'].astype('float32');assert hashlib.sha256(acc.tobytes()).hexdigest()==r['acc_sha256'];a=spec(acc);b=np.zeros_like(a)
            if i in allowed:
                audio=z['audio'].astype('float32');assert hashlib.sha256(audio.tobytes()).hexdigest()==r['audio_sha256'];b=spec(audio)
        return i,a,b,r['acc_sha256']
    aa=np.empty((len(records),40,32),dtype='float32');bb=np.zeros_like(aa)
    with ThreadPoolExecutor(max_workers=8) as workers:
        for i,a,b,digest in workers.map(read,enumerate(records)):
            assert digest not in seen;seen.add(digest);aa[i]=a;bb[i]=b;audios+=int(i in allowed)
            if (i+1)%500==0:print('PREPARE',i+1,len(records),flush=True)
    # One mean/std per physical modality, fitted only on Base-fit; identical for all identities.
    am=float(aa[fit].mean());astd=float(aa[fit].std());bm=float(bb[fit].mean());bstd=float(bb[fit].std());a=np.clip((aa-am)/(3*astd),-1,1)[:,None];b=np.zeros_like(a);valid=np.array(sorted(allowed));b[valid]=np.clip((bb[valid,None]-bm)/(3*bstd),-1,1)
    data=dict(a=a,b=b,fit=fit,dev=dev,audio_indices=valid,normalization=np.array([am,astd,bm,bstd]));np.savez_compressed(path,**data);dump(out/'preparation_public.json',dict(records=len(records),base_users=35,cal_users=12,test_users=12,gan_fit_records=len(fit),gan_develop_records=len(dev),gan_audio_records=audios,cal_test_audio_reads=0,exact_acc_duplicates=0,frontend='same centered2s/DC/STFT4096hop1024FFT16384/log40geometric50..7600Hz;timeinterpolation28to32;one Base-fit global mean/std permodality,clip3SD',shape=[40,32],normalization_fit='Base fit only',identity_roles_disjoint=True));return data

@torch.no_grad()
def validation(g,f,data,device):
    # Predeclared deterministic Base-dev subset, at most2 recordings perBase identity.
    ix=data['validation_indices'];totals=[]
    for s in range(0,len(ix),16):
        jj=ix[s:s+16];a=torch.tensor(data['a'][jj],device=device);b=torch.tensor(data['b'][jj],device=device);fakeb=g(a);fakea=f(b);ca=f(fakeb);cb=g(fakea);totals.append([float(F.l1_loss(fakeb,b)),float(F.l1_loss(fakea,a)),float(F.l1_loss(ca,a)),float(F.l1_loss(cb,b))])
    q=np.mean(totals,0);return dict(pair_audio=float(q[0]),pair_acc=float(q[1]),cycle_acc=float(q[2]),cycle_audio=float(q[3]),selection_loss=float(q.sum()))

def train(source,out,data,steps):
    d=dict(np.load(source/'inputs_private.npz'));ix=np.concatenate([data['dev'][d['user'][data['dev']]==u][:2] for u in BASE]);assert len(ix)==70;data['validation_indices']=ix;device='cuda';torch.manual_seed(42);np.random.seed(42);g=Generator().to(device);f=Generator().to(device);init=out/'initial_generators_private.pt'
    if not init.exists():torch.save(dict(G=g.state_dict(),F=f.state_dict()),init)
    final=out/'gan_complete_private.json';dest=out/'best_generators_private.pt'
    if final.exists():q=torch.load(dest,map_location=device,weights_only=False);g.load_state_dict(q['G']);f.load_state_dict(q['F']);return g,f
    da=Discriminator().to(device);db=Discriminator().to(device);og=torch.optim.Adam(list(g.parameters())+list(f.parameters()),lr=2e-4,betas=(.5,.999));od=torch.optim.Adam(list(da.parameters())+list(db.parameters()),lr=2e-4,betas=(.5,.999));start=0;best=float('inf');history=[];rng=np.random.RandomState(42);resume=out/'resume_private.pt'
    if resume.exists():
        ck=torch.load(resume,map_location=device,weights_only=False)
        for name,obj in [('G',g),('F',f),('DA',da),('DB',db),('OG',og),('OD',od)]:obj.load_state_dict(ck[name])
        start=ck['step'];best=ck['best'];history=ck['history'];rng.set_state(ck['rng'])
    else:dump(out/'initial_validation_public.json',validation(g,f,data,device))
    initialtime=time.time();a_all=torch.tensor(data['a'][data['fit']],device=device);b_all=torch.tensor(data['b'][data['fit']],device=device)
    for step in range(start+1,steps+1):
        lr=2e-4*(1 if step<=steps//2 else (steps-step)/max(steps-steps//2,1))
        for opt in [og,od]:
            for p in opt.param_groups:p['lr']=lr
        jj=rng.choice(len(a_all),16,replace=False);a=a_all[jj];b=b_all[jj]
        for p in list(da.parameters())+list(db.parameters()):p.requires_grad_(False)
        fb=g(a);fa=f(b);ca=f(fb);cb=g(fa);adv=((db(fb)-1)**2).mean()+((da(fa)-1)**2).mean();pair=F.l1_loss(fb,b)+F.l1_loss(fa,a);cyc=F.l1_loss(ca,a)+F.l1_loss(cb,b);ident=F.l1_loss(g(b),b)+F.l1_loss(f(a),a);loss=adv+10*pair+10*cyc+ident;assert torch.isfinite(loss);og.zero_grad();loss.backward();nn.utils.clip_grad_norm_(list(g.parameters())+list(f.parameters()),5);og.step()
        for p in list(da.parameters())+list(db.parameters()):p.requires_grad_(True)
        dl=.5*(((da(a)-1)**2).mean()+da(fa.detach()).square().mean()+((db(b)-1)**2).mean()+db(fb.detach()).square().mean());od.zero_grad();dl.backward();od.step()
        if step%100==0:print('GAN_STEP',step,steps,'loss',float(loss.detach()),'elapsed',round(time.time()-initialtime),flush=True)
        if step%250==0 or step==steps:
            g.eval();f.eval();v=validation(g,f,data,device);v['step']=step;history.append(v)
            if v['selection_loss']<best:best=v['selection_loss'];torch.save(dict(G=g.state_dict(),F=f.state_dict(),step=step,validation=v,fit_indices=data['fit'],validation_indices=ix),dest)
            dump(out/'gan_history_public.json',history);dump(out/'status_public.json',dict(state='training',step=step,total=steps,elapsed=time.time()-initialtime,best_base_dev_loss=best));torch.save(dict(G=g.state_dict(),F=f.state_dict(),DA=da.state_dict(),DB=db.state_dict(),OG=og.state_dict(),OD=od.state_dict(),step=step,best=best,history=history,rng=rng.get_state()),resume);g.train();f.train()
    q=torch.load(dest,map_location=device,weights_only=False);g.load_state_dict(q['G']);f.load_state_dict(q['F']);dump(final,dict(state='complete',steps=steps,selected_step=q['step'],fit_indices=data['fit'].tolist(),validation_indices=ix.tolist(),query_audio_reads=0));dump(out/'gan_training_public.json',dict(steps=steps,batch=16,selected_step=q['step'],fit_records=len(data['fit']),equivalent_epochs=steps*16/len(data['fit']),architecture='shared ResNet G/F,width32,4resblocks,nearestconvupsampling,PatchGAN;not old full772',loss='LSGAN+10pairedL1+10cycleL1+identityL1',selection='70 Base-dev recordings,2peridentity,lowest sum4paired/cycleMAEs;never Cal/Test',best_validation=q['validation']));return g,f

def pooled(z):return torch.cat([z.mean((-2,-1)),z.std((-2,-1),unbiased=False)],1)
def specstats(z):
    z=z[:,0];m=z.mean(-1);sd=z.std(-1,unbiased=False);shape=m-m.mean(1,keepdim=True);parts=torch.cat([z[:,:,j*8:(j+1)*8].mean(-1)-m for j in range(4)],1);return torch.cat([shape,sd,parts],1)
@torch.no_grad()
def features(g,f,a,indices):
    g.eval();f.eval();out={k:[] for k in ['input_encoder256','cycle_encoder256','generated_encoder256','cycle_statistics240','generated_statistics80']}
    for s in range(0,len(indices),32):
        x=torch.tensor(a[indices[s:s+32]],device='cuda');m=g(x);c=f(m);vals=[pooled(g.encoder(x)),pooled(g.encoder(c)),pooled(f.encoder(m)),specstats(c),specstats(m)[:,:80]]
        for k,v in zip(out,vals):out[k].append(v.cpu().numpy())
    return {k:np.concatenate(v) for k,v in out.items()}

def scores(features,d,u,group):
    en=np.flatnonzero((d['user']==u)&(d['role']=='registration'));q=np.flatnonzero(np.isin(d['user'],group)&(d['role']=='query'));assert len(en)==12;return dict(scores=(features[q]@norm(features[en].mean(0))).tolist(),labels=(d['user'][q]==u).tolist())

def evaluate(d,rows,out):
    methods=list(rows[35]);public={};users=[];tt={}
    for method in methods:
        cs=np.concatenate([rows[u][method]['scores'] for u in CAL]);cy=np.concatenate([rows[u][method]['labels'] for u in CAL]);ts=thresholds(cs,cy);tt[method]=ts;public[method]={};assert (cs[~cy]>=ts['Cal_FAR5']).mean()<=.05
        for point,tau in ts.items():
            ss=np.concatenate([rows[u][method]['scores'] for u in TEST]);yy=np.concatenate([rows[u][method]['labels'] for u in TEST]);r=metrics(ss,yy,tau);pp=[]
            for role,group in [('Cal',CAL),('Test',TEST)]:
                for j,u in enumerate(group,1):
                    row=rows[u][method];p=metrics(np.array(row['scores']),np.array(row['labels']),tau);users.append(dict(user=f'Vib59-{role}-{j:02}',role=role,method=method,point=point,**p))
                    if role=='Test':pp.append(p)
            r.update(macro_auc=float(np.mean([v['auc'] for v in pp])),worst_target_far=max(v['far'] for v in pp),lowest_target_tar=min(v['tar'] for v in pp));public[method][point]=r
    rng=np.random.RandomState(20261110);paired={}
    for method in methods:
        if method in ['shape40','baseline232']:continue
        delta=np.array([next(v['hter'] for v in users if v['user']==f'Vib59-Test-{i:02}' and v['method']==method and v['point']=='Cal_EER')-next(v['hter'] for v in users if v['user']==f'Vib59-Test-{i:02}' and v['method']=='baseline232' and v['point']=='Cal_EER') for i in range(1,13)]);boot=np.array([delta[rng.choice(12,12,replace=True)].mean() for _ in range(5000)]);paired[method]=dict(hter_delta=float(delta.mean()),identity_bootstrap95=np.quantile(boot,[.025,.975]).tolist(),improved_targets=int((delta<0).sum()))
    dump(out/'thresholds_private.json',tt);dump(out/'per_user_public.json',users);dump(out/'results_public.json',dict(state='complete',base=35,cal=12,test=12,K=12,seed=42,results=public,paired_vs_baseline232=paired,scope='shared Base-only compact CycleGAN features appended to existing shape40+V4student192;no query Audio;all11attackers;not full772 replication',limits=['single seed','previously explored VibraVox cohort','not same-prompt across users','single-axis forehead accelerometer not VR HMD velocity','12Cal split6/6 for feature choice;selection exploratory']))

def authenticate(source,prior,out,data,g,f):
    d=dict(np.load(source/'inputs_private.npz'));fit=data['fit'];x=StandardScaler().fit(d['x'][fit]).transform(d['x']).astype('float32');shape=norm(x[:,:40]);allfeature=out/'gan_features_private.npz';randomfeature=out/'random_features_private.npz'
    # Only Base/Cal are processed until route and feature-scaling are locked.
    precal=np.flatnonzero(d['user']<47)
    if not (out/'selection_public.json').exists():
        raw=features(g,f,data['a'],precal);standardizers={};z={}
        for k,v in raw.items():sc=StandardScaler().fit(v[np.isin(precal,fit)]);standardizers[k]=sc;z[k]=np.zeros((len(d['x']),v.shape[1]),dtype='float32');z[k][precal]=norm(sc.transform(v));
        import joblib
        joblib.dump(standardizers,out/'feature_scalers_private.joblib');choices=[];cache={}
        def baseline(u):
            net=Student();net.load_state_dict(torch.load(prior/'outer'/f'owner{u}'/'V4_rho0.3_student_private.pt',map_location='cpu',weights_only=False));group=CAL if u in CAL else TEST;ii=np.flatnonzero(((d['user']==u)&(d['role']=='registration'))|(np.isin(d['user'],group)&(d['role']=='query')));a=np.zeros((len(x),192),dtype='float32');a[ii]=encode(net,x[ii]);return np.concatenate([shape,a],1)/np.sqrt(2)
        for u in CAL:cache[u]=baseline(u)
        for branch in ['none']+list(z):
            for alpha in ([0.] if branch=='none' else [.25,.5,1.]):
                def collect(group):
                    rr=[]
                    for u in group:
                        zz=cache[u] if branch=='none' else np.concatenate([cache[u],alpha*z[branch]],1)/np.sqrt(1+alpha*alpha);rr.append(scores(zz,d,u,group))
                    return np.concatenate([r['scores'] for r in rr]),np.concatenate([r['labels'] for r in rr])
                cs,cy=collect(CAL[:6]);qs,qy=collect(CAL[6:]);q=metrics(qs,qy,thresholds(cs,cy)['Cal_EER']);choices.append(dict(branch=branch,alpha=alpha,hter=q['hter'],auc=q['auc'],tar=q['tar'],far=q['far']))
        best=min(choices,key=lambda q:(q['hter'],-q['auc'],q['alpha'],q['branch']));dump(out/'selection_public.json',dict(branch=best['branch'],alpha=best['alpha'],rule='Cal01-06 thresholds;Cal07-12 dev,minHTER then AUC then smalleralpha;allCal-only,no Test feature extraction beforelock;zeroallowed',choices=choices));print('CYCLE_BRANCH_LOCKED',best['branch'],best['alpha'],flush=True)
    # Freeze generator, scalers and selection before processing any Test input.
    import joblib
    sc=joblib.load(out/'feature_scalers_private.joblib');indices=np.arange(len(x));raw=dict(np.load(allfeature)) if allfeature.exists() else features(g,f,data['a'],indices)
    if not allfeature.exists():np.savez_compressed(allfeature,**raw)
    z={k:norm(sc[k].transform(v)) for k,v in raw.items()};choice=json.loads((out/'selection_public.json').read_text());init=torch.load(out/'initial_generators_private.pt',map_location='cuda',weights_only=False);rg=Generator().cuda();rf=Generator().cuda();rg.load_state_dict(init['G']);rf.load_state_dict(init['F']);randomraw=dict(np.load(randomfeature)) if randomfeature.exists() else features(rg,rf,data['a'],indices)
    if not randomfeature.exists():np.savez_compressed(randomfeature,**randomraw)
    randomz={k:norm(StandardScaler().fit(v[fit]).transform(v)) for k,v in randomraw.items()};rows={}
    for u in CAL+TEST:
        folder=out/f'owner{u}';folder.mkdir(exist_ok=True);path=folder/'scores_private.json';net=Student();net.load_state_dict(torch.load(prior/'outer'/f'owner{u}'/'V4_rho0.3_student_private.pt',map_location='cpu',weights_only=False));group=CAL if u in CAL else TEST;ii=np.flatnonzero(((d['user']==u)&(d['role']=='registration'))|(np.isin(d['user'],group)&(d['role']=='query')));student=np.zeros((len(x),192),dtype='float32');student[ii]=encode(net,x[ii]);base=np.concatenate([shape,student],1)/np.sqrt(2);zz=dict(shape40=shape,baseline232=base)
        for k,v in z.items():zz['fixed_'+k]=np.concatenate([base,.5*v],1)/np.sqrt(1.25)
        k=choice['branch'];alpha=choice['alpha'];zz['selected_cycle']=base if k=='none' else np.concatenate([base,alpha*z[k]],1)/np.sqrt(1+alpha*alpha);zz['matched_random_control']=base if k=='none' else np.concatenate([base,alpha*randomz[k]],1)/np.sqrt(1+alpha*alpha);zz['random_input_encoder_control']=np.concatenate([base,.5*randomz['input_encoder256']],1)/np.sqrt(1.25)
        rows[u]={name:scores(v,d,u,group) for name,v in zz.items()};dump(path,rows[u]);print('AUTH_OWNER',u,flush=True)
    dump(out/'scores_private.json',{str(k):v for k,v in rows.items()});evaluate(d,rows,out);dump(out/'status_public.json',dict(state='complete'));print('ALL_COMPLETE',flush=True)

def main(a):
    torch.set_num_threads(2);assert torch.cuda.is_available();torch.backends.cudnn.benchmark=True;source=Path(a.source);out=Path(a.out);out.mkdir(parents=True,exist_ok=True);data=prepare(source,Path(a.paired),out);g,f=train(source,out,data,a.steps);authenticate(source,Path(a.rkd02),out,data,g,f)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--paired',required=True);p.add_argument('--rkd02',required=True);p.add_argument('--out',required=True);p.add_argument('--steps',type=int,default=4000);main(p.parse_args())
