"""Nine known users, binary RBF heads, registration-only personal CycleGAN.

Nominal same-word and cross-word tests, fixed all-eight known attackers.
Audio wave cache may decompress as a whole; only allowed rows enter models.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import argparse,json,sys,copy,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from scipy import signal
import joblib,torch
from torch import nn
from torch.nn import functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from run_rkd01_vibravox import norm,supcon,rkd,metrics,thresholds
from run_mei9_shape_v4 import Student,encode,features as raw_features,rows,reg,replay
from vibravox_cycle_supplement import Generator as OriginalGenerator,Discriminator as OriginalDiscriminator,pooled,dump
USERS=list(range(1,10));BRANCHES=['input256','cycle256','cycle_stats720']

class Generator(OriginalGenerator):
    def __init__(self,ci,co):
        super().__init__()
        if ci!=1:self.encoder[1]=nn.Conv2d(ci,32,7)
        if co!=1:self.decoder[-2]=nn.Conv2d(32,co,7)
class Discriminator(OriginalDiscriminator):
    def __init__(self,ci):
        super().__init__()
        if ci!=1:self.net[0]=nn.Conv2d(ci,32,4,2,1)

def indices(d):
    ix=dict(fit=np.flatnonzero(d['role']=='fit'),dev=np.array([rows(d,u,w)[0] for u in USERS for w in [8,9,18,19]]))
    assert np.all(d['role'][ix['dev']]=='develop')
    for name,sl in [('register',slice(0,3)),('calA',slice(3,4)),('calB',slice(4,5)),('sameword',slice(5,9))]:ix[name]=np.concatenate([rows(d,u,w)[sl] for u in USERS for w in range(20,24)])
    ix['crossword']=np.concatenate([rows(d,u,w)[:4] for u in USERS for w in range(24,30)])
    assert len(ix['register'])==108 and len(ix['calA'])==36 and len(ix['calB'])==36 and len(ix['sameword'])==144 and len(ix['crossword'])==216
    for i,a in enumerate(ix):
        for b in list(ix)[:i]:assert not set(ix[a])&set(ix[b])
    return ix

def prepare(a,out):
    d=dict(np.load(a.frontend));ix=indices(d);path=out/'prepared_private.npz'
    if path.exists():return d,ix,dict(np.load(path))
    raw=np.load(Path(a.raw)/'raw9_full_features_private.npz');mapping={s:i for i,s in enumerate(raw['sample_id'])};ri=np.array([mapping[s] for s in d['sample_id']]);assert np.array_equal(raw['user'][ri],d['user']) and np.array_equal(raw['word'][ri],d['word']);allowed=np.unique(np.r_[ix['fit'],ix['dev'],ix['register']]);selected=np.unique(np.r_[np.concatenate([replay(d,u) for u in USERS]),ix['register']]);assert len(selected)==144
    wave=raw['wave'][ri[allowed]].astype('float32');wave-=wave.mean(1,keepdims=True);slots={int(i):j for j,i in enumerate(allowed)}
    def audio(w):
        f,_,z=signal.stft(w.astype('float64'),16000,nperseg=4096,noverlap=3072,nfft=16384,boundary=None,padded=False);lp=np.log(abs(z)**2+1e-20);b=np.stack([np.interp(np.geomspace(50,7600,40),f,lp[:,j]) for j in range(lp.shape[1])],1);return np.stack([np.interp(np.linspace(0,b.shape[1]-1,32),np.arange(b.shape[1]),v) for v in b]).astype('float32')
    av=F.interpolate(torch.tensor(d['spectrum']),size=(40,32),mode='bilinear',align_corners=True).numpy();bv=np.zeros((len(av),1,40,32),dtype='float32')
    with ThreadPoolExecutor(max_workers=8) as workers:
        for i,b in zip(allowed,workers.map(audio,wave)):bv[i,0]=b
    am=av[ix['fit']].mean((0,2,3),keepdims=True);asd=av[ix['fit']].std((0,2,3),keepdims=True);bm=float(bv[ix['fit']].mean());bsd=float(bv[ix['fit']].std());av=np.clip((av-am)/(3*asd),-1,1);bv[allowed]=np.clip((bv[allowed]-bm)/(3*bsd),-1,1);xr,vr=raw_features(d);sc=StandardScaler().fit(xr[ix['fit']]);x=sc.transform(xr).astype('float32');shape=norm(x[:,:120]);stats=norm(StandardScaler().fit(vr[ix['fit']]).transform(vr));twave=wave[[slots[int(i)] for i in selected]];data=dict(a=av,b=bv,x=x,shape=shape,stats=stats,audio_indices=allowed,teacher_indices=selected,teacher_wave=twave,input_mean=am,input_std=asd,audio_mean=np.array(bm),audio_std=np.array(bsd));np.savez_compressed(path,**data);dump(out/'split_private.json',{k:v.tolist() for k,v in ix.items()});dump(out/'preparation_public.json',dict(records=len(av),users=9,identity_protocol='closed:all9fit,all8knownnegativeidentities',counts={k:len(v) for k,v in ix.items()},teacher_audio_records=144,gan_audio_allowed_records=len(allowed),query_audio_model_forwards=0,source='exact-timestamp raw9 common2695;XYZ OpenVR tracking vVelocity,not raw accelerometer',limits=['nominal word order inferred,not transcript-verified','cached microphone left-channel wavefloat16 convertedfloat32;archive may decompress whole wave array','same headset,no redon/session/device labels']));return d,ix,data

def teacher_student(a,d,ix,data,out):
    final=out/'student_private.pt';torch.manual_seed(42);net=Student()
    if final.exists():net.load_state_dict(torch.load(final,map_location='cpu',weights_only=False));return net
    torch.manual_seed(42);rng=np.random.RandomState(42);head=nn.Linear(192,9,bias=False);opt=torch.optim.AdamW(list(net.parameters())+list(head.parameters()),lr=1e-4,weight_decay=1e-4);fitwords=sorted(set(d['word'][ix['fit']]));lookup={(u,w):np.intersect1d(rows(d,u,w),ix['fit']) for u in USERS for w in fitwords}
    for step in range(700):
        ww=rng.choice(fitwords,4,replace=False);ii=np.concatenate([[rng.choice(lookup[u,w]) for w in ww] for u in USERS]);y=torch.tensor(d['user'][ii]-1,dtype=torch.long);z=net(torch.tensor(data['x'][ii]));loss=F.cross_entropy(16*F.linear(z,F.normalize(head.weight,dim=1)),y)+.1*supcon(z,y);opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(list(net.parameters())+list(head.parameters()),1);opt.step()
    torch.save(net.state_dict(),out/'initial_student_private.pt');sys.path.insert(0,'/root');from run_ecapa_pilot import teacher_load
    for n in ['model','deps']:
        p=out/n
        if not p.exists():p.symlink_to(Path(a.raw)/n,target_is_directory=True)
    model=teacher_load(out);emb=model.mods.embedding_model;device=next(emb.parameters()).device;ii=data['teacher_indices'];wave=torch.tensor(data['teacher_wave'],device=device);ff=[]
    with torch.no_grad():
        for s in range(0,len(wave),16):ff.append(model.mods.mean_var_norm(model.mods.compute_features(wave[s:s+16]),torch.ones(len(wave[s:s+16]),device=device)))
    feat=torch.cat(ff);lens=torch.ones(len(feat),device=device);yy=torch.tensor(d['user'][ii]-1,device=device,dtype=torch.long);emb.eval()
    for p in emb.parameters():p.requires_grad_(True)
    def forward(j):return F.normalize(emb(feat[j],lens[j]).squeeze(1),dim=1)
    with torch.no_grad():orig=forward(torch.arange(len(ii),device=device)).detach()
    h=nn.Linear(192,9,bias=False).to(device)
    with torch.no_grad():h.weight.copy_(F.normalize(torch.stack([orig[yy==u].mean(0) for u in range(9)]),dim=1))
    op=torch.optim.AdamW([dict(params=emb.parameters(),lr=1e-6),dict(params=h.parameters(),lr=1e-4)],weight_decay=1e-4);rng=np.random.RandomState(4242)
    for step in range(100):
        j=np.concatenate([rng.choice(np.flatnonzero(d['user'][ii]==u),4,replace=False) for u in USERS]);jt=torch.tensor(j,device=device);z=forward(jt);y=yy[jt];loss=F.cross_entropy(16*F.linear(z,F.normalize(h.weight,dim=1)),y)+.1*supcon(z.cpu(),y.cpu()).to(device)+.1*(1-F.cosine_similarity(z,orig[jt],dim=1)).mean();op.zero_grad();loss.backward();nn.utils.clip_grad_norm_(list(emb.parameters())+list(h.parameters()),1);op.step()
    for p in emb.parameters():p.requires_grad_(False)
    with torch.no_grad():targets=forward(torch.arange(len(ii),device=device)).cpu().numpy()
    np.savez_compressed(out/'teacher_targets_private.npz',indices=ii,targets=targets);torch.save(emb.state_dict(),out/'audio_teacher_private.pt');del model,emb,feat,wave,op,h;torch.cuda.empty_cache();start=encode(net,data['x'][ii]);head=nn.Linear(192,9,bias=False);labels=torch.tensor(d['user'][ii]-1,dtype=torch.long)
    with torch.no_grad():head.weight.copy_(torch.tensor(norm(np.stack([start[labels.numpy()==u].mean(0) for u in range(9)]))))
    opt=torch.optim.AdamW(list(net.parameters())+list(head.parameters()),lr=1e-4,weight_decay=1e-4);params=list(net.parameters());rng=np.random.RandomState(42000);ei=ek=None;trace=[]
    for step in range(1,101):
        j=np.concatenate([rng.choice(np.flatnonzero(d['user'][ii]==u),4,replace=False) for u in USERS]);z=net(torch.tensor(data['x'][ii[j]]));y=labels[j];identity=F.cross_entropy(16*F.linear(z,F.normalize(head.weight,dim=1)),y)+.1*supcon(z,y)+.1*(1-F.cosine_similarity(z,torch.tensor(start[j]),dim=1)).mean();kd=rkd(z,torch.tensor(targets[j]),y);gi=torch.autograd.grad(identity,params,retain_graph=True);gk=torch.autograd.grad(kd,params,retain_graph=True);ni=float(torch.sqrt(sum(g.square().sum() for g in gi)));nk=float(torch.sqrt(sum(g.square().sum() for g in gk)));ei=ni if ei is None else .9*ei+.1*ni;ek=nk if ek is None else .9*ek+.1*nk;lam=min(100.,.3*ei/max(ek,1e-8));loss=identity+lam*kd;opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(list(net.parameters())+list(head.parameters()),1);opt.step()
        if step in [1,25,50,100]:trace.append(dict(step=step,lambda_rkd=lam,gradient_ratio=lam*nk/max(ni,1e-8)))
    torch.save(net.state_dict(),final);dump(out/'student_training_private.json',dict(base_fit_indices=ix['fit'].tolist(),distill_indices=ii.tolist(),base_steps=700,distill_steps=100,all_training_identities=USERS,trace=trace,query_audio_forwards=0));print('CLOSED_TEACHER_STUDENT_COMPLETE',flush=True);return net

@torch.no_grad()
def val(g,f,data,ix):
    values=[]
    for s in range(0,len(ix['dev']),9):
        j=ix['dev'][s:s+9];a=torch.tensor(data['a'][j],device='cuda');b=torch.tensor(data['b'][j],device='cuda');fb=g(a);fa=f(b);values.append([float(F.l1_loss(fb,b)),float(F.l1_loss(fa,a)),float(F.l1_loss(f(fb),a)),float(F.l1_loss(g(fa),b))])
    return np.mean(values,0).tolist()

def train_gan(data,ix,out):
    dest=out/'shared_gan_private.pt';torch.manual_seed(42);g=Generator(3,1).cuda();f=Generator(1,3).cuda();da=Discriminator(3).cuda();db=Discriminator(1).cuda()
    if (out/'gan_complete_public.json').exists():ck=torch.load(dest,map_location='cpu',weights_only=False);g.load_state_dict(ck['G']);f.load_state_dict(ck['F']);da.load_state_dict(ck['DA']);db.load_state_dict(ck['DB']);return g,f,da,db
    og=torch.optim.Adam(list(g.parameters())+list(f.parameters()),lr=2e-4,betas=(.5,.999));od=torch.optim.Adam(list(da.parameters())+list(db.parameters()),lr=2e-4,betas=(.5,.999));rng=np.random.RandomState(42);best=float('inf');history=[];start=0;resume=out/'gan_resume_private.pt'
    if resume.exists():
        ck=torch.load(resume,map_location='cuda',weights_only=False)
        for k,obj in [('G',g),('F',f),('DA',da),('DB',db),('OG',og),('OD',od)]:obj.load_state_dict(ck[k])
        start=ck['step'];best=ck['best'];history=ck['history'];rng.set_state(ck['rng'])
    aa=torch.tensor(data['a'][ix['fit']],device='cuda');bb=torch.tensor(data['b'][ix['fit']],device='cuda');t0=time.time()
    for step in range(start+1,4001):
        lr=2e-4*(1 if step<=2000 else (4000-step)/2000)
        for opt in [og,od]:
            for group in opt.param_groups:group['lr']=lr
        j=rng.choice(len(aa),16,replace=False);a=aa[j];b=bb[j]
        for p in list(da.parameters())+list(db.parameters()):p.requires_grad_(False)
        fb=g(a);fa=f(b);pair=F.l1_loss(fb,b)+F.l1_loss(fa,a);cycle=F.l1_loss(f(fb),a)+F.l1_loss(g(fa),b);adv=((db(fb)-1)**2).mean()+((da(fa)-1)**2).mean();loss=adv+10*pair+10*cycle;assert torch.isfinite(loss);og.zero_grad();loss.backward();nn.utils.clip_grad_norm_(list(g.parameters())+list(f.parameters()),5);og.step()
        for p in list(da.parameters())+list(db.parameters()):p.requires_grad_(True)
        dl=.5*(((da(a)-1)**2).mean()+da(fa.detach()).square().mean()+((db(b)-1)**2).mean()+db(fb.detach()).square().mean());od.zero_grad();dl.backward();od.step()
        if step%100==0:print('CLOSED_GAN_STEP',step,4000,'seconds',round(time.time()-t0),flush=True)
        if step%250==0:
            vv=val(g,f,data,ix);history.append(dict(step=step,pair_audio=vv[0],pair_acc=vv[1],cycle_acc=vv[2],cycle_audio=vv[3]));score=sum(vv)
            if score<best:best=score;torch.save(dict(G=g.state_dict(),F=f.state_dict(),DA=da.state_dict(),DB=db.state_dict(),step=step,fit_indices=ix['fit'],dev_indices=ix['dev']),dest)
            torch.save(dict(G=g.state_dict(),F=f.state_dict(),DA=da.state_dict(),DB=db.state_dict(),OG=og.state_dict(),OD=od.state_dict(),step=step,best=best,history=history,rng=rng.get_state()),resume);dump(out/'gan_history_public.json',history)
    ck=torch.load(dest,map_location='cpu',weights_only=False)
    for k,obj in [('G',g),('F',f),('DA',da),('DB',db)]:obj.load_state_dict(ck[k])
    dump(out/'gan_complete_public.json',dict(steps=4000,batch=16,equivalent_epochs=4000*16/len(ix['fit']),selected_step=ck['step'],fit_records=len(ix['fit']),develop_records=len(ix['dev']),all9known_training_users=True,loss='LSGAN+10pairedL1+10cycleL1;3ACCchannels/1Audiochannel,identitymapping omitted due channel mismatch'));return g,f,da,db

@torch.no_grad()
def gan_features(g,f,a,ii):
    out={k:[] for k in BRANCHES}
    for s in range(0,len(ii),32):
        x=torch.tensor(a[ii[s:s+32]],device='cuda');c=f(g(x));m=c.mean(-1);lev=m.mean(-1,keepdim=True);std=c.std(-1,unbiased=False);parts=torch.cat([c[:,:,:,j*8:(j+1)*8].mean(-1)-m for j in range(4)],2);stat=torch.cat([(m-lev).flatten(1),std.flatten(1),parts.flatten(1)],1);values=[pooled(g.encoder(x)),pooled(g.encoder(c)),stat]
        for k,v in zip(out,values):out[k].append(v.cpu().numpy())
    return {k:np.concatenate(v) for k,v in out.items()}

def micro(owner,g,f,da,db,data,ix,out):
    folder=out/f'owner{owner}';folder.mkdir(exist_ok=True)
    if all((folder/f'gan{s}_private.pt').exists() for s in [25,100]):return
    gg=Generator(3,1).cuda();ff=Generator(1,3).cuda();gg.load_state_dict(g.state_dict());ff.load_state_dict(f.state_dict());en=ix['register'][d_global['user'][ix['register']]==owner];assert len(en)==12;aa=torch.tensor(data['a'][en],device='cuda');bb=torch.tensor(data['b'][en],device='cuda');opt=torch.optim.Adam(list(gg.parameters())+list(ff.parameters()),lr=2e-5,betas=(.5,.999));rng=np.random.RandomState(9900+owner)
    for model in [da,db]:
        for p in model.parameters():p.requires_grad_(False)
    for step in range(1,101):
        j=rng.choice(12,6,replace=False);a=aa[j];b=bb[j];fb=gg(a);fa=ff(b);loss=((db(fb)-1)**2).mean()+((da(fa)-1)**2).mean()+10*(F.l1_loss(fb,b)+F.l1_loss(fa,a))+10*(F.l1_loss(ff(fb),a)+F.l1_loss(gg(fa),b));opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(list(gg.parameters())+list(ff.parameters()),5);opt.step()
        if step in [25,100]:torch.save(dict(G=gg.state_dict(),F=ff.state_dict(),train_indices=en,owner=owner,step=step),folder/f'gan{step}_private.pt')
    dump(folder/'gan_training_private.json',dict(owner=owner,train_indices=en.tolist(),steps=100,reset_shared=True,other_user_records=0,query_audio_model_forwards=0));print('CLOSED_PERSONAL_GAN',owner,flush=True)

def matrices(base,z,c):return base if c['step']==0 else np.concatenate([base,c['alpha']*z[c['step']][c['branch']]],1)/np.sqrt(1+c['alpha']**2)
def svm(x,d,ix,u):return SVC(C=10,gamma='scale',class_weight='balanced').fit(x[ix['register']],d['user'][ix['register']]==u)
def evaluate(d,rows,out):
    results={};users=[];threshold_private={};paired={}
    for method in rows[1]:
        results[method]={};threshold_private[method]={}
        for u in USERS:
            c=rows[u][method]['calibration'];cs=np.array(c['scores']);cy=np.array(c['labels']);ts=thresholds(cs,cy);threshold_private[method][str(u)]=ts
            for cond in ['sameword','crossword']:
                r=rows[u][method][cond];s=np.array(r['scores']);y=np.array(r['labels'])
                for point,tau in ts.items():users.append(dict(user=f'Mei9-U{u:02}',method=method,condition=cond,point=point,**metrics(s,y,tau)))
        for cond in ['sameword','crossword']:
            results[method][cond]={}
            for point in ['Cal_EER','Cal_FAR5']:
                pp=[v for v in users if v['method']==method and v['condition']==cond and v['point']==point];results[method][cond][point]={k:float(np.mean([p[k] for p in pp])) for k in ['tar','far','frr','hter','auc','eer']};results[method][cond][point]['worst_target_far']=max(p['far'] for p in pp)
    rng=np.random.RandomState(20261110)
    for method in rows[1]:
        if method=='baseline312':continue
        paired[method]={}
        for cond in ['sameword','crossword']:
            delta=np.array([next(v['hter'] for v in users if v['user']==f'Mei9-U{u:02}' and v['method']==method and v['condition']==cond and v['point']=='Cal_EER')-next(v['hter'] for v in users if v['user']==f'Mei9-U{u:02}' and v['method']=='baseline312' and v['condition']==cond and v['point']=='Cal_EER') for u in USERS]);boot=np.array([delta[rng.choice(9,9,replace=True)].mean() for _ in range(5000)]);paired[method][cond]=dict(hter_delta=float(delta.mean()),identity_bootstrap95=np.quantile(boot,[.025,.975]).tolist(),improved_users=int((delta<0).sum()))
    dump(out/'thresholds_private.json',threshold_private);dump(out/'per_user_public.json',users);dump(out/'results_public.json',dict(state='complete',users=9,seed=42,backend='9balanced binary RBF SVM C10 gammaScale;all8knownnegatives;no postconcat scaling',protocol='closed all9identitytrain;12register,8cal/person;16sameword/24crosswordquery/person;personalG/Fown12only;no query Audio forward',results=results,paired_vs_baseline312=paired,limits=['nominal inferred word order,not transcript-verified','raw9indices not necessarily historicaltrainbU mapping','OpenVR velocityXYZ not raw accelerometer','same headset/no redon or crossdevice','single seed/explored cohort','RBF closed backend differs from previous open cosine,compare methods only inside this experiment']))

def authenticate(d,ix,data,student,g,f,da,db,out):
    emb=encode(student,data['x']);base=np.concatenate([data['shape'],emb],1)/np.sqrt(2);scpath=out/'gan_scalers_private.joblib'
    if scpath.exists():sc=joblib.load(scpath)
    else:raw=gan_features(g,f,data['a'],ix['fit']);sc={k:StandardScaler().fit(v) for k,v in raw.items()};joblib.dump(sc,scpath)
    pre=np.unique(np.r_[ix['register'],ix['calA'],ix['calB']]);cache={}
    def owner_features(u,ii):
        zz={}
        for step in [25,100]:
            folder=out/f'owner{u}';path=folder/f'features{step}_{len(ii)}_private.npz'
            if path.exists():raw=dict(np.load(path))
            else:
                ck=torch.load(folder/f'gan{step}_private.pt',map_location='cpu',weights_only=False);gg=Generator(3,1).cuda();ff=Generator(1,3).cuda();gg.load_state_dict(ck['G']);ff.load_state_dict(ck['F']);raw=gan_features(gg,ff,data['a'],ii);np.savez_compressed(path,**raw)
            zz[step]={}
            for k,v in raw.items():a=np.zeros((len(base),v.shape[1]),dtype='float32');a[ii]=norm(sc[k].transform(v));zz[step][k]=a
        return zz
    for u in USERS:micro(u,g,f,da,db,data,ix,out);cache[u]=owner_features(u,pre)
    lock=out/'selection_public.json'
    if not lock.exists():
        choices=[dict(step=0,branch='none',alpha=0.)]+[dict(step=s,branch=k,alpha=w) for s in [25,100] for k in BRANCHES for w in [.25,.5,1.]]
        for c in choices:
            pp=[]
            for u in USERS:
                x=matrices(base,cache[u],c);model=svm(x,d,ix,u);a=ix['calA'];b=ix['calB'];cs=model.decision_function(x[a]);s=model.decision_function(x[b]);y=d['user'][a]==u;yy=d['user'][b]==u;pp.append(metrics(s,yy,thresholds(cs,y)['Cal_EER']))
            c.update(hter=float(np.mean([v['hter'] for v in pp])),auc=float(np.mean([v['auc'] for v in pp])))
        best=min(choices,key=lambda c:(c['hter'],-c['auc'],c['alpha'],c['step'],c['branch']));dump(lock,dict(selected=best,choices=choices,rule='SVMtrain registration108;CalA threshold CalB featureselection across9users;no finalquery features extracted beforelock;minHTER/AUC/smallerweight/fewersteps;zeroallowed'));print('CLOSED_SELECTION_LOCKED',best,flush=True)
    choice=json.loads(lock.read_text())['selected'];allix=np.unique(np.r_[pre,ix['sameword'],ix['crossword']]);results={}
    for u in USERS:
        zz=owner_features(u,allix);variants=dict(raw120=data['shape'],statistics720=data['stats'],student192=emb,baseline312=base)
        for step in [25,100]:
            for k in BRANCHES:variants[f'personal{step}_{k}']=matrices(base,zz,dict(step=step,branch=k,alpha=.5))
        variants['selected_cycle']=matrices(base,zz,choice);variants['cycle_only100']=zz[100]['cycle256'];rowsout={};models={};cal=np.r_[ix['calA'],ix['calB']]
        for method,x in variants.items():
            model=svm(x,d,ix,u);models[method]=model;rowsout[method]={}
            for cond,j in [('calibration',cal),('sameword',ix['sameword']),('crossword',ix['crossword'])]:rowsout[method][cond]=dict(scores=model.decision_function(x[j]).tolist(),labels=(d['user'][j]==u).tolist())
        joblib.dump(models,out/f'owner{u}'/'rbf_heads_private.joblib');dump(out/f'owner{u}'/'scores_private.json',rowsout);results[u]=rowsout;print('CLOSED_AUTH_OWNER',u,flush=True)
    dump(out/'scores_private.json',{str(k):v for k,v in results.items()});evaluate(d,results,out);print('MEI9_CLOSED_COMPLETE',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--frontend',required=True);p.add_argument('--raw',required=True);p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2);torch.backends.cudnn.benchmark=True;d,ix,data=prepare(a,out);d_global=d;student=teacher_student(a,d,ix,data,out);g,f,da,db=train_gan(data,ix,out);authenticate(d,ix,data,student,g,f,da,db,out)
