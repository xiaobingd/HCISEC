"""ABCS43 open-set feature supplementation, Base-only G/F + own registration fine-tune."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import argparse,json,time
from pathlib import Path
import numpy as np
import torch,joblib
from torch import nn
from torch.nn import functional as F
from sklearn.preprocessing import StandardScaler
from vibravox_cycle_supplement import Generator,Discriminator,features,dump
from run_rkd01_vibravox import Student,encode,norm,metrics,thresholds
from explore_abcs43_ecapa import evaluate
BASE=list(range(23));CAL=list(range(23,31));TEST=list(range(31,43));BRANCHES=['input_encoder256','cycle_encoder256','generated_encoder256','cycle_statistics240','generated_statistics80']

def score(z,d,u,group):
    role='calibration' if u in CAL else 'query';en=np.flatnonzero((d['user']==u)&(d['role']=='registration'));q=np.flatnonzero(np.isin(d['user'],group)&(d['role']==role));assert len(en)==12 and not set(en)&set(q);return dict(scores=(z[q]@norm(z[en].mean(0))).tolist(),labels=(d['user'][q]==u).tolist())

@torch.no_grad()
def validate(g,f,data):
    values=[];weights=[]
    for s in range(0,len(data['dev']),16):
        j=data['dev'][s:s+16];a=torch.tensor(data['a'][j],device='cuda');b=torch.tensor(data['b'][j],device='cuda');fb=g(a);fa=f(b);values.append([float(F.l1_loss(fb,b)),float(F.l1_loss(fa,a)),float(F.l1_loss(f(fb),a)),float(F.l1_loss(g(fa),b))]);weights.append(len(j))
    return np.average(values,axis=0,weights=weights).tolist()

def train(data,out):
    torch.manual_seed(42);g=Generator().cuda();f=Generator().cuda();da=Discriminator().cuda();db=Discriminator().cuda();initial=out/'initial_generators_private.pt';bestpath=out/'shared_gan_private.pt'
    if not initial.exists():torch.save(dict(G=g.state_dict(),F=f.state_dict()),initial)
    if not (out/'gan_complete_public.json').exists():
        og=torch.optim.Adam(list(g.parameters())+list(f.parameters()),lr=2e-4,betas=(.5,.999));od=torch.optim.Adam(list(da.parameters())+list(db.parameters()),lr=2e-4,betas=(.5,.999));rng=np.random.RandomState(42);start=0;best=float('inf');history=[];resume=out/'gan_resume_private.pt'
        if resume.exists():
            ck=torch.load(resume,map_location='cuda',weights_only=False)
            for k,obj in [('G',g),('F',f),('DA',da),('DB',db),('OG',og),('OD',od)]:obj.load_state_dict(ck[k])
            start=ck['step'];best=ck['best'];history=ck['history'];rng.set_state(ck['rng'])
        else:dump(out/'initial_gan_validation_public.json',dict(zip(['pair_audio','pair_acc','cycle_acc','cycle_audio'],validate(g,f,data))))
        aa=torch.tensor(data['a'][data['fit']],device='cuda');bb=torch.tensor(data['b'][data['fit']],device='cuda');t0=time.time()
        for step in range(start+1,4001):
            lr=2e-4*(1 if step<=2000 else (4000-step)/2000)
            for opt in [og,od]:
                for group in opt.param_groups:group['lr']=lr
            j=rng.choice(len(aa),16,replace=False);a=aa[j];b=bb[j]
            for p in list(da.parameters())+list(db.parameters()):p.requires_grad_(False)
            fb=g(a);fa=f(b);adv=((db(fb)-1)**2).mean()+((da(fa)-1)**2).mean();pair=F.l1_loss(fb,b)+F.l1_loss(fa,a);cycle=F.l1_loss(f(fb),a)+F.l1_loss(g(fa),b);identity=F.l1_loss(g(b),b)+F.l1_loss(f(a),a);loss=adv+10*pair+10*cycle+identity;assert torch.isfinite(loss);og.zero_grad();loss.backward();nn.utils.clip_grad_norm_(list(g.parameters())+list(f.parameters()),5);og.step()
            for p in list(da.parameters())+list(db.parameters()):p.requires_grad_(True)
            dl=.5*(((da(a)-1)**2).mean()+da(fa.detach()).square().mean()+((db(b)-1)**2).mean()+db(fb.detach()).square().mean());od.zero_grad();dl.backward();od.step()
            if step%100==0:print('ABCS_GAN_STEP',step,4000,'seconds',round(time.time()-t0),flush=True)
            if step%250==0:
                vv=validate(g,f,data);history.append(dict(step=step,pair_audio=vv[0],pair_acc=vv[1],cycle_acc=vv[2],cycle_audio=vv[3]));value=sum(vv)
                if value<best:best=value;torch.save(dict(G=g.state_dict(),F=f.state_dict(),DA=da.state_dict(),DB=db.state_dict(),step=step,fit_indices=data['fit'],develop_indices=data['dev']),bestpath)
                torch.save(dict(G=g.state_dict(),F=f.state_dict(),DA=da.state_dict(),DB=db.state_dict(),OG=og.state_dict(),OD=od.state_dict(),step=step,best=best,history=history,rng=rng.get_state()),resume);dump(out/'gan_history_public.json',history)
        ck=torch.load(bestpath,map_location='cpu',weights_only=False);dump(out/'gan_complete_public.json',dict(steps=4000,batch=16,equivalent_epochs=4000*16/len(data['fit']),selected_step=ck['step'],base_users=23,fit_records=len(data['fit']),develop_records=len(data['dev']),architecture='shared width32 fourRes G/F,PatchGAN,40x32',loss='LSGAN+10pairedL1+10cycleL1+identitymapping1',selection='only Base23 develop345,weighted mean sum4MAEs'))
    ck=torch.load(bestpath,map_location='cpu',weights_only=False)
    for k,obj in [('G',g),('F',f),('DA',da),('DB',db)]:obj.load_state_dict(ck[k])
    for obj in [da,db]:
        for p in obj.parameters():p.requires_grad_(False)
    return g,f,da,db

def personalize(u,d,data,g,f,da,db,out):
    folder=out/f'owner{u}';folder.mkdir(exist_ok=True)
    if all((folder/f'gan{s}_private.pt').exists() for s in [25,100]):return
    gg=Generator().cuda();ff=Generator().cuda();gg.load_state_dict(g.state_dict());ff.load_state_dict(f.state_dict());assert all(torch.equal(v,g.state_dict()[k]) for k,v in gg.state_dict().items());en=np.flatnonzero((d['user']==u)&(d['role']=='registration'));assert len(en)==12;aa=torch.tensor(data['a'][en],device='cuda');bb=torch.tensor(data['b'][en],device='cuda');op=torch.optim.Adam(list(gg.parameters())+list(ff.parameters()),lr=2e-5,betas=(.5,.999));rng=np.random.RandomState(9900+u);history=[]
    for step in range(1,101):
        j=rng.choice(12,6,replace=False);a=aa[j];b=bb[j];fb=gg(a);fa=ff(b);pair=F.l1_loss(fb,b)+F.l1_loss(fa,a);cycle=F.l1_loss(ff(fb),a)+F.l1_loss(gg(fa),b);loss=((db(fb)-1)**2).mean()+((da(fa)-1)**2).mean()+10*pair+10*cycle+F.l1_loss(gg(b),b)+F.l1_loss(ff(a),a);assert torch.isfinite(loss);op.zero_grad();loss.backward();nn.utils.clip_grad_norm_(list(gg.parameters())+list(ff.parameters()),5);op.step()
        if step in [1,25,100]:history.append(dict(step=step,pair=float(pair.detach()),cycle=float(cycle.detach()),loss=float(loss.detach())))
        if step in [25,100]:torch.save(dict(G=gg.state_dict(),F=ff.state_dict(),train_indices=en,owner=u,step=step),folder/f'gan{step}_private.pt')
    dump(folder/'training_audit_private.json',dict(owner=u,train_indices=en.tolist(),steps=100,batch=6,lr=2e-5,other_user_train_records=0,query_audio_model_forwards=0,reset_shared=True,history=history));print('ABCS_PERSONAL_GAN',u,flush=True)

def join(base,branch,alpha):return np.concatenate([base,alpha*branch],1)/np.sqrt(1+alpha**2)
def authenticate(source,out,d,data,g,f,da,db):
    fit=data['fit'];x=StandardScaler().fit(d['x'][fit]).transform(d['x']).astype('float32');shape=norm(x[:,:40]);stats=norm(StandardScaler().fit(d['statistics'][fit]).transform(d['statistics']));scpath=out/'feature_scalers_private.joblib'
    if scpath.exists():sc=joblib.load(scpath)
    else:raw=features(g,f,data['a'],fit);sc={k:StandardScaler().fit(v) for k,v in raw.items()};joblib.dump(sc,scpath)
    cache={}
    def make(u):
        personalize(u,d,data,g,f,da,db,out);group=CAL if u in CAL else TEST;role='calibration' if u in CAL else 'query';ii=np.flatnonzero(((d['user']==u)&(d['role']=='registration'))|(np.isin(d['user'],group)&(d['role']==role)));student=Student();student.load_state_dict(torch.load(source/f'owner{u}'/'student_private.pt',map_location='cpu',weights_only=False));emb=np.zeros((len(x),192),dtype='float32');emb[ii]=encode(student,x[ii]);base=np.concatenate([shape,emb],1)/np.sqrt(2);z={}
        for step in [25,100]:
            folder=out/f'owner{u}';path=folder/f'features{step}_private.npz'
            if path.exists():raw=dict(np.load(path))
            else:
                ck=torch.load(folder/f'gan{step}_private.pt',map_location='cpu',weights_only=False);gg=Generator().cuda();ff=Generator().cuda();gg.load_state_dict(ck['G']);ff.load_state_dict(ck['F']);raw=features(gg,ff,data['a'],ii);np.savez_compressed(path,**raw)
            z[step]={}
            for k,v in raw.items():zz=np.zeros((len(x),v.shape[1]),dtype='float32');zz[ii]=norm(sc[k].transform(v));z[step][k]=zz
        return base,emb,z
    for u in CAL:cache[u]=make(u)
    lock=out/'selection_public.json'
    if not lock.exists():
        choices=[dict(step=0,branch='none',alpha=0.)]+[dict(step=s,branch=k,alpha=w) for s in [25,100] for k in BRANCHES for w in [.25,.5,1.]]
        for c in choices:
            def collect(group):
                rr=[]
                for u in group:
                    base,emb,z=cache[u];rr.append(score(base if c['step']==0 else join(base,z[c['step']][c['branch']],c['alpha']),d,u,group))
                return np.concatenate([v['scores'] for v in rr]),np.concatenate([v['labels'] for v in rr])
            cs,cy=collect(CAL[:4]);s,y=collect(CAL[4:]);v=metrics(s,y,thresholds(cs,cy)['Cal_EER']);c.update({k:v[k] for k in ['tar','far','hter','auc']})
        selected=min(choices,key=lambda c:(c['hter'],-c['auc'],c['alpha'],c['step'],c['branch']));dump(lock,dict(selected=selected,choices=choices,rule='Cal01-04threshold Cal05-08selection,attackers restricted to corresponding4;minHTER/AUC/smallerweight/fewersteps;zeroallowed;before Testpersonaltraining'));print('ABCS_CYCLE_LOCKED',selected,flush=True)
    choice=json.loads(lock.read_text())['selected'];allix=np.arange(len(x));sharedpath=out/'shared_features_private.npz';randompath=out/'random_features_private.npz'
    if sharedpath.exists():sharedraw=dict(np.load(sharedpath))
    else:sharedraw=features(g,f,data['a'],allix);np.savez_compressed(sharedpath,**sharedraw)
    sharedz={k:norm(sc[k].transform(v)) for k,v in sharedraw.items()};initial=torch.load(out/'initial_generators_private.pt',map_location='cpu',weights_only=False);rg=Generator().cuda();rf=Generator().cuda();rg.load_state_dict(initial['G']);rf.load_state_dict(initial['F'])
    if randompath.exists():randomraw=dict(np.load(randompath))
    else:randomraw=features(rg,rf,data['a'],allix);np.savez_compressed(randompath,**randomraw)
    randomz={k:norm(StandardScaler().fit(v[fit]).transform(v)) for k,v in randomraw.items()};rows={}
    for u in CAL+TEST:
        base,emb,z=cache[u] if u in CAL else make(u);group=CAL if u in CAL else TEST;variants=dict(shape40=shape,statistics240=stats,student192=emb,current_supplement=base,cycle_only100=z[100]['cycle_encoder256'])
        for step in [25,100]:
            for short,k in [('input','input_encoder256'),('cycle','cycle_encoder256'),('statistics','cycle_statistics240')]:variants[f'personal{step}_{short}_supplement']=join(base,z[step][k],.5)
        for name,zz in [('shared',sharedz),('random',randomz)]:
            for short,k in [('input','input_encoder256'),('cycle','cycle_encoder256')]:variants[f'{name}_{short}_supplement']=join(base,zz[k],.5)
        if choice['step']==0:variants.update(selected_cycle_supplement=base,matched_shared_supplement=base,matched_random_supplement=base)
        else:
            step,k,alpha=choice['step'],choice['branch'],choice['alpha'];variants.update(selected_cycle_supplement=join(base,z[step][k],alpha),matched_shared_supplement=join(base,sharedz[k],alpha),matched_random_supplement=join(base,randomz[k],alpha))
        rows[u]={name:score(v,d,u,group) for name,v in variants.items()};dump(out/f'owner{u}'/'scores_private.json',rows[u]);print('ABCS_OPEN_AUTH_OWNER',u,flush=True)
    dump(out/'scores_private.json',{str(u):v for u,v in rows.items()});evaluate(rows,out,'ABCS43 open set: Base23 shared CycleGAN pretrain;own12paired registration-only personalG/F25/100updates;existing shape40+V4student192 fixed;no queryAudio;Cal4/4selection;all11unseenattackers;fixedshared/randomcontrols predeclared')
    p=out/'results_public.json';r=json.loads(p.read_text());r['channels']='ch0 provisional vibration,ch1 provisional Audio;physical mapping unresolved';r['limits']+=['full stereo WAV read for CRC/ACC;query ch1 not transformed or forwarded','fixed current232 reference unchanged;teacher Base replay content differs from owner registration content','previously explored Test cohort,not sealed confirmatory'];dump(p,r);print('ABCS_OPEN_CYCLE_COMPLETE',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--out',required=True);a=p.parse_args();torch.set_num_threads(1);torch.backends.cudnn.benchmark=True;source=Path(a.source);out=Path(a.out);out.mkdir(parents=True,exist_ok=True);d=dict(np.load(source/'inputs_private.npz'));data=dict(np.load(out/'spectra_private.npz'));assert np.array_equal(data['row_ids'],np.arange(len(d['x'])));assert np.all((d['user'][data['fit']]<23)&(d['role'][data['fit']]=='fit'));g,f,da,db=train(data,out);authenticate(source,out,d,data,g,f,da,db)
