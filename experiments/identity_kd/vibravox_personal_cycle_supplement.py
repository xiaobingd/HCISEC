"""Registration-only personal G/F fine-tuning, on top of Base-only CycleGAN.

Each owner resets shared G/F and uses ONLY their own 12 paired registrations.
Fixed shared Base-fit feature scalers avoid target-dependent or query scaling.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import argparse,json,copy,hashlib
from pathlib import Path
import numpy as np
from scipy import signal
import joblib,torch
from torch import nn
from torch.nn import functional as F
from sklearn.preprocessing import StandardScaler
from vibravox_cycle_supplement import Generator,Discriminator,features,scores,evaluate,dump,BASE,CAL,TEST
from run_rkd01_vibravox import Student,encode,norm,metrics,thresholds

def registration_audio(source,paired,shared,out):
    path=out/'registration_audio_spectra_private.npz'
    if path.exists():return dict(np.load(path))
    d=dict(np.load(source/'inputs_private.npz'));records=json.loads((source/'manifest_private.json').read_text())['records'];data=dict(np.load(shared/'spectra_private.npz'));am,ast,bm,bst=data['normalization'];ix=np.flatnonzero(np.isin(d['user'],CAL+TEST)&(d['role']=='registration'));assert len(ix)==288;bb=[]
    for i in ix:
        r=records[i]
        with np.load(paired/'paired'/r['file']) as z:w=z['audio'].astype('float32')
        assert hashlib.sha256(w.tobytes()).hexdigest()==r['audio_sha256'];w=w.astype('float64');j=(len(w)-32000)//2;w=w[j:j+32000].copy();w-=w.mean();freq,_,zz=signal.stft(w,16000,nperseg=4096,noverlap=3072,nfft=16384,boundary=None,padded=False);lp=np.log(abs(zz)**2+1e-20);b=np.stack([np.interp(np.geomspace(50,7600,40),freq,lp[:,j]) for j in range(lp.shape[1])],1);b=np.stack([np.interp(np.linspace(0,b.shape[1]-1,32),np.arange(b.shape[1]),row) for row in b]);bb.append(np.clip((b-bm)/(3*bst),-1,1).astype('float32'))
    q=dict(indices=ix,b=np.stack(bb)[:,None]);np.savez_compressed(path,**q);dump(out/'preparation_public.json',dict(registration_audio_records=288,owners=24,each_owner=12,query_audio_reads=0,normalization='same shared Base-fit modality constants',all_other_users_excluded_from_personal_gradient=True));return q

def personal(u,d,data,regaudio,base,discriminators,out):
    folder=out/f'owner{u}';folder.mkdir(parents=True,exist_ok=True);en=np.flatnonzero((d['user']==u)&(d['role']=='registration'));group=CAL if u in CAL else TEST;query=np.flatnonzero(np.isin(d['user'],group)&(d['role']=='query'));ii=np.r_[en,query];assert len(en)==12 and len(query)==288;allout={};g=Generator().cuda();f=Generator().cuda();g.load_state_dict(base['G']);f.load_state_dict(base['F']);assert all(torch.equal(v,base['G'][k].cuda()) for k,v in g.state_dict().items());torch.manual_seed(9900+u);rng=np.random.RandomState(9900+u);og=torch.optim.Adam(list(g.parameters())+list(f.parameters()),lr=2e-5,betas=(.5,.999));a=torch.tensor(data['a'][en],device='cuda');slots={int(i):j for j,i in enumerate(regaudio['indices'])};b=torch.tensor(regaudio['b'][[slots[int(i)] for i in en]],device='cuda');da,db=discriminators
    if all((folder/f'features{step}_private.npz').exists() for step in [25,100]):
        return {step:dict(np.load(folder/f'features{step}_private.npz')) for step in [25,100]},ii
    history=[];initial={k:v.detach().cpu().clone() for k,v in g.state_dict().items()}
    for step in range(1,101):
        jj=rng.choice(12,6,replace=False);x=a[jj];y=b[jj];fb=g(x);fa=f(y);ca=f(fb);cb=g(fa);adv=((db(fb)-1)**2).mean()+((da(fa)-1)**2).mean();pair=F.l1_loss(fb,y)+F.l1_loss(fa,x);cycle=F.l1_loss(ca,x)+F.l1_loss(cb,y);identity=F.l1_loss(g(y),y)+F.l1_loss(f(x),x);loss=adv+10*pair+10*cycle+identity;assert torch.isfinite(loss);og.zero_grad();loss.backward();nn.utils.clip_grad_norm_(list(g.parameters())+list(f.parameters()),5);og.step()
        if step in [1,25,100]:history.append(dict(step=step,loss=float(loss.detach()),pair=float(pair.detach()),cycle=float(cycle.detach())))
        if step in [25,100]:
            torch.save(dict(G=g.state_dict(),F=f.state_dict(),owner=u,registration_indices=en,steps=step),folder/f'generators{step}_private.pt');z=features(g,f,data['a'],ii);np.savez_compressed(folder/f'features{step}_private.npz',**z);allout[step]=z;g.train();f.train()
    delta=max(float((v.detach().cpu()-initial[k]).abs().max()) for k,v in g.state_dict().items());assert delta>0;dump(folder/'training_audit_private.json',dict(owner=u,train_indices=en.tolist(),query_indices=query.tolist(),registration_count=12,query_audio_reads=0,other_user_train_records=0,steps=100,learning_rate=2e-5,batch=6,reset_from_shared=True,updated_g_maxdelta=delta,discriminators='frozen shared Base-only final-step D',history=history));print('PERSONAL_GAN_COMPLETE',u,flush=True);return allout,ii

def main(a):
    torch.set_num_threads(2);source=Path(a.source);shared=Path(a.shared);prior=Path(a.rkd02);out=Path(a.out);out.mkdir(parents=True,exist_ok=True);d=dict(np.load(source/'inputs_private.npz'));data=dict(np.load(shared/'spectra_private.npz'));regaudio=registration_audio(source,Path(a.paired),shared,out);base=torch.load(shared/'best_generators_private.pt',map_location='cpu',weights_only=False);ds=torch.load(shared/'resume_private.pt',map_location='cpu',weights_only=False);da=Discriminator().cuda();db=Discriminator().cuda();da.load_state_dict(ds['DA']);db.load_state_dict(ds['DB']);del ds
    for model in [da,db]:
        model.eval()
        for p in model.parameters():p.requires_grad_(False)
    sc=joblib.load(shared/'feature_scalers_private.joblib');fit=data['fit'];x=StandardScaler().fit(d['x'][fit]).transform(d['x']).astype('float32');shape=norm(x[:,:40]);sharedraw=dict(np.load(shared/'gan_features_private.npz'));sharedz={k:norm(sc[k].transform(v)) for k,v in sharedraw.items()};randomraw=dict(np.load(shared/'random_features_private.npz'));randomz={k:norm(StandardScaler().fit(v[fit]).transform(v)) for k,v in randomraw.items()};sharedscores=json.loads((shared/'scores_private.json').read_text());calcache={};rows={}
    def make(u):
        z,ii=personal(u,d,data,regaudio,base,(da,db),out);net=Student();net.load_state_dict(torch.load(prior/'outer'/f'owner{u}'/'V4_rho0.3_student_private.pt',map_location='cpu',weights_only=False));emb=encode(net,x[ii]);b=np.zeros((len(x),232),dtype='float32');b[ii]=np.concatenate([shape[ii],emb],1)/np.sqrt(2);zz={}
        for step,zstep in z.items():
            zz[step]={}
            for branch,v in zstep.items():q=np.zeros((len(x),v.shape[1]),dtype='float32');q[ii]=norm(sc[branch].transform(v));zz[step][branch]=q
        return b,zz
    for u in CAL:calcache[u]=make(u)
    lock=out/'selection_public.json'
    if not lock.exists():
        choices=[dict(step=0,branch='none',alpha=0.)]
        for step in [25,100]:
            for branch in sc:
                for alpha in [.25,.5,1.]:choices.append(dict(step=step,branch=branch,alpha=alpha))
        for c in choices:
            def collect(group):
                rr=[]
                for u in group:
                    b,zz=calcache[u];z=b if c['step']==0 else np.concatenate([b,c['alpha']*zz[c['step']][c['branch']]],1)/np.sqrt(1+c['alpha']**2);rr.append(scores(z,d,u,group))
                return np.concatenate([v['scores'] for v in rr]),np.concatenate([v['labels'] for v in rr])
            cs,cy=collect(CAL[:6]);qs,qy=collect(CAL[6:]);r=metrics(qs,qy,thresholds(cs,cy)['Cal_EER']);c.update({k:r[k] for k in ['hter','auc','tar','far']})
        chosen=min(choices,key=lambda c:(c['hter'],-c['auc'],c['alpha'],c['step'],c['branch']));dump(lock,dict(selected=chosen,choices=choices,rule='only Cal01-06 threshold,Cal07-12 selection;minHTER/AUC/smallerweight/fewersteps;zeroallowed;before personal Test models trained'));print('PERSONAL_SELECTION_LOCKED',chosen,flush=True)
    chosen=json.loads(lock.read_text())['selected']
    for u in CAL+TEST:
        b,zz=calcache[u] if u in CAL else make(u);group=CAL if u in CAL else TEST;variants={'shape40':shape,'baseline232':b}
        for step in [25,100]:
            for branch in ['input_encoder256','cycle_encoder256','cycle_statistics240']:
                variants[f'personal{step}_{branch}']=np.concatenate([b,.5*zz[step][branch]],1)/np.sqrt(1.25)
        if chosen['step']==0:variants.update(personal_selected=b,matched_shared_branch=b,matched_random_branch=b)
        else:
            step=chosen['step'];branch=chosen['branch'];alpha=chosen['alpha'];variants.update(personal_selected=np.concatenate([b,alpha*zz[step][branch]],1)/np.sqrt(1+alpha*alpha),matched_shared_branch=np.concatenate([b,alpha*sharedz[branch]],1)/np.sqrt(1+alpha*alpha),matched_random_branch=np.concatenate([b,alpha*randomz[branch]],1)/np.sqrt(1+alpha*alpha))
        rows[u]={k:scores(v,d,u,group) for k,v in variants.items()};rows[u]['shared_selected']=sharedscores[str(u)]['selected_cycle'];dump(out/f'owner{u}'/'scores_private.json',rows[u]);print('PERSONAL_AUTH_OWNER',u,flush=True)
    dump(out/'scores_private.json',{str(k):v for k,v in rows.items()});evaluate(d,rows,out);p=out/'results_public.json';r=json.loads(p.read_text());r['scope']='Registration-only personalized G/F;each12ownAudio/vibrationpairs,25/100updates,frozen shared discriminators;sharedG/F pretrained35Base;existing shape40+V4student192 preserved;no query Audio';r['limits'].append('personalization may imprint owner style onto attacker outputs;fixed branch and shared/random controls included');dump(p,r);dump(out/'status_public.json',dict(state='complete',personal_models=24,checkpoints_per_owner=2));print('PERSONAL_ALL_COMPLETE',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--paired',required=True);p.add_argument('--shared',required=True);p.add_argument('--rkd02',required=True);p.add_argument('--out',required=True);main(p.parse_args())
