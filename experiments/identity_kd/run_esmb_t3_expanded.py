"""Expanded existing ESMB:7Base/2Cal/5unseen targets, same T0/T3 controls."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import argparse,json,time,hashlib,gc,ctypes
from pathlib import Path
import numpy as np
from scipy.io import wavfile
from scipy import ndimage
from scipy.interpolate import interp1d
import torch
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score
import run_esmb_t3_joint as r

ROOT=Path('/autodl-fs/data/esmb_t3_expanded_20261109')
BASE=['0005','0006','0007','0021','0022','0023','0003'];CAL=['0026','0027'];TARGET=['0001','0002','0004','0024','0025'];USERS=BASE+CAL+TARGET
METHODS=r.METHODS;NB=len(BASE);NC=len(CAL);NU=len(USERS)
def source(name):
    if name in ['0001','0002']:return Path('/autodl-fs/data/esmb_baseline_20261020/two_speakers')/(name+'_4.wav')
    if name in ['0003','0004']:return Path('/autodl-fs/data/esmb_baseline_20261020/attack_speakers')/(name+('_12.wav' if name=='0003' else '_4.wav'))
    return r.RAW/(name+'_4.wav')

def prepare():
    cache=ROOT/'inputs_private.npz'
    if cache.exists():return {k:v for k,v in np.load(cache).items()}
    diagnostic=json.loads(Path('/autodl-fs/data/esmb_observable_audio_20261109/summary_public.json').read_text())
    selected=np.flatnonzero(np.any(np.isclose(r.CENTERS[:,None],diagnostic['observable_band_centers'],rtol=1e-6),axis=1));assert len(selected)==17
    x=[];y=[];user=[];index=[];manifest=[];hashes={};duplicates=[]
    for uid,name in enumerate(USERS,1):
        path=source(name);sr,w=wavfile.read(path,mmap=True);assert sr==16000 and w.shape[1]==2 and len(w)>=719*16000,(name,w.shape)
        manifest.append(dict(user=name,file=str(path),sample_rate=sr,duration=len(w)/sr,role='Base' if uid<=NB else 'Cal' if uid<=NB+NC else 'Target'))
        for j in range(240):
            bc=np.array(w[j*3*16000:j*3*16000+32000,1]);assert len(bc)==32000
            key=hashlib.sha256(bc.tobytes()).hexdigest()
            if key in hashes:duplicates.append([hashes[key],[uid,j]])
            else:hashes[key]=[uid,j]
            power,_=r.spectrum(bc);freq=interp1d(r.BINS,np.log(power+1e-12),axis=1)(np.geomspace(50,1200,80)).T
            x.append(ndimage.zoom(freq,(1,80/freq.shape[1]),order=1).astype('float32'))
            target=np.zeros((18,80),dtype='float32')
            if uid<=NB and j<200:
                pp,env=r.spectrum(np.array(w[j*3*16000:j*3*16000+32000,0]));bands=np.stack([np.log(pp[:,(r.BINS>=a)&(r.BINS<b)].mean(1)+1e-12) for a,b in zip(r.EDGES[:-1],r.EDGES[1:])],1);bands-=bands.mean(1,keepdims=True);yy=np.column_stack([bands[:,selected],env]).T;target=ndimage.zoom(yy,(1,80/yy.shape[1]),order=1).astype('float32')
            y.append(target);user.append(uid);index.append(j)
        r.write(ROOT/'status.json',dict(state='preparing',completed_users=uid,total_users=NU));print('PREPARED',name,flush=True)
    d=dict(x=np.stack(x),audio=np.stack(y),user=np.array(user),index=np.array(index));fit=(d['user']<=NB)&(d['index']<160);allowed=(d['user']<=NB)&(d['index']<200)
    lo,hi=np.percentile(d['x'][fit],[1,99]);d['x']=np.clip(2*(d['x']-lo)/(hi-lo)-1,-1,1).astype('float32');mean=d['audio'][fit].mean((0,2),keepdims=True);std=np.maximum(d['audio'][fit].std((0,2),keepdims=True),1e-4);d['audio'][allowed]=(d['audio'][allowed]-mean)/std;assert np.all(d['audio'][~allowed]==0)
    r.write(ROOT/'preprocessing_private.json',dict(manifest=manifest,input_low=float(lo),input_high=float(hi),audio_mean=mean.ravel().tolist(),audio_std=std.ravel().tolist(),selected_centers=r.CENTERS[selected].tolist(),exact_duplicate_pairs=duplicates,canonical_files='one file per identity;copies not counted as new identities;0003 uses first archived12suffix,not chosen by auth results'))
    np.savez_compressed(cache,**d);return d

def train(d,method,steps):
    folder=ROOT/method;folder.mkdir(exist_ok=True)
    if (folder/'complete.json').exists():return
    torch.manual_seed(42);net=r.Model(NB);init=hashlib.sha256(b''.join(t.detach().numpy().tobytes() for t in net.state_dict().values())).hexdigest();opt=torch.optim.AdamW(net.parameters(),lr=1e-4,weight_decay=1e-4);rng=np.random.RandomState(42)
    fit=np.flatnonzero((d['user']<=NB)&(d['index']<160));dev=np.flatnonzero((d['user']<=NB)&(d['index']>=160)&(d['index']<200));lookup={(int(u),int(j)):i for i,(u,j) in enumerate(zip(d['user'],d['index']))};history=[];best=-1;gradient=None;started=time.time()
    for step in range(1,steps+1):
        net.train();ix=np.concatenate([rng.choice(fit[d['user'][fit]==u],2,replace=False) for u in range(1,NB+1)]);yy=torch.tensor(d['user'][ix]-1);target=torch.tensor(d['audio'][ix].copy())
        if method=='T3_wrong_time':target=torch.roll(target,14,-1)
        if method=='T3_wrong_user':target=torch.tensor(d['audio'][[lookup[int(d['user'][i])%NB+1,int(d['index'][i])] for i in ix]])
        z,logits,pred=net(torch.tensor(d['x'][ix]));identity=F.cross_entropy(logits,yy)+.1*r.supcon(z,yy);spec=F.mse_loss(pred[:,:17],target[:,:17]);env=F.mse_loss(pred[:,17:],target[:,17:]);aux=.1*(spec+env);loss=identity if method=='T0_matched' else identity+aux
        if step==1 and method!='T0_matched':gradient=float(torch.autograd.grad(aux,net.body.conv1.weight,retain_graph=True)[0].norm());assert gradient>0
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
        if step==1 or step%5==0:
            r.write(ROOT/'status.json',dict(state='training',method=method,step=step,steps=steps,identity_loss=float(identity.detach()),spectrum_loss=float(spec.detach()),envelope_loss=float(env.detach()),aux_to_encoder_gradient_norm=gradient,elapsed_seconds=time.time()-started));print('TRAIN',method,step,float(loss.detach()),flush=True)
        if step%50==0 or step==steps:
            zz=r.encode(net,d['x'][dev]);auc=r.pair_auc(zz,d['user'][dev],d['index'][dev]);history.append(dict(step=step,base_dev_auc=auc,identity_loss=float(identity.detach()),spectrum_loss=float(spec.detach()),envelope_loss=float(env.detach())))
            if auc>best:best=auc;torch.save(dict(state=net.state_dict(),step=step,dev_auc=auc),folder/'best_private.pt')
            r.write(folder/'history.json',dict(init_sha256=init,history=history,aux_to_encoder_gradient_norm=gradient));torch.save(dict(model=net.state_dict(),optimizer=opt.state_dict(),step=step),folder/'last_private.pt')
    del net,opt;gc.collect();ctypes.CDLL('libc.so.6').malloc_trim(0)
    net=r.Model(NB);ck=torch.load(folder/'best_private.pt',map_location='cpu',weights_only=True);net.load_state_dict(ck['state']);np.save(folder/'embeddings_private.npy',r.encode(net,d['x']));r.write(folder/'complete.json',dict(method=method,steps=steps,selected_step=ck['step'],base_dev_auc=ck['dev_auc'],elapsed_seconds=time.time()-started,aux_to_encoder_gradient_norm=gradient));del net,ck;gc.collect();ctypes.CDLL('libc.so.6').malloc_trim(0);print('MODEL_COMPLETE',method,flush=True)

def evaluate(d):
    cal=list(range(NB+1,NB+NC+1));targets=list(range(NB+NC+1,NU+1));result={}
    for method in METHODS:
        z=np.load(ROOT/method/'embeddings_private.npy');protos={}
        for u in cal+targets:
            en=np.flatnonzero((d['user']==u)&(d['index']<120));assert len(en)==120;v=z[en].mean(0);protos[u]=v/max(np.linalg.norm(v),1e-12)
        def scores(u,attacks):
            ix=np.flatnonzero(np.isin(d['user'],[u]+attacks)&(d['index']>=200));return z[ix]@protos[u],d['user'][ix]==u,d['user'][ix]
        cs=[];cy=[]
        for u in cal:
            s,y,_=scores(u,[v for v in cal if v!=u]);cs.extend(s);cy.extend(y)
        cs=np.array(cs);cy=np.array(cy);result[method]={}
        for rule in ['eer','cal_far05']:
            if rule=='eer':tau=r.tau_eer(cs[cy],cs[~cy])
            else:tau=float(min(t for t in np.r_[np.unique(cs[~cy]),np.nextafter(cs[~cy].max(),np.inf)] if (cs[~cy]>=t).mean()<=.05))
            entry=dict(threshold=tau,cal_tar=float((cs[cy]>=tau).mean()),cal_far=float((cs[~cy]>=tau).mean()),targets={})
            for u in targets:
                entry['targets'][USERS[u-1]]={}
                for scope,attacks in [('unseen_attackers',[v for v in targets if v!=u]),('all_other_sources',[v for v in range(1,NU+1) if v!=u])]:
                    s,y,who=scores(u,attacks);tar=float((s[y]>=tau).mean());far=float((s[~y]>=tau).mean());entry['targets'][USERS[u-1]][scope]=dict(tar=tar,far=far,hter=(1-tar+far)/2,auc=float(roc_auc_score(y,s)),genuine_n=int(y.sum()),impostor_n=int((~y).sum()),per_attacker_far={USERS[v-1]:float((s[(who==v)&~y]>=tau).mean()) for v in attacks})
            result[method][rule]=entry
    r.write(ROOT/'results_private.json',result)

def main(a):
    ROOT.mkdir(exist_ok=True);torch.set_num_threads(1)
    r.write(ROOT/'protocol.json',dict(base=BASE,cal=CAL,targets=TARGET,users=USERS,methods=METHODS,steps=a.steps,batch=2*NB,seed=42,fit='160/Base=1120total;dev40/Base=280total',registration='120BC-only pertarget',query='index200..239 allusers;40genuine/160unseenattacks/520allsourceattacks pertarget;20directed targetattackpairs',physical_teacher='same17bandsandlogRMS;only7Basefirst200Audio;noCal/targetAudio;old4frequencyselectionBaseusersaresubsetofnewBase',frontend='same50..1200Hz80x80aspreviousT3;Basefitonlyscales;alluserssame',model='sameResNet18/192/18channelaux;identityclassifier7;allencoderjointlytrained',loss='CE+.1SupCon+.1sharedspecMSE+.1envMSE',selection='BaseDevidentityAUC every50;CalEERandCalFAR05 predeclared;no finaltestselection',budget_rationale='400updates*2windows/person/160records=5draws/record;previous200*2/80=5',source='14existingidentities,onecanonicalrecordingeach;0003_12 different suffix disclosed;not new blind data',limits='only2Calidentities;single seed;no verifiedtext;newqueries/roles/budget differ,do notattributeold/newdifferenceonlytoBasecount'))
    d=prepare();gc.collect();ctypes.CDLL('libc.so.6').malloc_trim(0)
    for method in METHODS:train(d,method,a.steps)
    evaluate(d);r.write(ROOT/'status.json',dict(state='complete',methods=METHODS,steps=a.steps));print('ALL_COMPLETE',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--steps',type=int,default=400);args=ap.parse_args()
    try:main(args)
    except Exception as error:
        ROOT.mkdir(exist_ok=True);r.write(ROOT/'error.json',dict(error=repr(error)));raise