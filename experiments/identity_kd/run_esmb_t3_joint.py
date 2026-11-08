"""Joint BC identity + observable Audio spectrum/envelope; matched controls."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import json,time,hashlib,gc,argparse
from pathlib import Path
import numpy as np
from scipy.io import wavfile
from scipy import ndimage
from scipy.interpolate import interp1d
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18
from sklearn.metrics import roc_auc_score

ROOT=Path('/autodl-fs/data/esmb_t3_joint_20261109')
RAW=Path('/autodl-fs/data/esmb_shared_identity_20261020/raw')
USERS=['0005','0006','0007','0021','0022','0023','0026','0027','0024','0025']
METHODS=['T3_paired','T0_matched','T3_wrong_time','T3_wrong_user']
FS=16000;WIN=1024;HOP=256;EDGES=np.geomspace(50,8000,33)
CENTERS=np.sqrt(EDGES[:-1]*EDGES[1:]);BINS=np.fft.rfftfreq(2048,1/FS)

def write(path,obj):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(path)

def spectrum(w):
    w=w.astype('float64')/32768.;w-=w.mean();fr=np.lib.stride_tricks.sliding_window_view(w,WIN)[::HOP].copy();fr-=fr.mean(1,keepdims=True)
    return np.abs(np.fft.rfft(fr*np.hanning(WIN),n=2048,axis=1))**2,np.log(np.sqrt(np.mean(fr**2,1)+1e-12))

def prepare():
    path=ROOT/'inputs_private.npz'
    if path.exists():return {k:v for k,v in np.load(path).items()}
    diagnostic=json.loads(Path('/autodl-fs/data/esmb_observable_audio_20261109/summary_public.json').read_text())
    selected=np.flatnonzero(np.any(np.isclose(CENTERS[:,None],diagnostic['observable_band_centers'],rtol=1e-6),axis=1));assert len(selected)==17
    x=[];targets=[];user=[];index=[];hashes={};dups=[]
    for u,name in enumerate(USERS,1):
        sr,w=wavfile.read(RAW/(name+'_4.wav'),mmap=True);assert sr==FS and w.shape[1]==2
        for j in range(200):
            bc=np.array(w[j*3*FS:j*3*FS+2*FS,1]);assert len(bc)==32000
            h=hashlib.sha256(bc.tobytes()).hexdigest()
            if h in hashes:dups.append([hashes[h],[u,j]])
            else:hashes[h]=[u,j]
            p,_=spectrum(bc);db=np.log(p+1e-12);freq=interp1d(BINS,db,axis=1)(np.geomspace(50,1200,80)).T
            xx=ndimage.zoom(freq,(1,80/freq.shape[1]),order=1).astype('float32');assert xx.shape==(80,80)
            tt=np.zeros((18,80),dtype='float32')
            # Only Base fit/develop Audio is ever accessed for teacher targets.
            if u<=4 and j<100:
                pp,env=spectrum(np.array(w[j*3*FS:j*3*FS+2*FS,0]));bands=np.stack([np.log(pp[:,(BINS>=a)&(BINS<b)].mean(1)+1e-12) for a,b in zip(EDGES[:-1],EDGES[1:])],1);bands-=bands.mean(1,keepdims=True)
                yy=np.column_stack([bands[:,selected],env]).T;tt=ndimage.zoom(yy,(1,80/yy.shape[1]),order=1).astype('float32')
            x.append(xx);targets.append(tt);user.append(u);index.append(j)
        write(ROOT/'status.json',dict(state='preparing',completed_users=u,total_users=10));print('PREPARED',name,flush=True)
    d=dict(x=np.stack(x),audio=np.stack(targets),user=np.array(user),index=np.array(index))
    fit=(d['user']<=4)&(d['index']<80);lo,hi=np.percentile(d['x'][fit],[1,99]);d['x']=np.clip(2*(d['x']-lo)/(hi-lo)-1,-1,1).astype('float32')
    mean=d['audio'][fit].mean((0,2),keepdims=True);std=np.maximum(d['audio'][fit].std((0,2),keepdims=True),1e-4);allowed=(d['user']<=4)&(d['index']<100)
    d['audio'][allowed]=(d['audio'][allowed]-mean)/std;assert np.all(d['audio'][~allowed]==0)
    np.savez_compressed(path,**d);write(ROOT/'preprocessing_private.json',dict(bc_low=float(lo),bc_high=float(hi),audio_mean=mean.ravel().tolist(),audio_std=std.ravel().tolist(),selected_band_centers=CENTERS[selected].tolist(),audio_access='onlyBase first100;no Cal or targetAudio',exact_bc_duplicate_pairs=dups));return d

class Model(nn.Module):
    def __init__(self):
        super().__init__();self.body=resnet18(weights=None);self.body.fc=nn.Linear(512,192);self.classifier=nn.Linear(192,4);self.aux=nn.Sequential(nn.Conv1d(64,64,3,padding=1),nn.GELU(),nn.Conv1d(64,18,1))
    def forward(self,x):
        b=self.body;x=x[:,None].repeat(1,3,1,1);x=b.maxpool(b.relu(b.bn1(b.conv1(x))));early=b.layer1(x)
        prediction=F.interpolate(self.aux(early.mean(2)),size=80,mode='linear',align_corners=False)
        y=b.layer4(b.layer3(b.layer2(early)));z=F.normalize(b.fc(torch.flatten(b.avgpool(y),1)),dim=1)
        return z,self.classifier(z),prediction

def supcon(z,y):
    sim=z@z.T/.1;diag=torch.eye(len(y),dtype=torch.bool);pos=(y[:,None]==y[None,:])&~diag
    lp=sim-torch.logsumexp(sim.masked_fill(diag,-1e9),dim=1,keepdim=True)
    return -lp.masked_fill(~pos,0).sum(1).div(pos.sum(1).clamp_min(1)).mean()

@torch.no_grad()
def encode(net,x):
    net.eval();out=[]
    for k in range(0,len(x),8):out.append(net(torch.tensor(x[k:k+8]))[0].numpy())
    return np.concatenate(out)

def pair_auc(z,u,j):
    sim=z@z.T;upper=np.triu(np.ones(sim.shape,bool),1);block=j//10
    pos=upper&(u[:,None]==u[None,:])&(block[:,None]!=block[None,:]);neg=upper&(u[:,None]!=u[None,:])&(block[:,None]==block[None,:])
    return float(roc_auc_score(np.r_[np.ones(pos.sum()),np.zeros(neg.sum())],np.r_[sim[pos],sim[neg]]))

def train(d,method,steps):
    folder=ROOT/method;folder.mkdir(exist_ok=True)
    if (folder/'complete.json').exists():return
    torch.manual_seed(42);net=Model();init_hash=hashlib.sha256(b''.join(t.detach().numpy().tobytes() for t in net.state_dict().values())).hexdigest()
    opt=torch.optim.AdamW(net.parameters(),lr=1e-4,weight_decay=1e-4);rng=np.random.RandomState(42)
    fit=np.flatnonzero((d['user']<=4)&(d['index']<80));dev=np.flatnonzero((d['user']<=4)&(d['index']>=80)&(d['index']<100));lookup={(int(u),int(j)):i for i,(u,j) in enumerate(zip(d['user'],d['index']))}
    best=-1;history=[];started=time.time();gradient=None
    for step in range(1,steps+1):
        net.train();ix=np.concatenate([rng.choice(fit[d['user'][fit]==u],2,replace=False) for u in range(1,5)]);yy=torch.tensor(d['user'][ix]-1);target=torch.tensor(d['audio'][ix].copy())
        if method=='T3_wrong_time':target=torch.roll(target,14,dims=-1)
        if method=='T3_wrong_user':target=torch.tensor(d['audio'][[lookup[(int(d['user'][i])%4+1,int(d['index'][i]))] for i in ix]])
        z,logits,pred=net(torch.tensor(d['x'][ix]));identity=F.cross_entropy(logits,yy)+.1*supcon(z,yy);spec=F.mse_loss(pred[:,:17],target[:,:17]);env=F.mse_loss(pred[:,17:],target[:,17:]);aux=.1*spec+.1*env
        loss=identity if method=='T0_matched' else identity+aux
        if step==1 and method!='T0_matched':
            gradient=float(torch.autograd.grad(aux,net.body.conv1.weight,retain_graph=True)[0].norm());assert gradient>0
        opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
        if step==1 or step%5==0:
            write(ROOT/'status.json',dict(state='training',method=method,step=step,steps=steps,identity_loss=float(identity.detach()),spectrum_loss=float(spec.detach()),envelope_loss=float(env.detach()),aux_to_encoder_gradient_norm=gradient,elapsed_seconds=time.time()-started))
            print('TRAIN',method,step,float(loss.detach()),flush=True)
        if step%50==0 or step==steps:
            dz=encode(net,d['x'][dev]);auc=pair_auc(dz,d['user'][dev],d['index'][dev]);row=dict(step=step,base_dev_identity_auc=auc,identity_loss=float(identity.detach()),spec_loss=float(spec.detach()),env_loss=float(env.detach()));history.append(row)
            if auc>best:best=auc;torch.save(dict(state=net.state_dict(),step=step,dev_auc=auc),folder/'best_private.pt')
            write(folder/'history.json',dict(method=method,seed=42,init_sha256=init_hash,history=history,aux_to_encoder_gradient_norm=gradient))
            torch.save(dict(model=net.state_dict(),optimizer=opt.state_dict(),step=step),folder/'last_private.pt')
    del opt,net;gc.collect();net=Model();ck=torch.load(folder/'best_private.pt',map_location='cpu',weights_only=True);net.load_state_dict(ck['state']);z=encode(net,d['x']);np.save(folder/'embeddings_private.npy',z)
    write(folder/'complete.json',dict(method=method,steps=steps,selected_step=ck['step'],base_dev_auc=ck['dev_auc'],aux_to_encoder_gradient_norm=gradient,elapsed_seconds=time.time()-started));print('MODEL_COMPLETE',method,flush=True)

def tau_eer(g,n):
    vals=np.unique(np.r_[g,n]);vals=np.r_[vals,np.nextafter(vals[-1],np.inf)]
    return float(min(vals,key=lambda t:(abs((g<t).mean()-(n>=t).mean()),(g<t).mean()+(n>=t).mean(),-t)))

def evaluate(d):
    result={}
    for method in METHODS:
        z=np.load(ROOT/method/'embeddings_private.npy');protos={}
        for u in range(5,11):
            en=np.flatnonzero((d['user']==u)&(d['index']<120));v=z[en].mean(0);protos[u]=v/max(np.linalg.norm(v),1e-12)
        def scores(u,attacks):
            ix=np.flatnonzero(np.isin(d['user'],[u]+attacks)&(d['index']>=160));return z[ix]@protos[u],d['user'][ix]==u
        cs=[];cy=[]
        for u in range(5,9):
            s,y=scores(u,[v for v in range(5,9) if v!=u]);cs.extend(s);cy.extend(y)
        cs=np.array(cs);cy=np.array(cy);result[method]={}
        for rule in ['eer','cal_far05']:
            if rule=='eer':tau=tau_eer(cs[cy],cs[~cy])
            else:
                options=np.r_[np.unique(cs[~cy]),np.nextafter(cs[~cy].max(),np.inf)];tau=float(min(t for t in options if (cs[~cy]>=t).mean()<=.05))
            entry=dict(threshold=tau,cal_tar=float((cs[cy]>=tau).mean()),cal_far=float((cs[~cy]>=tau).mean()),targets={})
            for u in [9,10]:
                entry['targets'][USERS[u-1]]={}
                for scope,attacks in [('unseen_attacker',[19-u]),('all_nine_sources',[v for v in range(1,11) if v!=u])]:
                    s,y=scores(u,attacks);tar=float((s[y]>=tau).mean());far=float((s[~y]>=tau).mean());entry['targets'][USERS[u-1]][scope]=dict(tar=tar,far=far,hter=(1-tar+far)/2,auc=float(roc_auc_score(y,s)),genuine_n=int(y.sum()),impostor_n=int((~y).sum()))
            result[method][rule]=entry
    write(ROOT/'results_private.json',result)

def main(a):
    ROOT.mkdir(exist_ok=True);torch.set_num_threads(1)
    write(ROOT/'protocol.json',dict(methods=METHODS,base=USERS[:4],cal=USERS[4:8],targets=USERS[8:],frontend='BC logpower64ms/16ms,80logfreq50..1200Hz,80time;Basefitglobal1/99percentile;alluserssame',teacher='17Base-selected Audio relative logbands centers74..940Hz+logRMS;AudioBasefirst100only',training='ResNet18fromscratch192;auxheadfromlayer1 frequencymean64x20->18x80;allBCencoderparameters+heads jointlytrained;CE+.1SupCon+.1specMSE+.1envMSE',budget=dict(seed=42,steps=a.steps,batch=8,optimizer='AdamW1e-4weightdecay1e-4',selection='BaseDevidentityAUC every50;earlier ties'),controls='same initialization/sampling/input/budget;T0auxweight0;wrongtimecircularshift14of80frames;wrongusersamewindowindex',backend='120BC-onlyregistrationmeanprototype;40query/user;Cal4 mutual attacks;EERandCalFAR05 fixedrules',limits='exploratory2target identities previouslyevaluated;single recording;no verifiedtext;not same/crosswordclaim;newfrontT0mustnotreplaceoldmodelasmatchedcontrol'))
    d=prepare()
    for method in METHODS:train(d,method,a.steps)
    evaluate(d);write(ROOT/'status.json',dict(state='complete',methods=METHODS,steps=a.steps));print('ALL_COMPLETE',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--steps',type=int,default=200);main(ap.parse_args())