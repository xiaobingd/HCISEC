"""F0/voicing first, deterministic harmonic scaffold second. Own DDSP-inspired pilot."""
import os,json,copy
from pathlib import Path
import numpy as np
from scipy import signal
import librosa
import torch
from torch import nn
from torch.nn import functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent;torch.set_num_threads(2);device='cuda'
grid=np.arange(60.,401.,2.);cents_grid=1200*np.log2(grid/60)
D={};cache=ROOT/'pitch_labels.npz'
labels=dict(np.load(cache)) if cache.exists() else {}
for s in ('train','val'):
    a=dict(np.load(ROOT.parent/'codec_adapter_20261017'/('pairs_'+s+'.npz')))
    ax=dict(np.load(ROOT/('axes_'+s+'.npz')));assert np.array_equal(ax['ids'],a['ids'])
    if s+'_f0' not in labels:
        fs=[];vs=[];probs=[]
        for i,w in enumerate(a['audio16']):
            f,v,p=librosa.pyin(w,sr=16000,fmin=60,fmax=400,frame_length=1024,hop_length=160,center=True,pad_mode='constant')
            fs.append(f);vs.append(v);probs.append(p)
        labels[s+'_f0']=np.array(fs);labels[s+'_voice']=np.array(vs);labels[s+'_prob']=np.array(probs)
        np.savez_compressed(cache,**labels)
    a.update(f0=labels[s+'_f0'],voice=labels[s+'_voice'],prob=labels[s+'_prob'])
    _,_,az=signal.stft(a['audio16'],fs=16000,nperseg=1024,noverlap=864)
    a['audio_abs']=abs(az);a['rms']=np.sqrt(np.mean(abs(az)**2,axis=1)).clip(1e-7)
    ff,_,z=signal.stft(ax['axes'],fs=1000,nperseg=128,noverlap=118,nfft=1024,axis=-1)
    power=np.sqrt(np.mean(abs(z)**2,axis=1));logmag=np.log(abs(z)+1e-12)
    features=[];contrasts=[]
    contrast=logmag-signal.savgol_filter(logmag,101,2,axis=2)
    for h in (1,2):
        for axis in range(3):
            features.append(np.stack([np.stack([np.interp(grid*h,ff,logmag[i,axis,:,t],right=-28) for t in range(81)],1) for i in range(len(z))]))
            contrasts.append(np.stack([np.stack([np.interp(grid*h,ff,contrast[i,axis,:,t],right=0) for t in range(81)],1) for i in range(len(z))]))
    a['salience_input']=np.stack(features,1)
    a['contrast_input']=np.stack(contrasts,1)
    a['physical_score']=np.stack([np.stack([np.interp(grid,ff,np.log(power[i,:,t]+1e-12)) for t in range(81)],1) for i in range(len(z))])
    assert a['f0'].shape==(len(z),81)
    D[s]=a
assert not(set(D['train']['ids'])&set(D['val']['ids']))
ftrain=D['train']['f0'][D['train']['voice']];median=float(np.median(ftrain));print('pitch targets',median,np.percentile(ftrain,[5,95]),flush=True)
lo,hi=np.percentile(D['train']['salience_input'],[1,99]);rmean=float(np.log(D['train']['rms']).mean());rstd=float(np.log(D['train']['rms']).std())
clo,chi=np.percentile(D['train']['contrast_input'],[1,99])
for s,d in D.items():
    x=np.clip(2*(d['salience_input']-lo)/(hi-lo)-1,-1,1)
    coords=np.broadcast_to((grid/400)[None,None,:,None],(len(x),1,len(grid),81))
    mask=np.broadcast_to((2*grid<=500)[None,None,:,None],coords.shape)
    d['sx']=torch.tensor(np.concatenate([x,coords,mask],1),dtype=torch.float32,device=device)
    cx=np.clip(2*(d['contrast_input']-clo)/(chi-clo)-1,-1,1)
    d['cx']=torch.cat([d['sx'],torch.tensor(cx,dtype=torch.float32,device=device)],1)
    d['ox']=torch.tensor(d['x'],device=device)
    d['target_f0']=torch.tensor(np.nan_to_num(d['f0'],nan=median),dtype=torch.float32,device=device)
    d['v']=torch.tensor(d['voice'],dtype=torch.float32,device=device)
    d['r']=torch.tensor((np.log(d['rms'])-rmean)/rstd,dtype=torch.float32,device=device)

class OriginalPitch(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Conv1d(80,64,5,padding=2),nn.GELU(),nn.Dropout(.1),nn.Conv1d(64,64,5,padding=2),nn.GELU(),nn.Conv1d(64,len(grid)+2,1))
    def forward(self,x):
        y=self.net(x);return y[:,:len(grid)],y[:,len(grid)],y[:,len(grid)+1]
class AxisPitch(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Conv2d(8,16,3,padding=1),nn.GELU(),nn.Conv2d(16,16,3,padding=1),nn.GELU())
        self.pitch=nn.Conv2d(16,1,1);self.head=nn.Sequential(nn.Conv1d(32,32,5,padding=2),nn.GELU(),nn.Conv1d(32,2,1))
    def forward(self,x):
        h=self.net(x);p=self.pitch(h)[:,0];v=self.head(torch.cat([h.mean(2),h.max(2).values],1));return p,v[:,0],v[:,1]
class AxisContrastPitch(AxisPitch):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(14,16,3,padding=1),nn.GELU(),nn.Conv2d(16,16,3,padding=(4,1),dilation=(4,1)),nn.GELU(),nn.Conv2d(16,16,3,padding=(8,1),dilation=(8,1)),nn.GELU())
gt=torch.tensor(grid,dtype=torch.float32,device=device)[None,:,None]
def loss_fn(outputs,d,ix=None):
    p,v,r=outputs;vv=d['v'] if ix is None else d['v'][ix];ff=d['target_f0'] if ix is None else d['target_f0'][ix];rr=d['r'] if ix is None else d['r'][ix]
    soft=torch.exp(-.5*((gt-ff[:,None])/3)**2);soft=soft/soft.sum(1,keepdim=True).clamp_min(1e-8)
    ce=-(soft*F.log_softmax(p,dim=1)).sum(1)
    return 2*(ce*vv).sum()/vv.sum().clamp_min(1)+F.binary_cross_entropy_with_logits(v,vv)+.2*F.mse_loss(r,rr)
def path_decode(logits):
    # Fixed temporal prior, same for all inputs. Compare against constant-pitch control.
    trans=.2*np.minimum(((cents_grid[:,None]-cents_grid[None,:])/80)**2,25)
    paths=[]
    for q in logits:
        q=q-q.max(0,keepdims=True);scores=q[:,0];back=[]
        for t in range(1,q.shape[1]):
            a=scores[:,None]-trans;idx=np.argmax(a,axis=0);back.append(idx);scores=a[idx,np.arange(len(grid))]+q[:,t]
        out=[int(np.argmax(scores))]
        for b in back[::-1]:out.append(int(b[out[-1]]))
        paths.append(grid[out[::-1]])
    return np.array(paths)
def pitch_metrics(f,v):
    d=D['val'];target=d['f0'];mask=d['voice'].copy();mask[:,:4]=False;mask[:,-4:]=False
    err=abs(1200*np.log2(np.maximum(f,1)/np.nan_to_num(target,nan=median)))
    interior=np.ones_like(mask);interior[:,:4]=False;interior[:,-4:]=False
    y=d['voice'][interior];vv=v[interior];tp=np.sum(y&vv);fp=np.sum(~y&vv);fn=np.sum(y&~vv)
    return dict(voicing_f1=float(2*tp/max(2*tp+fp+fn,1)),voiced_f0_median_error_cents=float(np.median(err[mask])),voiced_f0_mean_error_cents=float(err[mask].mean()),within50cents_on_voiced=float(np.mean(err[mask]<50)),within50cents_including_missed_voicing=float(np.mean((err[mask]<50)&v[mask])),frames=int(mask.sum()))

pred={};fit={};old=json.loads((ROOT/'results.json').read_text()) if (ROOT/'results.json').exists() else {}
for route,cls,key in [('original_pitch',OriginalPitch,'ox'),('signed_axes_pitch',AxisPitch,'sx'),('signed_axes_contrast',AxisContrastPitch,'cx')]:
    torch.manual_seed(42);np.random.seed(42);net=cls().to(device);opt=torch.optim.AdamW(net.parameters(),lr=5e-4,weight_decay=.02);best=None;history=[]
    saved=ROOT/(route+'.pt')
    epochs=0 if saved.exists() else 250
    if epochs==0:
        cp=torch.load(saved,map_location=device,weights_only=True);best=(old['fit'][route]['val_loss'],old['fit'][route]['epoch'],cp['net'])
    for epoch in range(1,epochs+1):
        net.train();groups=np.unique(D['train']['ids']);np.random.shuffle(groups)
        for g in np.array_split(groups,4):
            ix=[np.random.choice(np.flatnonzero(D['train']['ids']==r)) for r in g]
            loss=loss_fn(net(D['train'][key][ix]),D['train'],ix);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1);opt.step()
        net.eval()
        with torch.no_grad():
            losses=[]
            for start in range(0,len(D['val']['ids']),4):
                ix=np.arange(start,min(start+4,len(D['val']['ids'])));losses.append(float(loss_fn(net(D['val'][key][ix]),D['val'],ix)))
            val=np.mean(losses)
        history.append(dict(epoch=epoch,val_loss=float(val)))
        if best is None or val<best[0]:best=(val,epoch,copy.deepcopy(net.state_dict()))
        if epoch%50==0:print(route,epoch,val,'best',best[:2],flush=True)
    net.load_state_dict(best[2]);net.eval()
    with torch.no_grad():outs=[net(z) for z in D['val'][key].split(4)]
    p,v,r=[torch.cat([o[i] for o in outs]).cpu().numpy() for i in range(3)]
    pred[route]=(path_decode(p),v>0,np.exp(r*rstd+rmean));fit[route]=dict(epoch=best[1],val_loss=float(best[0]),parameters=sum(p.numel() for p in net.parameters()))
    if epochs:
        torch.save({'net':best[2],'route':route,'scalar_scale':torch.tensor([lo,hi,rmean,rstd,clo,chi])},ROOT/(route+'.pt'))
        (ROOT/(route+'_history.json')).write_text(json.dumps(history))
physical=path_decode(D['val']['physical_score']);voice=pred['signed_axes_pitch'][1];rms=pred['signed_axes_pitch'][2]
pred['physical_acc_pitch']=(physical,voice,rms)
pred['constant_train_pitch']=(np.full_like(physical,median),voice,rms)
pred['oracle_audio_pitch']=(np.nan_to_num(D['val']['f0'],nan=median),D['val']['voice'],D['val']['rms'])
result={'fit':fit,'validation':{k:pitch_metrics(f,v) for k,(f,v,r) in pred.items()},'train_f0_median':median,'label_voiced_frames':{s:int(d['voice'].sum()) for s,d in D.items()},'notes':'Exploratory same-person validation; F0 truth from microphone pYIN, not human labels. First/last40ms excluded from pitch metrics. Constant-F0 and physical-F0 controls share signed-axis student voicing/RMS. No test/reserve read. Harmonic scaffold is not full audio or identity validation.'}
(ROOT/'results.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

# Training-only mean harmonic envelope; no validation timbre supplied to students.
freq=np.fft.rfftfreq(1024,1/16000);hh=np.arange(1,49);profiles=[]
for i in range(len(D['train']['ids'])):
    for t in np.flatnonzero(D['train']['voice'][i]):
        f=D['train']['f0'][i,t];a=np.interp(hh*f,freq,D['train']['audio_abs'][i,:,t],right=0);profiles.append(a/max(np.linalg.norm(a),1e-8))
profile=np.mean(profiles,0);profile/=max(np.linalg.norm(profile),1e-8)
def synth(f,v,r):
    waves=[];n=12800;tt=np.arange(n)/16000;frame_t=np.arange(81)*.01
    for ff,vv,rr in zip(f,v,r):
        p=np.interp(tt,frame_t,ff);phase=2*np.pi*np.cumsum(p)/16000
        # RMS proxy is an STFT-bin RMS: apply one global training-derived factor.
        gain=np.interp(tt,frame_t,rr*vv)
        w=np.zeros(n)
        for h,a in zip(hh,profile):w+=a*np.sin(h*phase)*(h*p<7500)
        waves.append(w*gain)
    return np.array(waves)
import sys
sys.path.insert(0,str(ROOT.parent))
from data0723_frontend_helpers import create_mel_filterbank
fb=create_mel_filterbank(16000,512,80,20,7500)
# One global amplitude calibration from TRAIN references only; not per-record equalization.
target_rms=np.sqrt(np.mean(D['train']['audio16']**2));stft_rms=np.sqrt(np.mean(D['train']['rms']**2));gain=target_rms/max(stft_rms,1e-8)
images={}
for name,(f,v,r) in pred.items():
    w=synth(f,v,r)*gain;_,_,z=signal.stft(w,fs=16000,nperseg=512,noverlap=352)
    a=10*np.log10(np.einsum('mf,nft->nmt',fb,abs(z)**2)+1e-10)
    images[name]=np.clip(2*(a+100)/72.15651143799875-1,-1,1)
real=np.load(ROOT.parent/'data_original/val/imu_audio.npy');ids=D['val']['ids'];indices=[np.flatnonzero(ids==g)[0] for g in np.unique(ids)]
rows=[('Real microphone',real),('ORACLE F0 / voice / RMS + TRAIN mean envelope',images['oracle_audio_pitch']),('Original ACC -> predicted harmonic scaffold',images['original_pitch']),('Signed XYZ -> predicted harmonic scaffold',images['signed_axes_pitch']),('Constant train-median pitch control',images['constant_train_pitch'])]
rows.insert(-1,('Signed XYZ + local peak contrast',images['signed_axes_contrast']))
fig,axes=plt.subplots(len(rows),4,figsize=(12,11),squeeze=False)
for rr,(label,arr) in enumerate(rows):
    for col,i in enumerate(indices):
        axes[rr,col].imshow(arr[i],origin='lower',aspect='auto',cmap='magma',vmin=-1,vmax=1,interpolation='nearest');axes[rr,col].set_title(label if col==0 else str(ids[i]).replace('recording_',''),fontsize=8);axes[rr,col].tick_params(labelsize=7)
fig.suptitle('Explicit pitch/voicing before harmonic synthesis: structural pilot, NOT full speech restoration\nAll four validation recordings, first window each; microphone labels are used ONLY in the ORACLE row.',fontsize=10)
fig.tight_layout(rect=(0,0,1,.95));fig.savefig(ROOT/'harmonic_comparison.png',dpi=105)
fig,axes=plt.subplots(4,1,figsize=(12,9))
for ax,i in zip(axes,indices):
    t=np.arange(81)*.01;mask=D['val']['voice'][i];ax.plot(t,np.where(mask,D['val']['f0'][i],np.nan),'k',label='Microphone pYIN reference',linewidth=2)
    for name in ('original_pitch','signed_axes_pitch','signed_axes_contrast','physical_acc_pitch'):
        f,v,r=pred[name];ax.plot(t,np.where(v[i],f[i],np.nan),label=name,alpha=.8)
    ax.axhline(median,color='gray',linestyle='--',label='train median');ax.set_ylim(60,400);ax.set_title(str(ids[i]));ax.set_ylabel('F0 / Hz')
axes[0].legend(fontsize=8,ncol=3);axes[-1].set_xlabel('Time / s');fig.tight_layout();fig.savefig(ROOT/'f0_tracks.png',dpi=110)
np.savez_compressed(ROOT/'validation_scaffold.npz',real=real,ids=ids,reference_f0=D['val']['f0'],reference_voice=D['val']['voice'],**images,**{k+'_f0':v[0] for k,v in pred.items()})
