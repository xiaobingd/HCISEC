"""Own pretrained-codec adaptation pilot, NOT a paper reproduction or ASV result."""
import os,sys,json,copy,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'deps'))
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import MimiModel
from scipy import signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torchaudio

torch.set_num_threads(2);torch.manual_seed(42);np.random.seed(42)
device='cuda';model=MimiModel.from_pretrained(str(ROOT/'mimi'),local_files_only=True,use_safetensors=True,attn_implementation='eager').to(device).eval()
for p in model.parameters():p.requires_grad_(False)
data={s:dict(np.load(ROOT/('pairs_'+s+'.npz'))) for s in ('train','val')}
def decode(z):
    h=model.upsample(z)
    h=model.decoder_transformer(h.transpose(1,2),use_cache=False,return_dict=True).last_hidden_state.transpose(1,2)
    return model.decoder(h)[..., :19200]

targets={};oracle={}
with torch.no_grad():
    for split,d in data.items():
        zs=[];ws=[]
        for a in np.array_split(d['audio24'],max(1,int(np.ceil(len(d['audio24'])/4)))):
            a=torch.tensor(a[:,None],device=device)
            codes=model.encode(a,return_dict=True).audio_codes
            z=model.quantizer.decode(codes)
            w=decode(z)
            official=model.decode(codes,return_dict=True).audio_values[...,:19200]
            assert torch.allclose(w,official,atol=1e-5,rtol=1e-4)
            zs.append(z.cpu());ws.append(w[:,0].cpu().numpy())
        targets[split]=torch.cat(zs).to(device);oracle[split]=np.concatenate(ws)
        print('encoded',split,tuple(targets[split].shape),flush=True)
    torch.save({s:v.cpu() for s,v in targets.items()},ROOT/'targets.pt')

# Training-only latent calibration; no per-record/user gain correction.
center=targets['train'].mean((0,2),keepdim=True)
scale=targets['train'].std((0,2),keepdim=True).clamp_min(.1)
ys={s:(z-center)/scale for s,z in targets.items()}
xs={s:torch.tensor(d['x'],device=device) for s,d in data.items()}
times=targets['train'].shape[-1]

class Adapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Conv1d(80,128,5,padding=2),nn.GELU(),nn.Dropout(.15),nn.Conv1d(128,128,5,padding=2),nn.GELU(),nn.Conv1d(128,512,1))
    def forward(self,x):return F.adaptive_avg_pool1d(self.net(x),times)

def matched_loss(p,y):
    # Negatives are different recordings of SAME PERSON: content alignment only.
    a=F.normalize(p.mean(2),dim=1);b=F.normalize(y.mean(2),dim=1)
    return F.cross_entropy(a@b.T/.15,torch.arange(len(a),device=device))

preds={'codec_oracle':oracle['val']};training={};latent_predictions={}
mean=targets['train'].mean(0,keepdim=True).expand(len(xs['val']),-1,-1)
latent_predictions['mean_latent']=mean
# Ridge uses the same original ACC and local +/-20ms context, regularization selected on validation.
feat={}
for s,x in xs.items():
    q=torch.cat([torch.roll(x,k,2) for k in (-2,0,2)],1)
    feat[s]=F.adaptive_avg_pool1d(q,times).transpose(1,2)
mu=feat['train'].mean((0,1));sd=feat['train'].std((0,1)).clamp_min(.1)
A=((feat['train']-mu)/sd).reshape(-1,240);Y=ys['train'].transpose(1,2).reshape(-1,512)
V=((feat['val']-mu)/sd).reshape(-1,240);ridge_best=None
for alpha in (1.,10.,100.,1000.):
    W=torch.linalg.solve(A.T@A+alpha*torch.eye(240,device=device),A.T@Y)
    z=(V@W).reshape(len(xs['val']),times,512).transpose(1,2)
    loss=float(F.mse_loss(z,ys['val']))
    if ridge_best is None or loss<ridge_best[0]:ridge_best=(loss,alpha,z,W)
latent_predictions['ridge']=ridge_best[2]*scale+center
training['ridge']={'validation_latent_mse':ridge_best[0],'alpha':ridge_best[1]}

for route in ('regression','contrastive'):
    torch.manual_seed(42);np.random.seed(42);net=Adapter().to(device);opt=torch.optim.AdamW(net.parameters(),lr=5e-4,weight_decay=.02)
    best=None;hist=[]
    for epoch in range(1,301):
        net.train()
        # One random window per recording prevents duplicated overlapping recordings dominating batches.
        groups=np.unique(data['train']['ids']);np.random.shuffle(groups)
        for g in np.array_split(groups,3):
            ix=[np.random.choice(np.flatnonzero(data['train']['ids']==v)) for v in g]
            p=net(xs['train'][ix]);y=ys['train'][ix]
            loss=F.mse_loss(p,y)
            if route=='contrastive':loss=loss+.2*matched_loss(p,y)
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1);opt.step()
        net.eval()
        with torch.no_grad():
            p=net(xs['val']);val=float(F.mse_loss(p,ys['val']))
            tr=float(F.mse_loss(net(xs['train']),ys['train']))
        hist.append({'epoch':epoch,'train_latent_mse':tr,'val_latent_mse':val})
        if best is None or val<best[0]:best=(val,epoch,copy.deepcopy(net.state_dict()))
        if epoch%50==0:print(route,epoch,tr,val,'best',best[:2],flush=True)
    net.load_state_dict(best[2]);net.eval()
    with torch.no_grad():latent_predictions[route]=net(xs['val'])*scale+center
    training[route]={'epoch':best[1],'validation_latent_mse':best[0],'parameters':sum(p.numel() for p in net.parameters())}
    torch.save({'adapter':best[2],'center':center.cpu(),'scale':scale.cpu(),'route':route},ROOT/(route+'.pt'))
    (ROOT/(route+'_history.json')).write_text(json.dumps(hist))

# Inspired by 2026 BC key-value memory, using Mimi rather than the paper's SSL features.
# Values contain TRAIN recordings only; leave the entire query recording out during training.
bank=ys['train'].transpose(1,2).reshape(-1,512).detach()
bank_ids=np.repeat(data['train']['ids'],times)
class MemoryAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.query=nn.Sequential(nn.Conv1d(80,128,5,padding=2),nn.GELU(),nn.Conv1d(128,128,5,padding=2),nn.GELU())
        self.key=nn.Linear(512,128)
    def forward(self,x,query_ids=None):
        q=F.adaptive_avg_pool1d(self.query(x),times).transpose(1,2)
        logits=F.normalize(q,dim=-1)@F.normalize(self.key(bank),dim=-1).T/.1
        if query_ids is not None:
            blocked=torch.tensor(np.array(query_ids)[:,None]==bank_ids[None,:],device=device)
            logits=logits.masked_fill(blocked[:,None,:],-1e4)
        return (logits.softmax(-1)@bank).transpose(1,2)
for route in ('memory','memory_contrastive'):
    torch.manual_seed(42);np.random.seed(42);net=MemoryAdapter().to(device)
    opt=torch.optim.AdamW(net.parameters(),lr=5e-4,weight_decay=.02);best=None;hist=[]
    for epoch in range(1,301):
        net.train();groups=np.unique(data['train']['ids']);np.random.shuffle(groups)
        for g in np.array_split(groups,3):
            ix=[np.random.choice(np.flatnonzero(data['train']['ids']==v)) for v in g]
            p=net(xs['train'][ix],data['train']['ids'][ix]);y=ys['train'][ix]
            loss=F.mse_loss(p,y)
            if route=='memory_contrastive':loss=loss+.2*matched_loss(p,y)
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1);opt.step()
        net.eval()
        with torch.no_grad():val=float(F.mse_loss(net(xs['val']),ys['val']))
        hist.append({'epoch':epoch,'val_latent_mse':val})
        if best is None or val<best[0]:best=(val,epoch,copy.deepcopy(net.state_dict()))
        if epoch%100==0:print(route,epoch,val,'best',best[:2],flush=True)
    net.load_state_dict(best[2]);net.eval()
    with torch.no_grad():latent_predictions[route]=net(xs['val'])*scale+center
    training[route]={'epoch':best[1],'validation_latent_mse':best[0],'parameters':sum(p.numel() for p in net.parameters()),'bank':'training only; entire query recording excluded during training'}
    torch.save({'adapter':best[2],'center':center.cpu(),'scale':scale.cpu(),'route':route},ROOT/(route+'.pt'))
    (ROOT/(route+'_history.json')).write_text(json.dumps(hist))

# Decoder-space supervision avoids insisting on one arbitrary phase-sensitive latent code.
# A new pilot route: frozen decoder gradients reach ONLY the small ACC adapter.
sys.path.insert(0,str(ROOT.parent))
from data0723_frontend_helpers import create_mel_filterbank
fb_t=torch.tensor(create_mel_filterbank(16000,512,80,20,7500),device=device,dtype=torch.float32)
hann=torch.hann_window(512,device=device)
refmel={s:torch.tensor(np.load(ROOT.parent/'data_original'/s/'imu_audio.npy'),device=device) for s in ('train','val')}
def differentiable_mel(z):
    w=torchaudio.functional.resample(decode(z)[:,0],24000,16000)[...,:12800]
    spec=torch.stft(w,512,160,512,window=hann,center=True,pad_mode='constant',return_complex=True)/hann.sum()
    db=10*torch.log10(torch.einsum('mf,nft->nmt',fb_t,spec.abs().square())+1e-10)
    return (2*(db+100)/72.15651143799875-1).clamp(-1,1)
def spectral_objective(p,y):
    return F.l1_loss(p,y)+.3*F.l1_loss(p[:,1:]-p[:,:-1],y[:,1:]-y[:,:-1])+.1*F.l1_loss(p.mean(1),y.mean(1))
torch.manual_seed(42);np.random.seed(42);net=Adapter().to(device)
net.load_state_dict(torch.load(ROOT/'regression.pt',map_location=device,weights_only=True)['adapter'])
opt=torch.optim.AdamW(net.parameters(),lr=1e-4,weight_decay=.02)
net.eval()
with torch.no_grad():
    init=torch.cat([differentiable_mel(net(x)*scale+center) for x in xs['val'].split(2)])
    best=(float(spectral_objective(init,refmel['val'])),0,copy.deepcopy(net.state_dict()))
hist=[]
for epoch in range(1,41):
    net.train();groups=np.unique(data['train']['ids']);np.random.shuffle(groups)
    for g in np.array_split(groups,10):
        ix=[np.random.choice(np.flatnonzero(data['train']['ids']==v)) for v in g]
        p=net(xs['train'][ix]);spec=differentiable_mel(p*scale+center)
        loss=spectral_objective(spec,refmel['train'][ix])+.05*F.mse_loss(p,ys['train'][ix])
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1);opt.step()
    if epoch%2==0:
        net.eval()
        with torch.no_grad():
            p=torch.cat([differentiable_mel(net(x)*scale+center) for x in xs['val'].split(2)])
            val=float(spectral_objective(p,refmel['val']))
        hist.append({'epoch':epoch,'val_spectral_loss':val})
        if val<best[0]:best=(val,epoch,copy.deepcopy(net.state_dict()))
        print('decoder_supervised',epoch,val,'best',best[:2],flush=True)
net.load_state_dict(best[2]);net.eval()
with torch.no_grad():latent_predictions['decoder_supervised']=net(xs['val'])*scale+center
training['decoder_supervised']={'epoch':best[1],'validation_spectral_loss':best[0],'parameters':sum(p.numel() for p in net.parameters()),'decoder':'frozen, gradient enabled only to train adapter'}
torch.save({'adapter':best[2],'center':center.cpu(),'scale':scale.cpu()},ROOT/'decoder_supervised.pt')
(ROOT/'decoder_supervised_history.json').write_text(json.dumps(hist))

with torch.no_grad():
    for name,z in latent_predictions.items():
        waves=[]
        for zz in z.split(4):waves.append(decode(zz)[:,0].cpu().numpy())
        preds[name]=np.concatenate(waves)

# Exact original 16kHz, 512 FFT, 160-hop, HTK80 Mel and fixed TRAIN scales.
sys.path.insert(0,str(ROOT.parent))
from data0723_frontend_helpers import create_mel_filterbank
fb=create_mel_filterbank(16000,512,80,20,7500)
lo,hi=-100.,-27.843488562001255
def mel(w,rate=24000):
    if rate==24000:w=signal.resample_poly(w,2,3,axis=-1)[...,:12800]
    _,_,z=signal.stft(w,fs=16000,nperseg=512,noverlap=352,axis=-1)
    p=np.einsum('mf,nft->nmt',fb,abs(z)**2)
    return np.clip(2*(10*np.log10(p+1e-10)-lo)/(hi-lo)-1,-1,1)
real=np.load(ROOT.parent/'data_original/val/imu_audio.npy')
trainreal=np.load(ROOT.parent/'data_original/train/imu_audio.npy')
assert np.max(abs(mel(data['val']['audio16'],16000)-real))<1e-4,'Waveform export/reference mismatch'
assert np.mean(abs(mel(data['val']['audio24'])-real))<.002,'Unexpected resampling distortion'
ids=data['val']['ids'];wrong=np.roll(np.arange(len(ids)),4);assert np.all(ids!=ids[wrong])
def corr(a,b):
    a=a.reshape(len(a),-1);b=b.reshape(len(b),-1);a=a-a.mean(1,keepdims=True);b=b-b.mean(1,keepdims=True)
    return float(np.mean(np.sum(a*b,1)/np.maximum(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1),1e-10)))
def detail(a):
    # Same nine-bin reflected moving average used by the previous texture audit.
    padded=np.pad(a,((0,0),(4,4),(0,0)),mode='reflect')
    return a-np.mean(np.stack([padded[:,k:k+80] for k in range(9)]),axis=0)
def metrics(p):
    d=detail(p);r=detail(real)
    distances=np.mean(abs(p[:,None]-real[None,:]),axis=(2,3));dd=np.mean(abs(d[:,None]-r[None,:]),axis=(2,3));mask=ids[:,None]!=ids[None,:]
    return dict(audio_mae=float(np.mean(abs(p-real))),matching_gap=float(distances[mask].mean()-distances.diagonal().mean()),detail_corr=corr(d,r),detail_matching_gap=float(dd[mask].mean()-dd.diagonal().mean()),envelope_corr=corr(p.mean(1),real.mean(1)),variance_ratio=float(p.var(0).mean()/real.var(0).mean()))
images={name:mel(w) for name,w in preds.items()}
images['resampling_control']=mel(data['val']['audio24'])
images['train_mean_mel']=np.broadcast_to(trainreal.mean(0),real.shape)
results={'training':training,'validation':{k:metrics(p) for k,p in images.items()},'interpretation':'Exploratory reused validation, single person. No identity/FAR/FRR conclusions. Clean microphone only supplies targets/oracle, never student inference. No tests or authentication reserve accessed.'}
(ROOT/'results.json').write_text(json.dumps(results,indent=2));print(json.dumps(results),flush=True)
np.savez_compressed(ROOT/'validation_outputs.npz',real=real,ids=ids,**images)
indices=[np.flatnonzero(ids==g)[0] for g in np.unique(ids)]
rows=[('Real microphone',real)]+list(images.items())
fig,axes=plt.subplots(len(rows),4,figsize=(15,2.5*len(rows)),squeeze=False)
for r,(label,a) in enumerate(rows):
    for col,i in enumerate(indices):
        axes[r,col].imshow(a[i],origin='lower',aspect='auto',cmap='magma',vmin=-1,vmax=1)
        axes[r,col].set_title(label+' / '+str(ids[i]).replace('recording_',''),fontsize=9)
fig.suptitle('Frozen pretrained Mimi decoder / original ACC input / first window of each validation recording\nOracle uses real microphone: decoder ceiling ONLY. Student inference uses ACC only.')
fig.tight_layout();fig.savefig(ROOT/'validation.png',dpi=115)
