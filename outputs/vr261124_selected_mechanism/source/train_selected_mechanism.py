"""Coarse local Audio supervision with matched wrong-repeat/user controls.
No calibrated physical synchronization is assumed; no final-query Audio is read.
"""
import json
from pathlib import Path
import numpy as np
from scipy import signal
from scipy.io import wavfile
import torch
from torch import nn
from torch.nn import functional as F
from train_audio_multiword_fitonly import supcon

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'conditional_correspondence/stage4_selected';OUT.mkdir(exist_ok=True)
DEV=[8,18,24,30]

def crossword_supcon(z,y,words):
    logits=z@z.T/.1
    same=y[:,None]==y[None,:]
    different_word=words[:,None]!=words[None,:]
    positive=same&different_word
    valid=(~same)|positive
    assert positive.any(1).all()
    logp=logits-torch.logsumexp(logits.masked_fill(~valid,-torch.inf),dim=1,keepdim=True)
    return -((logp*positive).sum(1)/positive.sum(1)).mean()

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.body=nn.Sequential(nn.Conv1d(147,32,3,padding=1),nn.GroupNorm(4,32),nn.GELU(),
            nn.Conv1d(32,32,3,padding=1),nn.GroupNorm(4,32),nn.GELU())
        self.identity=nn.Linear(64,192)
        self.local=nn.Conv1d(32,26,1)
    def forward(self,x):
        h=self.body(x)
        z=F.normalize(self.identity(torch.cat([h.mean(2),h.std(2)],1)),dim=1)
        return z,F.adaptive_avg_pool1d(self.local(h),12)

def main():
    torch.set_num_threads(2)
    d=np.load(ROOT/'conditional_correspondence/coarse_pilot/raw_inputs_private.npz')
    u,w,r,fit=d['user'],d['word'],d['repeat'],d['fit'];ids=d['sample_id']
    raw=d['acc'];assert raw.shape==(3200,3,3000) and fit.sum()==1280
    dev=np.isin(w,DEV);assert not np.any(fit&dev)
    manifest=json.loads((ROOT/'mapping_44100/consistent_timebase/aligned_manifest_private.json').read_text(encoding='utf8'))['records']
    assert [a['sample_id'] for a in manifest]==list(ids)
    # Fresh fixed-frame ACC spectrum from raw time series, not old762statistics.
    spec=[]
    for i in range(0,3200,80):
        f,t,z=signal.stft(raw[i:i+80],1000,nperseg=100,noverlap=80,nfft=100,boundary=None,padded=False)
        spec.append(np.log(np.maximum(abs(z[:,:,1:50])**2,1e-20)).reshape(-1,147,len(t)))
    spec=np.concatenate(spec).astype('float32')
    mean=spec[fit].mean((0,2),keepdims=True);std=spec[fit].std((0,2),keepdims=True)
    x=torch.tensor((spec-mean)/np.maximum(std,1e-5));del raw,spec
    cache=ROOT/'conditional_correspondence/stage2/audio_targets_private.npz'
    allowed=fit|dev
    if cache.exists():
        c=np.load(cache);assert np.array_equal(c['sample_id'],ids) and np.array_equal(c['allowed'],allowed)
        targets=c['target']
    else:
        targets=np.zeros((3200,26,12),dtype='float32');groups={}
        for i in np.flatnonzero(allowed):groups.setdefault(manifest[i]['record_group'],[]).append(int(i))
        assert len(groups)==200
        for count,(stem,ix) in enumerate(groups.items()):
            p=Path(stem);meta=json.loads(p.with_name(p.name+'_meta.json').read_text(encoding='utf-8-sig'))
            fs,pcm=wavfile.read(p.parent/meta['full_audio']);assert fs==48000 and pcm.dtype==np.int16
            pcm=pcm.astype('float32')/32768
            if pcm.ndim==2:pcm=pcm[:,0]
            for i in ix:
                center=(manifest[i]['audio_start_s']+manifest[i]['audio_end_s'])/2
                frames=np.rint((center+np.arange(132300)/44100-1.5)*44100).astype(int)
                valid=(frames>=0)&(frames<len(pcm));wave=np.zeros(132300,dtype='float32');wave[valid]=pcm[frames[valid]]
                wave=signal.resample_poly(wave,160,441)
                f,t,z=signal.stft(wave,16000,nperseg=1600,noverlap=1280,nfft=1600,boundary=None,padded=False)
                power=abs(z)**2;low=np.stack([np.interp(np.arange(50,301,10),f,col) for col in power.T],1)
                lp=np.log(np.maximum(low,1e-12))
                for k in range(12):targets[i,:,k]=lp[:,(t>=k*.25)&(t<(k+1)*.25)].mean(1)
            if (count+1)%40==0:print('LOCAL_AUDIO',count+1,flush=True)
        np.savez_compressed(cache,target=targets,sample_id=ids,allowed=allowed)
    am=targets[fit].mean((0,2),keepdims=True);asd=targets[fit].std((0,2),keepdims=True)
    target=torch.tensor((targets-am)/np.maximum(asd,1e-5));y=torch.tensor(u-1,dtype=torch.long)
    words=np.unique(w[fit]);groups=np.array([[np.flatnonzero(fit&(u==us)&(w==wo)) for wo in words] for us in range(1,11)])
    lookup={(int(us),int(wo),int(re)):i for i,(us,wo,re) in enumerate(zip(u,w,r))}
    screening=json.loads((ROOT/'conditional_correspondence/band_screen/summary_public.json').read_text())
    selected=screening['selected_hz'];assert selected==list(range(70,301,10))
    bandmask=np.array([(hz-50)//10 for hz in selected])
    methods=['A_ACC','B_correct','C_wrong_repeat','D_wrong_user']
    protocol=dict(stage='stage3 single-variable cross-word identity constraint;reuse stage2 Audio targets;physical sync unverified',users=10,
        train_words=words.tolist(),development_words=DEV,train_records=1280,development_records=320,
        encoder='fresh147channel ACC log-power100msSTFT/20ms hop from3axis raw;Conv32-GN-GELU twice;mean/std->192identity;separate26channel local head',
        audio='continuous PCM interpreted44100 resampled16k;100msSTFT/20ms hop;50-300Hz/10Hz bins;12fixed250ms bins over3sec',
        loss='CE+.1crossword-only SupCon allarms;sameuser sameword ignored,not negatives;B/C/D localSmoothL1;body auxiliary gradient ratio.1 cap10,identityhead receives no direct local loss',
        controls='C fixed next repeat sameuserword;D fixed nextuser samewordrepeat;matched Audio local-feature marginals;wrongrepeat remains identity positive',
        scope='3sec raw windows,no per-record VAD zero mask;VAD centers still reused;adjacent windows may overlap and are not independent subjects',
        optimizer='fresh per-seed matched init/batch;400steps AdamW.001 wd.01 clip1;10users x2words x2repeats',
        methods=methods,seeds=[42,43,44],selection='fixed allsettings before development retrieval and query auth;no finalAudio features')
    protocol.update(stage='fixed screened-band authentication confirmation on reused cohort;not new blind evidence',selected_hz=selected,selection_evidence='sharedcandidate only;previous independentidentitysupport empty',comparison='reuse exactstage3 noAudio+fullAudio baseline;newselectedB/C/D same400steps/init/batches')
    (OUT/'protocol_public.json').write_text(json.dumps(protocol,indent=2))
    logs=[];retrieval=[];views={}
    for seed in (42,43,44):
        for method in methods:
            torch.manual_seed(seed);rng=np.random.default_rng(seed);net=Network();head=nn.Linear(192,10,bias=False)
            opt=torch.optim.AdamW([*net.parameters(),*head.parameters()],lr=.001);cp=OUT/f'{method}_{seed}_private.pt';trace=[]
            if method=='A_ACC':
                net.load_state_dict(torch.load(ROOT/f'conditional_correspondence/stage3/A_ACC_{seed}_private.pt',map_location='cpu',weights_only=True));torch.save(net.state_dict(),cp)
            elif cp.exists():net.load_state_dict(torch.load(cp,map_location='cpu',weights_only=True))
            else:
                for step in range(400):
                    chosen=rng.choice(16,2,replace=False)
                    ix=np.array([np.concatenate([rng.choice(groups[us,wo],2,replace=False) for wo in chosen]) for us in range(10)]).ravel()
                    z,pred=net(x[ix]);identity=F.cross_entropy(16*F.linear(z,F.normalize(head.weight,dim=1)),y[ix])+.1*crossword_supcon(z,y[ix],torch.tensor(w[ix]))
                    aux=identity.new_zeros(());weight=0.;cosine=0.
                    if method!='A_ACC':
                        jj=ix if method=='B_correct' else np.array([lookup[int(u[i]),int(w[i]),(int(r[i])+1)%8] if method=='C_wrong_repeat' else lookup[int(u[i])%10+1,int(w[i]),int(r[i])] for i in ix])
                        assert np.all(fit[jj])
                        aux=F.smooth_l1_loss(pred[:,bandmask],target[jj][:,bandmask])
                        gi=torch.autograd.grad(identity,net.body.parameters(),retain_graph=True);ga=torch.autograd.grad(aux,net.body.parameters(),retain_graph=True)
                        g=torch.cat([a.ravel() for a in gi]);a=torch.cat([v.ravel() for v in ga])
                        weight=min(10.,.1*float(g.norm())/max(float(a.norm()),1e-8));cosine=float(F.cosine_similarity(g[None],a[None]))
                    loss=identity+weight*aux;opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_([*net.parameters(),*head.parameters()],1);opt.step()
                    if (step+1)%100==0:
                        trace.append(dict(step=step+1,identity=float(identity.detach()),local=float(aux.detach()),weight=weight,body_gradient_cosine=cosine));print('LOCAL_TRAIN',seed,method,step+1,flush=True)
                torch.save(net.state_dict(),cp)
            with torch.inference_mode():
                allz=[];allp=[]
                for i in range(0,3200,80):
                    z,pred=net.eval()(x[i:i+80]);allz.append(z.numpy());allp.append(pred.numpy())
                views[f'{seed}_{method}']=np.concatenate(allz);pred=np.concatenate(allp)
            for scope,mask in [('fit',fit),('development_unseen_words',dev)]:
                for us in range(1,11):
                    for wo in np.unique(w[mask]):
                        ix=np.flatnonzero(mask&(u==us)&(w==wo));assert len(ix)==8
                        a=pred[ix][:,bandmask].reshape(8,-1);b=target[ix][:,bandmask].numpy().reshape(8,-1)
                        score=-((a[:,None]-b[None,:])**2).mean(2);diag=np.diag(score)
                        top=np.mean([np.mean(np.flatnonzero(row==row.max())==i) for i,row in enumerate(score)])
                        retrieval.append(dict(seed=seed,method=method,scope=scope,user=us,word=int(wo),top1=float(top),margin=float((diag-score[~np.eye(8,dtype=bool)].reshape(8,7).mean(1)).mean())))
            logs.append(dict(seed=seed,method=method,trace=trace))
            (OUT/'training_public.json').write_text(json.dumps(logs,indent=2));(OUT/'retrieval_public.json').write_text(json.dumps(retrieval))
    reference=np.load(ROOT/'conditional_correspondence/stage3/views_private.npz')
    for seed in (42,43,44):
        assert np.array_equal(views[f'{seed}_A_ACC'],reference[f'{seed}_A_ACC'])
        views[f'{seed}_B_fullSpectrum']=reference[f'{seed}_B_correct']
    np.savez_compressed(OUT/'views_private.npz',**views)
    (OUT/'training_verification_public.json').write_text(json.dumps(dict(status='PASS',models=12,train_audio=1280,development_audio=320,final_query_audio=0,fit_only_normalizers=True,physical_sync_verified=False),indent=2))
    print('LOCAL_MECHANISM_TRAIN_COMPLETE',flush=True)

if __name__=='__main__':main()
