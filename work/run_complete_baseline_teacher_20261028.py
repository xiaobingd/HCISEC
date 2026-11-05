"""Frozen existing G/F, original 772-D trainb binary head, optional teachers.

First audited cohort: U1/U4/U5 from the existing 150/150 experiment.
No generator training. Teacher fitting/selection uses only the upstream 150.
"""
import copy, hashlib, importlib.util, json, os, random, sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

PROJECT=Path('/root/autodl-tmp/cycle1')
OUT=Path('/autodl-fs/data/complete_baseline_teacher_20261028')
os.chdir(str(PROJECT)); sys.path.insert(0,str(PROJECT))
# Pin the historical ResNet definition; the mutable sibling module is now U-Net.
_spec=importlib.util.spec_from_file_location('model_drop',str(PROJECT/'experiments/samebo200safe_user3/model_drop.py'))
_module=importlib.util.module_from_spec(_spec); sys.modules['model_drop']=_module; _spec.loader.exec_module(_module)
import trainb_target_binary as original
from trainb_target_150 import fusion_inputs
from target_auth_protocol_150 import split_user_data
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
torch.set_num_threads(2)

def seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv=nn.Sequential(nn.Conv2d(1,16,5,2,2),nn.GroupNorm(4,16),nn.SiLU(),
            nn.Conv2d(16,32,3,2,1),nn.GroupNorm(4,32),nn.SiLU(),
            nn.Conv2d(32,64,3,2,1),nn.GroupNorm(8,64),nn.SiLU(),nn.AdaptiveAvgPool2d((10,1)))
        self.fc=nn.Linear(640,64)
    def forward(self,x):
        x=x-x.mean((2,3),keepdim=True)
        return F.normalize(self.fc(self.conv(x).flatten(1)),dim=1)

class Teacher(nn.Module):
    def __init__(self):
        super().__init__(); self.acc=Encoder(); self.audio=Encoder(); self.binary=nn.Linear(64,1)

def tensor(x): return torch.as_tensor(x[:,None],dtype=torch.float32,device=DEVICE)

def supcon(z,y):
    eye=torch.eye(len(y),dtype=torch.bool,device=z.device)
    pos=(y[:,None]==y[None,:]) & ~eye
    logits=(z@z.T/.1).masked_fill(eye,-1e4)
    lp=logits-torch.logsumexp(logits,dim=1,keepdim=True)
    return -((lp*pos).sum(1)/pos.sum(1).clamp_min(1)).mean()

def fit_teacher(target,sources):
    seed(2800+target); users=[1,2,4,5,6,7,9]
    aa=[]; mm=[]; yy=[]; ids={}
    for u in users:
        name='bo' if u==1 else 'user%d'%u
        idx=np.asarray(sources[name]['cyclegan_indices'])
        idx=np.random.RandomState(4200+u).permutation(idx)
        a=np.load(str(PROJECT/'meiy'/str(u)/'acc.npy')).astype('float32')[:,:80,:80]
        m=np.load(str(PROJECT/'meiy'/str(u)/'audio.npy')).astype('float32')[:,:80,:80]
        assert len(a)==len(m) and np.isfinite(a).all() and np.isfinite(m).all()
        aa.append(a[idx]); mm.append(m[idx]); yy.append(np.full(150,int(u==target)))
        ids[name]={'fit':idx[:120].tolist(),'selection':idx[120:].tolist()}
        assert not set(idx)&set(sources[name]['auth_indices'])
    aa=np.stack(aa); mm=np.stack(mm); yy=np.stack(yy)
    xa=aa[:,:120].reshape(-1,80,80); xm=mm[:,:120].reshape(-1,80,80); y=yy[:,:120].ravel()
    va=tensor(aa[:,120:].reshape(-1,80,80)); vm=tensor(mm[:,120:].reshape(-1,80,80))
    vy=torch.as_tensor(np.tile(yy[:,120:].ravel(),2),dtype=torch.float32,device=DEVICE)
    model=Teacher().to(DEVICE); opt=torch.optim.AdamW(model.parameters(),lr=.0007,weight_decay=.001)
    rng=np.random.RandomState(2800+target); pos=np.flatnonzero(y); neg=np.flatnonzero(y==0)
    best=None; history=[]
    for step in range(1,601):
        model.train(); ix=np.r_[rng.choice(pos,24,False),rng.choice(neg,24,False)]
        a=tensor(xa[ix]); m=tensor(xm[ix]); sh=int(rng.randint(-3,4))
        a=torch.roll(a,sh,-1)+.005*torch.randn_like(a); m=torch.roll(m,sh,-1)+.005*torch.randn_like(m)
        za=model.acc(a); zm=model.audio(m); z=torch.cat([za,zm])
        labels=torch.as_tensor(np.tile(y[ix],2),dtype=torch.float32,device=DEVICE)
        loss=F.binary_cross_entropy_with_logits(model.binary(z).squeeze(1),labels)+.15*supcon(z,labels)+.5*(1-(za*zm).sum(1)).mean()
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1); opt.step()
        if step%25==0:
            model.eval()
            with torch.no_grad():
                za=model.acc(va); zm=model.audio(vm); z=torch.cat([za,zm])
                vl=F.binary_cross_entropy_with_logits(model.binary(z).squeeze(1),vy,pos_weight=torch.tensor(6.,device=DEVICE))+.15*supcon(z,vy)+.5*(1-(za*zm).sum(1)).mean()
            value=float(vl); history.append({'step':step,'validation_loss':value})
            if best is None or value<best[0]: best=(value,step,copy.deepcopy(model.state_dict()))
    model.load_state_dict(best[2]); model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    meta={'target':target,'source_indices':ids,'fit_steps':600,'selected_step':best[1],
          'selected_loss':best[0],'history':history,'authentication_overlap':0}
    torch.save({'state':model.state_dict(),'metadata':meta},OUT/('U%d_teacher.pt'%target))
    return model

def main():
    OUT.mkdir(parents=True,exist_ok=True); sources=split_user_data(42)
    # Verify existing 150/150 manifest against current arrays before fitting anything.
    recorded=json.loads(Path('/autodl-fs/data/auth_full_pipeline_150x150_20260924/data_split_manifest.json').read_text())['users']
    for name,source in sources.items():
        for field in ('sha256','n_total','cyclegan_indices','auth_indices'):
            assert source[field]==recorded[name][field], (name,field)
    (OUT/'source_manifest.json').write_text(json.dumps(sources,indent=2))
    original_create=original.create_multi_source_model
    original_precompute=original.precompute_features
    original_evaluate=original.evaluate
    results_path=OUT/'results.json'
    results=json.loads(results_path.read_text()) if results_path.exists() else {}
    for target in (1,4,5):
        print('TARGET_START',target,flush=True)
        teacher_path=OUT/('U%d_teacher.pt'%target)
        if teacher_path.exists():
            saved=torch.load(str(teacher_path),map_location=DEVICE,weights_only=False)
            for name,part in saved['metadata']['source_indices'].items():
                assert set(part['fit']+part['selection'])==set(sources[name]['cyclegan_indices'])
            teacher=Teacher().to(DEVICE); teacher.load_state_dict(saved['state']); teacher.eval()
            for p in teacher.parameters(): p.requires_grad_(False)
        else: teacher=fit_teacher(target,sources)
        checkpoint='/autodl-fs/data/auth_full_pipeline_150x150_20260924/user%d/cyclegan/model/checkpoint_epoch_200.pth'%target
        assert Path(checkpoint).exists()
        for mode in ('baseline','baseline_teacher'):
            folder=OUT/('U%d_%s'%(target,mode))
            seed(42)
            def create(*args,**kwargs):
                model,info=original_create(*args,**kwargs)
                assert model.fusion[0].in_features==772
                if mode=='baseline_teacher':
                    state=torch.get_rng_state()
                    model.fusion[0]=nn.Linear(900,512).to(DEVICE)
                    torch.set_rng_state(state)
                return model,info
            @torch.no_grad()
            def precompute(model,specs,labels,device,batch_size,augment=None):
                if mode=='baseline': return original_precompute(model,specs,labels,device,batch_size,augment)
                dataset=original.MultiUserSpectrogramDataset(specs,labels,augment=augment)
                loader=DataLoader(dataset,batch_size=batch_size,shuffle=False,num_workers=0)
                cached=[]; labs=[]; model.eval(); teacher.eval()
                for x,y in loader:
                    x=x.to(device)
                    features=fusion_inputs(model,x)
                    za=teacher.acc(x); zm=teacher.audio(model.g_acc2audio(x))
                    cached.append(torch.cat([features,za,zm],1).cpu()); labs.append(y.cpu())
                return torch.cat(cached),torch.cat(labs)
            def evaluate(model,binary,cached,protocol,target_user,final=False):
                # Never score final queries before checkpoint and threshold selection.
                if not final:
                    cached={k:v for k,v in cached.items() if k not in ('final_genuine','final_impostor')}
                return original_evaluate(model,binary,cached,protocol,target_user,final)
            original.create_multi_source_model=create; original.precompute_features=precompute; original.evaluate=evaluate
            sys.argv=['trainb_target_binary.py','--target_user',str(target),'--cyclegan_ckpt',checkpoint,
                      '--save_dir',str(folder),'--log_dir',str(folder/'logs'),'--feature_batch_size','16']
            original.main()
            result=json.loads((folder/'trainb_binary_results.json').read_text())
            result['input_features']=772 if mode=='baseline' else 900
            result['teacher_inference_inputs']='ACC and G(ACC) only; no real test Audio'
            result['generator_training']='none; frozen existing checkpoint; see cyclegan_ckpt for version/epoch'
            result['content_protocol']='Original random-recording 150/150 split; not unseen-word verification'
            results['U%d_%s'%(target,mode)]=result
            (OUT/'results.json').write_text(json.dumps(results,indent=2))
            print('ROUTE_DONE',target,mode,result['final_attack'],flush=True)
        del teacher
    print('COMPLETE',flush=True)

if __name__=='__main__': main()
