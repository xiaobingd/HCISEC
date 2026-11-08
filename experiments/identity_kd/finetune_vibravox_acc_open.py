"""Full embedding-backbone adaptation, with outer identities strictly excluded.

Three-axis mean embedding; gradient caching permits logical batches on CPU.
Feature frontend and BN running statistics are fixed; embedding weights train.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import argparse
import copy
import json
from pathlib import Path
import sys
import time
import numpy as np
from scipy.signal import resample_poly
import torch
from torch import nn
from torch.nn import functional as F
sys.path.insert(0,'/root')
import run_strict_open_residual as c
from run_ecapa_pilot import teacher_load

ROOT=Path('/autodl-fs/data');SOURCE=ROOT/'identity_kd_all9_raw_full_multisource_20261103';CACHE=ROOT/'vr_acc_pretrain_control_20261108'


def main(args):
    out=Path(args.out);out.mkdir(exist_ok=True,parents=True)
    for name in ['model','deps']:
        if not(out/name).exists():(out/name).symlink_to(SOURCE/name,target_is_directory=True)
    device='cuda' if torch.cuda.is_available() else 'cpu';torch.set_num_threads(1)
    meta=np.load(SOURCE/'raw9_full_features_private.npz');keep=meta['user']!=9
    d={k:meta[k][keep] for k in ['user','word','role','sample_id']}
    a=np.load(CACHE/'vr_acceleration_transfer_private.npz');assert np.array_equal(a['sample_id'],meta['sample_id'])
    wave=a['acceleration'][keep]
    teacher=np.load(ROOT/'mei9_vibravox_teacher_transfer_20261107/vibravox_audio_embeddings_private.npy')[keep].astype('float32')
    frozen=np.load(CACHE/'vibravox_acc_embedding_private.npy')[keep]
    assert np.isfinite(frozen).all() and np.all(np.linalg.norm(frozen,axis=1)>.9)
    def status(**v):c.dump(out/'status.json',dict(updated_unix=time.time(),**v))
    c.dump(out/'protocol.json',dict(scope='8 outer targets x3 training seeds;4base/2cal/target/unseenattacker',initialization='VibraVox ACC ECAPA best_backbone_private.pt, reinitialized for each fold/seed/method',trainable='ALL embedding_model parameters; frontend fixed;BatchNorm running statistics fixed;not adapter-only',input='same timestamp-derived acceleration XYZ as frozen transfer;per-axis embedding then normalize mean;no per-user axis selection',modes=['identity_ft','audio_relation_ft'],loss='cosine CE+.1SupCon+.1frozenACCfeaturepreservation;Audio arm adds .1balancedrelationKD;no final user participates',budget=dict(steps=args.steps,lr_backbone=1e-5,lr_head=1e-3,batch=8,microbatch='one recording,three axes;two-pass gradient cache'),selection='base develop: first2 repeats per word,32records/fold;every50steps;step0 eligible;same subset acrossseeds/methods',authentication='same locked sameword/crossword queries,12shot enrollment,sharedcalibrationthreshold and identity split as strict_open_residual',limits='timestamp derivative is not hardware accelerometer;words inferred;comparison frozen vs FT isolates adaptation on same frontend'))
    results={}
    for seed in args.seeds:
        for target in args.targets:
            f=c.fold_users(target);fit=np.flatnonzero(np.isin(d['user'],f['base'])&(d['role']=='fit'))
            dev=np.concatenate([np.flatnonzero((d['user']==u)&(d['word']==w)&(d['role']=='develop'))[:2] for u in f['base'] for w in np.unique(d['word'][d['role']=='develop'])])
            words=np.unique(d['word'][fit]);lookup={(u,w):fit[(d['user'][fit]==u)&(d['word'][fit]==w)] for u in f['base'] for w in words};labels={u:j for j,u in enumerate(f['base'])}
            frozen_result=c.evaluate(frozen,d,f)
            for mode in ['identity_ft','audio_relation_ft']:
                folder=out/f'target{target}_seed{seed}'/mode;folder.mkdir(exist_ok=True,parents=True)
                resultpath=folder/'results_private.json'
                if resultpath.exists():results[str(folder.relative_to(out))]=json.loads(resultpath.read_text());continue
                torch.manual_seed(seed);rng=np.random.RandomState(seed)
                model=teacher_load(out);emb=model.mods.embedding_model
                ck=torch.load(ROOT/'vibravox_dual_teacher_20261105/acc_teacher/best_backbone_private.pt',map_location='cpu',weights_only=False);emb.load_state_dict(ck['embedding_model']);del ck
                for p in emb.parameters():p.requires_grad_(True)
                emb.eval();head=nn.Linear(192,4,bias=False).to(device)
                opt=torch.optim.AdamW([dict(params=emb.parameters(),lr=1e-5),dict(params=head.parameters(),lr=1e-3)],weight_decay=1e-4)
                watched=next(p for p in emb.parameters() if p.ndim>1);initial=watched.detach().clone()
                def features(i):
                    w=resample_poly(wave[i],16,1,axis=1).astype('float32');w=torch.tensor(w,device=device)
                    with torch.no_grad():return model.mods.mean_var_norm(model.mods.compute_features(w),torch.ones(3,device=device)).detach()
                def vector(feat):
                    z=F.normalize(emb(feat,torch.ones(3,device=device)).squeeze(1),dim=1)
                    return F.normalize(z.mean(0,keepdim=True),dim=1)[0]
                def encode(ii):
                    emb.eval()
                    with torch.no_grad():return np.stack([vector(features(int(i))).cpu().numpy() for i in ii])
                best=c.r.pair_auc(frozen[dev],d['user'][dev],d['word'][dev]);selected=0
                bestpath=folder/'best_backbone_private.pt';resume=folder/'resume_private.pt';start=0
                if resume.exists():
                    state=torch.load(resume,map_location=device);emb.load_state_dict(state['embedding_model']);head.load_state_dict(state['head']);opt.load_state_dict(state['optimizer']);rng.set_state(state['rng']);start=state['step'];best=state['best'];selected=state['selected_step'];del state
                else:torch.save(dict(embedding_model=emb.state_dict(),selected_step=0,base_users=f['base']),bestpath)
                for step in range(start+1,args.steps+1):
                    emb.eval();ww=rng.choice(words,2,replace=False);ii=np.array([int(rng.choice(lookup[u,w])) for u in rng.permutation(f['base']) for w in ww]);y=torch.tensor([labels[u] for u in d['user'][ii]],device=device)
                    feats=[features(int(i)) for i in ii]
                    with torch.no_grad():z0=torch.stack([vector(feat) for feat in feats])
                    z=z0.detach().requires_grad_(True)
                    loss=F.cross_entropy(16*F.linear(z,F.normalize(head.weight,dim=1)),y)+.1*c.r.p.sc_loss(z,y)+.1*(1-(z*torch.tensor(frozen[ii],device=device)).sum(1)).mean()
                    if mode=='audio_relation_ft':loss=loss+.1*c.relational(z,torch.tensor(teacher[ii],device=device),y,torch.ones((4,4),device=device))
                    opt.zero_grad();loss.backward();grad=z.grad.detach()
                    for j,feat in enumerate(feats):(vector(feat)*grad[j]).sum().backward()
                    assert watched.grad is not None and torch.isfinite(watched.grad).all()
                    nn.utils.clip_grad_norm_(list(emb.parameters())+list(head.parameters()),1.);opt.step()
                    delta=float((watched-initial).abs().max().detach());assert delta>0,'embedding backbone was not updated'
                    if step%50==0:
                        zz=encode(dev);q=c.r.pair_auc(zz,d['user'][dev],d['word'][dev])
                        if q>best:best=q;selected=step;torch.save(dict(embedding_model=emb.state_dict(),selected_step=step,base_users=f['base']),bestpath)
                    if step==1 or step%10==0:
                        torch.save(dict(embedding_model=emb.state_dict(),head=head.state_dict(),optimizer=opt.state_dict(),rng=rng.get_state(),step=step,best=best,selected_step=selected),resume)
                        status(state='finetuning_full_acc_backbone',target=target,seed=seed,mode=mode,step=step,total_steps=args.steps,backbone_max_update=delta,device=device)
                        print('FT',target,seed,mode,step,delta,flush=True)
                emb.load_state_dict(torch.load(bestpath,map_location=device)['embedding_model'])
                z=frozen.copy()
                # Final inference is after checkpoint selection; does not update weights.
                ii=np.flatnonzero(d['role']=='final');z[ii]=encode(ii)
                result=dict(target=target,seed=seed,mode=mode,fold=f,selected_step=selected,base_develop_pair_auc=best,frozen=frozen_result,finetuned=c.evaluate(z,d,f))
                c.dump(resultpath,result);results[str(folder.relative_to(out))]=result;c.dump(out/'results_private.json',results)
                del model,emb,head,opt
    status(state='complete',completed_adaptations=len(results))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',default=str(ROOT/'mei8_vibravox_acc_fullft_20261108'));ap.add_argument('--targets',nargs='+',type=int,default=list(range(1,9)));ap.add_argument('--seeds',nargs='+',type=int,default=[42,43,44]);ap.add_argument('--steps',type=int,default=200);main(ap.parse_args())
