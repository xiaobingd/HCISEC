"""Recording-held-out stopping, expanded paired crops and feature cycles.

All final candidates retain both cycles. No audio/latent bypass to return ACC.
"""
import os, sys, json, time, copy, argparse
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE/'cycle_audio_domain_20261018'))
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(Path('outputs/imu_cyclegan_trial_20261016').resolve()))
from model_drop import GeneratorAccToAudio
from train_cycle_audio_domain import AudioDiscriminator, Discriminator, acc_loss, adv, fm, smooth, corr, detail, load
from train_cycle_resnet_curriculum import no_instance_norm

ROUTES = {
    'fixed_full': dict(dense=False, small=False, feature_cycle=False),
    'dense_full': dict(dense=True, small=False, feature_cycle=False),
    'dense_small': dict(dense=True, small=True, feature_cycle=False),
    'dense_small_feature': dict(dense=True, small=True, feature_cycle=True),
}

class AudioG(nn.Module):
    def __init__(self, small=False):
        super().__init__()
        self.projection = nn.Conv1d(251, 80, 1)
        self.decoder = GeneratorAccToAudio(input_dim=1, output_dim=1, hidden_dim=8 if small else 16, num_residual_blocks=1 if small else 2, dropout=0., upsample='resize')
        no_instance_norm(self.decoder)
    def forward(self, x):
        return self.decoder(self.projection(x)[:, None])[:, 0, :80, :81]

class FrameNorm(nn.Module):
    def __init__(self, c):
        super().__init__(); self.norm = nn.LayerNorm(c)
    def forward(self, x):
        return self.norm(x.transpose(1, 2)).transpose(1, 2)

class ReturnF(nn.Module):
    def __init__(self, small=False):
        super().__init__(); c = 32 if small else 64
        self.enc = nn.Conv1d(80, c, 1); self.level = nn.Conv1d(80, 251, 1)
        self.blocks = nn.ModuleList([nn.Sequential(nn.Conv1d(c, c, 3, padding=d, dilation=d), FrameNorm(c), nn.SiLU(), nn.Conv1d(c, c, 1)) for d in ([1, 2, 4] if small else [1, 2, 4, 8, 16])])
        self.out = nn.Conv1d(c, 251, 1)
    def forward(self, x):
        z = self.enc(x)
        for block in self.blocks:
            z = z+.25*block(z)
        return self.level(x)+.25*self.out(z)

def feature_cycle_loss(p, x):
    # Fixed summaries in the same ACC frequency channels; no trainable collapsing encoder.
    means, stds = [], []
    for k in (5, 15):
        pool = lambda z: F.avg_pool1d(z, k, stride=1, padding=k//2, count_include_pad=False)
        pm, xm = pool(p), pool(x)
        ps = (pool(p*p)-pm*pm).clamp_min(1e-6).sqrt()
        xs = (pool(x*x)-xm*xm).clamp_min(1e-6).sqrt()
        means.append(acc_loss(pm, xm)); stds.append(acc_loss(ps, xs))
    return .25*acc_loss(p, x)+.75*(sum(means)+sum(stds))/4

def evaluate(g, f, data, templates=None):
    g.eval(); f.eval(); ga, fb, rec, reverse, blurred = [], [], [], [], []
    with torch.no_grad():
        for first in range(0, len(data['ids']), 24):
            x = torch.as_tensor(data['x'][first:first+24], device='cuda')
            y = torch.as_tensor(data['y'][first:first+24], device='cuda')
            a = g(x); b = f(y)
            for dest, z in [(ga,a),(fb,b),(rec,f(a)),(reverse,g(b)),(blurred,f(smooth(a)))]:
                dest.append(z.cpu().numpy())
    ga, fb, rec, reverse, blurred = [np.concatenate(z) for z in (ga, fb, rec, reverse, blurred)]
    ids = data['ids']; groups = np.unique(ids); permutation = np.empty(len(ids), dtype=int)
    for k, name in enumerate(groups):
        ii = np.flatnonzero(ids == name); jj = np.flatnonzero(ids == groups[(k+1)%len(groups)])
        permutation[ii] = np.resize(jj, len(ii))
    wrong_audio = ga[permutation]
    rows = []
    for name in groups:
        ii = ids == name; x, y, a, c = data['x'][ii], data['y'][ii], ga[ii], rec[ii]
        cm = float(np.mean(abs(c-x)))
        row = dict(recording=str(name), audio_mae=float(np.mean(abs(a-y))), audio_detail_corr=corr(detail(a), detail(y)), audio_envelope_corr=corr(a.mean(1), y.mean(1)), audio_profile_mae=float(np.mean(abs(a.mean(2)-y.mean(2)))), audio_curvature_ratio=float(np.mean(abs(np.diff(a,n=2,axis=1)))/max(np.mean(abs(np.diff(y,n=2,axis=1))),1e-9)), ACC_cycle_mae=cm, ACC_cycle_corr=corr(c,x), ACC_detail_corr=corr(detail(c[:,:195]),detail(x[:,:195])), ACC_energy_corr=corr(c[:,195:],x[:,195:]), paired_ACC_mae=float(np.mean(abs(fb[ii]-x))), reverse_audio_cycle_mae=float(np.mean(abs(reverse[ii]-y))), blur_cycle_mae_increase=float(np.mean(abs(blurred[ii]-x))-cm), wrong_ACC_input_audio_mae_increase=float(np.mean(abs(wrong_audio[ii]-y))-np.mean(abs(a-y))))
        if templates is not None:
            row['audio_template_mae'] = float(np.mean(abs(templates[1]-y)))
            row['ACC_template_mae'] = float(np.mean(abs(templates[0]-x)))
        rows.append(row)
    return dict(macro={k:float(np.mean([r[k] for r in rows])) for k in rows[0] if k!='recording'}, per_record=rows), dict(generated_audio=ga, cycle_ACC=rec, generated_ACC=fb, ids=ids)

def subset(data, groups):
    mask = np.isin(data['ids'], groups)
    return {k:v[mask] for k,v in data.items() if len(v)==len(mask)}

def fit(cfg, seed, data, root, name, args, internal=None, cycle_epochs=None):
    torch.manual_seed(seed); rng = np.random.default_rng(seed); noise_rng = np.random.default_rng(seed+555)
    g = AudioG(cfg['small']).cuda(); f = ReturnF(cfg['small']).cuda()
    da, db = Discriminator().cuda(), AudioDiscriminator().cuda()
    opt = torch.optim.AdamW(list(g.parameters())+list(f.parameters()), lr=.0003, betas=(.5,.999), weight_decay=.001)
    od = torch.optim.AdamW(list(da.parameters())+list(db.parameters()), lr=.0002, betas=(.5,.999))
    x = torch.as_tensor(data['x'],device='cuda'); y = torch.as_tensor(data['y'],device='cuda')
    groups = [np.flatnonzero(data['ids']==r) for r in np.unique(data['ids'])]
    active = [ix[data['activity_fraction'][ix]>=.1] for ix in groups]
    templates = (data['x'].mean(0),data['y'].mean(0))
    history=[]; best_score=float('inf'); best_ep=None; teacher=None; start=time.time()
    parameters=dict(G=sum(p.numel() for p in g.parameters()), F=sum(p.numel() for p in f.parameters()))
    for stage, count in [('prewarm',args.warm_epochs), ('full_cycle',cycle_epochs or args.cycle_epochs)]:
        for ep in range(1,count+1):
            for m in (g,f,da,db):m.train()
            losses=[]
            for step in range(args.updates):
                gi=rng.choice(len(groups),8,replace=False)
                ii=[rng.choice(active[j] if len(active[j]) and rng.random()<.75 else groups[j]) for j in gi]
                a,b=x[ii],y[ii]
                S=smooth if stage=='full_cycle' and noise_rng.random()<.5 else lambda z:z
                for d in (da,db):d.requires_grad_(True)
                with torch.no_grad():generated=g(a); back=f(S(b))
                od.zero_grad();dl=.5*((da(a)-1).square().mean()+da(back).square().mean()+adv(db(b),1)+adv(db(generated),0));dl.backward();od.step()
                for d in (da,db):d.requires_grad_(False)
                generated=g(a);back=f(S(b));pd=db(generated)
                loss=20*abs(generated-b).mean()+5*acc_loss(back,a)+min(1.,.1+(ep-1)/20)*(adv(pd,1)+.05*(da(back)-1).square().mean())+abs(torch.diff(generated,dim=1)-torch.diff(b,dim=1)).mean()+abs(generated.mean(1)-b.mean(1)).mean()+2*abs(generated.mean(2)-b.mean(2)).mean()+2*fm(pd,db(b))
                if stage=='full_cycle':
                    reconstruction=f(S(generated)+.005*torch.randn_like(generated))
                    cl=feature_cycle_loss(reconstruction,a) if cfg['feature_cycle'] else acc_loss(reconstruction,a)
                    with torch.no_grad():reference=teacher(a)
                    loss=loss+(10+10*min(1.,(ep-1)/40))*cl+10*abs(g(f(S(b)))-b).mean()+5*abs(generated-reference).mean()
                opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(list(g.parameters())+list(f.parameters()),5);opt.step();losses.append(float(loss.detach()))
            if ep%15==0 or ep==count:
                row=dict(stage=stage,epoch=ep,loss=float(np.mean(losses)),seconds=round(time.time()-start,1))
                if stage=='full_cycle' and internal is not None:
                    metric,_=evaluate(g,f,internal,templates)
                    m=metric['macro'];score=.5*(m['audio_mae']/m['audio_template_mae']+m['ACC_cycle_mae']/m['ACC_template_mae'])
                    row.update(internal=metric,selection_score=score)
                    if score<best_score:
                        best_score=score;best_ep=ep
                        torch.save(dict(g=g.state_dict(),f=f.state_dict(),epoch=ep,score=score),root/(name+'_internal_best.pt'))
                history.append(row);print(name,json.dumps(row),flush=True)
                (root/(name+'_history.json')).write_text(json.dumps(history,indent=2))
        if stage=='prewarm':
            teacher=copy.deepcopy(g).eval().requires_grad_(False)
    if internal is None:
        torch.save(dict(g=g.state_dict(),f=f.state_dict()),root/(name+'_weights.pt'))
    return g,f,dict(best_cycle_epoch=best_ep,best_score=best_score if internal is not None else None,parameters=parameters,seconds=round(time.time()-start,1))

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--external-data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--warm-epochs',type=int,default=120);p.add_argument('--cycle-epochs',type=int,default=120);p.add_argument('--updates',type=int,default=4);p.add_argument('--seeds',nargs='+',type=int,default=[42,43]);p.add_argument('--routes',nargs='+',default=list(ROUTES));args=p.parse_args()
    root=args.output;root.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    dp=json.loads((args.data/'data_protocol.json').read_text());fixed=load(args.data/'fixed_train.npz');dense=load(args.data/'crop_bank_train.npz')
    all_groups=np.unique(fixed['ids']);held=dp['internal_holdout_recordings'];fit_groups=[g for g in all_groups if g not in held]
    assert len(fit_groups)==16 and len(held)==4
    internal=subset(fixed,held)
    protocol=dict(routes={r:ROUTES[r] for r in args.routes},data=dp,seeds=args.seeds,selection='For EACH route/seed fit16TRAIN, evaluate4internalTRAIN recordings every15full-cycle epochs. Choose minimum 0.5*(audioMAE/audioTRAIN-templateMAE + pixelACCcycleMAE/ACCTRAIN-templateMAE), then reinitialize same seed and refit20TRAIN for selected cycle epochs. Prewarm120 fixed. No externalVAL selection.',training=dict(warm_epochs=args.warm_epochs,max_cycle_epochs=args.cycle_epochs,updates_per_epoch=args.updates,batch_size=8,weight_decay=.001),capacity='full G original project 2DResNet hidden16 blocks2 with axialattention, F64/dilations1,2,4,8,16; small Ghidden8 blocks1 (no axialattention), F32/dilations1,2,4. Package capacity reduction, not width-only ablation.',feature_cycle='25%originalpixel +75%fixed per-frequency temporal means/stds over5/15frames; samefeaturechannel weighting; reverseaudio cycle unchanged. No learned identity loss, no external encoder, no bypass.',common='pairedprewarm thencompletebidirectionalcycle; same audioGAN+pairedlosses+TRAIN-onlyprewarmteacher; Gaussian.005 and50%3x3Mel bottleneck duringcycle; raw G evaluated/shown, cleanF(G) mainmetric; all preprocessing/scales frozen and same',scope='singleperson, exploratory4VAL reused, noidentityverification; test4/auth28 unopened; finite40mscropbanknotindependentnewrecordings',torch=torch.__version__)
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2));results=[]
    # Do not even load external VAL until after an individual refit has finished.
    for seed in args.seeds:
        for route in args.routes:
            cfg=ROUTES[route];bank=dense if cfg['dense'] else fixed;train16=subset(bank,fit_groups)
            name=route+'_'+str(seed)
            g,f,selected=fit(cfg,seed,train16,root,name+'_selection',args,internal=internal)
            del g,f;torch.cuda.empty_cache()
            g,f,final=fit(cfg,seed,bank,root,name,args,cycle_epochs=selected['best_cycle_epoch'])
            external=load(args.external_data/'data_val.npz');assert not set(external['ids'])&set(all_groups)
            templates=(fixed['x'].mean(0),fixed['y'].mean(0))
            vm,preds=evaluate(g,f,external,templates);tm,tp=evaluate(g,f,fixed,templates)
            np.savez_compressed(root/(name+'_predictions.npz'),**preds)
            # Same fixed first window in each of the first four TRAIN recordings for visual diagnostics.
            ix=[np.flatnonzero(fixed['ids']==n)[0] for n in all_groups[:4]]
            np.savez_compressed(root/(name+'_train_examples.npz'),real_audio=fixed['y'][ix],generated_audio=tp['generated_audio'][ix],ids=fixed['ids'][ix])
            result=dict(route=route,seed=seed,selection=selected,refit=final,train=tm,validation=vm)
            results.append(result);(root/'results.json').write_text(json.dumps(results,indent=2));print('RESULT',json.dumps(result),flush=True)
            del g,f;torch.cuda.empty_cache()
if __name__=='__main__':main()
