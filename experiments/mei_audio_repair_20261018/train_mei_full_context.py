"""Long-context complete-cycle adaptation, with shared-target interior comparisons."""
import os,sys,json,argparse,time
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('MKL_THREADING_LAYER','SEQUENTIAL')
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from train_mei_audio_repair import AudioG,ReturnF,AudioDiscriminator,Discriminator,acc_loss,adv,contrast,extra_metrics,load

class FullG(AudioG):
    def forward(self,x):return self.decoder(self.projection(x)[:,None])[:,0,:80,:x.shape[-1]]

def legacy_predictions(g,f,data,legacy):
    g.eval();f.eval();aa=[];cc=[]
    with torch.no_grad():
        for i in range(0,len(data['ids']),12):
            x=torch.as_tensor(data['x'][i:i+12],device='cuda');y=g(x);aa.append(y.cpu().numpy());cc.append(f(y).cpu().numpy())
    aa,cc=np.concatenate(aa),np.concatenate(cc);lookup={str(n):i for i,n in enumerate(data['ids'])};ga=[];rec=[];yy=[];xx=[];ids=[]
    for name in np.unique(legacy['ids']):
        ii=np.flatnonzero(legacy['ids']==name);idx=lookup[str(name)]
        for j in range(3):
            ga.append(aa[idx,:,j*40+8:j*40+73]);rec.append(cc[idx,:,j*40+8:j*40+73]);yy.append(legacy['y'][ii[j],:,8:-8]);xx.append(legacy['x'][ii[j],:,8:-8]);ids.append(name)
    d=dict(x=np.array(xx),y=np.array(yy),ids=np.array(ids));pred=dict(generated_audio=np.array(ga),cycle_ACC=np.array(rec),ids=d['ids'])
    return d,pred,dict(generated_audio=aa,cycle_ACC=cc,ids=data['ids'])

def metrics(d,p,alo,ahi,threshold):
    extra=extra_metrics(p,d,alo,ahi,threshold);rows=[]
    from train_mei_audio_repair import corr,detail
    for name in np.unique(d['ids']):
        ii=d['ids']==name;a,b=p['generated_audio'][ii],d['y'][ii];c,x=p['cycle_ACC'][ii],d['x'][ii]
        rows.append(dict(recording=str(name),audio_mae=float(abs(a-b).mean()),audio_detail_corr=corr(detail(a),detail(b)),ACC_cycle_mae=float(abs(c-x).mean()),ACC_detail_corr=corr(detail(c[:,:195]),detail(x[:,:195]))))
    return dict(macro={k:float(np.mean([r[k] for r in rows])) for k in rows[0] if k!='recording'},per_record=rows,extra=extra)

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--study',type=Path,required=True);p.add_argument('--epochs',type=int,default=120);p.add_argument('--seeds',nargs='+',type=int,default=[42,43]);args=p.parse_args();root=args.study/'full_context_results';root.mkdir(exist_ok=True);torch.set_num_threads(2)
    cp_proto=json.loads((args.study/'results/protocol.json').read_text());alo,ahi=cp_proto['normalization_audio'];threshold=cp_proto['audio_energy_threshold'];span=ahi-alo
    tr=load(args.study/'full_context_data/full_train.npz');va=load(args.study/'full_context_data/full_validation.npz');legacy=load(args.source/'data/mei_val.npz');assert not set(tr['ids'])&set(va['ids'])
    x=torch.as_tensor(tr['x'],device='cuda');y=torch.as_tensor(tr['y'],device='cuda');results=[]
    proto=dict(data=json.loads((args.study/'full_context_data/protocol.json').read_text()),training='Oldsame-seedmei180longG/F init, Gonlyoutputcrop generalizes81->inputT=201; SAME network weights/architecture. NewD40warmup.120epochs4updates8uniqueTRAINrecordings. CompleteACCcycle20/audio10/pairedACC5/pairedMel20; no S oroldteacher; audioGAN.1/ACCGAN.05,localcontrast.5,freqgrad1,timegrad1,physicalMelenergy3. Fixedendpoint,noVALselection.',seeds=args.seeds,epochs=args.epochs,scope='Exploratory longcontext/objective adaptation package, not clean attribution tocontext alone. Offline2s latency. Common interior scoring comparedoldLOCALandinitialFULLGandTRAINtime-template; not directly againstolderall-frameMAE.')
    (root/'protocol.json').write_text(json.dumps(proto,indent=2))
    for seed in args.seeds:
        torch.manual_seed(seed);cp=torch.load(args.source/'results_long'/('mei180_long_'+str(seed)+'_weights.pt'),map_location='cuda',weights_only=True);g=FullG().cuda();f=ReturnF().cuda();g.load_state_dict(cp['g']);f.load_state_dict(cp['f'])
        d,before,_=legacy_predictions(g,f,va,legacy);initial=metrics(d,before,alo,ahi,threshold)
        torch.manual_seed(seed+2026);da,db=Discriminator().cuda(),AudioDiscriminator().cuda();opt=torch.optim.AdamW(list(g.parameters())+list(f.parameters()),lr=.00015,betas=(.5,.999),weight_decay=.001);od=torch.optim.AdamW(list(da.parameters())+list(db.parameters()),lr=.0002,betas=(.5,.999));rng=np.random.default_rng(seed);noise=torch.Generator(device='cuda').manual_seed(seed+777)
        def dl(a,b,ga,fb):return .5*((da(a)-1).square().mean()+da(fb).square().mean()+adv(db(b),1)+adv(db(ga),0))
        for _ in range(40):
            ii=rng.choice(len(x),8,replace=False);a,b=x[ii],y[ii]
            with torch.no_grad():ga=g(a);fb=f(b)
            od.zero_grad();loss=dl(a,b,ga,fb);loss.backward();od.step()
        start=time.time();history=[]
        env=lambda a:torch.log10(torch.pow(10.,(a-1)*span/20).mean(1).clamp_min(1e-12))
        for ep in range(1,args.epochs+1):
            g.train();f.train();losses=[]
            for _ in range(4):
                ii=rng.choice(len(x),8,replace=False);a,b=x[ii],y[ii]
                for z in [da,db]:z.requires_grad_(True)
                with torch.no_grad():ga=g(a);fb=f(b)
                od.zero_grad();loss=dl(a,b,ga,fb);loss.backward();od.step()
                for z in [da,db]:z.requires_grad_(False)
                ga=g(a);fb=f(b);rec=f(ga+.005*torch.randn(ga.shape,device='cuda',generator=noise));ry=g(fb)
                loss=20*abs(ga-b).mean()+20*acc_loss(rec,a)+10*abs(ry-b).mean()+5*acc_loss(fb,a)+.1*adv(db(ga),1)+.05*(da(fb)-1).square().mean()+.5*abs(contrast(ga)-contrast(b)).mean()+abs(torch.diff(ga,dim=1)-torch.diff(b,dim=1)).mean()+abs(torch.diff(ga,dim=2)-torch.diff(b,dim=2)).mean()+3*abs(env(ga)-env(b)).mean()/(span/10)
                opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(list(g.parameters())+list(f.parameters()),5);opt.step();losses.append(float(loss.detach()))
            if ep%30==0 or ep==args.epochs:
                row=dict(epoch=ep,loss=float(np.mean(losses)),seconds=round(time.time()-start,1));history.append(row);(root/('full_'+str(seed)+'_history.json')).write_text(json.dumps(history,indent=2));print(seed,json.dumps(row),flush=True)
        d,preds,full=legacy_predictions(g,f,va,legacy);final=metrics(d,preds,alo,ahi,threshold)
        old=load(args.source/'results_long'/('mei180_long_'+str(seed)+'_predictions.npz'));idx=np.concatenate([np.flatnonzero(legacy['ids']==n)[:3] for n in np.unique(legacy['ids'])]);oldpred=dict(generated_audio=old['generated_audio'][idx,:,8:-8],cycle_ACC=old['cycle_ACC'][idx,:,8:-8],ids=legacy['ids'][idx]);oldmetric=metrics(d,oldpred,alo,ahi,threshold)
        result=dict(seed=seed,initial_full=initial,old_local=oldmetric,final=final);results.append(result);(root/'results.json').write_text(json.dumps(results,indent=2))
        np.savez_compressed(root/('full_'+str(seed)+'_common_predictions.npz'),**preds,real_audio=d['y'],real_ACC=d['x']);np.savez_compressed(root/('full_'+str(seed)+'_predictions.npz'),**full);torch.save(dict(g=g.state_dict(),f=f.state_dict()),root/('full_'+str(seed)+'_weights.pt'));print('RESULT',json.dumps(result),flush=True)
        del g,f,da,db;torch.cuda.empty_cache()
if __name__=='__main__':main()
