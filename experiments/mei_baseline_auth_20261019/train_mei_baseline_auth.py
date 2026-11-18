"""Original trainb SupCon/fusion head, with an explicit current-baseline adapter."""
import os,sys,json,argparse,hashlib,copy,time
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader,TensorDataset
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'mei_audio_repair_20261018'))
from train_mei_full_context import FullG,ReturnF,load
from original_trainb_core import SupConLoss,ProjectionHead,FeatureAggregator

class TrainBHead(nn.Module):
    def __init__(self,input_dim):
        super().__init__()
        # Identical trainb modelb fusion and projector, only input feature width changes.
        self.fusion=nn.Sequential(nn.Linear(input_dim,512),nn.BatchNorm1d(512),nn.ReLU(inplace=True),nn.Dropout(.1),nn.Linear(512,512),nn.BatchNorm1d(512))
        self.projector=ProjectionHead(512,256,128)
    def forward(self,x,embedding=False):
        h=self.fusion(x)
        return F.normalize(h,p=2,dim=1) if embedding else self.projector(h)

@torch.no_grad()
def extract(g,f,data):
    out={k:[] for k in ['fusion','raw_ACC','cycle_ACC']};g.eval();f.eval();maps={}
    hooks=[g.decoder.model[6].register_forward_hook(lambda m,i,o:maps.update(shallow=o)),g.decoder.model[12].register_forward_hook(lambda m,i,o:maps.update(deep=o))]
    gap=FeatureAggregator(32,'gap');deepagg=FeatureAggregator(64,'gap_std')
    for i in range(0,len(data['ids']),12):
        x=torch.as_tensor(data['x'][i:i+12],device='cuda');a=g(x);rec=f(a);shallow=maps['shallow'];deep=maps['deep'];assert shallow.shape[1]==32 and deep.shape[1]==64
        diff=x-rec;flat=diff.flatten(1);stats=torch.stack([flat.mean(1),flat.std(1),flat.abs().amax(1),flat.square().mean(1).sqrt()],1)
        # The shared 251->80 projection is the existing G input interface;
        # subtract projected x/rec to cancel its bias rather than projecting zero diff.
        diffmap=(g.projection(x)-g.projection(rec))[:,None]
        diff_feat=g.decoder.model[:7](diffmap)
        fused=torch.cat([gap(shallow),deepagg(deep),gap(diff_feat),stats],1);assert fused.shape[1]==196
        out['fusion'].append(fused.cpu().numpy());out['raw_ACC'].append(torch.cat([x.mean(-1),x.std(-1)],1).cpu().numpy());out['cycle_ACC'].append(torch.cat([rec.mean(-1),rec.std(-1)],1).cpu().numpy())
    for h in hooks:h.remove()
    return {k:np.concatenate(v) for k,v in out.items()}

@torch.no_grad()
def embed(model,x):
    model.eval();out=[]
    for i in range(0,len(x),64):out.append(model(torch.as_tensor(x[i:i+64],device='cuda'),embedding=True).cpu().numpy())
    return np.concatenate(out)

def separation(z,labels):
    sim=z@z.T;eq=labels[:,None]==labels[None];off=~np.eye(len(z),dtype=bool)
    intra=float(sim[eq&off].mean());inter=float(sim[(~eq)&off].mean());return dict(intra=intra,inter=inter,sep=intra-inter)

def choose_threshold(genuine,impostor):
    vals=np.unique(np.r_[genuine,impostor]);candidates=np.r_[vals[0]-1e-6,(vals[:-1]+vals[1:])/2,vals[-1]+1e-6] if len(vals)>1 else vals;best=None
    for t in candidates:
        far=float(np.mean(impostor>=t));frr=float(np.mean(genuine<t));key=(abs(far-frr),(far+frr)/2)
        if best is None or key<best[0]:best=(key,float(t),far,frr)
    return best[1]

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--seeds',nargs='+',type=int,default=[42,43]);args=p.parse_args();root=args.root;torch.set_num_threads(2);assert not (root/'final_test.npz').exists(),'Final features must not exist until head selection/calibration are locked'
    manifest=json.loads((root/'split_manifest.json').read_text());cp=torch.load(root/'baseline_weights.pt',map_location='cuda',weights_only=True);g=FullG().cuda();f=ReturnF().cuda();g.load_state_dict(cp['g']);f.load_state_dict(cp['f']);g.requires_grad_(False);f.requires_grad_(False)
    cache={};groups={}
    for group in ['head_train','head_validation','enrollment','calibration']:
        data=load(root/(group+'.npz'));groups[group]=data;cache[group]=extract(g,f,data);np.savez_compressed(root/(group+'_features.npz'),**cache[group],labels=data['labels'],ids=data['ids']);print('CACHED',group,len(data['ids']),flush=True)
    proto=dict(core_origin='User E:/Edge下载/train_core/对比学习特征处理/modelb.py; SupConLoss,ProjectionHead,FeatureAggregator copied verbatim via AST source extraction, whole original file SHA25605c7c8e40e427e5529346bd360940ef35a7eebd028c421b3fc119d37e1f6f153.',adapter='CurrentG251->80 shared input projection, decoder shallow[:7]=32channels, truebottleneck[:13]=64channels, residual uses projectedx-projectedF(Gx) to cancel inputprojectionbias, 4originaldiffstats onx-F(Gx). Fusion196->512->512; proj512->256->128. Legacyoldmodel usesdifferentchannelwidth/cuts, so not exact wholetrainb replay.',head='Original Adam lr.001 WD.0001, cosine100epoch eta1e-6, SupCon temperature/base.07, clip1,32batchshuffle/drop_last, best validation512embedding intra-inter separation; inference normalizedfusion512 notprojector128.',sampling='Cached frozen G/F; each class duplicated to2*largestclass=240items; no fakeusers or simulatedidentitylabels. No old80x80imageaugment onheterogeneous251spectrum+energy rows; no signalaugment, balancing duplicates are not new recordings.',routes=['fusion','raw_ACC','cycle_ACC'],controls='Same SupCon/fusion/projector and splits for rawmean/std502 and cycleACCmean/std502; inputwidthdiffers fromfusion196. SingletrainedGseed42, two headinitializations42/43. Allheadgroupsrecorddisjoint; thresholds fixed on separatecalibration beforefinal export. Ke identity seen inTRAIN; notunseen-useropen-set.',seeds=args.seeds)
    (root/'head_protocol.json').write_text(json.dumps(proto,indent=2));locked=[];histories={}
    for route in proto['routes']:
        for seed in args.seeds:
            torch.manual_seed(seed);rng=np.random.default_rng(seed);model=TrainBHead(cache['head_train'][route].shape[1]).cuda();opt=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0001);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=100,eta_min=1e-6);criterion=SupConLoss(.07)
            features=cache['head_train'][route];labels=groups['head_train']['labels'];count=max(np.bincount(labels))*2;idx=np.concatenate([np.resize(rng.permutation(np.flatnonzero(labels==c)),count) for c in [0,1]])
            ds=TensorDataset(torch.as_tensor(features[idx]),torch.as_tensor(labels[idx]));loader=DataLoader(ds,batch_size=32,shuffle=True,drop_last=True);best_sep=-np.inf;best=None;history=[];best_epoch=None
            for ep in range(1,101):
                model.train();losses=[]
                for a,y in loader:
                    loss=criterion(model(a.cuda()),y.cuda());opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();losses.append(float(loss.detach()))
                scheduler.step();val=embed(model,cache['head_validation'][route]);score=separation(val,groups['head_validation']['labels']);row=dict(epoch=ep,loss=float(np.mean(losses)),validation=score);history.append(row)
                if score['sep']>best_sep:best_sep=score['sep'];best=copy.deepcopy(model.state_dict());best_epoch=ep
            model.load_state_dict(best);path=root/(route+'_'+str(seed)+'_head.pt');torch.save(dict(state=best,input_dim=features.shape[1],route=route,seed=seed,epoch=best_epoch),path)
            enr=embed(model,cache['enrollment'][route]);template=enr.mean(0);template/=max(np.linalg.norm(template),1e-9);cal=embed(model,cache['calibration'][route])@template;labs=groups['calibration']['labels'];threshold=choose_threshold(cal[labs==0],cal[labs==1])
            item=dict(route=route,seed=seed,best_epoch=best_epoch,validation_separation=float(best_sep),threshold=float(threshold),template=template.tolist(),head_file=path.name,head_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),calibration=dict(genuine_n=int((labs==0).sum()),impostor_n=int((labs==1).sum()),FAR=float(np.mean(cal[labs==1]>=threshold)),FRR=float(np.mean(cal[labs==0]<threshold))))
            locked.append(item);histories[route+'_'+str(seed)]=history;print('HEAD_LOCKED',route,seed,best_epoch,item['calibration'],flush=True);del model;torch.cuda.empty_cache()
    (root/'head_histories.json').write_text(json.dumps(histories,indent=2));(root/'heads_locked.json').write_text(json.dumps(dict(models=locked,protocol=proto,baseline_sha256=hashlib.sha256((root/'baseline_weights.pt').read_bytes()).hexdigest(),barrier='No final arrays were present or read before these models, templates and thresholds were frozen.'),indent=2));print('ALL_HEADS_LOCKED',flush=True)
if __name__=='__main__':main()
