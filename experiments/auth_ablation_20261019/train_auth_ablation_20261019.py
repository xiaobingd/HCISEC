"""Predeclared matched-budget label and enrollment-validation ablations."""
import os,sys,json,argparse,hashlib,copy,time
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
import torch
from torch.utils.data import DataLoader,TensorDataset
from sklearn.metrics import roc_auc_score

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def episode_metric(z,labels,splits):
    neg=np.flatnonzero(labels==1);aucs=[]
    for support,query in splits:
        center=z[support].mean(0);center/=max(np.linalg.norm(center),1e-12)
        ii=np.r_[query,neg];scores=z[ii]@center
        aucs.append(float(roc_auc_score(np.r_[np.ones(len(query)),np.zeros(len(neg))],scores)))
    return dict(mean_AUC=float(np.mean(aucs)),std_AUC=float(np.std(aucs)))

def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args();base=args.baseline;source=args.source;root=args.output;root.mkdir(exist_ok=True);assert not (root/'selection_locked.json').exists();sys.path.insert(0,str(base))
    from train_mei_baseline_auth import TrainBHead,embed,separation,choose_threshold
    from original_trainb_core import SupConLoss
    torch.set_num_threads(2);train=dict(np.load(source/'head_train_features.npz'));val=dict(np.load(source/'head_validation_features.npz'));users=sorted({x.split('|')[0] for x in train['ids']});lookup={u:i for i,u in enumerate(users)};identity=np.array([lookup[x.split('|')[0]] for x in train['ids']]);pos=np.flatnonzero(val['labels']==0);assert len(pos)==10
    rng=np.random.default_rng(20261021);splits=[]
    for _ in range(10):
        perm=rng.permutation(pos);splits.extend([(perm[:5],perm[5:]),(perm[5:],perm[:5])])
    protocol=dict(scope='Exploratory known-impostor identity recording holdout. All sources and final partition historically inspected. No new blind confirmation.',changes=dict(binary_short='Original binary SupCon, original raw binary intra-inter selector,25epochs x60updates=1500, cosine compressed to25epochs.',identity_short='Same initialization, batches, optimizer schedule, updates; only SupCon labels changed to five user identities. Same binary validation selector.',identity_episode='Shares identity_short optimization trajectory exactly. Only checkpoint selection changed to mean20 validation support5/query5 registration episodes AUC against120validation attacks; tie lower episode std then positive cohesion.'),fixed=dict(GAN='Same frozen mei model',input='Same source cached features, shared frontend, no signal or feature normalization changes',architecture='Same512fusion and128projector, inference512',training='Adam.001 WD.0001 temperature.07 clip1 dropout.1 batch32;original balanced1920cacheditems/epoch,25epochs,1500updates;cosine eta1e-6',seeds=[42,43],routes=['fusion','raw_ACC','cycle_ACC'],splits='Source data0720_head_training_20261019 unchanged',registration='Actual10enroll/130calibration read only after ALL checkpoint choices frozen; threshold original calibration EER-balance rule',test='No final arrays/scores read until all18head states/templates/thresholds frozen'),episode_seed=20261021,episode_members=[dict(support=val['ids'][a].tolist(),query=val['ids'][b].tolist()) for a,b in splits],identity_labels=lookup,source_feature_sha256={role:sha(source/(role+'_features.npz')) for role in ['head_train','head_validation']},baseline_sha256=sha(base/'baseline_weights.pt'),limits='Short vs previous6000-update training also compresses the cosine schedule; tests total optimizer-budget package, not step count alone. Validation already explored. No outcome-based source deletion.')
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2));heads=[];histories={}
    for route in protocol['fixed']['routes']:
        for seed in [42,43]:
            for labelmode in ['binary','identity']:
                torch.manual_seed(seed);prng=np.random.default_rng(seed);x=train[route];labs=train['labels'] if labelmode=='binary' else identity;count=max(np.bincount(train['labels']))*2
                idx=np.concatenate([np.resize(prng.permutation(np.flatnonzero(train['labels']==c)),count) for c in [0,1]])
                model=TrainBHead(x.shape[1]).cuda();opt=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0001);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,25,eta_min=1e-6);criterion=SupConLoss(.07)
                loader=DataLoader(TensorDataset(torch.as_tensor(x[idx]),torch.as_tensor(labs[idx])),batch_size=32,shuffle=True,drop_last=True);assert len(loader)==60
                selected={};hist=[];conditions=['binary_short'] if labelmode=='binary' else ['identity_short','identity_episode']
                for epoch in range(1,26):
                    model.train();losses=[]
                    for a,y in loader:
                        loss=criterion(model(a.cuda()),y.cuda());opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();losses.append(float(loss.detach()))
                    scheduler.step();z=embed(model,val[route]);sep=separation(z,val['labels']);ep=episode_metric(z,val['labels'],splits);pp=pos[:,None]!=pos[None,:];psim=z[pos]@z[pos].T;cohesion=float(psim[pp].mean());row=dict(epoch=epoch,updates=epoch*60,loss=float(np.mean(losses)),binary_separation=sep,episodes=ep,positive_cohesion=cohesion);hist.append(row)
                    for condition in conditions:
                        metric=(sep['sep'],) if condition!='identity_episode' else (ep['mean_AUC'],-ep['std_AUC'],cohesion)
                        if condition not in selected or metric>selected[condition]['metric']:selected[condition]=dict(metric=metric,epoch=epoch,state=copy.deepcopy(model.state_dict()),selection=row)
                for condition,v in selected.items():
                    filename=f'{condition}_{route}_{seed}.pt';torch.save(dict(state=v['state'],input_dim=x.shape[1],condition=condition,route=route,seed=seed,epoch=v['epoch']),root/filename);item=dict(condition=condition,route=route,seed=seed,head_file=filename,head_sha256=sha(root/filename),best_epoch=v['epoch'],selection=v['selection']);heads.append(item);print('CHECKPOINT_SELECTED',condition,route,seed,v['epoch'],flush=True)
                histories[f'{labelmode}_{route}_{seed}']=hist;del model;torch.cuda.empty_cache()
    assert len(heads)==18
    (root/'histories.json').write_text(json.dumps(histories,indent=2));(root/'selection_locked.json').write_text(json.dumps(dict(models=heads,barrier='All checkpoint choices frozen using head train/validation only. Actual enrollment/calibration/final arrays and scores have not been read.'),indent=2));print('ALL_SELECTION_LOCKED',flush=True)
    # Post-selection enrollment and calibration are isolated from checkpoint selection.
    enrollment=dict(np.load(source/'enrollment_features.npz'));cal=dict(np.load(source/'calibration_features.npz'));locked=[]
    for item in heads:
        cp=torch.load(root/item['head_file'],map_location='cuda',weights_only=True);model=TrainBHead(cp['input_dim']).cuda();model.load_state_dict(cp['state']);z=embed(model,enrollment[item['route']]);template=z.mean(0);template/=max(np.linalg.norm(template),1e-12);scores=embed(model,cal[item['route']])@template;positive=cal['labels']==0;t=choose_threshold(scores[positive],scores[~positive]);item=dict(item,template=template.tolist(),threshold=float(t),calibration=dict(genuine_n=int(positive.sum()),impostor_n=int((~positive).sum()),FAR=float((scores[~positive]>=t).mean()),FRR=float((scores[positive]<t).mean())));locked.append(item);del model
    (root/'heads_locked.json').write_text(json.dumps(dict(models=locked,baseline_sha256=protocol['baseline_sha256'],selection_lock_sha256=sha(root/'selection_locked.json'),barrier='All18 model states, enrollment templates and calibration thresholds frozen before final export/evaluation.'),indent=2));print('ALL_HEADS_LOCKED',flush=True)
if __name__=='__main__':main()
