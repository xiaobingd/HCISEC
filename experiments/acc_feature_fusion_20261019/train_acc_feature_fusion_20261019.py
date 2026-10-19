"""Add pre-cycle ACC statistics to frozen CycleGAN features, using the fixed episodic protocol."""
import os,sys,json,argparse,hashlib,copy,zipfile,shutil
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
import torch
from torch.utils.data import DataLoader,TensorDataset
from sklearn.metrics import roc_auc_score

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def combine(d,route):
    other='fusion' if route=='fusion_plus_ACC' else 'cycle_ACC'
    return np.concatenate([d[other],d['raw_ACC']],axis=1)

def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--prior',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args();base=args.baseline;source=args.source;prior=args.prior;root=args.output;root.mkdir(exist_ok=True);assert not (root/'selection_locked.json').exists();sys.path.insert(0,str(base));sys.path.insert(0,str(base.parent))
    from train_mei_baseline_auth import TrainBHead,embed,choose_threshold
    from original_trainb_core import SupConLoss
    from train_auth_ablation_20261019 import episode_metric
    from evaluate_mei_baseline_auth import interval
    torch.set_num_threads(2);train=dict(np.load(source/'head_train_features.npz'));val=dict(np.load(source/'head_validation_features.npz'));priorprotocol=json.loads((prior/'protocol.json').read_text());index={str(i):k for k,i in enumerate(val['ids'])};episodes=[(np.array([index[i] for i in e['support']]),np.array([index[i] for i in e['query']])) for e in priorprotocol['episode_members']];lookup=priorprotocol['identity_labels'];identity=np.array([lookup[i.split('|')[0]] for i in train['ids']]);pos=np.flatnonzero(val['labels']==0);routes=['fusion_plus_ACC','cycle_plus_ACC']
    protocol=dict(method='Concatenate original frozen per-record feature vectors, then original TrainBHead/SupCon with five identity labels and prior validation5support5query episode selector. No new signal processing, normalization, per-user filter, or trainable signal encoder.',routes=dict(fusion_plus_ACC='196CycleGANmultisource +502precycleACCmeanstd =698',cycle_plus_ACC='502cycleACCmeanstd +502precycleACCmeanstd =1004'),training='Same previous identity_episode:1500updates=25epochs60batches32,Adam.001WD.0001,cosine25 eta1e-6,temp.07,clip1,dropout.1, seeds42/43. Same binary-balance sampling then identity loss labels.',selection='Exactly previous20 validation episodes and tie rules; all checkpoint choices fixed before actual enrollment/calibration access.',registration='Independent original10enrollment,130calibration; original threshold selection; final30genuine477attacks afterall4locks.',feature_definition='Raw/inputACC means temporalmean/std of same preprocessed251-channel input (502D), NOT raw waveform samples.',source_hashes={k:sha(source/(k+'_features.npz')) for k in ['head_train','head_validation']},prior_protocol_sha256=sha(prior/'protocol.json'),baseline_sha256=sha(base/'baseline_weights.pt'),limits='Previously explored final partition, known-impostor identities. Wider input increases first layer parameters, so feature and capacity changes are not isolated. No outcome-based removals. Same generator single seed, two headinitializations only.')
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2));selected=[];histories={}
    for route in routes:
        x=combine(train,route);vx=combine(val,route);assert x.shape[1]==(698 if route=='fusion_plus_ACC' else 1004)
        for seed in [42,43]:
            torch.manual_seed(seed);rng=np.random.default_rng(seed);count=max(np.bincount(train['labels']))*2;idx=np.concatenate([np.resize(rng.permutation(np.flatnonzero(train['labels']==c)),count) for c in [0,1]])
            model=TrainBHead(x.shape[1]).cuda();opt=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0001);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,25,eta_min=1e-6);criterion=SupConLoss(.07);loader=DataLoader(TensorDataset(torch.as_tensor(x[idx]),torch.as_tensor(identity[idx])),batch_size=32,shuffle=True,drop_last=True);assert len(loader)==60;best=None;history=[]
            for epoch in range(1,26):
                model.train();losses=[]
                for a,y in loader:
                    loss=criterion(model(a.cuda()),y.cuda());opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();losses.append(float(loss.detach()))
                scheduler.step();z=embed(model,vx);ep=episode_metric(z,val['labels'],episodes);sim=z[pos]@z[pos].T;cohesion=float(sim[~np.eye(len(pos),dtype=bool)].mean());metric=(ep['mean_AUC'],-ep['std_AUC'],cohesion);row=dict(epoch=epoch,updates=epoch*60,loss=float(np.mean(losses)),episodes=ep,positive_cohesion=cohesion);history.append(row)
                if best is None or metric>best['metric']:best=dict(metric=metric,state=copy.deepcopy(model.state_dict()),epoch=epoch,selection=row)
            name=f'{route}_{seed}.pt';torch.save(dict(state=best['state'],input_dim=x.shape[1],route=route,seed=seed,epoch=best['epoch']),root/name);selected.append(dict(route=route,seed=seed,head_file=name,head_sha256=sha(root/name),input_dim=x.shape[1],parameter_count=sum(p.numel() for p in model.parameters()),best_epoch=best['epoch'],selection=best['selection']));histories[f'{route}_{seed}']=history;print('SELECTED',route,seed,best['epoch'],flush=True);del model
    (root/'histories.json').write_text(json.dumps(histories,indent=2));(root/'selection_locked.json').write_text(json.dumps(dict(models=selected,barrier='All4selections fixed usingheadtrain/valonly beforeactualenrollment/calibration/finalaccess.'),indent=2));print('ALL_SELECTION_LOCKED',flush=True)
    enrollment=dict(np.load(source/'enrollment_features.npz'));cal=dict(np.load(source/'calibration_features.npz'));locked=[]
    for m in selected:
        cp=torch.load(root/m['head_file'],map_location='cuda',weights_only=True);model=TrainBHead(cp['input_dim']).cuda();model.load_state_dict(cp['state']);z=embed(model,combine(enrollment,m['route']));template=z.mean(0);template/=max(np.linalg.norm(template),1e-12);scores=embed(model,combine(cal,m['route']))@template;positive=cal['labels']==0;t=choose_threshold(scores[positive],scores[~positive]);locked.append(dict(m,template=template.tolist(),threshold=float(t),calibration=dict(FAR=float((scores[~positive]>=t).mean()),FRR=float((scores[positive]<t).mean()))));del model
    (root/'heads_locked.json').write_text(json.dumps(dict(models=locked,selection_sha256=sha(root/'selection_locked.json'),baseline_sha256=protocol['baseline_sha256'],barrier='All4heads/templates/thresholds frozen before finalfeatures/scores read.'),indent=2));lh=sha(root/'heads_locked.json');print('ALL_HEADS_LOCKED',flush=True)
    assert json.loads((source/'final_audit.json').read_text())['processed_inputs_disjoint'];final=dict(np.load(prior/'final_features.npz'));manifest=json.loads((source/'split_manifest.json').read_text());assert final['ids'].tolist()==manifest['memberships']['final_test'] and len(final['ids'])==507;genuine=final['labels']==0;attack=~genuine;users=np.array([i.split('|')[0] for i in final['ids']]);results=[];private={}
    for m in locked:
        assert sha(root/m['head_file'])==m['head_sha256'];cp=torch.load(root/m['head_file'],map_location='cuda',weights_only=True);model=TrainBHead(cp['input_dim']).cuda();model.load_state_dict(cp['state']);scores=embed(model,combine(final,m['route']))@np.array(m['template'],dtype='float32');accepted=scores>=m['threshold'];by={}
        for u in sorted(set(users[attack])):
            mask=users==u;n=int(mask.sum());fp=int(accepted[mask].sum());by[u]=dict(n=n,accepted=fp,FAR=fp/n,Wilson95=interval(fp,n))
        tp=int(accepted[genuine].sum());fp=int(accepted[attack].sum());far=fp/int(attack.sum());frr=1-tp/int(genuine.sum());row=dict(route=m['route'],seed=m['seed'],best_epoch=m['best_epoch'],threshold=m['threshold'],input_dim=m['input_dim'],parameter_count=m['parameter_count'],genuine=dict(n=int(genuine.sum()),accepted=tp,TAR=1-frr,FRR=frr),attack_sources=by,TAR=1-frr,FAR=far,FRR=frr,HTER=(far+frr)/2,AUC=float(roc_auc_score(genuine.astype(int),scores)));results.append(row);private[m['route']+'_'+str(m['seed'])]=dict(scores=scores.tolist(),ids=final['ids'].tolist(),labels=final['labels'].tolist(),users=users.tolist());print('FINAL',json.dumps(row),flush=True);del model
    priorresult=json.loads((prior/'summary.json').read_text());aggregate={route:priorresult['aggregate']['identity_episode'][route] for route in ['raw_ACC','fusion','cycle_ACC']}
    for route in routes:
        rr=[r for r in results if r['route']==route];a={k:float(np.mean([r[k] for r in rr])) for k in ['TAR','FAR','FRR','HTER','AUC']};a['sources']={u:dict(n=rr[0]['attack_sources'][u]['n'],FAR=float(np.mean([r['attack_sources'][u]['FAR'] for r in rr]))) for u in rr[0]['attack_sources']};aggregate[route]=a
    assert sha(root/'heads_locked.json')==lh;assert sha(base/'baseline_weights.pt')==protocol['baseline_sha256'];assert all(sha(source/(r+'_features.npz'))==h for r,h in protocol['source_hashes'].items())
    summary=dict(protocol=protocol,results=results,aggregate=aggregate,controls='Previous identity_episode rawACC/fusion/cycleACC onidentical507sources andsamelearningrecipe; reusedfrozenresults.',counts=manifest['counts'],assertions=dict(same_source_ids=True,all_source_isolation_checks_passed=True,same_shared_frontend=True,no_actual_enrollment_used_for_selection=True,heads_thresholds_locked_before_final=True,source_and_generator_hashes_unchanged=True))
    (root/'summary.json').write_text(json.dumps(summary,indent=2));(root/'private_scores.json').write_text(json.dumps(private));print('EVALUATION_DONE',json.dumps(aggregate),flush=True)
    with zipfile.ZipFile(root/'models_private.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in ['protocol.json','histories.json','selection_locked.json','heads_locked.json','summary.json']+[m['head_file'] for m in locked]:z.write(root/name,name)
    backup=Path('/autodl-fs/data/acc_feature_fusion_20261019');backup.mkdir(exist_ok=True)
    for name in ['models_private.zip','summary.json','private_scores.json']:shutil.copy2(root/name,backup/name)
    print('BACKUP_DONE',flush=True)
if __name__=='__main__':main()
