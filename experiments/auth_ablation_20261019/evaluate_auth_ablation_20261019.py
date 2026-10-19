"""One final evaluation of all prelocked ablation heads; prior conditions on identical sources."""
import os,sys,json,argparse,hashlib,zipfile,shutil
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--root',type=Path,required=True);args=p.parse_args();base=args.baseline;source=args.source;root=args.root;sys.path.insert(0,str(base))
    from train_mei_baseline_auth import FullG,ReturnF,TrainBHead,extract,embed,load
    from evaluate_mei_baseline_auth import interval
    torch.set_num_threads(2);lock=json.loads((root/'heads_locked.json').read_text());lh=sha(root/'heads_locked.json');assert len(lock['models'])==18
    assert sha(root/'selection_locked.json')==lock['selection_lock_sha256'];assert sha(base/'baseline_weights.pt')==lock['baseline_sha256']
    for m in lock['models']:assert sha(root/m['head_file'])==m['head_sha256']
    protocol=json.loads((root/'protocol.json').read_text());assert all(sha(source/(role+'_features.npz'))==h for role,h in protocol['source_feature_sha256'].items());assert json.loads((source/'final_audit.json').read_text())['processed_inputs_disjoint']
    data=load(source/'final_test.npz');manifest=json.loads((source/'split_manifest.json').read_text());assert data['ids'].tolist()==manifest['memberships']['final_test'];assert len(data['ids'])==507
    cp=torch.load(base/'baseline_weights.pt',map_location='cuda',weights_only=True);g=FullG().cuda();f=ReturnF().cuda();g.load_state_dict(cp['g']);f.load_state_dict(cp['f']);g.eval();f.eval();g.requires_grad_(False);f.requires_grad_(False);bank=extract(g,f,data);np.savez_compressed(root/'final_features.npz',**bank,ids=data['ids'],labels=data['labels'])
    prior=json.loads((source/'summary.json').read_text());results=[];private={};genuine=data['labels']==0;attacks=~genuine;users=np.array([x.split('|')[0] for x in data['ids']]);enrollment=dict(np.load(source/'enrollment_features.npz'));train=dict(np.load(source/'head_train_features.npz'))
    for m in lock['models']:
        cp=torch.load(root/m['head_file'],map_location='cuda',weights_only=True);model=TrainBHead(cp['input_dim']).cuda();model.load_state_dict(cp['state']);route=m['route'];template=np.array(m['template'],dtype='float32');scores=embed(model,bank[route])@template;accepted=scores>=m['threshold'];by={}
        ztrain=embed(model,train[route]);center=ztrain[train['labels']==0].mean(0);center/=np.linalg.norm(center);enr=embed(model,enrollment[route]);np.testing.assert_allclose(enr.mean(0)/np.linalg.norm(enr.mean(0)),template,atol=2e-6)
        for u in sorted(set(users[attacks])):
            mask=users==u;n=int(mask.sum());fp=int(accepted[mask].sum());by[u]=dict(n=n,accepted=fp,FAR=fp/n,Wilson95=interval(fp,n))
        ng=int(genuine.sum());ni=int(attacks.sum());tp=int(accepted[genuine].sum());fp=int(accepted[attacks].sum());frr=1-tp/ng;far=fp/ni
        row=dict(condition=m['condition'],route=route,seed=m['seed'],best_epoch=m['best_epoch'],threshold=m['threshold'],selection=m['selection'],calibration=m['calibration'],genuine=dict(n=ng,accepted=tp,TAR=tp/ng,FRR=frr),attack_sources=by,FAR=far,macro_FAR=float(np.mean([v['FAR'] for v in by.values()])),FRR=frr,HTER=(far+frr)/2,AUC=float(roc_auc_score(genuine.astype(int),scores)),TAR=tp/ng,enrollment_to_training_center=float(center@template),enrollment_mean_norm=float(np.linalg.norm(enr.mean(0))))
        results.append(row);private[m['condition']+'_'+route+'_'+str(m['seed'])]=dict(scores=scores.tolist(),ids=data['ids'].tolist(),labels=data['labels'].tolist(),users=users.tolist());print('FINAL',json.dumps(row),flush=True);del model
    priorrows=[]
    for r in prior['results']:
        row=dict(r);row['condition']='old_ke_only' if r['condition']=='ke_only' else 'binary_long';row['TAR']=r['genuine']['TAR'];priorrows.append(row)
    results=priorrows+results;aggregate={}
    for c in ['old_ke_only','binary_long','binary_short','identity_short','identity_episode']:
        aggregate[c]={}
        for route in ['fusion','raw_ACC','cycle_ACC']:
            rr=[r for r in results if r['condition']==c and r['route']==route];assert len(rr)==2;agg={k:float(np.mean([r[k] for r in rr])) for k in ['FAR','macro_FAR','FRR','HTER','AUC','TAR']};agg['sources']={u:dict(n=rr[0]['attack_sources'][u]['n'],FAR=float(np.mean([r['attack_sources'][u]['FAR'] for r in rr]))) for u in rr[0]['attack_sources']};aggregate[c][route]=agg
    assert sha(root/'heads_locked.json')==lh
    summary=dict(protocol=protocol,counts=manifest['counts'],results=results,aggregate=aggregate,assertions=dict(no_deployment_changes_after_final=True,heads_thresholds_locked_before_final=True,all_source_isolation_checks_passed=True,same_final_sources_all_conditions=True,same_frozen_GAN_and_frontend=True),limits='Previously evaluated final partition, exploratory known-impostor identity record holdout. All groups/calibration unchanged. No new blind identity/cross-day claim. All18 candidates fixed before final; do not select a new production winner by these test scores.')
    (root/'summary.json').write_text(json.dumps(summary,indent=2));(root/'private_scores.json').write_text(json.dumps(private));print('EVALUATION_DONE',json.dumps(aggregate),flush=True)
    # Public upload is not attempted: prior report approval rejections remain in force.
    (root/'model_sha256.json').write_text(json.dumps({m['head_file']:m['head_sha256'] for m in lock['models']},indent=2))
    with zipfile.ZipFile(root/'models_private.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in ['protocol.json','heads_locked.json','selection_locked.json','histories.json','summary.json','model_sha256.json']+[m['head_file'] for m in lock['models']]:z.write(root/name,name)
    backup=Path('/autodl-fs/data/auth_ablation_20261019');backup.mkdir(exist_ok=True)
    for name in ['models_private.zip','summary.json','private_scores.json']:shutil.copy2(root/name,backup/name)
    print('BACKUP_DONE',flush=True)
if __name__=='__main__':main()
