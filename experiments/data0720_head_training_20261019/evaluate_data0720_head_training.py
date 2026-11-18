"""Compare expanded and previous heads on identical recording-disjoint final sources."""
import os,sys,json,argparse,hashlib,zipfile
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True);p.add_argument('--root',type=Path,required=True);args=p.parse_args();base=args.baseline;root=args.root;sys.path.insert(0,str(base))
    from train_mei_baseline_auth import FullG,ReturnF,TrainBHead,extract,embed,load
    from evaluate_mei_baseline_auth import interval
    torch.set_num_threads(2);locked=json.loads((root/'heads_locked.json').read_text());locksha=sha(root/'heads_locked.json');assert len(locked['models'])==6
    assert sha(root/'baseline_weights.pt')==sha(base/'baseline_weights.pt')==locked['baseline_sha256']
    for item in locked['models']:assert sha(root/item['head_file'])==item['head_sha256']
    final=load(root/'final_test.npz');manifest=json.loads((root/'split_manifest.json').read_text());assert final['ids'].tolist()==manifest['memberships']['final_test'];assert len(final['ids'])==507
    assert int((final['labels']==0).sum())==30 and int((final['labels']==1).sum())==477
    assert json.loads((root/'final_audit.json').read_text())['processed_inputs_disjoint']
    # Verify original sources copied byte-for-byte into their original roles.
    for role in ['head_train','head_validation','enrollment','calibration','final_test']:
        old=load(base/(role+'.npz'));new=load(root/(role+'.npz'));n=len(old['ids']);assert np.array_equal(old['ids'],new['ids'][:n]) and np.array_equal(old['x'],new['x'][:n])
    cp=torch.load(root/'baseline_weights.pt',map_location='cuda',weights_only=True);g=FullG().cuda();f=ReturnF().cuda();g.load_state_dict(cp['g']);f.load_state_dict(cp['f']);g.eval();f.eval();g.requires_grad_(False);f.requires_grad_(False);bank=extract(g,f,final)
    old_lock=json.loads((base/'heads_locked.json').read_text());old_scores=json.loads((base/'private_scores.json').read_text());external=json.loads((base.parent/'data0720_attacks_20261019/private_scores.json').read_text());users=np.array([s.split('|')[0] for s in final['ids']]);genuine=final['labels']==0;attacks=~genuine;results=[];private={}
    for condition,models in [('ke_only',old_lock['models']),('expanded_negatives',locked['models'])]:
        for item in models:
            route,seed=item['route'],item['seed'];key=route+'_'+str(seed)
            if condition=='ke_only':
                lookup=dict(zip(old_scores[key]['ids'],old_scores[key]['scores']));lookup.update(zip(external[key]['ids'],external[key]['scores']));scores=np.array([lookup[i] for i in final['ids']])
            else:
                cp=torch.load(root/item['head_file'],map_location='cuda',weights_only=True);model=TrainBHead(cp['input_dim']).cuda();model.load_state_dict(cp['state']);scores=embed(model,bank[route])@np.array(item['template'],dtype='float32');del model
            accepted=scores>=item['threshold'];sources={}
            for u in sorted(set(users[attacks])):
                mask=users==u;n=int(mask.sum());fp=int(accepted[mask].sum());sources[u]=dict(n=n,accepted=fp,FAR=fp/n,Wilson95=interval(fp,n),AUC=float(roc_auc_score(np.r_[np.ones(int(genuine.sum())),np.zeros(n)],np.r_[scores[genuine],scores[mask]])))
            ng=int(genuine.sum());ni=int(attacks.sum());fn=int((~accepted&genuine).sum());fp=int((accepted&attacks).sum());far=fp/ni;frr=fn/ng;newmask=attacks&(users!='chengkeyi')
            row=dict(condition=condition,route=route,seed=seed,threshold=item['threshold'],best_epoch=item['best_epoch'],genuine=dict(n=ng,accepted=ng-fn,TAR=(ng-fn)/ng,FRR=frr,TAR_Wilson95=interval(ng-fn,ng)),attack_sources=sources,FAR=far,macro_FAR=float(np.mean([v['FAR'] for v in sources.values()])),FRR=frr,HTER=(far+frr)/2,AUC=float(roc_auc_score(genuine.astype(int),scores)),new_attack_FAR=float(accepted[newmask].mean()),calibration=item['calibration'])
            results.append(row);private[condition+'_'+key]=dict(scores=scores.tolist(),ids=final['ids'].tolist(),users=users.tolist(),labels=final['labels'].tolist(),tasks=final['tasks'].tolist());print('FINAL',json.dumps(row),flush=True)
    aggregate={}
    for condition in ['ke_only','expanded_negatives']:
        aggregate[condition]={}
        for route in ['fusion','raw_ACC','cycle_ACC']:
            rr=[r for r in results if r['condition']==condition and r['route']==route];agg={k:float(np.mean([r[k] for r in rr])) for k in ['FAR','macro_FAR','FRR','HTER','AUC','new_attack_FAR']};agg['TAR']=1-agg['FRR'];agg['sources']={u:dict(n=rr[0]['attack_sources'][u]['n'],FAR=float(np.mean([r['attack_sources'][u]['FAR'] for r in rr]))) for u in rr[0]['attack_sources']};aggregate[condition][route]=agg
    assert sha(root/'heads_locked.json')==locksha
    public=dict(target='wangjingmei',counts=manifest['counts'],scope=manifest['scope'],limits=manifest['limits'],partition_rule=manifest['partition_rule'],partition_seeds=manifest['partition_seeds'],results=results,aggregate=aggregate,assertions=dict(manifest['assertions'],actual_model_inputs_all_roles_disjoint=True,original_group_arrays_unchanged=True,heads_and_thresholds_locked_before_final_export=True,one_identical_frozen_generator=True),comparison='Ke-only and expanded heads evaluated on same30genuine/477attacks, including359 Data0720. Original condition reuses exact prior frozen scores; expanded refit uses same original algorithm but60vs15updates/epoch and expanded130calibration. Entire899 Data0720 sources previously diagnostic-scored; heldout partition is exploratory, not pristine.',training=dict(positive_unique=30,negative_unique=480,negative_users=4,head_epochs=100,updates_per_epoch=60,prior_updates_per_epoch=15,seeds=[42,43],normalization='Same frozen original shared frontend; no per-user changes; generation model not retrained.'))
    (root/'summary.json').write_text(json.dumps(public,indent=2));(root/'private_scores.json').write_text(json.dumps(private));print('EVALUATION_DONE',json.dumps(aggregate),flush=True)
    with zipfile.ZipFile(root/'delivery.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in ['summary.json','private_scores.json','split_manifest.json','heads_locked.json','head_protocol.json','head_histories.json','development_audit.json','final_audit.json']+[i['head_file'] for i in locked['models']]:z.write(root/name,name)
        # The baseline remains the identical previously delivered model.
        z.writestr('baseline_reference.txt',locked['baseline_sha256'])
    print('DELIVERY_DONE',flush=True)
if __name__=='__main__':main()
