"""External attacker evaluation with frozen mei models and thresholds; no training."""
import os,sys,json,argparse,hashlib,zipfile,io
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True);p.add_argument('--archive',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    sys.path.insert(0,str(args.baseline));from train_mei_baseline_auth import FullG,ReturnF,TrainBHead,extract,embed,load
    from prepare_mei_auth_data import frontend
    from evaluate_mei_baseline_auth import interval
    root=args.baseline;out=args.output;out.mkdir(exist_ok=True);torch.set_num_threads(2)
    locksha=sha(root/'heads_locked.json');lock=json.loads((root/'heads_locked.json').read_text());m=json.loads((root/'split_manifest.json').read_text())
    assert sha(root/'baseline_weights.pt')==lock['baseline_sha256']
    for r in lock['models']:assert sha(root/r['head_file'])==r['head_sha256']
    # Audit raw ACC against ALL original sources, including generation TRAIN.
    raw_seen=set()
    with zipfile.ZipFile('/autodl-fs/data/data0723_identity_cyclegan/raw_pairs/data0723_paired_samples.zip') as z:
        for rows in m['groups'].values():
            for r in rows:raw_seen.add(hashlib.sha256(z.read(r['task']+'/'+r['user']+'/'+r['key']+'.npy')).hexdigest())
    original_processed=set()
    cg=load(root.parent/'mei_size_cycle_20261018/full_context_data/full_train.npz') if (root.parent/'mei_size_cycle_20261018/full_context_data/full_train.npz').exists() else None
    # CG cache is in the audio-repair study, resolved without altering it.
    if cg is None:
        import glob
        candidates=glob.glob(str(root.parent/'*/full_context_data/full_train.npz'))
        assert len(candidates)==1,candidates
        cg=load(Path(candidates[0]))
    for x in cg['x']:original_processed.add(hashlib.sha256(x.tobytes()).hexdigest())
    for role in ['gan_check','head_train','head_validation','enrollment','calibration','final_test']:
        for x in load(root/(role+'.npz'))['x']:original_processed.add(hashlib.sha256(x.tobytes()).hexdigest())
    xx=[];ids=[];users=[];tasks=[];excluded=[];quality=[];processed_seen={}
    with zipfile.ZipFile(args.archive) as z:
        rows=json.loads(z.read('manifest.json'))['records']
        for r in rows:
            b=z.read(r['path']);assert hashlib.sha256(b).hexdigest()==r['sha256']
            assert r['sha256'] not in raw_seen,('Raw overlap original',r['path'])
            a=np.load(io.BytesIO(b));t=np.sort(np.unique(a[:,0]));duration=float(t[-1]-t[0])
            # Same frontend only accepts full two-second records. Document input failures;
            # never filter based on a model score, clip rate, or user-specific properties.
            if len(np.arange(0,min(duration,2.),.001))<1997:
                excluded.append(dict(path=r['path'],reason='Too short for frozen 2s frontend',duration=duration));continue
            x,_=frontend(r,z,m['shared_normalization']);h=hashlib.sha256(x.tobytes()).hexdigest()
            assert h not in original_processed,('Processed overlap original',r['path'])
            if h in processed_seen:
                assert processed_seen[h]['user']==r['user'],('Cross-user identical input',r['path'])
                excluded.append(dict(path=r['path'],reason='Identical processed input',duplicate_of=processed_seen[h]['path']));continue
            processed_seen[h]=r;xx.append(x);ids.append(r['user']+'|'+r['key']);users.append(r['user']);tasks.append(r['task']);quality.append(dict(user=r['user'],duration=duration,samples=len(a),clip_fraction=float((abs(x[:195])>=1).mean()),input_mean=float(x.mean()),input_std=float(x.std())))
    data=dict(x=np.array(xx),ids=np.array(ids));assert len(xx)>0
    cp=torch.load(root/'baseline_weights.pt',map_location='cuda',weights_only=True);g=FullG().cuda();f=ReturnF().cuda();g.load_state_dict(cp['g']);f.load_state_dict(cp['f']);g.eval();f.eval();g.requires_grad_(False);f.requires_grad_(False)
    bank=extract(g,f,data);users=np.array(users);tasks=np.array(tasks)
    prior=json.loads((root/'private_scores.json').read_text());summary=[];private={}
    for item in lock['models']:
        route,seed=item['route'],item['seed'];key=route+'_'+str(seed);cp=torch.load(root/item['head_file'],map_location='cuda',weights_only=True);model=TrainBHead(cp['input_dim']).cuda();model.load_state_dict(cp['state']);scores=embed(model,bank[route])@np.array(item['template'],dtype='float32');accepted=scores>=item['threshold'];by={}
        genuine=np.array(prior[key]['scores'])[np.array(prior[key]['labels'])==0];ke=np.array(prior[key]['scores'])[np.array(prior[key]['labels'])==1]
        for user in sorted(set(users)):
            keep=users==user;n=int(keep.sum());nacc=int(accepted[keep].sum());by[user]=dict(n=n,accepted=nacc,FAR=nacc/n,Wilson95=interval(nacc,n),AUC=float(roc_auc_score(np.r_[np.ones(len(genuine)),np.zeros(n)],np.r_[genuine,scores[keep]])),by_task={task:dict(n=int((keep&(tasks==task)).sum()),accepted=int(accepted[keep&(tasks==task)].sum())) for task in sorted(set(tasks[keep]))})
        old_fp=int((ke>=item['threshold']).sum());n=len(scores);fp=int(accepted.sum());frr=float((genuine<item['threshold']).mean());all_scores=np.r_[ke,scores]
        row=dict(route=route,seed=seed,threshold=item['threshold'],genuine=dict(n=len(genuine),accepted=int((genuine>=item['threshold']).sum()),FRR=frr),attack_sources=by,new_attack=dict(n=n,accepted=fp,micro_FAR=fp/n,macro_FAR=float(np.mean([r['FAR'] for r in by.values()])),AUC=float(roc_auc_score(np.r_[np.ones(len(genuine)),np.zeros(n)],np.r_[genuine,scores]))),combined_with_ke=dict(n=len(all_scores),accepted=fp+old_fp,micro_FAR=(fp+old_fp)/len(all_scores),macro_FAR=float(np.mean([r['FAR'] for r in by.values()]+[old_fp/len(ke)])),HTER=((fp+old_fp)/len(all_scores)+frr)/2,AUC=float(roc_auc_score(np.r_[np.ones(len(genuine)),np.zeros(len(all_scores))],np.r_[genuine,all_scores]))))
        summary.append(row);private[key]=dict(scores=scores.tolist(),ids=ids,users=users.tolist(),tasks=tasks.tolist());print('RESULT',json.dumps(row),flush=True);del model
    assert sha(root/'heads_locked.json')==locksha
    public=dict(protocol='Frozen existing mei CycleGAN, six heads, enrollment templates and thresholds. Data0720 identities not in head training/calibration. Added as external attacks only; original 30 genuine and118 ke scores reused, no retraining/tuning. Different collection dates can confound identity. Same shared frozen frontend; no user-specific adjustments.',source_counts={u:int((users==u).sum()) for u in sorted(set(users))},records_before_frontend=len(rows),exclusions=excluded,assertions=dict(raw_overlap_original=False,processed_overlap_original=False,within_new_input_duplicates=False,head_threshold_lock_unchanged=True,baseline_sha256=lock['baseline_sha256'],heads_lock_sha256=locksha),results=summary,input_quality={u:{k:float(np.mean([q[k] for q in quality if q['user']==u])) for k in ['duration','samples','clip_fraction','input_mean','input_std']} for u in sorted(set(users))})
    (out/'summary.json').write_text(json.dumps(public,indent=2));(out/'private_scores.json').write_text(json.dumps(private));(out/'input_quality_private.json').write_text(json.dumps(quality));print('EXTERNAL_EVALUATION_DONE',flush=True)
if __name__=='__main__':main()
