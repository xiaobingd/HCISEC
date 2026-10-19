"""Extend original authentication splits with recording-disjoint Data0720 negatives."""
import os,sys,json,argparse,hashlib,zipfile,shutil
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True);p.add_argument('--archive',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--stage',choices=['development','final'],default='development');args=p.parse_args()
    base=args.baseline;root=args.output;root.mkdir(exist_ok=True);sys.path.insert(0,str(base))
    from prepare_mei_auth_data import frontend,partition
    original=json.loads((base/'split_manifest.json').read_text());norm=original['shared_normalization']
    with zipfile.ZipFile(args.archive) as z:
        rows=json.loads(z.read('manifest.json'))['records'];users=sorted({r['user'] for r in rows});added={k:[] for k in ['head_train','head_validation','calibration','final_test']}
        for i,u in enumerate(users):
            rr=[r for r in rows if r['user']==u];assert len(rr) in [299,300]
            groups=partition(rr,[120,30,30,len(rr)-180],20261020+i)
            for role,g in zip(added,groups):added[role].extend(g)
        memberships={role:original['memberships'][role]+[r['user']+'|'+r['key'] for r in added.get(role,[])] for role in original['groups']}
        for a,aa in memberships.items():
            for b,bb in memberships.items():
                if a!=b:assert not set(aa)&set(bb),(a,b)
        if args.stage=='development':
            assert not (root/'heads_locked.json').exists() and not (root/'final_test.npz').exists()
            rawhashes={}
            with zipfile.ZipFile('/autodl-fs/data/data0723_identity_cyclegan/raw_pairs/data0723_paired_samples.zip') as zz:
                for role,rr in original['groups'].items():
                    for r in rr:rawhashes[r['user']+'|'+r['key']]=hashlib.sha256(zz.read(r['task']+'/'+r['user']+'/'+r['key']+'.npy')).hexdigest()
            for r in rows:
                b=z.read(r['path']);assert hashlib.sha256(b).hexdigest()==r['sha256'];rawhashes[r['user']+'|'+r['key']]=r['sha256']
            hs={k:{rawhashes[i] for i in ids} for k,ids in memberships.items()}
            for a,aa in hs.items():
                for b,bb in hs.items():
                    if a!=b:assert not aa&bb,('raw duplicate',a,b)
            counts={role:{u:sum(i.split('|')[0]==u for i in ids) for u in ['wangjingmei','chengkeyi']+users} for role,ids in memberships.items()}
            proto=dict(scope='All four attack identities are seen in head training. Recording-disjoint exploratory authentication, not unseen-impostor identity open-set.',counts=counts,memberships=memberships,groups={role:rr+added.get(role,[]) for role,rr in original['groups'].items()},shared_normalization=norm,added_splits=added,partition_seeds={u:20261020+i for i,u in enumerate(users)},partition_rule='Each additional identity120headtrain/30headvalidation/30thresholdcalibration/remaining120or119final; original mei and ke splits unchanged. Fixed task-stratified RNG, independent of score. No removals by outcome.',assertions=dict(source_ids_disjoint=True,raw_ACC_hashes_disjoint=True,normalization_frozen=True,original_target_and_ke_memberships_unchanged=True),limits='All899 Data0720 sources already scored by the earlier frozen-model diagnostic; future final subset therefore not pristine independent confirmation. No final scores used for new head/threshold selection. Both dates/users can confound identity. Larger negative count also increases original-trainer updates per epoch (60 vs15), so improvements are not data-only causal estimates.',baseline_sha256=hashlib.sha256((base/'baseline_weights.pt').read_bytes()).hexdigest())
            (root/'split_manifest.json').write_text(json.dumps(proto,indent=2))
            os.link(base/'baseline_weights.pt',root/'baseline_weights.pt')
            shutil.copyfile(base/'enrollment.npz',root/'enrollment.npz')
            selected=['head_train','head_validation','calibration']
        else:
            assert (root/'heads_locked.json').exists(),'Freeze heads/templates/thresholds before final export'
            proto=json.loads((root/'split_manifest.json').read_text());assert proto['memberships']==memberships
            selected=['final_test']
        for role in selected:
            old=np.load(base/(role+'.npz'));xx=[x for x in old['x']];ids=old['ids'].tolist();tasks=old['tasks'].tolist();labs=old['labels'].tolist()
            for r in added[role]:
                x,_=frontend(r,z,norm);xx.append(x);ids.append(r['user']+'|'+r['key']);tasks.append(r['task']);labs.append(1)
            np.savez_compressed(root/(role+'.npz'),x=np.array(xx),ids=np.array(ids),tasks=np.array(tasks),labels=np.array(labs));print('EXPORTED',role,len(xx),flush=True)
        # Stage-wise input content checks cover every available role and original CG train.
        from train_mei_baseline_auth import load
        cg=load(base.parent/'mei_audio_repair_20261018/full_context_data/full_train.npz')
        arrays={'gan_train':cg['x'],'gan_check':load(base/'gan_check.npz')['x']}
        for role in ['head_train','head_validation','enrollment','calibration','final_test']:
            if (root/(role+'.npz')).exists():arrays[role]=np.load(root/(role+'.npz'))['x']
        hs={role:{hashlib.sha256(x.tobytes()).hexdigest() for x in arr} for role,arr in arrays.items()}
        for a,aa in hs.items():
            assert len(aa)==len(arrays[a]),('within-role duplicate',a)
            for b,bb in hs.items():
                if a!=b:assert not aa&bb,('processed duplicate',a,b)
        (root/(args.stage+'_audit.json')).write_text(json.dumps(dict(processed_inputs_disjoint=True,counts={k:len(v) for k,v in arrays.items()},stage=args.stage),indent=2))
        print('PREPARE_DONE',args.stage,flush=True)
if __name__=='__main__':main()
