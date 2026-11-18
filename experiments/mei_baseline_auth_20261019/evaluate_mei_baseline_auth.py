"""One final evaluation after frozen models, enrollment and thresholds."""
import os,sys,json,argparse,hashlib,zipfile
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
import torch
from sklearn.metrics import roc_auc_score,roc_curve
from train_mei_baseline_auth import FullG,ReturnF,TrainBHead,extract,embed,load
from train_mei_full_context import metrics
from prepare_mei_auth_data import frontend

def interval(k,n):
    z=1.95996398454;p=k/n;den=1+z*z/n;mid=(p+z*z/(2*n))/den;half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [float(max(0,mid-half)),float(min(1,mid+half))]

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--prior',type=Path,required=True);args=p.parse_args();root=args.root;torch.set_num_threads(2)
    locked=json.loads((root/'heads_locked.json').read_text());assert len(locked['models'])==6
    assert hashlib.sha256((root/'baseline_weights.pt').read_bytes()).hexdigest()==locked['baseline_sha256']
    for item in locked['models']:assert hashlib.sha256((root/item['head_file']).read_bytes()).hexdigest()==item['head_sha256']
    cp=torch.load(root/'baseline_weights.pt',map_location='cuda',weights_only=True);g=FullG().cuda();f=ReturnF().cuda();g.load_state_dict(cp['g']);f.load_state_dict(cp['f']);g.eval();f.eval();g.requires_grad_(False);f.requires_grad_(False)
    manifest=json.loads((root/'split_manifest.json').read_text());cg_train=load(args.prior/'full_context_data/full_train.npz');lookup={str(n).split('__',1)[1]:i for i,n in enumerate(cg_train['ids'])}
    with zipfile.ZipFile('/autodl-fs/data/data0723_identity_cyclegan/raw_pairs/data0723_paired_samples.zip') as zf:
        parity=0.
        for row in manifest['groups']['gan_train'][:5]:
            xx,_=frontend(row,zf,manifest['shared_normalization']);parity=max(parity,float(abs(xx-cg_train['x'][lookup[row['key']]]).max()))
    assert parity<1e-6,('Frontend changed',parity)
    # Generation check remains outside every authentication group and never selects G.
    check=load(root/'gan_check.npz');gg=[];cc=[]
    with torch.no_grad():
        for i in range(0,len(check['ids']),12):
            x=torch.as_tensor(check['x'][i:i+12],device='cuda');a=g(x);gg.append(a.cpu().numpy());cc.append(f(a).cpu().numpy())
    gen=np.concatenate(gg);rec=np.concatenate(cc);proto=json.loads((args.prior/'results/protocol.json').read_text());alo,ahi=proto['normalization_audio'];generation=metrics(check,dict(generated_audio=gen,cycle_ACC=rec),alo,ahi,proto['audio_energy_threshold'])
    # Fixed first2sources/task, same rule used before in generation plots.
    ii=[]
    for task in ['0-9','word']:ii.extend(np.flatnonzero(check['tasks']==task)[:2].tolist())
    np.savez_compressed(root/'generation_plots.npz',real_audio=check['y'][ii],generated_audio=gen[ii],real_ACC=check['x'][ii],cycle_ACC=rec[ii],ids=check['ids'][ii],tasks=check['tasks'][ii])
    final=load(root/'final_test.npz');assert int((final['labels']==0).sum())==30 and int((final['labels']==1).sum())==118
    role_arrays={'gan_train':cg_train['x'],'gan_check':check['x'],'final_test':final['x']}
    for name in ['head_train','head_validation','enrollment','calibration']:role_arrays[name]=load(root/(name+'.npz'))['x']
    hashsets={k:{hashlib.sha256(x.tobytes()).hexdigest() for x in a} for k,a in role_arrays.items()}
    for a,aa in hashsets.items():
        assert len(aa)==len(role_arrays[a]),('Duplicate modelinput within role',a)
        for b,bb in hashsets.items():
            if a!=b:assert not aa&bb,('Duplicate modelinput across roles',a,b)
    bank=extract(g,f,final);results=[];private={}
    for item in locked['models']:
        route,seed=item['route'],item['seed'];cphead=torch.load(root/item['head_file'],map_location='cuda',weights_only=True);model=TrainBHead(cphead['input_dim']).cuda();model.load_state_dict(cphead['state']);z=embed(model,bank[route]);template=np.array(item['template'],dtype='float32');scores=z@template;t=item['threshold'];genuine=final['labels']==0;impostor=~genuine;accepted=scores>=t
        fn=int((~accepted&genuine).sum());fp=int((accepted&impostor).sum());ng=int(genuine.sum());ni=int(impostor.sum());truth=genuine.astype(int);auc=float(roc_auc_score(truth,scores));fpr,tpr,cuts=roc_curve(truth,scores)
        row=dict(route=route,head_seed=seed,best_epoch=item['best_epoch'],threshold=t,calibration=item['calibration'],genuine=dict(n=ng,accepted=ng-fn,rejected=fn,TAR=(ng-fn)/ng,FRR=fn/ng,TAR_Wilson95=interval(ng-fn,ng)),attacks=dict(chengkeyi=dict(n=ni,accepted=fp,FAR=fp/ni,FAR_Wilson95=interval(fp,ni))),FAR=fp/ni,FRR=fn/ng,HTER=(fp/ni+fn/ng)/2,AUC=auc,by_task={})
        for task in ['0-9','word']:
            keep=final['tasks']==task;gm=keep&genuine;im=keep&impostor
            row['by_task'][task]=dict(genuine_n=int(gm.sum()),TAR=float(accepted[gm].mean()) if gm.any() else None,attack_n=int(im.sum()),FAR=float(accepted[im].mean()) if im.any() else None)
        results.append(row);private[route+'_'+str(seed)]=dict(scores=scores.tolist(),labels=final['labels'].tolist(),ids=final['ids'].tolist(),tasks=final['tasks'].tolist(),ROC=dict(FAR=fpr.tolist(),TAR=tpr.tolist()))
        print('FINAL',json.dumps(row),flush=True);del model;torch.cuda.empty_cache()
    aggregate={}
    for route in ['fusion','raw_ACC','cycle_ACC']:
        rr=[r for r in results if r['route']==route];aggregate[route]={k:float(np.mean([r[k] for r in rr])) for k in ['FAR','FRR','HTER','AUC']};aggregate[route]['TAR']=1-aggregate[route]['FRR']
    m=manifest;public=dict(target='wangjingmei',attack_sources=['chengkeyi'],counts=m['counts'],generation_check=dict(recordings=len(check['ids']),metric_frame_scope='Full2s201frames of30GENcheck sources; NOT former60sources180commoninteriors, so do not directly compare to .3071.',macro=generation['macro'],extra=generation['extra']['macro']),authentication=results,aggregate=aggregate,head_protocol=locked['protocol'],data_assertions=dict(m['assertions'],all_preprocessed_inputs_pairwise_disjoint=True,frontend_parity_max=parity),scope=m['scope'],limits=m['limits'],lock_barrier=locked['barrier'],baseline_sha256=locked['baseline_sha256'])
    (root/'summary.json').write_text(json.dumps(public,indent=2));(root/'private_scores.json').write_text(json.dumps(private,indent=2))
    with zipfile.ZipFile(root/'review.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in ['summary.json','private_scores.json','generation_plots.npz','cyclegan_protocol.json','head_protocol.json','head_histories.json','full_history.json','fresh_local_history.json']:
            z.write(root/name,name)
    print('EVALUATION_DONE',json.dumps(aggregate),flush=True)
if __name__=='__main__':main()
