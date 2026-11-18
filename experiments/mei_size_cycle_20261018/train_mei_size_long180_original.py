"""Optimization-budget control for the larger clean mei set, not another model."""
import json,argparse
from pathlib import Path
import numpy as np
import torch
from train_mei_size_cycle import fit,evaluate,load

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--seeds',nargs='+',type=int,default=[42,43]);p.add_argument('--warm-epochs',type=int,default=360);p.add_argument('--cycle-epochs',type=int,default=360);p.add_argument('--updates',type=int,default=4);args=p.parse_args()
    root=args.root;out=root/'results_long';out.mkdir(exist_ok=True);torch.set_num_threads(2)
    protocol=json.loads((root/'results/protocol.json').read_text())
    protocol.update(routes=['mei180_long'],seeds=args.seeds,budget_control='Additional larger-set optimization control after first short-budget seed showed higher training error. SAME model/frontend/loss/frozen calibration and clean180TRAIN, 3xprewarm and3xfull-cycle updates, new same-seed initialization. Fixed360/360 endpoints, no externalVAL checkpoint selection. Exploratory follow-up, not predeclared with the original4routes.')
    protocol['training'].update(prewarm_epochs=args.warm_epochs,full_cycle_epochs=args.cycle_epochs)
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2));results=[]
    data=load(root/'data/mei180_train.npz');reference=load(root/'data/mei20_train.npz')
    for seed in args.seeds:
        name='mei180_long_'+str(seed)
        g,f,meta=fit(dict(small=False,feature_cycle=False),seed,data,out,name,args,cycle_epochs=args.cycle_epochs)
        va=load(root/'data/mei_val.npz');assert not set(data['ids'])&set(va['ids'])
        templates=(data['x'].mean(0),data['y'].mean(0));vm,preds=evaluate(g,f,va,templates);tm,tp=evaluate(g,f,data,templates);rm,_=evaluate(g,f,reference,templates)
        np.savez_compressed(out/(name+'_predictions.npz'),**preds)
        result=dict(route='mei180_long',seed=seed,fit=meta,n_train_recordings=len(np.unique(data['ids'])),n_train_windows=len(data['ids']),train=tm,common_small_train=rm,validation=vm)
        results.append(result);(out/'results.json').write_text(json.dumps(results,indent=2));print('RESULT',json.dumps(result),flush=True)
        del g,f;torch.cuda.empty_cache()
if __name__=='__main__':main()
