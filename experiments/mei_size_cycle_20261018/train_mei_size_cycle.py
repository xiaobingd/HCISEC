"""Fixed-update within-mei sample-size comparison; all candidates are full cycles."""
import os,sys,json,argparse
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_THREADING_LAYER','SEQUENTIAL')
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'cycle_generalization_20261018'))
from train_cycle_generalization import fit,evaluate,load

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--seeds',nargs='+',type=int,default=[42,43]);p.add_argument('--warm-epochs',type=int,default=120);p.add_argument('--cycle-epochs',type=int,default=120);p.add_argument('--updates',type=int,default=4);p.add_argument('--routes',nargs='+',default=['personal20','mei20','mei70','mei180']);p.add_argument('--output-name',default='results');args=p.parse_args()
    root=args.root;out=root/args.output_name;out.mkdir(exist_ok=True);torch.set_num_threads(2)
    dp=json.loads((root/'data_protocol.json').read_text())
    protocol=dict(data=dp,seeds=args.seeds,routes=args.routes,model='SAME G original project2DResNet hidden16 blocks2 resize/noIN/projection251->80, F temporal64 fiveblocks; train_cycle_generalization.py fixed_full configuration',training=dict(prewarm_epochs=args.warm_epochs,full_cycle_epochs=args.cycle_epochs,updates_per_epoch=args.updates,batch_size=8,optimizer='AdamW lr.0003 beta.5/.999 WD.001; Dlr.0002',selection='fixed endpoints, no validation checkpoint/model selection, all runs from scratch. Same number gradient updates, not same full passes through each dataset.'),losses='Identical previous full-cycle losses:20pairedaudio5pairedACC/audioGAN1/ACCGAN.05, spectralgradient/envelope/profile/FM; fullcycleACC ramp10->20 audio10 TRAIN-onlyprewarmteacher5;50%MelS/.005noise; no bypass.',scope='Within-mei20/70/180 changes training-source record count only;20mei is nested and has same sharedcalibration as larger sets. Personal20 is a secondary different-source control, NOT a causal quantity-only comparison. Noidentityhead/FAR/FRR claims.')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2));results=[]
    for seed in args.seeds:
        for route in args.routes:
            data=load(root/'data'/(route+'_train.npz'));domain='personal' if route=='personal20' else 'mei'
            name=route+'_'+str(seed)
            g,f,meta=fit(dict(small=False,feature_cycle=False),seed,data,out,name,args,cycle_epochs=args.cycle_epochs)
            va=load(root/'data'/(domain+'_val.npz'));assert not set(data['ids'])&set(va['ids'])
            templates=(data['x'].mean(0),data['y'].mean(0))
            vm,preds=evaluate(g,f,va,templates);tm,tp=evaluate(g,f,data,templates)
            # Also evaluate common nested20 TRAIN windows for comparable train/val gaps across mei sizes.
            reference=data if domain=='personal' else load(root/'data'/'mei20_train.npz')
            rm,_=evaluate(g,f,reference,templates)
            np.savez_compressed(out/(name+'_predictions.npz'),**preds)
            ug=np.unique(data['ids']);ix=[np.flatnonzero(data['ids']==n)[0] for n in ug[:4]]
            np.savez_compressed(out/(name+'_train_examples.npz'),real_audio=data['y'][ix],generated_audio=tp['generated_audio'][ix],ids=data['ids'][ix])
            result=dict(route=route,seed=seed,fit=meta,n_train_recordings=len(ug),n_train_windows=len(data['ids']),train=tm,common_small_train=rm,validation=vm)
            results.append(result);(out/'results.json').write_text(json.dumps(results,indent=2));print('RESULT',json.dumps(result),flush=True)
            del g,f;torch.cuda.empty_cache()
if __name__=='__main__':main()
