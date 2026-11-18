"""Expose best NONZERO MKD and matched shuffle so fallback never disguises CE as MKD."""
import sys,json,time
import numpy as np
import torch
sys.path.insert(0,'/root')
import run_relaxed_audio_feature_kd_20261030 as run
prev=run.prev;old=run.old

def load(path):
 ck=torch.load(str(path),map_location=run.D,weights_only=False);net=prev.Classifier().to(run.D);net.load_state_dict(ck['state']);net.eval()
 for p in net.parameters():p.requires_grad_(False)
 return net,ck

def main():
 data=prev.data_load();allrows={}
 for s in prev.SEEDS:
  folder=run.ROOT/('seed%d'%s);deadline=time.monotonic()+1800
  while True:
   try:lock=json.loads((folder/'locked_before_final.json').read_text());assert (folder/'results_public.json').exists();break
   except (FileNotFoundError,json.JSONDecodeError,AssertionError):assert time.monotonic()<deadline;time.sleep(5)
  selected={k:min(rows,key=lambda m:prev.rank(m['quality'])+(m['lambda_mkd'],m['margin'])) for k,rows in lock['selection'].items()};chosen={k:dict(key=m['key'],lambda_mkd=m['lambda_mkd'],margin=m['margin']) for k,m in selected.items()};old.save(folder/'nonzero_choices_before_final.json',dict(chosen=chosen,rule='best strictly positive lambda by existing validation only;not final',reason='E0 fallback must not be reported as actual MKD',final_scores_seen=False))
  models={};sources={}
  for family,meta in selected.items():models[family+'_nonzero'],_=load(folder/(meta['key']+'_private.pt'));sources[family+'_nonzero']=dict(directory='current',key=meta['key'])
  paired=selected['E2_AudioMKD'];main_paired=lock['metadata']['E2_AudioMKD']
  if main_paired is not None and main_paired['key']==paired['key']:
   models['E2_nonzero_Shuffled'],_=load(folder/'E2_Shuffled_private.pt');sources['E2_nonzero_Shuffled']=dict(directory='current',key='E2_Shuffled')
  else:
   teacher,_=run.load_previous(prev.ROOT/('seed%d'%s),'T1_Audio');z,_=prev.inference(teacher,data,'train','audio');mapping=np.load(str(folder/'shuffle_map_private.npy'));net,projection,meta=run.train(data,s,folder,'E2_nonzero_Shuffled',prev.tensor(z),paired['lambda_mkd'],paired['margin'],mapping);models['E2_nonzero_Shuffled']=net;sources['E2_nonzero_Shuffled']=dict(directory='current',key='E2_nonzero_Shuffled')
  locks={}
  for key,net in models.items():
   z={role:prev.inference(net,data,role)[0] for role in ['enroll','validation','calibration']};locks[key]=prev.backend_lock(z,data,folder,key)
  old.save(folder/'nonzero_locked_before_final.json',dict(sources=sources,chosen=chosen,routes=locks,final_scores_seen=False))
  rows={k:prev.evaluate_final(net,data,folder,k,locks[k]) for k,net in models.items()};out=dict(seed=s,routes=rows,chosen=chosen,sources=sources);old.save(folder/'nonzero_results_public.json',out);allrows[str(s)]=out;old.save(run.ROOT/'nonzero_results_public.json',allrows);print('NONZERO_COMPLETE',s,flush=True)
 old.save(run.ROOT/'nonzero_COMPLETE.json',dict(status='complete',seeds=prev.SEEDS))
if __name__=='__main__':main()
