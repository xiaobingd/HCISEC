import argparse,json,os
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from modelb import create_multi_source_model
from target_auth_protocol_150 import build_target_protocol

def extract(model,x,device,batch=64):
 out={'shallow':[],'bottleneck':[],'cycle_diff':[]}
 model.eval()
 for i in range(0,len(x),batch):
  t=torch.from_numpy(x[i:i+batch]).float().unsqueeze(1).to(device)
  with torch.no_grad():
   s=model.shallow_agg(model.shallow_encoder(t))
   b=model.bottleneck_agg(model.bottleneck_encoder(t))
   dm,ds=model._compute_cycle_diff(t)
   c=torch.cat([model.diff_agg(model.diff_encoder(dm)),ds],1)
  for k,v in [('shallow',s),('bottleneck',b),('cycle_diff',c)]:
   out[k].append(v.cpu().numpy())
 out={k:np.concatenate(v) for k,v in out.items()}
 out['combined']=np.concatenate([out['shallow'],out['bottleneck'],out['cycle_diff']],1)
 return out

def threshold(dev_pos,dev_neg):
 candidates=np.unique(np.concatenate([dev_pos,dev_neg]))
 best=None
 for t in candidates:
  frr=np.mean(dev_pos<t);far=np.mean(dev_neg>=t)
  q=(0.5*(far+frr),abs(far-frr),float(t),far,frr)
  if best is None or q[:2]<best[:2]:best=q
 return best

def eer(y,s):
 fpr,tpr,_=roc_curve(y,s);fnr=1-tpr;j=np.argmin(np.abs(fpr-fnr))
 return float((fpr[j]+fnr[j])/2)

def main():
 p=argparse.ArgumentParser();p.add_argument('--target',type=int,required=True)
 p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
 a=p.parse_args();device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 protocol=build_target_protocol(a.target)
 model,_=create_multi_source_model(a.checkpoint,device=device)
 groups=['auth_train','model_validation','target_dev_pool','impostor_calibration','final_genuine','final_impostor']
 feats={g:extract(model,protocol.arrays[g],device) for g in groups}
 labels=protocol.labels;target=a.target-1;results={}
 for source in ['shallow','bottleneck','cycle_diff','combined']:
  best=None
  for C in [0.01,0.1,1.0,10.0]:
   clf=make_pipeline(StandardScaler(),LogisticRegression(C=C,max_iter=3000,class_weight='balanced'))
   clf.fit(feats['auth_train'][source],labels['auth_train'])
   val=clf.score(feats['model_validation'][source],labels['model_validation'])
   if best is None or val>best[0]:best=(val,C,clf)
  val_acc,C,clf=best
  idx=list(clf.classes_).index(target)
  score=lambda g:clf.predict_proba(feats[g][source])[:,idx]
  dp,dn=score('target_dev_pool'),score('impostor_calibration')
  hter_dev,_,th,far_dev,frr_dev=threshold(dp,dn)
  gp,gn=score('final_genuine'),score('final_impostor')
  y=np.r_[np.ones(len(gp)),np.zeros(len(gn))];ss=np.r_[gp,gn]
  results[source]={
   'dim':int(feats['auth_train'][source].shape[1]),'C':C,'val_multiclass_acc':float(val_acc),
   'dev_threshold':th,'dev_far':float(far_dev),'dev_frr':float(frr_dev),'dev_hter':float(hter_dev),
   'final_auc':float(roc_auc_score(y,ss)),'final_eer':eer(y,ss),
   'final_far':float(np.mean(gn>=th)),'final_frr':float(np.mean(gp<th)),
  }
  results[source]['final_hter']=0.5*(results[source]['final_far']+results[source]['final_frr'])
 payload={'target_user':a.target,'target_name':protocol.manifest['target_name'],'checkpoint':a.checkpoint,'results':results}
 os.makedirs(os.path.dirname(a.output),exist_ok=True)
 with open(a.output,'w') as f:json.dump(payload,f,indent=2)
 print(json.dumps(payload,indent=2))
if __name__=='__main__':main()

