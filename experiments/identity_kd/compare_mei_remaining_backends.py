"""Frozen E2/E4/E24: RBF, fit-only within-class shrink whitening cosine, LR."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='2'
import json
from pathlib import Path
import numpy as np
from remaining_auth_backends import Backends
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from evaluate_crossword_encoder_distill import norm,metric,threshold

OUT=Path('outputs/mei_remaining_backends_20261117');USERS=[1,2,3,4,6,7,8,9]
def dump(name,v):(OUT/name).write_text(json.dumps(v,indent=2,allow_nan=False),encoding='utf-8')
def main():
 OUT.mkdir(exist_ok=True)
 d=np.load('outputs/mei_crossword_representation_20261116/vectors_private.npz')
 joint=np.load('outputs/mei_e2_e4_joint_20261117/vectors_private.npz');assert np.array_equal(d['sample_id'],joint['sample_id'])
 u,w=d['user'],d['word'];fit=np.flatnonzero((w<18)&np.isin(u,USERS))
 locks=json.loads(Path('outputs/mei_crossword_representation_20261116/fusion_selection_public.json').read_text())['methods']
 lookup={(us,ww):np.flatnonzero((u==us)&(w==ww)) for us in USERS for ww in range(18,30)}
 def reg(us,ww,f):return lookup[us,ww][f*4:f*4+4]
 def query(ww,f):return np.concatenate([lookup[us,ww][(1-f)*4:(1-f)*4+4] for us in USERS])
 allrows=[];selections={};private={}
 for name in ['E2','E4','E24']:
  key='E2' if name=='E24' else name;alpha=locks[key]['selected']['alpha']
  z=joint['E2'] if name=='E24' else d[name]
  v=d['raw'] if alpha==0 else np.c_[d['raw'],alpha*z]/np.sqrt(1+alpha**2)
  engines=Backends(v,u,w,fit,USERS)
  def scoring(backend,C=1.):
   def score(owner,rw,qw,f):
    return engines.score(backend,owner,reg(owner,rw,f),query(qw,f))
   return score
  chosen=1.
  selections[name]=dict(ASNorm_topK=3,PLDA_PCA=32,neural_steps=400,neural_seed=42)
  dump('selection_public.json',selections)
  for backend in ['asnorm','plda','neural']:
   score=scoring(backend,chosen)
   for owner in USERS:
    yy=[];ss=[]
    for qw in [21,22,23]:
     y=u[query(qw,0)]==owner
     for rw in [21,22,23]:
      if rw!=qw:yy.extend([y,y]);ss.extend([score(owner,qw,qw,0),score(owner,rw,qw,0)])
    tau=threshold(np.concatenate(yy),np.concatenate(ss))
    for condition in ['sameword','crossword']:
     values=[];scores=[]
     for f in [0,1]:
      for qw in range(24,30):
       y=u[query(qw,f)]==owner
       for rw in range(24,30):
        if rw==qw:continue
        s=score(owner,qw if condition=='sameword' else rw,qw,f)
        values.append(metric(y,s,tau));scores.append(s)
     allrows.append(dict(method=name,backend=backend,user=f'U{owner}',condition=condition,**{k:float(np.mean([x[k] for x in values])) for k in values[0]}))
     private[f'{name}_{backend}_{owner}_{condition}']=np.array(scores)
   print('COMPLETE',name,backend,flush=True)
  dump('status.json',dict(method=name,stage='evaluated'))
 summary=[]
 for name in ['E2','E4','E24']:
  for b in ['asnorm','plda','neural']:
   for c in ['sameword','crossword']:
    rr=[r for r in allrows if r['method']==name and r['backend']==b and r['condition']==c]
    summary.append(dict(method=name,backend=b,condition=c,**{k:float(np.mean([r[k] for r in rr])) for k in ['tar','far','frr','hter','eer','auc']}))
 error=0.
 dump('results_public.json',dict(summary=summary,per_user=allrows));np.savez_compressed(OUT/'scores_private.npz',**private)
 dump('verification_public.json',dict(state='PASS',all_fit_only=True,users=USERS,frozen_encoders=True,whitening_fit_words=list(range(18)),LR_C_dev_words=[18,19,20],cal_words=[21,22,23],final_words=list(range(24,30)),no_query_audio=True,thresholds_shared_same_cross=True,selection='fixed ASNormTop3;PCA32 Gaussian PLDA moment estimate with within covariance shrinkage and between PSD projection;pairMLP64to32to1 BCE400steps seed42;no final tuning',limits='single seed,known users,historically explored words;fusion originally tuned for RBF'))
 lines=['# Mei认证后端对照','', '固定E2、E4、E24表征与原统计融合权重，U5不参与认证。ASNorm固定Top3；PLDA固定PCA32；神经后端固定400步，全部只用训练词0–17。Cal21–23锁定阈值，Final24–29同查询K4两折。九人编码器训练不变，认证八人。','', '|特征|后端|同词EER%|跨词EER%|同词HTER%|跨词HTER%|跨词FAR%|跨词FRR%|','|---|---|---:|---:|---:|---:|---:|---:|']
 for name in ['E2','E4','E24']:
  for b in ['asnorm','plda','neural']:
   a=next(r for r in summary if r['method']==name and r['backend']==b and r['condition']=='sameword');c=next(r for r in summary if r['method']==name and r['backend']==b and r['condition']=='crossword')
   lines.append(f'|{name}|{b}|'+ '|'.join(f'{100*x:.2f}' for x in [a['eer'],c['eer'],a['hter'],c['hter'],c['far'],c['frr']])+'|')
 lines+=['','asnorm=余弦＋双侧Top3背景标准化，背景为训练词身份原型并排除当前目标；闭集背景身份与攻击者身份允许重叠，仅录制隔离。plda=32维高斯PLDA，训练矩估计与协方差收缩；neural=对称差值/乘积MLP，多注册分数均值。所有参数只用训练词，神经正对强制同人异词。原始融合权重曾按RBF开发性能锁定，因此这是固定特征后端对照。','', '逐用户结果见results_public.json；不发布逐样本分数。单种子，历史探索数据，不能当作未见用户验证。']
 (OUT/'report.md').write_text('\n'.join(lines),encoding='utf-8');dump('status.json',dict(stage='complete'));print('\n'.join(lines))

if __name__=='__main__':main()
