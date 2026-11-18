"""Follow-up hypothesis from Audio lag peaks: 100-row period10 + period20 tail.

Data-driven exploratory order diagnosis, not a verified transcript manifest.
"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
ROOT=Path('/autodl-fs/data/mei_word_order_audit_20261029');SRC=Path('/root/autodl-tmp/train_core/data')
def norm(x):return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
def center(x):return x-x.mean(0,keepdims=True)
def ids(n):
 i=np.arange(n);return np.where(i<100,i%10,10+(i-100)%20),np.where(i<100,i//10,(i-100)//20)
def main():
 aa={u:np.load(str(SRC/('audio.%d.npy'%u))).astype('float64')[:,:80,:80] for u in range(1,10)}
 ff={u:np.concatenate([a.mean(2),a.std(2)],1) for u,a in aa.items()};full=[u for u in aa if len(aa[u])==300]
 result=dict(hypothesis='first100 rows repeated10-item rounds, remainder repeated20-item rounds; group=index%10 below100 else10+(index-100)%20',status='Exploratory Audio-based evidence, not transcript identification. Hypothesis selected after period diagnostics, no authentication scores used.',within={},transfer=[],pair_retrieval=[],role_overlap_old_wordmajor={})
 for u,z in ff.items():
  y,r=ids(len(z));hits=[]
  for rr in range(10):
   tr=r!=rr;va=~tr;sc=StandardScaler().fit(z[tr]);zt=sc.transform(z[tr]);zv=sc.transform(z[va]);proto=norm(np.stack([zt[y[tr]==w].mean(0) for w in range(30)]));pred=np.argmax(norm(zv)@proto.T,axis=1);hits.extend((pred==y[va]).tolist())
  result['within'][str(u)]=dict(top1=float(np.mean(hits)),chance=1/30,rows=len(z))
  roles={'upstream_fit':np.arange(120),'upstream_select':np.arange(120,150),'registration':np.arange(150,210),'calibration':np.arange(210,240),'test':np.arange(240,min(len(z),297))}
  result['role_overlap_old_wordmajor'][str(u)]={k:np.unique(y[v]).tolist() for k,v in roles.items()}
 for u in range(1,10):
  train=[v for v in full if v!=u];xt=np.concatenate([center(ff[v]) for v in train]);yt=np.concatenate([ids(len(ff[v]))[0] for v in train]);xv=center(ff[u]);yv,_=ids(len(xv))
  sc=StandardScaler().fit(xt);pc=PCA(n_components=48,svd_solver='randomized',random_state=42).fit(sc.transform(xt));m=LogisticRegression(C=1,max_iter=1500).fit(pc.transform(sc.transform(xt)),yt);pred=m.predict(pc.transform(sc.transform(xv)))
  result['transfer'].append(dict(user=u,rows=len(xv),top1=float(np.mean(pred==yv)),first100_top1=float(np.mean(pred[:100]==yv[:100])),after100_top1=float(np.mean(pred[100:]==yv[100:])),train_users=train))
 proto={u:norm(center(np.stack([aa[u][ids(len(aa[u]))[0]==w].mean((0,2)) for w in range(30)]))) for u in full}
 for u in full:
  for v in full:
   if u<v:
    s=proto[u]@proto[v].T;result['pair_retrieval'].append(dict(users=[u,v],top1=float(np.mean(np.argmax(s,1)==np.arange(30)))))
 result['mean_pair_retrieval']=float(np.mean([x['top1'] for x in result['pair_retrieval']]))
 # Local period evidence from disjoint regions: preserve all users, no identity-based selection.
 result['period_by_segment']={}
 for u,z in ff.items():
  parts={}
  for start,end in [(0,100),(100,len(z))]:
   zz=norm(center(z[start:end]));sim=zz@zz.T;vals=[float(np.diag(sim,k).mean()) for k in range(1,61)];parts[str((start,end))]=dict(lags=list(range(1,61)),mean_cosine=vals)
  result['period_by_segment'][str(u)]=parts
 (ROOT/'mixed_results_public.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='period_by_segment'}),flush=True)
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 old=json.loads((ROOT/'results_public.json').read_text());fig,axes=plt.subplots(1,3,figsize=(15,4))
 for u,parts in result['period_by_segment'].items():
  for j,part in enumerate(parts.values()):axes[j].plot(part['lags'],part['mean_cosine'],label='U'+u)
 for j in (0,1):axes[j].set(xlabel='Row lag',ylabel='Centered Audio cosine',title='Rows 0-99' if j==0 else 'Rows 100-end');axes[j].axvline(10 if j==0 else 20,color='black',linestyle='--')
 x=np.arange(9);width=.25
 for j,layout in enumerate(('word_major','round_major','mixed')):
  values=[r['top1'] for r in result['transfer']] if layout=='mixed' else [r['top1'] for r in old['layouts'][layout]['audio_prompt_transfer']]
  axes[2].bar(x+(j-1)*width,values,width,label=layout)
 axes[2].set(xticks=x,xticklabels=['U%d'%u for u in range(1,10)],ylabel='Prompt-index accuracy',title='Held-speaker Audio transfer');axes[2].axhline(1/30,color='black',linestyle=':');axes[2].legend();axes[0].legend(ncol=3,fontsize=7)
 fig.tight_layout();fig.savefig(str(ROOT/'mixed_order_diagnostic.png'),dpi=150);fig.savefig(str(ROOT/'mixed_order_diagnostic.svg'));print('MIXED_ORDER_AUDIT_COMPLETE')
if __name__=='__main__':main()
