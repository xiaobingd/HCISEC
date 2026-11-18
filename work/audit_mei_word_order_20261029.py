"""Audio-only word/round layout diagnostics, not authentication optimization."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json,hashlib
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
ROOT=Path('/autodl-fs/data/mei_word_order_audit_20261029');SRC=Path('/root/autodl-tmp/train_core/data')
def norm(x):return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
def labels(n,layout):return np.arange(n)//10 if layout=='word_major' else np.arange(n)%30
def repeat(n,layout):return np.arange(n)%10 if layout=='word_major' else np.arange(n)//30
def center(x):return x-x.mean(0,keepdims=True)
def main():
 ROOT.mkdir(exist_ok=True);arrays={};features={};flat={};sources={}
 for u in range(1,10):
  p=SRC/('audio.%d.npy'%u);raw=p.read_bytes();a=np.load(str(p)).astype('float64')[:,:80,:80];assert np.isfinite(a).all()
  arrays[u]=a;features[u]=np.concatenate([a.mean(2),a.std(2)],1);flat[u]=a[:,:,::4].reshape(len(a),-1)
  sources[str(u)]=dict(rows=len(a),sha256=hashlib.sha256(raw).hexdigest())
 full=[u for u in arrays if len(arrays[u])==300];result=dict(source=sources,complete_users=full,chance=1/30,method='Audio-only diagnostics. No identity/authentication outcomes used; mean-removal for diagnostics only, does not alter source data or authentication frontend.',layouts={},lags={})
 for layout in ('word_major','round_major'):
  blocks={};pairs=[];within={};transfer=[]
  for u in full:
   y=labels(300,layout);z=np.stack([arrays[u][y==w].mean((0,2)) for w in range(30)]);blocks[u]=norm(center(z))
  for u in full:
   for v in full:
    if u>=v:continue
    sim=blocks[u]@blocks[v].T;ranks=np.argsort(np.argsort(-sim,axis=1),axis=1)
    pairs.append(dict(users=[u,v],top1=float(np.mean(np.argmax(sim,axis=1)==np.arange(30))),median_rank=float(np.median(np.diag(ranks)+1)),diagonal_cosine=float(np.diag(sim).mean())))
  for u,a in arrays.items():
   # Compare balanced first270 for both layouts; word-major27 groups and round-major30.
   n=270;y=labels(n,layout);r=repeat(n,layout);rows=[]
   for name,z in [('mean_std160',features[u][:n]),('spectro_time_stride4',flat[u][:n])]:
    hits=[]
    for rr in np.unique(r):
     tr=r!=rr;va=~tr;sc=StandardScaler().fit(z[tr]);zt=sc.transform(z[tr]);zv=sc.transform(z[va]);classes=np.unique(y)
     proto=norm(np.stack([zt[y[tr]==w].mean(0) for w in classes]));pred=classes[np.argmax(norm(zv)@proto.T,axis=1)];hits.extend((pred==y[va]).tolist())
    rows.append(dict(feature=name,top1=float(np.mean(hits)),chance=1/len(np.unique(y)),rows=n,groups=len(np.unique(y))))
   within[str(u)]=rows
  # Balanced independent speaker transfer: complete users train, one target excluded.
  for target in range(1,10):
   train_users=[u for u in full if u!=target];xt=np.concatenate([center(features[u]) for u in train_users]);yt=np.concatenate([labels(len(features[u]),layout) for u in train_users]);xv=center(features[target]);yv=labels(len(xv),layout)
   sc=StandardScaler().fit(xt);pc=PCA(n_components=48,svd_solver='randomized',random_state=42).fit(sc.transform(xt));clf=LogisticRegression(C=1,max_iter=1500).fit(pc.transform(sc.transform(xt)),yt)
   pred=clf.predict(pc.transform(sc.transform(xv)));blockscores=[]
   for b in range(0,len(xv),30):blockscores.append(dict(start_row=b,n=min(30,len(xv)-b),top1=float(np.mean(pred[b:b+30]==yv[b:b+30]))))
   transfer.append(dict(user=target,rows=len(xv),train_users=train_users,top1=float(accuracy_score(yv,pred)),first270_top1=float(accuracy_score(yv[:270],pred[:270])),by_30row_segment=blockscores))
  result['layouts'][layout]=dict(cross_user_block_retrieval=pairs,mean_block_top1=float(np.mean([p['top1'] for p in pairs])),within_user_repeat_holdout=within,audio_prompt_transfer=transfer)
  print('LAYOUT_DONE',layout,result['layouts'][layout]['mean_block_top1'],[(r['user'],r['top1']) for r in transfer],flush=True)
 for u in arrays:
  z=norm(center(features[u]));s=z@z.T;vals=[float(np.diag(s,k).mean()) for k in range(1,61)];result['lags'][str(u)]=dict(lags=list(range(1,61)),cosine=vals,lag1=vals[0],lag10=vals[9],lag30=vals[29],lag60=vals[59],largest_lag=int(np.argmax(vals)+1))
 (ROOT/'results_public.json').write_text(json.dumps(result,indent=2))
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,axs=plt.subplots(1,2,figsize=(12,4))
 for u in arrays:axs[0].plot(range(1,61),result['lags'][str(u)]['cosine'],label='U%d'%u,alpha=.8)
 axs[0].axvline(30,color='black',linestyle='--');axs[0].set(xlabel='Separation in array rows',ylabel='Mean centered Audio feature cosine',title='Audio sequence periodicity');axs[0].legend(ncol=3,fontsize=8)
 x=np.arange(9);w=.36
 for j,layout in enumerate(('word_major','round_major')):
  vals=[r['top1'] for r in result['layouts'][layout]['audio_prompt_transfer']];axs[1].bar(x+(j-.5)*w,vals,w,label=layout)
 axs[1].axhline(1/30,color='black',linestyle=':');axs[1].set(xticks=x,xticklabels=['U%d'%u for u in range(1,10)],ylabel='Prompt-index classification accuracy',title='Held-speaker Audio transfer (diagnostic)');axs[1].legend()
 fig.tight_layout();fig.savefig(str(ROOT/'word_order_diagnostic.png'),dpi=150);fig.savefig(str(ROOT/'word_order_diagnostic.svg'));plt.close(fig)
 print('ORDER_AUDIT_COMPLETE',flush=True)
if __name__=='__main__':main()
