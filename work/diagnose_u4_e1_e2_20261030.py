"""Existing-feature E1/E2 diagnostics for same-content U4 authentication."""
import sys,json
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
sys.path.insert(0,'/root')
import run_nine_audio_supplement_20261030 as a
import run_nine_sameword_balanced_20261030 as b
old=b.old;OUT=Path('/autodl-fs/data/u4_e1_e2_sameword_20261030')

def corr_lags(x,y,L=5):
 x=np.asarray(x,float);y=np.asarray(y,float);rows=[]
 for lag in range(-L,L+1):
  xx=x[max(0,lag):min(len(x),len(x)+lag)]; yy=y[max(0,-lag):min(len(y),len(y)-lag)]
  xx=xx-xx.mean();yy=yy-yy.mean();rows.append(float(xx@yy/(np.linalg.norm(xx)*np.linalg.norm(yy)+1e-12)))
 return rows[L],max(rows),int(np.argmax(rows)-L)

def envelope(x):
 x=np.asarray(x,float);return np.sqrt(np.mean((x-x.mean(axis=1,keepdims=True))**2,axis=0))

def main():
 OUT.mkdir(exist_ok=True);M=b.same.manifest();data=a.dataset(4,M);train=data['auth_train'];test=data['test'];base= b.dataset(4,M)
 acc=[];aud=[];indices=[]
 for u in range(1,10):
  x=np.load(str(old.DATA/('acc.%d.npy'%u))).astype('float32')[:,:80,:80];m=np.load(str(old.DATA/('audio.%d.npy'%u))).astype('float32')[:,:80,:80]
  roles=M['sources'][str(u)]['roles']
  for role in ['auth_train','test']:
   for j in roles[role]:indices.append((role,u-1,int(M['sources'][str(u)]['groups'][j]),j));acc.append(x[j]);aud.append(m[j])
 acc=np.asarray(acc);aud=np.asarray(aud);ntrain=len(train['labels']);ntest=len(test['labels']);assert ntrain==675 and ntest==270 and len(acc)==945
 # data helper order is users x indices, train then test; reproduce metadata order.
 meta_train=[];meta_test=[]
 for role in ['auth_train','test']:
  dest=meta_train if role=='auth_train' else meta_test
  for u in range(1,10):
   src=M['sources'][str(u)]
   for j in src['roles'][role]:dest.append((u-1,int(src['groups'][j]),int(src['repeats'][j]),j))
 assert np.array_equal(train['labels'],np.asarray([v[0] for v in meta_train])) and np.array_equal(test['labels'],np.asarray([v[0] for v in meta_test]))
 # E1: transformed-spectrum dynamic envelopes; descriptive correlations only.
 at=np.asarray([envelope(x) for x in acc[ntrain:]]);mt=np.asarray([envelope(x) for x in aud[ntrain:]])
 idxby_uidword={(u,w):[i for i,(uu,w2,rep,j) in enumerate(meta_test) if uu==u and w2==w] for u in range(9) for w in set(z[1] for z in meta_test)}
 paired=[];sameuid_crossword=[];otheruid_sameword=[]
 for i,(u,w,rep,j) in enumerate(meta_test):
  paired.append(corr_lags(at[i],mt[i]))
  cand=[k for k,(uu,ww,rp,jj) in enumerate(meta_test) if uu==u and ww!=w]
  k=cand[(u+i)%len(cand)];sameuid_crossword.append(corr_lags(at[i],mt[k]))
  cand=[k for k,(uu,ww,rp,jj) in enumerate(meta_test) if uu!=u and ww==w]
  k=cand[(u+i)%len(cand)];otheruid_sameword.append(corr_lags(at[i],mt[k]))
 def corr_summary(v):
  x=np.asarray(v);return dict(n=len(x),zero_lag_mean=float(x[:,0].mean()),zero_lag_median=float(np.median(x[:,0])),best_pm5_mean=float(x[:,1].mean()),best_pm5_median_frames=int(np.median(x[:,2])),best_lag_frames={str(k):int((x[:,2]==k).sum()) for k in range(-5,6)})
 e1={k:corr_summary(v) for k,v in [('paired_same_recording',paired),('same_identity_different_word',sameuid_crossword),('different_identity_same_word',otheruid_sameword)]}
 # E2: feature-only and two-block cosine distance distributions. Scalers fit on auth_train only.
 methods=['raw','old_audio','identity_acc','base'];scaled={}
 for name in methods:
  sc=StandardScaler().fit(train['features'][name]);q=sc.transform(test['features'][name]);scaled[name]=q/(np.linalg.norm(q,axis=1,keepdims=True)+1e-12)
 e2={}
 for names in [('raw',),('old_audio',),('identity_acc',),('base',),('raw','old_audio'),('base','raw')]:
  z=np.concatenate([scaled[k]/np.sqrt(len(names)) for k in names],axis=1) if len(names)>1 else scaled[names[0]];sim=z@z.T;groups={k:[] for k in ['same_identity_same_word','same_identity_cross_word','different_identity_same_word','different_identity_cross_word']}
  for i in range(ntest):
   for j in range(i+1,ntest):
    u,w=test['labels'][i],test['words'][i];v,x=test['labels'][j],test['words'][j];key=('same_identity_' if u==v else 'different_identity_')+('same_word' if w==x else 'cross_word');groups[key].append(float(sim[i,j]))
  auc=float(roc_auc_score(np.r_[np.ones(len(groups['same_identity_cross_word'])),np.zeros(len(groups['different_identity_same_word']))],np.r_[groups['same_identity_cross_word'],groups['different_identity_same_word']]))
  def desc(x):return dict(n=len(x),mean_similarity=float(np.mean(x)),median_similarity=float(np.median(x)),q10=float(np.quantile(x,.1)),q90=float(np.quantile(x,.9)))
  key='+'.join(names);groups={k:desc(v) for k,v in groups.items()};groups['cross_word_genuine_vs_same_word_impostor_auc']=auc
  e2[key]=groups
 # clipping audit and provenance limitations.
 audit={}
 for name,x in [('ACC',acc),('Audio',aud)]:audit[name]=dict(n=int(x.size),fraction_at_or_beyond_0p99=float(np.mean(np.abs(x)>=.99)),fraction_exact_edges=float(np.mean(np.abs(x)==1.)))
 result=dict(protocol='same 15 words; U4 target; heldout two words; all8 attackers retained',e1=e1,e2=e2,stored_spectrum_clip_audit=audit,physical_sync_waveforms_available=False,wear_sessions_available=False,limits=['ACC/Audio are stored normalized log-spectrum arrays; cannot recover physical waveform coherence, cross-spectrum phase, absolute band-energy transfer, or impulse response','envelope correlations are across 80 transformed time bins, have no physical frequency calibration and are descriptive','same user different word is a content-identity proxy; no independent rewear/session split','pairwise distances are descriptive and dependent; this is not a calibrated authentication result'])
 old.save(OUT/'diagnostics_public.json',result);print(json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()


