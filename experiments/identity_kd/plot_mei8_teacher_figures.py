import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
OUT=Path('outputs/mei8_teacher_figures_20261108');OUT.mkdir(exist_ok=True)
a=np.load(OUT/'heldout_embeddings_private.npz');sel=[]
for u in range(1,9):
 for w in range(24,30):sel.extend(np.flatnonzero((a['user']==u)&(a['word']==w))[:8])
u=a['user'][sel];w=a['word'][sel];names=['bo','bro','jin','ke','mei','qi','qis','xin']
z={k:a[k][sel] for k in ['audio','acc']}
z={k:v/np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1e-12) for k,v in z.items()}
upper=np.triu(np.ones((len(u),len(u)),bool),1);pos=upper&(u[:,None]==u)&(w[:,None]!=w);neg=upper&(u[:,None]!=u)&(w[:,None]==w)
s=z['audio']@z['audio'].T;p=s[pos];n=s[neg]
fig,ax=plt.subplots(figsize=(8,4.8),layout='constrained');bins=np.linspace(-1,1,65)
ax.hist(p,bins=bins,density=True,alpha=.6,label=f'Same user, different words (n={len(p):,})',color='#237f9c');ax.hist(n,bins=bins,density=True,alpha=.55,label=f'Different users, same word (n={len(n):,})',color='#db8650')
auc=roc_auc_score(np.r_[np.ones(len(p)),np.zeros(len(n))],np.r_[p,n]);ax.set(xlabel='Cosine similarity',ylabel='Density',title=f'Frozen Audio ECAPA: held-out word identity stability\nPair AUC = {auc:.3f}');ax.legend(fontsize=9);fig.savefig(OUT/'01_ecapa_similarity_distribution.png',dpi=190);plt.close(fig)
mats={};crossword=w[:,None]!=w
for key in z:
 s=z[key]@z[key].T;m=np.array([[s[(u[:,None]==i)&(u[None,:]==j)&crossword].mean() for j in range(1,9)] for i in range(1,9)]);mats[key]=m
vmin=min(m.min() for m in mats.values());vmax=1.
for idx,key in enumerate(['audio','acc'],2):
 fig,ax=plt.subplots(figsize=(7,6),layout='constrained');im=ax.imshow(mats[key],vmin=vmin,vmax=vmax,cmap='viridis');ax.set_xticks(range(8),names);ax.set_yticks(range(8),names);ax.set_title(('Audio ECAPA' if key=='audio' else 'VR ACC encoder')+' identity similarity\nMean cross-word pair cosine; identical recordings');
 for i in range(8):
  for j in range(8):ax.text(j,i,f'{mats[key][i,j]:.2f}',ha='center',va='center',fontsize=9,color='white' if mats[key][i,j]<(vmin+vmax)/2 else 'black')
 fig.colorbar(im,ax=ax,label='Cosine similarity');fig.savefig(OUT/f'{idx:02d}_{key}_identity_heatmap.png',dpi=190);plt.close(fig)
o=np.triu_indices(8,1);corr=float(spearmanr(mats['audio'][o],mats['acc'][o]).statistic)
result=dict(users=names,n_recordings=len(u),words=['Down','Up','Zoom','Record','Play','Reset'],repeats_per_user_word=2,audio_same_user_different_word_mean=float(p.mean()),audio_different_user_same_word_mean=float(n.mean()),audio_pair_auc=float(auc),audio_positive_pairs=len(p),audio_negative_pairs=len(n),crossmodal_offdiagonal_spearman=corr,matrices={k:v.tolist() for k,v in mats.items()},limits='Frozen encoders; eight registered identities; held-out recordings and words; inferred word order; correlated pairs are not independent trials; descriptive diagnostic, not authentication FAR/EER. VR input is HMD velocity-derived representation, not hardware accelerometer.')
(OUT/'diagnostics.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='matrices'},indent=2))
