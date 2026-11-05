"""Verify and summarize all phase-1 scores; no model/threshold selection."""
import json,sys,zipfile
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
ROOT=Path(sys.argv[1] if len(sys.argv)>1 else '/autodl-fs/data/pira_stage1_20261028');TARGETS=(1,2,4,5,7)
allrows=[];counts={};verified=0
for u in TARGETS:
 folder=ROOT/('U%d'%u);assert (folder/'COMPLETE').exists(),u
 data=json.loads((folder/'results.json').read_text());manifest=json.loads((folder/'manifest.json').read_text());scores=np.load(str(folder/'scores_private.npz'))
 assert len(data)==576
 uid=scores['negative_uid'];ids=scores['negative_rows'];unknown=np.isin(uid,[3,8]);known=~unknown
 gw=np.asarray(manifest['rows'][str(u)]['test'])//10;nw=ids//10
 counts[str(u)]={'genuine':len(gw),'unknown':int(unknown.sum()),'known':int(known.sum()),'calibration':len(manifest['rows'][str(u)]['cal'])}
 for row in data:
  tag='seed%d_n%d_%s_%s'%(row['seed'],row['n_enroll'],row['feature'],row['method']);g=scores[tag+'_g'];n=scores[tag+'_n'];t=row['threshold']
  assert len(g)==len(gw) and len(n)==len(uid)
  assert sum(v['accepted'] for v in row['attacks'].values())==int(np.sum(n>=t))
  for kind,mask in [('unknown',unknown),('known',known)]:
   r=row[kind];m=n[mask]
   assert abs(r['tar']-np.mean(g>=t))<1e-10 and abs(r['far']-np.mean(m>=t))<1e-10
   assert abs(r['auc']-roc_auc_score(np.r_[np.ones(len(g)),np.zeros(len(m))],np.r_[g,m]))<1e-10
   per=[]
   for w in range(20,30):
    a=g[gw==w];b=n[(nw==w)&mask]
    if len(a) and len(b):per.append(roc_auc_score(np.r_[np.ones(len(a)),np.zeros(len(b))],np.r_[a,b]))
   r['same_word_auc']=float(np.mean(per));r['same_word_auc_words']=len(per)
   rates=[np.mean(n[(uid==v)&(nw==w)]>=t) for v in np.unique(uid[mask]) for w in range(20,30) if np.any((uid==v)&(nw==w))]
   r['identity_word_macro_far']=float(np.mean(rates))
  verified+=1
 allrows.extend(data)
public=[]
for u in TARGETS:
 for n in (10,20,40,60):
  for feat in ('raw_profile320','cycle772','cycle772_teacher64'):
   methods=sorted({r['method'] for r in allrows})
   for method in methods:
    z=[r for r in allrows if r['target']==u and r['n_enroll']==n and r['feature']==feat and r['method']==method];assert len(z)==3
    out={'target':u,'n_enroll':n,'feature':feat,'method':method,'seeds':[42,43,44]}
    for kind in ('known','unknown'):
     out[kind]={k:{'mean':float(np.mean([r[kind][k] for r in z])),'std':float(np.std([r[kind][k] for r in z]))} for k in ('tar','far','auc','hter','tar_at_far5_roc','same_word_auc','identity_word_macro_far')}
    public.append(out)
(ROOT/'summary_public.json').write_text(json.dumps({'protocol':json.loads((ROOT/'locked_protocol.json').read_text()),'counts':counts,'verified_score_sets':verified,'rows':public},indent=2))
(ROOT/'verification.json').write_text(json.dumps({'all_score_sets_verified':verified,'expected':2880,'checks':['accepted counts','FAR/TAR','known and unknown AUC','word matched AUC','all cohorts complete']},indent=2))
# Scores/identities stay on the private server; export only summaries and manifests/locks.
with zipfile.ZipFile(str(ROOT/'review_bundle.zip'),'w',zipfile.ZIP_DEFLATED) as z:
 for p in [ROOT/'summary_public.json',ROOT/'verification.json',ROOT/'locked_protocol.json']:
  z.write(str(p),p.name)
 for u in TARGETS:
  p=ROOT/('U%d'%u)/'manifest.json';z.write(str(p),'U%d_manifest_private.json'%u)
# Figure uses prespecified 60 enrollment and K=2, not the post hoc best K.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axs=plt.subplots(1,3,figsize=(15,4.4),constrained_layout=True)
styles={'kmeans_K1_cosine':('Single prototype','o'),'kmeans_K2_cosine':('Two prototypes','s'),'kmeans_K2_local':('Two + local scale','^'),'deep_svdd':('Deep SVDD','x'),'rbf':('Known-negative RBF','D')}
for ax,key,title in zip(axs,['auc','same_word_auc','far'],['Unknown-attacker AUC','Same-word unknown AUC','Unknown FAR (positive-only threshold)']):
 for method,(label,marker) in styles.items():
  rr=[next(r for r in public if r['target']==u and r['n_enroll']==60 and r['feature']=='cycle772_teacher64' and r['method']==method) for u in TARGETS]
  ax.plot(range(len(TARGETS)),[r['unknown'][key]['mean'] for r in rr],marker=marker,label=label)
 ax.set_xticks(range(len(TARGETS)),['U%d'%u for u in TARGETS]);ax.set_title(title);ax.set_ylim(-.03,1.03);ax.grid(alpha=.2)
axs[0].legend(fontsize=8);fig.suptitle('Frozen CycleGAN 772 + ACC teacher 64 | 60 enrollment | mean of 3 seeds\nAuthentication words held out; historical recordings; unknown identities U3/U8')
fig.savefig(str(ROOT/'stage1_comparison.png'),dpi=160);fig.savefig(str(ROOT/'stage1_comparison.svg'));plt.close(fig)
print('VERIFIED_AND_EXPORTED',verified,len(public),counts,flush=True)
for u in TARGETS:
 print('USER',u,flush=True)
 for method in styles:
  row=next(r for r in public if r['target']==u and r['n_enroll']==60 and r['feature']=='cycle772_teacher64' and r['method']==method)
  print(method,{k:round(row['unknown'][k]['mean'],4) for k in ['tar','far','auc','same_word_auc']},flush=True)
