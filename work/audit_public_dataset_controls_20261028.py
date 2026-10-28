"""Prespecified dataset-confound controls; no generated features or model selection on final."""
import sys,json
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
sys.path.insert(0,'/root');import evaluate_public_baselines_20261028 as e
ROOT=e.ROOT;results={}
for dataset in ['esmb','vibravox']:
 p=json.loads((ROOT/dataset/'protocol_private.json').read_text());d=dict(np.load(str(ROOT/dataset/'inputs.npz')));x=d['acc'];raw=np.concatenate([x.mean(2),x.std(2,ddof=1),x.mean(1),x.std(1,ddof=1)],1)
 amplitude=np.stack([x.mean((1,2)),x.std((1,2)),x.min((1,2)),x.max((1,2))],1)
 feats={k:{int(i):a[d['unit_id']==i].mean(0) for i in np.unique(d['unit_id'])} for k,a in [('raw320',raw),('amplitude4',amplitude)]};rows={}
 for owner in p['owners']:
  known=next(u for u in p['owners'] if u!=owner);fitids=p['roles'][owner]['auth_train']+p['roles'][known]['auth_train'];y=np.r_[np.ones(10,dtype=bool),np.zeros(10,dtype=bool)];rr={}
  for key,bank in feats.items():
   get=lambda ids:np.stack([bank[int(i)] for i in ids]);tx=get(fitids);folds=list(StratifiedKFold(5,shuffle=True,random_state=42).split(tx,y));candidates=[]
   for C in (1.,10.,100.):
    for gamma in ('scale',.1):
     auc=[roc_auc_score(y[v],e.score(e.fit(tx[a],y[a],C,gamma),tx[v])) for a,v in folds];candidates.append({'C':C,'gamma':gamma,'cv_auc':float(np.mean(auc))})
   cfg=min(candidates,key=lambda z:(-z['cv_auc'],z['C'],str(z['gamma'])));m=e.fit(tx,y,cfg['C'],cfg['gamma']);score=lambda ids:e.score(m,get(ids));t=e.threshold(score(p['roles'][owner]['calibration']),score(p['roles'][known]['calibration']));g=score(p['roles'][owner]['final']);unk=np.concatenate([score(p['roles'][u]['final']) for u in p['unknown']]);rr[key]={'config':cfg,'threshold':t,'unknown':e.metrics(g,unk,t)}
  if dataset=='vibravox':
   text={int(r['unit']):r['text'] for r in p['source']};vec=TfidfVectorizer(analyzer='char_wb',ngram_range=(2,4));z=vec.fit_transform([text[i] for i in fitids]);m=LogisticRegression(C=1.,class_weight='balanced',max_iter=1000,random_state=42).fit(z,y);score=lambda ids:m.decision_function(vec.transform([text[i] for i in ids]));t=e.threshold(score(p['roles'][owner]['calibration']),score(p['roles'][known]['calibration']));g=score(p['roles'][owner]['final']);unk=np.concatenate([score(p['roles'][u]['final']) for u in p['unknown']]);rr['text_only_diagnostic']={'unknown':e.metrics(g,unk,t),'method':'Fixed char2–4 TFIDF + logistic C1; fit vocabulary on authentication train only. Diagnostic, not authentication candidate.'}
  else:rr['text_only_diagnostic']={'status':'Unavailable: cached ESMB source files have no verified transcripts.'}
  rows[owner]=rr;print('CONTROL_DONE',dataset,owner,{k:v.get('unknown',{}).get('auc') for k,v in rr.items()},flush=True)
 results[dataset]=rows
(ROOT/'confound_controls_public.json').write_text(json.dumps(results,indent=2));print('CONTROLS_COMPLETE',flush=True)
