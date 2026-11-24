"""Descriptive per-query-word strata using already locked single-query protocol."""
import json
from pathlib import Path
import numpy as np
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits
from calibrate_threeword import measure
from prepare_phrase_auth import REG,SHORT,ENGLISH,CHINESE

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'conditional_correspondence/enrollment_boundary_singleword'

def main():
    d=np.load(ROOT/'phrase_auth/frontend_private.npz');u,w=d['user'],d['word'];reg=np.flatnonzero(np.isin(w,REG))
    views=np.load(ROOT/'conditional_correspondence/stage3/views_private.npz')
    choices=json.loads((OUT/'selection_private.json').read_text());locks=json.loads((OUT/'thresholds_private.json').read_text())
    vocabulary=json.loads((ROOT/'paired_scope.json').read_text(encoding='utf8'))['words']
    rows=[]
    with threadpool_limits(limits=2):
        for choice in choices:
            seed,method=choice['seed'],choice['method'];ix=np.array(choice['indices']);z=views[f'{seed}_A_ACC']
            heads=[]
            for us in range(1,11):
                train=np.r_[ix[u[ix]==us],reg[u[reg]!=us]]
                heads.append(SVC(C=10,class_weight='balanced').fit(z[train],u[train]==us).decision_function(z))
            scores={'RBF':np.column_stack(heads)}
            templates=np.stack([z[ix[u[ix]==us]].mean(0) for us in range(1,11)])
            templates/=np.maximum(np.linalg.norm(templates,axis=1,keepdims=True),1e-8)
            scores['cosine']=z@templates.T
            for backend,score in scores.items():
                for family,words in [('english_word',SHORT),('english_long',ENGLISH),('chinese_long',CHINESE)]:
                    for word in words:
                        qi=np.flatnonzero(w==word);assert len(qi)==80 and len(set(d['sample_id'][qi]))==80
                        for us in range(1,11):
                            tau=next(a['threshold'] for a in locks if a['seed']==seed and a['method']==method and a['backend']==backend and a['user']==us and a['words']==1)
                            m=measure(score[qi,us-1],u[qi]==us,tau)
                            assert m['genuine_trials']==8 and m['attack_trials']==72
                            rows.append(dict(seed=seed,method=method,backend=backend,family=family,query_word_id=int(word),query_word=vocabulary[word],user=us,**m))
            print('BOUNDARY_WORD',seed,method,flush=True)
    assert len(rows)==7200
    (OUT/'per_word_seed_user_public.json').write_text(json.dumps(rows,ensure_ascii=False),encoding='utf8')
    macros=[];users=[]
    for method in sorted(set(a['method'] for a in rows)):
        for backend in scores:
            for word in SHORT+ENGLISH+CHINESE:
                sub=[a for a in rows if a['method']==method and a['backend']==backend and a['query_word_id']==word];assert len(sub)==30
                cfg={key:sub[0][key] for key in ['method','backend','family','query_word_id','query_word']}
                macros.append(dict(**cfg,**{key:float(np.mean([a[key] for a in sub])) for key in ['far','frr','hter','eer','auc']}))
                for us in range(1,11):
                    part=[a for a in sub if a['user']==us];assert len(part)==3
                    users.append(dict(**cfg,user=us,**{key:float(np.mean([a[key] for a in part])) for key in ['far','frr','hter','eer','auc']}))
    (OUT/'per_word_summary_public.json').write_text(json.dumps(dict(macro=macros,per_user=users),ensure_ascii=False),encoding='utf8')
    print('BOUNDARY_PER_WORD_COMPLETE',flush=True)

if __name__=='__main__':main()
