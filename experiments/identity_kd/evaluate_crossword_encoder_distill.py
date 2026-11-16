"""Fixed-query K4 authentication after representation choices are locked."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='2'
import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score,roc_curve
from sklearn.svm import SVC

USERS=list(range(1,10));TARGETS=[1,2,3,4,6,7,8,9]
DEV=[18,19,20];CAL=[21,22,23];FINAL=list(range(24,30))
KEYS=['tar','far','frr','hter','eer','auc']


def dump(path,value):Path(path).write_text(json.dumps(value,indent=2),encoding='utf-8')
def norm(x):return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-8)
def metric(y,s,t):
    far=float((s[~y]>=t).mean());frr=float((s[y]<t).mean())
    f,p,_=roc_curve(y,s,drop_intermediate=False);j=np.argmin(abs(f-(1-p)))
    return dict(tar=1-frr,far=far,frr=frr,hter=(far+frr)/2,
                eer=float((f[j]+1-p[j])/2),auc=float(roc_auc_score(y,s)))
def threshold(y,s):
    f,p,t=roc_curve(y,s,drop_intermediate=False)
    j=min(range(len(t)),key=lambda j:(abs(f[j]-(1-p[j])),f[j]))
    return float(t[j])


def main(a):
    out=Path(a.out);data=np.load(out/'vectors_private.npz')
    protocol=json.loads((out/'protocol_public.json').read_text(encoding='utf-8'))
    selection=json.loads((out/'selection_public.json').read_text())
    user,word=data['user'],data['word'];raw=data['raw']
    lookup={(u,w):np.flatnonzero((user==u)&(word==w)) for u in USERS for w in DEV+CAL+FINAL}
    assert all(len(v)>=8 for v in lookup.values())
    def reg(w,fold,u):return lookup[u,w][fold*4:fold*4+4]
    def query(w,fold,users):return np.concatenate([lookup[u,w][(1-fold)*4:(1-fold)*4+4] for u in users])
    pools=[set(np.flatnonzero(word<18)),set(np.flatnonzero(np.isin(word,DEV))),
           set(np.flatnonzero(np.isin(word,CAL))),set(np.flatnonzero(np.isin(word,FINAL)))]
    assert all(not pools[i]&pools[j] for i in range(4) for j in range(i))
    def fused(z,alpha):return raw if alpha==0 else np.c_[raw,alpha*z]/np.sqrt(1+alpha**2)
    def score_fn(v,backend):
        models={}
        def score(owner,rw,qw,fold,q_users):
            rr=reg(rw,fold,owner);qq=query(qw,fold,q_users)
            assert not set(rr)&set(qq)
            if backend=='cos':return v[qq]@norm(v[rr].mean(0))
            key=(owner,rw,fold)
            if key not in models:
                ix=np.concatenate([reg(rw,fold,u) for u in USERS])
                models[key]=SVC(C=10,gamma='scale',class_weight='balanced').fit(v[ix],user[ix]==owner)
            return models[key].decision_function(v[qq])
        return score
    def dev_quality(v):
        score=score_fn(v,'rbf');values=[]
        for u in USERS:
            for fold in [0,1]:
                for qw in DEV:
                    y=user[query(qw,fold,USERS)]==u
                    for rw in DEV:
                        if rw!=qw:values.append(roc_auc_score(y,score(u,rw,qw,fold,USERS)))
        return float(np.mean(values))
    # No final query metric is computed before all fusion weights are locked.
    locks={}
    for name in ['E0','E1','E2','E3','E4']:
        choices=[dict(alpha=alpha,dev_cross_rbf_auc=dev_quality(fused(data[name],alpha))) for alpha in [0,.25,.5,1.]]
        chosen=max(choices,key=lambda r:(r['dev_cross_rbf_auc'],-r['alpha']))
        locks[name]=dict(selected=chosen,candidates=choices)
    dump(out/'fusion_selection_public.json',dict(rule='Dev-only RBF cross AUC,then lower alpha;zero allowed',methods=locks))
    views={'raw_rbf':(raw,'rbf'),'raw_cos':(raw,'cos')}
    for name in ['E0','E1','E2','E3','E4']:
        views[name+'_embedding_rbf']=(data[name],'rbf')
        views[name+'_embedding_cos']=(data[name],'cos')
        views[name+'_fused_rbf']=(fused(data[name],locks[name]['selected']['alpha']),'rbf')
    rows=[];private_scores={};private_thresholds={}
    for name,(v,backend) in views.items():
        score=score_fn(v,backend)
        cache={}
        def scores(owner,rw,qw,fold,users):
            key=(owner,rw,qw,fold,tuple(users))
            if key not in cache:cache[key]=score(owner,rw,qw,fold,users)
            return cache[key]
        for u in TARGETS:
            labels=[];values=[]
            for qw in CAL:
                y=user[query(qw,0,USERS)]==u
                for rw in CAL:
                    if rw==qw:continue
                    labels.extend([y,y]);values.extend([scores(u,qw,qw,0,USERS),scores(u,rw,qw,0,USERS)])
            tau=threshold(np.concatenate(labels),np.concatenate(values))
            private_thresholds[f'{name}_{u}']=tau
            for fold in [0,1]:
                for qw in FINAL:
                    qq=query(qw,fold,TARGETS);y=user[qq]==u
                    assert y.sum()==4 and (~y).sum()==28
                    for rw in FINAL:
                        if rw==qw:continue
                        for condition,registration_word in [('sameword',qw),('crossword',rw)]:
                            s=scores(u,registration_word,qw,fold,TARGETS)
                            key=f'{name}_{u}_{fold}_{qw}_{rw}_{condition}'
                            private_scores[key]=s
                            rows.append(dict(method=name,user=f'U{u}',fold=fold,query_word=qw,
                                    cross_registration_word=rw,condition=condition,**metric(y,s,tau)))
        dump(out/'status.json',dict(stage='authentication',method=name,completed_methods=len(set(r['method'] for r in rows)),total_methods=len(views)))
        print('AUTH_COMPLETE',name,flush=True)
    np.savez_compressed(out/'scores_private.npz',**private_scores)
    dump(out/'thresholds_private.json',private_thresholds)
    error=0.
    for r in rows:
        name=r['method'];u=int(r['user'][1:]);fold=r['fold'];qw=r['query_word'];rw=r['cross_registration_word']
        y=user[query(qw,fold,TARGETS)]==u
        ss=private_scores[f'{name}_{u}_{fold}_{qw}_{rw}_{r["condition"]}']
        m=metric(y,ss,private_thresholds[f'{name}_{u}'])
        error=max(error,max(abs(m[k]-r[k]) for k in KEYS))
    assert error==0 and len(rows)==len(views)*8*2*30*2
    per_user=[];summary=[];per_word=[]
    def aggregate(rr):return {k:float(np.mean([r[k] for r in rr])) for k in KEYS}
    for name in views:
        for condition in ['sameword','crossword']:
            rr=[r for r in rows if r['method']==name and r['condition']==condition]
            summary.append(dict(method=name,condition=condition,**aggregate(rr)))
            for u in TARGETS:per_user.append(dict(method=name,condition=condition,user=f'U{u}',**aggregate([r for r in rr if r['user']==f'U{u}'])))
            for w in FINAL:per_word.append(dict(method=name,condition=condition,word=w,**aggregate([r for r in rr if r['query_word']==w])))
    geometry=[]
    ix=np.concatenate([lookup[u,w][:8] for u in TARGETS for w in FINAL]);y=user[ix];w=word[ix]
    positive=(y[:,None]==y[None,:])&(w[:,None]!=w[None,:]);negative=(y[:,None]!=y[None,:])&(w[:,None]==w[None,:])
    for name in ['raw','E0','E1','E2','E3','E4']:
        z=norm(data[name][ix]);sim=z@z.T
        geometry.append(dict(method=name,same_user_cross_word=float(sim[positive].mean()),
                             different_user_same_word=float(sim[negative].mean()),
                             similarity_gap=float(sim[positive].mean()-sim[negative].mean())))
    dump(out/'per_user_public.json',per_user);dump(out/'per_word_public.json',per_word)
    dump(out/'results_public.json',dict(state='complete',seed=protocol['seed'],summary=summary,geometry=geometry,
        audio_development_choice=selection['deployment_choice'],positive_audio_rho=selection['best_positive']['rho']))
    dump(out/'verification_public.json',dict(state='PASS',workpoints=len(rows),saved_score_metric_error=error,
        fit_dev_cal_final_word_disjoint=True,registration_query_record_disjoint=True,
        identical_queries_for_same_cross=True,thresholds_shared_across_conditions_and_folds=True,
        final_U5_target_and_attack_removed=True,training_registration_and_calibration_still_nine_users=True,
        all_alpha_choices_locked_on_Dev_before_final_scores=True,
        limits='saved-score replay and index checks;not independent full retraining'))
    lines=['# Audio辅助跨词身份表征学习：首轮结果','',
        '独立于已冻结创新点一；九人参与原始训练，U5不作为最终目标或攻击者，保留八名目标及七名攻击来源。所有编码器使用相同均值/标准差池化、128维投影和训练批次。',
        '', 'E0=CE；E1=CE+跨词SupCon；E2=最佳正rho Audio关系KL；E3=对应rho跨身份错配；E4=对应rho ACC自关系。Audio的rho=0选择由E1提供，不能把E2正权重诊断冒充必然被开发集选择。',
        '', f"开发选择Audio部署分支：{selection['deployment_choice']}；最佳正rho={selection['best_positive']['rho']}。", '',
        '新结构从400步CE共同预训练起步，每组继续1000步。ReDimNet2-B3冻结训练词缓存，不更新教师。主结果看统一RBF，余弦用于表征诊断；原始统计融合alpha只在Dev选，包含0。',
        '', '|方法|条件|TAR%|FAR%|HTER%|EER%|AUC|','|---|---|---:|---:|---:|---:|---:|']
    for r in summary:lines.append(f"|{r['method']}|{r['condition']}|{100*r['tar']:.2f}|{100*r['far']:.2f}|{100*r['hter']:.2f}|{100*r['eer']:.2f}|{r['auc']:.4f}|")
    lines+=['','## 开发集融合选择','', '|方法|alpha|Dev跨词RBF AUC|','|---|---:|---:|']
    for name,r in locks.items():lines.append(f"|{name}|{r['selected']['alpha']}|{r['selected']['dev_cross_rbf_auc']:.4f}|")
    lines+=['','## 逐用户','', '|用户|方法|条件|TAR%|FAR%|HTER%|EER%|AUC|','|---|---|---|---:|---:|---:|---:|---:|']
    for r in sorted(per_user,key=lambda r:(r['user'],r['method'],r['condition'])):lines.append(f"|{r['user']}|{r['method']}|{r['condition']}|{100*r['tar']:.2f}|{100*r['far']:.2f}|{100*r['hter']:.2f}|{100*r['eer']:.2f}|{r['auc']:.4f}|")
    lines+=['','固定查询K4、六词30方向、两折；Cal21–23同词/跨词等权池确定共享阈值，最终24–29不调参。同一查询在词对中重复，不是独立新试次。第一点模型和结果未修改。',
        '', '一个种子的定位实验，不是三种子正式确认。词序名义映射、历史同会话、未见用户和重戴均未在本轮验证。表征和融合的增益需分别与相同后端E0/E1/错配/自关系比较；alpha=0表示开发未选择补充分支。',
        '', f'核查PASS：{len(rows)}工作点保存分数重算误差0，记录/词隔离与相同查询检查通过。训练梯度与编码器参数变化见各方法gradient/complete文件。私有数据、特征、模型、分数和阈值不公开。']
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    dump(out/'status.json',dict(stage='complete'))
    protocol=json.loads((out/'protocol_public.json').read_text(encoding='utf-8'))
    protocol['state']='complete';dump(out/'protocol_public.json',protocol)
    print('REPRESENTATION_AUTH_ALL_COMPLETE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);main(p.parse_args())
