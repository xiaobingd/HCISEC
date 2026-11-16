"""Frozen personal CycleGAN supplement to the locked representation pilot."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='2'
import json
from pathlib import Path
import numpy as np
import joblib
from sklearn.svm import SVC
from evaluate_crossword_encoder_distill import norm,metric,threshold,dump,KEYS,USERS,TARGETS,CAL,FINAL

def main():
    source=Path('outputs/mei_crossword_representation_20261116')
    assets=Path('outputs/mei_cycle_matched_assets_20261116')
    cache=Path('outputs/mei_cycle_matched_content_20261116')
    out=Path('outputs/mei_representation_cycle_supplement_20261116');out.mkdir(exist_ok=True)
    d=np.load(source/'vectors_private.npz');front=np.load('outputs/mei_common_frontend_private.npz')
    assert np.array_equal(d['sample_id'],front['sample_id'])
    user,word=d['user'],d['word'];raw=d['raw']
    locks=json.loads((source/'fusion_selection_public.json').read_text())['methods']
    scalers=joblib.load(assets/'gan_scalers_private.joblib')
    lookup={(u,w):np.flatnonzero((user==u)&(word==w)) for u in USERS for w in CAL+FINAL}
    def reg(u,w,f):return lookup[u,w][f*4:f*4+4]
    def query(w,f,uu):return np.concatenate([lookup[u,w][(1-f)*4:(1-f)*4+4] for u in uu])
    rows=[];saved={};taus={};audit=[]
    for owner in TARGETS:
        c=np.load(cache/f'owner{owner}_features_private.npz');ix=c['indices']
        z={}
        for branch in ['input256','cycle256']:
            z[branch]=np.zeros((len(user),256),dtype='float32')
            z[branch][ix]=norm(scalers[branch].transform(c[branch]))
        needed=np.concatenate([v[:8] for v in lookup.values()])
        assert set(needed).issubset(set(ix))
        # Historical personal generator used only its owner's first three
        # repeats of words20..23. Cal query uses repeats4..7; final words24..29.
        original=json.loads((assets/'split_private.json').read_text())
        own_train=np.array(original['register']);own_train=own_train[user[own_train]==owner]
        assert len(own_train)==12
        qq=np.r_[np.concatenate([query(w,0,USERS) for w in CAL]),
                 np.concatenate([query(w,f,TARGETS) for w in FINAL for f in [0,1]])]
        assert not set(own_train)&set(qq)
        audit.append(dict(user=f'U{owner}',cached_records=len(ix),personal_training_query_overlap=0))
        views={'raw':raw}
        for name in ['E0','E1','E2','E4']:
            alpha=locks[name]['selected']['alpha']
            base=raw if alpha==0 else np.c_[raw,alpha*d[name]]/np.sqrt(1+alpha**2)
            views[name]=base
            for branch in z:views[name+'_'+branch]=np.c_[base,.5*z[branch]]/np.sqrt(1.25)
        for name,v in views.items():
            models={};scores={}
            def score(rw,qw,f,uu):
                key=(rw,qw,f,tuple(uu))
                if key not in scores:
                    if (rw,f) not in models:
                        rr=np.concatenate([reg(u,rw,f) for u in USERS])
                        models[rw,f]=SVC(C=10,gamma='scale',class_weight='balanced').fit(v[rr],user[rr]==owner)
                    rr=reg(owner,rw,f);qq=query(qw,f,uu);assert not set(rr)&set(qq)
                    scores[key]=models[rw,f].decision_function(v[qq])
                return scores[key]
            yy=[];ss=[]
            for qw in CAL:
                y=user[query(qw,0,USERS)]==owner
                for rw in CAL:
                    if rw!=qw:
                        yy.extend([y,y]);ss.extend([score(qw,qw,0,USERS),score(rw,qw,0,USERS)])
            tau=threshold(np.concatenate(yy),np.concatenate(ss));taus[f'{owner}_{name}']=tau
            for f in [0,1]:
                for qw in FINAL:
                    y=user[query(qw,f,TARGETS)]==owner
                    for rw in FINAL:
                        if rw==qw:continue
                        for condition,r in [('sameword',qw),('crossword',rw)]:
                            s=score(r,qw,f,TARGETS);key=f'{owner}_{name}_{f}_{qw}_{rw}_{condition}'
                            saved[key]=s
                            rows.append(dict(user=f'U{owner}',method=name,condition=condition,fold=f,qw=qw,rw=rw,**metric(y,s,tau)))
        dump(out/'status.json',dict(stage='evaluation',last_user=owner));print('OWNER_COMPLETE',owner,flush=True)
    error=0.
    for r in rows:
        u=int(r['user'][1:]);name=r['method'];f=r['fold'];qw=r['qw'];rw=r['rw']
        s=saved[f'{u}_{name}_{f}_{qw}_{rw}_{r["condition"]}']
        m=metric(user[query(qw,f,TARGETS)]==u,s,taus[f'{u}_{name}'])
        error=max(error,max(abs(m[k]-r[k]) for k in KEYS))
    assert error==0
    np.savez_compressed(out/'scores_private.npz',**saved)
    def aggregate(rr):return {k:float(np.mean([r[k] for r in rr])) for k in KEYS}
    summary=[];peruser=[]
    for name in views:
        for condition in ['sameword','crossword']:
            rr=[r for r in rows if r['method']==name and r['condition']==condition]
            summary.append(dict(method=name,condition=condition,**aggregate(rr)))
            for u in TARGETS:peruser.append(dict(method=name,condition=condition,user=f'U{u}',**aggregate([r for r in rr if r['user']==f'U{u}'])))
    # Reproduce previous no-supplement controls exactly before interpreting gain.
    prior=json.loads((source/'results_public.json').read_text())['summary']
    parity=0.
    for r in summary:
        if r['method'] not in ['raw','E0','E1','E2','E4']:continue
        oldname='raw_rbf' if r['method']=='raw' else r['method']+'_fused_rbf'
        old=next(q for q in prior if q['method']==oldname and q['condition']==r['condition'])
        parity=max(parity,max(abs(r[k]-old[k]) for k in KEYS))
    assert parity<1e-12,parity
    dump(out/'results_public.json',dict(summary=summary,per_user=peruser))
    dump(out/'verification_public.json',dict(state='PASS',workpoints=len(rows),score_replay_error=error,
         prior_control_error=parity,owner_audit=audit,fixed_cycle_weight=.5,no_new_generator_training=True,
         historical_personal_adaptation='own12 records words20..23;unchanged',
         cal_query_and_final_not_in_personal_adaptation=True,
         final_targets_and_attackers=TARGETS,training_registration_calibration_users=USERS,
         limitation='fixed historical personal adaptation;not a new K4-only adaptation protocol'))
    lines=['# 既有身份表征补充个人CycleGAN特征','',
      '沿用表征实验全部锁定模型、原始统计融合权重及相同RBF后端。额外CycleGAN分支固定权重0.5，不用最终测试或已参与个人微调的开发词重新选权重。',
      'input256=循环一致性训练的G输入编码；cycle256=ACC→Audio→ACC后再编码。个人G/F保留历史本人12条100步微调，不重新训练。九人训练与负类注册，最终八人（排除U5目标及攻击者）。',
      '最终同词/跨词使用完全相同查询，六词30方向两折K4；Cal21..23注册前4条、查询4..7，阈值在条件和折之间共享。E0=CE；E1=跨词SupCon；E2=Audio关系蒸馏；E4=ACC自蒸馏。',
      '', '|方法|条件|TAR%|FAR%|HTER%|EER%|AUC|','|---|---|---:|---:|---:|---:|---:|']
    for r in summary:lines.append(f"|{r['method']}|{r['condition']}|{100*r['tar']:.2f}|{100*r['far']:.2f}|{100*r['hter']:.2f}|{100*r['eer']:.2f}|{r['auc']:.4f}|")
    lines+=['','## 逐用户','', '|用户|方法|条件|TAR%|FAR%|HTER%|EER%|AUC|','|---|---|---|---:|---:|---:|---:|---:|']
    for r in peruser:lines.append(f"|{r['user']}|{r['method']}|{r['condition']}|{100*r['tar']:.2f}|{100*r['far']:.2f}|{100*r['hter']:.2f}|{100*r['eer']:.2f}|{r['auc']:.4f}|")
    lines+=['',f'核查PASS：{len(rows)}工作点，保存分数重算误差{error}，与上一轮无补充对照误差{parity}。单种子、同会话、名义词序；固定历史个人适配包含额外本人注册数据，故结果回答现有个人CycleGAN特征能否补充，不单独归因于循环损失。创新点一定版未修改。']
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    dump(out/'status.json',dict(stage='complete'));print('COMPLETE',flush=True)

if __name__=='__main__':main()
