"""Paired same/cross-word verification on identical query recordings.

Frozen nine-user models; U5 remains a training identity and attacker,
but is excluded as a target. K4 is required by available repeats.
"""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ[key]='2'
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from crossword_neural_verifier import ACCEncoder,Verifier
from run_crossword_neural_verifier import norm,pair_features,threshold,metrics,dump

TARGETS=[1,2,3,4,6,7,8,9]
CAL=[21,22,23]
FINAL=[24,25,26,27,28,29]
NAMES={24:'Down',25:'Up',26:'Zoom',27:'Record',28:'Play',29:'Reset'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(a):
    torch.set_num_threads(2)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    source=Path(a.source);d=np.load(a.frontend)
    assert set(d['user'])==set(range(1,10))
    n=np.load(source/'normalization_private.npz');b=d['spectrum']
    fit=np.flatnonzero(d['word']<18)
    raw=np.concatenate([b.mean(-1).reshape(len(b),-1),b.std(-1).reshape(len(b),-1),
        *[b[...,j].mean(-1).reshape(len(b),-1) for j in np.array_split(np.arange(28),4)]],1)
    scaler=StandardScaler().fit(raw[fit])
    assert np.array_equal(scaler.mean_,n['raw_mean'])
    assert np.array_equal(scaler.scale_,n['raw_scale'])
    raw=norm(scaler.transform(raw)).astype('float32')
    lookup={(u,w):np.flatnonzero((d['user']==u)&(d['word']==w))
            for u in range(1,10) for w in CAL+FINAL}
    assert all(len(j)>=8 for j in lookup.values())
    metadata={}
    def support(word,fold,u):return lookup[u,word][fold*4:fold*4+4]
    def query(word,fold):
        start=(1-fold)*4
        return np.concatenate([lookup[u,word][start:start+4] for u in range(1,10)])
    # All physical recordings used for Cal stay disjoint from final evaluation.
    cp=set(np.concatenate([v[:8] for (u,w),v in lookup.items() if w in CAL]))
    fp=set(np.concatenate([v[:8] for (u,w),v in lookup.items() if w in FINAL]))
    assert not cp&fp and not set(fit)&(cp|fp)
    for fold in [0,1]:
        for word in CAL+FINAL:
            ss=np.concatenate([support(word,fold,u) for u in range(1,10)])
            qq=query(word,fold)
            assert not set(ss)&set(qq)
    dump(out/'protocol_public.json',dict(state='running',source='frozen nine-user verifier pilot',
        target_users=TARGETS,training_users=list(range(1,10)),attack_sources='all other eight identities including U5',
        fit_words=list(range(18)),calibration_words=CAL,final_words=FINAL,
        registration_per_target=4,query_per_identity_per_word_per_fold=4,
        folds='0:register repeat0:4/query4:8;1:register4:8/query0:4',
        ordered_crossword_pairs=30,identical_queries_in_each_same_cross_pair=True,
        thresholds='per target/method/fold on equal-weight pooled same/cross Cal scores;shared final conditions',
        rbf='C10,gamma scale,balanced;fit K4 genuine and4 from each other identity per registration word',
        models='raw_rbf;frozen local verifier;frozen Audio-decision verifier',
        limits=['nominal chronological word labels','K4 evaluation differs from previous K12',
            'neural verifiers trained with K12,not retrained for K4',
            'pairs share queries and supports;not independent trials',
            'single seed and historically explored single-session data']))
    t=np.load(source/'tasks_private.npz')
    train={k:t['train_'+k] for k in ['q','r','y','owner']}
    base=LogisticRegression(C=1,max_iter=1000).fit(pair_features(raw,train),train['y'])
    enc=ACCEncoder().eval();enc.load_state_dict(torch.load(source/'encoder_private.pt',weights_only=True))
    with torch.no_grad():
        tokens=torch.cat([enc(torch.tensor((b[j:j+64]-n['mean'])/n['std'])) for j in range(0,len(b),64)])
    networks={}
    for name in ['local','audio']:
        net=Verifier().eval();net.load_state_dict(torch.load(source/(name+'_best_private.pt'),weights_only=True))
        networks[name]=net
    rows=[];scores_archive={};threshold_archive={}
    for method in ['raw_rbf','local','audio']:
        for fold in [0,1]:
            cache={};rbf={}
            def score(owner,registered,queried):
                key=(owner,registered,queried)
                if key in cache:return cache[key]
                qi=query(queried,fold);rr=support(registered,fold,owner)
                assert not set(qi)&set(rr)
                if method=='raw_rbf':
                    if (owner,registered) not in rbf:
                        ix=np.concatenate([support(registered,fold,u) for u in range(1,10)])
                        rbf[owner,registered]=SVC(C=10,gamma='scale',class_weight='balanced').fit(raw[ix],d['user'][ix]==owner)
                    value=rbf[owner,registered].decision_function(raw[qi])
                else:
                    task=dict(q=qi,r=np.tile(rr,(len(qi),1)))
                    logits=base.decision_function(pair_features(raw,task)).astype('float32')
                    values=[]
                    with torch.no_grad():
                        for start in range(0,len(qi),12):
                            sl=slice(start,start+12)
                            values.append(networks[method](tokens[qi[sl]],tokens[task['r'][sl]],torch.tensor(logits[sl]))[0].numpy())
                    value=np.concatenate(values)
                cache[key]=value
                return value
            for owner in TARGETS:
                cy=[];cs=[]
                for qw in CAL:
                    yy=d['user'][query(qw,fold)]==owner
                    for rw in CAL:
                        if rw==qw:continue
                        # Duplicate same-word scores once per cross comparison:
                        # exactly balanced same/cross weight in threshold calibration.
                        cy.extend([yy,yy]);cs.extend([score(owner,qw,qw),score(owner,rw,qw)])
                tau=threshold(np.concatenate(cy),np.concatenate(cs))
                threshold_archive[f'{method}_{fold}_{owner}']=tau
                for qw in FINAL:
                    qi=query(qw,fold);yy=d['user'][qi]==owner
                    for rw in FINAL:
                        if rw==qw:continue
                        for condition,regword in [('sameword',qw),('crossword',rw)]:
                            ss=score(owner,regword,qw)
                            row=dict(method=method,user=f'U{owner}',fold=fold,query_word=qw,
                                     cross_registration_word=rw,condition=condition,
                                     **metrics(yy,ss,tau))
                            rows.append(row)
                            scores_archive[f'{method}_{fold}_{owner}_{qw}_{rw}_{condition}']=ss
            print('SCORED',method,'fold',fold,flush=True)
            dump(out/'status.json',dict(stage='scoring',method=method,fold=fold))
    np.savez_compressed(out/'scores_private.npz',**scores_archive)
    dump(out/'thresholds_private.json',threshold_archive)
    # Recompute every saved workpoint from scores and locked thresholds.
    cached=np.load(out/'scores_private.npz');max_error=0.
    for row in rows:
        method,fold,owner=row['method'],row['fold'],int(row['user'][1:])
        qw,rw=row['query_word'],row['cross_registration_word']
        yy=d['user'][query(qw,fold)]==owner
        tau=threshold_archive[f'{method}_{fold}_{owner}']
        ss=cached[f'{method}_{fold}_{owner}_{qw}_{rw}_{row["condition"]}']
        mm=metrics(yy,ss,tau)
        max_error=max(max_error,max(abs(mm[k]-row[k]) for k in mm))
        if row['condition']=='sameword':
            baseline=cached[f'{method}_{fold}_{owner}_{qw}_{next(w for w in FINAL if w!=qw)}_sameword']
            assert np.array_equal(ss,baseline)
    assert max_error==0
    assert len(rows)==3*2*8*30*2
    summary=[];per_user=[];per_word=[];direction=[]
    keys=['tar','far','frr','hter','eer','auc']
    def aggregate(group):return {k:float(np.mean([r[k] for r in group])) for k in keys}
    for method in ['raw_rbf','local','audio']:
        for condition in ['sameword','crossword']:
            rr=[r for r in rows if r['method']==method and r['condition']==condition]
            summary.append(dict(method=method,condition=condition,**aggregate(rr)))
            for u in TARGETS:
                per_user.append(dict(method=method,condition=condition,user=f'U{u}',
                                     **aggregate([r for r in rr if r['user']==f'U{u}'])))
            for w in FINAL:
                per_word.append(dict(method=method,condition=condition,word=NAMES[w],
                                     **aggregate([r for r in rr if r['query_word']==w])))
            for name,regwords,qwords in [('A_to_B',FINAL[:3],FINAL[3:]),('B_to_A',FINAL[3:],FINAL[:3])]:
                direction.append(dict(method=method,condition=condition,direction=name,
                    **aggregate([r for r in rr if r['query_word'] in qwords and r['cross_registration_word'] in regwords])))
    dump(out/'results_public.json',dict(state='complete',summary=summary,directions=direction))
    dump(out/'per_user_public.json',per_user);dump(out/'per_word_public.json',per_word)
    dump(out/'verification_public.json',dict(state='PASS',workpoints=len(rows),
        max_saved_score_metric_error=max_error,queries_identical_within_pairs=True,
        registration_query_disjoint_within_fold=True,calibration_final_disjoint=True,
        thresholds_shared_across_conditions=True,registration_count_equal=True,
        all_30_word_directions=True,two_repeat_folds=True,
        unique_final_recordings=len(fp),query_reuse='each physical query appears in five word-pair comparisons',
        source_hashes={p:digest(source/p) for p in ['encoder_private.pt','local_best_private.pt','audio_best_private.pt','normalization_private.npz']},
        limits='saved-score replay,not a second implementation of the scoring networks'))
    lines=['# 固定相同查询的同词／跨词认证对照','',
        '九人训练模型冻结；统计除U5外八名目标，保留U5攻击。K4注册＋K4查询，两组录制交换，共六词30个有向异词组合。每个组合的同词与跨词条件读取完全相同的查询和标签，只改变注册词。',
        '', '注册每词只有约10次重复，因此本轮K4，不与之前K12的绝对指标直接比较。固定Cal词21–23，共享同词/跨词阈值；最终六词24–29，不使用最终结果调参。',
        '', '|方法|条件|TAR%|FAR%|HTER%|EER%|AUC|','|---|---|---:|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f"|{r['method']}|{r['condition']}|{100*r['tar']:.2f}|{100*r['far']:.2f}|{100*r['hter']:.2f}|{100*r['eer']:.2f}|{r['auc']:.4f}|")
    lines+=['','## 查询词分解','', '|方法|查询词|条件|HTER%|TAR%|FAR%|','|---|---|---|---:|---:|---:|']
    for r in per_word:lines.append(f"|{r['method']}|{r['word']}|{r['condition']}|{100*r['hter']:.2f}|{100*r['tar']:.2f}|{100*r['far']:.2f}|")
    lines+=['','## 原两组三词交换方向','', '|方法|方向|条件|HTER%|TAR%|FAR%|','|---|---|---|---:|---:|---:|']
    for r in direction:lines.append(f"|{r['method']}|{r['direction']}|{r['condition']}|{100*r['hter']:.2f}|{100*r['tar']:.2f}|{100*r['far']:.2f}|")
    lines+=['','A=Down/Up/Zoom，B=Record/Play/Reset。A_to_B中跨词注册A、查询B，同词条件也查询相同B录制；B_to_A相反。词名是名义顺序映射，未独立转写核验。',
        '', '## 逐用户','', '|用户|方法|条件|HTER%|TAR%|FAR%|AUC|','|---|---|---|---:|---:|---:|---:|']
    for r in sorted(per_user,key=lambda r:(r['user'],r['method'],r['condition'])):
        lines.append(f"|{r['user']}|{r['method']}|{r['condition']}|{100*r['hter']:.2f}|{100*r['tar']:.2f}|{100*r['far']:.2f}|{r['auc']:.4f}|")
    lines+=['','主问题看原始ACC＋RBF。local/audio为已有冻结验证器的附加诊断；它们原先按K12训练，本轮K4没有重训。阈值从同等权重的Cal同词/跨词分数拟合，每目标/方法/折固定共用，不是每个最终词单独调阈值。',
        '', f'核查PASS：{len(rows)}工作点，保存分数重算误差0；注册/查询折内互斥、Cal与最终互斥。同一查询在五个异词组合中重复，组合不是独立试次；两折交换录制也不产生新会话。',
        '', '本轮控制的是固定查询条件下的注册内容变化，注册录制随条件改变属于该操作的一部分。没有新的Audio提取、生成模型训练或教师微调。原始波形、特征、分数、阈值及模型保持私有。']
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    dump(out/'status.json',dict(stage='complete'))
    print('MATCHED_QUERY_CONTROL_COMPLETE',json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--frontend',required=True);p.add_argument('--source',required=True);p.add_argument('--out',required=True)
    main(p.parse_args())
