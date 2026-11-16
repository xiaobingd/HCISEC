"""Remove U5 final attack queries only; retain every trained model and threshold."""
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score,roc_curve

SOURCE=Path('outputs/mei_cycle_matched_content_20261116')
OUT=Path('outputs/mei_cycle_matched_without_u5_attacks_20261116')
OUT.mkdir(exist_ok=True,parents=True)
scores=np.load(SOURCE/'scores_private.npz')
thresholds=json.loads((SOURCE/'thresholds_private.json').read_text())
prior=json.loads((SOURCE/'per_user_public.json').read_text())
targets=[1,2,3,4,6,7,8,9]
methods=['statistics720','baseline312','selected_input256','cycle256_fused','cycle256_only']
query_user=np.repeat(np.arange(1,10),4)
keep=query_user!=5

def metric(y,s,t):
    far=float((s[~y]>=t).mean());frr=float((s[y]<t).mean())
    fp,tp,_=roc_curve(y,s,drop_intermediate=False);j=np.argmin(abs(fp-(1-tp)))
    return dict(tar=1-frr,far=far,frr=frr,hter=(far+frr)/2,
                auc=float(roc_auc_score(y,s)),eer=float((fp[j]+1-tp[j])/2))

rows=[];max_error=0.;workpoints=0
for u in targets:
    for method in methods:
        for condition in ['sameword','crossword']:
            old=[];new=[]
            tau=thresholds[f'{u}_{method}']
            for fold in [0,1]:
                for qw in range(24,30):
                    for rw in range(24,30):
                        if rw==qw:continue
                        s=scores[f'{u}_{method}_{fold}_{qw}_{rw}_{condition}']
                        assert len(s)==36
                        y=query_user==u
                        assert y[keep].sum()==4 and (~y[keep]).sum()==28
                        old.append(metric(y,s,tau));new.append(metric(y[keep],s[keep],tau));workpoints+=1
            before=next(r for r in prior if r['user']==f'U{u}' and r['method']==method and r['condition']==condition)
            for k in ['tar','far','frr','hter','auc']:
                err=abs(np.mean([r[k] for r in old])-before[k]);max_error=max(max_error,float(err));assert err<1e-12
            values={k:float(np.mean([r[k] for r in new])) for k in new[0]}
            assert abs(values['tar']-before['tar'])<1e-12
            rows.append(dict(user=f'U{u}',method=method,condition=condition,**values))
summary=[]
for method in methods:
    for condition in ['sameword','crossword']:
        rr=[r for r in rows if r['method']==method and r['condition']==condition]
        summary.append(dict(method=method,condition=condition,
                     **{k:float(np.mean([r[k] for r in rr])) for k in ['tar','far','frr','hter','auc','eer']}))
def save(name,value):(OUT/name).write_text(json.dumps(value,indent=2),encoding='utf-8')
save('results_public.json',dict(scope='U5 excluded as target and final attacker only;all nine training and original calibration retained',summary=summary))
save('per_user_public.json',rows)
save('verification_public.json',dict(state='PASS',recomputed_workpoints=workpoints,
    prior_score_aggregation_error=max_error,genuine_acceptance_unchanged=True,
    each_pair_genuine=4,each_pair_impostor=28,excluded_user=5,
    thresholds_models_and_registration_unchanged=True,
    limits='posthoc attack-subset recomputation,not retraining or a new blind test'))
lines=['# 第一创新点：U5不作为目标或最终攻击者','',
    '在固定查询K4对照上，仅从最终查询中移除U5攻击，并排除U5目标。保留U1/U2/U3/U4/U6/U7/U8/U9。原九人生成器/学生训练、RBF负类注册、Cal阈值和融合权重全部不变。每个词对/折/目标剩4条本人、七个攻击来源各4条，共28条攻击。',
    '', '|方法|条件|TAR%|FAR%|HTER%|EER%|AUC|','|---|---|---:|---:|---:|---:|---:|']
for r in summary:lines.append(f"|{r['method']}|{r['condition']}|{100*r['tar']:.2f}|{100*r['far']:.2f}|{100*r['hter']:.2f}|{100*r['eer']:.2f}|{r['auc']:.4f}|")
lines+=['','## 逐用户','', '|用户|方法|条件|TAR%|FAR%|HTER%|EER%|AUC|','|---|---|---|---:|---:|---:|---:|---:|']
for r in sorted(rows,key=lambda r:(r['user'],r['method'],r['condition'])):lines.append(f"|{r['user']}|{r['method']}|{r['condition']}|{100*r['tar']:.2f}|{100*r['far']:.2f}|{100*r['hter']:.2f}|{100*r['eer']:.2f}|{r['auc']:.4f}|")
lines+=['',f'保存分数重算{workpoints}个工作点，移除前重算与源逐用户指标最大误差{max_error:.3g}，移除攻击不改变任何本人接受率。',
    '', '这是已观察结果后的限定攻击来源统计，不表示模型在没有U5数据时训练，也没有重新校准更有利的阈值。词标签名义映射、单会话、K4、两折及词对重复使用查询等范围沿用源实验。selected_input256仍为原定版；cycle256_fused为诊断分支，不因当前成绩重选。',
    '', '源报告：https://github.com/xiaobingd/HCISEC/blob/main/outputs/mei_cycle_matched_content_20261116/report.md']
(OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(summary,indent=2))
