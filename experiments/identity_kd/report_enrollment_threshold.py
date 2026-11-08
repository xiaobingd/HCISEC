"""Aggregate registration-conditioned threshold policies; no test retuning."""
import json
from pathlib import Path
from collections import Counter
import numpy as np
public={};lines=['# 注册条件个体化阈值对照结果','', '采用图片建议的任务边界：Base模拟注册学习g，独立Cal选验证策略，目标仅使用合法注册Audio/HMD生成阈值，查询HMD-only。ECAPA/ERes2NetV2分别24折。固定E0/E2模板和编码器，对照阈值策略。','', '## 主要结论','', '当前低维Ridge版g没有稳定改善。ECAPA E2固定阈值TAR75.35%、FAR20.31%、HTER22.48%；个体化阈值为74.65%、20.66%、23.00%。ERes E2个体化仅小幅降低HTER（23.87%→23.70%），仍劣于原E0的23.09%，且FAR上升。','', 'Cal约束可使校准集FAR不恶化，但不能保证外层未见攻击FAR。所有折均未采用背景下界候选；不能宣称背景下界已有效。多数折回退到全局阈值，反映该g/校准规则尚未获得可靠个体化策略。此结论不否定所有注册条件校准方案。','']
for teacher in ['ecapa','eres']:
    rows=list(json.loads(Path(f'outputs/enrollment_threshold_{teacher}_20261108/results_private.json').read_text()).values());assert len(rows)==24
    public[teacher]=dict(folds=24,metrics={},policy_counts={})
    lines+=['## '+teacher,'','|攻击口径|内容|方法|TAR|FAR|HTER|原余弦AUC|校准margin AUC|','|---|---|---|---:|---:|---:|---:|---:|']
    for scope in ['unseen_attacker','all_seven_sources']:
        public[teacher]['metrics'][scope]={}
        for cond in ['sameword','crossword']:
            public[teacher]['metrics'][scope][cond]={}
            for method in rows[0]['methods']:
                keys=['tar','far','hter','raw_auc','margin_auc'];v=np.array([[np.mean([r['methods'][method][cond][scope][k] for r in rows if r['seed']==seed])*100 for k in keys] for seed in [42,43,44]])
                public[teacher]['metrics'][scope][cond][method]=dict(mean=dict(zip(keys,v.mean(0).tolist())),seed_std=dict(zip(keys,v.std(0,ddof=1).tolist())))
                lines+=['|'+scope+'|'+cond+'|'+method+'|'+'|'.join(f'{x:.2f}±{y:.2f}' for x,y in zip(v.mean(0),v.std(0,ddof=1)))+'|']
    for m in ['E0','E2']:
        counts=Counter(f"eta={r['cal_policy'][m]['adaptive_guarded']['eta']},floor={r['cal_policy'][m]['adaptive_guarded']['background_floor']}" for r in rows);public[teacher]['policy_counts'][m]=dict(counts);lines+=['',m+'策略分布：'+json.dumps(dict(counts),ensure_ascii=False)]
        for r in rows:
            for pol in ['adaptive','adaptive_guarded']:assert r['cal_policy'][m][pol]['cal_far']<=r['cal_policy'][m][pol]['baseline_cal_far']+1e-12
    lines+=['','|目标|E2固定TAR|E2固定FAR|E2个体化TAR|E2个体化FAR|E2个体化减固定HTER(pp)|','|---|---:|---:|---:|---:|---:|']
    peruser={}
    for u in range(1,9):
        rr=[r for r in rows if r['target']==u];v={m:{k:float(np.mean([r['methods'][m]['crossword']['unseen_attacker'][k] for r in rr])*100) for k in ['tar','far','hter']} for m in ['E2_fixed','E2_adaptive_guarded']};peruser[str(u)]=v;a=v['E2_fixed'];b=v['E2_adaptive_guarded'];lines+=['|'+['bo','bro','jin','ke','mei','qi','qis','xin'][u-1]+'|'+'|'.join(f'{x:.2f}' for x in [a['tar'],a['far'],b['tar'],b['far'],b['hter']-a['hter']])+'|']
    public[teacher]['per_user']=peruser
lines+=['','## 解释边界','', '模板未变，原余弦AUC不变；margin AUC变化只说明注册条件阈值导致的方向间分数校准变化，不是多提取了身份信息。固定分支逐折复现P1的TAR/FAR/HTER。','', 'g使用6维注册embedding一致性及3维相对Base背景原型统计，Ridge正则20，非高维任意阈值网络。注册端预测相对Cal注册预测中位数的偏移限制±0.1，Cal从0/.25/.5/1选收缩幅度；背景下界为可选候选。阈值策略针对Cal EER工作点，不是经过验证的低FAR安全保证。尚未新增正式分数归一化对照。','', '当前只有4位Base身份；96个模拟注册任务不等于96名独立用户。内层映射器留人而固定编码器已见Base、内外背景人数3→4、目标与Cal注册质量分布偏移等均限制泛化。词标签未逐条语义核验，八人群体此前排除yue属探索性选择。不能将该版g的失败扩大为所有个体化校准不可行。','', '[执行协议](2026-11-08-enrollment-threshold-protocol.md) · [E3与阈值迁移诊断](2026-11-08-e3-quality-gating-results.md)']
Path('docs/experiments/2026-11-08-enrollment-threshold-results.md').write_text('\n'.join(lines),encoding='utf-8');Path('outputs/enrollment_threshold_public_summary_20261108.json').write_text(json.dumps(public,indent=2));print('48fold report complete; all Cal FAR constraints verified')
