"""Report all folds, including zero-Audio fallback; never select on final scores."""
import json
from pathlib import Path
from collections import Counter
import numpy as np

ROOT=Path('outputs');lines=['# 实验一P1：冻结编码器的双模态注册对照结果','', '对应最新方案P1 E0/E1/E2。E0仅HMD注册，E1仅注册Audio经映射建立HMD模板，E2融合两种模板；所有方法查询仅使用HMD。每人12次相同注册发声、完全相同的查询录制，每方法同词/跨词共享独立校准阈值。','', '此前蒸馏/残差不包含新用户注册Audio，不能替代本实验。本轮不训练大主干，只复用24个严格Base-only ACC检查点及两种冻结Audio编码器。','']
public={}
for teacher in ['ecapa','eres']:
    root=ROOT/f'p1_enrollment_{teacher}_20261108';rows=[json.loads(p.read_text()) for p in sorted(root.glob('target*_seed*/results_private.json'))]
    assert len(rows)==24,(teacher,len(rows))
    public[teacher]={'folds':24,'lambda_counts':dict(Counter(str(r['selection']['selected_lambda']) for r in rows)),'metrics':{},'paired_deltas':{},'per_user':{}}
    lines += ['## '+teacher,'','|攻击口径|内容|方法|TAR|FAR|HTER|AUC|','|---|---|---|---:|---:|---:|---:|']
    for scope in ['unseen_attacker','all_seven_sources']:
        public[teacher]['metrics'][scope]={}
        for cond in ['sameword','crossword']:
            public[teacher]['metrics'][scope][cond]={}
            for method in ['E0','E1','E2']:
                v=np.array([[np.mean([r['methods'][method][cond][scope][k] for r in rows if r['seed']==seed])*100 for k in ['tar','far','hter','auc']] for seed in [42,43,44]])
                public[teacher]['metrics'][scope][cond][method]={'mean':dict(zip(['tar','far','hter','auc'],v.mean(0).tolist())),'seed_std':dict(zip(['tar','far','hter','auc'],v.std(0,ddof=1).tolist())),'by_seed':dict(zip(['42','43','44'],v.tolist()))}
                lines+=['|'+scope+'|'+cond+'|'+method+'|'+'|'.join(f'{x:.2f}±{y:.2f}' for x,y in zip(v.mean(0),v.std(0,ddof=1)))+'|']
    lines+=['','lambda由Base内部开发选择，分布：'+json.dumps(public[teacher]['lambda_counts'],ensure_ascii=False)+'。lambda=0为合法回退，仍纳入统计。','', '|目标|E0跨词TAR|E0跨词FAR|E2跨词TAR|E2跨词FAR|E2-E0 HTER变化(pp)|','|---|---:|---:|---:|---:|---:|']
    for u in range(1,9):
        rr=[r for r in rows if r['target']==u];v={m:{k:float(np.mean([r['methods'][m]['crossword']['unseen_attacker'][k] for r in rr])*100) for k in ['tar','far','hter','auc']} for m in ['E0','E1','E2']};public[teacher]['per_user'][str(u)]=v
        lines+=['|'+['bo','bro','jin','ke','mei','qi','qis','xin'][u-1]+'|'+'|'.join(f'{x:.2f}' for x in [v['E0']['tar'],v['E0']['far'],v['E2']['tar'],v['E2']['far'],v['E2']['hter']-v['E0']['hter']])+'|']
    for seed in [42,43,44]:
        rr=[r for r in rows if r['seed']==seed];public[teacher]['paired_deltas'][str(seed)]={k:float(np.mean([r['methods']['E2']['crossword']['unseen_attacker'][k]-r['methods']['E0']['crossword']['unseen_attacker'][k] for r in rr])*100) for k in ['tar','far','hter','auc']}
    lines+=['','跨词严格未见攻击者E2-E0配对种子差值（百分点）：'+json.dumps(public[teacher]['paired_deltas'],ensure_ascii=False),'']
lines+=['## 路线与解释边界','','主干冻结 → Base训练轻量Audio→HMD映射 → Base内部开发锁定步数及融合lambda → 目标/Cal用同步注册对构建模板 → 各方法独立Cal阈值（同词/跨词共享）→ HMD-only最终查询。E0所有24折与旧原ACC结果完全复现，核对TAR/FAR/HTER/AUC及阈值最大差异0。','', '内层仅映射器留一Base身份，固定编码器曾见过这4名Base用户，不是完整内层未见身份重训。外层目标/攻击者未参与任何训练或开发选择。两条非精确配对fit样本仅从映射训练统一排除；注册池192条均有精确时间戳配对。词语编号仍未逐条语义核验，八人群体此前排除yue具有探索性局限。','', '均值及±标准差按先8目标宏平均、再3种子汇总；不是独立会话重复，不据小幅变化宣称显著收益。全部七来源含训练/校准已见身份，须与严格未见攻击者分开。E2融合有效也不能证明真实同步生理耦合，尚需C系列错配/错位控制。','', '验证评分函数只读取HMD查询和固定注册模板，不访问查询Audio、不运行Audio教师。每用户最终模板192维float32为768字节（不含协议和用户标识）；Audio及映射只用于注册。','', '[详细执行协议](2026-11-08-p1-enrollment-execution-protocol.md) · [原方案](2026-11-08-multimodal-enrollment-acc-only-verification-plan.md)']
Path('docs/experiments/2026-11-08-p1-enrollment-results.md').write_text('\n'.join(lines),encoding='utf-8');(ROOT/'p1_public_summary_20261108.json').write_text(json.dumps(public,indent=2))
for teacher,v in public.items():
    print(teacher,v['lambda_counts'])
    for method,m in v['metrics']['unseen_attacker']['crossword'].items():print(method,m['mean'],m['seed_std'])
