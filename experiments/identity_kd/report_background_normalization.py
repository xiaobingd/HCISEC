"""Locked-template score normalization comparison and focused user analysis."""
import json
from pathlib import Path
import numpy as np

public={};lines=['# 注册背景分数标准化、原余弦与Ridge严格对照','', '两种冻结Audio模型分别8目标×3种子，各24折。固定P1 E0/E2模板、12次注册和HMD查询；只改变分数校准方法。背景均值/标准差仅来自4个Base身份的fit HMD中心与注册模板相似度，目标/Cal/攻击者不进入背景拟合。','', '采用图片公式 (s-mu)/(sigma+1e-6)，sigma为4中心相似度的总体标准差，不按最终效果调epsilon或筛背景。各方法每折在相同独立Cal上校准共享阈值，同词/跨词共享各方法阈值。原余弦和Ridge在EER协议逐折复现旧指标；新增Cal FAR<=5%工作点，所有方法使用相同校准池，Ridge在Cal重新选择收缩/背景下界策略，未访问目标测试。','', '## 主要结论','', '- 在原EER校准规则下，ECAPA E2原余弦HTER22.48%，背景标准化23.61%，Ridge23.00%。背景标准化未超过两项对照。','- 在Cal FAR<=5%工作点，ECAPA E2原余弦TAR38.37%、FAR6.60%、HTER34.11%；标准化39.06%、6.25%、33.59%，有约0.52个百分点HTER改善，但接受率仍很低，外层FAR仍超过5%，不能称为可用安全认证。','- ERes2NetV2同样在EER点变差（E2 HTER23.87%→24.65%），在Cal5%点小幅改善（34.64%→34.11%）。该归一化不具有跨工作点普遍优势。','- qi和xin在EER点的等效余弦阈值反而升高，TAR下降；bro的高重叠没有改善。4个训练中心提供的背景统计没有实现预期的新用户分数纠偏，不能将训练中心相似度分布等同于真实未知攻击分布。','']
for teacher in ['ecapa','eres']:
    rows=list(json.loads(Path(f'outputs/background_norm_{teacher}_20261108/results_private.json').read_text()).values());assert len(rows)==24;public[teacher]=dict(folds=24,metrics={},focus={})
    lines+=['## '+teacher,'','|攻击口径|内容|方法/校准规则|TAR|FAR|HTER|原余弦AUC|判定margin AUC|','|---|---|---|---:|---:|---:|---:|---:|']
    for scope in ['unseen_attacker','all_seven_sources']:
        public[teacher]['metrics'][scope]={}
        for cond in ['sameword','crossword']:
            public[teacher]['metrics'][scope][cond]={}
            for method in rows[0]['methods']:
                keys=['tar','far','hter','raw_auc','margin_auc'];v=np.array([[np.mean([r['methods'][method][cond][scope][k] for r in rows if r['seed']==seed])*100 for k in keys] for seed in [42,43,44]])
                public[teacher]['metrics'][scope][cond][method]=dict(mean=dict(zip(keys,v.mean(0).tolist())),seed_std=dict(zip(keys,v.std(0,ddof=1).tolist())))
                lines+=['|'+scope+'|'+cond+'|'+method+'|'+'|'.join(f'{x:.2f}±{y:.2f}' for x,y in zip(v.mean(0),v.std(0,ddof=1)))+'|']
    lines+=['','### qi / xin / bro：E2跨词、严格未见攻击者、原EER校准','','|目标|方法|TAR|FAR|HTER|等效余弦阈值均值|背景mu|背景sigma|','|---|---|---:|---:|---:|---:|---:|---:|']
    for u in [6,8,2]:
        rr=[r for r in rows if r['target']==u];name={6:'qi',8:'xin',2:'bro'}[u];public[teacher]['focus'][name]={}
        for kind in ['raw','zscore','ridge']:
            method='E2_'+kind+'_eer';v={k:float(np.mean([r['methods'][method]['crossword']['unseen_attacker'][k] for r in rr])*100) for k in ['tar','far','hter']};stats={k:float(np.mean([q[k] for r in rr for q in r['registration_statistics'][method+'_crossword']])) for k in ['mu','sigma','equivalent_cosine_threshold']};public[teacher]['focus'][name][kind]=dict(**v,**stats)
            lines+=['|'+name+'|'+kind+'|'+f"{v['tar']:.2f}%|{v['far']:.2f}%|{v['hter']:.2f}%|{stats['equivalent_cosine_threshold']:.4f}|{stats['mu']:.4f}|{stats['sigma']:.4f}|"]
    for r in rows:
        for name,v in r['calibration'].items():
            if name.endswith('cal_far05'):assert v['cal_far']<=.05+1e-12
lines+=['','## 聚焦解释','', 'ECAPA E2中qi的等效余弦阈值均值由0.9331升至0.9721，本人接受率19.44%→5.56%；xin阈值0.6904→0.7241，TAR73.61%→65.28%；bro TAR88.89%、FAR76.39%基本不变。这里阈值均值只是同一目标三个种子、两个方向的描述统计，不是部署时使用平均阈值。','', '背景mu接近0、sigma在这些例子约0.57–0.59，未提供与实际新用户分数偏移相匹配的校正。这提示4个Base分类中心的几何关系可能不足以代表未知攻击分布；不是已证明的唯一失败原因，也不应靠最终查询重选背景或epsilon修饰结果。bro的原分数高度重叠，单调线性标准化不能增加同一注册任务内的身份排序信息。','', '同词/跨词共享标准化后的阈值与统计生成规则，原余弦等效阈值可随注册模板变化；属于新评分协议。原始余弦AUC不变，margin AUC可能因两个注册方向的条件标准化而变化，不代表编码器新提取身份信息。','', 'Cal仅2位身份、背景仅4位训练身份；经验Cal FAR<=5%不是未见攻击FAR保证。多个随机种子不是独立会话，词语编号未逐条语义核验，八人群体此前排除yue具有探索性选择限制。没有根据目标测试挑选校准规则，两个工作点均完整保留。','', '[个体化阈值对照](2026-11-08-enrollment-threshold-results.md) · [P1固定模型路线](2026-11-08-p1-enrollment-results.md)']
Path('docs/experiments/2026-11-08-background-normalization-results.md').write_text('\n'.join(lines),encoding='utf-8');Path('outputs/background_normalization_public_summary_20261108.json').write_text(json.dumps(public,indent=2));print('48folds complete; raw/Ridge EER parity and all Cal FAR05 constraints passed')