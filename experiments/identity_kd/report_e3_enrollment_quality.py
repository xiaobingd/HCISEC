"""Summarize quality-gated enrollment and separate threshold diagnostics."""
import json
from pathlib import Path
import numpy as np

public={};lines=['# E3注册质量自适应与阈值迁移诊断','', '两种冻结Audio模型各8目标×3训练种子，均完成24折。E0/E2复用P1；E3仅学习注册模板的融合权重，不用目标测试样本调整阈值。主干及P1 Audio→HMD映射不更新。','', '当前协议不是按最终目标测试分数选阈值。每折4位Base、2位Cal、目标、未见攻击者互斥；每方法阈值由Cal确定，同词/跨词共享。不同外层折的编码器、校准身份与阈值不同，所以不能把跨折分数尺度差异直接当成用户生理差异。','']
for teacher in ['ecapa','eres']:
    root=Path(f'outputs/e3_enrollment_{teacher}_20261108');rows=[json.loads(p.read_text()) for p in sorted(root.glob('target*_seed*/results_private.json'))];assert len(rows)==24
    public[teacher]={'folds':24,'metrics':{},'threshold_diagnosis':{}}
    lines+=['## '+teacher,'','|攻击口径|内容|方法|TAR|FAR|HTER|AUC|','|---|---|---|---:|---:|---:|---:|']
    for scope in ['unseen_attacker','all_seven_sources']:
        public[teacher]['metrics'][scope]={}
        for cond in ['sameword','crossword']:
            public[teacher]['metrics'][scope][cond]={}
            for method in ['E0','E2','E3','HMD_quality','random_quality']:
                v=np.array([[np.mean([r['methods'][method][cond][scope][k] for r in rows if r['seed']==s])*100 for k in ['tar','far','hter','auc']] for s in [42,43,44]])
                public[teacher]['metrics'][scope][cond][method]=dict(mean=dict(zip(['tar','far','hter','auc'],v.mean(0).tolist())),seed_std=dict(zip(['tar','far','hter','auc'],v.std(0,ddof=1).tolist())))
                lines+=['|'+scope+'|'+cond+'|'+method+'|'+'|'.join(f'{x:.2f}±{y:.2f}' for x,y in zip(v.mean(0),v.std(0,ddof=1)))+'|']
    lines+=['','### E0阈值诊断（各目标三个种子的均值）','','|目标|校准阈值|Cal本人分数中位数|目标本人中位数|目标攻击者中位数|固定阈值HTER|回看测试EER工作点HTER|','|---|---:|---:|---:|---:|---:|---:|']
    for u in range(1,9):
        rr=[r for r in rows if r['target']==u];keys=['cal_threshold','cal_genuine_median','target_genuine_median','target_impostor_median','frozen_threshold_hter','retrospective_test_hter'];v={k:float(np.mean([r['threshold_diagnosis']['E0'][k] for r in rr])) for k in keys};public[teacher]['threshold_diagnosis'][str(u)]=v
        lines+=['|'+['bo','bro','jin','ke','mei','qi','qis','xin'][u-1]+'|'+'|'.join(f'{v[k]:.3f}' for k in keys[:4])+'|'+f"{v[keys[4]]*100:.2f}%|{v[keys[5]]*100:.2f}%|"]
lines+=['','## 方法与限制','','每折只在4位Base身份上产生96个注册任务：每身份24个支持集，每集3词×4次发声=12条，查询为Base独立develop HMD。映射器按留一Base身份训练，编码器已见全部Base，与P1相同限制。注册质量只由支持集embedding估计：Audio/HMD平均一致性、跨词一致性、两个原型一致性、逐对映射一致性；不是原始传感器SNR或生理耦合测量。','', '用Base develop查询得到最佳lambda训练标签（并列取更小lambda），StandardScaler+Ridge(alpha10)拟合6维注册质量到lambda，再裁剪[0,1]。结构、正则与规则固定，不根据目标结果修改。HMD_quality只用2维HMD质量；random_quality从Base门控输出分布抽权重，不读取目标查询，不保证与外层E3权重分布完全相同。','', '外层仅Cal及目标的12条注册Audio/HMD构模板，验证评分只读HMD查询；每方法独立Cal阈值，同词/跨词共享。E3个性化的是融合权重，不是目标测试阈值。所有E0/E2的TAR/FAR/HTER/AUC/阈值逐折复现P1（容差1e-6）。','', '阈值诊断中的回看测试EER工作点使用最终标签，仅作事后诊断，未用于门控、阈值校准、模型选择或部署结果；不是最低可能HTER，也不应当作可实现的提升。qi呈明显校准分数偏移；bro本人/攻击者分布接近，单调阈值不能修复身份可分性不足。这两类原因并存。','', '只有4位Base身份，门控容易过拟合；内层不等于完整编码器未见用户验证。八人群体排除yue具有探索性选择，词标签未逐条语义核验。现结果不能宣称同步耦合有效。均值±标准差先8目标宏平均再3种子汇总，不是独立会话不确定性。','', '[P1结果与路线](2026-11-08-p1-enrollment-results.md)']
Path('docs/experiments/2026-11-08-e3-quality-gating-results.md').write_text('\n'.join(lines),encoding='utf-8');Path('outputs/e3_public_summary_20261108.json').write_text(json.dumps(public,indent=2))
for t,x in public.items():
    print(t)
    for m,v in x['metrics']['unseen_attacker']['crossword'].items():print(m,v['mean'])
