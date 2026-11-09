"""Merge completed owner outputs, verify aggregates, write scientific summary."""
import json
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0,'experiments/identity_kd')
import run_mei9_controlled_identity as r
root=Path('outputs/mei9_controlled_identity_20261109');rows={str(i):json.loads((root/f'target{i}_neural_private.json').read_text()) for i in range(1,10)}
for target,v in rows.items():
    f=v['fold'];assert len(set(f['base']+f['calibration']+[f['target']]+f['unseen_attackers']))==9 and f['target']==int(target)
    for method,conditions in v['methods'].items():
        for cond,scopes in conditions.items():
            e=scopes['two_unseen_attackers'];assert e['genuine_n']==24 and e['impostor_n']==48;np.testing.assert_allclose(e['hter'],(1-e['tar']+e['far'])/2)
summary=r.macro(rows);unique={v['fold']['owner']:v for v in rows.values()};assert len(unique)==8
training={method:{k:float(np.mean([v['training'][method][k] for v in unique.values()])) for k in ['parameters','fit_pair_auc','dev_pair_auc','seconds']} for method in ['stats_mlp','resnet18']}
probes={method:{group:float(np.median([v['probe_target_query_r2'][method][group] for v in rows.values()])) for group in ['shape','std','time','amplitude']} for method in ['original_statistics','statistics_pca192','stats_mlp','resnet18']}
obj=dict(state='complete',completed_targets=9,unique_training_folds=8,results=summary,training_means=training,target_probe_median_r2=probes)
(root/'neural_summary_public.json').write_text(json.dumps(obj,indent=2))
text='# 九人统一前端：原始统计、PCA、统计MLP与ResNet18\n\n全部九目标已完成，8个独立训练分组、每分组两种神经编码器各200步、seed42。第九目标与第一目标共享同一Base/Cal训练分组，因此不是九个独立模型。\n\n'
text+='## 完整控制条件\n\n2695条原始HMD速度记录，统一3轴×40频点×28帧前端（1000Hz插值、2秒、256Hann/64hop/1024FFT、10–300Hz），没有复用旧80×80前端权重。每折4Base/2Cal/1未见目标/2未见攻击者；目标/攻击者不参与尺度、训练、开发或Cal。神经网络同200步、相同批次抽样与CE+0.1SupCon、AdamW1e-4，仅Base开发词选checkpoint。\n\n'
text+='原统计720维；PCA192仅Base-fit；MLP720→256→192，GELU；标准ResNet18直接输入三轴时频，192输出。模型最后均L2归一化。所有方法共同12-shot原型余弦，Cal互相攻击确定同词/跨词共享阈值；24本人/48未见攻击每目标，所有原始查询相同。\n\n'
for cond,label in [('sameword','同词划分假设'),('crossword','跨词划分假设')]:
    text+='## '+label+'：两位未见攻击者\n\n|表示|TAR|FAR|HTER|AUC|\n|---|---:|---:|---:|---:|\n'
    for method in ['original_statistics','statistics_pca192','stats_mlp','resnet18']:
        v=summary[cond][method]['two_unseen_attackers'];text+='|'+method+'|'+ '|'.join(f'{100*v[k]:.2f}%' for k in ['tar','far','hter'])+f"|{v['auc']:.5f}|\n"
text+='\n## 含已见身份的全部八来源攻击\n\n|表示|同词TAR|同词FAR|同词HTER|跨词TAR|跨词FAR|跨词HTER|\n|---|---:|---:|---:|---:|---:|---:|\n'
for method in ['original_statistics','statistics_pca192','stats_mlp','resnet18']:
    text+='|'+method+'|'+ '|'.join(f"{100*summary[c][method]['all_eight_sources'][k]:.2f}%" for c in ['sameword','crossword'] for k in ['tar','far','hter'])+'|\n'
text+='\n这些攻击包含Base/Cal，不能标为全部未见攻击。\n\n## 训练—开发诊断\n\n下表为8个训练分组平均，开发录音仍来自相同Base用户的留出词，并非未见身份。\n\n|编码器|参数量|训练配对AUC|开发配对AUC|\n|---|---:|---:|---:|\n'
for method,v in training.items():text+=f"|{method}|{int(v['parameters']):,}|{v['fit_pair_auc']:.5f}|{v['dev_pair_auc']:.5f}|\n"
text+='\n参数量不同、相同更新数也不等于各网络均已充分调优。ResNet标准stem与四阶段将40×28缩到最后2×1，再全局平均，存在强空间/时间压缩；但卷积通道也可能保存动态，不能仅凭尺寸声称时间身份信息全部丢失。需要额外stem/归一化干预才能定位具体结构原因。\n\n'
text+='## 统计线性读出探针\n\n固定Ridge alpha10只在Base-fit拟合，预测未见目标查询的统计分量，下表为九目标R²中位数。\n\n|表示|平均形状|波动|相对时间|幅度|\n|---|---:|---:|---:|---:|\n'
for method,v in probes.items():text+='|'+method+'|'+ '|'.join(f'{v[k]:.4f}' for k in ['shape','std','time','amplitude'])+'|\n'
text+='\n负R²表示跨身份线性读出迁移失败，不是信息完全不存在。即使原统计参照也可能出现负值：表示的L2归一化及未见身份分布偏移使线性反演困难。因此不能仅用这张表定论深度编码器删除了某项生理特征；认证与训练/开发差距是主要证据。\n\n'
text+='## 结论边界\n\n本轮独立比较了共同信号前端下的统计表示、固定压缩和监督编码，不引入Audio或生成任务。结果只能说明这些表示在当前协议和预算下保留了多少可用于认证的用户相关差异。结合统计消融，原差异中混有幅度、佩戴/录制日期与采样条件，原统计优于神经网络并不自动证明深度网络丢失的全部是稳定生理身份。\n\n'
text+='用户确认同一台头显且无独立重戴/跨天/换设备重复，词标签仍是采集顺序假设。单seed、已探索群体、相同设备下的个人佩戴条件混杂依然存在。不同训练折分配到本地/服务器，同折所有路线在同一环境；环境版本差异可能造成微小数值变化。本表没有按最终目标调参数或选择用户。\n\n'
text+='全部原始信号、具体身份/日期关系、逐目标阈值和成绩保持私有；公开代码、协议、宏平均。来源消融详见同日mei9-statistics-factor-results。\n'
Path('docs/experiments/2026-11-09-mei9-controlled-encoder-results.md').write_text(text,encoding='utf-8')
print('PASS9targets8traininggroups; merged summaries and report written');print(json.dumps(obj,indent=2))
