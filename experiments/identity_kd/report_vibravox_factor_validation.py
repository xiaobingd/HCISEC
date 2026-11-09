"""Verify complete public-data validation outputs and write aggregate-only report."""
import json
from pathlib import Path
import numpy as np
from run_vibravox_factor_validation import aggregate
root=Path('outputs/vibravox_factor_validation_20261109');ab=json.loads((root/'ablation_summary_public.json').read_text());ar=json.loads((root/'ablation_results_private.json').read_text());nr={str(i):json.loads((root/f'target{i}_neural_private.json').read_text()) for i in range(1,10)};ns=aggregate(nr);pre=json.loads((root/'preprocessing_public.json').read_text());d=dict(np.load(root/'inputs_private.npz'))
assert len(set(d['sample_id']))==len(d['sample_id'])==1040
sets=[set(d['text'][d['role']==k]) for k in ['fit','develop','registration','query']];assert all(not sets[i]&sets[j] for i in range(4) for j in range(i))
for rows in [ar,nr]:
    assert len(rows)==9
    for v in rows.values():
        f=v['fold'];assert len(set(f['base']+f['calibration']+[f['target']]+f['unseen_attackers']))==9
        for scopes in v['methods'].values():
            assert scopes['two_unseen_attackers']['genuine_n']==24 and scopes['two_unseen_attackers']['impostor_n']==48
            for e in scopes.values():np.testing.assert_allclose(e['hter'],(1-e['tar']+e['far'])/2)
for method,scopes in ab['results'].items():
    for scope,metrics in scopes.items():
        for key,value in metrics.items():np.testing.assert_allclose(value,np.mean([v['methods'][method][scope][key] for v in ar.values()]))
unique={v['fold']['owner']:v for v in nr.values()};assert len(unique)==8
train={m:{k:float(np.mean([v['training'][m][k] for v in unique.values()])) for k in ['fit_pair_auc','dev_pair_auc','parameters']} for m in ['stats_mlp','resnet18']}
summary=dict(state='complete',targets=9,unique_training_folds=8,seed=42,results=ns,training_means=train,protocol='1040uniqueVibraVoxforeheadutterances;true sentence groups disjoint;no matched same-content claim;4Base2Cal1target2unseen;12shot;24genuine48attacks pertarget;noAudio')
(root/'neural_summary_public.json').write_text(json.dumps(summary,indent=2))
text='# VibraVox复核：哪些本地结论得到支持\n\n九目标及8个唯一训练分组全部完成。现有speech_clean原始额头加速度缓存中按文本覆盖和固定哈希选择9人1040原始录音；不是按认证成绩选人。采集字段是实际单轴额头加速度，不是本地OpenVR三轴速度。\n\n'
text+='## 数据与协议\n\n1040个ACC缓存载荷哈希全部核对，无完全重复；源说话人/句子/Parquet行映射已单独复核。572拟合、144开发、108注册、216查询，真实规范化文本全局不跨角色。每人12注册、24查询。选中子集的文本均只对应一个用户，所以只能称跨句身份隔离，不能报告同内容攻击或严格排除用户间文本分布差异。\n\n'
text+='每折4Base/2Cal/1目标/2未见攻击者；仅Base-fit尺度/PCA/训练，Base-develop选checkpoint，Cal相互攻击选阈值。目标与未见攻击者从未参与这些步骤。单seed42，第九目标与第一目标共享训练分组。原型余弦后端相同，所有目标都保留。每目标24本人/48未见攻击，九目标共216本人决策、432攻击决策。\n\n'
text+='主前端为50–7600Hz、1×40×28对数功率；2秒中心窗、256msHann/64ms hop，所有架构同源。真实传感器具有宽频响应，不硬套HMD的300Hz。另有10–300Hz和50–1200Hz原统计对照。四路线为原统计240维、Base-fit PCA192、MLP240→256→192、单通道ResNet18/192；神经网络从头200步、相同CE+0.1SupCon/AdamW1e-4批次16，不使用Audio或预训练教师。\n\n'
text+='## 统计分量：两位未见攻击者\n\n全分块是平均形状40、波动40、相对时间160、幅度6，共246维；幅度含谱水平及真实波形5项。各块按Base拟合列标准化并除sqrt(维数)。全分块与原240维存在额外幅度统计与块权重变化，不能当作单一因素增益。\n\n|处理|TAR|FAR|HTER|AUC|\n|---|---:|---:|---:|---:|\n'
for method in ['original_statistics','all_factors','without_shape','without_std','without_time','without_amplitude','shape_only','std_only','time_only','amplitude_only','no_axis_level','low_band_statistics','mid_band_statistics']:
    v=ab['results'][method]['two_unseen_attackers'];text+='|'+method+'|'+ '|'.join(f'{100*v[k]:.2f}%' for k in ['tar','far','hter'])+f"|{v['auc']:.5f}|\n"
text+='\n## 均匀增益敏感性：固定注册和阈值\n\n|查询变换|TAR|FAR|HTER|AUC|\n|---|---:|---:|---:|---:|\n'
for name,v in [('原统计无扰动',ab['results']['original_statistics']['two_unseen_attackers']),*ab['simulated_gain_query_only'].items()]:text+='|'+name+'|'+ '|'.join(f'{100*v[k]:.2f}%' for k in ['tar','far','hter'])+f"|{v['auc']:.5f}|\n"
text+='\nshape_gain为去整体幅度的平均频谱形状，其无扰动指标与两项增益扰动指标相同。原统计增益后排序AUC变化较小但HTER变化较大，提示操作点迁移是重要组成，不能全解释为身份信息消失。这里只是线性增益，没有真实重戴或ADC饱和。\n\n'
text+='## 统一前端编码器对照\n\n|表示|TAR|FAR|HTER|AUC|\n|---|---:|---:|---:|---:|\n'
for method in ['original_statistics','statistics_pca192','stats_mlp','resnet18']:
    v=ns[method]['two_unseen_attackers'];text+='|'+method+'|'+ '|'.join(f'{100*v[k]:.2f}%' for k in ['tar','far','hter'])+f"|{v['auc']:.5f}|\n"
text+='\n## 全部八来源攻击（含已见身份，不能标为全未见）\n\n|表示|TAR|FAR|HTER|AUC|\n|---|---:|---:|---:|---:|\n'
for method in ['original_statistics','statistics_pca192','stats_mlp','resnet18']:
    v=ns[method]['all_eight_sources'];text+='|'+method+'|'+ '|'.join(f'{100*v[k]:.2f}%' for k in ['tar','far','hter'])+f"|{v['auc']:.5f}|\n"
text+='\n## Base拟合/开发诊断\n\n|模型|编码器参数量|拟合配对AUC|开发配对AUC|\n|---|---:|---:|---:|\n'
for m,v in train.items():text+=f"|{m}|{int(v['parameters']):,}|{v['fit_pair_auc']:.5f}|{v['dev_pair_auc']:.5f}|\n"
text+='\n开发仍是同一批Base用户的不同句子，不是未见身份。神经模型的参数量不同，200更新也不是各自最优训练预算的证明；本轮只能评价四位Base从头监督训练，不代表大规模预训练声纹模型都会弱于统计。\n\n'
text+='## 对本地结论的支持边界\n\n平均频谱形状的重要性复现：删除形状AUC下降；幅度不是必要身份线索，去掉幅度可提高排序。相对时间块单独较弱的现象也复现；波动块在VibraVox比本地更有可分性，不能将本地相对贡献比例直接推广。\n\n均匀增益下，原统计操作点敏感、去幅度形状稳定，支持将幅度与形状分开。但没有重戴/跨设备采集标签，不能因此证明真实佩戴鲁棒。宽频与10–300Hz的差距说明需按传感器定频段。\n\n本地的日期/采样条件混杂无法由本数据直接验证或否定，现有缓存没有相同的采集时间戳/重戴标签；真正的额头加速度数据复现某些统计规律也不能证明本地vVelocity的生理来源。所有结论是探索性数据集复核，不是盲测试或完整VibraVox benchmark。\n\n'
text+='官方VibraVox论文另使用预训练ECAPA2做说话人验证；那不是本轮四Base从头ResNet/MLP协议，不能将两者混为一谈。公开只保存代码、协议、聚合数据；原音频、逐身份/文本关系、阈值和逐目标结果保持私有。\n\n来源：https://huggingface.co/datasets/Cnam-LMSSC/vibravox ；Hauret等，Vibravox，Speech Communication，2025：https://arxiv.org/html/2407.11828v2 。\n'
Path('docs/experiments/2026-11-09-vibravox-factor-validation-results.md').write_text(text,encoding='utf-8');print('PASS ninefold metrics and true-text isolation; aggregate report written');print(json.dumps(summary,indent=2))
