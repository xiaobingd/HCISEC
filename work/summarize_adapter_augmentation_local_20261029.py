import json
from pathlib import Path
root=Path('outputs/mei_adapter_augmentation_20261029');d=json.loads((root/'verified_summary_public.json').read_text())
assert d['status']=='PASS' and d['targets']==3
aug_names={'none':'无增强','noise':'轻度频谱噪声','time_shift':'轻度时间平移','noise_shift':'噪声加平移'}
mode_names={'combined':'跨词及注册查询联合','episode_only':'仅注册查询'}
lines=['# 认证训练增强三目标三seed结果','',
'在冻结个人G/F上，仅对认证训练ACC频谱增加轻度噪声、时间平移或组合，重新提取772维并训练相同64维适配器。U1/U4/U5均保留全部八个攻击身份。标准化、验证、校准、测试与注册中心均使用干净输入。','',
'训练每身份50条，未见词验证30条，校准20条，最终本人47至50条。每组400步、batch450，seed42/43/44；每步每条录制50%干净、25%各增强视图，共两个固定视图。无增强对照复用前一轮相同配置的原模型/分数。','',
'频谱噪声标准差为450条训练ACC逐条标准差中位数的2%，按训练min/max裁剪；不是物理时域噪声仿真。时间平移±1/2帧，沿时间轴用原边缘填充、不循环。两种目标的全部条件采用相同初始化和独立的增强RNG，保持episode词与录制抽样配对。','',
'## 三目标平均','', '|训练目标|增强|TAR|FAR|AUC|HTER|','|---|---|---:|---:|---:|---:|']
for mode,title in mode_names.items():
 for aug,name in aug_names.items():
  r=d['macro'][mode+'__'+aug];lines.append('|%s|%s|%.2f%%|%.2f%%|%.4f|%.2f%%|'%(title,name,r['tar']*100,r['far']*100,r['auc'],r['hter']*100))
lines+=['','## 每目标三seed均值及样本标准差','', '|目标|训练目标|增强|TAR 均值±SD|FAR 均值±SD|AUC 均值±SD|HTER均值|','|---|---|---|---:|---:|---:|---:|']
for u in ['1','4','5']:
 for mode,title in mode_names.items():
  for aug,name in aug_names.items():
   r=d['results'][u]['seed_summary'][mode+'__'+aug]
   lines.append('|U%s|%s|%s|%.2f±%.2f%%|%.2f±%.2f%%|%.4f±%.4f|%.2f%%|'%(u,title,name,r['tar']['mean']*100,r['tar']['std']*100,r['far']['mean']*100,r['far']['std']*100,r['auc']['mean'],r['auc']['std'],r['hter']['mean']*100))
lines+=['','## 阶段判断','',
'联合版的轻度时间平移使宏FAR从17.99%降至15.66%，HTER从19.85%降至19.02%，AUC从0.8742升至0.8781，本人接受率从78.29%略降至77.63%。是小幅、存在权衡的收益，不是显著性结论。',
'','U4联合版本人接受率维持70.21%，FAR23.18%降至19.71%，AUC0.8281升至0.8455；U5本人接受率不变、FAR下降，但AUC略降；U1本人接受率与FAR同时下降，HTER反而稍差。不能只报告FAR更低。',
'','仅注册查询版时间平移的HTER19.01%到18.99%，几乎持平；噪声和组合没有一致收益。当前只支持保留轻度时间平移继续验证，不支持训练增强普遍有效。相对匹配RBF的HTER17.07%，联合版加平移仍未超过其综合性能；两者读出方式不同。',
'','## 验证与边界','',
'54个新增强模型与18个原始控制共72条路线均已核验。从原训练ACC重新生成抽样增强，经同一冻结G/F与保存的增强特征银行比较；核对行ID哈希、训练/开发/测试无交集、干净Scaler与控制完全一致、检查点选模、权重重提向量、校准阈值、逐来源和逐词AUC及接受数。平移边缘另有确定性小矩阵检查。',
'','所有增强只发生在auth_train。支持和查询仍是不同词的独立录制，增强视图不算新录制。没有根据最终成绩选seed、变增强强度或删用户/攻击者。',
'','结果说明当前轻度频谱扰动和固定两视图方案的作用，不代表所有增强方法都有效或无效。没有噪声测试集，不能据此宣称物理传感器抗噪能力。历史最终数据已被观察，内容映射仍依赖10/20周期和尾部缺行假设；单会话、已知身份闭集。',
'','三seed标准差只反映初始化与episode抽样，不是跨用户总体置信区间。增强增加的是视图多样性，不增加用户、词语或真实独立录制数量。']
Path('docs/experiments/2026-10-29-adapter-augmentation-results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
compact={k:d[k] for k in ['status','targets','routes_verified','macro','augmentation_checks','verification']};compact['results']={}
keep=['tar','far','auc','hter','same_word_mean_auc','genuine_n','impostor_n','genuine_accepted','attack_accepted','mode','augmentation','seed','selected_step']
for u,a in d['results'].items():compact['results'][u]=dict(seed_summary=a['seed_summary'],training_feature_median_cosine_drift=a['training_feature_median_cosine_drift'],routes={key:{k:r[k] for k in keep if k in r} for key,r in a['routes'].items()})
(root/'summary_public.json').write_text(json.dumps(compact,indent=2),encoding='utf-8');print('\n'.join(lines))

