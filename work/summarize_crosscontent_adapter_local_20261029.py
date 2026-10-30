import json
from pathlib import Path
root=Path('outputs/mei_crosscontent_adapter_20261029');d=json.loads((root/'verified_summary_public.json').read_text())
assert d['status']=='PASS' and d['targets']==3
names={'rbf_matched':'同验证内容选模RBF','raw772_cosine':'未训练772维余弦','ordinary_supcon':'同尺寸普通SupCon','crossword_only':'仅跨词身份对比','episode_only':'仅注册查询训练','combined':'跨词加注册查询联合'}
lines=['# 跨内容身份适配器三目标三seed初始结果','',
'新增772到128、GELU、128到64、L2适配器，上游个人CycleGAN和772维特征冻结。U1/U4/U5各自训练，八个攻击身份全部保留。四种训练目标均400步、seed42/43/44，不按最终成绩挑seed。','',
'跨词对比以同人不同词为正对，同人同词不进入分母，异人同词负对权重2，温度0.07。注册查询训练每步选不同支持词和查询词，各身份各2条支持和查询，九类原型CE温度0.1。联合版两个损失等权。','',
'每身份50条训练、30条未见内容验证、20条独立内容校准，最终47至50条本人、其余八人全部攻击。模型选模不读校准和最终成绩；阈值只读校准。RBF也用同一验证内容选参数，因此不直接覆盖历史仅训练CV的RBF成绩。','',
'## 三目标平均','', '|路线|TAR|FAR|AUC|HTER|同词平均AUC|','|---|---:|---:|---:|---:|---:|']
for mode in names:
 r=d['macro'][mode];lines.append('|%s|%.2f%%|%.2f%%|%.4f|%.2f%%|%.4f|'%(names[mode],r['tar']*100,r['far']*100,r['auc'],r['hter']*100,r['same_word_mean_auc']))
lines+=['','## 每目标三seed均值及样本标准差','', '|目标|路线|TAR 均值±SD|FAR 均值±SD|AUC 均值±SD|HTER 均值|','|---|---|---:|---:|---:|---:|']
for u in ['1','4','5']:
 a=d['results'][u]
 for route in ['rbf_matched','raw772_cosine']:
  r=a['routes'][route];lines.append('|U%s|%s|%.2f%%|%.2f%%|%.4f|%.2f%%|'%(u,names[route],r['tar']*100,r['far']*100,r['auc'],r['hter']*100))
 for mode,r in a['seed_summary'].items():
  lines.append('|U%s|%s|%.2f±%.2f%%|%.2f±%.2f%%|%.4f±%.4f|%.2f%%|'%(u,names[mode],r['tar']['mean']*100,r['tar']['std']*100,r['far']['mean']*100,r['far']['std']*100,r['auc']['mean'],r['auc']['std'],r['hter']['mean']*100))
lines+=['','## 初始判断','',
'联合版相对同尺寸普通SupCon，三目标平均AUC仅从0.8723到0.8742，HTER从19.81%到19.85%，没有显示稳定增量。单独跨词对比也未改善平均HTER。仅注册查询训练HTER19.01%，比普通SupCon低约0.80个百分点，是值得继续验证的分量，但仍低于当前匹配RBF的17.07%综合表现。',
'','训练后的适配器相对未训练772维余弦有改善，但这不能单独说明跨词损失有效，因为普通SupCon也有改善。跨词联合方案目前不是已经成功的创新点2，只是已实现并完成首轮消融的候选机制。',
'','RBF和适配器使用不同最终读出：RBF为非线性二分类，适配器为注册原型余弦。二者比较反映完整认证流程，不能单独归因于表征质量。若进一步定位，应增加冻结适配器64维加同RBF读出的控制，不应按最终表现挑seed替换当前记录。',
'','## 解释边界','',
'适配器全体36模型、两类对照共42条路线已核验。独立重载checkpoint重提向量，核对验证选模排名、训练Scaler仅450条、校准阈值、最终分数与全部来源接受数和词内AUC。另用手工小矩阵核验加权跨词对比损失、梯度非零有限及不同词episodic采样。',
'','本轮没有教师补充分支、Diffusion、AS-Norm或测试知情删人。最终注册中心仍来自认证训练录制，非独立新注册池；八个负类身份均已知，不证明未见用户开放集。历史最终集已多次观察，词序依赖用户词表加10/20周期及尾部缺行假设。',
'','三seed反映初始化及episode采样变化，不等同于新增用户或独立确认实验。最终模型经验证集选checkpoint，本轮结果也不能仅凭较好个例宣称创新点2成立。新颖性仍需相对已有内容稳健身份学习方法明确区分。']
Path('docs/experiments/2026-10-29-crosscontent-adapter-results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
compact={k:d[k] for k in ['status','targets','routes_verified','macro','loss_checks','verification']};compact['results']={}
keep=['tar','far','auc','hter','same_word_mean_auc','genuine_n','impostor_n','genuine_accepted','attack_accepted','mode','seed','selected_step']
for u,a in d['results'].items():compact['results'][u]=dict(seed_summary=a['seed_summary'],routes={key:{k:r[k] for k in keep if k in r} for key,r in a['routes'].items()})
(root/'summary_public.json').write_text(json.dumps(compact,indent=2),encoding='utf-8')
print('\n'.join(lines))

