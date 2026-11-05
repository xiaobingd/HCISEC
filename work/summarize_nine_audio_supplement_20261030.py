import json,statistics
from pathlib import Path
root=Path('outputs/mei_nine_audio_supplement_20261030');data=json.loads((root/'verified_results_public.json').read_text());assert json.loads((root/'verification_public.json').read_text())['status']=='PASS' and len(data)==9
lines=['# 九用户生成Audio身份特征补充消融','',
'同一15词、同一重复录制划分：每身份训练75、校准30、最终30，全部八个攻击者，最终30本人/240攻击，每词2本人/16攻击。保留完整772维基线，G/F和教师冻结，认证推理只输入ACC。','',
'新增的是E_M(G(x))的64维身份表征，不是真实测试Audio。原772维已包含G内部编码。复用上一轮基线/原始统计/ACC教师分类器及OOF，仅为旧目标、身份保持、稳健三类生成Audio表征新训RBF分支，采用同一重复5外折3内折和平衡HTER规则。','',
'A：ACC教师可选（40候选，逐项复现前轮）；B：生成Audio教师可选（40）；C：ACC＋Audio独立选择家族和权重（400）；D：固定A已选ACC/原始统计，只允许新增Audio（10）。所有权重0/.25/.5/1，允许零补充。选择只用训练OOF，阈值只用校准本人和攻击者最小化HTER，不设接受率优先。','',
'## 宏平均','', '|路线|本人接受率|FAR|AUC|逐同词AUC|HTER|','|---|---:|---:|---:|---:|---:|']
names={'baseline772':'完整772维','acc':'A：只补ACC教师','audio':'B：只补生成Audio教师','both':'C：ACC与Audio联合选路','fixed_acc_audio':'D：固定ACC后补Audio'};macro={}
for key,name in names.items():
 m={k:statistics.mean(a['routes'][key][k] for a in data.values()) for k in ['tar','far','auc','same_word_mean_auc','hter']};macro[key]=m;lines.append('|%s|%.2f%%|%.2f%%|%.4f|%.4f|%.2f%%|'%(name,100*m['tar'],100*m['far'],m['auc'],m['same_word_mean_auc'],100*m['hter']))
lines+=['','## 各用户HTER','', '|用户|完整基线|A仅ACC|B仅Audio|C联合|D固定ACC补Audio|','|---|---:|---:|---:|---:|---:|']
for u,a in data.items():lines.append('|U%s|%s|'%(u,'|'.join('%.2f%%'%(100*a['routes'][k]['hter']) for k in names)))
lines+=['','## 全部路线与选中权重','', '|用户|路线|TAR|FAR|AUC|逐同词AUC|HTER|ACC教师/权重|Audio教师/权重|原始统计权重|','|---|---|---:|---:|---:|---:|---:|---|---|---:|']
for u,a in data.items():
 for k,name in names.items():
  r=a['routes'][k];lines.append('|U%s|%s|%.2f%%|%.2f%%|%.4f|%.4f|%.2f%%|%s / %.2f|%s / %.2f|%.2f|'%(u,name,100*r['tar'],100*r['far'],r['auc'],r['same_word_mean_auc'],100*r['hter'],r['acc'],r['acc_weight'],r['audio'],r['audio_weight'],r['raw_weight']))
lines+=['','## 相对A的结果','']
for k in ['audio','both','fixed_acc_audio']:
 wins=sum(a['routes'][k]['hter']<a['routes']['acc']['hter']-1e-12 for a in data.values());losses=sum(a['routes'][k]['hter']>a['routes']['acc']['hter']+1e-12 for a in data.values());included=sum(a['routes'][k]['audio_weight']>0 for a in data.values());lines.append('- %s：HTER改善%d、持平%d、变差%d；%d/9用户选入生成Audio。'%(names[k],wins,9-wins-losses,losses,included))
lines+=['','## 初步判断','',
'固定原ACC组合后只补Audio，宏HTER28.10%至26.71%，宏TAR72.59%至73.70%，FAR28.80%至27.13%；5人改善、4人持平、无退步。该受控增量支持当前生成Audio表征有一定补充价值，但仅为历史小样本探索。宏AUC0.7958至0.7945略降，不能把操作点改善等同整体排序能力提高。',
'独立只补Audio宏HTER26.67%，优于只补ACC28.10%；自由联合搜索27.31%反而差于只补Audio和固定ACC增补，说明更大的搜索空间不保证泛化。U4自由联合为TAR90%、FAR18.33%、HTER14.17%，固定ACC增补为90%、25.83%、17.92%；前者比本轮A好，但仍不是历史旧900维教师HTER13.75%的全面替代。',
'U5固定ACC增补后TAR86.67%降至73.33%，FAR46.67%降至29.58%，HTER30.00%至28.12%，改善来自拒绝更多攻击者同时也拒绝更多本人。不能只报告FAR下降。总体绝对错误率仍高，未达到实用认证效果。','',
'## 核验与边界','',
'独立重拟合135个新增Audio外折模型重建OOF，重放每用户490个候选和阈值，逐项核验A与前轮完全一致、D不改变ACC组合。重算45条路线最终分数、逐词和逐攻击计数，并直接从原始ACC经冻结G、Audio教师重提校准/测试样本，确认缓存来源为生成Audio。',
'C比A/B搜索空间更大，不能将提升全部归于模态信息，D提供更小搜索范围的增量检查。已有772维包含G内部特征，A并非完全不依赖Audio相关生成训练。即使生成Audio补充有效，也不证明恢复了ACC不存在的信息，可能是表示重编码。',
'历史最终集已观察，属于探索性已知用户闭集；词映射依赖10/20周期和跨用户相位，未逐录音转写。OOF元选择并非全新完整嵌套，存在搜索过拟合。没有按最终成绩回退、删用户或重调阈值。']
Path('docs/experiments/2026-10-30-nine-generated-audio-supplement-results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');(root/'macro_public.json').write_text(json.dumps(macro,indent=2),encoding='utf-8');print('\n'.join(lines[:35]))

