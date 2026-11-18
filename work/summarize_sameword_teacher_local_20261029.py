import json
from pathlib import Path
root=Path('outputs/mei_sameword_teacher_20261029');d=json.loads((root/'verified_summary_public.json').read_text())
assert d['status']=='PASS' and d['targets']==3
lines=['# U1 U4 U5 同词匹配教师baseline结果','',
'用户确认数字及20指令的有序词表；行标签沿用Audio审计支持的前100条10项周期、后200条20项周期。本实验逐词/逐重复/逐身份在该映射下完全平衡，但没有逐录音独立转写，因此不声称词义清单已逐条独立核实。','',
'## 认证数据','',
'认证使用15个尚未进入冻结上游模型训练的词：five、six、seven、eight、nine、Next、Back、Off、Grab、Down、Up、Zoom、Record、Play、Reset。不是全部30词。','',
'所有九人使用同一词集，训练重复0至4，校准5/6，测试7/8。每身份75条训练、30条校准、30条测试；第9次重复对所有身份统一不用，以避免缺行不平衡。每目标75本人/600负类训练、30/240校准、30/240测试。每个测试词2条本人、16条攻击。所有八个攻击身份保留，没有按效果筛人。','',
'## baseline与教师','',
'完整772维RBF baseline与固定拼接的900维教师特征分别训练，后者追加64维ACC教师和64维生成Audio教师。并列三种冻结教师家族；只用认证训练重复折选择教师的结果为主路线。不重训G/F或教师，不加Diffusion/AS-Norm。推理只输入ACC，不使用真实认证Audio。','',
'|目标|路线|本人接受/测试|TAR|攻击接受/测试|FAR|总体AUC|逐同词平均AUC|','|---|---|---|---:|---|---:|---:|---:|']
names={'baseline772':'完整772维baseline','old_teacher900':'旧目标教师900维','identity_teacher900':'身份保持教师900维','robust_teacher900':'稳健教师900维','train_selected_teacher':'训练内选择教师'}
for u in ['1','4','5']:
 for route,name in names.items():
  r=d['results'][u]['routes'][route]
  lines.append('|U%s|%s|%d/%d|%.2f%%|%d/%d|%.2f%%|%.4f|%.4f|'%(u,name,r['genuine_accepted'],r['genuine_n'],100*r['tar'],r['attack_accepted'],r['impostor_n'],100*r['far'],r['auc'],r['same_word_mean_auc']))
lines+=['','三位目标均由训练折选择identity_teacher900，未根据测试成绩选教师。认证模型为RBF，不能将本结果称为重跑SupCon。','',
'## 三目标宏平均','', '|路线|TAR|FAR|总体AUC|逐同词平均AUC|','|---|---:|---:|---:|---:|']
for route in ['baseline772','train_selected_teacher']:
 r=d['macro'][route];lines.append('|%s|%.2f%%|%.2f%%|%.4f|%.4f|'%(names[route],100*r['tar'],100*r['far'],r['auc'],r['same_word_mean_auc']))
lines+=['','## 阶段结果','',
'训练内选择教师后，U1为30/30本人接受、0/240攻击误接受；零误接受仅针对当前有限测试。U4从24/30、46/240改善为25/30、26/240；U5本人22/30不变，误接受94/240增至96/240，教师未改善该用户。没有因U5最终成绩更差而事后改选另一教师。',
'','三目标平均TAR提高1.11个百分点、FAR降低3.75个百分点、AUC提高约0.0195。逐同词平均AUC与总体AUC接近，减少了不同身份词频不一致的解释；仍不能据此排除会话或设备等身份相关混杂因素。',
'','## 解释范围','',
'同一用户的训练与测试录制不同，但词相同；本人和攻击者词频及重复设计一致，逐词AUC进一步控制了内容混合影响。它不等同于只有身份因素变化：会话、佩戴、幅度及原数组前端统计仍可能有影响。',
'','上游和认证的词类别仍分开。该设计验证认证头在同词独立重复上的效果，不是全流程所有词均重叠。由于旧上游占用其余15词全部重复，若要在全部30词建立从生成训练到认证测试均同词的实验，必须重划重复并重训上游。',
'','本轮训练75条、测试30条、词集15个，与此前跨内容80条训练、47至50条测试不是仅改变一个因素的配对实验。不可直接相减声称跨内容损失。历史录制及特征曾被使用，仍为探索性复测、单种子。',
'','外层按5次重复留一、内层按重复3折选SVM参数；各折保持全部15词，不使用错误的统一row除10标签。阈值来自校准，不利用最终分数调参。独立核验包括上游索引无交集、所有人内容/重复完全一致、675条Scaler拟合数量、OOF家族选择、模型分数、每词2/16和逐来源计数。']
Path('docs/experiments/2026-10-29-sameword-teacher-baseline-results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('\n'.join(lines))

