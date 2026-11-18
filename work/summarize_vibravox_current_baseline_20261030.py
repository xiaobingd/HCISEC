import json,statistics
from pathlib import Path
root=Path('outputs/vibravox_current_baseline_20261030');data=json.loads((root/'verified_results_public.json').read_text());assert json.loads((root/'verification_public.json').read_text())['status']=='PASS'
names={'full772':'完整772维','raw320':'原始ACC统计','input_encoder640':'原输入G编码','cycle_encoder640':'循环ACC编码','acc':'基线＋ACC教师','audio':'基线＋生成Audio教师','both':'ACC/Audio联合选择','fixed_acc_audio':'固定ACC后补Audio'}
lines=['# VibraVox当前基线迁移：两注册者先导','',
'VibraVox speech_clean前额加速度计与头戴麦克风。复用已有两位用户的原个人CycleGAN200轮（原生成训练每人30句/60窗口），本轮重新训练两位用户的个人ACC/Audio身份教师，并按当前HTER等权规则重训RBF及补充分数选择。没有重新训练生成器。','',
'每个注册者30句生成训练、6句生成验证、10句认证训练、6句校准、8句最终本人。另一注册者作为10句训练负类、6句校准负类、8句已知攻击；另外两位用户各24句仅作未见攻击，共48句。每句取两个相邻0.8秒窗口，先平均特征，再按完整原句评分一次。所有人的预处理一致，归一化只拟合两位注册者GAN TRAIN；未见用户不参与教师、认证器、参数选择或校准。','',
'官方文本核验无完全相同文本跨角色，但不同用户不是同一句，不能称同内容认证或完全内容无关。此前已经探索过该缓存人群，历史最终数据已观察，属于探索性先导。','',
'## 未见攻击者结果','', '|目标|路线|本人接受/8|TAR|未知误接受/48|未知FAR|未知AUC|未知HTER|','|---|---|---|---:|---|---:|---:|---:|']
for i,(u,a) in enumerate(data.items(),1):
 for key,name in names.items():
  r=a['routes'][key]['unseen'];lines.append('|V%d (%s)|%s|%d/%d|%.2f%%|%d/%d|%.2f%%|%.4f|%.2f%%|'%(i,u,name,round(r['tar']*r['genuine_n']),r['genuine_n'],100*r['tar'],round(r['far']*r['impostor_n']),r['impostor_n'],100*r['far'],r['auc'],100*r['hter']))
macro={};lines+=['','## 两注册者宏平均：未见攻击','', '|方法|TAR|未知FAR|未知AUC|未知HTER|','|---|---:|---:|---:|---:|']
for key,name in names.items():
 m={k:statistics.mean(a['routes'][key]['unseen'][k] for a in data.values()) for k in ['tar','far','auc','hter']};macro[key]=m;lines.append('|%s|%.2f%%|%.2f%%|%.4f|%.2f%%|'%(name,100*m['tar'],100*m['far'],m['auc'],100*m['hter']))
lines+=['','## 已知负类与选中组合','', '|目标|路线|已知FAR|完整772权重|原始统计权重|ACC教师权重|生成Audio教师权重|','|---|---|---:|---:|---:|---:|---:|']
for i,(u,a) in enumerate(data.items(),1):
 for key in ['full772','acc','audio','both','fixed_acc_audio']:
  r=a['routes'][key];w=r['weights'];lines.append('|V%d|%s|%.2f%%|%.2f|%.2f|%.2f|%.2f|'%(i,names[key],100*r['known']['far'],w.get('full772',0),w.get('raw320',0),w.get('teacher_acc',0),w.get('teacher_audio',0)))
lines+=['','## 初步解释','',
'完整772维平均未知FAR39.58%、HTER19.79%；ACC教师补充后33.33%、16.67%，有小幅改善，但纯ACC统计平均FAR3.12%、HTER4.69%，明显更强。不能据此声称跨模态基线优于纯ACC。',
'固定ACC后补Audio的训练选择对两个目标都给Audio零权重：本轮未复现自采九人中的生成Audio增量。V1全部教师补充均退回原完整基线；V2联合选择偏向Audio而固定ACC仍保持ACC补充，最终前者未知HTER30.21%，后者27.08%，没有根据未知测试改回较好版本。',
'每个目标只有一个已知训练负类，已知类别可分不等于未知攻击可拒绝；少量训练句和校准句不足以支持稳定开放集结论。纯ACC统计高分也可能混入幅值、佩戴、句子或录制条件差异，不能自动解释为生理身份。','',
'## 训练与核验','',
'个人教师只使用两位已知注册者的生成训练/验证句，600步；BCE＋两身份CE＋不同句正对对比＋真实跨模态身份一致性，验证平均ACC/真实Audio AUC选教师。认证仅输入ACC以及其生成Audio的教师表征。生成验证不使用认证句。',
'认证RBF每分支C1/10/100、gamma scale/.1，5外折3内折均按整句划分。分数标准化只用训练分数。补充权重0/.25/.5/1：ACC16候选、Audio16、联合64、固定ACC增补4。校准同时用本人和已知负类最小化HTER，不设置本人接受率优先，不用未知攻击结果改阈值。',
'独立核验60个外折认证模型重建OOF，重放融合和阈值，检查按句池化、各角色文本及索引隔离、教师训练范围及原模型SHA。直接用原G/F重提完整772与生成Audio教师特征，确认来源。',
'只有8条独立本人测试、48条未知攻击，两个窗口不是两条独立测试；仅一个训练负类身份，校准各6句，结果方差很大。没有训练完整VibraVox或八用户个人模型。专用前额传感器不等同VR IMU。历史选择与OOF元搜索有过拟合风险，不作显著性、部署安全或新增生理信息结论。','',
'官方来源：https://huggingface.co/datasets/Cnam-LMSSC/vibravox/blob/main/README.md 。']
Path('docs/experiments/2026-10-30-vibravox-current-baseline-results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');(root/'macro_public.json').write_text(json.dumps(macro,indent=2),encoding='utf-8');print('\n'.join(lines[:40]))

