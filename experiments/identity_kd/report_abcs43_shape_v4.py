"""Publishable anonymous aggregate and every-user tables."""
import json
from pathlib import Path
root=Path('outputs/abcs43_shape_v4_20261110');r=json.loads((root/'results_public.json').read_text());users=json.loads((root/'per_user_public.json').read_text())
lines=['# ABCS43：固定频谱形状＋个人 Audio 蒸馏补充','', '日期：2026-11-10。43位现有可用参与者全部纳入身份划分：23 Base、8 Cal、12 Test；并非43人均作为未见目标。物理通道尚未确认，暂以ch0作振动、ch1作Audio；结论有此前提。','', '## 协议与技术路线','', '- 固定身份哈希划分，不按认证效果选人。2445条记录通过CRC核查，无完全相同PCM重复。','- Base每人60训练＋15开发；Cal/Test每人12注册＋24查询。开发数据本轮保留未用于调参。各角色文本全局不重叠；12位Test查询相同24句、注册相同12句；8位Cal另用相同24句。属于跨句子开放集，不能称同句重复验证。','- ch0：整段去直流，4096点Hann/1024步长/16384 FFT；50–7600Hz几何40频带、16时间格。没有只保留100–500Hz。提取均值40、标准差40、分段160、幅度6；StandardScaler仅拟合Base训练。','- Base学生700步，CE＋0.1 SupCon，平衡身份和同句困难负样本。','- 每个Cal/Test注册用户独立从原始VoxCeleb ECAPA重置，使用本人12条注册Audio＋23位Base各4条回放；主体微调30步，主体LR=1e-6、分类头1e-4。','- 个人学生100步；身份损失＋Base锚定＋梯度EMA平衡蒸馏，rho=0.3、EMA=0.9、lambda上限100。保持shape40不变，与学生192维等权拼接为232维。无CycleGAN或Diffusion。','- 注册允许Audio；查询仅使用ch0。不使用查询Audio进行训练或推断。阈值只由独立8位Cal选择，Test不挑阈值、不剔除攻击来源。所有超参数固定，seed42。','- 每位Test有24本人、264攻击比较；共288本人、3168攻击比较。重复参与的查询不是独立3168条录制。','', '## 主结果（独立Cal-EER阈值，百分比）','', '| 特征 | TAR | FAR | HTER | pooled AUC |','|---|---:|---:|---:|---:|']
for m,v in r['results'].items():
 q=v['Cal_EER'];lines.append(f"| {m} | {100*q['tar']:.2f} | {100*q['far']:.2f} | {100*q['hter']:.2f} | {q['auc']:.4f} |")
lines+=['','补充相比shape40的HTER下降1.56个百分点，12人中9人改善；身份bootstrap 95%区间为[-3.63,+0.28]个百分点，仍跨零。不能确认稳定超过原始形状。相对学生单独使用下降3.44个百分点，也不等同于已证明Audio的因果收益：还缺同预算无Audio个人适配、错配Audio对照与多种子。','', '在Cal-FAR≤5%的规则下，补充特征Test TAR=96.53%、FAR=6.50%、HTER=4.99%；Cal安全约束不保证Test FAR≤5%。主工作点最差目标FAR=14.02%、最低TAR=83.33%。','', '## 每位未见用户：所有方法（Cal-EER）','', '| 用户 | 方法 | TAR% | FAR% | HTER% | AUC | 本人/攻击 |','|---|---|---:|---:|---:|---:|---:|']
for v in users:
 if v['role']=='Test' and v['point']=='Cal_EER':lines.append(f"| {v['user']} | {v['method']} | {100*v['tar']:.2f} | {100*v['far']:.2f} | {100*v['hter']:.2f} | {v['auc']:.4f} | {v['genuine_n']}/{v['impostor_n']} |")
for role,point,title in [('Test','Cal_FAR5','每位未见用户：补充特征的Cal-FAR5工作点'),('Cal','Cal_EER','8位校准用户诊断（用于选阈值，不算独立测试）')]:
 lines+=['','## '+title,'','| 用户 | 方法 | TAR% | FAR% | HTER% | AUC | 本人/攻击 |','|---|---|---:|---:|---:|---:|---:|']
 for v in users:
  if v['role']==role and v['point']==point and (role=='Cal' or v['method']=='supplement232'):lines.append(f"| {v['user']} | {v['method']} | {100*v['tar']:.2f} | {100*v['far']:.2f} | {100*v['hter']:.2f} | {v['auc']:.4f} | {v['genuine_n']}/{v['impostor_n']} |")
lines+=['','## 完整性与边界','','独立复核20个个人模型、160个用户工作点：保存分数与重新推断最大差1.19e-7；每个教师重置后的Base输出差为0；训练/注册/查询文本隔离通过，查询Audio模型前向次数为0。完整匿名逐用户双工作点见per_user_public.json。23位Base不报告未见用户认证结果。','','缺少.z01，不能称59人实验；不能将与旧8人结果的差异归因为扩大训练量，因为身份、文本、攻击池也变化。单随机种子、同一语料探索性实验；无重戴/跨设备验证；通道物理归属未确认。不得将高AUC直接等同于可部署安全性。']
out=Path('docs/experiments/2026-11-10-abcs43-shape-v4-results.md');out.write_text('\n'.join(lines)+'\n',encoding='utf-8');print(out)
