"""Local independent score verification and complete experiment report."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'work/public_signal_runtime_20261028'))
import numpy as np
R=Path('outputs/mei_error_trim_20261028')
def main():
 d=json.loads((R/'results_public.json').read_text());z=np.load(str(R/'scores_private.npz'));checked=0
 lines=['# Mei九人数据：训练折外错误筛除实验（2026-10-28）','',
 '结论：固定规则筛除反复判错的认证训练样本，没有带来稳定改善。困难样本不等同于传感器坏数据；不建议按认证错误直接删除最终测试或未见攻击数据。','',
 '## 协议','',
 '- 目标U1/U2/U4/U5/U7；U3/U8为未见攻击者，不参与认证拟合、标准化、选参、筛除或校准。其余身份为已知负类，所有八个最终攻击来源完整保留。',
 '- 冻结已有个人CycleGAN，不重新训练生成器。比较完整772维和完整772维＋已有个人ACC教师64维，两条路径均报告未筛选/筛选结果。教师不是本轮新训练的教师。',
 '- 仅使用与上游随机150条不重叠的认证补集。词0–14认证拟合，15–19校准，20–29最终测试。上游生成器见过这些词类别，因此只称认证阶段跨词，不称全流程内容隔离。',
 '- 训练数据按词分组5折选择RBF的C=1/10/100、gamma=scale/0.1；每折独立标准化，各分支除以维度平方根。按平均验证AUC选参。筛除前后共用同一参数，分别重新拟合和校准。',
 '- 用seed42/43/44的三次按词分组折外预测，分数0为固定判错边界；至少2次判错才候选。每个身份最多删除该身份训练样本的floor(10%)，优先错误次数多、平均正确方向间隔小的样本。无迭代删除，无最终结果调参。',
 '- 折外筛选采用在训练集整体交叉验证中选出的参数，未做完全嵌套选参；故折外错误频率只是训练筛选信号，不作为无偏泛化指标。校准与最终样本从不影响删除名单。',
 '- 本次没有按用户改滤波或归一化算法，没有修改源数组。所有最终样本原样保留，仅改变认证拟合名单。',
 '- 历史测试录制已多次使用，本轮是探索性对照，不是新盲测。U4/U7词序仍依赖“缺失在末组”假设；单会话/设备条件混杂未排除。','',
 '## 未见攻击者结果','',
 '每个目标未见攻击88条（U3与U8合计）。本人测试数因上游随机150条及缺行不同而不同。表中百分数使用各自锁定校准阈值。','',
 '|目标|特征|筛选|本人接受|TAR|未见攻击误接受|FAR|AUC|剔除训练数|','|---|---|---|---|---:|---|---:|---:|---:|']
 appendix=['','## 全部攻击与逐来源结果','','以下结果同样保留已知和未知攻击来源，不删除难攻击者。','']
 for u,features in d.items():
  for mode,row in features.items():
   removed=sum(v['removed'] for v in row['removed_by_identity'].values())
   label='772' if mode=='baseline772' else '772＋教师64'
   for arm,detail in row['routes'].items():
    key='U%s_%s_%s'%(u,mode,arm);s=z[key+'_scores'];uid=z[key+'_users'];ix=z[key+'_rows'];y=uid==int(u);t=detail['fit']['threshold']
    assert np.all(ix//10>=20)
    for scope in ('all','unknown','known'):
     mask=np.ones(len(y),bool) if scope=='all' else y|np.isin(uid,[3,8]) if scope=='unknown' else ~np.isin(uid,[3,8])
     ss=s[mask];yy=y[mask];p=ss[yy];n=ss[~yy];auc=float(((p[:,None]>n[None,:])+.5*(p[:,None]==n[None,:])).mean());m=detail[scope]
     assert abs(auc-m['AUC'])<1e-12 and int((p>=t).sum())==m['genuine_accepted'] and int((n>=t).sum())==m['attack_accepted']
    for v,m in detail['by_attacker'].items():assert int((s[uid==int(v)]>=t).sum())==m['accepted']
    m=detail['unknown'];lines.append('|U%s|%s|%s|%d/%d|%.2f%%|%d/%d|%.2f%%|%.4f|%d|'%(u,label,'是' if arm=='trimmed' else '否',m['genuine_accepted'],m['genuine_count'],100*m['TAR'],m['attack_accepted'],m['attack_count'],100*m['FAR'],m['AUC'],removed if arm=='trimmed' else 0))
    appendix.extend(['### U%s / %s / %s'%(u,label,arm),'', '|范围|本人接受|攻击误接受|FAR|AUC|','|---|---|---|---:|---:|'])
    for scope in ('all','known','unknown'):
     a=detail[scope];appendix.append('|%s|%d/%d|%d/%d|%.2f%%|%.4f|'%(scope,a['genuine_accepted'],a['genuine_count'],a['attack_accepted'],a['attack_count'],a['FAR']*100,a['AUC']))
    appendix.extend(['','|攻击者|误接受/总数|FAR|','|---|---|---:|'])
    for v,a in detail['by_attacker'].items():appendix.append('|U%s|%d/%d|%.2f%%|'%(v,a['accepted'],a['count'],a['FAR']*100))
    appendix.append('');checked+=1
 lines.extend(['','## 解释与边界','',
 '筛除负类难例可能移走最重要的决策边界约束；筛除本人难例可能缩窄可接受的本人变化范围。FAR略降同时TAR大降，不能解释为认证整体改善。固定阈值随重新校准变化，AUC与TAR/FAR需一起读。',
 '本轮不修改最终测试集，不按未见攻击表现删样本或挑目标，因此与“删除误接受攻击后重算结果”有本质区别。也不证明任何去噪策略无效：本轮试的是训练错误驱动筛选，不是物理噪声诊断。',
 '这轮基线为完整772维RBF读出，不是旧SupCon神经头；与以前不同协议或不同认证器的数字不可直接相减。逐来源/所有攻击结果见下。'])
 lines+=appendix
 assert checked==20
 (R/'local_verification_public.json').write_text(json.dumps(dict(status='PASS',routes_checked=checked,independent_pairwise_auc=True,per_attacker_counts_verified=True),indent=2))
 path=Path('docs/experiments/2026-10-28-mei-training-error-trim-results.md');path.write_text('\n'.join(lines)+'\n',encoding='utf-8')
 print('\n'.join(lines[:50]))
if __name__=='__main__':main()
