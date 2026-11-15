"""Report the fresh CPU rerun without replacing the historical report."""
import json
from pathlib import Path
p=Path('outputs/mei9_closed_rerun_20261110');r=json.loads((p/'results_public.json').read_text());per=json.loads((p/'per_user_public.json').read_text());s=json.loads((p/'selection_public.json').read_text());audit=json.loads((p/'verification_public.json').read_text());assert audit['state']=='PASS'
lines=['# Mei9闭集同词／跨词：独立CPU重训结果','','2026-11-15完成审核。共享CycleGAN4000步、九个人模型各100步、九个二分类RBF认证头均已完成。重放审核PASS：18个人检查点、432个工作点，缓存特征/认证分数重放误差0。审核在CPU执行；旧审核JSON内含gpu_reextraction历史字段名，不表示本轮使用GPU。','','输入为三轴OpenVR tracking速度频谱。词标签按采集顺序推断，未逐条转写。全九人参与身份学习，八个已知攻击来源全部保留。共享fit1438条，develop36条；每人12条注册、8条校准、16条同词查询、24条跨词查询，录制索引互斥。Audio只用合法训练/注册录制；个人G/F仅用本人12条配对微调。','','模型：ACC统计残差编码器192维、九人共享Audio ECAPA微调/冻结蒸馏；原始形状120+学生192构成baseline312。共享G/F采用LSGAN+配对/循环L1，3→1→3；个人100步。RBF C10/gamma=scale/balanced，108条注册特征拟合。CalA设阈值、CalB选特征，最终CalA+B设个人阈值；同词和跨词共用阈值。','','校准选中个人100步的input256，alpha0.5。它是ACC→Audio生成器的输入编码特征，不是循环输出；不能归因于循环后身份增益。','','|方法|条件|TAR%|FAR%|HTER%|AUC|','|---|---|---:|---:|---:|---:|']
for m in ['statistics720','baseline312','selected_cycle']:
 for c in ['sameword','crossword']:
  v=r['results'][m][c]['Cal_EER'];lines.append(f'|{m}|{c}|'+ '|'.join(f'{100*v[k]:.2f}' for k in ['tar','far','hter'])+f'|{v["auc"]:.4f}|')
lines+=['','## 逐用户全部方法及工作点','','|用户|方法|条件|校准规则|TAR%|FAR%|HTER%|AUC|','|---|---|---|---|---:|---:|---:|---:|']
for v in per:lines.append(f'|{v["user"]}|{v["method"]}|{v["condition"]}|{v["point"]}|'+ '|'.join(f'{100*v[k]:.2f}' for k in ['tar','far','hter'])+f'|{v["auc"]:.4f}|')
lines+=['','单种子、已探索小数据；不证明未见用户、跨天或重新佩戴泛化。同词和跨词的词集合及查询数量不同，并非完全单变量对照。Cal-FAR5仅是校准目标，不保证查询FAR≤5%。本次没有新训练同预算无Audio学生对照，因此不能把整体改善全部归因于Audio。此前个人物理Audio辅助头实验是独立实验，不应与本轮混合。']
(p/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf8');print(r['paired_vs_baseline312']['selected_cycle'])
