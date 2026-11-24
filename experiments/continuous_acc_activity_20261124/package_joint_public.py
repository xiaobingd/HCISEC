"""Copy an explicit anonymous publication allowlist; never copy private caches."""
import json
from pathlib import Path
import shutil

root = Path(__file__).resolve().parents[2]
base = root / 'outputs/vr261119_10x40_raw_acc/continuous_acc_activity'
dest = root / 'outputs/vr261125_joint_identity_activity'
dest.mkdir(exist_ok=True)
names = ['report.md', 'decision_public.json', 'protocol_public.json',
         'verification_public.json', 'independent_verification_public.json',
         'analysis_plan_public.json', 'summary_public.json',
         'per_seed_user_public.json', 'training_public.json',
         'selection_public.json', 'exclusion_audit_public.json',
         'activity_quality_public.json']
files = []
for name in names:
    target = dest / name
    shutil.copyfile(base / 'joint_identity' / name, target)
    files.append(target)
shutil.copyfile(base / 'word_split_public.json', dest / 'word_split_public.json')
files.append(dest / 'word_split_public.json')
readme = dest / 'README.md'
readme.write_text('''# 联合身份与 Audio 活动监督实验

完整中文方法、逐用户结果与限制见 [report.md](report.md)。
公开内容仅为匿名汇总和代码；不含原始信号、embedding、权重、身份映射、个人阈值或逐样本分数。

## 运行

需要 Python、PyTorch、NumPy、SciPy、scikit-learn、threadpoolctl，以及授权私有缓存。
数据准备与缓存字段沿用此前连续 ACC 实验，公开目录不足以重新生成私人数据。

```powershell
python experiments/continuous_acc_activity_20261124/test_joint_identity.py
python experiments/continuous_acc_activity_20261124/run_joint_identity.py --root outputs/vr261119_10x40_raw_acc
python experiments/continuous_acc_activity_20261124/verify_joint_identity.py --root outputs/vr261119_10x40_raw_acc
python experiments/continuous_acc_activity_20261124/report_joint_identity.py --root outputs/vr261119_10x40_raw_acc
```

训练固定400步、三个种子；联合组共享开发选择的辅助权重。测试阶段只用 ACC。
本轮测试词曾被历史实验查看，属于探索复核。未改变冻结的创新点一。
''', encoding='utf-8')
files.append(readme)
for name in ['run_joint_identity.py', 'test_joint_identity.py',
             'verify_joint_identity.py', 'report_joint_identity.py',
             'joint_analysis_plan.json', 'run_continuous.py',
             'package_joint_public.py']:
    files.append(Path(__file__).resolve().parent / name)
manifest = [{'path': p.relative_to(root).as_posix(), 'local': str(p), 'bytes': p.stat().st_size} for p in files]
(dest / 'upload_manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
print(json.dumps(manifest))
