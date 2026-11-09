"""Cohort-only KD-RKD-01 tables and scientific figure, no speaker identifiers."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
LABELS={'S0_shape':'S0 频谱形状','S0_allstats':'S0 全统计','S1_student':'S1 纯振动残差','S2_student':'S2 Cosine KD','S3_student':'S3 RKD-D','S4_student':'S4 RKD-D+A','S5_student':'S5 MMD','S6_student':'S6 错误身份RKD','S7_student':'S7 振动几何正则'}
def main(a):
    root=Path(a.out);r=json.loads((root/'results_public.json').read_text());diag=json.loads((root/'teacher_diagnostic_public.json').read_text());methods=[m for m in LABELS if m in r['results']];ks=['3','5','12'];text='# KD-RKD-01：VibraVox 关系蒸馏首轮结果\n\n'
    text+='35 Base / 12 Cal / 12 Test，三训练种子42/43/44，所有Test身份均未参与学生训练或超参数选择。注册与查询均仅使用振动；12位Test互相攻击。每目标24条本人查询、264条攻击查询。K=3/5/12为相同12条注册记录的嵌套子集。\n\n'
    text+=f"外部冻结VoxCeleb ECAPA在Base开发集AUC={diag['base_dev_auc']:.4f}、EER={100*diag['base_dev_eer']:.2f}%，Audio/振动身份关系Spearman={diag['identity_geometry_spearman']:.3f}。Base开发集中异人同文本配对为{diag['different_user_same_text_pairs']}，不能声称教师已通过相同文本困难负类验证。\n\n"
    text+='本轮是**纯统计模块3**，未加入CycleGAN。固定40维相对谱形状为强参考，并保留246维统计参考；所有学生统一输入246维。全程未读取Cal/Test的Audio。\n\n'
    for k in ks:
        text+=f'## {k}-shot，Cal-EER阈值\n\n| 方法 | 宏AUC | 宏EER | TAR | FAR | HTER | 最困难目标FAR |\n|---|---:|---:|---:|---:|---:|---:|\n'
        for m in methods:
            q=r['results'][m][k]['Cal_EER'];text+=f"| {LABELS[m]} | {q['macro_target_auc']:.4f} | {100*q['macro_target_eer']:.2f}% | {100*q['tar']:.2f}% | {100*q['far']:.2f}% | {100*q['hter']:.2f}% | {100*q['worst_target_far']:.2f}% |\n"
        text+='\n'
    text+='## 12-shot安全工作点\n\nCal-FAR5阈值在Cal上满足FAR≤5%，不保证Test同样满足。\n\n| 方法 | Test TAR | Test FAR | HTER |\n|---|---:|---:|---:|\n'
    for m in methods:
        q=r['results'][m]['12']['Cal_FAR5'];text+=f"| {LABELS[m]} | {100*q['tar']:.2f}% | {100*q['far']:.2f}% | {100*q['hter']:.2f}% |\n"
    text+='\n## 统计与学生融合（12-shot，Cal-EER）\n\n| 方法 | 宏AUC | TAR | FAR | HTER |\n|---|---:|---:|---:|---:|\n'
    for m in r['results']:
        if not m.endswith('_fusion'):continue
        q=r['results'][m]['12']['Cal_EER'];text+=f"| {m} | {q['macro_target_auc']:.4f} | {100*q['tar']:.2f}% | {100*q['far']:.2f}% | {100*q['hter']:.2f}% |\n"
    text+='\n## 目标身份级配对差值\n\n负值代表RKD降低HTER。先对同一身份平均三种子，再对12个身份bootstrap5000次；区间不能代表跨数据集不确定性。\n\n| 对照 | HTER差值(pp) | 身份bootstrap95%区间(pp) | 改善身份数/12 | 各种子差值(pp) |\n|---|---:|---|---:|---|\n'
    for key,v in r['paired'].items():text+=f"| {key} | {100*v['hter_delta']:.2f} | [{100*v['identity_bootstrap95'][0]:.2f}, {100*v['identity_bootstrap95'][1]:.2f}] | {v['improved_identities']} | {', '.join(f'{100*z:.2f}' for z in v['seed_mean_deltas'])} |\n"
    if (root/'selection_public.json').exists():
        selection=json.loads((root/'selection_public.json').read_text());text+='\n## Base内部锁定的超参数\n\n| 方法 | λ | 检查点步数 | 融合α |\n|---|---:|---:|---:|\n'
        for m,p in selection.items():text+=f"| {m} | {p['lam']} | {p['step']} | {p['alpha']} |\n"
        text+='\n零λ表示该开发流程没有选择Audio约束，不能将相应方法称为有效蒸馏。零α表示融合评分完全由原统计分支决定。\n'
    text+='\n## 解释边界\n\nAUC/EER按目标身份计算，再平均身份及训练种子；固定阈值TAR/FAR/HTER按相同数量查询汇总。最困难目标FAR为每身份先平均种子后取最大值。JSON另保留跨目标池化AUC/EER，不能与宏平均混淆。EER由Test分数作描述性排序诊断，部署阈值只能来自Cal。\n\nS3必须同时超过S1、S6和非Audio控制，才可讨论Audio身份关系的增量；S3超过S2才支持关系蒸馏优于逐样本匹配。不能只依据融合增益掩盖学生退化。\n\n这是VibraVox一次固定身份划分，之前已有探索；没有相同文本攻击、重新佩戴或跨设备控制。ABCS需先确认通道物理归属，HMD需完成词标签及采样率审核；两项尚不能冒充本轮已完成的严格复现。完整[协议](2026-11-09-kd-rkd01-protocol.md)。\n'
    Path(a.report).write_text(text,encoding='utf8')
    fig,axes=plt.subplots(1,2,figsize=(13,4.8));x=np.arange(len(methods));w=.25
    for off,key,label in [(-1,'tar','TAR'),(0,'far','FAR'),(1,'hter','HTER')]:axes[0].bar(x+off*w,[100*r['results'][m]['12']['Cal_EER'][key] for m in methods],w,label=label)
    axes[0].set_xticks(x,[m.replace('_student','').replace('_shape',' shape').replace('_allstats',' all') for m in methods],rotation=35);axes[0].set_ylim(0,100);axes[0].set_ylabel('Percent; mean across 3 seeds');axes[0].legend();axes[0].set_title('12-shot unseen users, Cal-EER threshold')
    for m in ['S0_shape','S1_student','S2_student','S3_student']:axes[1].plot([3,5,12],[100*r['results'][m][k]['Cal_EER']['hter'] for k in ks],marker='o',label=m.replace('_student',''))
    axes[1].set_xlabel('Enrollment recordings');axes[1].set_ylabel('HTER (%)');axes[1].set_xticks([3,5,12]);axes[1].legend();axes[1].set_title('Nested K-shot registration; identical queries');fig.tight_layout();fig.savefig(root/'rkd01_authentication.png',dpi=180);fig.savefig(root/'rkd01_authentication.pdf');plt.close(fig)
    hh=json.loads((root/'similarity_histograms_public.json').read_text());bins=np.array(hh['bins']);fig,ax=plt.subplots(figsize=(7,4))
    for key,label in [('same_user_different_text','Same speaker, different text'),('different_user','Different speakers')]:
        counts=np.array(hh[key]);ax.stairs(counts/counts.sum()/np.diff(bins),bins,label=label)
    ax.set_xlabel('Audio ECAPA cosine similarity');ax.set_ylabel('Density');ax.set_title('Base development only; external frozen teacher');ax.legend();fig.tight_layout();fig.savefig(root/'rkd01_teacher_diagnostic.png',dpi=180);plt.close(fig)
    print('REPORT_COMPLETE',len(methods),'methods',r['seeds'])
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--report',required=True);main(p.parse_args())
