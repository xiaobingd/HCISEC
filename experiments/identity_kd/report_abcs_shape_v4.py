"""ABCS provisional-channel V4 cohort report, never per-speaker publication."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
LABELS={'shape40':'通道0固定谱形状40维','statistics240':'通道0完整统计240维','student192':'V4学生单独192维','supplement232':'谱形状＋V4补充232维'}
def main(a):
    root=Path(a.out);r=json.loads((root/'results_public.json').read_text());assert r['integrity']['state']=='pass';text='# ABCS：谱形状保留＋注册教师微调/梯度平衡学生补充\n\n'
    text+='**通道物理归属未确认。** 本轮仅按“通道0暂作振动、通道1暂作Audio”的假设运行，不因效果好坏确认AC/BC。因此以下是这个通道方向的结果，不能冒充已核实的BC身份认证验证。\n\n'
    text+='八身份、八折4 Base/2 Cal/1目标/1未见攻击者；seed42、12-shot。每目标24本人/24未见攻击查询。注册/Cal查询/Test查询分别为12/13/24句，实际文本互斥；不同用户同一角色使用相同真实文本。只有跨句注册认证，没有同人同句独立重复数据，不造同内容训练/测试。\n\n'
    text+='只训练V4：每位注册者Audio微调原始VoxCeleb ECAPA主体30步并冻结，学生身份+梯度平衡RKD100步；每折Base-only初始化700步。保持固定40维无学习映射，直接拼接192维学生并除sqrt2，共232维。alpha1/rho0.3沿用外部开发值；目标数据仅用于合法注册适配，不用于选择超参数或最终阈值。\n\n'
    for scope,title in [('unseen_attacker','主结果：一个完全未见攻击身份'),('all_seven_mixed','辅助结果：全部7来源（含Base/Cal）')]:
        text+='## '+title+'\n\n'
        for point,ptitle in [('Cal_EER','Cal-EER'),('Cal_FAR5','Cal-FAR5')]:
            text+='### '+ptitle+'\n\n| 表示 | TAR | FAR | HTER | 宏AUC | 宏EER | 最差目标FAR | 最低目标TAR |\n|---|---:|---:|---:|---:|---:|---:|---:|\n'
            for m,label in LABELS.items():
                v=r['results'][m][scope][point];text+=f"| {label} | {100*v['tar']:.2f}% | {100*v['far']:.2f}% | {100*v['hter']:.2f}% | {v['auc']:.4f} | {100*v['eer']:.2f}% | {100*v['worst_target_far']:.2f}% | {100*v['lowest_target_tar']:.2f}% |\n"
            text+='\n'
    b=r['results']['shape40']['unseen_attacker']['Cal_EER'];s=r['results']['supplement232']['unseen_attacker']['Cal_EER'];p=r['paired_vs_shape'];text+='## 判断\n\n'
    text+=f"232维相对40维的HTER改善{100*(b['hter']-s['hter']):.2f}个百分点，目标级配对bootstrap95%差值区间[{100*p['identity_bootstrap95'][0]:.2f}, {100*p['identity_bootstrap95'][1]:.2f}]个百分点，仅{p['improved_targets']}/8目标改善，尚无稳定增益证据。\n\n"
    text+=f"补充后的FAR仍{100*s['far']:.2f}%，最差目标FAR={100*s['worst_target_far']:.2f}%、最低本人接受率={100*s['lowest_target_tar']:.2f}%；不能用队列均值掩盖个别完全失效。AUC由{b['auc']:.4f}变为{s['auc']:.4f}，没有证明整体排序改善；学生单独的HTER还略低于补充路线。\n\n"
    text+='固定40维和240维参考与上一轮ABCS统计对照一致。本轮Cal只有两个身份、合计26次负查询，Cal-FAR5无法保证Test FAR≤5%；这既是校准限制，也不能解释全部排序误差。整体尚不能称为可靠认证。\n\n'
    v=r['integrity'];text+=f"## 核查与限制\n\n24个个人ECAPA主体参数均更新；重置后的Base特征最大差={v['teacher_reset_base_embedding_max_difference']}。128条合法fit/注册Audio进入教师特征提取，查询Audio前向为0；19条短录制提供真实有效长度。24个只用通道0特征的推理模型、{v['pairwise_auc_workpoint_checks']}项逐目标/工作点pairwise AUC及全部宏指标复算通过，最大重算分数误差={v['max_score_recompute_error']}。40维未被替换。\n\n"
    text+=f"梯度比抽查平均{100*v['realized_gradient_ratio_mean']:.2f}%，抽查范围[{100*v['realized_gradient_ratio_range'][0]:.2f}%,{100*v['realized_gradient_ratio_range'][1]:.2f}%]。EMA规则存在瞬时超调，目标rho30%不能解释为每步30%；尚无因果对照证明它导致当前性能。\n\n"
    text+='单种子、不完整语料的八人子集、一个未见攻击者/目标、没有独立佩戴/跨设备控制。辅助全部7来源不是完全未见身份结果。通道0为整句时频统计、通道1教师为中心最多2秒，关系蒸馏不宣称逐帧时间对齐。没有CycleGAN，也不是原完整772维baseline。所有样本处理方法相同，未挑选用户或攻击源。\n\n[完整协议](2026-11-10-abcs-shape-v4-protocol.md)。个人数据、模型、分数和阈值不上传GitHub。\n'
    Path(a.report).write_text(text,encoding='utf8');fig,axes=plt.subplots(1,3,figsize=(11,3.8));methods=['shape40','student192','supplement232'];labels=['Shape40','V4 student192','Shape + V4']
    for ax,k,title in zip(axes,['tar','far','hter'],['Genuine acceptance (%)','FAR (%)','HTER (%)']):
        values=[100*r['results'][m]['unseen_attacker']['Cal_EER'][k] for m in methods];bars=ax.bar(labels,values,color=['#668eaa','#bfc4cb','#168c84']);ax.set_title(title);ax.set_ylim(0,min(100,max(values)*1.2));ax.tick_params(axis='x',rotation=23)
        for bar,value in zip(bars,values):ax.text(bar.get_x()+bar.get_width()/2,value,f'{value:.2f}',ha='center',va='bottom',fontsize=9)
    fig.suptitle('ABCS: channel0 query, channel1 registration teacher (provisional mapping)');fig.tight_layout();fig.savefig(root/'abcs_shape_v4.png',dpi=180);fig.savefig(root/'abcs_shape_v4.pdf');print('REPORT_COMPLETE')
    if a.manifest and (root/'user_results_private.md').exists():
        try:raw=Path(a.manifest).read_text(encoding='utf8')
        except UnicodeDecodeError:raw=Path(a.manifest).read_text(encoding='gb18030')
        rows=json.loads(raw);mapping={int(v['user']):v['speaker'] for v in rows};path=root/'user_results_private.md';private=path.read_text(encoding='utf8')
        for u,name in mapping.items():private=private.replace(f'| ABCS-{u} |',f'| {name}（ABCS-{u}） |')
        path.write_text(private,encoding='utf8')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--report',required=True);p.add_argument('--manifest');main(p.parse_args())
