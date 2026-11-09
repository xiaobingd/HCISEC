"""Aggregate Mei9 V4 report; individual results stay private/local."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
LABELS={'shape120':'固定三轴谱形状120维','statistics720':'完整统计720维','student192':'微调教师＋梯度平衡学生192维','supplement312':'谱形状＋学生补充312维'}
def main(a):
    root=Path(a.out);r=json.loads((root/'results_public.json').read_text());assert r['integrity']['state']=='pass';text='# Mei9：谱形状保留＋Audio辅助学生补充\n\n'
    text+='九人全部完成，seed42；每折4 Base/2 Cal/1目标/2未见攻击者，九折均衡轮换，未剔除任何用户或预定来源。每人12条注册，24条本人查询、48条未见攻击查询。注册允许本人Audio+振动，查询只有振动。\n\n'
    text+='只训练注册Audio微调ECAPA＋梯度平衡的V4路线。Mei是三轴，每轴保留40维，共120维固定谱形状；追加192维学生，形成312维。没有把三轴压成40维，没有学习新的融合投影。alpha1、rho0.3直接沿用VibraVox开发值，未在Mei测试上选择。固定120维、720维统计仅作评分参考，不是额外训练方法。\n\n'
    text+='## 主要结果：两个未见攻击来源\n\n词标签尚未逐录制验证，以下同词/跨词均为名义标签条件。每折/方法使用同词＋跨词Cal分数确定共同阈值。\n'
    for point,title in [('Cal_EER','Cal-EER工作点'),('Cal_FAR5','Cal-FAR5工作点')]:
        text+='\n### '+title+'\n\n| 条件 | 方法 | 本人接受率 | FAR | HTER | 宏AUC | 宏EER |\n|---|---|---:|---:|---:|---:|---:|\n'
        for c,label in [('sameword','同词异重复'),('crossword','跨词')]:
            for m,name in LABELS.items():
                v=r['results'][m]['two_unseen'][c][point];text+=f"| {label} | {name} | {100*v['tar']:.2f}% | {100*v['far']:.2f}% | {100*v['hter']:.2f}% | {v['auc']:.4f} | {100*v['eer']:.2f}% |\n"
    text+='\n## 判断\n\n'
    same=r['paired']['sameword'];cross=r['paired']['crossword'];text+=f"同词补充相对固定谱形状HTER降低{-100*same['hter_delta']:.2f}个百分点，{same['improved_targets']}/9目标改善；目标级bootstrap95%差值区间[{100*same['identity_bootstrap95'][0]:.2f}, {100*same['identity_bootstrap95'][1]:.2f}]个百分点。\n\n"
    text+=f"跨词HTER差值为{100*cross['hter_delta']:+.2f}个百分点，{cross['improved_targets']}/9改善，区间[{100*cross['identity_bootstrap95'][0]:.2f}, {100*cross['identity_bootstrap95'][1]:.2f}]跨零；**没有复现VibraVox的跨内容增益**。跨词完整720维统计HTER为{100*r['results']['statistics720']['two_unseen']['crossword']['Cal_EER']['hter']:.2f}%，优于补充路线。不能只与120维比较后宣称超过完整统计基线。\n\n"
    b=r['results']['shape120']['two_unseen']['crossword']['Cal_EER'];v=r['results']['supplement312']['two_unseen']['crossword']['Cal_EER'];text+=f"跨词AUC从{b['auc']:.4f}降至{v['auc']:.4f}，提示退化不仅是阈值工作点；两位Cal的样本量及个人分数漂移可能同时参与，但本轮未分离因果。部分目标改善不能抵消其他目标退化，不能按最终结果挑人。\n\n"
    text+='同词下的收益不等于内容无关身份表征。与VibraVox相比，同时变化了传感器物理量、三轴频带、说话时长/内容、Base/Cal规模及折划分，不能把性能差距只归因于某一个因素。\n\n'
    text+='## 辅助结果：全部八个来源\n\n该范围包含Base/Cal，不能当作完全未见攻击实验。\n\n| 条件 | 方法 | TAR | FAR | HTER | AUC |\n|---|---|---:|---:|---:|---:|\n'
    for c,label in [('sameword','同词'),('crossword','跨词')]:
        for m,name in LABELS.items():
            v=r['results'][m]['all_eight_mixed'][c]['Cal_EER'];text+=f"| {label} | {name} | {100*v['tar']:.2f}% | {100*v['far']:.2f}% | {100*v['hter']:.2f}% | {v['auc']:.4f} |\n"
    q=r['integrity'];text+=f"\n## 核查与限制\n\n2695条精确时间戳配对，27个独立个人教师均发生主体更新；Base教师重置特征最大差={q['teacher_reset_base_embedding_max_difference']}。Audio前向仅限144个跨折合法fit/注册索引，与同词/跨词查询索引无交叉。27个仅振动推理模型、{q['pairwise_auc_workpoint_checks']}项逐目标/工作点指标及所有汇总复算通过，最大分数误差{q['max_score_recompute_error']:.3g}。梯度比抽查平均{100*q['realized_gradient_ratio_mean']:.2f}%，范围[{100*q['realized_gradient_ratio_range'][0]:.2f}%,{100*q['realized_gradient_ratio_range'][1]:.2f}%]；目标rho30%不等于每步实际比例。\n\n"
    text+='实际信号为OpenVR vVelocity，非原始IMU。词ID依赖录制顺序推断，缓存Audio为左通道16kHz/2秒float16派生数据；统一前端继承原审核，不声称重新核实全部语义或佩戴。单头显、无重戴/跨天/跨设备控制、单种子、重叠Base集合，目标bootstrap仅为本实验描述性不确定性。Cal-FAR5只约束同词/跨词汇总Cal，不保证Test或每个条件的FAR≤5%。\n\n与旧八用户环＋固定第九攻击者不同，本轮九折均衡，阈值/注册/查询也重新匹配；不能与旧报告绝对数值拼接归因。无CycleGAN、无完整772维旧baseline。逐用户表仅保留本地私有报告，GitHub不发布个人分数、模型、映射或阈值。[完整协议](2026-11-10-mei9-shape-v4-protocol.md)。\n'
    Path(a.report).write_text(text,encoding='utf8');fig,axes=plt.subplots(1,2,figsize=(10,4));labels=['Shape120','Statistics720','V4 student192','Shape + V4']
    for ax,c,title in zip(axes,['sameword','crossword'],['Same nominal words, different repetitions','Cross nominal words']):
        values=[100*r['results'][m]['two_unseen'][c]['Cal_EER']['hter'] for m in LABELS];bars=ax.bar(labels,values,color=['#668eaa','#aab5c5','#c3c7cc','#168c84']);ax.set_ylabel('HTER (%)');ax.set_ylim(0,max(values)*1.2);ax.set_title(title);ax.tick_params(axis='x',rotation=22)
        for bar,value in zip(bars,values):ax.text(bar.get_x()+bar.get_width()/2,value,f'{value:.2f}',ha='center',va='bottom',fontsize=9)
    fig.suptitle('Mei9: all 9 targets; 4 Base / 2 Cal / 1 target / 2 unseen attackers');fig.tight_layout();fig.savefig(root/'mei9_shape_v4.png',dpi=180);fig.savefig(root/'mei9_shape_v4.pdf');print('REPORT_COMPLETE')
    if a.manifest and (root/'user_results_private.md').exists():
        mf=json.loads(Path(a.manifest).read_text(encoding='utf8'));mapping={int(rec['user_id']):rec['user'] for rec in mf['records']};p=root/'user_results_private.md';private=p.read_text(encoding='utf8')
        for u,name in mapping.items():private=private.replace(f'| U{u} |',f'| {name}（raw9-{u}） |')
        private=private.replace('编号沿用原始raw9身份映射。','编号沿用原始raw9身份映射，不保证与旧trainb数组的U编号一致。');p.write_text(private,encoding='utf8')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--report',required=True);p.add_argument('--manifest');main(p.parse_args())
