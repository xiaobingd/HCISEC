"""Cohort-level report for registration-Audio ECAPA and gradient balancing."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
LABELS={'S0_shape':'固定谱形状','R_frozen':'原冻结学生','V0':'仅振动个人适配','V1':'冻结ECAPA＋旧λ','V2':'冻结ECAPA＋梯度平衡','V3':'微调ECAPA＋旧λ','V4':'微调ECAPA＋梯度平衡','V5':'错误身份＋梯度平衡','V6':'无Audio几何控制'}
def main(a):
    root=Path(a.out);r=json.loads((root/'results_public.json').read_text());assert r['integrity']['state']=='pass';keys=[k for k in LABELS if k in r['results']];text='# RKD-02：注册者Audio微调与梯度平衡结果\n\n'
    text+='协议已改变：注册可用本人12条同步Audio+振动，查询只有振动。35 Base、12 Cal、12 Test，seed42；每目标24条本人查询及264条未见攻击查询。注册与查询文本互斥，所有用户保留。不能与“注册也没有Audio”的RKD-01混称同一任务。\n\n'
    text+='ECAPA embedding backbone全部参数可训练，固定BN统计，30步，LR1e-6，Base回放与原教师锚定；之后冻结供个人学生使用。各个人学生100步、相同初始化/采样，保留Base回放与原学生锚定。ρ由Base内部模拟注册选择为'+str(r['rho'])+'，不使用最终用户效果挑选。\n\n'
    q=r['results'];text+='## 本轮判断\n\n'
    text+=f"振动个人适配本身有效果：原冻结学生HTER={100*q['R_frozen']['Cal_EER']['hter']:.2f}%，仅振动个人适配={100*q['V0']['Cal_EER']['hter']:.2f}%。不能将这部分收益归给Audio。\n\n"
    text+=f"梯度平衡使冻结教师HTER从{100*q['V1']['Cal_EER']['hter']:.2f}%降至{100*q['V2']['Cal_EER']['hter']:.2f}%，但目标身份级区间仍跨零。注册Audio微调没有提供额外收益：旧权重下V3与V1决策指标相同；梯度平衡下V4为{100*q['V4']['Cal_EER']['hter']:.2f}%，略差于冻结教师V2。\n\n"
    text+=f"完整组合相对无Audio个人适配只改善{100*(q['V0']['Cal_EER']['hter']-q['V4']['Cal_EER']['hter']):.2f}个百分点，且对错误身份/无Audio几何控制的优势均不确定。固定谱形状参考HTER={100*q['S0_shape']['Cal_EER']['hter']:.2f}%，仍优于所有学生。本轮未证明Audio知识带来稳定增益；也不能据此否定更强或经独立开发选择的Audio域适配。\n\n"
    for point,title in [('Cal_EER','Cal-EER工作点'),('Cal_FAR5','Cal-FAR5工作点')]:
        text+='## '+title+'\n\n| 方法 | 宏AUC | 宏EER | TAR | FAR | HTER | 最困难目标FAR |\n|---|---:|---:|---:|---:|---:|---:|\n'
        for k in keys:
            v=r['results'][k][point];text+=f"| {LABELS[k]} | {v['macro_auc']:.4f} | {100*v['macro_eer']:.2f}% | {100*v['tar']:.2f}% | {100*v['far']:.2f}% | {100*v['hter']:.2f}% | {100*v['worst_target_far']:.2f}% |\n"
        text+='\n'
    text+='## 分离两种改动\n\n下列差值为前者减后者，负值表示HTER降低。目标身份级bootstrap5000次，只有12位Test、单种子。\n\n| 对比 | HTER差值(pp) | 身份bootstrap95%区间(pp) | 改善目标数/12 |\n|---|---:|---|---:|\n'
    for k,v in r['paired'].items():text+=f"| {k} | {100*v['hter_delta']:.2f} | [{100*v['identity_bootstrap95'][0]:.2f}, {100*v['identity_bootstrap95'][1]:.2f}] | {v['improved_targets']} |\n"
    text+='\nV2−V1检验冻结教师下提高梯度影响的作用；V3−V1检验旧权重下微调教师的增量；V4−V2检验相同梯度策略下注册Audio微调的增量；V4−V5/V6检查真实Audio身份知识相对错误身份和无Audio几何的增量。\n\n'
    text+='## 实际梯度与权重\n\n目标ρ不等于每步实现比例。下表为24个外层个人模型第1/25/50/100步的摘要，梯度比为裁剪前加权RKD/身份梯度范数。\n\n| 方法 | λ平均 | 实际梯度比平均 | 实际梯度比范围 |\n|---|---:|---:|---|\n'
    for m,v in r['gradient_audit'].items():text+=f"| {LABELS[m]} | {v['lambda_mean']:.3f} | {100*v['realized_gradient_ratio_mean']:.2f}% | [{100*v['realized_gradient_ratio_range'][0]:.2f}%, {100*v['realized_gradient_ratio_range'][1]:.2f}%] |\n"
    text+='\n## 核查与边界\n\n'
    audit=r['integrity'];text+=f"已核查24个独立ECAPA，主体参数更新范围{audit['teacher_parameter_update_range']}。重置后的Base原教师特征与RKD-01外部教师最小余弦={audit['frozen_base_teacher_min_cosine_vs_original']:.6f}，防止前一注册者权重串入下一人。所有教师输入索引只含Base-fit及本人registration，查询Audio读取为0。\n\n"
    text+=f"逐目标/工作点pairwise AUC及Cal-FAR5检查{audit['pairwise_auc_workpoint_checks']}项通过；抽查{audit['inference_models_spot_checked']}个仅振动推理模型，最大分数差={audit['max_spot_inference_score_difference']}。固定阈值来自独立Cal，不以Test重调。\n\n"
    text+='单种子、一次身份划分、同一语料采集条件；没有重新佩戴/跨设备验证。VibraVox用户间文本并不一致，不能充当ABCS异人同句实验。没有CycleGAN；这是统计特征上的注册适配/蒸馏实验。只有同时超过纯振动个人适配、错误身份和无Audio控制，才可讨论Audio身份知识的增量。\n\n[完整协议](2026-11-10-registration-rkd02-protocol.md)。原始音频、特征、个人模型、分数及阈值保持私有。\n'
    Path(a.report).write_text(text,encoding='utf8');fig,ax=plt.subplots(1,2,figsize=(13,4.5));xx=np.arange(len(keys));w=.25
    for offset,metric,label in [(-1,'tar','TAR'),(0,'far','FAR'),(1,'hter','HTER')]:ax[0].bar(xx+offset*w,[100*r['results'][k]['Cal_EER'][metric] for k in keys],w,label=label)
    ax[0].set_xticks(xx,keys,rotation=30);ax[0].set_ylim(0,100);ax[0].legend();ax[0].set_ylabel('%');ax[0].set_title('12-shot; Cal-EER threshold; 12 unseen Test users')
    gradkeys=list(r['gradient_audit']);ax[1].bar(gradkeys,[100*r['gradient_audit'][k]['realized_gradient_ratio_mean'] for k in gradkeys]);ax[1].axhline(100*r['rho'],color='black',linestyle='--',label='Target ratio');ax[1].set_ylabel('Weighted RKD / identity gradient norm (%)');ax[1].set_title('Observed gradient ratio; outer registration adaptation');ax[1].legend();fig.tight_layout();fig.savefig(root/'registration_rkd02.png',dpi=180);fig.savefig(root/'registration_rkd02.pdf');print('REPORT_COMPLETE')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--report',required=True);main(p.parse_args())
