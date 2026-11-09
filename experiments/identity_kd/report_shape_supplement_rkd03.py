"""Report the single V4 supplementation route and immutable references."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def main(a):
    root=Path(a.out);r=json.loads((root/'results_public.json').read_text());sel=json.loads((root/'selection_public.json').read_text());text='# RKD-03：40维频谱形状＋V4学生补充\n\n'
    text+='只运行“注册Audio微调ECAPA＋梯度平衡”这一条补充路线；外层24个既有V4模型直接复用，未重训其他方法。新训练仅为Base内部6个V4开发学生及其29身份初始化，用于选择权重。35 Base/12 Cal/12 Test、seed42、K12，每目标24条本人查询、264条来自其他11位未见用户的攻击查询。查询只有振动，注册Audio边界沿用RKD-02。\n\n'
    text+=f"保留40维相对谱形状`s`，拼接192维Audio辅助振动学生`z`：`e=[s;alpha*z]/sqrt(1+alpha²)`。两分支先各自L2归一化，40维没有可学习映射或蒸馏梯度。联合232维建立注册原型，以余弦认证。Base内部锁定alpha={r['alpha']}、rho={r['rho']}，没有根据Test扫权重。\n\n"
    methods=[('shape_only','固定40维谱形状'),('student_only_reused_reference','既有V4学生单独192维'),('supplement','40维＋V4补充232维')]
    for point,title in [('Cal_EER','Cal-EER工作点'),('Cal_FAR5','Cal-FAR5工作点')]:
        text+='## '+title+'\n\n| 表示 | 宏AUC | 宏EER | 本人接受率 | FAR | HTER | 最困难目标FAR |\n|---|---:|---:|---:|---:|---:|---:|\n'
        for key,label in methods:
            v=r[key][point];text+=f"| {label} | {v['macro_auc']:.4f} | {100*v['macro_eer']:.2f}% | {100*v['tar']:.2f}% | {100*v['far']:.2f}% | {100*v['hter']:.2f}% | {100*v['worst_target_far']:.2f}% |\n"
        text+='\n'
    p=r['paired_vs_shape'];text+=f"## 判断\n\n补充相对40维的HTER差值为{100*p['hter_delta']:.3f}个百分点，12身份bootstrap95%区间[{100*p['identity_bootstrap95'][0]:.2f}, {100*p['identity_bootstrap95'][1]:.2f}]个百分点，{p['improved_targets']}/12目标改善。队列平均本人接受率和FAR同时改善，但区间跨零，单种子不足以证明稳定增益。\n\n"
    text+='本轮支持继续研究“保留强统计主干、让学生补充”的路线，不能仅据此证明增益必然由Audio身份知识造成：按要求没有新增其他学生补充路线。不能把队列均值理解为每位目标都改善，也不能把alpha=1解释为忽略40维：此时两条L2分支具有相同权重，40维仍完整存在。\n\n'
    text+='## Base内部选择\n\nalpha0只作参考，正权重候选按HTER、宏AUC、较小alpha依次选择。\n\n| alpha | 开发HTER | 开发宏AUC |\n|---|---:|---:|\n'
    for v in sel['choices']:text+=f"| {v['alpha']} | {100*v['hter']:.2f}% | {v['auc']:.4f} |\n"
    v=r['integrity'];text+=f"\n## 核查\n\n拼接前40维与原分支乘统一常数一致；独立展开两分支内积，最大分数误差{v['independent_branch_formula_max_error']:.3g}；{v['pairwise_auc_workpoint_checks']}项逐目标/工作点pairwise AUC检查通过。Cal-FAR5仅在Cal满足≤5%，不保证Test各目标都满足。原始数据、标准化、注册集和查询集沿用同一固定划分，未删除攻击源、未用Test重选alpha。\n\n[实验协议](2026-11-10-shape-supplement-rkd03-protocol.md)。私有模型、特征、映射、个人分数与阈值不公开。\n"
    Path(a.report).write_text(text,encoding='utf8');fig,axes=plt.subplots(1,3,figsize=(11,3.6));labels=['Shape40','V4 student192','Shape40 + V4']
    for ax,metric,title in zip(axes,['tar','far','hter'],['Genuine acceptance (%)','FAR (%)','HTER (%)']):
        values=[100*r[k]['Cal_EER'][metric] for k,_ in methods];bars=ax.bar(labels,values,color=['#668eaa','#b8bec6','#168c84']);ax.set_title(title);ax.tick_params(axis='x',rotation=22)
        for b,value in zip(bars,values):ax.text(b.get_x()+b.get_width()/2,b.get_height(),f'{value:.2f}',ha='center',va='bottom',fontsize=9)
        ax.set_ylim(0,max(values)*1.15)
    fig.suptitle('VibraVox: unseen users; K=12; seed42; Cal-EER threshold');fig.tight_layout();fig.savefig(root/'shape_supplement_rkd03.png',dpi=180);fig.savefig(root/'shape_supplement_rkd03.pdf');print('REPORT_COMPLETE')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--report',required=True);main(p.parse_args())
