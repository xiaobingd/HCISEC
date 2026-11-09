"""Publish only cohort-level results; require all eight folds."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_abcs_factor_validation import aggregate

def main(a):
    out=Path(a.out);rows={str(u):json.loads((out/f'target{u}_neural_private.json').read_text()) for u in range(1,9)}
    for row in rows.values():
        f=row['fold'];assert not set(f['base'])&set(f['calibration']);assert f['target'] not in f['base']+f['calibration']+f['unseen_attackers']
        for r in row['methods'].values():
            assert r['unseen_attacker']['genuine_n']==r['unseen_attacker']['impostor_n']==24
            assert r['all_seven_sources']['impostor_n']==168
    n=aggregate(rows);ab=json.loads((out/'ablation_summary_public.json').read_text());s='unseen_attacker'
    summary=dict(state='complete',targets=8,results=n,training_macro={name:{k:float(np.mean([r['training'][name][k] for r in rows.values()])) for k in ['fit_pair_auc','dev_pair_auc','selected_step','parameters']} for name in ['stats_mlp','resnet18']},protocol='4Base/2Cal/1target/1unseen;12shot;matched actual query texts;cross-text enrollment;channel0 provisionalBC;seed42;200steps')
    (out/'neural_summary_public.json').write_text(json.dumps(summary,indent=2))
    names=['original_statistics','statistics_pca192','stats_mlp','resnet18'];labels=['Statistics240','PCA192','Statistics MLP','ResNet18']
    fig,ax=plt.subplots(1,2,figsize=(12,4.8));x=np.arange(4);w=.25
    for off,key,label in [(-1,'tar','TAR'),(0,'far','FAR'),(1,'hter','HTER')]:ax[0].bar(x+off*w,[100*n[k][s][key] for k in names],w,label=label)
    ax[0].set_xticks(x,labels,rotation=15);ax[0].set_ylim(0,100);ax[0].set_ylabel('Percent (macro across 8 folds)');ax[0].legend();ax[0].set_title('Unseen-user authentication; Cal-only threshold')
    ks=['all_factors','without_shape','without_amplitude','shape_only','time_only','amplitude_only'];ax[1].barh(np.arange(len(ks)),[ab['results'][k][s]['auc'] for k in ks]);ax[1].set_yticks(np.arange(len(ks)),ks);ax[1].set_xlim(.5,1);ax[1].set_xlabel('AUC');ax[1].set_title('Factor ablation; not equal feature dimensions')
    fig.suptitle('ABCS: channel 0 (provisional BC); matched query prompts, cross-text enrollment',fontsize=11);fig.tight_layout();fig.savefig(out/'abcs_factor_validation.png',dpi=180);fig.savefig(out/'abcs_factor_validation.pdf');plt.close(fig)
    def table(source,keys):
        t=['| 方法 | TAR | FAR | HTER | AUC |','|---|---:|---:|---:|---:|']
        for k in keys:
            r=source[k][s];t.append(f"| {k} | {100*r['tar']:.2f}% | {100*r['far']:.2f}% | {100*r['hter']:.2f}% | {r['auc']:.4f} |")
        return '\n'.join(t)
    text='# ABCS：统计因素与身份编码器复现实验结果\n\n八个目标轮换，4 Base / 2 Cal / 1目标 / 1未见攻击者；每目标12条注册、24条本人查询及24条未见攻击查询。所有人查询相同实际文本，注册与查询文本分离。结果为八折宏平均。通道0暂按 BC，索引尚未独立确认；通道1未融合。\n\n## 统一前端对照\n\n'+table(n,names)+'\n\n## 因素消融\n\n'+table(ab['results'],list(ab['results']))+'\n\n## 查询增益压力测试\n\n| 变换 | TAR | FAR | HTER | AUC |\n|---|---:|---:|---:|---:|\n'
    for k,r in ab['simulated_gain_query_only'].items():text+=f"| {k} | {100*r['tar']:.2f}% | {100*r['far']:.2f}% | {100*r['hter']:.2f}% | {r['auc']:.4f} |\n"
    text+='\n## 解释与限制\n\n频谱形状消融、网络对照必须分别解释。AUC 只衡量排序，不能替代独立阈值下的 FAR / TAR。只有四个 Base 用户，200步、单种子；本轮不能证明深度网络普遍不如统计特征。\n\n通道1为独立控制，不能据其数值推断通道语义。没有可靠重戴、设备或跨天标注，也未测量个体组织传递函数，因此不能把可分性直接归因于生物特征。没有同人同句的独立重复采集，不能冒称完整同内容/跨内容单变量对照。\n\n整句时频表示统一16帧，无原波形拉伸或零填充；中心1.25秒仅作统计控制。125共同文本中124句参与实验，五用途文本分离，992条录制通过 CRC / 精确重复检查。\n\n完整协议：[实验设计](2026-11-09-abcs-factor-validation-protocol.md)。私有数据、映射、逐人分数与阈值不公开。\n'
    Path(a.report).write_text(text,encoding='utf-8');print(json.dumps(summary,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--report',required=True);main(p.parse_args())
