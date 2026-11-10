"""Publish only anonymous aggregates and per-user metrics; never private scores."""
import json,csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
SETS={'Mei9':'mei9_closed_cross3_20261110','ABCS43':'abcs43_closed_cross_20261110','VibraVox59':'vibravox59_closed_cross_20261110'}
CORE=['raw_statistics','no_audio_baseline','audio_baseline','selected_cycle']
LABELS=['Raw statistics','ACC identity only','Audio assisted','Cal-selected CycleGAN']
def p(x):return f'{100*x:.2f}'
def main():
    report=['# 三数据集：闭集跨内容认证','', '所有身份参与学习与负类注册，查询录制独立；这是已知身份的新内容认证，不是未见用户开放集。每用户注册12条、校准8条、查询24条，保留全部其他身份攻击。', '', 'Mei按名义词组隔离（未逐条核验转写）；ABCS/VibraVox按完整句子隔离，不保证词汇不重叠或不同用户说相同句子。ABCS通道0=振动仍未经独立证实。', '', '主工作点：仅由校准数据设置每用户Cal-EER阈值；下表为逐用户宏平均百分比。', '', '|数据集|方法|TAR|FAR|HTER|','|---|---|---:|---:|---:|']
    fig,axes=plt.subplots(1,3,figsize=(13,4),sharey=True)
    for ax,(name,folder) in zip(axes,SETS.items()):
        out=ROOT/'outputs'/folder;r=json.loads((out/'results_public.json').read_text());users=json.loads((out/'per_user_public.json').read_text());verify=json.loads((out/'verification_public.json').read_text());assert verify['state']=='PASS'
        selection=json.loads((out/'selection_public.json').read_text())['selected']
        for point in ['Cal_EER','Cal_FAR5']:
            rows=[v for v in users if v['point']==point]
            with (out/f'per_user_{point}.csv').open('w',newline='',encoding='utf-8') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        lines=[f'# {name}：闭集跨内容逐用户报告','','全部用户均参与身份学习，个人CycleGAN仅用本人12条注册配对微调。全部其他身份作为训练和测试负类。','',f'用户数{r["users"]}；每用户已知攻击来源{r["known_attackers_each"]}；seed42。', '',f'校准锁定的CycleGAN配置：`{json.dumps(selection,ensure_ascii=False)}`。不根据最终查询结果改选配置。','','|方法|工作点|TAR%|FAR%|HTER%|AUC|','|---|---|---:|---:|---:|---:|']
        for method,points in r['results'].items():
            for point,v in points.items():lines.append(f'|{method}|{point}|{p(v["tar"])}|{p(v["far"])}|{p(v["hter"])}|{v["auc"]:.4f}|')
        lines+=['','## 全部用户：Cal-EER工作点','','每格为TAR / FAR / HTER（%）；完整14方法及两个工作点见同目录CSV。','','|用户|原始统计|纯ACC基线|Audio辅助|校准选择CycleGAN|','|---|---|---|---|---|']
        for user in sorted(set(v['user'] for v in users)):
            cells=[]
            for method in CORE:
                v=next(v for v in users if v['user']==user and v['method']==method and v['point']=='Cal_EER');cells.append(' / '.join(p(v[k]) for k in ['tar','far','hter']))
            lines.append('|'+user+'|'+'|'.join(cells)+'|')
        lines+=['','## 循环特征相对Audio基线的逐身份配对变化','','负数表示HTER下降；区间是逐身份bootstrap95%，不能替代训练随机种子。','','|方法|HTER变化pp|95%区间pp|改善用户数|','|---|---:|---|---:|']
        for method,v in r['paired_vs_audio_baseline'].items():lines.append(f'|{method}|{p(v["hter_delta"])}|'+ ' / '.join(p(t) for t in v['identity_bootstrap95'])+f'|{v["improved_users"]}|')
        lines+=['','## 审核与边界','',f'独立重放审核：`{json.dumps(verify,ensure_ascii=False)}`。','','单种子探索结果；Mei为tracking速度而非原始ACC；公开数据为跨句而非词汇互斥。ABCS物理通道归属待确认。查询Audio没有模型前向；注册Audio允许用于教师/个人模型适配。Cal-FAR5是校准目标，不保证查询FAR≤5%。没有跨天/重新佩戴结论，也没有本轮随机GAN对照。','', '[完整协议](../../docs/experiments/2026-11-10-three-dataset-closed-cross-protocol.md)']
        (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        for method in CORE:
            v=r['results'][method]['Cal_EER'];report.append(f'|{name}|{method}|{p(v["tar"])}|{p(v["far"])}|{p(v["hter"])}|')
        vals=[100*r['results'][m]['Cal_EER']['hter'] for m in CORE];ax.bar(range(4),vals,color=['#777777','#387cb0','#24a585','#a66bbc']);ax.set_title(name);ax.set_xticks(range(4),LABELS,rotation=35,ha='right');ax.grid(axis='y',alpha=.2)
        for i,v in enumerate(vals):ax.text(i,v+.15,f'{v:.2f}',ha='center',fontsize=8)
    axes[0].set_ylabel('Macro HTER (%) — lower is better');fig.tight_layout();dest=ROOT/'outputs'/'three_closed_cross_20261110';dest.mkdir(exist_ok=True);fig.savefig(dest/'comparison.png',dpi=180);plt.close(fig)
    report+=['','## 当前结论','','三套均未显示Audio蒸馏相对同预算纯ACC的明确增益。Mei校准选择不加入循环；ABCS选择25步循环编码、权重0.5，HTER下降0.37个百分点，但逐身份95%区间[-1.25,+0.53]跨零。VibraVox的循环选择虽使HTER下降0.69个百分点，区间[-1.84,+0.41]也跨零。', '', 'VibraVox需特别谨慎：Audio基线到循环补充，TAR为89.05%→90.82%，FAR为0.52%→0.91%，AUC为0.9985→0.9969，最差目标FAR仍为16.09%。这说明工作点有变化，不能将HTER改善直接等同为排序或安全性改善。ABCS补充后的最差目标FAR仍为12.10%，平均值不能代替逐用户评估。', '', '本轮支持“已知身份、不同录制/内容可做认证”，尚不支持“Audio辅助稳定改善”或“循环生成增加身份信息”。公开数据只用文本的身份识别仍高于随机（ABCS4.46%对2.33%，VibraVox3.95%对1.69%），不能声称彻底排除内容分配混杂。', '', '当前路线是统计残差编码器＋RKD Audio蒸馏＋个人CycleGAN＋RBF，并非历史完整772维SupCon基线。', '', '## 解读规则','','首先比较同预算纯ACC与Audio辅助；再比较Audio辅助与仅校准选择的CycleGAN。固定分支的查询表现可作诊断，不能替代预先锁定方案。小幅差异不代表稳定提升；逐用户置信区间不能替代多训练种子。闭集优于开放集也不能单归因于内容，因为身份训练覆盖和负类协议同时不同。','','## 逐数据集完整结果','']
    for name,folder in SETS.items():report.append(f'- [{name}全部用户及方法](../../outputs/{folder}/report.md)')
    (ROOT/'docs/experiments/2026-11-10-three-dataset-closed-cross-results.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
if __name__=='__main__':main()
