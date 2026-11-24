"""Target-cluster descriptive uncertainty for common-subspace audit."""
import argparse,json
from collections import Counter
from pathlib import Path
import numpy as np

def main(root):
    out=root/'conditional_correspondence/linear_shared_identity';read=lambda n:json.loads((out/n).read_text())
    s=read('summary_public.json');chosen=read('selection_public.json');stability=read('stability_public.json')
    rng=np.random.default_rng(20261124);boot=rng.integers(0,10,(10000,10))
    names=['raw_ACC','raw_Audio','rcca_correct','rcca_identity_wrong','lowrank_ridge_correct','lowrank_ridge_identity_wrong']
    comparisons=[];supported=[]
    for method in ['rcca','lowrank_ridge']:
      for scope in ['fit','newwords']:
        a=np.array([next(r['top1'] for r in s['per_user'] if r['method']==method+'_correct' and r['scope']==scope and r['repeats']=='all' and r['user']==u) for u in range(1,11)])
        b=np.array([next(r['top1'] for r in s['per_user'] if r['method']==method+'_identity_wrong' and r['scope']==scope and r['repeats']=='all' and r['user']==u) for u in range(1,11)])
        chance=np.quantile(a[boot].mean(1),[.025,.975]);difference=np.quantile((a-b)[boot].mean(1),[.025,.975])
        comparisons.append(dict(method=method,scope=scope,top1=float(a.mean()),target_cluster_ci95=chance.tolist(),correct_minus_wrong=float((a-b).mean()),difference_ci95=difference.tolist()))
        if scope=='newwords' and chance[0]>.5 and difference[0]>0:supported.append(method)
    decision=('以下方法在映射留出身份与保留词上有超过机会及错配控制的共同结构证据：'+','.join(supported)+'。可作为进一步独立确认的候选，但本轮没有认证增益结论。' if supported else '当前未建立可跨身份推广的低维共同身份结构证据；不进入最终认证或扩大投影网络。该结论限于现有冻结表征与小样本线性协议，不等于原始信号不含共同信息。')
    lines=['# 低容量 Audio–ACC 共同身份子空间审计','',
      '## 核心问题与范围','',
      '不计算最终认证 EER/HTER，不做分数融合。本轮只检查是否存在能够在映射未见身份上保持正确对应的低维共同结构。仅使用 16 个 fit 词的 1280 条 ECAPA Audio 与纯 ACC embedding，所有原始编码器、历史模型和创新点一不变。',
      '', '10 位用户、3 个 ACC 种子、45 组两身份留出，共 135 个外层身份折。每折只用其余 8 位身份训练映射；内部逐个留出 1 位，用 7 人训练、1 人开发，选择维度与正则。两种词方向均检查。重复录制首先汇成每人原型，映射训练有效身份数是 8，最大中心化秩是 7，不是上千个独立身份。',
      '', '## 方法与固定协议','',
      '- 正则化 CCA：训练身份均值中心化、按非零特征值均值缩放 ridge 的协方差白化，再做跨协方差 SVD；无交叉关系的零空间丢弃。',
      '- 低秩线性映射：ridge ACC→Audio 回归，对预测 Audio 截断 SVD；Audio 与 ACC 映射到同一低维坐标后计算余弦。',
      '- 候选维度 1/2/4，ridge 0.1/1/10。只根据 8 个内部身份开发折的平均匹配 AUC 选择；开发 AUC 统一保留 12 位小数消除浮点尾数，再按同分优先低维、更强正则。',
      '- 正确映射与身份错配映射使用相同维度和正则，错配组在外层训练 8 人内部做无固定点身份置换后重新拟合；不把外层身份引入训练。',
      '- 原 ACC、原 Audio 各自的跨词余弦作为可分性参照。Audio-only 只使用 fit 词缓存，不代表部署时使用查询 Audio。',
      '', '12 个映射训练词分为前 6/后 6，构建跨词身份原型；每模态每身份 48 条记录。其余 4 词分为前 2/后 2，每模态每身份 16 条，作为映射保留词。保留词在内部开发身份上参与选超参，因此不是对超参完全盲的词；外层两身份在全部选择中隔离。原 ACC 主干此前见过全部 10 人和 16 词，所以留出只针对线性映射，不是整套系统的开放集。',
      '', '主要匹配任务只在外层两位未见身份中选择（机会 50%）；同时报告十人候选（含 8 位映射见过的身份）的 Top1（机会 10%）。另把每词 8 遍拆为偶数 4 遍与奇数 4 遍检查重复稳定性。1 维余弦可能大量并列，所有 Top1 与 AUC 都按并列均分处理。',
      '', '## 留出身份匹配结果','','全重复原型；数值为十目标宏平均。这里的 AUC 是原型匹配 AUC，不是最终认证 AUC。','','|方法|词范围|两人 Top1%|两人匹配 AUC|十人 Top1%|本人−最难冒充原型间隔|','|---|---|---:|---:|---:|---:|']
    for name in names:
      for scope in ['fit','newwords']:
        r=next(r for r in s['macro'] if r['method']==name and r['scope']==scope and r['repeats']=='all')
        lines.append(f"|{name}|{scope}|{100*r['top1']:.2f}|{r['auc']:.4f}|{100*r['top1_10way']:.2f}|{r['margin']:+.4f}|")
    lines+=['','## 用户簇不确定性','','先将每个用户在不同搭档、方向、种子上的表现平均，再重采样十个用户，未把 45 对身份、多个词和录制当独立受试者。身份搭档仍有共享依赖，区间用于探索性描述。','','|方法|词范围|Top1 95%区间|正确−错配/百分点|差异95%区间/百分点|','|---|---|---|---:|---|']
    for r in comparisons:
      lo,hi=r['target_cluster_ci95'];dl,dh=r['difference_ci95'];lines.append(f"|{r['method']}|{r['scope']}|[{100*lo:.2f}%, {100*hi:.2f}%]|{100*r['correct_minus_wrong']:+.2f}|[{100*dl:+.2f}, {100*dh:+.2f}]|")
    lines+=['','## 跨重复稳定性','','|方法|词范围|偶数重复 Top1%|奇数重复 Top1%|偶/奇身份关系 Spearman|','|---|---|---:|---:|---:|']
    for name in names[2:]:
      for scope in ['fit','newwords']:
        even=next(r for r in s['macro'] if r['method']==name and r['scope']==scope and r['repeats']=='0')['top1']
        odd=next(r for r in s['macro'] if r['method']==name and r['scope']==scope and r['repeats']=='1')['top1']
        rho=np.mean([r['even_odd_relation_spearman'] for r in stability if r['method']==name and r['scope']==scope])
        lines.append(f'|{name}|{scope}|{100*even:.2f}|{100*odd:.2f}|{rho:.4f}|')
    lines+=['','重复关系稳定本身不证明身份对应正确，错误映射也可能稳定；必须同时比较正确身份匹配与错配控制。常数关系的 Spearman 无定义，脚本将其记作 0，不作为相关证据。',
      '', '## 超参数选择','']
    for method in ['rcca','lowrank_ridge']:
      counts=Counter(f"d={r['dimension']},ridge={r['ridge']}" for r in chosen if r['method']==method)
      lines.append(f'- {method}：`{json.dumps(dict(counts),ensure_ascii=False)}`。')
    lines+=['','## 逐用户结果（映射保留词、全部重复）','','|用户|方法|两人 Top1%|十人 Top1%|','|---|---|---:|---:|']
    for user in range(1,11):
      for name in names:
        r=next(r for r in s['per_user'] if r['user']==user and r['method']==name and r['scope']=='newwords' and r['repeats']=='all')
        lines.append(f"|U{user:02d}|{name}|{100*r['top1']:.2f}|{100*r['top1_10way']:.2f}|")
    lines+=['','## 判断','',decision,
      '', '这里寻找的是可迁移的统计子空间，未把某条轴解释为基频、组织传导或其他生理因素；这种物理解释需要独立测量。若仅在映射训练词上成立、保留词不成立，应先解决跨内容稳定性；若两者都不成立，不再通过增大网络将训练身份拟合率当作共享身份信息。',
      '', '无论是否发现线性共同结构，本轮都不声称蒸馏有效或最终身份认证改善。当前数据规模只允许探索性判断；需要真正与编码器训练隔离的新身份确认。',
      '', '## 校验与公开范围','',
      '合成测试先验证已知共享潜变量在真正留出身份上的恢复、身份错配负控制、训练集中心化以及并列处理。实际数据检查 sample_id、词 ID、重复编号及所有外层/内层身份隔离。公开脚本、协议、匿名逐用户汇总；不上传音频、embedding、身份映射、逐样本分数或私有线性投影矩阵。']
    (out/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    result=dict(decision=decision,supported_methods=supported,comparisons=comparisons)
    (out/'decision_public.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(decision=decision,macro=[r for r in s['macro'] if r['repeats']=='all']),ensure_ascii=False))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
