"""Report cross-modal retrieval and registration benefit separately."""
import argparse,json
from collections import Counter
from pathlib import Path
import numpy as np

def main(root):
    out=root/'conditional_correspondence/crossmodal_identity_projection';read=lambda n:json.loads((out/n).read_text())
    s=read('summary_public.json');alphas=read('alpha_selection_public.json');ret=read('retrieval_public.json')
    methods=['raw_ACC','ACC_ACC_fusion','Audio_correct','Audio_other','Audio_shuffled','AudioModel_ACCprototype','Audio_other_Bthreshold','Audio_shuffled_Bthreshold']
    families=['english_word','english_long','chinese_long'];rng=np.random.default_rng(20261124);boot=rng.integers(0,10,(10000,10))
    comparisons=[]
    for control in [m for m in methods if m!='Audio_correct']:
      for metric in ['eer','hter']:
        delta=[]
        for us in range(1,11):
          b=np.mean([r[metric] for r in s['per_user'] if r['method']=='Audio_correct' and r['user']==us])
          a=np.mean([r[metric] for r in s['per_user'] if r['method']==control and r['user']==us]);delta.append(100*(b-a))
        delta=np.array(delta);ci=np.quantile(delta[boot].mean(1),[.025,.975])
        comparisons.append(dict(control=control,metric=metric,delta_percentage_points=float(delta.mean()),target_cluster_ci95=ci.tolist(),improved_users=int((delta<0).sum())))
    retrieval=[]
    for us in range(1,11):
      rr=[r for r in ret if r['user']==us];retrieval.append(dict(user=us,**{k:float(np.mean([r[k] for r in rr])) for k in ['top1','crossmodal_target_auc','same_query_retrieval_auc','genuine_minus_hard_margin']}))
    retmacro={k:float(np.mean([r[k] for r in retrieval])) for k in retrieval[0] if k!='user'}
    top=np.array([r['top1'] for r in retrieval]);retci=np.quantile(top[boot].mean(1),[.025,.975]).tolist()
    counts={v:{str(k):n for k,n in sorted(Counter(r['alpha'] for r in alphas if r['variant']==v).items())} for v in ['crossmodal','acc_only']}
    passed=all(r['target_cluster_ci95'][1]<0 for r in comparisons if r['control'] in ['raw_ACC','ACC_ACC_fusion','Audio_other','Audio_shuffled'])
    decision=('正确注册 Audio 同时超过纯 ACC、容量匹配和错配控制，建议进入独立录制确认。' if passed else '本轮未通过 Audio 独立认证增益标准；不继续在现有查询词上搜索融合权重。跨模态检索结果与最终认证结果分开解释。')
    lines=['# 跨词跨模态身份投影与 Audio 注册原型实验','',
      '## 方法与数据','',
      '冻结现有纯 ACC 192 维编码器与官方 ECAPA 192 维 Audio 编码器，仅训练两个 192→64→32 投影头（GELU，L2 归一化）。双向身份监督对比学习，温度 0.1。同一批中 Audio 全部来自同一训练词，ACC 全部来自另一个训练词；同人跨词为正、异人同词来源为负。没有复制教师相似度矩阵。',
      '', '全部 10 人保留、种子 42/43/44。每折留出目标及下一位循环身份；其余 8 人训练投影 400 步。ACC→ACC 对照使用相同两投影头、初始化、episode、优化预算，把 Audio 输入替换为相应词的 ACC。共 60 个模型。',
      '', '投影优化只使用前 12 个 fit 词；其余 4 个 fit 词用于内部注册、校准和开发：前两词各 6 条支持，第三词 8 条本人＋64 条攻击校准，第四词 8＋64 条选择融合权重。开发本人身份未参与投影优化；最终目标身份不参与此权重选择。权重候选为 0/0.1/0.25/0.5/1，按内部 HTER、EER、更小权重排序。该内部样本量较小，权重选择存在不确定性。',
      '', '实际注册沿用旧 acc_crossword12：4 个 REG 词、每词 3 条，共 12 条相同 ACC，Audio 也取对应 12 条。没有修改注册样本排序。最终查询为单条 ACC，未读取查询 Audio；注册词、CAL 词与三个查询内容组的词 ID 完全分离。',
      '', '分数＝原 ACC 余弦＋α×查询 ACC 投影与注册 Audio 投影原型的余弦。异人 Audio、192 坐标打乱 Audio 使用相同正确模型和 α；同时报告分别校准阈值及沿用正确 Audio 阈值的对照。AudioModel_ACCprototype 使用同一 Audio 联合训练的 ACC 投影头，将原型换成 ACC，α 保持相同。',
      '', '240 个 CAL 阈值全部锁定后才评分最终查询。阈值规则沿用本人 Q10 与冒充者 Q95 中点。ACC 主干此前见过所有 10 人与全部 16 个 fit 词，因此这不是完整未见身份开放集，也不将内部 4 词称为编码器未见词。',
      '', '## 第一层：跨模态检索','',
      f"CAL 的目标 ACC 查询在 10 个注册 Audio 原型中 Top1 为 {100*retmacro['top1']:.2f}%（随机期望 10%；目标簇 bootstrap 95% 区间 [{100*retci[0]:.2f}%, {100*retci[1]:.2f}%]）。目标身份跨模态验证 AUC={retmacro['crossmodal_target_auc']:.4f}；同一查询在本人/异人原型间检索 AUC={retmacro['same_query_retrieval_auc']:.4f}。CAL 只用于此描述和固定规则阈值，不用于选择投影模型或 α。",'',
      '|用户|Top1%|目标验证 AUC|同查询检索 AUC|本人−最难冒充原型间隔|','|---|---:|---:|---:|---:|']
    for r in retrieval:lines.append(f"|U{r['user']:02d}|{100*r['top1']:.2f}|{r['crossmodal_target_auc']:.4f}|{r['same_query_retrieval_auc']:.4f}|{r['genuine_minus_hard_margin']:+.4f}|")
    diagnostic=out/'fit_diagnostics_public.json'
    if diagnostic.exists():
      fit=json.loads(diagnostic.read_text())
      lines+=['','### 事后拟合诊断（不用于选择模型或权重）','','|范围|Top1|原型候选/支持量|','|---|---:|---|']
      labels={'train_identity_fit_word_disjoint':'投影训练身份、训练词内跨词','held_identity_fit_word_disjoint':'留出目标身份、训练词内跨词','inner_dev_identity_inner_words':'留出开发身份、内部开发词跨词'}
      for a in fit['macro']:
        sample=next(r for r in fit['rows'] if r['scope']==a['scope'])
        lines.append(f"|{labels[a['scope']]}|{100*a['top1']:.2f}%|{sample['candidate_users']} 人；每人 {sample['support_records_per_user']} 条 Audio 支持|")
      lines+=['','训练身份上的跨词匹配达到 100%，但留出身份与保留注册/CAL 词上的匹配弱。这支持小投影头拟合了训练身份对应、泛化不足的解释。上述事后检查的候选人数与支持预算不完全相同，不能将所有差值只归因于词汇；它们也不能替代最终固定 12 条注册的认证结果。']
    lines+=['','## 第二层：ACC-only 单条跨内容认证','','三种子、十目标宏平均。EER 与 HTER 以及 FAR/FRR 均为百分比。','','|方法|查询内容|EER|HTER|FAR|FRR|AUC|','|---|---|---:|---:|---:|---:|---:|']
    for m in methods:
      for f in families:
        r=next(r for r in s['macro'] if r['method']==m and r['family']==f)
        lines.append(f"|{m}|{f}|{100*r['eer']:.3f}|{100*r['hter']:.3f}|{100*r['far']:.3f}|{100*r['frr']:.3f}|{r['auc']:.5f}|")
    lines+=['','## 配对不确定性','','正确 Audio−对照，负值为好；每位目标先平均三种子、三种内容，再按目标重采样。重复发声不视为独立受试者。','','|对照|指标|差/百分点|95%区间|改善人数|','|---|---|---:|---|---:|']
    for r in comparisons:
      lo,hi=r['target_cluster_ci95'];lines.append(f"|{r['control']}|{r['metric']}|{r['delta_percentage_points']:+.4f}|[{lo:+.4f}, {hi:+.4f}]|{r['improved_users']}/10|")
    lines+=['','## 权重选择与判断','',f'30 折权重计数：`{json.dumps(counts)}`。α=0 表示开发数据未选择 Audio 分数，最终与原 ACC 分数完全相同，不能计作 Audio 收益。','',decision,
      '', '跨模态检索好，只能说明小投影头能够学习身份对应；认证必须同时超过原 ACC、ACC→ACC 容量对照和错误注册 Audio，才支持独立收益。第一层指标高不替代第二层。',
      '', '## 逐用户认证结果（三种子平均）','','|用户|查询内容|方法|EER%|HTER%|FAR%|FRR%|','|---|---|---|---:|---:|---:|---:|']
    for us in range(1,11):
      for f in families:
        for m in methods[:6]:
          r=next(r for r in s['per_user'] if r['user']==us and r['family']==f and r['method']==m)
          lines.append(f"|U{us:02d}|{f}|{m}|{100*r['eer']:.3f}|{100*r['hter']:.3f}|{100*r['far']:.3f}|{100*r['frr']:.3f}|")
    lines+=['','## 校验与公开边界','',
      '独立检查 12000 个训练任务的目标/开发身份隔离、跨词采样与标签对应，复核 60 次 α 选择、240 阈值和 720 个评估工作点；原 ACC 对照必须与历史同口径完全一致。所有匿名逐用户逐种子数据附在 JSON。',
      '', '当前为反复查看过的既有队列。44100 Hz Audio 解释与物理延迟未独立确认；约 3 秒 ACC 窗口可能含相邻重复发声。两个编码器、创新点一和历史结果均保持不变。只发布代码与匿名汇总，不上传 Audio、embedding、模型权重、身份映射、采样索引、逐样本分数或实际阈值。']
    (out/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    result=dict(decision=decision,alpha_counts=counts,retrieval_macro=retmacro,retrieval_top1_ci95=retci,paired_comparisons=comparisons)
    (out/'decision_public.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(decision=decision,alpha_counts=counts,retrieval_macro=retmacro,macro=s['macro'][:9]),ensure_ascii=False))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
