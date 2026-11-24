"""Anonymous per-user report; uncertainty is clustered by target, not repetitions."""
import argparse,json
from pathlib import Path
import numpy as np

def main(root):
    out=root/'conditional_correspondence/selective_gradient_audit';s=json.loads((out/'summary_public.json').read_text());p=json.loads((out/'protocol_public.json').read_text())
    users=s['per_user'];virtual=s['virtual_updates'];rng=np.random.default_rng(20261124);boot=rng.integers(0,10,(10000,10))
    comparisons=[]
    by={r['method']:r for r in virtual}
    for other in ['identity_only','uniform_correct','selected_wrong']:
        a=np.array([r['check_margin_delta'] for r in by['selected_correct']['per_user']])
        b=np.zeros(10) if other=='identity_only' else np.array([r['check_margin_delta'] for r in by[other]['per_user']])
        delta=a-b;ci=np.quantile(delta[boot].mean(1),[.025,.975])
        comparisons.append(dict(reference=other,mean_margin_delta=float(delta.mean()),target_cluster_ci95=ci.tolist(),improved_users=int((delta>0).sum())))
    supports=all(a['target_cluster_ci95'][0]>0 for a in comparisons)
    decision=('筛选式监督具备训练词内的方向性与虚拟更新证据，可进入固定协议的多步认证对照。' if supports else '尚未获得选择性 Audio 迁移的完整支持：不直接扩大蒸馏训练；先保留本轮诊断，后续只能在新的受控训练/开发协议中检验。')
    lines=['# Audio 身份关系与 ACC 梯度方向审计','',
      '## 本轮回答什么','',
      '只使用当前 10 人的 16 个训练词、1280 条记录与三个纯 ACC 检查点。检查 Audio 的困难身份关系、梯度方向，以及可回滚的一步更新。没有读取最终查询 Audio，没有计算或优化最终 EER/HTER，也没有改写既有模型或创新点一。',
      '', '## 固定实验设计','',
      '每种子 20 个任务，共 60 个任务、600 个目标关系梯度。每个任务为全部 10 人分别取两个不同训练词；同人异词为正，ACC 排名前 3 的异人为困难冒充者。局部关系损失为本人及这三名冒充者的温度 0.1 KL，不要求复制 Audio embedding。',
      '', 'ACC 参照是原跨词 SupCon 身份项；旧训练 CE 分类头未保存，不能计算原完整 CE＋SupCon 梯度。梯度方向在共享卷积 body 参数上测量，实际虚拟更新同时作用于 body 与身份投影层。',
      '', '候选规则：Audio 在偶数/奇数重复两组里都令本人分数高于选中的困难冒充者，并且局部关系梯度与 ACC 全任务身份梯度余弦大于零。正确和错误 Audio 使用相同困难冒充者；selected-wrong 复用正确组的筛选掩码。错误组对 Audio 关系的身份索引同时置换行列，保留本人对角结构。',
      '', f"虚拟更新任务使用前 12 个 fit 词 {p['optimization_words']}，更新后检查另外 4 个 fit 词 {p['virtual_check_words']}。这些词均已被历史编码器训练过，因此后者是未参与本次虚拟更新的词，并不是编码器未见词。", 
      '', '所有虚拟更新从同一原始检查点出发：SGD 0.001，辅助梯度目标为身份梯度范数的 0.1 倍，倍率最大 10。每次更新立即回滚；以 identity-only 同一步更新为参照。检查词只记录结果，不参与筛选。',
      '', '## 身份排序与方向','', '|用户|困难冒充排序 Spearman|Top3 重合率|Audio 胜 ACC 最难冒充者|正确梯度同向率|错 Audio 同向率|入选率|','|---|---:|---:|---:|---:|---:|---:|']
    for r in users:
      lines.append(f"|U{r['user']:02d}|{r['ranking_spearman']:.3f}|{100*r['top3_overlap']:.1f}%|{100*r['audio_beats_acc_hard_rate']:.1f}%|{100*r['gradient_positive_rate']:.1f}%|{100*r['wrong_gradient_positive_rate']:.1f}%|{100*r['selected_rate']:.1f}%|")
    avg=lambda key:float(np.mean([r[key] for r in users]))
    lines+=['',f"总体：排序 Spearman {avg('ranking_spearman'):.4f}，Top3 重合率 {100*avg('top3_overlap'):.2f}%，正确梯度同向率 {100*avg('gradient_positive_rate'):.2f}%，错误 Audio 同向率 {100*avg('wrong_gradient_positive_rate'):.2f}%，入选率 {100*avg('selected_rate'):.2f}%。正确 Audio 平均梯度余弦 {avg('gradient_cosine'):.4f}。",'',
      '排序指标对每次任务的 9 位冒充者计算，再按目标汇总。其机会 Top3 重合率为 3/9≈33.3%。Audio 胜困难冒充者只说明教师自己能判别，不等于 ACC 能学习该关系。',
      '', '## 虚拟更新结果','','负的损失差、正的身份间隔差为好；均相对 identity-only 更新。损失为整个检查任务的 SupCon，不将它复制到多个用户后当成独立证据。','','|方法|跨词检查损失差|本人−最难冒充者间隔差|','|---|---:|---:|']
    for r in virtual:lines.append(f"|{r['method']}|{r['mean_check_loss_delta']:+.8f}|{r['mean_check_margin_delta']:+.8f}|")
    lines+=['','|selected-correct 对照|间隔改善差|按用户重采样95%区间|改善人数|','|---|---:|---|---:|']
    for c in comparisons:
      lo,hi=c['target_cluster_ci95'];lines.append(f"|{c['reference']}|{c['mean_margin_delta']:+.8f}|[{lo:+.8f}, {hi:+.8f}]|{c['improved_users']}/10|")
    lines+=['','## 逐用户虚拟更新','','|用户|方法|检查间隔差|间隔改善任务比例|','|---|---|---:|---:|']
    for u in range(1,11):
      for m in virtual:
        r=next(r for r in m['per_user'] if r['user']==u);lines.append(f"|U{u:02d}|{m['method']}|{r['check_margin_delta']:+.8f}|{100*r['margin_improvement_rate']:.1f}%|")
    lines+=['','## 判断与下一步','',decision,
      '', '梯度同向的意义是局部一阶兼容，而非认证收益。只有在另一批词上的更新也有利，且优于相同掩码的错误 Audio，才值得进入正式多步训练。即便本轮方向有利，也不能宣称最终认证已改善。',
      '', '若继续，唯一建议的候选是：ACC 困难身份局部关系 KL，重复录制稳定性过滤，身份梯度同向门控，辅助梯度小比例上限。正式实验必须重新训练只见优化词的 ACC 起点，保留无 Audio、均匀正确、选择正确、选择错误四组，固定预算并使用独立未见词验证 EER/HTER。不得把本轮历史 fit 检查词当成真正未见词。',
      '', '## 校验','',json.dumps(s['verification'],ensure_ascii=False),
      '', '所有原始权重逐字节保持一致；重建纯 ACC 特征与旧缓存误差小于 1e-4。合成测试覆盖 KL 自匹配静止、正反梯度余弦、SupCon 有限梯度和精确回滚。只公开匿名逐用户汇总，不发布原始信号、embedding、身份映射、逐关系或逐样本分数。']
    (out/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    (out/'decision_public.json').write_text(json.dumps(dict(comparisons=comparisons,decision=decision),ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(decision=decision,virtual=virtual,rank_average={k:float(np.mean([a[k] for a in users])) for k in users[0] if k!='user'}),ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
