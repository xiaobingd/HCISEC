"""Anonymous complete B0-B3 authentication report, all users, seeds and backends."""
import json
from pathlib import Path
import numpy as np

OUT=Path('outputs/mei_local_relation_auth_controls_20261117')
NAMES=dict(B0='纯ACC CE',B1='CE＋ACC自关系',B2='CE＋正确Audio局部关系',B3='CE＋错误Audio局部关系')

def main():
    r=json.loads((OUT/'results_public.json').read_text());protocol=json.loads((OUT/'protocol_public.json').read_text())
    compact=json.loads((OUT/'per_user_public.json').read_text());keys=compact['metric_order']
    users=[dict(method=x[0],backend=x[1],condition=x[2],user=x[3],mean=dict(zip(keys,x[4])),std=dict(zip(keys,x[5]))) for x in compact['rows']]
    lookup={(v['method'],v['backend'],v['condition']):v for v in r['summary']}
    def item(method,backend='rbf',condition='crossword'):return lookup[method,backend,condition]
    def pm(v,key,percentage=False):
        scale=100 if percentage else 1
        return f"{scale*v['mean'][key]:.2f}±{scale*v['std'][key]:.2f}"
    a=item('B0')['mean'];b=item('B2')['mean']
    better=all(b['eer']<item(m)['mean']['eer'] and b['hter']<item(m)['mean']['hter'] and b['hard_identity_margin']>item(m)['mean']['hard_identity_margin'] for m in ['B0','B1','B3'])
    conclusion='B2在主后端三个平均指标上均优于三组对照；仍需查看逐seed稳定性，并在新数据确认。' if better else '正确Audio局部关系没有同时改善EER、固定阈值HTER与困难身份间隔，未取得独立认证收益。本轮停止这版局部关系KL方案，不增加权重、不从验证结果重新选邻居或用户。'
    lines=['# B0–B3：Audio局部关系与ACC-only未见词认证（三随机种子）','',
      '## 结论','',conclusion,'',
      f"主RBF跨词：B0 EER={100*a['eer']:.2f}%、HTER={100*a['hter']:.2f}%；B2 EER={100*b['eer']:.2f}%、HTER={100*b['hter']:.2f}%。B2−B0分别为{100*(b['eer']-a['eer']):+.3f}、{100*(b['hter']-a['hter']):+.3f}个百分点。不能把小于种子波动的差值解读为稳定损害，但平均结果没有改善。",
      '', '## 数据、预算和范围','',
      '- 全部九人作为训练身份、认证目标和攻击来源，未按效果选人。编码器训练词0–11每格前6次，共648条；第7–8次共216条只作校准查询；词12–17每格前8次，共432条用于未见词注册/查询。所有方法数据完全相同。',
      '- 重新拟合648条训练ACC的每轴/频点标准化。ReDimNet2-B3冻结，只有同一648条训练Audio供B2/B3监督；校准和验证不使用Audio。没有使用词18–29。',
      '- 三seed42/43/44；每seed随机初始化IdentityEncoder，CE共同预训练400步后复制到四组各继续400步。每组总800步、同结构/同批次序列/同优化器规则，续训AdamW均重置。固定最后一步，不用验证选检查点。',
      '- 结构：CNN32/64/128、GroupNorm、时间均值/标准差256→Linear128→L2。AdamW lr3e-4、wd1e-3、clip1、batch36（九人×两词×两重复）。',
      '- B0仅CE；B1冻结共同预训练ACC教师的自关系；B2真实训练Audio局部关系；B3错误Audio关系但保留B2相同邻居掩码。没有SupCon、生成器或循环损失。创新点一V1及历史模型保持冻结。',
      '- 本表基线是新建纯CE对照，数据量、预算和特征与此前完整特征基线不同，绝对指标不能跨协议直接比较。',
      '- 词12–17此前已查看，本轮属于探索性复核。已知用户/未见词、同会话、名义词序；不称新预留确认集，也不称未见用户开放集。',
      '', '## 局部关系与公平控制','',
      '训练图固定为各身份Top2邻居在全部训练样本、重复前3/后3、偶数/奇数训练词五视图的交集。B1只用ACC训练表示建图，B2/B3用相同训练Audio图。局部行KL温度0.2，只保留同人异词正对和稳定邻居的异人同词负对，不匹配整行其他身份关系。',
      '', 'B3对两批训练词使用不同的非零跨身份移位，保持词和重复序号，破坏真实跨词身份关系。它保留正确Audio掩码，因此主要检验正确关系值的贡献；不能把B3当作完全没有Audio信息的组。',
      '', '辅助项每步按当前编码器梯度范数匹配CE的3%，lambda上限1；三个辅助组同规则，不搜索权重。教师detach，当前编码器确有更新。CE和关系梯度直接相加，分类头保持纯CE梯度。B0与其他组优化更新数相同，辅助组另有计算教师目标梯度的开销。',
      '', '## 注册、阈值和指标','',
      '主后端为balanced RBF SVC(C10,gammaScale)，输入128维ACC表示。每注册词K4，背景八人同词K4；编码器不在注册时更新。余弦原型仅作第二后端诊断，不在结果出来后择优。',
      '', '校准用训练词0–11前4次注册、第7–8次查询，查询未参与编码器训练。按同词/跨词等权分数池选择每目标/方法/seed/后端的Cal-EER阈值，之后锁定用于两条件和两折。没有本人接受率约束。',
      '', '验证为六词、30有向异词组合、前4/后4两折交换；同词与跨词使用完全相同查询。每目标先池化各任务分数得到EER、固定Cal阈值HTER/FAR/FRR，再宏平均九人。测试ROC的EER仅为描述指标，不用于选择运行阈值。',
      '', '困难身份间隔=本人平均分−八个攻击身份中最高平均分；最差样本间隔=最低本人分−最高攻击样本分。两者除以校准攻击分数标准差，保留原始间隔作为尺度核查。正值越大越好，不跨后端解释其绝对大小。',
      '', '## 主RBF结果：三seed均值±标准差','',
      '|方法|条件|EER%|HTER%|TAR%|FAR%|困难身份间隔|最差样本间隔|','|---|---|---:|---:|---:|---:|---:|---:|']
    for method in ['B0','B1','B2','B3']:
        for condition in ['sameword','crossword']:
            v=item(method,'rbf',condition)
            lines.append(f"|{method} {NAMES[method]}|{condition}|{pm(v,'eer',True)}|{pm(v,'hter',True)}|{pm(v,'tar',True)}|{pm(v,'far',True)}|{pm(v,'hard_identity_margin')}|{pm(v,'worst_sample_margin')}|")
    lines+=['','## 余弦诊断后端：完整报告','', '|方法|条件|EER%|HTER%|TAR%|FAR%|困难身份间隔|','|---|---|---:|---:|---:|---:|---:|']
    for method in ['B0','B1','B2','B3']:
        for condition in ['sameword','crossword']:
            v=item(method,'cos',condition)
            lines.append(f"|{method}|{condition}|{pm(v,'eer',True)}|{pm(v,'hter',True)}|{pm(v,'tar',True)}|{pm(v,'far',True)}|{pm(v,'hard_identity_margin')}|")
    lines+=['','## 每个seed的主后端跨词结果','', '|seed|方法|EER%|HTER%|FAR%|困难身份间隔|原始困难间隔|','|---|---|---:|---:|---:|---:|---:|']
    for v in r['per_seed']:
        if v['backend']=='rbf' and v['condition']=='crossword':lines.append(f"|{v['seed']}|{v['method']}|{100*v['eer']:.3f}|{100*v['hter']:.3f}|{100*v['far']:.3f}|{v['hard_identity_margin']:.4f}|{v['raw_hard_identity_margin']:.4f}|")
    lines+=['','## 配对差值：B2减各组（负EER/HTER/FAR、正间隔为改善）','', '|后端|条件|对照|seed|ΔEER百分点|ΔHTER百分点|ΔFAR百分点|Δ困难身份间隔|','|---|---|---|---|---:|---:|---:|---:|']
    for v in r['paired_B2_differences']:
        for s in v['per_seed']:lines.append(f"|{v['backend']}|{v['condition']}|{v['B2_minus']}|{s['seed']}|{100*s['eer']:+.3f}|{100*s['hter']:+.3f}|{100*s['far']:+.3f}|{s['hard_identity_margin']:+.4f}|")
    comparison=next(v for v in r['paired_B2_differences'] if v['backend']=='rbf' and v['condition']=='crossword' and v['B2_minus']=='B0')['per_seed']
    lines+=['',f"B2相对B0的RBF跨词EER改善{sum(v['eer']<0 for v in comparison)}/3种子，HTER改善{sum(v['hter']<0 for v in comparison)}/3，困难身份间隔改善{sum(v['hard_identity_margin']>0 for v in comparison)}/3。全部seed都报告，没有选seed42的较好误差作为最终结论。三个种子是优化重复，不是三个新受试人群；不把重复词对当独立受试者，也不给不恰当的事件级显著性。", '',
      '## 训练关系覆盖及梯度快照','', '|用户|ACC邻居 seed42|ACC邻居 seed43|ACC邻居 seed44|Audio稳定邻居|','|---|---:|---:|---:|---:|']
    graphs=[json.loads((OUT/f'graphs_seed{seed}_public.json').read_text()) for seed in [42,43,44]]
    for u in range(9):lines.append(f"|U{u+1}|{graphs[0]['per_user'][u]['ACC_self_neighbors']}|{graphs[1]['per_user'][u]['ACC_self_neighbors']}|{graphs[2]['per_user'][u]['ACC_self_neighbors']}|{graphs[0]['per_user'][u]['Audio_neighbors']}|")
    lines+=['','没有稳定邻居的用户不作为该局部行损失的锚点，仍可能成为其他锚点的负对，且始终保留CE训练和全部认证角色。图只用训练数据，不根据验证误差删人。',
      '', '|seed|组|加权KD/CE范数中位数%|最大值%|平均方向余弦|平均有效锚点数|','|---|---|---:|---:|---:|---:|']
    gradient=[]
    for seed in [42,43,44]:
        for method in ['B1','B2','B3']:
            trace=json.loads((OUT/f'{method}_seed{seed}_trace_public.json').read_text())
            cos=[v['gradient_cosine'] for v in trace if v['gradient_cosine'] is not None]
            v=dict(seed=seed,method=method,snapshot_median_ratio=float(np.median([v['weighted_gradient_ratio'] for v in trace])),
                snapshot_max_ratio=float(max(v['weighted_gradient_ratio'] for v in trace)),
                snapshot_mean_cosine=float(np.mean(cos)) if cos else None,snapshot_mean_active=float(np.mean([v['active_anchors'] for v in trace])))
            gradient.append(v);co='—' if not cos else f"{v['snapshot_mean_cosine']:.3f}"
            lines.append(f"|{seed}|{method}|{100*v['snapshot_median_ratio']:.3f}|{100*v['snapshot_max_ratio']:.3f}|{co}|{v['snapshot_mean_active']:.1f}|")
    (OUT/'gradient_summary_public.json').write_text(json.dumps(gradient,indent=2)+'\n',encoding='utf-8')
    lines+=['','上表为第1/100/200/300/400步的快照；不冒充全部400步的平均。',
      '', '## 全部九人逐用户结果（三seed均值±标准差）','', '|用户|方法|后端|条件|EER%|HTER%|FAR%|FRR%|困难身份间隔|','|---|---|---|---|---:|---:|---:|---:|---:|']
    for v in sorted(users,key=lambda x:(x['user'],x['backend'],x['method'],x['condition'])):
        lines.append(f"|{v['user']}|{v['method']}|{v['backend']}|{v['condition']}|{pm(v,'eer',True)}|{pm(v,'hter',True)}|{pm(v,'far',True)}|{pm(v,'frr',True)}|{pm(v,'hard_identity_margin')}|")
    lines+=['','## 对研究问题的回答','',
      '这次直接检验认证，而非把关系预测Spearman提升当作认证成功。当前固定局部KL配置未让ACC编码器在未见词中更好区分本人和最困难冒充者；正确Audio没有稳定优于CE、自关系和错误关系。仅有某个seed或余弦后端某个小指标变好，不足以改变这一判断。',
      '', '这不否定训练Audio包含身份信息，而是表明本次挑选的局部相似度分布目标，没有提供对ACC认证有用的额外优化方向。保留这组负结果，不增加权重、不挑用户，也不把本轮词再次用于改规则后的独立确认。',
      '', '若后续提出新机制，应另行预先固定监督目标与真正新数据；本轮不自动追加模型搜索。创新点一继续采用既有V1。',
      '', '## 复现与核查','',
      '源代码：local_relation_control_losses.py、run_local_relation_auth_controls.py、evaluate_local_relation_auth_controls.py、check_local_relation_controls.py、verify_local_relation_auth_controls.py、report_local_relation_auth_controls.py。协议见docs/experiments/2026-11-17-local-relation-auth-controls-protocol.md。',
      '', '顺序：check → run → evaluate → verify → report。score_replay、校准阈值重算、实际训练索引、标准化和12个检查点/缓存一致性见verification_public.json。公开results_public.json、per_user_public.json（metric_order和row_order说明压缩数组）、训练摘要/梯度摘要及本报告；不公开音频、embedding、逐查询分数或真实身份映射。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print('REPORT_COMPLETE',flush=True)

if __name__=='__main__':main()
