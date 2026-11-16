"""Chinese report from locked, anonymous predictability aggregates only."""
import json
from pathlib import Path
from evaluate_relation_predictability import OUT

NAMES={'ACC_similarity':'直接ACC相似度','ACC_fixed_prior':'ACC训练身份关系均值',
 'Audio_fixed_identity_memory':'Audio固定身份对记忆（诊断）','ACC_self_Ridge':'同容量ACC自回归Ridge',
 'Audio_Ridge':'Audio监督Ridge（主方法）','Audio_Ridge_no_word_type':'Audio Ridge去词类特征',
 'Audio_residual_Ridge':'Audio身份记忆＋残差Ridge','Audio_Ridge_command_only':'仅训练指令词10/11的Ridge',
 'Audio_Ridge_heldout_dyad':'留身份对Audio监督Ridge','Audio_Ridge_heldout_endpoint':'留目标用户Audio监督Ridge'}

def main():
    r=json.loads((OUT/'results_public.json').read_text());p=json.loads((OUT/'protocol_public.json').read_text())
    c=json.loads((OUT/'identity_permutation_public.json').read_text())['primary_two_methods']
    methods=r['methods'];a=methods['ACC_similarity']['crossword'];b=methods['Audio_Ridge']['crossword'];m=methods['Audio_fixed_identity_memory']['crossword']
    lines=['# 新ACC编码器的Audio身份关系可预测性：词0–11训练、词12–17独立验证','',
      '## 结论','',
      f"重新训练后，直接ACC的跨词困难负类排序相关为{a['task_spearman_macro']:.3f}。训练词Audio监督将其提高到{b['task_spearman_macro']:.3f}，说明整体Audio排序具有部分可预测性；但固定训练身份对记忆达到{m['task_spearman_macro']:.3f}，而主方法最困难冒充者Top1由{a['top1']:.2%}降到{b['top1']:.2%}。因此本轮不进入局部困难关系蒸馏：整体排序改善尚未转化为最关键困难关系改善，也未超过固定身份记忆。",
      '', '## 实际训练与隔离','',
      f"- 全部九人保留。随机初始化新的IdentityEncoder，词0–11共{p['fit_records']}条参与CE训练；词12–17共{p['validation_records']}条完全不参加编码器训练或标准化拟合。",
      '- 模型结构沿用CNN32/64/128、GroupNorm、频率下采样、时间mean/std256→Linear128→L2。固定seed42、1400步、AdamW3e-4/wd1e-3、batch36、梯度裁剪1；使用最后一步，不用独立验证集选检查点。没有加载旧E0，没有Audio损失。',
      '- 关系预测阶段固定每人每词前8条：训练864条，验证432条。4条注册、4条查询，两折交换后平均关系。预测器只用训练词0–11的冻结ReDimNet2-B3 Audio关系标签，所有模型预测锁定后才使用12–17的Audio标签。没有读取词18–29的Audio。',
      '- 词/重复标签沿用历史名义映射；不是转写核验。规范化统计重新拟合0–11，并在评估脚本独立重算核对。编码器首层有非零梯度、参数确实变化。创新点一V1及历史模型/结果不改动。',
      '- 这是已知九人、未见词检验。留目标Audio监督只约束关系预测器，ACC主干仍见过九人的训练词，不是未见用户开放集。',
      '', '## 比较对象与固定信息约束','',
      '主方法为14维ACC关系统计→StandardScaler→Ridge(alpha10)。输入包含当前ACC相似度、训练ACC身份关系先验、残差、两端行列难度统计及本人相似度、同词/词类标记；不输入用户或词ID one-hot。所有标准化和回归拟合仅用允许的训练关系。',
      '', '关系原型按真实身份离线汇总；本人相似度、身份对先验等是审计统计。此预测器不是可直接部署的在线认证器，也没有使用未知查询Audio。Audio固定身份对记忆使用训练期身份查表，仅用于检验九人固定排序捷径。',
      '', '同容量ACC自回归Ridge不使用任何Audio；训练身份置换保持模型容量和输入不变，仅改变Audio监督。身份对留出排除该无向身份对两个方向所有监督；目标留出排除涉及该用户任一端的所有Audio监督。',
      '', '## 独立验证词身份认证诊断','', '|模态|同词AUC|跨词AUC|','|---|---:|---:|']
    for row in r['independent_word_authentication']:lines.append(f"|{row['modality']}|{row['macro']['sameword']:.4f}|{row['macro']['crossword']:.4f}|")
    lines+=['','这些是真正未参与当前ACC训练的词。上一轮E0在0–17训练词上得到的0.9952跨词AUC不能与其当作同协议改进比较；上一轮只证明训练表示关系，本轮修正了这一暴露问题。',
      '', '## 困难冒充者排序预测','',
      '每任务固定一个注册词、一个查询词和目标用户，对8个冒充身份排序，与真实验证Audio排序计算Spearman；先平均词任务，再平均九人。Top1为最困难冒充者是否相同；随机期望12.5%。汇总身份ρ先平均全部跨词任务再比较身份排序。残差ρ在每个固定有向身份对内部去掉验证词任务均值，不让固定身份排序产生虚假任务预测能力。',
      '', '|方法|同词任务ρ|跨词任务ρ|跨词Top1|跨词汇总身份ρ|身份对内任务残差ρ|','|---|---:|---:|---:|---:|---:|']
    for name,v in methods.items():
        x=v['crossword'];res='—（恒定预测）' if x['residual_spearman_macro'] is None else f"{x['residual_spearman_macro']:.3f}"
        lines.append(f"|{NAMES[name]}|{v['sameword']['task_spearman_macro']:.3f}|{x['task_spearman_macro']:.3f}|{x['top1']:.2%}|{x['aggregate_identity_ranking_spearman']:.3f}|{res}|")
    lines+=['','## 稳健性与负控制','', '|方法|跨词ρ词块95%区间|相对直接ACC的差值95%区间|相对固定Audio身份记忆的差值95%区间|','|---|---|---|---|']
    def interval(x):return f'[{x[0]:.3f}, {x[1]:.3f}]'
    for name in ['ACC_similarity','ACC_self_Ridge','Audio_Ridge','Audio_Ridge_heldout_dyad','Audio_Ridge_heldout_endpoint']:
        x=r['word_block_intervals'][name]
        lines.append(f"|{NAMES[name]}|{interval(x['word_block_95pct'])}|{interval(x['difference_from_ACC_95pct'])}|{interval(x['difference_from_fixed_Audio_memory_95pct'])}|")
    n=r['training_identity_permutation_control']
    lines+=['',f"99次训练Audio身份整节点置换并重新拟合预测器：正确监督跨词ρ={n['correct_training_audio_task_spearman']:.3f}，错误监督中位数{n['null_median']:.3f}，95%区间{interval(n['null_95pct'])}，单侧Monte Carlo p={n['p_one_sided']:.3f}。此检验不选择监督权重，也不把99个模型当99个受试者。", '',
      '|原始一致性补充检验|1999次整身份置换单侧p|两方法Holm后p|','|---|---:|---:|']
    for name,v in c.items():lines.append(f"|{NAMES[name]}|{v['p_one_sided']:.4f}|{v['p_holm_2']:.4f}|")
    lines+=['','该整身份置换同时改变Audio矩阵的行列，保留词结构和共享攻击者依赖。它是在结果汇总后的补充推断，不涉及重新训练、模型选择或新调参。直接ACC关系仍有弱正相关，但其检验接近0.05，不宜称强对应。',
      '', '词块区间为固定九人条件下500次相同的验证词抽样；跨词至少抽到两个不同词。没有把30个词对×9人当成270名独立受试者，也没有用六词bootstrap推断开放集人群。',
      '', '## 词类、身份难度与人数控制','',
      '验证12–17全部为指令词；训练0–11包含10个数字词、2个指令词。两阶段词类并不完全平衡。去掉词类标记以及仅用训练指令词10/11拟合的对照，仍保留部分监督排序收益，但没有解决最困难Top1不足；不能据此消除全部语义迁移影响。',
      '', f"留身份对监督ρ为{methods['Audio_Ridge_heldout_dyad']['crossword']['task_spearman_macro']:.3f}、留目标Audio监督ρ为{methods['Audio_Ridge_heldout_endpoint']['crossword']['task_spearman_macro']:.3f}，说明收益不完全依赖显式记住某一Audio身份对；但输入包含已知九人ACC统计，不能等同未见用户迁移。固定Audio身份记忆的跨词ρ约{m['task_spearman_macro']:.3f}，以及各方法很低的身份对内任务残差ρ，表明总体排序主要由稳定用户关系主导。",
      '', '|人数|全部组合数|直接ACC任务ρ|Audio监督任务ρ|改善组合比例|差值范围|ACC Top1|监督Top1|随机Top1|','|---|---:|---:|---:|---:|---|---:|---:|---:|']
    for v in r['all_combination_user_count_control']:
        lines.append(f"|{v['users']}|{v['cohorts']}|{v['baseline']:.3f}|{v['supervised']:.3f}|{v['fraction_improved']:.2%}|{interval(v['improvement_range'])}|{v['baseline_top1']:.2%}|{v['supervised_top1']:.2%}|{v['random_top1']:.2%}|")
    lines+=['','全部126个五人组合、36个七人组合及九人组均报告，无有利组合筛选。这里重新拟合关系预测器并改变候选攻击池，编码器仍固定为九人训练；组合相互重叠，不作为163次独立统计重复。',
      '', '## 全部逐用户跨词关系预测','', '|用户|方法|任务ρ|Top1|身份对内任务残差ρ|','|---|---|---:|---:|---:|']
    for u in range(9):
        for name,v in methods.items():
            x=v['crossword']['per_user'][u];res='—' if x['within_identity_pair_task_residual_spearman'] is None else f"{x['within_identity_pair_task_residual_spearman']:.3f}"
            lines.append(f"|U{u+1}|{NAMES[name]}|{x['task_spearman']:.3f}|{x['top1']:.2%}|{res}|")
    lines+=['','## 全部逐用户独立词认证AUC','', '|用户|Audio同词|Audio跨词|ACC同词|ACC跨词|','|---|---:|---:|---:|---:|']
    for u in range(9):
        aa,bb=[v['per_user'][u] for v in r['independent_word_authentication']]
        lines.append(f"|U{u+1}|{aa['sameword']:.4f}|{aa['crossword']:.4f}|{bb['sameword']:.4f}|{bb['crossword']:.4f}|")
    lines+=['','## 决策','',
      '本轮回答：①新的ACC表示能跨到未见词进行身份区分；②Audio身份关系在未见词中仍有弱至中等的可预测部分；③Audio监督确实提高完整困难排序的平均相关，超过同容量ACC自回归与多数错误监督；④它没有超过固定Audio身份记忆，且没有改善九人池的最难Top1，固定身份对内部的词任务残差预测很弱。',
      '', '**目前不启动局部困难关系蒸馏，也不增大KL或蒸馏权重。** 若后续继续此路线，先预先定义以Top1/Top2和身份对内任务残差为主目标的预测门槛，在新预留词或独立数据上验证能否同时超过直接ACC与固定关系记忆。不能拿本轮已查看的12–17词重新搜索预测器，再称独立验证。',
      '', '本轮没有给ACC编码器添加Audio损失，没有产生新的蒸馏认证结果；尚未证明开放集未见用户可迁移。',
      '', '## 文件与复现','',
      '- `train_relation_predictability_acc.py`：新ACC仅0–11训练。',
      '- `evaluate_relation_predictability.py`：预测锁定、对照、身份/词隔离与全部人数组合。',
      '- `check_relation_predictability.py`：合成预测、同容量回归、静态记忆残差、留词/留端点标签不可影响拟合、身份置换统计检查。',
      '- `report_relation_predictability.py`：匿名JSON转中文报告。',
      '- `verify_relation_predictability.py`：核查实际检查点训练索引、模型哈希及全部保存预测的指标重算；本轮最大重算误差0。',
      '- 协议：`docs/experiments/2026-11-17-relation-predictability-protocol.md`。输出：本目录protocol/history/prediction_lock/results/verification/identity_permutation公开JSON及报告。',
      '', '依次运行上述train、check、evaluate、report脚本；CPU两线程。私有模型、embedding、预测张量、真实身份映射均不上传。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print('REPORT_COMPLETE')

if __name__=='__main__':main()
