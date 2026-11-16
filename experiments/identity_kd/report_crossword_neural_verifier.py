"""Public aggregate and all-user report; excludes scores/thresholds/identities."""
import argparse
import json
from pathlib import Path
import numpy as np


def report(root, baseline=None):
    root=Path(root)
    result=json.loads((root/'results_public.json').read_text(encoding='utf-8'))
    rows=json.loads((root/'per_user_public.json').read_text(encoding='utf-8'))
    verification=json.loads((root/'verification_public.json').read_text(encoding='utf-8'))
    protocol=json.loads((root/'protocol_public.json').read_text(encoding='utf-8'))
    users=sorted(set(r['user'] for r in rows))
    excluded=protocol.get('excluded_training_users',[])
    if baseline is None and excluded:
        archived=Path('outputs/mei_neural_verifier_20261116')
        if (archived/'per_user_public.json').exists():
            baseline=archived
    assert result['state']=='complete' and verification['state']=='PASS'
    lines=[f'# Mei：{len(users)}名注册者的Audio决策蒸馏与跨词神经验证器结果','',
           f'本轮为{len(users)}名已知注册身份、单seed42、独立ACC认证方法，不使用CycleGAN。注册/查询均仅ACC；Audio教师为外部冻结ReDimNet2-B3，只使用合法训练词特征。词编号按历史顺序推断，尚未逐条语义核验。',
           '', '最终每人12条注册，同词/跨词各12条本人及96条攻击查询；保留全部八个攻击来源。训练词0–17，开发18–20，校准21–23，注册/同词24–26，跨词27–29。新协议不能与旧12/8/16/24协议的绝对数值直接比较。',
           '', '所有神经配置相同1000步预算，开发宏AUC选检查点。每人阈值由Cal确定，并在同词和跨词测试之间保持一致。EER由测试分数作描述性诊断，HTER/FAR/FRR使用锁定阈值。',
           '', '|方法|条件|TAR%|FAR%|FRR%|HTER%|EER%|AUC|', '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in result['summary']:
        lines.append('|'+r['method']+'|'+r['condition']+'|'+ '|'.join(f'{100*r[k]:.2f}' for k in ['tar','far','frr','hter','eer'])+f"|{r['auc']:.4f}|")
    if baseline:
        old=json.loads((Path(baseline)/'per_user_public.json').read_text(encoding='utf-8'))
        old=[r for r in old if r['user'] in users]
        comparison=[]
        lines+=['','## 相同八名目标：九人训练与排除U5训练','',
                '两边只统计相同八名目标，攻击查询均仍含U5。此表避免仅删除U5目标就改变宏平均；训练任务与Cal具体采样随用户集合变化，不等于严格固定批次的单变量实验。',
                '', '|方法|条件|旧九人训练HTER%|新八人训练HTER%|差值(pp)|','|---|---|---:|---:|---:|']
        for method in sorted(set(r['method'] for r in rows)):
            for cond in ['sameword','crossword']:
                previous=[r for r in old if r['method']==method and r['condition']==cond]
                current=[r for r in rows if r['method']==method and r['condition']==cond]
                oh=np.mean([r['hter'] for r in previous]);nh=np.mean([r['hter'] for r in current])
                comparison.append(dict(method=method,condition=cond,old_hter=float(oh),new_hter=float(nh),delta=float(nh-oh)))
                lines.append(f'|{method}|{cond}|{100*oh:.2f}|{100*nh:.2f}|{100*(nh-oh):+.2f}|')
        (root/'matched_targets_comparison_public.json').write_text(json.dumps(comparison,indent=2),encoding='utf-8')
    lines+=['','## 成对比较','', '|前者减后者|条件|HTER变化(pp)|FAR变化(pp)|TAR变化(pp)|HTER改善用户数|','|---|---|---:|---:|---:|---:|']
    pairs=[('local','global'),('audio','local'),('shuffle','audio')]
    summary={(r['method'],r['condition']):r for r in result['summary']}
    for a,b in pairs:
        for cond in ['sameword','crossword']:
            ra,rb=summary[a,cond],summary[b,cond]
            wins=sum(next(r['hter'] for r in rows if r['method']==a and r['condition']==cond and r['user']==u)<next(r['hter'] for r in rows if r['method']==b and r['condition']==cond and r['user']==u)-1e-10 for u in users)
            lines.append(f'|{a} − {b}|{cond}|'+ '|'.join(f'{100*(ra[k]-rb[k]):+.2f}' for k in ['hter','far','tar'])+f'|{wins}/{len(users)}|')
    if excluded:
        lines+=['','## 训练排除用户诊断','',
                f'排除身份编号{excluded}不进入编码器、Scaler、教师评分器、神经验证器训练、开发或Cal，也不建立该身份注册目标。最终保留其查询为未见攻击；这是已知目标对未见攻击者的验证，不是未见目标注册实验。',
                '', '教师仅复用外部冻结主干的逐录制独立特征，训练数组中排除用户的所有特征置零；未复用九人ACC编码器、标准化或验证器。',
                '', '|方法|条件|仅七名已知攻击FAR%|含本人FRR的限定八人HTER%|U5未见攻击FAR%|',
                '|---|---|---:|---:|---:|']
        for method in sorted(set(r['method'] for r in rows)):
            for cond in ['sameword','crossword']:
                rr=[r for r in rows if r['method']==method and r['condition']==cond]
                known=np.mean([r['known_attack_far'] for r in rr]);fr=np.mean([r['frr'] for r in rr]);unseen=np.mean([r['unseen_attack_far'] for r in rr])
                lines.append(f'|{method}|{cond}|{known*100:.2f}|{(known+fr)*50:.2f}|{unseen*100:.2f}|')
        lines+=['','|用户|条件|方法|已知攻击FAR%|U5攻击FAR%|','|---|---|---|---:|---:|']
        for r in sorted(rows,key=lambda r:(r['user'],r['condition'],r['method'])):
            lines.append(f"|{r['user']}|{r['condition']}|{r['method']}|{r['known_attack_far']*100:.2f}|{r['unseen_attack_far']*100:.2f}|")
    lines+=['','## 逐用户完整工作点','', '|用户|条件|方法|TAR%|FAR%|FRR%|HTER%|EER%|AUC|','|---|---|---|---:|---:|---:|---:|---:|---:|']
    for r in sorted(rows,key=lambda r:(r['user'],r['condition'],r['method'])):
        lines.append('|'+r['user']+'|'+r['condition']+'|'+r['method']+'|'+ '|'.join(f'{100*r[k]:.2f}' for k in ['tar','far','frr','hter','eer'])+f"|{r['auc']:.4f}|")
    lines+=['','## 模型选择与核查','', '|方法|开发选择步数|开发宏AUC|','|---|---:|---:|']
    for mode in ['global','local','audio','shuffle']:
        selected=json.loads((root/(mode+'_selection_public.json')).read_text())
        lines.append(f"|{mode}|{selected['step']}|{selected['dev_macro_auc']:.4f}|")
    lines+=['', f"独立重载编码器、四个验证器和两种原始统计后端，重建合法Cal阈值与全部{verification['replayed_workpoints']}个工作点。核查结论："+verification['state']+f"；最大指标误差{verification['max_metric_error']:.3g}。", '',
            '教师评分器按训练查询词三折交叉拟合，主干冻结；不是未见身份教师交叉验证。打乱对照保留身份，仅破坏具体词/录制配对。', '',
            'raw_rbf在最终合法注册数据拟合，并用Cal查询校准；神经验证器在Cal内部模拟注册校准，校准支持构造不同。主要方法消融看global/local/audio/shuffle，不能把RBF与神经组差异全部归因于表示。', '',
            'global仅使用三种全局分数；local增加注意力及更宽的局部输入，实际参与反传的参数更多。因此local对global是完整验证器结构比较，尚未通过等参数MLP控制单独归因于注意力。local/audio/shuffle结构与预算相同，可用于比较Audio监督。', '',
            '本轮没有完成ECAPA替换、传统特征蒸馏同结构对照或多种子验证；不将强教师本身的识别能力等同于ACC增益。历史同会话数据已探索，未验证重戴/跨天，词标签为名义映射。', '',
            '协议：docs/experiments/2026-11-16-neural-verifier-protocol.md。私有数据、模型、身份映射、分数和阈值不公开。']
    (root/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--baseline')
    a=p.parse_args();report(a.out,a.baseline)
