"""Public aggregate and all-user report; excludes scores/thresholds/identities."""
import argparse
import json
from pathlib import Path
import numpy as np


def report(root):
    root=Path(root)
    result=json.loads((root/'results_public.json').read_text(encoding='utf-8'))
    rows=json.loads((root/'per_user_public.json').read_text(encoding='utf-8'))
    verification=json.loads((root/'verification_public.json').read_text(encoding='utf-8'))
    assert result['state']=='complete' and verification['state']=='PASS'
    lines=['# Mei9：Audio决策蒸馏与跨词神经验证器首轮结果','',
           '本轮为九人已知身份、单seed42、独立ACC认证方法，不使用CycleGAN。注册/查询均仅ACC；Audio教师为冻结ReDimNet2-B3，仅提取训练词1618条录音。词编号按历史顺序推断，尚未逐条语义核验。',
           '', '最终每人12条注册，同词/跨词各12条本人及96条攻击查询；保留全部八个攻击来源。训练词0–17，开发18–20，校准21–23，注册/同词24–26，跨词27–29。新协议不能与旧12/8/16/24协议的绝对数值直接比较。',
           '', '所有神经配置相同1000步预算，开发宏AUC选检查点。每人阈值由Cal确定，并在同词和跨词测试之间保持一致。EER由测试分数作描述性诊断，HTER/FAR/FRR使用锁定阈值。',
           '', '|方法|条件|TAR%|FAR%|FRR%|HTER%|EER%|AUC|', '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in result['summary']:
        lines.append('|'+r['method']+'|'+r['condition']+'|'+ '|'.join(f'{100*r[k]:.2f}' for k in ['tar','far','frr','hter','eer'])+f"|{r['auc']:.4f}|")
    lines+=['','## 成对比较','', '|前者减后者|条件|HTER变化(pp)|FAR变化(pp)|TAR变化(pp)|HTER改善用户数|','|---|---|---:|---:|---:|---:|']
    pairs=[('local','global'),('audio','local'),('shuffle','audio')]
    summary={(r['method'],r['condition']):r for r in result['summary']}
    for a,b in pairs:
        for cond in ['sameword','crossword']:
            ra,rb=summary[a,cond],summary[b,cond]
            wins=sum(next(r['hter'] for r in rows if r['method']==a and r['condition']==cond and r['user']==f'U{u}')<next(r['hter'] for r in rows if r['method']==b and r['condition']==cond and r['user']==f'U{u}')-1e-10 for u in range(1,10))
            lines.append(f'|{a} − {b}|{cond}|'+ '|'.join(f'{100*(ra[k]-rb[k]):+.2f}' for k in ['hter','far','tar'])+f'|{wins}/9|')
    lines+=['','## 逐用户完整工作点','', '|用户|条件|方法|TAR%|FAR%|FRR%|HTER%|EER%|AUC|','|---|---|---|---:|---:|---:|---:|---:|---:|']
    for r in sorted(rows,key=lambda r:(r['user'],r['condition'],r['method'])):
        lines.append('|'+r['user']+'|'+r['condition']+'|'+r['method']+'|'+ '|'.join(f'{100*r[k]:.2f}' for k in ['tar','far','frr','hter','eer'])+f"|{r['auc']:.4f}|")
    lines+=['','## 模型选择与核查','', '|方法|开发选择步数|开发宏AUC|','|---|---:|---:|']
    for mode in ['global','local','audio','shuffle']:
        selected=json.loads((root/(mode+'_selection_public.json')).read_text())
        lines.append(f"|{mode}|{selected['step']}|{selected['dev_macro_auc']:.4f}|")
    lines+=['', '独立重载编码器、四个验证器和两种原始统计后端，重建合法Cal阈值与全部108个工作点。核查结论：'+verification['state']+f"；最大指标误差{verification['max_metric_error']:.3g}。", '',
            '教师评分器按训练查询词三折交叉拟合，主干冻结；不是未见身份教师交叉验证。打乱对照保留身份，仅破坏具体词/录制配对。', '',
            'raw_rbf在最终合法注册数据拟合，并用Cal查询校准；神经验证器在Cal内部模拟注册校准，校准支持构造不同。主要方法消融看global/local/audio/shuffle，不能把RBF与神经组差异全部归因于表示。', '',
            '本轮没有完成ECAPA替换、传统特征蒸馏同结构对照或多种子验证；不将强教师本身的识别能力等同于ACC增益。历史同会话数据已探索，未验证重戴/跨天，词标签为名义映射。', '',
            '协议：docs/experiments/2026-11-16-neural-verifier-protocol.md。私有数据、模型、身份映射、分数和阈值不公开。']
    (root/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);report(p.parse_args().out)
