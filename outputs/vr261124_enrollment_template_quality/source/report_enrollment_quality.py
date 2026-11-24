import csv
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'conditional_correspondence/enrollment_template_report';OUT.mkdir(exist_ok=True)
PHASES={'acoustic':'enrollment_quality_fixed_background','identity':'enrollment_identity_quality_fixed_background'}
FAMILIES={'english_word':'英文单词','english_long':'英文长词／短语','chinese_long':'中文词／短语'}

def main():
    verify=json.loads((OUT/'verification_public.json').read_text());assert verify['status']=='PASS'
    diagnostic=json.loads((OUT/'registration_diagnostic_public.json').read_text());comparisons=[];datasets={}
    for phase,folder in PHASES.items():
        src=ROOT/'conditional_correspondence'/folder;data=json.loads((src/'summary_public.json').read_text());datasets[phase]=data
        dest=OUT/phase;dest.mkdir(exist_ok=True)
        (dest/'macro_public.json').write_text(json.dumps(data['macro'],indent=2))
        for name,rows in [('per_user_public.csv',data['per_user']),('per_seed_user_public.csv',json.loads((src/'per_seed_user_public.json').read_text()))]:
            with (dest/name).open('w',newline='',encoding='utf8') as file:
                writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        for name in ('protocol_public.json','verification_public.json'):(dest/name).write_bytes((src/name).read_bytes())
        method='acc_audio12' if phase=='acoustic' else 'acc_audio_identity12'
        controls=['acc12','shuffled_audio12','random12','all32_reference'] if phase=='acoustic' else ['acc_crossword12','shuffled_audio_identity12','random12','all32_reference']
        for backend in ('RBF','cosine'):
            for family in FAMILIES:
                for k in (1,3):
                    for control in controls:
                        for metric in ('far','frr','hter','eer'):
                            def values(m):return np.array([next(a[metric] for a in data['per_user'] if a['method']==m and a['backend']==backend and a['family']==family and a['words']==k and a['user']==us) for us in range(1,11)])
                            delta=values(method)-values(control);rng=np.random.default_rng(4242)
                            boot=delta[rng.integers(0,10,(5000,10))].mean(1)
                            comparisons.append(dict(phase=phase,backend=backend,family=family,words=k,control=control,metric=metric,
                                audio_minus_control=float(delta.mean()),user_cluster95=np.quantile(boot,[.025,.975]).tolist(),improved_users=int((delta<0).sum()),worsened_users=int((delta>0).sum())))
    (OUT/'comparisons_public.json').write_text(json.dumps(comparisons,indent=2))
    # Keep the earlier variable-background acoustic pilot explicit,not a primary result.
    pilot=json.loads((ROOT/'conditional_correspondence/enrollment_quality/summary_public.json').read_text())
    (OUT/'variable_background_pilot_macro_public.json').write_text(json.dumps(pilot['macro'],indent=2))
    def lookup(phase,method,family,backend='RBF',k=3):return next(a for a in datasets[phase]['macro'] if a['method']==method and a['family']==family and a['backend']==backend and a['words']==k)
    def pct(a):return '|'.join(f'{100*a[key]:.3f}' for key in ('far','frr','hter','eer','auc'))
    lines=['# Audio辅助注册模板质量：冻结ACC、等采集预算对照','', '2026-11-24。完成两类质量定义、两个固定后端、三个编码器种子，共5400个主工作点；原始可变负类声学试验另保留2520个探索工作点的宏汇总。',
        '', '本轮判断：身份一致性Audio辅助注册在英文RBF上出现小幅候选收益，但不是稳定全面改善。相对ACC跨词筛选，英文单词EER/HTER由0.271/0.812%变为0.057/0.684%，英文长词由0.271/1.054%变为0.100/1.011%；中文EER由1.538%升至1.610%，HTER仅由7.407%降至7.350%。余弦原型没有对应的稳定收益。',
        '', '英文单词EER改善集中在U5、U8及种子43，种子42/44的两种方法英文EER都为0；HTER在两个种子改善、一个种子退化。不能把该均值写成所有用户和所有种子稳定改善。Audio能识别身份与能判断ACC样本质量是两个不同问题。',
        '', '## 任务变化与隔离','', '不更新ACC编码器、不做Audio→ACC蒸馏。复用第三轮A_ACC三个冻结编码器的192维表示；Audio只在注册时帮助评价样本。个人RBF按注册子集重新拟合，后端配置固定；不能说RBF权重也被冻结。查询阶段两个后端均仅使用ACC。',
        '', '全部10位已见用户。注册词固定[20,21,22,23]，每人4词×8遍=32条；每种主方法保留每词3条共12条。固定取样、随机取样、ACC筛选也具有相同32条候选池。只有Audio方法额外读取这320条注册Audio；不能称只采12条的注册或相同模态成本。全部32条为更高保留数量诊断参照。',
        '', '校准词固定[8,18,24,30,31]，查询英文单词[25,26,27,28,29]、英文长词/短语[3,9,12,19,32]、中文[35,36,37,38,39]。共享编码器仅在既定16训练词学习，角色词集互斥；校准和查询Audio从未读取。中文条件同时有英文注册/校准→中文查询的语言差异。',
        '', '每人每词全部8条用于校准/查询，单条40本人/360攻击；三词查询固定分组平均三个不同词分数，39条源录制组成13本人/117攻击试验，尾部一条不使用，组内不复用源ID。不同目标复用攻击录制，3秒窗口同词内可能重叠/含邻近发声，因此不把攻击比较或重复视为独立人。',
        '', 'RBF主后端C10 balanced；所有方法的背景负类固定为其他9人全部32条注册ACC，共288条，选择仅改变本人的12条正类。余弦后端为选中12条ACC归一化均值原型，只用注册合法样本。阈值统一采用校准本人Q10/攻击Q95中点；两阶段共1800阈值先各自落盘，再算三个查询组共5400指标。',
        '', '本协议注册/校准词数量与之前单注册词20折不同，只能在本轮匹配方法间归因，不能将绝对性能变化归于Audio。群体已反复探索，非未见用户开放集或新盲测。',
        '', '## 两组质量定义','', '|阶段|ACC评分|Audio评分|融合与对照|','|---|---|---|',
        '|声学质量|同人同词留一余弦中心性|短帧能量80/20分位比、周期性、削波率的负值，等权词内秩|ACC/Audio秩各0.5；同人同词Audio秩无固定点置换|',
        '|身份一致性|同人其余3个注册词的ACC原型余弦|同人其余3个注册词的Audio原型余弦，词内秩|跨词ACC/Audio秩各0.5；同人同词Audio秩无固定点置换|',
        '', '声学能量对比不是严格测得的信噪比；削波、周期性也不是身份质量真值。两阶段的ACC指标不同，不能将跨阶段差异单独归于Audio指标。第二阶段在第一阶段探索之后提出，所有规则与0.5权重在第二阶段评价前固定，两组都完整报告，没有依查询成绩挑方法或权重。',
        '', '身份Audio使用官方VoxCeleb预训练ECAPA，权重SHA核验，未在本地身份微调。仅提取320条注册录音，按44100解释重采样16k，右填零及真实长度mask处理；不读取测试Audio，不更新ECAPA。它只给注册样本质量评分，不进入ACC查询，也不是让ACC匹配Audio embedding。物理逐帧同步不是该声纹评分的前提；样本配对仍依赖采集ID及既有映射。',
        '', '## 注册数据本身的诊断','', f"注册词之间的Audio身份AUC={diagnostic['audio_registration_crossword_auc']:.4f}，ACC={diagnostic['acc_registration_crossword_auc_mean']:.4f}。Audio与ACC跨词质量秩的词内Spearman平均={diagnostic['quality_rank_spearman_mean']:.4f}。",'', '因此Audio能区分用户，不等于它能准确判断哪条ACC最适合建模；该相关性是注册数据内部诊断，不能替代认证收益检验。',
        '', '## 主结果：三词联合RBF认证','', '全部指标为百分数，先在每位用户平均三个编码器种子，再等权10用户。EER由评估ROC；FAR/FRR/HTER使用提前锁定阈值。',
        '', '|阶段|内容|方法|FAR|FRR|HTER|EER|AUC|','|---|---|---:|---:|---:|---:|---:|---:|']
    for phase in PHASES:
        for family in FAMILIES:
            methods=sorted(set(a['method'] for a in datasets[phase]['macro']))
            for method in methods:lines.append(f'|{phase}|{FAMILIES[family]}|{method}|{pct(lookup(phase,method,family))}|')
    lines+=['','## 身份一致性Audio融合：逐用户配对区间','','负差值表示融合更低，区间按十位目标整簇bootstrap5000次，种子先在用户内平均；没有按重复样本制造独立性。未多重校正确认。','','|内容|指标|对照|差值(pp)|用户簇95%区间(pp)|改善/退化用户数|','|---|---|---|---:|---|---|']
    for a in comparisons:
        if a['phase']=='identity' and a['backend']=='RBF' and a['words']==3 and a['control'] in ('acc_crossword12','shuffled_audio_identity12') and a['metric'] in ('hter','eer'):
            lo,hi=np.array(a['user_cluster95'])*100
            lines.append(f"|{FAMILIES[a['family']]}|{a['metric']}|{a['control']}|{100*a['audio_minus_control']:+.3f}|[{lo:+.3f},{hi:+.3f}]|{a['improved_users']}/{a['worsened_users']}|")
    lines+=['','## 身份阶段三个种子：三词RBF','','|内容|种子|ACC跨词HTER|ACC+Audio身份HTER|ACC跨词EER|ACC+Audio身份EER|','|---|---:|---:|---:|---:|---:|']
    seedrows=json.loads((ROOT/'conditional_correspondence/enrollment_identity_quality_fixed_background/per_seed_user_public.json').read_text())
    for family in FAMILIES:
        for seed in (42,43,44):
            vals=[]
            for metric,method in [('hter','acc_crossword12'),('hter','acc_audio_identity12'),('eer','acc_crossword12'),('eer','acc_audio_identity12')]:
                vals.append(np.mean([a[metric] for a in seedrows if a['family']==family and a['seed']==seed and a['method']==method and a['backend']=='RBF' and a['words']==3]))
            lines.append(f'|{FAMILIES[family]}|{seed}|'+ '|'.join(f'{100*v:.3f}' for v in vals)+'|')
    lines+=['','## 余弦身份原型诊断：三词查询','','|内容|方法|HTER%|EER%|','|---|---|---:|---:|']
    for family in FAMILIES:
        for method in ('acc_crossword12','acc_audio_identity12','shuffled_audio_identity12','all32_reference'):
            a=lookup('identity',method,family,'cosine');lines.append(f"|{FAMILIES[family]}|{method}|{100*a['hter']:.3f}|{100*a['eer']:.3f}|")
    lines+=['','## 全部逐用户：身份融合、三词RBF','','|内容|用户|FAR|FRR|HTER|EER|','|---|---|---:|---:|---:|---:|']
    for family in FAMILIES:
        for us in range(1,11):
            a=next(a for a in datasets['identity']['per_user'] if a['method']=='acc_audio_identity12' and a['backend']=='RBF' and a['family']==family and a['words']==3 and a['user']==us)
            lines.append(f'|{FAMILIES[family]}|U{us}|'+ '|'.join(f'{100*a[key]:.3f}' for key in ('far','frr','hter','eer'))+'|')
    lines+=['','两个阶段的全部1800条逐用户均值、5400条逐种子用户结果见acoustic/与identity/目录的CSV；包括单条/三条、RBF/余弦及全部方法，不筛选有利用户。',
        '', '## 探索性原始版本及验证','', '最先的声学质量版本同时按方法筛选了背景负类，改变了本人正类与负类，不能独立归因为模板质量。保留其macro汇总为variable_background_pilot_macro_public.json，主结果改用固定288负类；不是依据查询成绩选择负类。',
        '', '两个固定负类阶段的固定取样、随机取样、ACC同词筛选、全部32条参照，所有用户/种子/后端/指标精确一致，最大差0。5400工作点计数与错误率重算、每人每词3条预算、首个用户RBF及余弦单/三条独立索引重放通过，误差0。静默音频有限值/并列排序/采样率解释合成测试通过。',
        '', '没有更新全局身份编码器、读取查询Audio、删除困难用户或按查询结果选权重。公开匿名汇总和代码，不公开音频、embedding、身份映射、模型权重、选样ID、阈值或逐样本认证分数。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print('ENROLLMENT_REPORT_READY',len(comparisons),flush=True)

if __name__=='__main__':main()
