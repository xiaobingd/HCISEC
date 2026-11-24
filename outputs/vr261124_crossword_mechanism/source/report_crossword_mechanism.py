"""Stage3 identity-constraint experiment, compared with its matched stage2."""
import csv
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent;BASE=ROOT/'conditional_correspondence/stage2';OUT=ROOT/'conditional_correspondence/stage3'
METHODS=['A_ACC','B_correct','C_wrong_repeat','D_wrong_user']
FAMILIES={'english_word':'英文单词','english_long':'英文长词／短语','chinese_long':'中文词／短语'}

def main():
    old=json.loads((BASE/'summary_public.json').read_text());new=json.loads((OUT/'summary_public.json').read_text())
    ret=json.loads((OUT/'retrieval_summary_public.json').read_text());oldret=json.loads((BASE/'retrieval_summary_public.json').read_text())
    verify=json.loads((OUT/'verification_public.json').read_text());assert verify['independent_replay_max_error']<1e-10
    def find(r,m,f,k):return next(a for a in r['macro'] if a['method']==m and a['family']==f and a['words']==k)
    interactions=[]
    for f in FAMILIES:
        for k in (1,3):
            for metric in ('far','frr','hter','eer'):
                before=find(old,'B_correct',f,k)[metric]-find(old,'A_ACC',f,k)[metric]
                after=find(new,'B_correct',f,k)[metric]-find(new,'A_ACC',f,k)[metric]
                interactions.append(dict(family=f,words=k,metric=metric,audio_effect_stage2=before,audio_effect_stage3=after,interaction=after-before))
    (OUT/'interaction_public.json').write_text(json.dumps(interactions,indent=2))
    lines=['# 共享发声机制实验三：显式跨词身份约束',
        '', '日期：2026-11-24。12个新模型、3个种子、4组Audio监督、同400步预算。',
        '', '本轮结果：正确Audio相对同轮无Audio的三词EER/HTER均略低，但六个用户簇bootstrap95%区间均跨零，且错配对照在英文单词和中文HTER仍更低。显式跨词身份约束改善了部分正确Audio组指标，没有证明正确配对的独立身份收益。开发检索B为23.333%，仍高于错遍18.646%和异人19.271%，但不能将检索优势直接等同于认证增益。',
        '', '## 为什么不是重复实验二',
        '', '实验二已经含CE和普通SupCon，并不是完全没有身份监督。实验三只改变SupCon正对/分母掩码：同人不同词为正对，异人为负对，同人同词的不同遍从此对比项分母与正对中排除。它们仍具有相同身份CE标签，不因错遍成为身份负样本。',
        '', '网络、初始化种子、训练批次、Audio目标、标准化、学习率和400步训练预算均沿用实验二。四组新模型从匹配随机初始化训练，没有使用实验二训练后权重续训。A无Audio；B正确局部Audio；C同人同词错遍；D异人同词同遍。',
        '', '## 固定的数据与认证协议',
        '', '10个已见身份，16个训练词1280条，开发词8/18/24/30共320条，最终3个内容组共15词。仅训练词参与梯度和标准化；开发Audio仅训练完成后离线检索；不读取最终查询Audio。Audio沿用按44100解释/重采样16k的连续3秒局部目标，50–300Hz候选频点、100ms分析窗/20ms帧移、12个250ms块。没有换ECAPA或加Transformer。',
        '', 'ACC仍为三轴原始3秒窗口的100ms/20ms STFT(10–490Hz)、两层32通道CNN，身份均值/标准差池化输出192维；单独局部预测头。全部方法CE+0.1跨词SupCon；Audio组局部SmoothL1，公共骨干辅助梯度范数预算0.1、权重上限10。',
        '', '每组内容5词，遍历全部20个注册/校准有序组合；1词8条注册、1不同词8条校准、其余3词查询。RBF C10 balanced、全部9个攻击来源。单条24本人/216攻击，三词8本人/72攻击。阈值固定为校准本人Q10和攻击Q95中点，14400个阈值先落盘再算查询指标。EER由评估ROC计算，FAR/FRR/HTER使用校准阈值。',
        '', '物理同步仍未标定，这是名义时间映射下的粗时间尺度探索。窗口可能包含相邻发声并在同词内重叠；VAD中心仍保留。校准三条组合仅来自一个词8条源录制，不是56条独立录制。历史测试内容已反复查看，本轮不是新盲测；不将结果用于调参或筛选用户。',
        '', '## 开发词局部对应检索',
        '', '同人同词8次录制作为候选，真实Audio目标负MSE检索，随机Top1=12.5%。A的局部头未监督，仅作随机读出参照。',
        '', '|组别|实验二Top1%|实验三Top1%|实验三margin|','|---|---:|---:|---:|']
    for method in METHODS:
        a=next(v for v in ret['macro'] if v['method']==method and v['scope']=='development_unseen_words')
        b=next(v for v in oldret['macro'] if v['method']==method and v['scope']=='development_unseen_words')
        lines.append(f"|{method}|{100*b['top1']:.3f}|{100*a['top1']:.3f}|{a['margin']:.6f}|")
    lines+=['','## 同预算身份认证：实验二与实验三','','指标为百分数，等权平均20折、用户、种子。','','|内容|联合数|组别|实验二EER|实验三EER|实验二HTER|实验三HTER|实验三FAR|实验三FRR|','|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for f in FAMILIES:
        for k in (1,3):
            for m in METHODS:
                a=find(old,m,f,k);b=find(new,m,f,k)
                lines.append(f'|{FAMILIES[f]}|{k}|{m}|'+ '|'.join(f'{100*v:.3f}' for v in [a['eer'],b['eer'],a['hter'],b['hter'],b['far'],b['frr']])+'|')
    lines+=['','## 分开判断身份约束收益和Audio独立收益','','Audio效应定义为同一轮B−A，负值表示正确Audio比无Audio错误率更低。交互定义为(实验三B−A)−(实验二B−A)，不能只拿实验三B与实验二A比较。','','|内容|三词指标|实验二Audio效应(pp)|实验三Audio效应(pp)|交互(pp)|','|---|---|---:|---:|---:|']
    for a in interactions:
        if a['words']==3 and a['metric'] in ('hter','eer'):
            lines.append(f"|{FAMILIES[a['family']]}|{a['metric']}|{100*a['audio_effect_stage2']:+.3f}|{100*a['audio_effect_stage3']:+.3f}|{100*a['interaction']:+.3f}|")
    lines+=['','三词条件下，正确Audio相对同轮无Audio的差异如下（负数为错误率下降）：']
    for f in FAMILIES:
        a=find(new,'A_ACC',f,3);b=find(new,'B_correct',f,3)
        lines.append(f"- {FAMILIES[f]}：EER {100*(b['eer']-a['eer']):+.3f}个百分点，HTER {100*(b['hter']-a['hter']):+.3f}个百分点。")
    lines+=['','## 三种子配对结果：三词认证','','|内容|种子|旧A HTER|新A HTER|旧B HTER|新B HTER|新A EER|新B EER|','|---|---:|---:|---:|---:|---:|---:|---:|']
    oldseed=json.loads((BASE/'per_seed_user_public.json').read_text());newseed=json.loads((OUT/'per_seed_user_public.json').read_text())
    def seedmean(rows,f,seed,method,metric):return 100*np.mean([a[metric] for a in rows if a['family']==f and a['seed']==seed and a['method']==method and a['words']==3])
    for f in FAMILIES:
        for seed in (42,43,44):
            vals=[seedmean(rows,f,seed,m,key) for rows,m,key in [(oldseed,'A_ACC','hter'),(newseed,'A_ACC','hter'),(oldseed,'B_correct','hter'),(newseed,'B_correct','hter'),(newseed,'A_ACC','eer'),(newseed,'B_correct','eer')]]
            lines.append(f'|{FAMILIES[f]}|{seed}|'+ '|'.join(f'{a:.3f}' for a in vals)+'|')
    lines+=['','## 全部逐用户：B正确Audio、三词跨内容','','|内容|用户|FAR|FRR|HTER|EER|','|---|---|---:|---:|---:|---:|']
    for f in FAMILIES:
        for us in range(1,11):
            a=next(a for a in new['per_user'] if a['family']==f and a['words']==3 and a['method']=='B_correct' and a['user']==us)
            lines.append(f'|{FAMILIES[f]}|U{us}|'+ '|'.join(f'{100*a[key]:.3f}' for key in ('far','frr','hter','eer'))+'|')
    lines+=['','全部240条逐用户、720条逐种子用户指标见summary_public.json、per_seed_user_public.json、per_user_public.csv。逐用户开发检索见retrieval_summary_public.json。',
        '', '只有B稳定优于同轮A/C/D，才支持Audio配对特异性认证收益。若A也改善，应归因于跨词身份学习；若仅检索改善，不把它解释成身份信息增益。',
        '', f"验证：14400工作点计数与错误率检查通过；中文首折seed42 B U1单条/三词独立索引重放最大误差{verify['independent_replay_max_error']:.3g}。跨词损失有限梯度、置换不变、缺失跨词正对拒绝与局部梯度路由合成测试通过。",
        '', '仅公开匿名汇总、协议和代码，不公开私有波形、embedding、身份映射、模型参数、阈值或逐样本分数。创新点一冻结结果与实验二保持原文件。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    with (OUT/'per_user_public.csv').open('w',newline='',encoding='utf8') as file:
        writer=csv.DictWriter(file,fieldnames=list(new['per_user'][0]));writer.writeheader();writer.writerows(new['per_user'])
    print('CROSSWORD_REPORT_READY',flush=True)

if __name__=='__main__':main()
