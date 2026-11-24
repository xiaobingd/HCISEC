import csv
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'conditional_correspondence/stage4_selected'
FAMILIES={'english_word':'英文单词','english_long':'英文长词／短语','chinese_long':'中文词／短语'}
METHODS=['A_ACC','B_fullSpectrum','B_correct','C_wrong_repeat','D_wrong_user']

def main():
    result=json.loads((OUT/'summary_public.json').read_text());ret=json.loads((OUT/'retrieval_summary_public.json').read_text())
    verify=json.loads((OUT/'verification_public.json').read_text());assert verify['independent_replay_max_error']<1e-10
    def find(m,f,k):return next(a for a in result['macro'] if a['method']==m and a['family']==f and a['words']==k)
    comparisons=[]
    for f in FAMILIES:
        for k in (1,3):
            for control in ('A_ACC','B_fullSpectrum','C_wrong_repeat','D_wrong_user'):
                for metric in ('far','frr','hter','eer'):
                    def user_values(method):return np.array([next(a[metric] for a in result['per_user'] if a['method']==method and a['family']==f and a['words']==k and a['user']==us) for us in range(1,11)])
                    delta=user_values('B_correct')-user_values(control);rng=np.random.default_rng(4242)
                    boot=delta[rng.integers(0,10,(5000,10))].mean(1)
                    comparisons.append(dict(family=f,words=k,control=control,metric=metric,selected_minus_control=float(delta.mean()),
                        user_cluster95=np.quantile(boot,[.025,.975]).tolist(),improved_users=int((delta<0).sum()),worsened_users=int((delta>0).sum())))
    (OUT/'selected_comparisons_public.json').write_text(json.dumps(comparisons,indent=2))
    lines=['# 固定筛选频带认证实验：70–300Hz辅助监督',
        '', '日期：2026-11-24。9个新模型(3组×3种子)，另复用3个无Audio和3个完整低频监督模型；同400步预算与认证协议。',
        '', '实际结论：筛选正确Audio组在三类内容的三词EER和HTER均高于完整50–300Hz监督，也均高于无Audio。英文单词/长词/中文HTER分别为5.464/2.625/3.059%，完整监督为4.919/2.387/2.701%。不采用本轮70–300Hz硬筛频配置。',
        '', '单频点跨词身份显著性筛选并未产生更好的联合监督；不能从50/60Hz单独未通过检验推导它们在联合任务中无用，也不能将本次退化直接归因于这两个频点承载了独有生理身份。掩码改变了联合辅助梯度方向；当前证据是该固定配置没有改善，不是对所有频带选择方法的否定。',
        '', '## 固定的筛选与问题',
        '', '上轮开发集筛选留下70–300Hz共24个10Hz频点作为共享身份相关候选，但没有频点通过正确Audio独立身份收益的追加审计。本轮是对此候选的受控认证检验，不预设有收益。频带在本轮训练前固定，不再根据查询结果选择频点、权重、用户或检查点。',
        '', '仅改变Audio局部SmoothL1监督的频点掩码，排除50和60Hz；ACC身份输入仍是原三轴10–490Hz时序谱，不能将本实验描述成ACC高通70Hz。预测头仍为26通道，参数量/初始权重不变，身份CE+跨词SupCon不变，局部梯度预算不变。新错遍/异人模型也使用相同24频点掩码。',
        '', '## 五组与数据',
        '', '|组别|含义|','|---|---|','|A_ACC|复用第三轮无Audio模型及其ACC表示，未追加训练|',
        '|B_fullSpectrum|复用第三轮完整50–300Hz正确Audio模型及表示，未追加训练|',
        '|B_correct|新训练70–300Hz正确配对Audio监督|',
        '|C_wrong_repeat|新训练70–300Hz同人同词错遍Audio监督|',
        '|D_wrong_user|新训练70–300Hz异人同词同遍Audio监督|',
        '', '10人全部保留。16训练词1280条，4开发词320条离线检索；最终15个内容类Audio不读取。Audio按44100解释、16k重采样，连续3秒窗口、100ms STFT/20ms帧移、12个250ms局部块。新模型从与第三轮相同种子随机初始化；batch抽样相同，每组400步AdamW.001，身份CE+.1跨词SupCon，局部SmoothL1骨干梯度比例.1、系数上限10。',
        '', 'C沿用固定下一遍，D沿用固定下一用户循环置换，词标签保持一致。相邻窗口可能部分重叠，固定用户置换也可能被模型记忆；它们是匹配的历史对照，不是完全破坏所有可学习结构的随机负控制。不能将未超过它们直接解释成Audio不含任何有用信息。',
        '', '候选来自历史查看过的开发词；本轮认证查询内容也已被历史实验查看，所以是固定设置的探索性复核，不是全新盲测。物理同步仍未标定，3秒窗口可能含相邻发声/同词内重叠，VAD中心信息仍存在。',
        '', '## 固定认证协议',
        '', '三类内容各5词，每类全部20种注册/校准有序划分：1词8条注册、1不同词8条校准、3个其余词查询。全部9攻击来源，RBF C10 balanced。单条24本人/216攻击；三条跨词联合8本人/72攻击，平均决策分数。校准三条来自单个词8条的56个依赖组合，不是独立56条录制。阈值固定为校准本人Q10/攻击Q95中点，18000阈值先落盘再算查询指标。',
        '', '以下百分数为等权20折、用户、种子均值。EER按查询ROC；FAR/FRR/HTER使用提前锁定的校准阈值。',
        '', '|内容|联合数|组别|FAR|FRR|HTER|EER|AUC|','|---|---:|---|---:|---:|---:|---:|---:|']
    for f in FAMILIES:
        for k in (1,3):
            for method in METHODS:
                a=find(method,f,k);lines.append(f'|{FAMILIES[f]}|{k}|{method}|'+ '|'.join(f'{100*a[key]:.3f}' for key in ('far','frr','hter','eer','auc'))+'|')
    lines+=['','## 筛选相对完整低频监督：逐用户簇不确定性','','负差值为筛选后错误率更低。先在每位用户内平均词划分与种子，再整簇bootstrap十位用户5000次；不是以重复、词对或种子作为独立受试者。未做多重比较确认。','','|内容|三词指标|筛选−完整(pp)|95%区间(pp)|改善/退化用户数|','|---|---|---:|---|---|']
    for a in comparisons:
        if a['words']==3 and a['control']=='B_fullSpectrum' and a['metric'] in ('hter','eer'):
            lo,hi=np.array(a['user_cluster95'])*100
            lines.append(f"|{FAMILIES[a['family']]}|{a['metric']}|{100*a['selected_minus_control']:+.3f}|[{lo:+.3f},{hi:+.3f}]|{a['improved_users']}/{a['worsened_users']}|")
    lines+=['','## 24频点开发词对应检索','','以下四组均按24频点评价，八选一随机12.5%。没有把完整26频点的历史检索率直接当同尺度对照；A局部头未训练。','','|组别|开发Top1%|配对margin|','|---|---:|---:|']
    for a in ret['macro']:
        if a['scope']=='development_unseen_words':lines.append(f"|{a['method']}|{100*a['top1']:.3f}|{a['margin']:.6f}|")
    lines+=['','## 全部逐用户：筛选正确Audio、三词跨内容','','|内容|用户|FAR|FRR|HTER|EER|','|---|---|---:|---:|---:|---:|']
    for f in FAMILIES:
        for us in range(1,11):
            a=next(a for a in result['per_user'] if a['family']==f and a['method']=='B_correct' and a['words']==3 and a['user']==us)
            lines.append(f'|{FAMILIES[f]}|U{us}|'+ '|'.join(f'{100*a[key]:.3f}' for key in ('far','frr','hter','eer'))+'|')
    lines+=['','全部300条逐用户均值与900条逐种子用户指标见summary_public.json、per_seed_user_public.json、per_user_public.csv。完整控制差异与区间见selected_comparisons_public.json。',
        '', '只有筛选组稳定超过完整监督且超过同轮无Audio/错配，才支持筛选带来的独立身份收益。单个内容组下降或检索提高不能替代该判断。本轮不根据结果继续改变频带。',
        '', f"验证：18000工作点计数与错误率重算通过；首个中文划分seed42 B U1单条/三词独立重放误差{verify['independent_replay_max_error']:.3g}；被排除50/60Hz的局部输出梯度为零、选中频点梯度非零，身份投影不直接接收局部梯度的合成测试通过。",'', '仅公开匿名汇总、协议、代码，不公开波形、embedding、权重、身份映射、阈值或逐样本分数。历史第三轮、冻结创新点一均不修改。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    with (OUT/'per_user_public.csv').open('w',newline='',encoding='utf8') as file:
        writer=csv.DictWriter(file,fieldnames=list(result['per_user'][0]));writer.writeheader();writer.writerows(result['per_user'])
    print('SELECTED_REPORT_READY',flush=True)

if __name__=='__main__':main()
