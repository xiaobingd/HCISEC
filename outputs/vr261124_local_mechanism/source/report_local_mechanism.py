import csv
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'conditional_correspondence/stage2'
FAMILIES={'english_word':'英文单词','english_long':'英文长词／短语','chinese_long':'中文词／短语'}
METHODS=['A_ACC','B_correct','C_wrong_repeat','D_wrong_user']

def main():
    r=json.loads((OUT/'summary_public.json').read_text());ret=json.loads((OUT/'retrieval_summary_public.json').read_text());v=json.loads((OUT/'verification_public.json').read_text())
    assert v['rate_count_checks'] and v['independent_replay_max_error']<1e-10
    def lookup(method,family,k):return next(a for a in r['macro'] if a['method']==method and a['family']==family and a['words']==k)
    def pct(row):return ' | '.join(f'{100*row[k]:.3f}' for k in ('far','frr','hter','eer','auc'))
    lines=['# 共享发声机制实验二：局部低频Audio辅助ACC身份学习',
        '', '2026-11-24。CPU实际运行，10位用户，种子42/43/44，四组同预算对照。',
        '', '**范围：尚未完成物理同步标定，本轮是名义时间映射下的粗时间尺度探索，不是已经校准的逐帧对齐实验。**',
        '', '本轮结论：正确配对的开发词检索优于同人同词错遍和异人Audio对照，三个种子方向一致；但认证没有独立收益。三词查询的正确Audio EER在三个内容组都高于纯ACC；中文HTER小幅改善没有超过错配对照。当前局部监督学到了可预测的录制对应结构，尚未转化为跨词身份判别收益。',
        '', '## 改动与数据',
        '', '不使用ECAPA、声纹embedding或身份相似度矩阵。原始三轴硬件ACC按1kHz构造3秒窗口；100ms STFT、20ms帧移，保留10–490Hz共49个频点/轴，得到147通道时序输入。前端逐通道均值/标准差仅用16训练词1280条估计。没有旧762维统计分支或旧预热编码器。',
        '', 'Audio读取连续完整PCM，将原48000头的采样解释为44100，按160/441重采样16k。分段中心秒数同步换算，PCM中心样本位置不变。100ms分析窗、20ms帧移，取50–300Hz的26个10Hz频点，再汇总为3秒内12个固定250ms块；不是整句拉伸归一，也不是完整声纹目标。50–300Hz仅是候选频带，未据此宣称传递函数或真实频响已测定。',
        '', '只读取16个训练词的1280条和独立4个开发词的320条Audio，用于训练监督/离线检索；15个最终查询内容类Audio不读取。Audio标准化仅使用训练词；开发词不进入梯度、选择或阈值确定。训练/开发/查询词见protocol_public.json。',
        '', '连续窗口内部不套每遍录制的VAD零掩码，但中心仍来自历史VAD。窗口可能包含相邻重复发声，且同一词录制内窗口可能重叠，因此重复/词对不是独立受试者。不同身份认证角色来自不同词的独立录制文件；不能把同一录制重叠窗口分到训练与最终查询角色。',
        '', '## 四组与两个头',
        '', '|组别|额外局部监督|', '|---|---|',
        '|A_ACC|无Audio监督|','|B_correct|同次发声Audio局部低频目标|',
        '|C_wrong_repeat|同人同词下一遍Audio目标，循环置换，无固定点|',
        '|D_wrong_user|下一用户同词同遍Audio目标，循环置换，无固定点|',
        '', '四组使用相同新初始化、batch序列和400步AdamW，lr0.001、weight decay0.01、clip1；每批10人×2词×2重复=40条。编码器两层Conv1d(32通道)+GroupNorm+GELU；身份头时间均值/标准差池化→192维L2；局部头32→26通道并汇总12块。所有组均CE+0.1SupCon，同人跨词/跨重复为身份正样本。',
        '', 'B/C/D增加局部SmoothL1；只作用于独立局部头并回传公共骨干，不直接作用身份投影。辅助骨干梯度预算为身份梯度范数的0.1，系数上限10；记录梯度余弦。A也保留同尺寸未监督局部头，不把它的随机局部输出当成学到的配对能力。',
        '', '## 第一层：对应关系检索',
        '', '将局部预测与同人同词8个候选Audio目标分别比较标准化MSE；所有候选同身份、同词，仅重复不同。Top1随机水平12.5%；并列最大按1/并列数计分。margin为正确配对负MSE减其余7遍平均负MSE。该指标仍可能受发声长度及相邻录制影响，不直接等于身份信息。',
        '', '|组别|范围|Top1%|正确减错遍margin|','|---|---|---:|---:|']
    for a in ret['macro']:lines.append(f"|{a['method']}|{a['scope']}|{100*a['top1']:.3f}|{a['margin']:.6f}|")
    lines+=['','开发词正确Audio减同人同词错遍Top1：+5.417个百分点；目标用户整簇bootstrap95%区间[+3.333,+7.708]个百分点。减异人Audio为+4.062个百分点，区间[+1.250,+6.667]。先按用户平均词与种子；没有把每个词对当独立受试者，也没有做多重检验校正。',
        '', '|种子|正确Audio开发Top1%|错遍开发Top1%|异人开发Top1%|','|---:|---:|---:|---:|']
    for seed in (42,43,44):
        vals=[next(a['top1'] for a in ret['development_per_seed'] if a['seed']==seed and a['method']==method) for method in METHODS[1:]]
        lines.append(f'|{seed}|'+'|'.join(f'{100*a:.3f}' for a in vals)+'|')
    lines+=['','## 第二层：严格跨内容认证','','沿用每个内容组5词的全部20个注册/校准有序组合：1词8条/人注册、1不同词8条/人校准、其余3词查询。全部9个攻击来源，RBF C10 balanced。单条24本人/216攻击；三词联合8本人/72攻击。每个条件源查询窗口只使用一次，但物理窗口重叠造成的依赖仍在。',
        '', '阈值固定为校准本人Q10与攻击Q95中点；全部14400个锁定阈值先落盘后计算查询指标。三词校准是同一校准词8条的56个组合，非56条独立录制，与查询三个不同词的多样性不匹配。EER由查询ROC计算，HTER/FAR/FRR使用校准锁定阈值。',
        '', '指标均百分数，先等权20折再等权用户和种子；A不是历史11.94%/14.23%那套协议，必须只与本轮A进行归因比较。每条输入是3秒连续窗口，三条联合覆盖三个窗口且可能含相邻发声；不能把本轮数值与旧短语句统计编码器直接比较，或将差异归于采样率校正/Audio辅助。',
        '', '|内容组|联合数|组别|FAR|FRR|HTER|EER|AUC|','|---|---:|---|---:|---:|---:|---:|---:|']
    for family in FAMILIES:
        for k in (1,3):
            for method in METHODS:lines.append(f'|{FAMILIES[family]}|{k}|{method}|{pct(lookup(method,family,k))}|')
    lines+=['','## 全部逐用户：正确Audio、三词跨内容','','|内容组|用户|FAR|FRR|HTER|EER|AUC|','|---|---|---:|---:|---:|---:|---:|']
    for family in FAMILIES:
        for us in range(1,11):
            a=next(a for a in r['per_user'] if a['method']=='B_correct' and a['family']==family and a['words']==3 and a['user']==us)
            lines.append(f'|{FAMILIES[family]}|U{us}|{pct(a)}|')
    lines+=['','完整240条逐用户均值、720条逐种子用户均值见summary_public.json、per_seed_user_public.json；检索逐用户见retrieval_summary_public.json。',
        '', '## 判定规则与局限',
        '', 'B开发检索超过C只支持当前局部目标具有可学习的对应信息；只有B认证同时超过A/C/D，才支持正确Audio的独立身份收益。B/C/D认证共同改善首先考虑辅助正则化，不能归于配对特异性。',
        '', '没有训练后改权重、挑用户、换词或挑种子。当前队列已经多轮探索，不是新盲测；物理同步和频带筛选仍待共同事件核验。敲击标定不能替代语音传递函数测量。',
        '', '按预定停止原则，本轮不再增加局部预测损失权重或堆叠模型。配对检索优势只证明当前统计目标有可学习的对应；发声动态、时长和相邻重复都可能贡献，并未隔离成身份因子。如果继续利用Audio，应优先独立验证注册样本质量/模板选择，而不是将本轮谱预测直接当作新的身份教师。',
        '', f"验证：合成梯度路由/错遍无固定点通过；14400工作点计数和错误率重算通过；独立全局索引重放中文首折、seed42、U1、B单条/三词，最大误差{v['independent_replay_max_error']:.3g}。",'', '仅公开代码、匿名逐用户/种子汇总和协议；不上传PCM、ACC、embedding、模型权重、身份映射、逐样本分数或阈值。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    with (OUT/'per_user_public.csv').open('w',newline='',encoding='utf8') as file:
        writer=csv.DictWriter(file,fieldnames=list(r['per_user'][0]));writer.writeheader();writer.writerows(r['per_user'])
    print('LOCAL_REPORT_READY',flush=True)

if __name__=='__main__':main()
