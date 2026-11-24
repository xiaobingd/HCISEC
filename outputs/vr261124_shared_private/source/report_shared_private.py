"""Anonymous Chinese report for the fixed shared/private experiment."""
import csv
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'shared_private'
NAMES={'english_word':'英文单词','english_long':'英文长词／短语','chinese_long':'中文词／短语'}

def main():
    result=json.loads((OUT/'summary_public.json').read_text());macro=result['macro'];users=result['per_user']
    seeds=json.loads((OUT/'per_seed_user_public.json').read_text())
    comparisons=json.loads((OUT/'comparisons_public.json').read_text())
    diagnostics=json.loads((OUT/'branch_diagnostics_public.json').read_text())
    verify=json.loads((OUT/'verification_public.json').read_text())
    assert verify['status']=='PASS' and verify['independent_global_index_replay_max_error']<1e-10
    def find(method,family,k):return next(row for row in macro if row['method']==method and row['family']==family and row['words']==k)
    def pct(row,keys=('far','frr','hter','eer','auc')):return ' | '.join(f'{100*row[key]:.2f}' for key in keys)
    lines=['# Audio 共享身份分支＋ACC 补充分支：严格跨词认证实验',
        '', '日期：2026-11-24。状态：CPU 实际完成；3 个随机种子；全部 10 位用户、全部九个攻击来源。',
        '', '## 本轮回答的问题',
        '', '只对共享分支施加 Audio 身份关系监督，同时保留 ACC 补充分支，是否优于同结构无 Audio 对照、错误 Audio 身份关系对照和原 ACC？这不是 MISA 原论文复现，而是借用共享／私有分解思想的身份认证实验。',
        '', '参考：[MISA 原始论文](https://arxiv.org/abs/2005.03545)。原论文任务是多模态情感分析，并使用分布对齐、差异和重建目标；这里改用身份 CE 和有标签 Audio 关系监督，推理仅提取 ACC 两个分支。',
        '', '本轮结论：没有获得独立的 Audio 认证增益。三词 HTER 正确 Audio 在三个内容组均高于同尺寸无 Audio；与错误 Audio 关系对照几乎相当。拼接优于只读共享分支，支持保留 ACC 补充读出的必要性，但尚不能证明分解出两类独立身份信息。',
        '', '## 数据与隔离',
        '', '- 使用 2026-11-19 新采集原始硬件 ACC，按完整 40 词资格固定选取 10 人；不是旧 Mei9，也没有按效果删除用户。每人每词取已有规则选定的 8 条完整录制，共 3200 条。',
        '- 16 个既定训练词共 1280 条；预热 ACC、标准化器、Audio 教师适配和本轮学生训练均仅使用这部分。训练词见 protocol_public.json。',
        '- 沿用已核验样本 ID 对齐的官方 ECAPA 冻结主干缓存，只读取训练记录。复用仅在这 1280 条 Audio 上适配过的 projection＋残差教师；不读取注册、校准或查询 Audio embedding。',
        '- 新分支使用现有 ACC 192 维编码器作为起点，预热 700 步；每组再训练 400 步，种子 42／43／44。原模型和历史结果不修改。',
        '- 前端是既有 762 维 ACC 统计输入：三轴 10–300 Hz 对数频谱统计、时间分段、幅度及最高至 400 Hz 的频带统计。没有重新引入跟踪速度。',
        '- 缓存切分沿用采集元数据发声边界和主机时间映射，并非本轮独立验证的传感器硬件同步；认证网络不读取 Audio，但本轮离线前端仍使用历史语音分段边界。',
        '', '## 模型及训练',
        '', 'ACC 编码器：762→256→LayerNorm→GELU→192。双分支：归一化 latent 分别经两个 192→96 线性层；共享和补充各自 L2 归一化，拼接后再归一化为 192 维。两个投影由相同随机正交基初始化，解码器初始化为逆变换。',
        '', '| 组别 | 训练设置 |', '|---|---|',
        '| ACC | 原结构；仅身份 CE |',
        '| SplitNoAudio | 双分支身份 CE＋弱去冗余＋latent 重建 |',
        '| SplitAudio | 与 SplitNoAudio 相同，额外给共享分支正确 Audio 关系监督 |',
        '| SplitWrongAudio | 相同结构和损失，但将 Audio 身份做固定无不动点置换；词和重复编号保持匹配 |',
        '', '双分支身份损失＝拼接 CE＋0.1×共享 CE＋0.1×补充 CE；再加 0.01×中心化标准相关矩阵平方均值＋0.05×latent 重建 MSE。重建目标为当前 ACC latent 的停止梯度副本，不是原始频谱，也不保证分解唯一。',
        '', 'Audio 关系 MSE 等权比较同人异词与异人同词；仅施加在共享分支。权重由训练梯度尺度确定：Audio 对 ACC 编码器的梯度范数为身份梯度的 0.1 倍，系数上限 10。AdamW 1e-4，weight decay 0.01，clip 1；每批 10 人×5 词×2 重复，同一种子各组使用相同抽样。没有查询调参或检查点择优。',
        '', 'Audio 损失能更新公共 ACC 编码器，所以会间接改变补充分支输入；“没有直接对齐补充分支”不等于冻结保护原 ACC。原 ACC 参数较少，SplitNoAudio 是控制新增容量和正则化的主要对照。',
        '', '## 认证协议',
        '', '英文单词、英文长词／短语、中文各 5 个共享训练未见词，分别遍历所有 20 个有序注册／校准词组合：1 个词注册、1 个不同词校准、其余 3 词查询，三种角色词集严格不交叠。',
        '', '每人注册 8 条；个人 RBF SVM C=10、balanced，全部其他用户注册数据为负类。单条查询每人 24 次；三词联合每次取三个不同查询词同重复序号录制，平均三个决策分数，每人 8 次，每种条件内不复用查询源。',
        '', '阈值固定为 (校准本人 Q10＋校准攻击 Q95)/2。所有 36000 个阈值先落盘，再计算查询指标。三词校准只有一个词的八条源录制，使用 56 个三条不同重复组合；组合依赖、不是 56 条独立数据，且校准内容多样性低于查询。',
        '', '这是已见用户、未见词的闭集认证；不是未见身份开放集。EER 来自评估 ROC；FAR／FRR／HTER 使用提前锁定阈值。先等权平均 20 折，再等权用户和种子，不把重复词对当独立受试者。',
        '', '## 四组主对照',
        '', '以下均为百分数，AUC 也以百分数展示；“3”表示三个不同查询词联合。',
        '', '| 内容组 | 联合数 | 方法 | FAR | FRR | HTER | EER | AUC |',
        '|---|---:|---|---:|---:|---:|---:|---:|']
    for family in NAMES:
        for k in (1,3):
            for method in ('ACC','SplitNoAudio_fused','SplitAudio_fused','SplitWrongAudio_fused'):
                row=find(method,family,k);lines.append(f'| {NAMES[family]} | {k} | {method} | {pct(row)} |')
    lines+=['','## Audio 独立贡献与分支互补','','差值为正确 Audio 拼接减去对应对照，负值表示该错误率下降。区间按十位目标用户整簇 bootstrap，用户内折和种子先平均；是探索性不确定性描述，未做多重检验校正。','','| 内容组 | 指标 | 相对对照 | 差值（百分点） | 用户簇95%区间 | 改善／退化用户数 |','|---|---|---|---:|---|---|']
    for c in comparisons:
        if c['words']==3 and c['metric'] in ('hter','eer'):
            lo,hi=np.array(c['user_cluster_bootstrap95'])*100
            lines.append(f"| {NAMES[c['family']]} | {c['metric']} | {c['control']} | {100*c['audio_minus_control']:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {c['improved_users']}／{c['worsened_users']} |")
    lines+=['','三个分支读出的 RBF 后端分别重新训练，保持相同注册、校准和查询记录；不是训练时分类头准确率。','','| 内容组 | Audio 模型读出 | FAR | FRR | HTER | EER | AUC |','|---|---|---:|---:|---:|---:|---:|']
    for family in NAMES:
        for branch in ('shared','private','fused'):
            row=find('SplitAudio_'+branch,family,3);lines.append(f'| {NAMES[family]} | {branch} | {pct(row)} |')
    lines+=['','共享／补充分支的中心化线性 CKA（1 表示几何关系高度相似）与有效秩如下，先按种子平均。它们不是互信息估计或独有性证明。','','| 范围 | CKA | 共享有效秩／96 | 补充有效秩／96 |','|---|---:|---:|---:|']
    for scope in ('fit','held_english_word','held_english_long','held_chinese_long'):
        ds=[v for v in diagnostics if v['method']=='SplitAudio' and v['scope']==scope]
        means=[np.mean([v[key] for v in ds]) for key in ('linear_cka','shared_effective_rank','private_effective_rank')]
        lines.append(f'| {scope} | {means[0]:.4f} | {means[1]:.2f} | {means[2]:.2f} |')
    lines+=['','高 CKA 表明两个分支的样本几何仍大幅重复。当前弱去相关和重建设置不足以建立所期待的共享／补充分解；不能把这轮结果写成“ACC 不存在可供 Audio 帮助提取的信息”，也不能宣称成功找到了 ACC 独有身份特征。',
        '', '正确 Audio 相对无 Audio 的三词 HTER 平均变化分别为：'+ '、'.join(f"{NAMES[family]} {100*(find('SplitAudio_fused',family,3)['hter']-find('SplitNoAudio_fused',family,3)['hter']):+.3f} 个百分点" for family in NAMES)+'。区间均跨零，且没有跨三个内容组的一致改善。']
    lines+=['','## 三种子的方向','','| 内容组 | 种子 | Audio拼接 HTER | 无Audio拼接 HTER | 错误Audio拼接 HTER | ACC HTER |','|---|---:|---:|---:|---:|---:|']
    for family in NAMES:
        for seed in (42,43,44):
            vals=[100*np.mean([v['hter'] for v in seeds if v['family']==family and v['seed']==seed and v['words']==3 and v['method']==method]) for method in ('SplitAudio_fused','SplitNoAudio_fused','SplitWrongAudio_fused','ACC')]
            lines.append(f'| {NAMES[family]} | {seed} | '+' | '.join(f'{v:.3f}' for v in vals)+' |')
    lines+=['','## 全部逐用户结果：正确 Audio 双分支、三词跨内容','','| 内容组 | 匿名用户 | FAR | FRR | HTER | EER | AUC |','|---|---|---:|---:|---:|---:|---:|']
    for family in NAMES:
        for us in range(1,11):
            row=next(v for v in users if v['family']==family and v['words']==3 and v['method']=='SplitAudio_fused' and v['user']==us)
            lines.append(f'| {NAMES[family]} | U{us} | {pct(row)} |')
    lines+=['','全部方法、两个联合数、各分支和所有用户另见 [summary_public.json](summary_public.json) 与 [逐用户 CSV](per_user_public.csv)，逐种子见 [per_seed_user_public.json](per_seed_user_public.json)。',
        '', '## 如何判断本轮结果',
        '', '应分别判断：双分支是否超过原 ACC；正确 Audio 是否超过同尺寸无 Audio 和错误 Audio；拼接是否超过单共享分支。某项改善不能替代另外两项。共享／补充命名和低相关约束本身不能证明得到物理独有身份子空间。',
        '', '如果正确 Audio 只优于原 ACC，而没有超过 SplitNoAudio，就把改善归于双分支身份学习，不能归于 Audio。如果仅 EER 下降而 HTER／FAR／FRR 退化，则应先解决分数与阈值迁移，不把排序改善写成完整认证收益。',
        '', '本轮不依据结果再改损失、挑用户或选择最佳折。后续若要继续，须使用新的独立开发词验证；冻结原 ACC、只增加小共享残差可用于进一步区分“补充新表征”和“改变已有表征”的作用。',
        '', '## 验证与公开范围',
        '', f"合成梯度路由及内容重叠拒绝测试通过；36000 个工作点计数与错误率重算通过；独立全局样本索引重放首个中文划分、种子42、U1、Audio拼接的单条／三词结果，最大误差 {verify['independent_global_index_replay_max_error']:.3g}。",
        '', '公布匿名逐用户汇总、训练日志、代码、协议与文件哈希；不公布音频、ACC 波形、embedding、模型参数、身份映射、逐样本分数或个人阈值。原创新点一和历史模型保持原文件。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    with (OUT/'per_user_public.csv').open('w',newline='',encoding='utf-8') as file:
        writer=csv.DictWriter(file,fieldnames=list(users[0]));writer.writeheader();writer.writerows(users)
    print('SHARED_PRIVATE_REPORT_READY',len(macro),len(users))

if __name__=='__main__':main()
