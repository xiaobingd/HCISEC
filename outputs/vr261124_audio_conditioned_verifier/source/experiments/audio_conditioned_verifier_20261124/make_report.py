"""Publish only aggregate anonymous results; cluster uncertainty by target user."""
import argparse,json
from pathlib import Path
import numpy as np

def main(root):
    out=root/'conditional_correspondence/audio_conditioned_verifier'
    data=json.loads((out/'summary_public.json').read_text());rows=data['macro'];users=data['per_user']
    names=['ungated_ACC','A_zero_audio','B_correct_audio','C_other_audio','D_destroyed_audio','B_wrong_audio','B_audio_first2words','B_audio_last2words']
    families=['english_word','english_long','chinese_long']
    lines=['# 注册 Audio 条件化 ACC 验证器：实际实验报告','',
      '执行来源：`experiment/audio-conditioned-verifier-20261124`，固定提交 `b2953752d7156c9b54d62a7adb16df5d2b5add46`。',
      '', '10 人全部保留，3 个种子（42/43/44），10 个留一目标验证器折，120 个小模型，每模型 400 步。所有查询均为一条 ACC 记录；没有三词联合。注册包含 4 个词的 12 条相同 ACC，Audio 取相同记录。注册、校准、查询词互不重叠，属于已见身份的跨内容验证。',
      '', '冻结 ACC 身份编码器与官方 ECAPA。576→32→192 小网络根据 ACC 原型、Audio 原型、ACC 注册方差输出 [0.75,1.25] 通道权重；使用加权余弦验证。A 为零 Audio，B 为正确 Audio，C 为异人 Audio，D 为训练统计匹配的随机 Audio。优化 BCE 与门控正则，所有组共享 ACC 难负类和 episode。',
      '', 'G 只优化前 12 个 fit 词，另外 4 个 fit 词记录内部验证损失，不用于挑选 checkpoint；固定第 400 步。每折目标完全不进入 G 训练或 Audio 统计。冻结 ACC 编码器曾在全 10 人 fit 词上学习，因此这不是整套系统严格未见用户开放集。',
      '', '校准规则预先固定为本人 Q10 与冒充者 Q95 的中点；240 个阈值全部落盘后才评分查询 ACC。B 的错 Audio 和同人异词扰动沿用 B 原阈值。没有查询 Audio。',
      '', '## 三种查询内容的宏平均', '', '数值为百分比，AUC 除外；每行先按目标与种子平均。', '', '|方法|查询内容|EER|HTER|FAR|FRR|AUC|','|---|---|---:|---:|---:|---:|---:|']
    for n in names:
      for f in families:
        r=next(r for r in rows if r['method']==n and r['family']==f)
        lines.append(f"|{n}|{f}|{100*r['eer']:.3f}|{100*r['hter']:.3f}|{100*r['far']:.3f}|{100*r['frr']:.3f}|{r['auc']:.5f}|")
    comparisons=[];rng=np.random.default_rng(20261124);boot=rng.integers(0,10,(10000,10))
    lines+=['','## 以用户为簇的配对比较','','差值 B−对照，负值为改善。每位用户先平均三个种子；bootstrap 重采样 10 位用户，不将重复录制视为独立受试者。95% 区间为探索性百分位区间。','','|对照|指标|B−对照/百分点|95%区间|改善用户/10|','|---|---|---:|---|---:|']
    for n in names:
      if n=='B_correct_audio':continue
      for metric in ['eer','hter']:
        delta=[]
        for u in range(1,11):
          b=np.mean([r[metric] for r in users if r['method']=='B_correct_audio' and r['user']==u])
          a=np.mean([r[metric] for r in users if r['method']==n and r['user']==u]);delta.append(b-a)
        delta=np.array(delta)*100;lo,hi=np.quantile(delta[boot].mean(1),[.025,.975])
        comparisons.append(dict(reference=n,metric=metric,delta_percentage_points=float(delta.mean()),cluster_ci95=[float(lo),float(hi)],improved_users=int((delta<0).sum())))
        lines.append(f'|{n}|{metric}|{delta.mean():+.3f}|[{lo:+.3f}, {hi:+.3f}]|{(delta<0).sum()}|')
    passed=all(c['cluster_ci95'][1]<0 for c in comparisons if c['reference'] in names[:2]+names[3:5])
    decision='通过预定独立增益检查，可进入新录制确认。' if passed else '未通过预定独立增益标准；停止当前通道门控方案，不在已查看查询词上继续搜索权重。'
    lines+=['','## 判断','',decision,'','本轮直接检验注册 Audio 是否能改善 ACC 验证度量。B 与无 Audio、错 Audio、破坏 Audio 的配对差异才是 Audio 独立贡献的依据；不能只看某一内容组或用户的最好结果。',
      '', '## 逐用户结果（3 种子平均）','','|用户|查询内容|方法|EER%|HTER%|FAR%|FRR%|','|---|---|---|---:|---:|---:|---:|']
    for u in range(1,11):
      for f in families:
        for n in names[:5]:
          r=next(r for r in users if r['user']==u and r['family']==f and r['method']==n)
          lines.append(f"|U{u:02d}|{f}|{n}|{100*r['eer']:.3f}|{100*r['hter']:.3f}|{100*r['far']:.3f}|{100*r['frr']:.3f}|")
    sensitivity=json.loads((out/'gate_diagnostics_public.json').read_text())['sensitivity']
    lines+=['','## 门控诊断','',f"换成他人 Audio 后，权重平均绝对变化 {np.mean([r['audio_wrong_gate_mean_abs_difference'] for r in sensitivity]):.6f}；同人两组注册词的变化 {np.mean([r['sameuser_word_halves_gate_mean_abs_difference'] for r in sensitivity]):.6f}。后一诊断每组只有 6 条 Audio，主实验为 12 条，数量不匹配，仅用于敏感性诊断。",'',
      '## 检查与边界','','独立核验 12000 个 episode：目标隔离、support/query 词无交集、真假标签一致。无门控余弦与历史相同预算基线完全回归；合成测试验证全一初始化、加权余弦、梯度与随机矩匹配。全部逐用户逐种子匿名汇总在 JSON 中。',
      '', '44100 Hz 为现有时间解释候选，物理延迟未独立标定。ACC 缓存为约 3 秒窗口，可能包含相邻重复发声；本报告的单条查询不等同于经人工确认只含一次孤立发声。历史上已反复查看这些查询词，当前结果为探索性。',
      '', '仅上传代码、方案、匿名逐用户汇总与审计摘要；私有 Audio、embedding、采样索引、阈值和权重不上传。']
    (out/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    (out/'comparison_public.json').write_text(json.dumps(dict(comparisons=comparisons,decision=decision),ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(decision=decision,macro=rows[:15]),ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
