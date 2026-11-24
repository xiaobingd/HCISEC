import argparse,json
from pathlib import Path
import numpy as np
def main(root):
    out=root/'conditional_correspondence/raw_common_dynamics';read=lambda n:json.loads((out/n).read_text())
    event=read('event_correspondence_public.json');ident=read('identity_summary_public.json');cross=read('crossband_public.json')
    rng=np.random.default_rng(20261124);boot=rng.integers(0,10,(10000,10))
    primary=next(v for v in event['macro'] if v['residual'] and v['conditioned'] and v['lag_ms']==0)
    differences=[]
    for a,b in [('coupling_correct','wrongrepeat_fixedmask'),('coupling_correct','wronguser_fixedmask'),('coupling_correct','timeshift_fixedmask'),('concatenation','ACC_only_stats'),('ACC_stats_Audiomask','ACC_only_stats')]:
      for metric in ['top1_10way','prototype_verification_auc']:
        delta=np.array([next(v[metric] for v in ident['per_user'] if v['user']==us and v['scope']=='new_words' and v['method']==a)-next(v[metric] for v in ident['per_user'] if v['user']==us and v['scope']=='new_words' and v['method']==b) for us in range(1,11)])
        ci=np.quantile(delta[boot].mean(1),[.025,.975]);differences.append(dict(method=a,control=b,metric=metric,delta=float(delta.mean()),target_cluster_ci95=ci.tolist(),improved_users=int((delta>0).sum())))
    pair_supported=event['primary_exact_user_signflip_p']<.05 and primary['correlation_advantage']>0
    latent_supported=cross['correct_advantage_exact_user_p']<.05 and cross['correct_minus_wrong_exact_user_p']<.05
    sensitive=any(v['method']=='coupling_correct' and v['control']=='wrongrepeat_fixedmask' and v['target_cluster_ci95'][0]>0 for v in differences)
    decision=('本轮检出对正确配对敏感的跨词耦合统计，并发现 Audio 辅助活动掩码的候选收益；同次发声动态检索仍未可靠高于机会，尚未证实共有生理身份模式或 ACC-only 增益。优先检验 ACC 能否预测 Audio 活动/质量掩码，暂不继续全局身份几何蒸馏。' if sensitive else '本轮尚未建立可靠的同次发声动态对应证据。局部统计可分性不能直接归因于已验证的跨模态共享动态，暂不将这些耦合统计作为 ACC 身份教师。')
    lines=['# 原始 Audio–ACC 发声动态与跨词身份审计','',
      '## 本轮研究对象','',
      '从原始波形重新提取局部谱，不使用 ECAPA/ACC 身份 embedding，不训练身份网络，不做最终认证或融合权重搜索。仅使用 10 人、16 个既有 fit 词的 1280 条记录；所有最终测试 Audio 未读取。',
      '', '原 ACC 约 999 Hz（三轴求功率和，去窗口常量），分析 20–450 Hz；Audio 按既有 44100 Hz 候选解释、重采样 16 kHz。窗口 100 ms、步长 20 ms，每条固定 3 秒。7 个频带为 20–50、50–80、80–120、120–180、180–250、250–350、350–450 Hz。没有做逐记录延迟择优或逐帧波形相位监督。',
      '', '## 一、控制词汇后的同次发声对应','',
      '固定同一身份和同一词。前四遍只用于估计各模态稳定时频模板，后四遍各自减去模板，再进行 4 选 1 真配对检索。这样控制词 ID 与重复模板，仍不能去除所有发音速度/音素实现差异。候选对使用相同规则；低于 12 个有效帧按中性相关 0 处理，不删人。',
      '', '活动代理来自 Audio 300–3500 Hz 能量，阈值 max(log(1e-12),Q20＋0.25×(Q90−Q20))；低频运动代理来自 ACC＜20 Hz，阈值由其他身份训练词 Q95 决定。两者都不是人工真值，不能声称已完全去除静音、头动。',
      '', '|模板去除|活动/运动代理控制|零延迟真配对 Top1|真配对−错遍相关差|','|---|---|---:|---:|']
    for v in event['macro']:
      if v['lag_ms']==0:lines.append(f"|{v['residual']}|{v['conditioned']}|{100*v['top1']:.2f}%|{v['correlation_advantage']:+.5f}|")
    lines+=['',f"主检验为模板残差＋活动/运动代理＋零延迟：Top1={100*primary['top1']:.2f}%（机会25%），正向相关优势的十用户精确符号翻转 p={event['primary_exact_user_signflip_p']:.4f}。推断单位是用户，不是帧或词对。",'',
      '|频带/Hz|相关优势|正向用户/10|max-T 校正 p|','|---|---:|---:|---:|']
    bands=read('extraction_public.json')['bands_hz']
    for v in event['bands']:
      lo,hi=bands[v['band']];lines.append(f"|{lo}–{hi}|{v['advantage']:+.5f}|{v['positive_users']}|{v['maxT_p']:.4f}|")
    lines+=['','### 固定延迟敏感性','','未从这些延迟选择最优值。这里只列主残差/代理控制条件。','','|Audio 相对 ACC 偏移/ms|Top1%|相关优势|','|---|---:|---:|']
    for v in event['macro']:
      if v['residual'] and v['conditioned']:lines.append(f"|{v['lag_ms']}|{100*v['top1']:.2f}|{v['correlation_advantage']:+.5f}|")
    lines+=['','### 跨频率线性补充（次要探索）','',
      '固定 2 维、ridge=1 的正则 CCA，在八位基础身份、前 12 个词的模板残差帧上拟合。两位留出身份、其余 4 词用于同人同词错遍检索；错遍训练控制保持内容与身份标签。没有搜索维度、正则或延迟。',
      '', '|训练配对|真配对 Top1%|相关优势|','|---|---:|---:|']
    for v in cross['macro']:lines.append(f"|{v['method']}|{100*v['top1']:.2f}|{v['correlation_advantage']:+.5f}|")
    lines+=['',f"正确 CCA 的正向优势 p={cross['correct_advantage_exact_user_p']:.4f}；正确−错遍训练控制的优势 p={cross['correct_minus_wrong_exact_user_p']:.4f}。这些是补充探索的用户级检验，不作为事后更换主检验。",'',
      '## 二、这些统计是否保留跨词身份差异','',
      '**这是离线双模态身份信息审计，不是 ACC-only 认证。耦合、拼接及 Audio 掩码方法的查询端都使用 Audio。只有 ACC_only_stats 的特征与活动掩码不使用查询 Audio。Audio_only_stats 不使用 ACC 掩码。**',
      '', '耦合统计包含每频带局部相关、回归斜率、经验对数能量差，各 7 维，共 21 维。后者不是经标定的传递函数。另有各自 16 维谱形/波动/能量统计、32 维拼接和三个 7 维消融。',
      '', '同人错遍、异人同词、500 ms 循环移位均有自适应掩码及固定正确 Audio 掩码版本；固定版本只改变 Audio 值，保持活动和运动掩码完全一致。错用户来源在八位基础身份内部循环，外层两人互换，留出身份不会通过错误来源进入基础均值/标准差。',
      '', '45 组双身份留出：预处理只在其余八人、前 12 个词上拟合。身份原型使用前六词的第 0/1 遍共 12 条；跨词检查为后六词的 4–7 遍，或保留四词的全部 8 遍。少于 12 个有效帧的样本用基础训练均值填补，全部人和记录保留。没有阈值、EER 或分类器训练。',
      '', '|特征|保留词十人 Top1%|保留词两人 Top1%|离线原型验证 AUC|','|---|---:|---:|---:|']
    for v in ident['macro']:
      if v['scope']=='new_words':lines.append(f"|{v['method']}|{100*v['top1_10way']:.2f}|{100*v['top1_2way']:.2f}|{v['prototype_verification_auc']:.4f}|")
    lines+=['','## 配对不确定性','','十个目标分别平均九个基础身份组合后，按用户重采样。身份组合共享数据，区间仅作探索性描述。正差为好。','','|方法−对照|指标|差/百分点|95%区间/百分点|改善用户|','|---|---|---:|---|---:|']
    for v in differences:
      lo,hi=v['target_cluster_ci95'];lines.append(f"|{v['method']}−{v['control']}|{v['metric']}|{100*v['delta']:+.2f}|[{100*lo:+.2f}, {100*hi:+.2f}]|{v['improved_users']}/10|")
    lines+=['','## 逐用户：保留词十人 Top1%','','|用户|纯 ACC|纯 Audio|拼接|正确耦合|固定掩码错遍|能量差|','|---|---:|---:|---:|---:|---:|---:|']
    for us in range(1,11):
      values=[next(v['top1_10way'] for v in ident['per_user'] if v['user']==us and v['scope']=='new_words' and v['method']==name) for name in ['ACC_only_stats','Audio_only_stats','concatenation','coupling_correct','wrongrepeat_fixedmask','coupling_energy_difference']]
      lines.append(f"|U{us:02d}|"+'|'.join(f'{100*v:.2f}' for v in values)+'|')
    missing=float(np.mean([v['insufficient_record_rate'] for v in ident['quality']]))
    lines+=['','## 判断','',decision,
      '', '单模态或拼接身份可分，并不能证明同次发声的动态耦合。正确耦合相对固定掩码错遍的探索性改善，是配对敏感性证据；两者的差异也可能包含发声区域错位造成的静音/能量混入，尚不能单独认定为身份特异性的生理耦合。判断身份价值还要结合真配对可辨识及跨词、跨佩戴稳定性。能量差的可分性也可能包含录制增益和接触条件。',
      '', '下一步的最小候选是训练期用 Audio 活动代理监督 ACC 质量/分段分支，查询时由 ACC 自行预测掩码；以 ACC 自身掩码、Audio 辅助掩码上限、预测掩码、打乱监督对照，验证表中辅助掩码收益能否在查询端完全移除 Audio 后保留。这是基于本轮结果的待验证方案，不是已经实现的认证改善。',
      '',f'正确共同掩码下有效帧不足的记录平均约 {100*missing:.2f}%，全部保留并填补。独立单模态与不同错配方法的缺失率单独列在 identity_summary_public.json。',
      '', '## 本轮不能回答的边界','',
      '物理同步没有独立标记确认，44.1 kHz 只是既有候选解释；本轮阴性结果不能区分信号确实缺乏可观测对应与残余时延/噪声/表征选择造成的检出失败。活动和运动只有代理，没有人工标注；缺少独立佩戴/设备/跨天验证。三秒窗可能包含相邻发声，模板半组也不等同完全独立的录制会话。',
      '', '本轮只覆盖七个粗频带的能量动态及低容量跨频率线性关系，不排除其他时频、非线性或更精确同步后的共同结构。没有选择有利用户，没有修改既有基线、教师或创新点一。公开代码和匿名逐用户汇总，不上传原始波形、局部谱、身份映射、逐样本分数或私有 CCA 参数。']
    (out/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    (out/'decision_public.json').write_text(json.dumps(dict(decision=decision,primary_event=primary,event_p=event['primary_exact_user_signflip_p'],crossband=cross['macro'],identity_differences=differences),ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(decision=decision,primary_event=primary),ensure_ascii=False))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
