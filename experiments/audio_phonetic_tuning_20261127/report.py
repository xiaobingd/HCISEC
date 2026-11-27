"""Report all bounded-search outcomes, including failed development gates and negative controls."""
import json,hashlib
from pathlib import Path
import numpy as np
from train_select import ROOT,PRIV,OUT,PREVIOUS,CONFIGS,write

def read(name):return json.loads((OUT/name).read_text(encoding='utf-8'))
def p(v):return f'{100*v:.2f}'
def avg(rows,key):return float(np.mean([r[key] for r in rows]))
def table(headers,rows):return '\n'.join(['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|']+['|'+'|'.join(map(str,row))+'|' for row in rows])

def main():
    result=read('results_public.json');lock=read('selection_lock_public.json');verify=read('verification_public.json');assert verify['status']=='PASS'
    summary=result['summary'];users=result['per_seed_user'];wordrows=result['per_seed_word_user'];codes=['BASE','ORIGINAL','MATCHED','TUNED','SHUFFLED']
    primary={r['arm']:r for r in summary if r['family']=='english_primary'};base,orig,tuned=primary['BASE'],primary['ORIGINAL'],primary['TUNED']
    restored=(orig['EER']-tuned['EER'],orig['HTER']-tuned['HTER']);gain=(base['EER']-tuned['EER'],base['HTER']-tuned['HTER'])
    if gain[0]>0 and gain[1]>0:
        conclusion='调参后平均 EER 与 HTER 均低于纯 ACC；是否支持 Audio 独立增益，还需同时检查匹配无 Audio 和时间错配对照及用户配对统计。'
    elif restored[0]>0 and restored[1]>0:
        conclusion='减弱辅助约束修复了原配置的部分退化，但未同时超过纯 ACC 的 EER 与 HTER。不能将减少损伤写成额外 Audio 身份增益。'
    else:conclusion='本轮有限参数调整没有同时改善原配置的 EER 与 HTER，保留纯 ACC 神经对照。'
    labels=dict(BASE='既有纯 ACC',ORIGINAL='原参数真实 Audio(.1,.05)',MATCHED='与新参数匹配的无 Audio',TUNED='开发选择的真实 Audio',SHUFFLED='同参数 Audio 时间错配')
    def summary_table(family):
        rr=[r for r in summary if r['family']==family]
        return table(['方法','EER%','AUC','FAR%','FRR%','HTER%'],[[r['arm'],p(r['EER']),f"{r['AUC']:.4f}",p(r['FAR']),p(r['FRR']),p(r['HTER'])] for r in rr])
    vocab=json.loads((PREVIOUS/'role_split_public.json').read_text(encoding='utf-8'))['vocabulary']
    lines=['# Audio 音素辅助：有限权重优化与三种子复核','',conclusion,'',
      f"开发词选择 **{lock['selected']}**：音素 λ={lock['config']['phone']}，词对抗 λ={lock['config']['content']}。相对原配置，主要 EER 改善 {p(restored[0])} 个百分点、HTER 改善 {p(restored[1])} 个百分点；相对纯 ACC，分别改善 {p(gain[0])}、{p(gain[1])} 个百分点（正值为改善）。",'',
      '## 1. 本轮范围','',
      '这是用户在原五组实验后授权的小范围参数优化。原实验提交 `e6ec6c1f0caefd3caf7a9a1435feeef3c19ac0ea`、历史结果、既有模型及创新点一均保留。此轮新增独立代码/输出/私有检查点目录。','',
      '复用同一2179个原始硬件ACC连续窗口及818个允许训练/开发词的冻结Wav2Vec2音素后验；不重新读取任何Audio。身份网络、147通道前端、全词表392类KL、100ms时间核、学习率0.0003、AdamW、裁剪1、batch80、12条注册、校准规则均保持。只改变音素损失与词对抗权重。','',
      '当前仍为10人已见身份的闭集跨词认证。上一轮查询词已经看过，候选方案也由上轮负结果触发，因此本轮是探索性后续复核，不称为全新盲测。候选排序/参数锁定只用开发词，不依据本轮最终查询分数选择。','',
      '## 2. 事先固定的有限搜索','',
      table(['开发组','音素权重','词对抗权重'],[[c,CONFIGS[c]['phone'],CONFIGS[c]['content']] for c in CONFIGS]),'',
      '开发四组同seed42、同初始化、同批次、各400步，在12优化词上训练。四个开发词 Off/Okay/zero/Zoom 做4次循环：每次两个词各取2个窗口注册（K=4），另一个词校准、最后一个词查询；每折注册/校准/查询词及原始窗口互斥。四词全部轮流作为查询；四折相关结果仅服务参数选择，不当作独立受试者。','',
      '排序分数为(EER+HTER)/2。开发门槛同时要求EER与HTER不高于纯ACC，FAR和FRR各自不比纯ACC增加超过0.5个百分点。在通过门槛的A/B中选择最低分；若两者都不通过，仅将开发分较低的一套做一次探索性正式复核，不标记为已通过。','',
      table(['开发组','EER%','FAR%','FRR%','HTER%','排序分数%'],[[c,*[p(lock['summaries'][c][k]) for k in ['EER','FAR','FRR','HTER','selection_score']]] for c in CONFIGS]),'',
      f"开发联合门槛通过：**{lock['development_dual_metric_and_FAR_FRR_gate_passed']}**。选择和门槛标记均在正式训练/查询之前落盘，详见 `selection_lock_public.json`。开发K=4只用于内部选择，不能将其绝对成绩直接与正式K=12成绩比较。",'',
      '## 3. 正式复核','',
      f"全部16训练词、1000步、种子42/43/44，正式查询完全复用原协议。新训练 {result['newly_trained_final_models']} 个最终模型，其他条目只读复用既有对照；所有方法/种子共15个评估条目、150个预先锁定阈值。",'',
      table(['代号','方法'],[[c,labels[c]] for c in codes]),'',
      'MATCHED只关闭Audio损失，保留选择后的词对抗权重；若词对抗为0，MATCHED与BASE复用同一纯ACC模型，其相同结果不能当作两份独立证据。SHUFFLED与TUNED使用相同参数、教师有效帧、窗口后验边际与种子，只循环移位有效时间；完整模型、初始化与批次一致。','',
      '训练后先锁定所有150个校准阈值 `(本人Cal Q10 + 其余9身份Cal Q95)/2`，再计算查询。全部用户、全部15个查询词、全部913个实际查询窗口均保留；单窗口ACC-only认证。英文短词和英文长短语等权为主要指标，中文单独报告。','',
      '### 英文等权主要结果','',summary_table('english_primary'),'',
      '### 英文短词','',summary_table('english_short'),'',
      '### 英文长短语','',summary_table('english_long'),'',
      '### 中文迁移','',summary_table('chinese'),'',
      '### 逐种子英文主要结果','',
      table(['seed','方法','EER%','FAR%','FRR%','HTER%'],[[seed,c,*[p(avg([r for r in users if r['seed']==seed and r['arm']==c and r['family']!='chinese'],k)) for k in ['EER','FAR','FRR','HTER']]] for seed in [42,43,44] for c in codes]),'',
      '## 4. 全部逐用户和逐词','',
      '下表为英文两族等权、三个种子平均的EER/HTER（%）；逐用户FAR/FRR/AUC及所有逐种子、逐词指标见各方法匿名JSON。','',
      table(['用户']+codes,[[f'U{user:02}']+[p(avg([r for r in users if r['user']==user and r['arm']==c and r['family']!='chinese'],'EER'))+'/'+p(avg([r for r in users if r['user']==user and r['arm']==c and r['family']!='chinese'],'HTER')) for c in codes] for user in range(1,11)]),'',
      table(['词ID','词']+codes,[[word,vocab[str(word)]]+[p(avg([r for r in wordrows if r['word']==word and r['arm']==c],'EER'))+'/'+p(avg([r for r in wordrows if r['word']==word and r['arm']==c],'HTER')) for c in codes] for word in [25,26,27,28,29,3,9,12,19,32,35,36,37,38,39]]),'',
      '## 5. 配对统计','',
      '按用户先平均三个种子和两个英文词族，10个用户为统计簇；10000次用户簇bootstrap、1024种精确符号置换，三项主比较做Holm校正。负差值表示TUNED更好。MATCHED若与BASE重复，三项校正仍保留，避免将重复对照当作额外收益证据。','',
      table(['比较','ΔEER百分点','用户簇95%CI','Holm p','改善种子/用户'],[[c['comparison'],p(c['mean_difference']),f"[{p(c['user_cluster_bootstrap_95CI'][0])},{p(c['user_cluster_bootstrap_95CI'][1])}]",f"{c['Holm_corrected_p']:.4f}",f"{c['seeds_improved']}/3 · {c['users_improved']}/10"] for c in result['comparisons']]),'',
      '## 6. 结论与下一步','',conclusion,'',
      f"本轮TUNED：EER={p(tuned['EER'])}%、HTER={p(tuned['HTER'])}%；原参数：{p(orig['EER'])}%、{p(orig['HTER'])}%；纯ACC：{p(base['EER'])}%、{p(base['HTER'])}%。应分别回答是否优于原参数、是否优于纯ACC、是否优于同参数错配监督；三者不混用。",'',
      '本轮只从两套候选锁定一套进行最终复核，未查看查询后继续扫描λ、改变注册数量、挑用户或换阈值规则。后续若要确认独立Audio收益，应先冻结新假设并使用新录制/内容批次。匹配同词K=12诊断仍因每人每词独立窗口不足而不可比较。','',
      '## 7. 独立核验与交付','',
      f"核验通过：所有模型初始化/批次一致，标准化与原参考逐值相同；全部150阈值先锁定；移除训练头后重新提取特征和校准阈值，并由独立秩统计实现重放 {verify['independent_metric_fields']} 个指标字段。指标最大误差 {verify['metric_max_absolute_error']:.3g}，分数误差 {verify['score_max_absolute_error']:.3g}，阈值误差 {verify['threshold_max_absolute_error']:.3g}。原BASE/ORIGINAL参考指标重现最大差 {verify['previous_reference_max_error']:.3g}。",'',
      '公开文件仅含代码、协议、匿名聚合指标、日志和哈希。信号、后验、向量、模型、阈值数值及逐样本分数留在私有目录。','',
      '[复现](REPRODUCE.md) · [汇总](summary_public.json) · [参数选择](selection_lock_public.json) · [核验](verification_public.json)']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    slim={k:v for k,v in result.items() if k not in ['per_seed_user','per_seed_word_user']};slim['detail_files']=[f'{c}_anonymous_metrics_public.json' for c in codes];write('summary_public.json',slim)
    for c in codes:
        obj=dict(arm=c,units='fractions',per_seed_user=[r for r in users if r['arm']==c],per_seed_word_user=[r for r in wordrows if r['arm']==c])
        (OUT/f'{c}_anonymous_metrics_public.json').write_text(json.dumps(obj,separators=(',',':')),encoding='utf-8')
    src=Path(__file__).parent
    files=[p for p in src.iterdir() if p.is_file() and p.suffix in ['.py','.md']]
    files += [p for p in OUT.iterdir() if p.is_file() and p.suffix in ['.md','.json'] and p.name not in ['results_public.json','publication_manifest_public.json']]
    write('publication_manifest_public.json',dict(files=[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(files)],private_data_excluded=True))
    print('TUNING_REPORT_READY',flush=True)

if __name__=='__main__':
    main()
