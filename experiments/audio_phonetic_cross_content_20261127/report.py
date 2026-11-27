"""Chinese report and public anonymous aggregate-only publication bundle."""
import json,hashlib,platform,importlib.metadata
from pathlib import Path
import numpy as np
from prepare_acc import OUT,PRIVATE,ROOT,ROLES

def read(name):return json.loads((OUT/name).read_text(encoding='utf-8'))
def mean(rows,key):return float(np.mean([r[key] for r in rows]))
def pct(x):return f'{100*x:.2f}'
def table(headers,rows):return '\n'.join(['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|']+['|'+'|'.join(map(str,row))+'|' for row in rows])

def main():
    result=read('results_public.json');verification=read('verification_public.json');assert verification['status']=='PASS'
    data=read('data_audit_public.json');teacher=read('teacher_audit_public.json');development=read('development_lock_public.json');vocab=read('role_split_public.json')['vocabulary']
    summary=result['summary'];rows=result['per_seed_user'];wordrows=result['per_seed_word_user'];arms=['T0','T1','T2','T3','T4']
    primary={r['arm']:r for r in summary if r['family']=='english_primary'}
    gain=primary['T0']['EER']-primary['T3']['EER'];gain_hter=primary['T0']['HTER']-primary['T3']['HTER']
    rank=min(primary,key=lambda a:primary[a]['EER'])
    if gain_hter>0 and all(primary['T3']['EER']<primary[a]['EER'] for a in ['T0','T2','T4']) and all(c['Holm_corrected_p']<.05 for c in result['comparisons']):
        decision='真实 Audio 组在主要跨词 EER 上优于三项主对照，用户配对检验通过三重校正。本轮支持沿该方向做独立新录制确认。'
    elif gain>0 and gain_hter<=0:
        decision='真实 Audio 组虽降低平均跨词 EER，但固定校准阈值下 HTER 没有改善，未同时满足区分性和工作点目标；不能只据 EER 将其定为最终认证方案。'
    elif gain>0:
        decision='真实 Audio 组的平均跨词 EER 低于纯 ACC；但尚未同时通过三项主对照和校正检验。下一步依据逐对照差值判断收益来源，不在这批查询词上继续搜索权重。'
    else:
        decision='真实 Audio 监督未降低主要跨词 EER，本轮不支持将该固定音素监督方案作为有效认证改进；保留纯 ACC 对照，停止在这批查询词上继续扫描辅助权重。'
    labels=dict(T0='纯 ACC 身份学习',T1='T0＋全局词 GRL',T2='T1＋词典伪音素',T3='T1＋真实 Audio 后验',T4='T1＋同后验时间错配')
    def summary_table(family):
        rr=[r for r in summary if r['family']==family]
        return table(['组','EER%','AUC','FAR%','FRR%','HTER%'],[[r['arm'],pct(r['EER']),f"{r['AUC']:.4f}",pct(r['FAR']),pct(r['FRR']),pct(r['HTER'])] for r in rr])
    lines=['# Audio 引导的分层跨内容 ACC 身份表征学习：完整执行结果','',
      f"{decision} 真实 Audio 相对纯 ACC 的 EER 改善为 **{pct(gain)} 个百分点**，HTER 改善为 **{pct(gain_hter)} 个百分点**（正数为改善）。主要 EER 最低的是 **{rank}**。",'',
      '已完成 5 个开发模型（每组 200 步）与 **5 组 × 3 种子 = 15 个最终模型**（每组 1000 步）。开发设置先锁定，全部模型完成后先锁定 150 个校准阈值，再评分查询。最终查询、注册及校准阶段均不读取 Audio。','',
      '## 1. 协议与数据','',
      '对应 GitHub 方案提交 `65f31238c0e3eeed7e19bac79fd6f1bbe3d58ce9`。本轮为 **10 个已见身份的闭集跨词认证**；所有身份参加共享训练，词/录制角色隔离。历史查询词已被过往实验查看，结果属于固定协议下的探索性复核。','',
      f"原始硬件三轴 ACC：400 个录制组，{data['actual_windows']} 个独立、不重叠的 3 秒连续窗口。名义 3200 次发声不等于连续窗口数。窗口以 IMU 起点加 0.5 秒确定，此后每 3 秒一个；不按 Audio 活动定位、裁剪、删静音或筛选用户。",'',
      '前端：逐轴去均值，插值到 1000 Hz 处理网格，100 ms STFT、20 ms hop，每轴 10–490 Hz 的 49 个频点取 log-power，共 147 通道、146 帧；仅使用允许的训练词有效帧计算全局标准化。所有组输入完全相同。','',
      table(['角色','零基词ID','实际词','窗口数'],[[role,','.join(map(str,words)),', '.join(vocab[str(w)] for w in words),data['actual_role_windows'][role]] for role,words in ROLES.items()]),'',
      '词 ID 沿用当前缓存的词典排序，并非采集 W001 顺序；已与各录制提示逐条核对。注册每人固定 12 条 ACC 窗口，4 个注册词各取前 3 个独立窗口；所有组使用同一注册索引。校准与查询均保留全部实际窗口及其余 9 个冒充身份。单个 3 秒窗口查询，不做多词联合。','',
      '## 2. 教师、模型与五组对照','',
      '教师为冻结的官方 `facebook/wav2vec2-lv-60-espeak-cv-ft`，不是 ECAPA embedding 蒸馏。精确版本 `ae45363bf3413b374fecd9dc8bc1df0e24c3b7f4`；权重 SHA256 `3173bde9e9ce490fa0f989e413c42f25bc1820c020adc1e6b9b87025b3cfcc5e`。已实际下载并校验，使用 float32 CPU 推理。','',
      '通过 eSpeak-ng en-us 生成完整 IPA token，使用真实 Wav2Vec2PhonemeCTCTokenizer 核验。16 个训练/开发词均可映射 392 类词表；不按字符拆音素。将已知词转写按采集元数据的重复次数展开，用真实 Audio logits 做精确 CTC forward-backward，保留软不确定性。该后验是自动对齐结果，并非人工音素边界。','',
      f"仅提取 160 个训练/开发录制，对应 {teacher['actual_windows']} 个 ACC 窗口、{teacher['valid_frames']} 个有效监督帧。注册/校准/查询 Audio 及其时戳零读取。音频头采样率与 PCM 回调时基独立检查后进行抗混叠重采样到 16k；原始 CTC 感受野中心映射至主机时间，再用固定 σ=100 ms Gaussian 软映射。",'',
      f"采集元数据的提示段数为 {sorted(set(r['repeats'] for r in teacher['records']))}；已按每条元数据分别展开，而非强行假设都恰好10遍。教师贪心音素序列相对该转写的编辑错误率中位数为 {np.median([r['greedy_phone_error_rate'] for r in teacher['records']]):.3f}、P95 为 {np.quantile([r['greedy_phone_error_rate'] for r in teacher['records']],.95):.3f}；它用于诊断教师/转写匹配质量，不据此筛用户或查询词。",'',
      f"时钟校验在 U09、词33发现全段线性拟合失真（回调间断，初次提取在第143条安全停止）。已在任何最终查询/阈值评分之前，以纯时戳规则修复：100回调块的中位速率选44.1/48k，超过预期75ms的回调间断分段，段内至少20回调且0.5秒、P95残差不超过30ms；CTC感受野跨界及不可信区域只屏蔽辅助项。全部160录制统一采用该规则，分段后分别Gaussian软映射，不跨缺失时间插值。共 {sum(r['callback_discontinuities'] for r in teacher['records'])} 个回调间断，{sum(r['masked_clock_segments'] for r in teacher['records'])} 个短/不可信时钟段被屏蔽；不删用户或身份窗口。已通过已知44.1/48k及300ms跳变合成测试。",'',
      '尚无物理同步事件标定，因此这是**粗时间尺度软监督**；没有依据查询成绩搜索时延。CTC blank 不是静音标签，不将 blank 准确率当作发声定位精度。','',
      'ACC 网络：147→64 Conv(k5)+GroupNorm+GELU→64 Conv(k3,dilation2)+GELU→64 Conv(k3,dilation3)+GELU。注意力均值/标准差池化 128→192→L2 身份表示；训练专用帧级 64→392 音素头、192→64→16 的 GRL 词头、归一化 10 类身份头。最终仅保留 ACC 主干/池化/身份投影、标准化、模板和阈值。','',
      table(['组','方法'],[[a,labels[a]] for a in arms]),'',
      '全部组：身份 CE＋0.1 跨词 SupCon（温度 0.1）。同人异词正对，异人负对，同人同词不进入此 SupCon 分母。词 GRL 系数 1、词 CE 外部系数 0.05；未对整个 CE 额外取负。音素项为全词表有效帧平均 KL，权重 0.1。','',
      'T2 只用词典，将词音素均匀放在固定 3 秒窗口的 0.5–2.5 秒，使用同一软化核，不借用 Audio 时长/边界。T4 在每窗口有效时间内做种子固定的非零循环移位，保持 mask、词表语义及窗口后验边际分布。T3/T4 辅助有效帧完全一致。','',
      'AdamW lr=0.0003、weight decay=0.01、clip=1；batch80=10人×4个共同训练词×2个独立窗口。种子42/43/44，同 seed 同初始权重、同批次、同预算、不加数据增强。先用12词优化、4词开发，在开发后保留原定损失权重，再用全部16词从头重训1000步，无查询选模型或早停。','',
      '## 3. 主要认证结果','',
      '英文短词与英文长短语分开评估；主要指标是两族等权、用户宏平均、再跨种子平均。EER/AUC 描述分数区分性；FAR/FRR/HTER 使用事先锁定的阈值 `(本人校准Q10＋冒充者校准Q95)/2`。低 FAR 与低 FRR 均报告，不用 EER 替代固定工作点。','',
      '### 英文等权主要结果','',summary_table('english_primary'),'',
      '### 英文短词','',summary_table('english_short'),'',
      '### 英文长短语','',summary_table('english_long'),'',
      '### 中文迁移（单独分析）','',summary_table('chinese'),'',
      '### 三个随机种子的主要结果','',
      table(['seed','组','EER%','FAR%','FRR%','HTER%'],[[seed,arm,*[pct(mean([r for r in rows if r['seed']==seed and r['arm']==arm and r['family']!='chinese'],k)) for k in ['EER','FAR','FRR','HTER']]] for seed in [42,43,44] for arm in arms]),'',
      '## 4. 全部逐用户结果','',
      '下表为英文两族等权、三个种子平均的 EER / HTER（%）。所有身份、所有冒充者均保留；逐用户 FAR、FRR、AUC 和逐词/逐种子完整指标见各组匿名 JSON。','',
      table(['匿名用户']+[a+' EER/HTER' for a in arms],[[f'U{user:02}']+[pct(mean([r for r in rows if r['user']==user and r['arm']==arm and r['family']!='chinese'],'EER'))+' / '+pct(mean([r for r in rows if r['user']==user and r['arm']==arm and r['family']!='chinese'],'HTER')) for arm in arms] for user in range(1,11)]),'',
      '## 5. 全部逐词结果','',
      '下表为三种子、全部用户宏平均 EER / HTER（%），不删除困难词。','',
      table(['词ID','词','族']+[a+' EER/HTER' for a in arms],[[word,vocab[str(word)],family]+[pct(mean([r for r in wordrows if r['word']==word and r['arm']==arm],'EER'))+' / '+pct(mean([r for r in wordrows if r['word']==word and r['arm']==arm],'HTER')) for arm in arms] for family in ['english_short','english_long','chinese'] for word in ROLES[family]]),'',
      '## 6. 配对统计与内容诊断','',
      '先按用户聚合两英文词族和三个种子；10 个用户作为统计簇，而非将重复词对当独立受试者。10,000 次用户簇 bootstrap、全部1024种用户符号置换，三项主比较做 Holm 校正。负差值表示 T3 更好。','',
      table(['比较','ΔEER百分点','用户簇95%CI（百分点）','精确p','Holm p','改善种子/用户'],[[c['comparison'],pct(c['mean_difference']),f"[{pct(c['user_cluster_bootstrap_95CI'][0])}, {pct(c['user_cluster_bootstrap_95CI'][1])}]",f"{c['exact_user_signflip_p']:.4f}",f"{c['Holm_corrected_p']:.4f}",f"{c['seeds_improved']}/3 · {c['users_improved']}/10"] for c in result['comparisons']]),'',
      '匹配同词/跨词退化量 `G_content` **不可比较**：连续窗口不足以在每人每词提供 12 个同词注册窗口以及独立查询。保持预先固定 K=12，不复用窗口、不在看到成绩后降低 K，因此没有虚构同词对照或内容退化数字。','',
      '冻结身份向量的词内容探针仅为解释：对全部16训练词，用奇偶用户两折训练正则 LogisticRegression(C=1)，报告 balanced accuracy（随机6.25%）；编码器本身见过全部身份，不称为开放集。未将未见词分类成训练词的准确率当内容能力。','',
      table(['组','训练词内容探针 balanced accuracy%'],[[a,pct(mean([r for r in result['content_probes'] if r['arm']==a],'train_word_balanced_accuracy'))] for a in arms]),'',
      '### 独立开发词的音素指标（200步，正式16词重训之前）','',
      table(['组','完整后验KL','非blank条件KL（质量加权）','非blank条件Top1%（质量加权）'],[[r['arm'],f"{r['full_posterior_KL']:.4f}",f"{r['nonblank_mass_weighted_conditional_KL']:.4f}",pct(r['nonblank_mass_weighted_top1'])] for r in development['models']]),'',
      f"教师映射后有效帧平均 blank 概率为 **{verification['teacher']['mean_valid_blank_probability']:.4f}**；非blank占多数的帧为 {verification['teacher']['nonblank_majority_frames']}。完整 KL 的下降可能主要来自 blank，因此同时报告非blank条件指标。无辅助有效帧窗口 {verification['teacher']['windows_without_auxiliary']} 个，仍保留身份训练。",'',
      table(['seed','T4平均帧后验L1变化','后验有变化窗口比例%'],[[seed,f"{read('final_seed'+str(seed)+'_shuffle_public.json')['mean_frame_L1_change']:.5f}",pct(read('final_seed'+str(seed)+'_shuffle_public.json')['nonconstant_window_fraction'])] for seed in [42,43,44]]),'',
      '## 7. 结果判断与下一步','',decision,'',
      f"本轮直接给出的证据是：在相同原始 ACC 输入、同预算、同模板和校准规则下，T3 的 EER={pct(primary['T3']['EER'])}%、HTER={pct(primary['T3']['HTER'])}%，T0 分别为 {pct(primary['T0']['EER'])}%、{pct(primary['T0']['HTER'])}%。T3 与词典组 T2、时间错配组 T4 的差值决定是否可以归因于真实 Audio 后验及时间对应，而不是仅依据 T3 与 T0 的单项差异。",'',
      f"T3 相对 T0 的平均 EER 差为 {pct(result['comparisons'][0]['mean_difference'])} 个百分点，用户簇95%区间为 [{pct(result['comparisons'][0]['user_cluster_bootstrap_95CI'][0])}, {pct(result['comparisons'][0]['user_cluster_bootstrap_95CI'][1])}]，Holm校正 p={result['comparisons'][0]['Holm_corrected_p']:.4f}。改善种子为 {result['comparisons'][0]['seeds_improved']}/3，改善用户为 {result['comparisons'][0]['users_improved']}/10。这个结论基于全部用户和种子。",'',
      f"相对单独词对抗 T1，三种底层音素监督的平均 EER 变化分别为：T2 {pct(primary['T2']['EER']-primary['T1']['EER'])}、T3 {pct(primary['T3']['EER']-primary['T1']['EER'])}、T4 {pct(primary['T4']['EER']-primary['T1']['EER'])} 个百分点；T1 相对 T0 变化为 {pct(primary['T1']['EER']-primary['T0']['EER'])} 个百分点。真实 Audio 对词典与时间错配的两项 EER 比较均需结合上表区间和校正检验解读。",'',
      f"独立开发词上，真实 Audio 组完整后验 KL={development['models'][3]['full_posterior_KL']:.3f}，时间错配组为 {development['models'][4]['full_posterior_KL']:.3f}。发音结构预测与最终身份认证是不同目标，应分别依据这两层实际结果判断收益。本轮 T0/T3 的认证 AUC 分别为 {primary['T0']['AUC']:.4f}/{primary['T3']['AUC']:.4f}；若 EER与AUC均退化，问题包含分数排序退化，单纯重新选择接受阈值不能恢复这两项指标。",'',
      '本轮五组并未单独识别“去掉 GRL 后的 Audio 效应”。仅当主对照支持实际增益时，才另行锁定无 GRL 消融；不将该后续设想混作本轮已完成证据。现有查询池不再扫描 λ；若需确认可重复性，采用新录制/新内容批次以及明确同步标记。','',
      '## 8. 独立核验与复现','',
      f"核验通过：{verification['model_count']} 个最终模型、150 个阈值、全部查询。逐模型重新提取 ACC 特征，独立秩统计实现重算 AUC/ROC-EER 与固定阈值指标；重放 {verification['reported_metric_fields_replayed']} 个指标字段，最大绝对差 {verification['metric_max_absolute_error']:.3g}。检查同初始权重、同批次、归一化仅训练角色、窗口不重叠、T3/T4 mask和边际保留、3个预定真实 CTC 对齐重放，以及移除训练头后部署输出一致。",'',
      '公开资料仅含源码、协议、版本/哈希、匿名用户/词汇聚合指标和诊断。原始音频/ACC、向量、模型参数、阈值数值、音素边界、身份映射和逐样本认证分数均留在私有目录。冻结创新点一和历史实验未修改。','',
      '[复现步骤](REPRODUCE.md) · [匿名汇总](summary_public.json) · [独立核验](verification_public.json) · [锁定协议](protocol.json)']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    slim={k:v for k,v in result.items() if k not in ['per_seed_user','per_seed_word_user']}
    slim['detail_files']=[f'{a}_anonymous_metrics_public.json' for a in arms]
    (OUT/'summary_public.json').write_text(json.dumps(slim,ensure_ascii=False,indent=2),encoding='utf-8')
    for arm in arms:
        bundle=dict(arm=arm,units='fractions',per_seed_user=[r for r in rows if r['arm']==arm],per_seed_word_user=[r for r in wordrows if r['arm']==arm])
        (OUT/f'{arm}_anonymous_metrics_public.json').write_text(json.dumps(bundle,ensure_ascii=False,separators=(',',':')),encoding='utf-8')
    versions={k:importlib.metadata.version(k) for k in ['torch','numpy','scipy','pandas','scikit-learn','threadpoolctl','packaging','PyYAML']}
    (OUT/'software_public.json').write_text(json.dumps(dict(python=platform.python_version(),platform=platform.platform(),packages=versions,device='cpu',training_threads=4),indent=2),encoding='utf-8')
    src=ROOT/'experiments/audio_phonetic_cross_content_20261127'
    sources=[p for p in src.iterdir() if p.is_file() and p.suffix in ['.py','.ps1','.md']]
    names=['data_audit_public.json','role_split_public.json','lexicon_public.json','teacher_audit_public.json','teacher_model_public.json',
      'development_lock_public.json','protocol.json','threshold_lock_public.json','query_phase_public.json','verification_public.json',
      'summary_public.json','software_public.json','report.md','REPRODUCE.md','input_verification_public.json']
    outputs=[OUT/name for name in names]+list(OUT.glob('train_*_public.json'))+list(OUT.glob('*_shuffle_public.json'))+list(OUT.glob('T?_anonymous_metrics_public.json'))
    files=[]
    for path in sorted(sources+outputs):
        assert path.is_file() and 'private' not in path.name and path.suffix in ['.py','.ps1','.md','.json']
        files.append(dict(path=path.relative_to(ROOT).as_posix(),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    (OUT/'publication_manifest_public.json').write_text(json.dumps(dict(files=files,private_signals_embeddings_models_thresholds_scores_excluded=True),indent=2),encoding='utf-8')
    print('REPORT_COMPLETE',OUT/'report.md',flush=True)

if __name__=='__main__':main()
