"""Complete nonblank-conditional experiment report and anonymous publication manifest."""
import json,hashlib
from pathlib import Path
import numpy as np
from run import ROOT,OUT,CONFIGS,write

def read(name):return json.loads((OUT/name).read_text(encoding='utf-8'))
def pct(x):return f'{100*x:.2f}'
def mean(rows,k):return float(np.mean([r[k] for r in rows]))
def table(head,rows):return '\n'.join(['|'+'|'.join(head)+'|','|'+'|'.join(['---']*len(head))+'|']+['|'+'|'.join(map(str,r))+'|' for r in rows])

def main():
    result=read('results_public.json');verification=read('verification_public.json');assert verification['status']=='PASS'
    lock=read('selection_lock_public.json');audit=read('teacher_conditioning_audit_public.json');rows=result['per_seed_user'];words=result['per_seed_word_user'];summary=result['summary']
    codes=['BASE','MATCHED','ORIGINAL','TUNED','SHUFFLED'];labels=dict(BASE='原纯ACC',MATCHED='词对抗.01，无Audio',ORIGINAL='完整392类KL，λ=.01',TUNED='非blank条件391类KL，λ=.01',SHUFFLED='条件KL，同参数联合时间错配')
    primary={r['arm']:r for r in summary if r['family']=='english_primary'};c,b,f=primary['TUNED'],primary['BASE'],primary['ORIGINAL']
    if c['EER']<b['EER'] and c['HTER']<b['HTER']:
        conclusion='去blank条件监督的平均EER和HTER均低于纯ACC；是否构成独立Audio增益，继续依据匹配无Audio、同权重完整KL、时间错配及四重校正统计判断。'
    elif c['EER']<f['EER'] and c['HTER']<f['HTER']:
        conclusion='条件监督优于同权重完整KL，但未同时超过纯ACC。修正blank监督带来了损失设计层面的改善，尚未满足最终认证增益目标。'
    else:conclusion='本轮去blank条件监督未同时改善同权重完整KL与纯ACC的EER/HTER，不能将该改动定为有效的身份认证增强。'
    def result_table(family):
        rr={r['arm']:r for r in summary if r['family']==family}
        return table(['方法','EER%','AUC','FAR%','FRR%','HTER%'],[[code,pct(rr[code]['EER']),f"{rr[code]['AUC']:.4f}",*[pct(rr[code][k]) for k in ['FAR','FRR','HTER']]] for code in codes])
    vocab=json.loads((ROOT/'outputs/vr261127_audio_phonetic_cross_content/role_split_public.json').read_text(encoding='utf-8'))['vocabulary']
    text=['# 去blank有效音素条件蒸馏：闭集跨词认证受控实验','',conclusion,'',
      f"条件KL：EER **{pct(c['EER'])}%**、HTER **{pct(c['HTER'])}%**；同权重完整KL：{pct(f['EER'])}%、{pct(f['HTER'])}%；纯ACC：{pct(b['EER'])}%、{pct(b['HTER'])}%。",'',
      '## 1. 本轮修正什么','',
      '前两轮完整CTC后验中blank概率约95%。本轮按用户方案，直接让辅助目标关注非blank音素，不再将blank预测当作内容学习效果。保留原冻结教师及全部数据，未更换研究对象、网络或传感器前端。','',
      '这是接在损失权重探索提交 `53be0e2539dcecfb9637656ae7b0f645946ffc03` 之后的独立新目录实验。旧代码、既有模型、历史报告与创新点一不覆盖。','',
      '## 2. 精确损失与可信度','',
      '令m_t=Σ(k≠blank)p_teacher(t,k)，教师条件目标q_t(k)=p_teacher(t,k)/m_t；用实际非blank概率求和避免1−p_blank的float32消减误差。学生目标为log_softmax(student_logits[...,1:])，同样排除blank。只改教师归一化而保留学生392类softmax不是本轮实现。','',
      '权重 w_t = m_t × max_k q_t(k) × valid_t × 1[m_t≥0.001]；损失 L_cond=Σw_t KL(q_teacher||q_student)/Σw_t。0.001在模型评分前固定，用于屏蔽极低质量后验；无有效权重时辅助损失精确为0，身份损失仍保留全部窗口。教师和权重均离线冻结，不反传。','',
      '这里的可信度是CTC条件后验代理，不是人工音素置信度或物理发声真值；blank不等于静音。全部查询窗口仍由IMU起点确定，查询不读Audio、不用Audio裁剪或VAD定位。','',
      f"教师审计：{audit['windows']} 个训练/开发窗口，{audit['valid_teacher_frames']} 个有效教师帧，{audit['positive_weight_frames']} 帧有正条件权重；无条件辅助监督窗口 {audit['windows_without_conditional_supervision']} 个，仍全部参加身份训练。有效帧平均非blank概率质量 {audit['mean_valid_nonblank_mass']:.5f}，正权重帧平均条件最大后验 {audit['mean_positive_conditional_confidence']:.5f}。",'',
      '合成测试通过：已知两帧加权KL的解析值；教师和学生均条件化；改变学生blank分数不影响损失；blank辅助梯度精确为0；极低质量帧屏蔽；全blank时有限零损失/零梯度；错配时后验和权重联合移位。','',
      '## 3. 固定对照与协议','',
      table(['代号','方法'],[[code,labels[code]] for code in codes]),'',
      '本轮固定音素λ=0.01、词对抗λ=0.01，没有扫描参数。完整KL与条件KL使用相同λ，条件错配与正确条件监督使用相同后验边际、mask及权重分布；错配后重新计算条件目标和权重，相当于二者联合时间置换，防止只移动标签却保留正确活动权重。','',
      'BASE为既有纯ACC；MATCHED复用上轮词对抗.01的无Audio模型。四个开发模型MATCHED/完整KL/条件KL/条件错配均为seed42、400步、12词优化、4词循环注册/校准/查询（K4），只作训练与开发诊断，不据此改变本轮λ。','',
      table(['开发组','EER%','FAR%','FRR%','HTER%'],[[code,*[pct(lock['summaries'][code][k]) for k in ['EER','FAR','FRR','HTER']]] for code in CONFIGS]),'',
      '正式使用全部16训练词、1000步、种子42/43/44。新增9个模型（3监督×3种子），既有BASE和MATCHED只读复用，共15个方法/种子评估条目。架构仍为147→64卷积/TDNN→注意力均值标准差池化→192维L2，帧级头392维；CE+.1跨词SupCon、AdamW lr.0003/wd.01/clip1、batch80、3秒连续ACC窗口、100ms教师时间核均不改。','',
      '10人闭集、40词角色保持原协议；每人12条固定注册，原校准词锁定全部150阈值后才进行查询。全部15个查询词、913个实际查询窗口和其余9个攻击身份保留。单窗口ACC-only认证。英文短词/长短语等权是主要指标，中文单独报告。历史查询池已被过往实验查看，本轮仍属探索性复核，不能称全新盲测或未见身份实验。','',
      '## 4. 正式结果','',
      '### 英文等权主要结果','',result_table('english_primary'),'',
      '### 英文短词','',result_table('english_short'),'',
      '### 英文长短语','',result_table('english_long'),'',
      '### 中文迁移','',result_table('chinese'),'',
      '### 逐种子英文主要结果','',
      table(['seed','方法','EER%','FAR%','FRR%','HTER%'],[[seed,code,*[pct(mean([r for r in rows if r['seed']==seed and r['arm']==code and r['family']!='chinese'],k)) for k in ['EER','FAR','FRR','HTER']]] for seed in [42,43,44] for code in codes]),'',
      '## 5. 逐用户与逐词','',
      '以下为EER/HTER（%），均跨三种子平均；用户表是英文两族等权。每位用户、每个查询词的FAR、FRR、AUC及逐种子数据见全部五份匿名JSON。','',
      table(['用户']+codes,[[f'U{user:02}']+[pct(mean([r for r in rows if r['user']==user and r['arm']==code and r['family']!='chinese'],'EER'))+'/'+pct(mean([r for r in rows if r['user']==user and r['arm']==code and r['family']!='chinese'],'HTER')) for code in codes] for user in range(1,11)]),'',
      table(['词ID','词']+codes,[[word,vocab[str(word)]]+[pct(mean([r for r in words if r['word']==word and r['arm']==code],'EER'))+'/'+pct(mean([r for r in words if r['word']==word and r['arm']==code],'HTER')) for code in codes] for word in [25,26,27,28,29,3,9,12,19,32,35,36,37,38,39]]),'',
      '## 6. 四项主比较','',
      '先按用户聚合两个英文词族和三个种子；10个用户为统计簇，10000次用户簇bootstrap、1024种精确符号置换，四项比较统一Holm校正。负差表示条件监督更好。重复词/窗口不当作独立受试者。','',
      table(['比较','ΔEER百分点','95%用户簇CI','Holm p','改善种子/用户'],[[r['comparison'],pct(r['mean_difference']),f"[{pct(r['user_cluster_bootstrap_95CI'][0])},{pct(r['user_cluster_bootstrap_95CI'][1])}]",f"{r['Holm_corrected_p']:.4f}",f"{r['seeds_improved']}/3 · {r['users_improved']}/10"] for r in result['comparisons']]),'',
      '## 7. 判断与交付','',conclusion,'',
      '应区分条件监督是否优于同权重完整KL，以及是否真正超过无Audio和时间错配的认证能力。去掉blank是对监督目标的修正，认证是否有增益由上述实际结果判断。EER/AUC体现分数区分性；FAR/FRR/HTER使用锁定阈值；本轮不在查询分数上改λ或阈值规则。匹配同词K12诊断仍因独立窗口不足而不可比较。','',
      f"独立重放通过：15个方法/种子、150个阈值、{verification['independent_metric_fields']}个指标字段。指标最大差 {verification['metric_max_absolute_error']:.3g}；分数差 {verification['score_max_absolute_error']:.3g}，阈值差 {verification['threshold_max_absolute_error']:.3g}。复用BASE/MATCHED与上轮公开参考指标最大差 {verification['previous_reference_max_error']:.3g}。全部初始化/批次/前端/注册条件一致，部署只保留ACC编码器，不含Audio或训练辅助头。",'',
      '仅公开源码、协议、日志、哈希与匿名聚合指标；信号、条件后验、权重时间序列、向量、模型、阈值数值及逐样本分数均私有保存。','',
      '[复现](REPRODUCE.md) · [汇总](summary_public.json) · [教师条件化审计](teacher_conditioning_audit_public.json) · [核验](verification_public.json)']
    diagnostic=OUT/'phoneme_diagnostic_public.json'
    if diagnostic.exists():
        diag=read(diagnostic.name)
        text += ['', '## 附：独立开发词的音素学习', '',
          '四组400步模型均只训练12优化词，在4个独立开发词上比较同一套条件目标及权重。下列一致率是与自动教师后验的对照，不是人工标注音素准确率；未用于本轮权重选择。', '',
          table(['组','条件加权KL','教师argmax一致率%','学生argmax对应教师概率','最集中预测及权重占比'],[[r['arm'],f"{r['conditional_weighted_KL']:.4f}",pct(r['teacher_argmax_agreement']),f"{r['teacher_probability_at_student_argmax']:.4f}",f"`{r['predicted_phone_modes'][0]['symbol']}` {pct(r['predicted_phone_modes'][0]['weight_fraction'])}%"] for r in diag['rows']]),'',
          '音素损失下降需与预测分布一起判断。若开发词预测高度集中于少数类别，不能仅由去blank后的KL降低认定已经学会跨词音素解码；该诊断只描述400步开发模型，最终身份结论仍使用上面的1000步三种子认证结果。']
    (OUT/'report.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    slim={k:v for k,v in result.items() if k not in ['per_seed_user','per_seed_word_user']};slim['detail_files']=[f'{code}_anonymous_metrics_public.json' for code in codes];write('summary_public.json',slim)
    for code in codes:
        obj=dict(arm=code,units='fractions',per_seed_user=[r for r in rows if r['arm']==code],per_seed_word_user=[r for r in words if r['arm']==code])
        (OUT/f'{code}_anonymous_metrics_public.json').write_text(json.dumps(obj,separators=(',',':')),encoding='utf-8')
    files=[p for p in Path(__file__).parent.iterdir() if p.is_file() and p.suffix in ['.py','.md']]
    files += [p for p in OUT.iterdir() if p.is_file() and p.suffix in ['.md','.json'] and p.name not in ['results_public.json','publication_manifest_public.json']]
    write('publication_manifest_public.json',dict(files=[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(files)],private_data_excluded=True))
    print('CONDITIONAL_REPORT_READY',flush=True)

if __name__=='__main__':main()
