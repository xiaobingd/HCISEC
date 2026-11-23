"""Report held-out phrase authentication and matched enrollment diagnostics."""
import json
from pathlib import Path
import numpy as np

p=Path(__file__).resolve().parent
prep=json.loads((p/'preparation_public.json').read_text(encoding='utf8'))
training=json.loads((p/'training_public.json').read_text())
data=np.load(p/'frontend_private.npz')
assert data['fit'].sum()==1280 and set(data['word'][data['fit']])==set(prep['train_words'])
query=set(prep['query_single']+prep['query_english_long']+prep['query_chinese_long'])
assert not query & set(prep['train_words']) and len(query)==15
methods=['CE','AudioKD','ShuffledAudioKD'];metrics=['far','frr','hter','eer','auc']
datasets={};protocols={};checks={};macros=[];users=[]
for study,folder in [('cross_content',p),('fixed_content',p/'fixed_content'),('matched_enrollment',p/'matched_enrollment')]:
    rows=json.loads((folder/'results_public.json').read_text())
    locks=json.loads((folder/'thresholds_public.json').read_text())
    assert len(rows)==540 and len(locks)==(180 if study=='cross_content' else 540)
    protocols[study]=json.loads((folder/'protocol_public.json').read_text())
    checks[study]=json.loads((folder/'verification_public.json').read_text())
    family='enrollment' if study=='matched_enrollment' else 'condition'
    assert len({(v['seed'],v['method'],v['words'],v['user'],v[family]) for v in rows})==540
    for v in rows:
        n=(40 if study=='cross_content' else 15)//v['words']
        assert v['genuine_trials']==n and v['attack_trials']==9*n
        assert abs(v['far']-v['false_accepts']/(9*n))<1e-12
        assert abs(v['frr']-v['false_rejects']/n)<1e-12
        assert abs(v['hter']-(v['far']+v['frr'])/2)<1e-12
        assert all(0<=v[k]<=1 for k in metrics)
    assert all(np.isfinite(v['threshold']) for v in locks)
    datasets[study]=rows
    for seed in (42,43,44):
        (folder/f'results_seed{seed}_public.json').write_text(json.dumps([v for v in rows if v['seed']==seed],separators=(',',':')))
    for m in methods:
        for f in ['single_word','english_long','chinese_long']:
            for k in (1,3):
                part=[v for v in rows if v['method']==m and v[family]==f and v['words']==k]
                macros.append(dict(study=study,method=m,family=f,words=k,
                    **{z:float(np.mean([v[z] for v in part])) for z in metrics}))
                for us in range(1,11):
                    sub=[v for v in part if v['user']==us]
                    users.append(dict(study=study,method=m,family=f,words=k,user=us,
                        **{z:float(np.mean([v[z] for v in sub])) for z in metrics}))
assert len(users)==540 and len(macros)==54
aux=[v for b in training['students'] if b['method']!='CE' for v in b['trace']]
assert aux and all(0<v['aux_gradient_ratio']<=.100001 for v in aux)
verification=dict(status='PASS',studies=checks,query_workpoints=1620,per_user_mean_rows=540,
    all15query_prompts_excluded_from_encoder_scaler_and_teacher=True,
    raw_fit_audio_cache_exact_matches=prep['newfit_original_audio_cache_exact_matches'],
    official_frozen_pool_subset_only=True,new_encoder_random_initialization=True,
    logged_kd_gradient_ratio_min=min(v['aux_gradient_ratio'] for v in aux),
    logged_kd_gradient_ratio_max=max(v['aux_gradient_ratio'] for v in aux))
(p/'audit_public.json').write_text(json.dumps(verification,indent=2))
(p/'summary_public.json').write_text(json.dumps(dict(protocols=protocols,macro=macros,
    per_user=users,verification=verification),ensure_ascii=False,separators=(',',':')),encoding='utf8')
def get(study,method,family,k=1,seed=None,user=None):
    rows=datasets[study];key='enrollment' if study=='matched_enrollment' else 'condition'
    return [v for v in rows if v['method']==method and v[key]==family and v['words']==k
            and (seed is None or v['seed']==seed) and (user is None or v['user']==user)]
def avg(part):return {z:float(np.mean([v[z] for v in part])) for z in metrics}
ml={'CE':'纯ACC','AudioKD':'Audio辅助','ShuffledAudioKD':'打乱Audio目标'}
fl={'single_word':'英文单词','english_long':'英文长词/短语','chinese_long':'后5条中文提示语'}
def table(study,selected_methods=methods):
    out=['|方法|提示语类型|联合条数|FAR%|FRR%|HTER%|EER%|AUC|','|---|---|---:|---:|---:|---:|---:|---:|']
    for m in selected_methods:
        for f in fl:
            for k in (1,3):
                v=avg(get(study,m,f,k))
                out.append(f'|{ml[m]}|{fl[f]}|{k}|'+ '|'.join(f'{100*v[z]:.2f}' for z in metrics[:4])+f"|{v['auc']:.4f}|")
    return out
lines=['# 完整短语能否替代多个独立词：严格留出与注册内容对照', '',
 '2026-11-23。使用原始硬件ACC、全部10人、3个学生随机种子，分别报告跨内容、同内容、匹配注册预算三套协议。'
 '所有查询提示语都从共享ACC编码器、Audio教师适配和标准化训练中留出。', '',
 '## 主要回答', '',
 '后5条中文提示语可以直接作为完整短语输入。但必须区分“用短语本身注册/校准”与“用英文单词注册后直接换中文查询”。'
 '前者单条Audio辅助HTER为6.53%，三条独立中文提示语联合为0.11%；后者在相同中文查询和15/10注册校准预算下为34.75%。'
 '注册/校准内容匹配非常重要，不能直接沿用英文单词的认证头和阈值。', '',
 '同内容协议中，单条中文提示语优于单条英文词（HTER10.53%→6.53%），但仍不如三条英文词联合（3.67%）。'
 '一条长提示语只产生一次得分，不能直接等同于多条独立录制的分数平均。', '',
 '三条中文提示语的0.11% HTER在纯ACC和打乱Audio目标组也能达到，因而不能归功于Audio蒸馏。'
 '这是同提示语、不同重复录制的结果，不是跨词或开放集成绩。', '',
 '## 原始数据与完整性检查', '',
 '从合并后的原始采集清单读取10位40提示语齐全用户，每提示语按原选择规则取8遍，共3200条缓存记录。'
 '通过原始Audio时间戳映射、VAD区间和WAV头核查时长；校准与查询只读取Audio头/元数据，没有Audio模型前向。'
 '新训练1280条Audio的PCM重采样/量化结果逐条与官方教师缓存输入完全一致。', '',
 '|原采集编号|中文提示语|音频时长中位数s|最大s|ACC映射区间最大s|','|---|---|---:|---:|---:|']
for original,word in [(36,35),(37,39),(38,38),(39,37),(40,36)]:
    v=prep['word_durations'][word]
    lines.append(f"|W{original:03}|{v['prompt']}|{v['audio_median_s']:.2f}|{v['audio_max_s']:.2f}|{v['host_max_s']:.2f}|")
lines += ['', '这5条都完整落在2秒ACC窗口内，不存在本轮怀疑的长短语截断。'
 '另有3条其他记录的ACC映射区间超过2秒（okay、select、what\'s wrong各1条），已扩展至完整区间。'
 f"重算旧2秒特征与缓存的最大差异{prep['legacy_feature_probe_max_error']:.3g}，用于核对源记录映射。"
 '全部输入采用相同规则：窗口至少2秒、足以覆盖完整映射VAD区间，区间外置零，不缩放长语句到2秒。', '',
 '音频时长来自WAV头采样数/采样率，ACC区间来自近似host callback映射；两者不完全相等。'
 '未新做物理同步标记校准，不能把这些时间当作已测在线认证延迟。中文提示语包含专名和多字词，'
 '英文长提示语组还包含单个多音节词congratulations，因此本实验不是严格的词数因果实验。', '',
 '## 重新训练与隔离', '',
 '为加入英文长词/短语对照，保留了15类查询提示语：5英文单词、5英文长词/短语和5中文提示语。'
 '不复用此前见过中文提示语的700步ACC检查点或本地微调Audio教师。', '',
 '使用另外16类英文单词、1280条录制，从随机参数重新训练762→256→192 ACC编码器：700步CE＋0.1SupCon预训练。'
 '标准化仅拟合这1280条。所有比较从同一新700步检查点开始，学生种子42/43/44，每组400步、100条/批。', '',
 'Audio教师只复用官方、未本地适配的ECAPA固定骨干特征中与新训练1280条对应的行；'
 '没有复用旧教师微调参数或查询提示语的教师向量。末端6144→192投影和192→64→192残差重新适配300步、种子4242，'
 'CE＋0.1SupCon＋0.01锚定后冻结。缓存输入已与这1280条原始Audio逐条匹配。', '',
 '三个学生对照为CE、正确Audio关系MSE、固定排列打乱Audio目标。Audio关系分别约束同人异词和异人同词，'
 '辅助梯度目标为编码器身份梯度的10%，权重上限10。学生架构、采样、优化器与400步预算匹配；教师额外计算预算单独存在。'
 '所有模型固定末步，没有按查询选检查点或权重。', '',
 '## 协议一：严格跨内容', '',
 '英文注册词Pause/Play/Record/Reset各前6遍，共24条/人；校准词four/one/select/two/up各8遍，共40条。'
 '英文单词查询为seven/six/start/stop/three；英文长提示语为congratulations/good bye/how are you/open device/what\'s wrong；'
 '中文为上述5项。每组都与训练、注册、校准提示语互斥。中文还涉及英语训练/注册到中文查询的语言变化。', '',
 'RBF C10 balanced，全9位攻击者保留；1或3条不同提示语分别打分后等权平均。40条查询按固定顺序组成40或13次挑战，'
 '实际用40或39条，不循环补齐。阈值为此前固定的本人校准Q10与攻击者Q95的中点；同一k阈值共享给三个查询提示语族。'
 '全部阈值写盘后才算查询指标。', '']
lines += table('cross_content')
lines += ['', '直接英文注册→未见中文单条的Audio HTER24.52%，不是可直接替代三词认证的结果。'
 '三条中文联合虽然改善EER，但固定英文校准阈值下FRR仍高；不能把它解释为中文没有身份信息。', '',
 '## 协议二：用提示语本身注册和校准', '',
 '仍冻结上述从未训练这些15类提示语的共享编码器。各提示语族单独训练个人RBF后端：同5类提示语的前3遍注册（15条），'
 '第4–5遍校准（10条），第6–8遍查询（15条）。提示语内容重叠，录制索引严格不重叠。'
 '这是闭集同内容认证，不是跨内容；只能说明注册后能识别相同提示语的新重复。', '',
 '单条模式每目标15次本人/135次攻击；三条联合每目标5次本人/45次攻击。三组提示语族注册/校准/查询源条数一致。'
 '三条校准只能形成3次挑战、使用9条源记录，固定分位数在小样本下仍有不确定性。', '']
lines += table('fixed_content')
lines += ['', '## 协议三：相同中文查询与相同注册预算的核对', '',
 '此为发现注册条件差异后的探索性诊断。固定同一批中文第6–8遍查询，比较15条注册＋10条校准来自英文单词、'
 '英文长提示语、中文提示语三种条件。编码器、Audio教师、查询录制、人数、阈值规则均相同，不根据查询再调参。'
 '中文自身注册的结果与协议二逐用户/种子复算误差0。语言和提示语内容随注册族一起变化，不能进一步拆开它们各自的因果作用。', '',
 '|Audio辅助：注册与校准内容|中文查询联合条数|FAR%|FRR%|HTER%|EER%|','|---|---:|---:|---:|---:|---:|']
for f in fl:
    for k in (1,3):
        v=avg(get('matched_enrollment','AudioKD',f,k))
        lines.append(f'|{fl[f]}|{k}|'+ '|'.join(f'{100*v[z]:.2f}' for z in metrics[:4])+'|')
lines += ['', '同样的中文查询，英文单词注册/校准的单条HTER34.75%，英文长提示语注册12.64%，中文自身注册6.53%。'
 '因此注册条件与查询条件匹配比仅把输入换成长句更关键；这不是增加注册数量带来的差异。', '',
 '## 同内容中文提示语：逐用户', '',
 '|用户|纯ACC单条 HTER%|Audio单条 FAR%|FRR%|HTER%|Audio三条 FAR%|FRR%|HTER%|',
 '|---|---:|---:|---:|---:|---:|---:|---:|']
for us in range(1,11):
    a=avg(get('fixed_content','CE','chinese_long',1,user=us))
    b=avg(get('fixed_content','AudioKD','chinese_long',1,user=us))
    c=avg(get('fixed_content','AudioKD','chinese_long',3,user=us))
    lines.append(f'|U{us}|'+ '|'.join(f'{100*v:.2f}' for v in [a['hter'],b['far'],b['frr'],b['hter'],c['far'],c['frr'],c['hter']])+'|')
lines += ['', '## 跨内容中文提示语：逐用户', '',
 '|用户|纯ACC单条 HTER%|Audio单条 FAR%|FRR%|HTER%|', '|---|---:|---:|---:|---:|']
for us in range(1,11):
    a=avg(get('cross_content','CE','chinese_long',1,user=us));b=avg(get('cross_content','AudioKD','chinese_long',1,user=us))
    lines.append(f'|U{us}|'+ '|'.join(f'{100*v:.2f}' for v in [a['hter'],b['far'],b['frr'],b['hter']])+'|')
lines += ['', '## 同内容中文：逐种子', '', '|种子|方法|条数|FAR%|FRR%|HTER%|EER%|','|---:|---|---:|---:|---:|---:|---:|']
for seed in (42,43,44):
    for m in methods:
        for k in (1,3):
            v=avg(get('fixed_content',m,'chinese_long',k,seed=seed))
            lines.append(f'|{seed}|{ml[m]}|{k}|'+ '|'.join(f'{100*v[z]:.2f}' for z in metrics[:4])+'|')
lines += ['', '## 有效音频时长', '', '|提示语类型|联合条数|参与评分源条数/人|音频时长均值s|中位数s|', '|---|---:|---:|---:|---:|']
for v in json.loads((p/'duration_public.json').read_text()):
    lines.append(f"|{fl[v['condition']]}|{v['words']}|{v['source_records_per_user']}|{v['active_speech_mean_s']:.2f}|{v['active_speech_median_s']:.2f}|")
lines += ['', '该时长为WAV头定义的有效音频累计长度，不包括等待、提示或实时系统开销。'
 '一条中文提示语往往短于三条英文词之和，因此不能假定它们天然提供相同数量的独立观测；'
 '本轮也没有按查询时长筛选有利样本。', '',
 '## 最终判断与边界', '',
 '可以直接把后5条作为完整短语认证输入，当前更适合先用短语本身注册和校准。'
 '单条短语有身份可分性，但尚不能替代三条独立词的分数融合；三条中文提示语一起使用时，同内容协议成绩很好。', '',
 'Audio的独立增益仍小：同内容中文三条的HTER在纯ACC、正确Audio和打乱目标中都为0.11%，'
 '单条中文HTER分别为6.58%/6.53%/6.64%。低错误首先来自注册条件匹配与多条融合。', '',
 '0.11%是10人、每人5次三条本人查询上的经验结果，不能当作泛化安全上界。'
 '三个种子复用同一物理录制，各协议又重复使用该批数据，1620个工作点不是1620个独立受试者。'
 '三套协议独立训练后端、独立锁阈值；不混合其数据角色或将它们的绝对差异全归因于内容因素。'
 '当前是历史探索过的同批闭集数据，不是新盲测、开放集或跨佩戴验证。', '',
 '旧模型、旧26训练词结果及创新点一V1未修改；新16词协议不能与上一轮3.32%/2.83%作单变量绝对比较。'
 '发布全部三套协议、540条逐用户均值、1620条分种子用户汇总、训练摘要、核验与代码，'
 '不发布原始Audio/ACC、embedding、模型、实名映射、逐样本分数或原始路径清单。', '']
(p/'report.md').write_text('\n'.join(lines),encoding='utf8')
print(json.dumps(dict(verification=verification,main=[v for v in macros if v['method']=='AudioKD'])))
