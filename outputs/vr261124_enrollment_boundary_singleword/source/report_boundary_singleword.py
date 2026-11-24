import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'conditional_correspondence/enrollment_boundary_singleword'
WORKSPACE=ROOT.parents[1]
FAMILIES={'english_word':'英文单词','english_long':'英文长词／短语','chinese_long':'中文词／短语'}
METHODS=['acc_crossword12','audio_boundary','coin_boundary']+[f'shuffled_boundary_{i}' for i in range(5)]

def main():
    s=json.loads((OUT/'summary_public.json').read_text());v=json.loads((OUT/'verification_public.json').read_text())
    assert v['baseline_per_seed_user_max_difference']<1e-8 and v['independent_replay_max_error']<1e-10
    rows=json.loads((OUT/'per_seed_user_public.json').read_text())
    word=json.loads((OUT/'per_word_summary_public.json').read_text(encoding='utf8'))
    wordrows=json.loads((OUT/'per_word_seed_user_public.json').read_text(encoding='utf8'))
    assert len(wordrows)==7200 and len(word['macro'])==240 and len(word['per_user'])==2400
    errors=[]
    for a in rows:
        sub=[b for b in wordrows if all(b[k]==a[k] for k in ['seed','method','backend','family','user'])]
        assert len(sub)==5
        for key in ('far','frr','hter'):errors.append(abs(a[key]-np.mean([b[key] for b in sub])))
    assert max(errors)<1e-12
    v.update(per_word_rows=7200,per_word_macro_rows=240,per_word_user_rows=2400,wordwise_operating_rates_recombine_max_error=max(errors))
    (OUT/'verification_public.json').write_text(json.dumps(v,indent=2))
    export=[('per_user_public.csv',s['per_user']),('per_seed_user_public.csv',rows),('per_word_macro_public.csv',word['macro']),('per_word_user_public.csv',word['per_user']),('per_word_seed_user_public.csv',wordrows)]
    for name,data in export:
        with (OUT/name).open('w',newline='',encoding='utf8') as file:
            writer=csv.DictWriter(file,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    def find(method,family,backend='RBF'):return next(a for a in s['macro'] if a['method']==method and a['family']==family and a['backend']==backend)
    def pct(a):return '|'.join(f'{100*a[key]:.3f}' for key in ('far','frr','hter','eer','auc'))
    status=[]
    for backend in ('RBF','cosine'):
        for family in FAMILIES:
            real=find('audio_boundary',family,backend);base=find('acc_crossword12',family,backend);coin=find('coin_boundary',family,backend)
            better={metric:sum(real[metric]<find(m,family,backend)[metric] for m in METHODS[3:]) for metric in ('hter','eer')}
            passed=all(real[k]<base[k] and real[k]<coin[k] and better[k]>=3 for k in ('hter','eer'))
            status.append(dict(backend=backend,family=family,beats_shuffles_hter=better['hter'],beats_shuffles_eer=better['eer'],point_criteria_pass=passed))
    assert not any(a['point_criteria_pass'] for a in status)
    (OUT/'stop_decision_public.json').write_text(json.dumps(dict(stop_current_audio_ranking_registration=True,search_fusion_weights=False,
        reason='No content/backend satisfies simultaneous HTER/EER win over ACC,coin andmajorityof5shuffles;cross-user/seed/backend success not established',by_condition=status),indent=2))
    program=WORKSPACE/'experiments/enrollment_audio_boundary_20261124/enrollment_audio_boundary_singleword.py'
    (OUT/'execution_manifest_public.json').write_text(json.dumps(dict(upstream_branch='experiment/audio-enrollment-boundary-20261124',upstream_commit='59dfd0e99f19688a479171b186af196070e9499b',
        source_sha256={str(p.relative_to(WORKSPACE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [program,ROOT/'verify_boundary_singleword.py',ROOT/'evaluate_boundary_per_word.py',ROOT/'report_boundary_singleword.py']},
        modification='only query k1,workpoint/threshold counts and descriptive protocolmetadata;original selector/backend/regression gate unchanged'),indent=2))
    lines=['# 最新Audio边界注册方案：单词查询实验','', '2026-11-24。按GitHub专用实验分支方案执行；仅评估单条查询，不做三词分数融合。',
        '', '[上游方案README](https://github.com/xiaobingd/HCISEC/blob/59dfd0e99f19688a479171b186af196070e9499b/experiments/enrollment_audio_boundary_20261124/README.md)，锁定提交59dfd0e99f19688a479171b186af196070e9499b。主分支的旧docs/下一步实验.md不是本轮使用的方案。',
        '', '## 实际结论与停止决定','', '正确Audio在英文单词RBF的HTER/EER略低于ACC排名，但HTER只优于2/5打乱、EER只优于1/5打乱；英文长词/短语退化；中文EER也高于ACC。余弦后端HTER在三类上均退化。没有条件满足同时超过ACC/随机/多数打乱的双指标要求。',
        '', '**按上游成功/停止标准，停止当前Audio排序辅助注册方案，不继续搜索融合权重。** 不把该结果解释为所有Audio注册方法都无效。',
        '', '## “单个单词”的执行口径','', '每次认证只使用一个查询记录的ACC分数，不平均不同词或不同记录。注册按方案保留四词，每词3条，共12条；本人其他三个注册词的Audio原型用于评分，不能把注册也改成单词后仍声称复现同一方案。',
        '', '主单词内容组为seven、six、start、stop、three。其余英文长词/短语及中文词/短语按上游方案单列诊断，不合并成纯英文单词指标。所有15个查询标签另有逐词结果，没有按结果挑一个最好词。',
        '', '输入沿用方案要求的原3秒ACC缓存；窗口可能包含邻近同词重复，不能将本轮称为“重新切出且保证只含一次发声”的波形实验。其余物理同步/VAD中心限制不变。',
        '', '## 固定方法','', '10位已见用户，冻结ACC种子42/43/44；注册[20,21,22,23]每词8条共32条。按样本对本人其他3注册词ACC原型的余弦排序，固定前2条；第三条仅在ACC第3/4名之间选择。',
        '', '|方法|第三条选择|','|---|---|','|acc_crossword12|ACC第3名|','|audio_boundary|第3/4名中Audio对本人其他3注册词原型一致性更高者|','|coin_boundary|固定种子抛硬币|','|shuffled_boundary_0..4|五组无固定点的同人同词Audio分数置换后选择|',
        '', 'Audio仅使用既有320条注册ECAPA192缓存，无新PCM读取，无校准/查询Audio。所有方法每人12条正类；其他9人共288条注册ACC负类固定。主RBF C10 balanced；次后端平均余弦原型。没有更新全局ACC或Audio网络。',
        '', '校准词[8,18,24,30,31]与注册/查询词互斥。每内容组5个查询标签×8遍，每人40条本人及360条攻击比较；所有攻击来源保留。480个校准阈值先锁定，随后计算1440个用户/种子/后端/方法/内容组工作点。阈值固定为本人Q10与攻击Q95中点。',
        '', '## RBF主结果','', '全部为百分数，先平均三个种子再等权十用户。EER由查询ROC计算；FAR/FRR/HTER使用校准阈值。','','|内容|方法|FAR|FRR|HTER|EER|AUC|','|---|---|---:|---:|---:|---:|---:|']
    for family in FAMILIES:
        for method in METHODS:lines.append(f'|{FAMILIES[family]}|{method}|{pct(find(method,family))}|')
    lines+=['','## 余弦后端诊断','','|内容|方法|HTER%|EER%|','|---|---|---:|---:|']
    for family in FAMILIES:
        for method in ('acc_crossword12','audio_boundary','coin_boundary'):
            a=find(method,family,'cosine');lines.append(f"|{FAMILIES[family]}|{method}|{100*a['hter']:.3f}|{100*a['eer']:.3f}|")
    lines+=['','## 逐词RBF结果：同一已锁定阈值','','每个词、每个目标有8本人/72攻击；逐词EER重新由该词子集ROC计算，不等于五词混合EER的可加分解。FAR/FRR/HTER等数量平均可精确重合并；所有阈值均保持原校准值，不为词单独搜索。','','|查询词|ACC HTER%|Audio HTER%|ACC EER%|Audio EER%|','|---|---:|---:|---:|---:|']
    for wid in sorted(set(a['query_word_id'] for a in word['macro'])):
        a=next(a for a in word['macro'] if a['method']=='acc_crossword12' and a['backend']=='RBF' and a['query_word_id']==wid)
        b=next(a for a in word['macro'] if a['method']=='audio_boundary' and a['backend']=='RBF' and a['query_word_id']==wid)
        lines.append(f"|{a['query_word']}|{100*a['hter']:.3f}|{100*b['hter']:.3f}|{100*a['eer']:.3f}|{100*b['eer']:.3f}|")
    lines+=['','## 逐用户与种子','','|内容|用户|Audio FAR|Audio FRR|Audio HTER|Audio EER|','|---|---|---:|---:|---:|---:|']
    for family in FAMILIES:
        for us in range(1,11):
            a=next(a for a in s['per_user'] if a['method']=='audio_boundary' and a['backend']=='RBF' and a['family']==family and a['user']==us)
            lines.append(f'|{FAMILIES[family]}|U{us}|'+ '|'.join(f'{100*a[k]:.3f}' for k in ('far','frr','hter','eer'))+'|')
    lines+=['','全部480条逐用户均值、1440条逐种子指标、240个逐词方法/后端汇总、2400条逐词用户均值及7200条逐词种子用户结果均公开为CSV。逐词分层重复使用同一查询，7200不是新增独立样本。',
        '', 'Audio共替换60/120个边界决策，等于三种子合计360个保留位置中的60个（16.7%），平均每位用户每种子替换2条。抛硬币替换57；五组打乱替换63/71/66/44/63。所有方法的前两名均保持，第三名只在3/4名中选。逐种子换出数见swap_counts_public.json。',
        '', '十位目标用户簇bootstrap区间见summary_public.json的comparisons；不把重复、种子或试验次数当独立受试者。英文单词HTER差−0.157个百分点，95%区间[−0.819,+0.324]；EER差−0.097个百分点，区间[−0.537,+0.315]，未形成稳定收益。',
        '', '## 验证与公开范围','', 'ACC跨词基线按上游要求复现，宏指标和全部逐用户/种子指标最大差0（要求<1e-8）；独立RBF/余弦首用户重放误差0。每人每词3条预算、所有方法前2条不变、最后一条只来自3/4名均验证通过。逐词FAR/FRR/HTER重合并误差<1e-12。',
        '', '这是已见用户、历史查看过查询内容的探索性实验，不是未见身份开放集或新盲测。仍需采集/读取全部注册Audio，不能将相同12条ACC写成相同传感器成本。仅公开匿名汇总、代码及换出数量，不上传PCM、embedding、身份映射、选样ID、权重、阈值或逐样本分数。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print('BOUNDARY_SINGLEWORD_REPORT_READY',max(errors),flush=True)

if __name__=='__main__':main()
