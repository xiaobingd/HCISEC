"""Independently replay closed-set results and reconcile role/mask accounting."""
import json,itertools,hashlib
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from train_phoneme_conditioned_identity import BASE,OUT,PRIVATE,TRAIN,QUERY,ARMS,Net,embed
from run_continuous import choose_auth,measure
from audit_local_identity_residual import holm

def read(name):return json.loads((OUT/name).read_text(encoding='utf-8'))
def main():
    torch.set_num_threads(2);d=np.load(PRIVATE/'prepared_private.npz');c=np.load(PRIVATE/'inputs_private.npz')
    assert np.array_equal(d['indices'],c['indices'])
    x=torch.tensor((c['x']-d['mean'])/d['scale']);u,w=d['u'],d['w'];fit=d['fit'];reg=d['registration'];cal=d['calibration'];query=d['query']
    assert np.all(np.stack([fit,reg,cal,query]).sum(0)==1) and [int(v.sum()) for v in [fit,reg,cal,query]]==[240,120,120,240]
    assert np.isnan(d['Audio_teacher'][~fit]).all() and np.isfinite(d['Audio_teacher'][fit]).all()
    assert not set(w[fit])&set(w[query]) and np.allclose(d['mean'],c['x'][fit].mean((0,1)),atol=1e-6)
    rows=read('per_seed_user_public.json');wordrows=read('per_word_user_public.json');users=read('per_user_public.json');summary=read('summary_public.json')
    assert len(rows)==120 and len(wordrows)==360 and len(users)==40 and len(summary)==4
    protocol=read('protocol_public.json');ph=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    lockpath=PRIVATE/'thresholds_private.json';locks=json.loads(lockpath.read_text());assert len(locks)==120 and lockpath.stat().st_mtime<=(OUT/'per_seed_user_public.json').stat().st_mtime
    ri,ci,qi=[np.flatnonzero(v) for v in [reg,cal,query]];maxerr=0.
    for seed in [42,43,44]:
      b=np.load(PRIVATE/f'seed{seed}_batches_private.npz');batch=b['batches'];assert batch.shape==(400,60) and fit[batch].all()
      assert np.all(np.sort(u[batch],axis=1)==np.repeat(np.arange(1,11),6))
      for us in range(1,11):
        assert sorted(b['pseudo'][fit&(u==us)].tolist())==sorted(d['phone'][fit&(u==us)].tolist())
        assert not np.array_equal(b['pseudo'][fit&(u==us)],d['phone'][fit&(u==us)])
      for arm in ARMS:
        p=torch.load(PRIVATE/f'{seed}_{arm}_private.pt',weights_only=False);assert p['steps']==400 and p['protocol_sha256']==ph and [v['step'] for v in p['trace']]==[100,200,300,400]
        net=Net();net.load_state_dict(p['model']);rz=embed(net,x,ri);cz=embed(net,x,ci);qz=embed(net,x,qi)
        proto=np.stack([rz[u[ri]==us].mean(0) for us in range(1,11)]);proto/=np.maximum(np.linalg.norm(proto,axis=1,keepdims=True),1e-12)
        for us in range(1,11):
          tau=next(v['tau'] for v in locks if v['seed']==seed and v['arm']==arm and v['user']==us)
          assert tau==choose_auth(cz@proto[us-1],u[ci]==us)
          score=qz@proto[us-1];got=measure(score,u[qi]==us,tau);expected=next(v for v in rows if v['seed']==seed and v['arm']==arm and v['user']==us)
          assert expected['genuine_trials']==24 and expected['attack_trials']==216
          for key in ['eer','hter','far','frr','auc']:
            err=abs(got[key]-expected[key]);maxerr=max(maxerr,err);assert err<1e-10
          for wo in QUERY:
            mask=w[qi]==wo;got=measure(score[mask],u[qi][mask]==us,tau);expected=next(v for v in wordrows if v['seed']==seed and v['arm']==arm and v['user']==us and v['word']==wo)
            assert expected['genuine_trials']==8 and expected['attack_trials']==72
            for key in ['eer','hter','far','frr','auc']:assert abs(got[key]-expected[key])<1e-10
    for v in summary:
      rr=[a for a in rows if a['arm']==v['arm']]
      for key in ['eer','hter','far','frr','auc']:assert np.isclose(v[key],np.mean([a[key] for a in rr]),atol=1e-12)
    comparisons=[];signs=np.array(list(itertools.product([-1,1],repeat=10)))
    for control in ['B0_identity','B1_dictionary_ACC','B3_shuffled_phoneme']:
      delta=np.array([next(v['eer'] for v in users if v['user']==us and v['arm']=='B2_Audio_phoneme')-next(v['eer'] for v in users if v['user']==us and v['arm']==control) for us in range(1,11)])
      comparisons.append(dict(control=control,eer_delta=float(delta.mean()),improved_users=int((delta<0).sum()),p=float(np.mean((signs*delta).mean(1)<=delta.mean()+1e-12))))
    for v,p in zip(comparisons[:2],holm([v['p'] for v in comparisons[:2]])):v['holm_primary_p']=float(p)
    result=dict(status='PASS',models_replayed=12,personal_workpoints_replayed=120,per_word_metrics_replayed=360,max_metric_error=maxerr,
      used_ACC_role_support='920ms per recording;min center gap>'+str(protocol['minimum_repeat_center_gap_s']),Audio_teacher_outside_fit_missing=True,
      matched_batches_checked=1200,nontrivial_balanced_phone_shuffle=True,closed_identity_training=True,query_input_ACC_only=True,legacy_segmentation=True,final_raw_Audio_opened=False)
    (OUT/'verification_public.json').write_text(json.dumps(result,indent=2));(OUT/'comparison_public.json').write_text(json.dumps(comparisons,indent=2))
    names={'B0_identity':'纯身份训练','B1_dictionary_ACC':'词典音素＋ACC能量局部身份','B2_Audio_phoneme':'词典音素＋Audio局部身份','B3_shuffled_phoneme':'Audio位置＋音素标签打乱'}
    lines=['# 音素条件下的跨词身份学习：同预算神经网络对照','','## 本轮问题','','Audio辅助音素区域的身份监督，能否让ACC-only查询在未参与训练的词上更好认证？同时区分词典类别、ACC局部能量和Audio区域的贡献。','','## 固定协议','','10用户闭集：全部身份参与表示训练，因此不是未见用户开放集。使用历史9个英语词；训练Back、Cancel、Exit、five、Help、Menu，留出Grab、Next、nine查询。所有词都曾在历史诊断中查看，属于探索复核，不是全新盲测。','','每词前4遍训练表示；训练6词的第5/6遍注册（每人12条），第7/8遍校准（每人12条）。留出3词全部8遍查询（每人24条）。四个角色索引不交叉，训练/注册/校准也没有共享实际ACC采样：输入只使用中心920ms，最小相邻重复中心间隔约981ms，未使用重叠3秒缓存的外围。','','每组5057参数的同一网络：24维ACC局部统计序列→共享32维MLP→可学习注意力→均值/标准差＋原始统计跳连→32维L2身份嵌入。临时10分类头只训练。四组均CE＋0.1跨词SupCon；后三组再加0.1局部身份SupCon。局部正对为同人同音素跨词，负对为异人同音素；单词相同的重复不当跨词正对。','','B1使用词典类别与ACC能量确定局部软区域；B2使用同样词典类别与训练Audio自动音素区域；B3保留B2区域，但分别在每个用户的训练词间打乱音素类别，保持类别数量，排除仅有额外损失的解释。音素类别来自词典，不能直接当作Audio贡献。训练只有five一种AY，因此AY没有同音素跨词局部正对，nine仅作为迁移控制。','','Audio只在训练角色提供区域软权重；注册、校准和查询的Audio教师数组全部缺失，推理只输入ACC，由网络自己的注意力提取身份。3秒缓存本身依赖历史分段，因此仍不是完全不依赖Audio的连续端到端系统。','','三个种子42/43/44，每模型400步、每批60条、完全同初始化与采样，12模型无早停或超参数搜索。身份梯度直接回传共享编码器与投影层。全局重命名音素标签不会改变SupCon，因此负控制采用逐用户/逐词的非平凡打乱，不使用无效的统一类别改名。','','注册原型余弦认证，全部120个用户阈值由校准集锁定后才计算查询分数。EER由评估分数计算；HTER/FAR/FRR使用锁定阈值。未改变冻结创新点一、既有模型与历史结果。','','## 三种子、十用户宏平均','','|方法|EER%↓|HTER%↓|FAR%↓|FRR%↓|AUC↑|','|---|---:|---:|---:|---:|---:|']
    for v in summary:lines.append('|'+names[v['arm']]+'|'+ '|'.join(f"{100*v[k]:.3f}" for k in ['eer','hter','far','frr'])+f"|{v['auc']:.4f}|")
    lines+=['','## 每个种子的宏平均','','|种子|方法|EER%|HTER%|AUC|','|---|---|---:|---:|---:|']
    for seed in [42,43,44]:
      for arm in ARMS:
        rr=[v for v in rows if v['seed']==seed and v['arm']==arm];lines.append(f"|{seed}|{names[arm]}|{100*np.mean([v['eer'] for v in rr]):.3f}|{100*np.mean([v['hter'] for v in rr]):.3f}|{np.mean([v['auc'] for v in rr]):.4f}|")
    lines+=['','## Audio组相对对照','','EER差为Audio减对照，负值较好；每人先平均3种子，按10用户作精确符号置换。两个主要比较Holm校正，打乱控制为附加检验。共享模型和历史数据使这些检验仅供探索性判断。','','|对照|EER差/百分点|改善用户|单侧p|主要校正p|','|---|---:|---:|---:|---:|']
    for v in comparisons:lines.append(f"|{names[v['control']]}|{100*v['eer_delta']:+.3f}|{v['improved_users']}/10|{v['p']:.4f}|{v.get('holm_primary_p','附加控制')}|")
    lines+=['','## 逐用户结果（三种子平均）','','|用户|方法|EER%|HTER%|FAR%|FRR%|AUC|','|---|---|---:|---:|---:|---:|---:|']
    for us in range(1,11):
      for arm in ARMS:
        v=next(v for v in users if v['user']==us and v['arm']==arm);lines.append(f"|U{us:02d}|{names[arm]}|"+'|'.join(f"{100*v[k]:.3f}" for k in ['eer','hter','far','frr'])+f"|{v['auc']:.4f}|")
    lines+=['','## 留出词结果','','|词|方法|EER%|HTER%|AUC|','|---|---|---:|---:|---:|']
    for wo,name in [(10,'Grab'),(14,'Next'),(15,'nine')]:
      for arm in ARMS:
        rr=[v for v in wordrows if v['word']==wo and v['arm']==arm];lines.append(f"|{name}|{names[arm]}|{100*np.mean([v['eer'] for v in rr]):.3f}|{100*np.mean([v['hter'] for v in rr]):.3f}|{np.mean([v['auc'] for v in rr]):.4f}|")
    lines+=['','## 判据','','只有Audio组稳定超过纯身份、词典＋ACC控制和标签打乱，才能把收益归因于训练Audio提供的局部身份监督。若仅优于纯身份而不优于词典＋ACC，证据支持音素结构训练，尚不支持Audio独立贡献。没有根据这些留出词再选权重或用户。','','## 核验','','重放12个模型、120个校准阈值及360个逐词用户结果，检查角色、批次、平衡标签打乱与查询缺失Audio标签。合成测试验证局部身份损失回传编码器、推理不依赖教师权重以及全局类别改名不是有效负控制。公开代码和匿名汇总，不含权重、边界、身份映射或逐样本分数。']
    if (OUT/'additional_EH_summary_public.json').exists():
      extra=read('additional_EH_summary_public.json');plan=read('additional_EH_protocol_public.json');eu=read('additional_EH_per_user_public.json');er=read('additional_EH_per_seed_user_public.json');ew=read('additional_EH_per_word_user_public.json')
      assert plan['words']==[23,24,25] and not set(plan['words'])&set(TRAIN+QUERY)
      assert len(er)==120 and len(ew)==360 and len(eu)==40
      assert plan['threshold_sha256']==hashlib.sha256((PRIVATE/'thresholds_private.json').read_bytes()).hexdigest()
      for seed in [42,43,44]:
        for arm in ARMS:assert plan['model_sha256'][str(seed)+'_'+arm]==hashlib.sha256((PRIVATE/f'{seed}_{arm}_private.pt').read_bytes()).hexdigest()
      for v in extra:
        rr=[a for a in er if a['arm']==v['arm']]
        for key in ['eer','hter','far','frr','auc']:assert np.isclose(v[key],np.mean([a[key] for a in rr]),atol=1e-12)
      result.update(additional_EH_query_words=[23,24,25],additional_models_and_thresholds_unchanged=True,additional_query_Audio_unused=True)
      (OUT/'verification_public.json').write_text(json.dumps(result,indent=2))
      lines+=['','## 后续冻结验证：Reset、Select、seven','','初始Next结果引出事后假设：较多同音素训练词是否更有利？不重训、不改阈值，按词典规则选出所有尚未用于本轮、英语单词且首条词典发音含EH的提示，得到Reset、Select、seven。在读取这些ACC查询结果前保存规则、模型与阈值哈希。仅计算ACC输入，未读取这三词Audio。三词在历史其他实验中已查看，仍不是全新盲测。','','|方法|EER%↓|HTER%↓|FAR%↓|FRR%↓|AUC↑|','|---|---:|---:|---:|---:|---:|']
      for v in extra:lines.append('|'+names[v['arm']]+'|'+ '|'.join(f"{100*v[k]:.3f}" for k in ['eer','hter','far','frr'])+f"|{v['auc']:.4f}|")
      lines+=['','### 追加查询逐用户结果','','|用户|方法|EER%|HTER%|FAR%|FRR%|AUC|','|---|---|---:|---:|---:|---:|---:|']
      for us in range(1,11):
        for arm in ARMS:
          v=next(v for v in eu if v['user']==us and v['arm']==arm);lines.append(f"|U{us:02d}|{names[arm]}|"+'|'.join(f"{100*v[k]:.3f}" for k in ['eer','hter','far','frr'])+f"|{v['auc']:.4f}|")
      lines+=['','### 追加查询逐词结果','','|词|方法|EER%|HTER%|AUC|','|---|---|---:|---:|---:|']
      for wo,name in [(23,'Reset'),(24,'Select'),(25,'seven')]:
        for arm in ARMS:
          rr=[v for v in ew if v['word']==wo and v['arm']==arm];lines.append(f"|{name}|{names[arm]}|{100*np.mean([v['eer'] for v in rr]):.3f}|{100*np.mean([v['hter'] for v in rr]):.3f}|{np.mean([v['auc'] for v in rr]):.4f}|")
      lines+=['','## 本轮结论','','局部身份约束降低了校准阈值下的错误，尤其误拒率，但正确音素Audio没有形成稳定独立优势：初始三词没有超过词典＋ACC的身份排序；冻结追加三词中，标签打乱复现了大部分HTER改善，且EER/AUC不劣于正确Audio。因此Next的局部正向结果不能推广为已证明的音素身份迁移。当前更像局部视图身份正则与分数分布变化，这一解释仍需专门机制对照，不能直接当作确定生理结论。']
    lines+=['','## 模型解释范围','','本实现的局部身份损失直接更新共享编码器和投影层，但不直接监督查询注意力；注意力由全局身份损失训练。因此本轮检验的是共享参数带来的局部身份知识迁移，不等同于已经完成能自主识别各音素并聚合的模型。这个限制不能用来宣布成功，也说明本轮未覆盖所有音素级训练架构。']
    (OUT/'report.md').write_text('\n'.join(lines),encoding='utf-8');print(json.dumps(dict(verification=result,comparisons=comparisons)))
if __name__=='__main__':
    with threadpool_limits(limits=2):main()
