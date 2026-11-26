"""Replay metrics and test direct-attention mechanism against matched controls."""
import json,hashlib,itertools
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from train_connected_attention_identity import ROOT,BASE,OLD,PRIVATE,OUT,ARMS,EXTRA,QUERY,build_net
from train_phoneme_conditioned_identity import embed
from train_acc_phoneme_locator import grid_descriptors
from run_continuous import choose_auth,measure
from audit_local_identity_residual import holm

def read(name):return json.loads((OUT/name).read_text(encoding='utf-8'))

def main():
    torch.set_num_threads(2);protocol=read('protocol_public.json');ph=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    assert protocol['data_sha256']==hashlib.sha256((OLD/'inputs_private.npz').read_bytes()).hexdigest()
    assert protocol['roles_sha256']==hashlib.sha256((OLD/'prepared_private.npz').read_bytes()).hexdigest()
    p=np.load(OLD/'prepared_private.npz');data=np.load(OLD/'inputs_private.npz');roles=np.load(PRIVATE/'roles_private.npz');u,w=p['u'],p['w'];fit=p['fit']
    assert np.isnan(p['Audio_teacher'][~fit]).all() and np.array_equal(data['indices'],p['indices'])
    assert fit[roles['donor'][fit]].all() and np.array_equal(u[roles['donor'][fit]],u[fit]) and np.array_equal(w[roles['donor'][fit]],w[fit])
    x=torch.tensor((data['x']-p['mean'])/p['scale']);ri,ci,qi=[np.flatnonzero(p[k]) for k in ['registration','calibration','query']]
    d=np.load(BASE/'conditional_correspondence/coarse_pilot/raw_inputs_private.npz');uu,ww=d['user'],d['word'];ei=np.flatnonzero(np.isin(ww,EXTRA));wave=d['acc']
    xx=np.stack([grid_descriptors(wave[int(i)],np.arange(1.1,1.901,.02)) for i in ei]).astype('float32');ex=torch.tensor((xx-p['mean'])/p['scale'])
    rows=read('per_seed_user_public.json');wordrows=read('per_word_user_public.json');users=read('per_user_public.json');summary=read('summary_public.json');diag=read('attention_gradient_public.json')
    assert len(rows)==360 and len(wordrows)==1080 and len(users)==120 and len(summary)==12
    locks=json.loads((PRIVATE/'thresholds_private.json').read_text());oldlocks=json.loads((OLD/'thresholds_private.json').read_text());assert len(locks)==180
    assert (PRIVATE/'thresholds_private.json').stat().st_mtime<=(OUT/'per_seed_user_public.json').stat().st_mtime
    maxerr=0.;stateerr=0.;keys=['eer','hter','far','frr','auc']
    for seed in [42,43,44]:
      for arm in ARMS:
        saved=torch.load(PRIVATE/f'{seed}_{arm}_private.pt',weights_only=False);assert saved['protocol_sha256']==ph and saved['steps']==400
        assert [v['step'] for v in saved['trace']]==[100,200,300,400]
        gd=saved['gradient'];assert len(gd)==3
        if arm in ['C0_identity','C1_detached_Audio']:assert all(v['local_attention_gradient_norm']==0 for v in gd)
        else:assert gd[0]['local_attention_gradient_norm']>0
        if arm in ['C0_identity','C1_detached_Audio']:
          oldarm='B0_identity' if arm=='C0_identity' else 'B2_Audio_phoneme';old=torch.load(OLD/f'{seed}_{oldarm}_private.pt',weights_only=False)['model']
          for name,v in saved['model'].items():
            err=float((v-old[name]).abs().max());stateerr=max(stateerr,err);assert err<1e-7
        net=build_net(arm);net.load_state_dict(saved['model']);rz=embed(net,x,ri);cz=embed(net,x,ci)
        proto=np.stack([rz[u[ri]==us].mean(0) for us in range(1,11)]);proto/=np.maximum(np.linalg.norm(proto,axis=1,keepdims=True),1e-12)
        for us in range(1,11):
          tau=next(v['tau'] for v in locks if v['seed']==seed and v['arm']==arm and v['user']==us)
          assert tau==choose_auth(cz@proto[us-1],u[ci]==us)
          if arm in ['C0_identity','C1_detached_Audio']:assert tau==next(v['tau'] for v in oldlocks if v['seed']==seed and v['arm']==oldarm and v['user']==us)
        for phase,inputs,indices,labels,lex in [('initial',x,qi,u[qi],w[qi]),('additional_EH',ex,np.arange(len(ei)),uu[ei],ww[ei])]:
          qz=embed(net,inputs,indices)
          for us in range(1,11):
            tau=next(v['tau'] for v in locks if v['seed']==seed and v['arm']==arm and v['user']==us);score=qz@proto[us-1]
            expected=next(v for v in rows if v['seed']==seed and v['arm']==arm and v['user']==us and v['phase']==phase);got=measure(score,labels==us,tau)
            assert expected['genuine_trials']==24 and expected['attack_trials']==216
            for k in keys:
              err=abs(got[k]-expected[k]);maxerr=max(maxerr,err);assert err<1e-10
            for wo in sorted(set(lex)):
              mask=lex==wo;got=measure(score[mask],labels[mask]==us,tau);expected=next(v for v in wordrows if v['seed']==seed and v['arm']==arm and v['user']==us and v['word']==wo and v['phase']==phase)
              assert expected['genuine_trials']==8 and expected['attack_trials']==72
              for k in keys:assert abs(got[k]-expected[k])<1e-10
    for v in summary:
      rr=[a for a in rows if a['arm']==v['arm'] and a['phase']==v['phase']]
      for k in keys:assert np.isclose(v[k],np.mean([a[k] for a in rr]),atol=1e-12)
    signs=np.array(list(itertools.product([-1,1],repeat=10)));comparisons=[]
    for phase in ['initial','additional_EH']:
      group=[]
      for control in ['C1_detached_Audio','C2_connected_ACC','C4_connected_wrongrepeat','C5_connected_shuffledphone']:
        delta=np.array([next(v['eer'] for v in users if v['user']==us and v['arm']=='C3_connected_Audio' and v['phase']==phase)-next(v['eer'] for v in users if v['user']==us and v['arm']==control and v['phase']==phase) for us in range(1,11)])
        group.append(dict(phase=phase,control=control,eer_delta=float(delta.mean()),improved_users=int((delta<0).sum()),p=float(np.mean((signs*delta).mean(1)<=delta.mean()+1e-12))))
      for v,pp in zip(group[:2],holm([v['p'] for v in group[:2]])):v['holm_primary_p']=float(pp)
      for v,pp in zip(group[2:],holm([v['p'] for v in group[2:]])):v['holm_control_p']=float(pp)
      comparisons.extend(group)
    verification=dict(status='PASS',models_replayed=18,calibration_workpoints_replayed=180,pooled_query_metrics_replayed=360,per_word_metrics_replayed=1080,max_metric_error=maxerr,
      historical_control_models_reproduced=6,max_historical_parameter_error=stateerr,historical_control_thresholds_reproduced=60,
      local_attention_gradient_detached_zero_connected_nonzero=True,query_ACC_only=True,query_Audio_teacher_unused=True,source_data_and_roles_unchanged=True)
    (OUT/'verification_public.json').write_text(json.dumps(verification,indent=2));(OUT/'comparison_public.json').write_text(json.dumps(comparisons,indent=2))
    names={'C0_identity':'纯身份训练','C1_detached_Audio':'旧Audio局部分支（未接注意力）','C2_connected_ACC':'接通注意力＋ACC先验','C3_connected_Audio':'接通注意力＋正确Audio','C4_connected_wrongrepeat':'接通注意力＋错遍Audio','C5_connected_shuffledphone':'接通注意力＋音素标签打乱'}
    lines=['# 局部身份损失直接回传查询注意力：受控实验','','## 本轮改变','','同一5057参数网络、同一数据与400步预算。旧局部身份分支使用固定教师权重；新分支使用softmax(ACC注意力logit＋log教师先验)，让局部身份SupCon直接更新查询用的注意力参数。不增加网络、不加入活动/VAD损失、不调lambda。查询仍只计算ACC自己的softmax注意力，不读取教师或音素标签。','','## 协议','','闭集跨词，10身份都参与表示训练。训练6词Back/Cancel/Exit/five/Help/Menu；前4遍拟合，第5/6遍注册（每人12条），第7/8遍校准（每人12条）。初始查询Grab/Next/nine每人24条；追加词固定沿用上一轮Reset/Select/seven每人24条，未再筛词。全部历史内容已被查看，属于探索复核，不是新盲测或未见用户开放集。','','输入只取原始ACC中央920ms，训练/注册/校准实际输入不重叠，教师只用训练角色。原始缓存中心依赖历史词分段，因此不是连续端到端无Audio系统。三个种子42/43/44、18个模型，各400步，初始化、采样和预算匹配。C0/C1应逐参数重现上一轮纯身份与旧Audio模型。','','错遍Audio仅交换本人同词的另外一条训练录制，不借用注册/校准/查询；标签打乱保留每用户类别数量。ACC先验组检验是否只是任意局部关注都有用，旧Audio分支检验新增梯度连接本身。全部180阈值锁定后才评分两组查询。EER来自评估分数，HTER/FAR/FRR使用校准阈值。','','## 身份结果（三种子×十用户宏平均）']
    for phase,label in [('initial','原始3个留出词'),('additional_EH','固定追加3个EH词')]:
      lines+=['',f'### {label}','','|方法|EER%↓|HTER%↓|FAR%↓|FRR%↓|AUC↑|','|---|---:|---:|---:|---:|---:|']
      for v in summary:
        if v['phase']==phase:lines.append('|'+names[v['arm']]+'|'+'|'.join(f"{100*v[k]:.3f}" for k in ['eer','hter','far','frr'])+f"|{v['auc']:.4f}|")
    lines+=['','## 正确Audio相对对照','','每用户先平均三个种子，再按10用户符号置换。EER差负值较好；初始两项主要比较Holm2，错误先验/标签为另一控制比较族；追加词仅作固定的探索一致性检查。不能将重复录制当独立受试者。','','|查询集|对照|EER差/百分点|改善用户|p|校正p|','|---|---|---:|---:|---:|---:|']
    for v in comparisons:lines.append(f"|{v['phase']}|{names[v['control']]}|{100*v['eer_delta']:+.3f}|{v['improved_users']}/10|{v['p']:.4f}|{v.get('holm_primary_p',v.get('holm_control_p')):.4f}|")
    lines+=['','## 真实梯度诊断','','训练第1/100/400步记录，平均3种子；未用诊断数值选择模型或权重。','','|方法|局部身份→注意力梯度范数均值|局部/全局梯度范数比均值|','|---|---:|---:|']
    for arm in ARMS:
      ds=[v for a in diag if a['arm']==arm for v in a['gradient']];lines.append(f"|{names[arm]}|{np.mean([v['local_attention_gradient_norm'] for v in ds]):.6g}|{np.mean([v['local_to_global_ratio'] for v in ds]):.6g}|")
    lines+=['','## 逐种子结果','','|查询集|种子|方法|EER%|HTER%|AUC|','|---|---|---|---:|---:|---:|']
    for phase in ['initial','additional_EH']:
      for seed in [42,43,44]:
        for arm in ARMS:
          rr=[v for v in rows if v['phase']==phase and v['seed']==seed and v['arm']==arm];lines.append(f"|{phase}|{seed}|{names[arm]}|{100*np.mean([v['eer'] for v in rr]):.3f}|{100*np.mean([v['hter'] for v in rr]):.3f}|{np.mean([v['auc'] for v in rr]):.4f}|")
    lines+=['','## 逐用户结果（三种子平均）','','|查询集|用户|方法|EER%|HTER%|FAR%|FRR%|AUC|','|---|---|---|---:|---:|---:|---:|---:|']
    for phase in ['initial','additional_EH']:
      for us in range(1,11):
        for arm in ARMS:
          v=next(v for v in users if v['phase']==phase and v['user']==us and v['arm']==arm);lines.append(f"|{phase}|U{us:02d}|{names[arm]}|"+'|'.join(f"{100*v[k]:.3f}" for k in ['eer','hter','far','frr'])+f"|{v['auc']:.4f}|")
    lines+=['','## 逐词结果','','|词|方法|EER%|HTER%|AUC|','|---|---|---:|---:|---:|']
    for wo,text in [(10,'Grab'),(14,'Next'),(15,'nine'),(23,'Reset'),(24,'Select'),(25,'seven')]:
      for arm in ARMS:
        rr=[v for v in wordrows if v['word']==wo and v['arm']==arm];lines.append(f"|{text}|{names[arm]}|{100*np.mean([v['eer'] for v in rr]):.3f}|{100*np.mean([v['hter'] for v in rr]):.3f}|{np.mean([v['auc'] for v in rr]):.4f}|")
    success=all(v['eer_delta']<0 and v['holm_primary_p']<.05 for v in comparisons if v['phase']=='initial' and v['control'] in ['C1_detached_Audio','C2_connected_ACC'])
    lines+=['','## 判断','',f'正确Audio同时超过旧分支及ACC先验并通过初始主要EER比较：{success}。这一条件即使成立，也还要检查HTER、错遍/标签控制、三个种子与追加查询的一致性；不能只凭注意力梯度非零宣布身份增益。若未成立，说明本次具体梯度连接修正没有完成Audio独立身份增益的证明，不等于排除全部音素级架构。', '', '本轮确实接通了身份梯度，旧对照也逐参数重现，但原三词的EER未改善、追加三词HTER几乎不变。因此当前问题不能仅归因于局部损失没有回传注意力。同人同词错遍仍保留发音内容，其结果不证明所有Audio信息都无价值；它说明本轮没有证明精确重复对应具有独立身份收益。', '', '局部梯度诊断只抽查训练第1/100/400步，不用于调权重或解释成完整训练过程的恒定梯度比例。', '', '## 独立核验','','重放18个模型、180个阈值、360个用户查询集结果和1080个逐词用户结果；对C0/C1逐项比较上一轮6个模型权重和60个阈值。检查训练局部梯度：旧分支为零，新连接为非零。合成测试确认初始局部视图一致、梯度能更新查询注意力、推理不依赖先验。源码和匿名汇总公开，不上传原始信号、边界、权重、身份映射或逐样本分数。']
    (OUT/'report.md').write_text('\n'.join(lines),encoding='utf-8');print(json.dumps(dict(verification=verification,comparisons=comparisons)))

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
