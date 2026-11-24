import argparse,json
from collections import Counter
from pathlib import Path
import numpy as np

def main(root):
    base=root/'continuous_acc_activity';out=base/'joint_identity';read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    s=read(out/'summary_public.json');selection=read(out/'selection_public.json');q=read(out/'activity_quality_public.json');logs=read(out/'training_public.json');protocol=read(out/'protocol_public.json')
    rng=np.random.default_rng(20261125);boot=rng.integers(0,10,(10000,10));comparisons=[]
    for control in ['Identity_only','Joint_wrongwindow']:
      for metric in ['eer','hter','auc']:
        delta=np.array([next(v[metric] for v in s['per_user'] if v['user']==us and v['arm']=='Joint_correct')-next(v[metric] for v in s['per_user'] if v['user']==us and v['arm']==control) for us in range(1,11)])
        ci=np.quantile(delta[boot].mean(1),[.025,.975]);comparisons.append(dict(control=control,metric=metric,delta=float(delta.mean()),target_cluster_ci95=ci.tolist(),improved_users=int((delta>0 if metric=='auc' else delta<0).sum())))
    def supported(control):
      e=next(v for v in comparisons if v['control']==control and v['metric']=='eer');a=next(v for v in comparisons if v['control']==control and v['metric']=='auc')
      return e['target_cluster_ci95'][1]<0 and a['target_cluster_ci95'][0]>0
    hter=next(v['hter'] for v in s['macro'] if v['arm']=='Joint_correct');utility=all(hter<=v['hter'] for v in s['macro'])
    passed=all(supported(v) for v in ['Identity_only','Joint_wrongwindow']) and utility
    if passed:decision='联合正确 Audio 活动监督同时稳定超过纯身份训练与错误窗口监督，且平均 HTER 不退化，可作为新录制盲测确认的候选。'
    elif supported('Joint_wrongwindow'):decision='正确监督相对错误窗口存在稳定身份区分线索，但尚未同时超过纯身份训练并满足完整工作点标准，不能宣布整体认证增益成立。'
    else:decision='本轮未取得稳定独立 Audio 身份增益：正确联合监督未同时在 EER/AUC 上超过纯身份训练和错误窗口控制。保留部分指标与逐用户差异，不使用测试词继续搜索权重。'
    arms=['Identity_only','Joint_correct','Joint_wrongwindow'];counts=dict(Counter(str(v['shared_lambda']) for v in selection));activity={arm:float(np.mean([v['activity_proxy_auc'] for v in q if v['arm']==arm])) for arm in arms}
    diag=[]
    for arm in arms[1:]:
      traces=[]
      for v in selection:
        log=next(r for r in logs if (r['seed'],r['user'],r['arm'],r['lambda_value'])==(v['seed'],v['user'],arm,v['shared_lambda']))
        traces.extend(r for r in log['trace'] if 'body_gradient_cosine' in r)
      diag.append(dict(arm=arm,body_gradient_cosine_mean=float(np.mean([v['body_gradient_cosine'] for v in traces])),aux_to_identity_gradient_ratio_mean=float(np.mean([v['aux_to_identity_gradient_ratio'] for v in traces])),observations=len(traces)))
    lines=['# 联合身份损失＋Audio 活动辅助监督：连续 ACC 实验','',
      '## 问题与执行结果','',decision,'',
      '本轮把身份损失与活动监督同时回传到共享 ACC 编码器及时间权重，不再先生成硬活动掩码，再固定提取身份统计。没有 Audio/ACC embedding 全局对齐，没有生成器，没有改动既有创新点一。',
      '', '## 一、固定数据与预算','',
      '沿用 2179 个固定连续 IMU 窗：首个有效 IMU 时间＋0.5 秒起，每 3 秒一窗，互不重叠；训练/开发/测试词 24/8/8，窗口数 1314/432/433。测试八词的 Audio 与 Audio 时钟没有打开，标签全部缺失。不存在 Audio-VAD 居中定位。',
      '', '**同一八个测试词已在上一轮看过，本轮是探索性复核，不是新的盲测确认。** 所有模型结构、候选权重和开发选择规则在本轮训练前固定，不用这些测试词选模型。',
      '', '三个随机种子 42/43/44；每个外层目标及下一位循环开发身份均不进入网络训练，其余八人训练。每个模型固定 400 步、相同 AdamW/初始化/批次。每批 8 人×2 个共同训练词×每词2窗＝32；同人跨词为正，异人同词也是负，同人同词不当负样本。',
      '', '正确和错误监督各有 λ=0.03/0.1 两候选，每候选均 400 步；共享 λ 按二者内部开发身份跨词 AUC 的均值选择，随后看平均 EER、再优先较小 λ。这样不单独给正确组挑更有利的权重。纯身份组 λ=0，没有可搜索的辅助权重。实际训练 150 个候选配置，最终选择 90 个；两个联合组搜索预算相同，所有最终模型有效训练预算均为 400 步。',
      '', '## 二、可学习身份结构','',
      f"三组完全相同的 {protocol['parameters']} 参数网络：15 通道局部 ACC 输入→16通道时序 CNN 两层→可学习时间打分→softmax 加权均值/标准差。完整窗口 16 维原始统计作为跳连，与 32 维 CNN 池化结果一起经过可训练投影得到 32 维 L2 身份向量。原始跳连不是最终固定身份特征，身份投影和 CNN 共同训练。",
      '', '时间头既用于身份池化，也接收活动 BCE，因此身份梯度可以改变关注哪些时间片。没有活动阈值、没有硬切片、没有按 Audio 删除查询。跳连保留整窗信息，防止完全丢失已有身份差异。只有 IMU 无效帧被排除；当前缓存有效帧完整。',
      '', '身份损失：归一化临时八分类头 CE（尺度16）＋0.1×跨词 SupCon（温度0.1）。分类头仅训练，不作为最终用户分类器。认证阶段使用注册原型和余弦相似度。',
      '', '总损失：L_ID＋λ×L_Audio_activity。活动目标是上一轮的 Audio 能量代理；错误控制在同人同词录制内使用另一非重叠窗口的标签，保留身份与词条件，破坏时间对应，属于异窗/错遍代理。',
      '', '初始时间头为零，所有有效帧均匀参与；Audio 首先更新时间头，此后经时间头传到共享编码器。合成梯度测试核验了身份和活动损失对共享编码器、时间权重的影响，没有 detach 或冻结生成器后仅训练分类头。',
      '', '## 三、开发选择与认证隔离','',
      '共享 λ 仅由内部开发身份决定：该身份在训练词的12条窗口注册，开发八词查询，与八位基础身份的开发查询区分。当前最终目标不进入该选择。身份模型检查点固定第400步，无测试/开发早停挑 epoch。',
      '', '三组使用同样12条注册窗；认证阈值沿用上轮开发八词最小 HTER、平衡 FAR/FRR、再较高阈值规则。全部90个个人阈值锁定后才生成测试身份向量和分数。所有433个测试窗均计入；没有活动概率拒绝或有利用户筛选。',
      '', '这是目标身份不参与网络训练的跨词验证，但背景仍是既有十人，个人阈值使用目标开发词；不是严格未见攻击者的开放集。',
      '', '## 四、独立测试词身份结果','','十用户×三种子宏平均，EER/HTER/FAR/FRR/Top1 为百分比。','','|方法|EER|HTER|FAR|FRR|AUC|十人Top1|','|---|---:|---:|---:|---:|---:|---:|']
    for arm in arms:
      v=next(v for v in s['macro'] if v['arm']==arm);lines.append(f"|{arm}|{100*v['eer']:.3f}|{100*v['hter']:.3f}|{100*v['far']:.3f}|{100*v['frr']:.3f}|{v['auc']:.5f}|{100*v['top1_10way']:.2f}|")
    previous=read(base/'results/summary_public.json')['macro'];reference=next(v for v in previous if v['method']=='ACC_direct')
    lines+=['',f"背景参照：上一轮同窗口的手工 ACC 统计＋余弦 EER={100*reference['eer']:.2f}%、HTER={100*reference['hter']:.2f}%、AUC={reference['auc']:.4f}。它的结构与分段规则不同，不用于单独归因 Audio；真正的因果对照是本轮完全同结构的三组。",'',
      '## 五、目标簇配对差异','','正确联合监督−对照；EER/HTER 负差为好，AUC 正差为好。每目标先平均三种子，再按10用户重采样，不将窗口或帧当独立人。共享背景、既有测试曝光使区间仅作探索性描述。','','|对照|指标|差/百分点|95%区间/百分点|改善用户|','|---|---|---:|---|---:|']
    for v in comparisons:
      lo,hi=v['target_cluster_ci95'];lines.append(f"|{v['control']}|{v['metric']}|{100*v['delta']:+.3f}|[{100*lo:+.3f}, {100*hi:+.3f}]|{v['improved_users']}/10|")
    lines+=['','## 六、逐用户测试结果（三种子平均）','','|用户|方法|EER%|HTER%|FAR%|FRR%|AUC|','|---|---|---:|---:|---:|---:|---:|']
    for us in range(1,11):
      for arm in arms:
        v=next(v for v in s['per_user'] if v['user']==us and v['arm']==arm);lines.append(f"|U{us:02d}|{arm}|{100*v['eer']:.3f}|{100*v['hter']:.3f}|{100*v['far']:.3f}|{100*v['frr']:.3f}|{v['auc']:.4f}|")
    lines+=['','## 七、活动一致性与真实梯度','',
      '这里只在留出目标的开发词上评价时间打分与 Audio 活动代理的一致性；测试 Audio 不可用。纯身份时间头没有活动训练目标，不把其数值当独立 VAD 性能基准。',
      '', '|方法|时间打分的活动代理 AUC|','|---|---:|']
    for arm in arms:lines.append(f'|{arm}|{activity[arm]:.4f}|')
    lines+=['','|联合组|共享编码器梯度余弦均值|辅助/身份梯度范数比均值|观察点|','|---|---:|---:|---:|']
    for v in diag:lines.append(f"|{v['arm']}|{v['body_gradient_cosine_mean']:.4f}|{v['aux_to_identity_gradient_ratio_mean']:.4f}|{v['observations']}|")
    lines+=['',f'30折共享权重总体分布：`{json.dumps(counts)}`。损失权重不是梯度比例，没有根据测试结果再做梯度门控或加权。',
      '', '## 判断与限制','',decision,
      '', '必须同时区别：联合身份训练相对旧统计流程的变化，是否来自身份监督/结构；正确 Audio 相对本轮纯身份网络的变化，是否才是 Audio 独立贡献。正确只超过错误监督仍不足以证明比没有 Audio 更好。活动预测变好也不替代 EER/AUC 的证据。',
      '', '固定三秒窗可能包含静音或多次同词发声；本轮全部计入，包含检测与身份判断的保守任务。没有人工 VAD 真值或物理同步标记，缺少跨天/重戴/换设备数据，因此不解释为唯一生理传递机制。',
      '', '## 核验与公开范围','',
      '同一私有数据文件哈希与上轮一致，测试 Audio 标签全缺失。核验150个候选、12000个身份批次、30次共享λ选择及90个身份工作点；重放多个早期/后期模型的 ACC-only 测试分数。公开代码、匿名逐用户/种子汇总，不上传原始信号、身份向量、权重、抽样索引、个人阈值或身份映射。']
    (out/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    result=dict(decision=decision,full_support=passed,correct_vs_wrong_supported=supported('Joint_wrongwindow'),correct_vs_identity_supported=supported('Identity_only'),comparisons=comparisons,activity_proxy_auc=activity,selected_lambda_counts=counts,gradient_diagnostics=diag)
    (out/'decision_public.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(decision=decision,macro=s['macro'],comparisons=comparisons,activity=activity),ensure_ascii=False))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
