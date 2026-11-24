import argparse,json
from collections import Counter
from pathlib import Path
import numpy as np

def main(root):
    base=root/'continuous_acc_activity';out=base/'results';read=lambda path:json.loads(path.read_text(encoding='utf-8'))
    s=read(out/'summary_public.json');q=read(out/'mask_quality_public.json');selection=read(out/'mask_thresholds_private.json');split=read(base/'word_split_public.json');prep=read(base/'preparation_audit_public.json');protocol=read(out/'protocol_public.json')
    records=read(base/'recordings_private.json');vocabulary={r['word']:r['prompt'] for r in records};c=np.load(base/'continuous_private.npz');w=c['word']
    names=['ACC_direct','ACC_self_trained','Audio_correct','Audio_wrongwindow','Fixed_time_prior'];rng=np.random.default_rng(20261124);boot=rng.integers(0,10,(10000,10))
    comparisons=[]
    for control in ['ACC_self_trained','Audio_wrongwindow','Fixed_time_prior','ACC_direct']:
      for metric in ['eer','hter','auc']:
        delta=np.array([next(v[metric] for v in s['per_user'] if v['user']==us and v['method']=='Audio_correct')-next(v[metric] for v in s['per_user'] if v['user']==us and v['method']==control) for us in range(1,11)])
        ci=np.quantile(delta[boot].mean(1),[.025,.975]);comparisons.append(dict(control=control,metric=metric,delta=float(delta.mean()),target_cluster_ci95=ci.tolist(),improved_users=int((delta>0 if metric=='auc' else delta<0).sum())))
    def core(control):
        e=next(v for v in comparisons if v['control']==control and v['metric']=='eer');a=next(v for v in comparisons if v['control']==control and v['metric']=='auc')
        return bool(e['target_cluster_ci95'][1]<0 and a['target_cluster_ci95'][0]>0)
    utility=all(next(v['hter'] for v in s['macro'] if v['method']=='Audio_correct')<=next(v['hter'] for v in s['macro'] if v['method']==control) for control in names if control not in ['Audio_correct','ACC_direct'])
    quality=[];quality_comparisons=[]
    for name in names[1:]:
      rr=[v for v in q if v['method']==name];quality.append(dict(method=name,**{k:float(np.mean([v[k] for v in rr])) for k in ['f1','balanced_accuracy','proxy_auc']}))
    for control in ['Audio_wrongwindow','Fixed_time_prior','ACC_self_trained']:
      delta=np.array([np.mean([v['proxy_auc'] for v in q if v['user']==us and v['method']=='Audio_correct'])-np.mean([v['proxy_auc'] for v in q if v['user']==us and v['method']==control]) for us in range(1,11)])
      ci=np.quantile(delta[boot].mean(1),[.025,.975]);quality_comparisons.append(dict(control=control,proxy_auc_delta=float(delta.mean()),target_cluster_ci95=ci.tolist()))
    activity=all(v['target_cluster_ci95'][0]>0 for v in quality_comparisons if v['control'] in ['Audio_wrongwindow','Fixed_time_prior'])
    passed=all(core(v) for v in ['ACC_self_trained','Audio_wrongwindow','Fixed_time_prior']) and utility
    if passed:decision='正确 Audio 监督在去除 Audio 定位后，同时超过同预算纯 ACC、异窗监督和时间先验，形成值得独立录制确认的认证候选证据。'
    elif activity:decision='已获得超出固定位置的 Audio 活动代理预测信息；尚未通过稳定额外身份增益标准。活动可预测性与跨词身份区分收益需要分开，本轮不扩大网络或在测试词调权重。'
    else:decision='本轮没有同时建立活动代理迁移与额外身份收益的完整证据；不从 HTER 单项或个别用户结果判断成功。'
    counts={name:dict(Counter(str(v['threshold']) for v in selection if v['method']==name)) for name in names[1:]}
    rows_by_role={role:int(np.isin(w,split[role]).sum()) for role in ['train','development','test']}
    lines=['# 去除 Audio 定位后的连续 ACC 活动监督与跨词身份验证','',
      '## 主结论','',decision,'',
      '本轮没有做 embedding 全局对齐，没有复用 Audio-VAD 中心窗，没有按测试 Audio 选择有声样本。活动代理预测与身份验证分别报告。既有 CycleGAN 创新点一及历史模型未修改。',
      '', '## 一、数据与独立划分','',
      f"400 个完整原始录制组，10 位用户、40 个提示词。以首个有效 IMU 时间戳＋0.5 秒为起点，按 3 秒窗／3 秒步长切分，末端不足完整窗舍弃，全部规则仅依赖 IMU。共 {len(w)} 窗：训练 {rows_by_role['train']}、开发 {rows_by_role['development']}、测试 {rows_by_role['test']}。窗口互不重叠，可能含零次、一次或多次同一提示词发声。",
      '', '采集没有独立逐遍提示事件，故采用连续网格；异窗监督是同人同词录制内的非重叠窗轮换，不称为经过独立提示标记确认的错遍标签。',
      '', '固定随机种子 20261124，按数字／英文指令／英文短句／中文分层 24/8/8。词划分在波形提取前锁定并核对哈希；每个词全部用户和全部窗口属于同一角色。新模型、标准化、阈值选择都不使用测试词。此队列历史上已被研究，独立划分不等同新采集盲测。']
    for role in ['train','development','test']:
      lines+=['',f"- {role}："+'、'.join(f"{word}:{vocabulary[word]}" for word in split[role])+'。']
    lines+=['','本轮最终测试八词的 Audio 和 Audio 时间戳文件从未打开；测试 Audio 标签数组全部缺失。训练/开发 Audio 用于能量活动代理及其开发选择，不用于裁切任何 ACC 窗。源文件 400 个哈希均不同，词语标签跨用户一致。',
      '', '历史 manifest 仅用于枚举录制文件与匿名用户/词 ID，没有使用其语音段起止边界。测试词 IMU 重建从原始时间轴独立完成。',
      '', '## 二、匹配预算与控制','',
      '三个训练组使用相同 2081 参数时序 CNN、初始化、采样批次和 400 步 AdamW 预算；每批八位基础用户各四窗。输入为七个 ACC 局部频带功率、七个逐窗中位数中心化通道及低频运动代理。身份端始终是相同的 16 维 ACC 统计和余弦原型。',
      '', '- ACC_self_trained：只用 ACC 自身能量活动代理监督；概率阈值也只按 ACC 自身开发代理选择，全流程不需要 Audio。',
      '- Audio_correct：训练 Audio 的 300–3500 Hz 录制级能量活动代理；监督与当前 ACC 时段对应。',
      '- Audio_wrongwindow：同人同词不同非重叠窗的 Audio 活动代理；边际分布保持，但时刻对应被破坏。',
      '- Fixed_time_prior：训练 Audio 在固定三秒窗位置上的平均活动概率，不读取 ACC；是无优化器的非参数位置参照，未假称与神经组等训练量。',
      '- ACC_direct：ACC 自身能量掩码的额外手工参考，没有神经训练。',
      '', '每目标另留一位循环开发用户，二者均不进入掩码预测器训练，其余八位训练。输入/特征标准化、低频运动 Q95 和缺失填补只用八人训练词。概率阈值只在内部开发身份的开发词上选 F1；纯 ACC 组按自身代理，其余按 Audio 代理。没有使用身份 EER/HTER 选择活动模型。',
      '', 'Audio 活动标签按完整录制估计噪声/能量阈值；训练时用 host callback–PCM frame-end 仿射映射处理时间轴，不使用逐词/逐窗延迟择优。它仍是未独立物理标定的能量代理，不是人工 VAD 真值。',
      '', '## 三、真正无 Audio 的查询流程','',
      '输入原始固定 IMU 窗 → ACC 自行预测活动 → 与 IMU 有效帧、统一低频运动规则相交 → 16 维统计 → 余弦身份分数。注册使用 12 个分层训练词各首个完整窗，所有组完全一致，不能按 Audio 质量挑注册窗。',
      '', '认证校准使用开发八词的目标与其余九位用户窗口；规则预先固定为最低校准 HTER，同分优先 FAR/FRR 更平衡、更高阈值。150 个阈值全部落盘后才计算测试身份分数。目标开发身份数据用于个人认证校准，不进入活动模型训练。背景仍是这十人，因此不是未见攻击者开放集。',
      '', '活动不足 12 帧的窗一律拒绝（分数 −1.1，阈值不允许接纳），不从测试分母移除。以下指标包含全部测试窗；拒绝可能静音的合法录制窗也计入 FRR，是含检测/分段的保守连续窗任务。覆盖率只表示活动足够的窗比例，不是真实有声召回率。',
      '', '## 四、活动预测：留出目标身份的开发词','',
      '最终测试 Audio 不可用，因此不报告测试词 VAD F1；这里只在留出目标身份的开发词做事后代理质量评估，不参与其阈值选择。',
      '', '|方法|活动代理 F1|平衡准确率|活动代理 AUC|','|---|---:|---:|---:|']
    for v in quality:lines.append(f"|{v['method']}|{v['f1']:.4f}|{v['balanced_accuracy']:.4f}|{v['proxy_auc']:.4f}|")
    lines+=['','|正确 Audio−对照|代理 AUC 差|十目标簇95%区间|','|---|---:|---|']
    for v in quality_comparisons:
      lo,hi=v['target_cluster_ci95'];lines.append(f"|{v['control']}|{v['proxy_auc_delta']:+.4f}|[{lo:+.4f}, {hi:+.4f}]|")
    lines+=['','这些活动结果只支持 ACC 中存在可预测 Audio 活动代理的信息；输入含低频运动，尚未隔离声带振动和伴随动作的贡献，不能称为身份特异性生理信息。',
      '', '## 五、全部独立测试词窗口的身份结果','',
      '十用户、三个种子宏平均；EER、HTER、FAR、FRR、覆盖率和 Top1 为百分比。分数来自当前 16 维统计流程，不与历史神经身份编码器/CycleGAN 协议直接比较。',
      '', '|方法|EER|HTER|FAR|FRR|AUC|覆盖率|十人 Top1|','|---|---:|---:|---:|---:|---:|---:|---:|']
    for name in names:
      v=next(v for v in s['macro'] if v['method']==name);lines.append(f"|{name}|{100*v['eer']:.3f}|{100*v['hter']:.3f}|{100*v['far']:.3f}|{100*v['frr']:.3f}|{v['auc']:.5f}|{100*v['query_coverage']:.2f}|{100*v['top1_10way']:.2f}|")
    lines+=['','## 六、预先锁定的核心对照与不确定性','',
      '正确监督必须在 EER/AUC 同时稳定优于同预算纯 ACC、异窗和时间先验，且平均 HTER 不退化，才记为完整候选支持。分析规则在最终测试分数文件生成前锁定。按目标用户先平均三种子，再 bootstrap 十用户；不把窗口/帧作为独立受试者。共享背景仍有依赖，区间为历史队列探索性描述。',
      '', '|正确 Audio−对照|指标|差/百分点|95%区间/百分点|改善用户|','|---|---|---:|---|---:|']
    for v in comparisons:
      lo,hi=v['target_cluster_ci95'];lines.append(f"|{v['control']}|{v['metric']}|{100*v['delta']:+.3f}|[{100*lo:+.3f}, {100*hi:+.3f}]|{v['improved_users']}/10|")
    lines+=['','## 七、逐用户测试结果（三种子平均）','','|用户|方法|EER%|HTER%|FAR%|FRR%|AUC|覆盖率%|','|---|---|---:|---:|---:|---:|---:|---:|']
    for us in range(1,11):
      for name in names:
        v=next(v for v in s['per_user'] if v['user']==us and v['method']==name);lines.append(f"|U{us:02d}|{name}|{100*v['eer']:.3f}|{100*v['hter']:.3f}|{100*v['far']:.3f}|{100*v['frr']:.3f}|{v['auc']:.4f}|{100*v['query_coverage']:.2f}|")
    lines+=['','## 判断与下一步','',decision,
      '', f"核心分项：相对异窗监督稳定 EER/AUC 优势={core('Audio_wrongwindow')}；相对固定位置先验={core('Fixed_time_prior')}；相对同预算纯 ACC={core('ACC_self_trained')}；平均 HTER 不退化={utility}。",
      '', '若活动预测提高而身份性能未提高，瓶颈已从“是否只记住固定发声位置”转为“活动掩码选出的信息是否对身份有额外价值”。更精确的发声区域并不自动等于更好的身份特征，不能仅据代理 AUC 推断认证有效。',
      '', '本轮身份统计保持固定，检验的是硬活动掩码路线；身份损失没有参与活动模型训练。活动辅助与身份目标的联合优化未在本轮测试。',
      '', '本轮没有尝试更大网络、全局 embedding 对齐、测试词权重搜索或有利用户筛选。缺少独立物理同步标记和跨天/重戴确认，仍不能把任何提升归因于唯一的生理传递机制。',
      '', '## 独立核验与公开范围','',
      '从目标1八个测试词的原始 IMU 重新构造窗口，不打开 Audio 或 Audio 时钟；重放三种模型的 ACC-only 推理、掩码与最终指标。核查数据哈希、词划分、目标隔离、全部窗口计数和阈值提前锁定。',
      '', '公开脚本、词划分、匿名逐用户/逐种子指标和汇总阈值分布；不上传原始波形、窗口、身份映射、掩码、模型、逐样本分数或个人阈值。']
    (out/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    result=dict(decision=decision,activity_proxy_supported=activity,audio_specific_identity_supported=core('Audio_wrongwindow'),full_support=passed,quality=quality,quality_comparisons=quality_comparisons,identity_comparisons=comparisons)
    (out/'decision_public.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'mask_selection_summary_public.json').write_text(json.dumps({name:dict(Counter(str(v['threshold']) for v in selection if v['method']==name)) for name in names[1:]},indent=2))
    (out/'vocabulary_public.json').write_text(json.dumps(vocabulary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(decision=decision,activity_proxy_supported=activity,audio_specific_identity_supported=core('Audio_wrongwindow'),quality=quality,identity_comparisons=comparisons),ensure_ascii=False))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
