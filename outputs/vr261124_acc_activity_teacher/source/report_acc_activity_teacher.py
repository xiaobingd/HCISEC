import argparse,json
from collections import Counter
from pathlib import Path
import numpy as np
def main(root):
    out=root/'conditional_correspondence/acc_activity_teacher';read=lambda n:json.loads((out/n).read_text())
    s=read('summary_public.json');q=read('mask_quality_public.json');selection=read('mask_threshold_selection_private.json')
    names=['ACC_self','Audio_proxy_upperbound','Predicted_correct','Predicted_wrongrepeat','Time_prior'];rng=np.random.default_rng(20261124);boot=rng.integers(0,10,(10000,10));comparisons=[]
    for control in ['ACC_self','Predicted_wrongrepeat','Time_prior','Audio_proxy_upperbound']:
      for metric in ['eer','hter','auc']:
        delta=np.array([next(v[metric] for v in s['per_user'] if v['user']==us and v['method']=='Predicted_correct' and v['scope']=='new_words')-next(v[metric] for v in s['per_user'] if v['user']==us and v['method']==control and v['scope']=='new_words') for us in range(1,11)])
        ci=np.quantile(delta[boot].mean(1),[.025,.975]);comparisons.append(dict(control=control,metric=metric,delta=float(delta.mean()),target_cluster_ci95=ci.tolist(),improved_users=int((delta>0 if metric=='auc' else delta<0).sum())))
    passes=all(v['target_cluster_ci95'][1]<0 for v in comparisons if v['control'] in ['ACC_self','Predicted_wrongrepeat','Time_prior'] and v['metric'] in ['eer','hter'])
    decision=('预测掩码同时超过 ACC 自身掩码、错遍监督及时间先验，具备进入独立查询词确认的候选证据。' if passes else '本轮未通过稳定 ACC-only 认证增益标准。分别依据代理预测质量、正确/错遍监督差异和时间先验结果判断瓶颈；不在当前查询词上继续调阈值或训练预算。')
    quality=[]
    for method in ['Predicted_correct','Predicted_wrongrepeat','Time_prior']:
      rr=[v for v in q if v['method']==method and v['scope']=='new_words']
      quality.append(dict(method=method,**{k:float(np.mean([v[k] for v in rr])) for k in ['f1','balanced_accuracy','proxy_auc','positive_fraction']}))
    counts={method:dict(Counter(str(v['probability_threshold']) for v in selection if v['method']==method)) for method in ['Predicted_correct','Predicted_wrongrepeat','Time_prior']}
    lines=['# Audio 活动代理监督 ACC 掩码：查询端移除 Audio 的对照','',
      '## 方法与协议','',
      f"仅训练一个 {read('verification_public.json')['parameters']} 参数的小型时序 CNN，不改身份编码器或既有基线。输入为七个 ACC 对数频带功率、七个逐条减中位数的动态通道和低频运动通道。Conv15→16(k5)→16(k3)→1，GroupNorm/SiLU；输出逐帧活动概率。",
      '', '教师标签是上一轮 Audio 300–3500 Hz 能量活动代理，不是人工 VAD 真值。400 步 AdamW，正确监督与同人同词错四遍监督使用相同初始化、批次、预算及类别平衡规则。没有身份损失、没有用认证结果选检查点。',
      '', '10 人、3 种子、每目标排除本人及下一位开发身份，其余 8 人训练，共 60 个模型。输入标准化、特征标准化、填补、运动 Q95 仅用八位训练身份的前 12 个词。概率阈值在五个预设值中，按开发身份另外四词的活动代理 F1 选择，完全不看目标身份或身份认证指标。',
      '', '时间先验不读取信号，只使用基础训练录制在每个时间位置的平均 Audio 活动概率，以相同开发规则选择阈值。它检查模型是否仅学到 VAD 中心裁剪后的固定位置。',
      '', '所有方法输出相同 16 维 ACC 谱形/波动/能量统计特征和余弦原型；这不是已定版的 CycleGAN/RBF 基线，不能跨协议直接比较分数。注册为前六词各第 0/1 遍共 12 条；认证校准为后六训练词第 0–3 遍；检查为这些词第 4–7 遍，或另外四词全部 8 遍。150 个认证阈值在查询评分前锁定，规则沿用本人 Q10 与攻击 Q95 中点。',
      '', '**Predicted_correct、Predicted_wrongrepeat、ACC_self、Time_prior 的掩码和身份统计推理接口不读取 Audio；Audio_proxy_upperbound 需要注册及查询 Audio，仅作离线参照。** 输入仍沿用历史 Audio-VAD 中心裁切的三秒 ACC 窗，因此尚不是从连续流裁切开始的端到端 ACC-only 验证。活动预测质量的事后评估读取 Audio 标签用于打分，标签不进入目标推理。',
      '', '本轮是 16 个既有 fit 词库内的内部跨词验证，四词在开发身份上用于概率阈值选择，因此不是对超参数完全盲的词。掩码预测器目标身份隔离，但背景仍是同一十人队列，不能称完整开放集。原始最终十五查询词没有使用。',
      '', '## 一、留出身份＋保留词的活动代理预测','','先按用户与种子汇总；帧不作为独立受试者。','','|方法|代理 F1|平衡准确率|代理 AUC|预测活动比例|','|---|---:|---:|---:|---:|']
    for v in quality:lines.append(f"|{v['method']}|{v['f1']:.4f}|{v['balanced_accuracy']:.4f}|{v['proxy_auc']:.4f}|{100*v['positive_fraction']:.2f}%|")
    lines+=['','## 二、相同身份流程的 ACC-only 结果','','EER/HTER/FAR/FRR/Top1 为百分比。Audio 辅助参照不算部署成绩。','','|方法|词范围|EER|HTER|FAR|FRR|AUC|十人 Top1|','|---|---|---:|---:|---:|---:|---:|---:|']
    for name in names:
      for scope in ['fit_words','new_words']:
        v=next(v for v in s['macro'] if v['method']==name and v['scope']==scope)
        lines.append(f"|{name}|{scope}|{100*v['eer']:.3f}|{100*v['hter']:.3f}|{100*v['far']:.3f}|{100*v['frr']:.3f}|{v['auc']:.5f}|{100*v['top1_10way']:.2f}|")
    lines+=['','## 三、目标簇配对比较','','预测正确监督−对照，EER/HTER 负差为好，AUC 正差为好。先平均三个种子，再重采样十个用户；区间为探索性描述，词和重复不当作独立人。','','|对照|指标|差/百分点|95%区间/百分点|改善用户|','|---|---|---:|---|---:|']
    for v in comparisons:
      lo,hi=v['target_cluster_ci95'];lines.append(f"|{v['control']}|{v['metric']}|{100*v['delta']:+.3f}|[{100*lo:+.3f}, {100*hi:+.3f}]|{v['improved_users']}/10|")
    lines+=['','## 逐用户保留词结果（三种子平均）','','|用户|方法|EER%|HTER%|FAR%|FRR%|AUC|','|---|---|---:|---:|---:|---:|---:|']
    for us in range(1,11):
      for name in names:
        v=next(v for v in s['per_user'] if v['user']==us and v['method']==name and v['scope']=='new_words')
        lines.append(f"|U{us:02d}|{name}|{100*v['eer']:.3f}|{100*v['hter']:.3f}|{100*v['far']:.3f}|{100*v['frr']:.3f}|{v['auc']:.4f}|")
    lines+=['','## 判断','',decision,
      '', '本轮正确监督的 HTER 相对 ACC 自身掩码改善，但 EER 基本未变；FAR 上升、FRR 下降，因此收益同时包含工作点与拒绝折衷。正确监督相对错遍监督的 AUC 存在正向线索，但主要 EER/HTER 尚未稳定超过错遍和时间先验。这个结果支持继续核验候选，不支持宣布全面身份判别增益。',
      '', '固定时间先验的代理 F1 高于 ACC 预测器，说明历史 VAD 中心裁片本身提供强位置信息。下一步最必要的是在不使用 Audio 确定查询中心的固定提示时间窗或连续 ACC 切片上确认，而不是在当前查询词上增加网络或重新选择阈值。',
      '', '物理同步仍未独立标定，教师为能量代理，三秒时间窗可能相互重叠；代理 F1 高不等于真正 VAD 准确率。缺少独立重戴/跨天验证，仍是历史队列探索。全部人和样本保留，有效帧不足采用基础训练均值填补。',
      '', '## 核验与公开边界','',
      '复核 30 个身份折、60 模型、90 次掩码阈值选择、150 个提前锁定认证阈值、300 个评估工作点。ACC_self 与 Audio 辅助参照必须回归上一轮完全相同身份组合/注册预算的原始统计结果；独立重放预测函数只提供 ACC，检查掩码与身份分数一致。',
      '', '只发布代码、匿名逐用户指标、训练摘要与阈值选择的总体分布。不上传 Audio/ACC、掩码、概率曲线、身份映射、检查点或个人阈值。']
    (out/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    (out/'decision_public.json').write_text(json.dumps(dict(decision=decision,quality=quality,comparisons=comparisons),ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'mask_selection_summary_public.json').write_text(json.dumps(dict(candidate_grid=[.3,.4,.5,.6,.7],aggregate_counts=counts),indent=2))
    print(json.dumps(dict(decision=decision,quality=quality,macro=[v for v in s['macro'] if v['scope']=='new_words']),ensure_ascii=False))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
