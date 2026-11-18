"""Publish aggregate-only results, preserving every preselected route and target."""
import json
from pathlib import Path
R=Path('outputs/mei_strict_autosupplement_20261029')
def main():
 source=json.loads((R/'verified_results_public.json').read_text());v=json.loads((R/'local_verification_public.json').read_text());assert v['status']=='PASS' and v['routes']==27
 routes=('baseline_rbf','auto_without_diffusion','auto_with_diffusion');out={};macro={}
 for u,row in source['results'].items():
  dm=row['diffusion_training'];assert not any(k in dm for k in ('fit_ids','selection_ids'))
  out[u]=dict(routes=row['routes'],trainb_reference=row['trainb_reference'],diffusion_selected_epoch=dm['selected_epoch'],diffusion_validation=dm['selected_validation'])
 for route in routes:macro[route]={k:sum(row['routes'][route][k] for row in out.values())/9 for k in ('tar','far','auc','hter')}
 public=dict(status='PASS',targets=9,routes=27,results=out,macro=macro,scope='Closed-set strict mixed-content groups under inferred ordering; same-session historical exploratory data',selection='Original training-only nested word CV and optional score fusion, no final-based fallback')
 (R/'summary_public.json').write_text(json.dumps(public,indent=2));(R/'macro_public.json').write_text(json.dumps(macro,indent=2))
 lines=['# 严格跨内容组：教师与Diffusion自主选择结果（2026-10-29）','',
 '九位用户全部完成：复用新严格分组CycleGAN200轮；每位新训三类个人教师各600步，以及外部ESMB预训练初始化的双向个人Diffusion50轮。27条认证路线通过服务器独立复算和本地CRC、计数、pairwise AUC、上游Diffusion索引和训练OOF选路重放核验。没有按最终成绩换回其他路线。','',
 '## 主要判断','',
 'ACC教师/原始统计补充有平均增量，U4最明确，但不是所有人获益。Diffusion只在U4/U8/U9被训练折选中；相对于同用户不含Diffusion的自动路线，三人的HTER均变差。不能把允许Diffusion后的整体改善全归给Diffusion，也不能因三人结果否定所有Diffusion方法。','',
 '|方案|宏平均本人接受率|宏平均FAR|宏平均AUC|宏平均HTER|','|---|---:|---:|---:|---:|']
 labels={'baseline_rbf':'纯772维RBF','auto_without_diffusion':'自动补充，不含Diffusion','auto_with_diffusion':'允许Diffusion的自动补充'}
 for route in routes:
  a=macro[route];lines.append('|%s|%.2f%%|%.2f%%|%.4f|%.2f%%|'%(labels[route],a['tar']*100,a['far']*100,a['auc'],a['hter']*100))
 lines+=['','## 全部用户、全部路线','', '|目标|方案|本人通过/测试|TAR|攻击误接受/测试|FAR|AUC|HTER|权重|','|---|---|---|---:|---|---:|---:|---:|---|']
 for u in sorted(out,key=int):
  for route in routes:
   a=out[u]['routes'][route];t=a['threshold'];g=int(round(a['tar']*a['genuine_n']));n=sum(x['accepted'] for x in a['attack_by_user'].values())
   lines.append('|U%s|%s|%d/%d|%.2f%%|%d/%d|%.2f%%|%.4f|%.2f%%|%s|'%(u,labels[route],g,a['genuine_n'],a['tar']*100,n,a['impostor_n'],a['far']*100,a['auc'],a['hter']*100,json.dumps(a['weights'])))
 lines+=['','## 公平对照与隔离','',
 '本轮自主选择沿用此前RBF分数融合。纯772维RBF与两条补充路线共用同一冻结G/F、相同角色、SVM候选和校准规则；上一轮原trainb神经头的宏平均TAR64.04%/FAR36.02%/AUC0.6894作为参考保留，但不能把换RBF带来的提升当成教师或Diffusion贡献。',
 '相对纯772维RBF，不含Diffusion的自动补充宏平均TAR提高约2.98个百分点、FAR降低约2.49个百分点；增加Diffusion候选后，相对不含Diffusion的自动路线，平均TAR略降、FAR略升。单seed、九身份及少量内容组，不作显著性/普遍有效性结论。',
 'U4：RBF为TAR72.34%/FAR19.80%/AUC0.8523；ACC身份教师＋原始统计为78.72%/10.66%/0.9256；再选Diffusion潜变量后为74.47%/9.39%/0.9232。最后一项FAR虽更低，本人拒绝更多，不能视为全面改善。',
 'U5：自动路线FAR从21.99%降到17.39%，但本人接受率从80%降到62%，HTER反而变差。U7同样出现明显本人拒绝；U6本人接受改善同时FAR变高。自动选择并不保证在未见词上复现开发集收益。',
 'old/identity/robust是三种教师训练目标的名称，全部在本轮严格上游组上新训，并非复用旧Mei权重。教师仅补充认证特征，没有反过来用身份损失更新本轮CycleGAN。Diffusion使用独立外部ESMB初始化，在各目标120训练/30验证上新微调50轮，不是从零200轮Diffusion。',
 '组划分与上一轮完全一致：生成拟合120/验证30；认证拟合50、选模30、校准20、最终47–50。自主选择仅用认证拟合的5外折/3内折内容CV，选模30条在本轮只提取未用于选路；权重0/.25/.5/1，稳定性门槛沿用旧流程，允许零补充。OOF用于选择，不冒充最终泛化。校准只在选路后确定阈值，最后才提取最终特征。',
 '所有八个攻击身份均是已知负类，全部最终查询保留。内容组以用户确认的数字/指令组成和Audio周期推断，逐条词义/跨用户起始相位、尾部缺失及历史前端归一化来源仍有限制；同会话历史数据探索性复测，不声称未知攻击、跨会话或新增ACC不存在的身份信息。','',
 '## 复现与存档','',
 '协议：[严格自主选择协议](2026-10-29-strict-autosupplement-protocol.md)。服务器/autodl-fs/data/mei_strict_autosupplement_20261029，主队列和独立核验已完成。公开summary_public.json含每条路线全部逐攻击来源接受数，macro_public.json和local_verification_public.json；私有原始信号、模型、向量、逐样本分数和索引不上传。']
 Path('docs/experiments/2026-10-29-strict-autosupplement-results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 print(json.dumps(macro,indent=2))
if __name__=='__main__':main()
