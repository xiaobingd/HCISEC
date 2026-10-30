"""Local independent NumPy verification and report of nine completed baseline heads."""
import sys,json,zipfile
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'work/public_signal_runtime_20261028'))
import numpy as np
R=Path('outputs/mei_strict_baseline_20261029')
def main():
 with zipfile.ZipFile(R/'review_private.zip') as z:
  assert z.testzip() is None
  assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in z.namelist());z.extractall(R/'review_private')
 private=R/'review_private';m=json.loads((private/'manifest_private.json').read_text());summary=json.loads((R/'verified_summary_public.json').read_text());r=json.loads((R/'results_public.json').read_text());checked=[]
 assert len(summary['results'])==9 and summary['all_nine_complete']
 for u,s in m['sources'].items():
  groups=np.asarray(s['content_groups']);i=np.arange(s['rows']);assert np.array_equal(groups,np.where(i<100,i%10,10+(i-100)%20))
  assert sorted(sum(s['roles'].values(),[]))==list(range(s['rows']))
  for k,rows in s['roles'].items():assert set(groups[rows])==set(m['groups'][k])
 for u,a in summary['results'].items():
  d=np.load(private/('U%s/auth/scores_private.npz'%u));g=d['final_genuine'];n=d['final_impostor'];labels=d['attack_labels'];t=r[u]['final_attack']['threshold']
  assert len(g)==a['genuine_n']==len(m['sources'][u]['roles']['test'])
  assert len(n)==a['attack_n']==sum(len(s['roles']['test']) for v,s in m['sources'].items() if v!=u)
  assert np.isfinite(g).all() and np.isfinite(n).all()
  auc=float(((g[:,None]>n[None,:])+.5*(g[:,None]==n[None,:])).mean());assert abs(auc-a['AUC'])<1e-12
  assert int((g>=t).sum())==a['genuine_accepted'] and int((n>=t).sum())==a['attack_accepted']
  assert abs(np.mean(g>=t)-a['TAR'])<1e-12 and abs(np.mean(n>=t)-a['FAR'])<1e-12
  for v in range(1,10):
   if str(v)==u:continue
   name='bo' if v==1 else 'user%d'%v;z=n[labels==v-1];b=a['by_attacker'][name];assert len(z)==b['n'] and int((z>=t).sum())==b['accepted']
  c=json.loads((private/('U%s/cyclegan/COMPLETE.json'%u)).read_text());assert c['epoch']==200 and len(c['checkpoint_sha256'])==64
  assert len(r[u]['history'])==20 and r[u]['best_epoch']==a['selected_epoch'];checked.append(int(u))
 out=dict(status='PASS',verified_targets=sorted(checked),zip_crc=True,group_mapping_and_disjointness=True,source_rows_complete=True,original_query_counts=True,all_eight_attack_sources=True,pairwise_auc=True,epoch200_markers=True,auth100_epoch_history=True)
 (R/'local_verification_public.json').write_text(json.dumps(out,indent=2));print(out)
 vals=list(summary['results'].values());macro={k:float(np.mean([a[k] for a in vals])) for k in ('TAR','FAR','AUC')};macro['HTER']=float(np.mean([(1-a['TAR']+a['FAR'])/2 for a in vals]));(R/'macro_public.json').write_text(json.dumps(macro,indent=2))
 lines=['# Mei九人严格内容组隔离baseline：完整结果（2026-10-29）','',
 '九个个人CycleGAN均从头完成200轮，九个原trainb完整772维认证头均完成100轮训练及选模；服务器和本地独立计数、pairwise AUC核验通过。全部八个攻击来源保留。','',
 '严格性以数字10项/指令20项的推断分组为条件；逐条词义、跨用户起始相位未独立转写核验，缺行仍依赖尾部缺失假设。闭集已知身份攻击、同会话历史数据探索性复测；不声称开放集或跨会话。','',
 '|目标|本人接受|TAR|FRR|攻击误接受|FAR|AUC|头选中轮次|','|---|---|---:|---:|---|---:|---:|---:|']
 for u in sorted(summary['results'],key=int):
  a=summary['results'][u];lines.append('|U%s|%d/%d|%.2f%%|%.2f%%|%d/%d|%.2f%%|%.4f|%d|'%(u,a['genuine_accepted'],a['genuine_n'],100*a['TAR'],100*(1-a['TAR']),a['attack_accepted'],a['attack_n'],100*a['FAR'],a['AUC'],a['selected_epoch']))
 lines+=['','九人宏平均：TAR %.2f%%，FAR %.2f%%，AUC %.4f，HTER %.2f%%。'%(100*macro['TAR'],100*macro['FAR'],macro['AUC'],100*macro['HTER']),
 '合并本人计数283/441，合并攻击误接受1271/3528。同一录制在多个个人模型上作为攻击，不把这些配对当独立采集样本；宏平均AUC不是混合不同模型logit重算AUC。','',
 '## 方法与解释','',
 '生成训练120条/12组、验证30条/3组；认证训练50条/5组、选模30条/3组、校准20条/2组、最终47–50条/5组。角色组互斥、每条源行只分配一次。个人G/F仅使用所属用户的生成训练组，seed42，从头训练原归档4ResNet G/9UNetResNet F，固定200轮。',
 '认证完整772维=输入浅层128＋瓶颈均值/标准差512＋有符号循环差分浅层128＋差分统计4；没有直接拼接纯循环640维。旧trainb融合512、SupCon投影128、加权本人BCE，100轮、seed42、原增强/优化不变；不加教师、diffusion或删样本。',
 '每目标50正/400负拟合，30正/240负选模，20正/160负校准；原验证HTER/AUC选择过程使用校准阈值，因此校准参与checkpoint选择，不是纯事后校准。最终特征仅在checkpoint与阈值锁定后提取。',
 '全部八个攻击身份都是训练负类，本表FAR不是未见攻击者FAR。历史数组原始归一化统计来源未完全追溯。',
 'U1的47/50本人接受、19/391误接受较好；其余目标在当前阈值下FAR较高或本人拒绝明显，U3 AUC约0.505接近随机。本轮不支持多数用户已有可靠跨内容认证，也不以U1推广到九人。',
 '尚不能量化跨内容导致多少退化：旧同内容标签有误，历史随机划分还存在训练数、架构及查询差异。需要修正后的同内容匹配对照和直接ACC强基线，才能分离内容因素与转换增量。','',
 '## 逐攻击来源','', '|目标/攻击者|'+'|'.join('U%d'%v for v in range(1,10))+'|','|---|'+'|'.join(['---']*9)+'|']
 for u in range(1,10):
  cells=[];a=summary['results'][str(u)]
  for v in range(1,10):
   if v==u:cells.append('本人');continue
   b=a['by_attacker']['bo' if v==1 else 'user%d'%v];cells.append('%d/%d（%.2f%%）'%(b['accepted'],b['n'],100*b['far']))
  lines.append('|U%d|'%u+'|'.join(cells)+'|')
 lines+=['','[固定协议](2026-10-29-mei-strict-content-baseline-protocol.md)；[创新点判断与公平比较要求](2026-10-29-content-generalization-innovation-assessment.md)。',
 '服务器/autodl-fs/data/mei_strict_baseline_20261029，队列已全部完成。公开verified_summary_public.json包含全部逐来源计数，macro_public.json及local_verification_public.json；私有原始信号、角色索引、逐样本分数和权重不上传。']
 Path('docs/experiments/2026-10-29-mei-strict-content-baseline-results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
if __name__=='__main__':main()
