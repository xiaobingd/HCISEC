"""Report raw matched-target anti-smoothing outputs; never sharpen images."""
import os,json,hashlib
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path('outputs/mei_sharp_cycle_20261019');review=root/'review'
runs=json.loads((review/'results.json').read_text());assert len(runs)==12
prior=json.loads(Path('outputs/mei_audio_repair_20261018/full_context_review/summary.json').read_text())
summary={}
for route in ['continue','focal','focal_residual','structure','structure_residual','highpass']:
    rr=[r for r in runs if r['route']==route]
    summary[route]=dict(macro={k:float(np.mean([r['metrics']['macro'][k] for r in rr])) for k in rr[0]['metrics']['macro']},extra={k:float(np.mean([r['metrics']['extra']['macro'][k] for r in rr])) for k in rr[0]['metrics']['extra']['macro']},sharp={k:float(np.mean([r['sharp'][k] for r in rr])) for k in rr[0]['sharp'] if isinstance(rr[0]['sharp'][k],(int,float))},train_audio_mae=float(np.mean([r['train_audio_mae'] for r in rr])))
baseline={k:float(np.mean([r['final']['macro'][k] for r in prior['runs']])) for k in prior['runs'][0]['final']['macro']}
baseline_extra={k:float(np.mean([r['final']['extra']['macro'][k] for r in prior['runs']])) for k in prior['runs'][0]['final']['extra']['macro']}
paired={}
for route in ['focal','focal_residual','structure','structure_residual','highpass']:
    deltas=[]
    for seed in [42,43]:
        target=next(r for r in runs if r['route']==route and r['seed']==seed)
        ref=next(r for r in runs if r['route']=='continue' and r['seed']==seed)
        aa={r['recording']:r['audio_mae'] for r in target['metrics']['per_record']};bb={r['recording']:r['audio_mae'] for r in ref['metrics']['per_record']}
        deltas.append([aa[n]-bb[n] for n in sorted(aa)])
    delta=np.mean(deltas,axis=0);rng=np.random.default_rng(20261019);boot=delta[rng.integers(0,len(delta),size=(5000,len(delta)))].mean(1)
    paired[route]=dict(audio_mae_change_vs_continue=float(delta.mean()),conditional_record_bootstrap95=np.percentile(boot,[2.5,97.5]).tolist(),records_improved=int((delta<0).sum()),records=60,limits='Conditional on reused validation, seed-average predictions; not fresh sessions or architecture selection uncertainty.')
public=dict(baseline=baseline,baseline_extra=baseline_extra,routes=summary,paired=paired,seed_results=[dict(seed=r['seed'],route=r['route'],macro=r['metrics']['macro'],extra=r['metrics']['extra']['macro'],sharp=r['sharp'],train_audio_mae=r['train_audio_mae']) for r in runs],protocol=json.loads((review/'protocol.json').read_text()),source_hashes={name:hashlib.sha256((Path('work')/name).read_bytes()).hexdigest() for name in ['train_mei_sharp_cycle.py','report_mei_sharp_cycle.py','audit_mei_sharp_cycle.py','pack_mei_sharp_review.py']})
if (review/'controls.json').exists():public['controls']=json.loads((review/'controls.json').read_text())
initial_controls=[r for r in public.get('controls',{}).get('runs',[]) if r['route']=='initial']
baseline_sharp={k:float(np.mean([r['sharp'][k] for r in initial_controls])) for k in initial_controls[0]['sharp'] if isinstance(initial_controls[0]['sharp'][k],(float,int))} if initial_controls else {}
public['baseline_sharp']=baseline_sharp
(root/'summary.json').write_text(json.dumps(public,indent=2),encoding='utf-8')
old=dict(np.load('outputs/mei_audio_repair_20261018/full_context_review/full_42_whole_record_plots.npz'))
plots={r:dict(np.load(review/(r+'_42_plots.npz'))) for r in summary}
for p in plots.values():assert np.array_equal(p['ids'],old['ids']) and np.array_equal(p['real_audio'],old['real_audio'])
rows=[('Real microphone',old['real_audio']),('Previous full-context',old['generated_audio'])]+[(r,plots[r]['generated_audio']) for r in summary]
fig,axs=plt.subplots(len(rows),4,figsize=(15,11),layout='constrained')
for ri,(name,arr) in enumerate(rows):
    for col in range(4):
        axs[ri,col].imshow(arr[col],origin='lower',aspect='auto',cmap='magma',vmin=-1,vmax=1,extent=[0,2,0,80],interpolation='nearest')
        axs[ri,col].set_xlabel('Time (s)')
        if ri==0:axs[ri,col].set_title(str(old['ids'][col]),fontsize=7)
    axs[ri,0].set_ylabel(name+'\nMel bin')
fig.suptitle('Anti-smoothing complete-cycle ablation / seed42 / first2 sources per task\nRaw outputs, shared scales, no visual filtering; explored VAL, no identity claim')
fig.savefig(root/'comparison.png',dpi=150);plt.close(fig)
fig,axs=plt.subplots(2,3,figsize=(12,6),layout='constrained')
panels=[('Real microphone',old['real_audio'][0]),('Previous full-context',old['generated_audio'][0]),('Same-budget continue',plots['continue']['generated_audio'][0]),('Paired focal + full resolution',plots['focal_residual']['generated_audio'][0]),('Structure + full resolution',plots['structure_residual']['generated_audio'][0]),('Frozen base + highpass branch',plots['highpass']['generated_audio'][0])]
for ax,(name,arr) in zip(axs.flat,panels):
    ax.imshow(arr,origin='lower',aspect='auto',cmap='magma',vmin=-1,vmax=1,interpolation='nearest');ax.set_title(name,fontsize=10);ax.set_xlabel('Frame index');ax.set_ylabel('Mel bin')
fig.suptitle('Fixed first VAL recording / seed42 / raw output and common scale')
fig.savefig(root/'compact.png',dpi=160);plt.close(fig)
fig,axs=plt.subplots(8,4,figsize=(15,14),layout='constrained')
for col in range(4):
    entries=[('Real microphone',old['real_audio'][col])]+[(r,plots[r]['generated_audio'][col]) for r in summary]+[('Previous',old['generated_audio'][col])]
    for row,(name,a) in enumerate(entries):
        axs[row,col].imshow(a[5:45,40:120],origin='lower',aspect='auto',vmin=-1,vmax=1,cmap='magma',interpolation='nearest',extent=[.4,1.2,5,45])
        if col==0:axs[row,col].set_ylabel(name)
        axs[row,col].set_xlabel('Time(s)')
fig.suptitle('Fixed same .4-1.2s / Mel5-45 zoom / no per-panel contrast or sharpening')
fig.savefig(root/'detail_zoom.png',dpi=160);plt.close(fig)
lines=['# mei 完整循环频谱过度平滑：配对细节与结构消融','',
'2026-10-19。完成六设置×两随机种子，共12个最终模型。目标是改善真实频谱细节对应，而非通过显示锐化或无条件纹理令图像看起来更清楚。结果均为模型原始输出，没有后处理。','',
f"**结论：尚未达到修复标准，不替换前轮基线。** 保守highpass分支使局部峰召回从{baseline_sharp['near_target_peak_recall']:.1%}增至{summary['highpass']['sharp']['near_target_peak_recall']:.1%}、局部起伏RMS比从{baseline_sharp['active_highpass_rms_ratio']:.3f}增至{summary['highpass']['sharp']['active_highpass_rms_ratio']:.3f}，但真实细节相关性{baseline['audio_detail_corr']:.4f}→{summary['highpass']['macro']['audio_detail_corr']:.4f}基本不变，误发声率{baseline_extra['activity_false_positive_rate']:.1%}→{summary['highpass']['extra']['activity_false_positive_rate']:.1%}上升。视觉可见更多横向条纹，却没有恢复真实音频中清楚、随时间变化的谐波。更强结构约束的起伏较大，细节对应反而变差。仅延长训练也没有解决问题。",'',
'highpass的冻结基底设计更好地保留前轮模型行为，优于相同预算全量继续训练，但不优于前轮模型的真实细节对应。该结果不能归因于特定一个组件，也不能把局部峰增加当作身份信息提升。','',
'## 数据和训练','',
'沿用清理后的 Data0723/wangjingmei：180个独立训练录制、60个反复探索过的验证录制。同一冻结预处理和归一化、原时间戳，无逐人滤波或调增益。推理仅输入约2秒ACC，属于离线未来上下文模型。未打开测试或认证保留数据，也没有训练认证头。','',
'各组从前一轮相同种子的约2秒完整循环 G/F 初始化，180轮×4更新，批次8个不同训练录制，学习率0.0001。判别器均重新初始化并预热40次；相同种子的数据批次和中间Mel噪声随机流一致。固定终点，不通过验证集选择epoch。保留 ACC→生成Mel→ACC 和音频→ACC→音频双向循环；F没有ACC或潜在信息旁路。种子43的首个继续训练进程中断，未完成结果未使用，从相同初始权重重新完成。','',
'- continue：相同预算继续原损失，作为训练预算对照。',
'- focal：在continue之上增加局部16/32大小Mel图块的复数傅里叶配对误差，以及更强的配对局部对比误差。',
'- focal_residual：相同focal目标，再加不下采样、不使用归一化的全分辨率残差分支，零初始化输出头并限制增量至±0.35。',
'- structure：在continue之上增加16×16配对图块去均值余弦损失及局部对数标准差误差，以目标局部标准差加权。',
'- structure_residual：相同structure目标再加上述全分辨率分支。','',
'- highpass：最后追加的保守方案，冻结同种子前轮基础G，只训练细节分支与F。细节增量限制为±0.35、去除9频带局部均值，形状/幅度损失减为1/0.5；这是冻结基底＋受限分支＋弱结构目标的组合，不能归因于单一改动。','',
'结构组是看到首轮频域细节约束未改善后的追加探索，不是预先注册的独立确认实验。残差头的初始输出已与原G同批次逐元素相等核验。所有图使用相同色阶和nearest显示，无逐图增益、锐化或选择最好样本。','',
'Focal部分借鉴[Focal Frequency Loss原论文](https://openaccess.thecvf.com/content/ICCV2021/html/Jiang_Focal_Frequency_Loss_for_Image_Reconstruction_and_Synthesis_ICCV_2021_paper.html)的自适应难频率加权。这里是Mel图像的空间傅里叶系数，不是波形频谱，也不代表恢复了语音物理基频；图块实现是本项目适配。结构损失为本项目实现。','',
'## 相同目标片段上的结果','',
'沿用前轮共同口径：60录制×前三个原0.8秒窗口×各65个内部帧，共180片段。所有行的目标完全相同。不能与更早240个完整81帧窗口的0.277等数字直接比较。以下为两随机种子平均；↑/↓仅表示指标方向。','',
'|设置|音频MAE↓|细节相关↑|局部对比相关↑|频带起伏幅度比（真实=1）|局部峰召回↑|误发声率↓|ACC循环MAE↓|',
'|---|---:|---:|---:|---:|---:|---:|---:|',
f"|前轮2秒模型|{baseline['audio_mae']:.4f}|{baseline['audio_detail_corr']:.4f}|{baseline_extra['local_spectral_contrast_corr']:.4f}|{baseline_sharp['active_highpass_rms_ratio']:.3f}|{baseline_sharp['near_target_peak_recall']:.2%}|{baseline_extra['activity_false_positive_rate']:.2%}|{baseline['ACC_cycle_mae']:.4f}|"]
for route,r in summary.items():
    m,e,s=r['macro'],r['extra'],r['sharp']
    lines.append(f"|{route}|{m['audio_mae']:.4f}|{m['audio_detail_corr']:.4f}|{e['local_spectral_contrast_corr']:.4f}|{s['active_highpass_rms_ratio']:.3f}|{s['near_target_peak_recall']:.2%}|{e['activity_false_positive_rate']:.2%}|{m['ACC_cycle_mae']:.4f}|")
lines+=['',
'频带起伏幅度比是目标活动帧内，生成谱与真实谱经9频带局部均值去除后的RMS比，描述是否过平滑；接近1不保证细节位置正确。峰指标仅指归一化Mel图上0.025差值门限的局部极大值、允许±1频率bin，不是独立F0/谐波准确率。误发声由固定TRAIN-only能量门限代理定义，不是人工标注语音。','',
'## 局限与使用条件','',
'该验证集已经反复用于方法探索，不能作为新模型无偏泛化结论。增加起伏但不提高真实细节对应、增加误发声，或明显损伤ACC循环的方案，都不能称为成功修复。训练误差和逐种子结果见summary.json。记录级bootstrap区间以固定当前模型和该验证集为条件，未包含方法选择不确定性。','',
'量化与平滑控制仅作用于归一化Mel数值，不是音频信道或声码器测试；错录制输入影响只说明输入依赖。频谱质量或循环误差不证明身份信息保留。未生成声码器波形、未做听感/PESQ/STOI或身份认证评估。','',
'代码、协议、汇总指标公开；原始数据、私有信号图、派生信号数组和模型权重不上传公共仓库。']
(root/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(dict(baseline=baseline,routes=summary,paired=paired),indent=2))
