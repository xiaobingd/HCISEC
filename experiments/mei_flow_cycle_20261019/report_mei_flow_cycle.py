"""Report actual draws, not best-of-N or averaged generated spectrograms."""
import os,json,hashlib
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path('outputs/mei_flow_cycle_20261019');review=root/'review';runs=json.loads((review/'results.json').read_text());assert len(runs)==8
routes=['direct','velocity','data_prediction','decoupled'];controls=json.loads((review/'controls.json').read_text());assert len(controls['runs'])==8
def mean_dict(rows):return {k:float(np.mean([r[k] for r in rows])) for k in rows[0] if isinstance(rows[0][k],(int,float))}
seed_results=[]
for run in runs:
    draws=run['draws'];seed_results.append(dict(seed=run['seed'],route=run['route'],macro=mean_dict([r['metrics']['macro'] for r in draws]),extra=mean_dict([r['metrics']['extra']['macro'] for r in draws]),sharp=mean_dict([r['sharp'] for r in draws]),draw_macros=[r['metrics']['macro'] for r in draws]))
summary={}
for route in routes:
    rr=[r for r in seed_results if r['route']==route];summary[route]={k:mean_dict([r[k] for r in rr]) for k in ['macro','extra','sharp']}
prior=json.loads(Path('outputs/mei_audio_repair_20261018/full_context_review/summary.json').read_text());baseline=mean_dict([r['final']['macro'] for r in prior['runs']]);baseline_extra=mean_dict([r['final']['extra']['macro'] for r in prior['runs']])
paired={}
for route in routes:
    diffs=[]
    for seed in [42,43]:
        a=next(r for r in runs if r['route']==route and r['seed']==seed);ref=next(r for r in prior['runs'] if r['seed']==seed);b={r['recording']:r['audio_mae'] for r in ref['final']['per_record']}
        aa=[]
        for draw in a['draws']:aa.append({r['recording']:r['audio_mae'] for r in draw['metrics']['per_record']})
        diffs.append([np.mean([d[n] for d in aa])-b[n] for n in sorted(b)])
    delta=np.mean(diffs,axis=0);rng=np.random.default_rng(20261019);boot=delta[rng.integers(0,len(delta),size=(5000,len(delta)))].mean(1)
    paired[route]=dict(audio_mae_change_vs_previous=float(delta.mean()),record_bootstrap95=np.percentile(boot,[2.5,97.5]).tolist(),records_lower_error=int((delta<0).sum()),records=60,limits='Conditional on reused samepersonVAL/currentmodels; average seeds and randomdraw metrics, no selection uncertainty or new-session guarantee.')
pubcontrols=[]
for c in controls['runs']:
    cc={k:v for k,v in c.items() if k not in ['zero_prior_noise','one_NFE_same_weights']}
    for name in ['zero_prior_noise','one_NFE_same_weights']:
        val=c[name];cc[name]=dict(macro=val['metrics']['macro'],extra=val['metrics']['extra']['macro'],sharp=val['sharp']) if val is not None else None
    pubcontrols.append(cc)
public=dict(baseline=baseline,baseline_extra=baseline_extra,routes=summary,seed_results=seed_results,paired_audio_changes=paired,controls=pubcontrols,protocol=json.loads((review/'protocol.json').read_text()),source_hashes={n:hashlib.sha256((Path('work')/n).read_bytes()).hexdigest() for n in ['train_mei_flow_cycle.py','audit_mei_flow_cycle.py','pack_mei_flow_review.py','report_mei_flow_cycle.py']})
control_summary={}
for route in routes:
    cc=[c for c in pubcontrols if c['route']==route]
    control_summary[route]=dict(diversity=float(np.mean([c['mean_pointwise_std_across3draws'] for c in cc])),wrong_record_audio_mae_change=float(np.mean([d['wrong_record_audio_mae_change'] for c in cc for d in c['draws']])),constant_ACC_audio_mae_change=float(np.mean([d['constant_ACC_audio_mae_change'] for c in cc for d in c['draws']])),quant8_cycle_mae_increase=float(np.mean([d['quant8_cycle_mae_increase'] for c in cc for d in c['draws']])),blur3_cycle_mae_increase=float(np.mean([d['blur3_cycle_mae_increase'] for c in cc for d in c['draws']])))
    if route!='direct':control_summary[route]['zero_noise_macro']=mean_dict([c['zero_prior_noise']['macro'] for c in cc])
public['control_summary']=control_summary
(root/'summary.json').write_text(json.dumps(public,indent=2),encoding='utf-8')
old=dict(np.load('outputs/mei_audio_repair_20261018/full_context_review/full_42_whole_record_plots.npz'));plots={r:dict(np.load(review/(r+'_42_plots.npz'))) for r in routes}
for pp in plots.values():assert np.array_equal(pp['real_audio'],old['real_audio']) and np.array_equal(pp['ids'],old['ids'])
rows=[('Real microphone',old['real_audio']),('Previous full-context',old['generated_audio'])]+[(r,plots[r]['generated_audio']) for r in routes]
fig,axs=plt.subplots(6,4,figsize=(16,12),layout='constrained')
for ri,(label,arr) in enumerate(rows):
    for col in range(4):
        axs[ri,col].imshow(arr[col],origin='lower',aspect='auto',vmin=-1,vmax=1,cmap='magma',extent=[0,2,0,80],interpolation='nearest');axs[ri,col].set_xlabel('Time(s)')
        if ri==0:axs[ri,col].set_title(str(old['ids'][col]),fontsize=7)
    axs[ri,0].set_ylabel(label+'\nMel bin')
fig.suptitle('Literature-inspired complete-cycle pilot / seed42 / draw0 / fixed4VAL sources\nRaw outputs shared scales, no sharpening or draw selection; not waveform or identity claims')
fig.savefig(root/'comparison.png',dpi=145);plt.close(fig)
fig,axs=plt.subplots(2,3,figsize=(12,6),layout='constrained')
for ax,(label,arr) in zip(axs.flat,rows):
    ax.imshow(arr[0],origin='lower',aspect='auto',vmin=-1,vmax=1,cmap='magma',extent=[0,2,0,80],interpolation='nearest');ax.set_title(label);ax.set_xlabel('Time(s)');ax.set_ylabel('Mel bin')
fig.suptitle('Fixed first mei VAL recording / seed42 / draw0, not best-of-N')
fig.savefig(root/'compact.png',dpi=160);plt.close(fig)
fig,axs=plt.subplots(6,4,figsize=(15,11),layout='constrained')
for row,(label,arr) in enumerate(rows):
    for col in range(4):
        axs[row,col].imshow(arr[col,5:45,40:120],origin='lower',aspect='auto',vmin=-1,vmax=1,cmap='magma',interpolation='nearest',extent=[.4,1.2,5,45]);axs[row,col].set_xlabel('Time(s)')
    axs[row,0].set_ylabel(label)
fig.suptitle('Matched detail crop / fixed .4-1.2s Mel5-45 / raw output shared scale')
fig.savefig(root/'detail_zoom.png',dpi=150);plt.close(fig)
fig,axs=plt.subplots(4,2,figsize=(10,9),layout='constrained')
zero=dict(np.load(review/'data_prediction_42_zero_noise_plots.npz'))
for col in range(2):
    for row,(label,a) in enumerate([('Real microphone',old['real_audio'][col]),('DP-FM fixed draw0',plots['data_prediction']['generated_audio'][col]),('DP-FM zero prior noise',zero['generated_audio'][col]),('Absolute difference (0 to .1)',abs(plots['data_prediction']['generated_audio'][col]-zero['generated_audio'][col]))]):
        axs[row,col].imshow(a,origin='lower',aspect='auto',vmin=0 if row==3 else -1,vmax=.1 if row==3 else 1,cmap='viridis' if row==3 else 'magma',interpolation='nearest',extent=[0,2,0,80]);axs[row,col].set_title(label,fontsize=10);axs[row,col].set_xlabel('Time(s)')
fig.suptitle('Randomness diagnostic / same checkpoint and ACC / no selected random seed')
fig.savefig(root/'noise_control.png',dpi=145);plt.close(fig)
lines=['# 最新研究启发的完整循环：直接回归、速度FM、数据预测FM、残差/噪声解耦','',
'2026-10-19。四设置×两模型种子，共8次最终训练。完整论文适配依据见同目录literature.md；本轮是方法启发的Mel域小试验，不是原论文算法/训练规模/波形系统复现。','',
f"**结论：DP-FM是本轮生成式组里较稳定的适配，但还没有超过原基线，不替换正式模型。** 原基线音频MAE/细节相关为{baseline['audio_mae']:.4f}/{baseline['audio_detail_corr']:.4f}；DP-FM为{summary['data_prediction']['macro']['audio_mae']:.4f}/{summary['data_prediction']['macro']['audio_detail_corr']:.4f}，速度场为{summary['velocity']['macro']['audio_mae']:.4f}/{summary['velocity']['macro']['audio_detail_corr']:.4f}，解耦场为{summary['decoupled']['macro']['audio_mae']:.4f}/{summary['decoupled']['macro']['audio_detail_corr']:.4f}。随机噪声确实产生了更多细粒度变化，但速度/解耦输出的斑点并没有恢复真实谐波；DP输出仍明显平滑。这只是当前小数据、小模型、有限预算实现的结果，不能据此否定论文原方法。",'',
'## 训练与评价边界','',
'沿用clean mei180个独立TRAIN/60个反复探索VAL录制，约2秒251×201 ACC输入、80×201 Mel目标。固定共享前端和归一化，没有逐人滤波/增益、没有改时间戳，未读取测试或认证保留数据，也未引入公开/LLM/伪负数据或外部预训练语音权重。','',
'相同种子前轮完整循环G作为冻结条件估计，另加24通道、不下采样、受flow-time条件调节的小网络。直接组1次前向；流匹配组从条件估计加σ=0.2 Gaussian噪声开始，5步Euler采样。速度组预测目标减初始状态，DP组在插值状态和时刻条件下直接预测目标，再换成更新方向。追加的解耦组用同主干的两个头分别预测真实目标相对基底的残差和初始Gaussian扰动。解耦组是依据FRSE概念设计，未获得其全部方法正文，不称精确复现。','',
'所有模型保留完整ACC→生成Mel→ACC和音频→ACC→音频，F仅看到生成Mel。180轮×4更新，批次8个不同训练录制，固定终点，无VAL选epoch。各组共享数据批次、主干宽度、初始G/F及损失系数；直接回归与5步生成计算量不同，解耦组多一个输出头。与前轮相比，配对Mel L1从20减至2，并增加20倍核心MSE（直接输出/速度/DP/解耦分量），其余循环、梯度、能量和对抗项保留。因此前轮是参考，当前direct才是同预算目标对照，不能把前后差异单独归因于FM。','',
'普通DP-FM只借鉴2026数据预测目标与条件中心先验，不含MeanFlow的JVP训练。该先验噪声作用在归一化Mel，不是论文原复数STFT噪声；没有照搬pink噪声或声码器。推理使用整段未来ACC，仍为离线，不声称实时。解析oracle测试检查1/5步采样符号与终点；初始化零头检查DP/直接输出等于原基底。','',
'评分仍是60录制×前三个旧0.8秒窗口×65内部帧，180共同片段，与前轮相同。每个流匹配模型固定3次噪声抽样，先计算每次指标再平均，图固定draw0；不挑最好结果，不把多张图平均后评分。','',
'## 同口径两种子平均结果','',
'|设置|音频MAE↓|细节相关↑|局部对比相关↑|局部起伏RMS比（真实=1）|局部峰召回↑|误发声率↓|ACC循环MAE↓|',
'|---|---:|---:|---:|---:|---:|---:|---:|',
f"|前轮基线|{baseline['audio_mae']:.4f}|{baseline['audio_detail_corr']:.4f}|{baseline_extra['local_spectral_contrast_corr']:.4f}|—|—|{baseline_extra['activity_false_positive_rate']:.2%}|{baseline['ACC_cycle_mae']:.4f}|"]
for route,row in summary.items():
    m,e,s=row['macro'],row['extra'],row['sharp'];lines.append(f"|{route}|{m['audio_mae']:.4f}|{m['audio_detail_corr']:.4f}|{e['local_spectral_contrast_corr']:.4f}|{s['active_highpass_rms_ratio']:.3f}|{s['near_target_peak_recall']:.2%}|{e['activity_false_positive_rate']:.2%}|{m['ACC_cycle_mae']:.4f}|")
lines+=['',
'峰是Mel图局部极大值（幅差0.025、频率bin容差±1）而非真实基频/谐波准确率；局部RMS接近1只说明起伏强度，不能证明位置正确。活动门限为TRAIN-only能量代理，不是人工VAD。所有VAL已反复探索，记录级bootstrap不包含方法选择及新会话/新用户不确定性。','',
'## 已完成的控制','',
'每个抽样在真实ACC、错录制ACC、TRAIN均值ACC之间保持相同噪声；比较TRAIN均值音频时间模板。另查初始σ=0和同权重1步结果、3抽样逐点标准差、8bit数值Mel量化/3×3平滑后循环ACC误差。这些是输入依赖与数值瓶颈诊断，不是音频信道鲁棒性或身份认证。全部指标见summary.json。','',
'|设置|3抽样逐点标准差|换错录制ACC的音频MAE增加|改用TRAIN均值ACC的MAE增加|σ=0时音频MAE/细节相关|',
'|---|---:|---:|---:|---:|']
for route,c in control_summary.items():
    zero=c.get('zero_noise_macro');ztext=f"{zero['audio_mae']:.4f}/{zero['audio_detail_corr']:.4f}" if zero else '确定性'
    lines.append(f"|{route}|{c['diversity']:.4f}|{c['wrong_record_audio_mae_change']:.4f}|{c['constant_ACC_audio_mae_change']:.4f}|{ztext}|")
lines+=['',
'速度/解耦组去掉初始随机噪声后，细节相关恢复到约0.24，而加噪采样时约0.17，支持“新增纹理含较多噪声残留”的诊断；不是所有细节都来自真实发声。DP组换错录制ACC后误差上升，表明依赖ACC输入，但尚不证明身份信息保留。','',
'无声码器、波形听感、PESQ/STOI或身份头；不把纹理、随机性或较低循环误差解释为身份保持。后续PASE式MIC训练期教师、声学/身份双路约束在本轮仅为文献方向，没有实际实施或声称验证成功。','',
'## 下一阶段优先改变的地方（本轮尚未实施）','',
'优先验证训练期MIC教师提供的、随录制变化的声学结构监督，再让ACC生成器学习这些结构，推理仍只输入ACC。参考PASE将内容线索与低层声学线索分开约束，但两路都只能通过最终生成Mel进入F，不能建立ACC直连F的旁路。只用语义教师可能丢失说话人属性；只固定一个注册者声纹向量又可能把攻击者也映射成注册者，两者都不能作为身份保持证据。','',
'身份信息须用后续隔离的注册和攻击实验验证，特别检查其他用户输入是否也被专属模型改成注册者风格。音频先验或教师的增加是待验证假设，不能由本轮失败直接推定一定有效。耳机论文的400Hz转折只作物理研究线索，不套作VR滤波参数；处理规则继续对所有用户一致。','',
'仅公开方法、协议和汇总。私有信号图、派生信号数组、原始录制、模型权重只保留在本地/服务器。']
(root/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(dict(baseline=baseline,routes=summary,paired=paired),indent=2))
