"""Review fixed-threshold, record-disjoint two-user authentication results."""
import os,json,hashlib
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path('outputs/mei_baseline_auth_20261019');review=root/'review';summary=json.loads((review/'summary.json').read_text());scores=json.loads((review/'private_scores.json').read_text());routes=['fusion','raw_ACC','cycle_ACC'];colors=['#1f77b4','#d95f02','#2ca02c']
summary['source_hashes']={name:hashlib.sha256((Path('work')/name).read_bytes()).hexdigest() for name in ['retrain_mei_auth_baseline.py','prepare_mei_auth_data.py','original_trainb_core.py','train_mei_baseline_auth.py','evaluate_mei_baseline_auth.py','pack_mei_auth_delivery.py','report_mei_baseline_auth.py']}
(root/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
fig,axs=plt.subplots(1,3,figsize=(14,4),layout='constrained')
for ax,route in zip(axs,routes):
    row=next(r for r in summary['authentication'] if r['route']==route and r['head_seed']==42);d=scores[route+'_42'];s=np.array(d['scores']);y=np.array(d['labels']);bins=np.linspace(min(s.min(),row['threshold']),max(s.max(),row['threshold'])+1e-6,16)
    ax.hist(s[y==0],bins=bins,density=True,alpha=.55,label='mei genuine (30)',color='#1f77b4');ax.hist(s[y==1],bins=bins,density=True,alpha=.55,label='ke impostor (118)',color='#d95f02');ax.axvline(row['threshold'],color='black',ls='--',label='Locked calibration threshold')
    ax.set_title(route+' / seed42\nTAR %.1f%%, FAR %.2f%%'%(100*row['genuine']['TAR'],100*row['FAR']));ax.set_xlabel('Cosine similarity to enrollment');ax.set_ylabel('Density');ax.legend(fontsize=7)
fig.suptitle('Final score distributions / thresholds fixed before final export\nSingle-session, two-user exploratory test; impostor identity seen in head training')
fig.savefig(root/'authentication_scores.png',dpi=160);plt.close(fig)
fig,ax=plt.subplots(figsize=(6,5),layout='constrained')
for route,color in zip(routes,colors):
    for seed in [42,43]:
        d=scores[route+'_'+str(seed)]['ROC'];row=next(r for r in summary['authentication'] if r['route']==route and r['head_seed']==seed)
        ax.plot(d['FAR'],d['TAR'],color=color,ls='-' if seed==42 else '--',label='%s s%d AUC %.3f'%(route,seed,row['AUC']));ax.scatter([row['FAR']],[row['genuine']['TAR']],color=color,s=24)
ax.plot([0,1],[0,1],color='gray',ls=':');ax.set_xlim(0,1);ax.set_ylim(0,1.02);ax.set_xlabel('False acceptance rate');ax.set_ylabel('Genuine acceptance rate');ax.set_title('Final ROC; dots=locked operating points\nSame30 genuine/118 impostor records per head');ax.legend(fontsize=8);fig.savefig(root/'authentication_roc.png',dpi=170);plt.close(fig)
fig,ax=plt.subplots(figsize=(7,4),layout='constrained');xx=np.arange(3);means=np.array([[summary['aggregate'][r]['TAR'],summary['aggregate'][r]['FAR']] for r in routes])*100
ax.bar(xx-.18,means[:,0],width=.35,label='Genuine accepted %',color='#1f77b4');ax.bar(xx+.18,means[:,1],width=.35,label='Impostor accepted %',color='#d95f02');ax.set_xticks(xx);ax.set_xticklabels(routes);ax.set_ylim(0,105);ax.set_ylabel('%');ax.legend();ax.set_title('Mean of two head initializations / one trained CycleGAN\nSame final records, not double the sample count');fig.savefig(root/'authentication_comparison.png',dpi=160);plt.close(fig)
p=dict(np.load(review/'generation_plots.npz'));n=len(p['ids']);fig,axs=plt.subplots(4,n,figsize=(3.7*n,9),squeeze=False,layout='constrained')
for col in range(n):
    arr=[p['real_audio'][col],p['generated_audio'][col],p['real_ACC'][col,:195].reshape(3,65,201).mean(0),p['cycle_ACC'][col,:195].reshape(3,65,201).mean(0)]
    for row,a in enumerate(arr):axs[row,col].imshow(a,origin='lower',aspect='auto',vmin=-1,vmax=1,cmap='magma',extent=[0,2,0,80 if row<2 else 500],interpolation='nearest');axs[row,col].set_xlabel('Time(s)')
    axs[0,col].set_title(str(p['tasks'][col])+' GENcheck '+str(col),fontsize=9)
for row,label in enumerate(['Real microphone','Fresh baseline generated audio','Real ACC XYZ mean','Cycle ACC XYZ mean']):axs[row,0].set_ylabel(label)
fig.suptitle('Generation-check sources excluded from all authentication groups\nFirst2 per task, raw output and shared scales, freshly trained baseline');fig.savefig(root/'generation_check.png',dpi=150);plt.close(fig)
labels={'fusion':'CycleGAN多源融合','raw_ACC':'原始ACC均值/标准差','cycle_ACC':'循环ACC均值/标准差'}
lines=['# 原基线重训与身份认证头：mei注册、ke攻击','',
'2026-10-19。重新从随机初始化训练最近报告中2秒完整循环基线，再训练基于用户原trainb监督对比学习的认证头。不是读取旧权重后直接报分，也不是旧80×80 CycleGAN未经修改的整套复现。','',
'## 主要结果','',
'多源头攻击误接受较少，但拒绝本人较多；原始ACC对照的平均整体误差更低，本轮不能证明CycleGAN提高了认证性能。以下是同一最终30本人/118攻击样本上的两个头种子平均，不能把重复评分当成60/236个独立样本。','',
'|特征/认证头|本人接受率TAR↑|本人拒绝率FRR↓|ke攻击接受率FAR↓|HTER↓|AUC↑|',
'|---|---:|---:|---:|---:|---:|']
for route in routes:
    a=summary['aggregate'][route];lines.append(f"|{labels[route]}|{a['TAR']:.2%}|{a['FRR']:.2%}|{a['FAR']:.2%}|{a['HTER']:.2%}|{a['AUC']:.4f}|")
lines+=['','多源头逐次结果：','', '|头种子|选中epoch|锁定阈值|本人接受/总数|攻击接受/总数|AUC|','|---|---:|---:|---:|---:|---:|']
for r in summary['authentication']:
    if r['route']=='fusion':lines.append(f"|{r['head_seed']}|{r['best_epoch']}|{r['threshold']:.6f}|{r['genuine']['accepted']}/30|{r['attacks']['chengkeyi']['accepted']}/118|{r['AUC']:.4f}|")
lines+=['',
'校准集只有10条本人和30条ke。多源两种子在校准上均为0错误，最终本人却拒绝6–8/30，说明小校准集上的完美结果不能代表最终性能。没有根据最终分数重新选择阈值或删除误接受/拒绝样本。各头是在相同的校准程序下运行，测试TAR/FAR不是强制匹配的操作点。','',
'原始ACC对照和多源头AUC相近，但锁定操作点的取舍不同；不能仅凭多源FAR更低宣称整体更好。循环ACC对照本轮也没有超过原始ACC。不同特征的维数与聚合方式不同，消融是描述性比较，不是完整因果证明。','',
'## 数据隔离与人数限制','',
'来源为同一Data0723原始配对集合：mei去重后300条、ke298条。两个用户都使用相同的时间戳排序/去重、1kHz XYZ插值、整段去DC、0–500Hz分轴幅度谱、56维多尺度能量、冻结共享量程；无逐人增益/滤波。认证只读ACC，不用麦克风输入或音频活动标签。固定2秒处理与原CG输入在5个训练源上逐元素核验；最大差见summary.json。','',
'|角色|mei|ke|','|---|---:|---:|']
names={'gan_train':'CycleGAN梯度训练','gan_check':'仅生成效果检查','head_train':'认证头梯度训练','head_validation':'认证头选epoch','enrollment':'注册模板','calibration':'阈值校准','final_test':'最终评分'}
for group,counts in summary['counts'].items():lines.append(f"|{names[group]}|{counts['wangjingmei']}|{counts['chengkeyi']}|")
lines+=['',
'每条2秒原始录制作为单位，不把重叠窗口拆进不同认证集合。CG180保持前轮训练来源完全一致；从原60VAL固定抽30作生成检查，其余可用90mei预先分给认证。ke298全部预先划入认证训练/验证/校准/最终，按任务分层，没有按攻击效果删除来源或样本。原始source key、raw pair/ACC文件哈希及所有实际预处理模型输入均跨角色无重复。归一化中的mei20源均在CG梯度训练内，量程继续冻结。','',
'这只是两用户、同次采集的探索实验。ke身份在训练认证头时已见过，最终只隔离了录制样本，不是未见攻击者身份的开集验证；也不能证明跨日、重戴、所有用户或面部传导的生理身份机制。mei旧VAL/test历史上已被探索，不是全新独立确认集。','',
'## 代码来自哪里、哪些地方做了适配','',
'CycleGAN重训直接调用之前基线的fit：360×4次预热＋360×4次完整循环，再将相同G权重拓展至201帧，按之前full_context原损失继续120×4次完整循环。初始化种子42，不复用旧checkpoint；固定端点，不通过新生成检查/认证选择G。这里的轮为固定4更新，不是遍历数据一遍。','',
'认证核心来自用户本地 E:/Edge下载/train_core/对比学习特征处理/modelb.py，原文件SHA256为05c7c8e40e427e5529346bd360940ef35a7eebd028c421b3fc119d37e1f6f153。FeatureAggregator、ProjectionHead、SupConLoss三个类通过AST逐字提取，并与源文件逐字核验。融合层结构与原代码相同：输入→512→512，投影512→256→128；训练用SupCon投影，推理用L2归一化的512融合向量。Adam lr1e-3/WD1e-4、100epoch余弦调度、温度0.07、梯度裁剪1、验证intra-inter差选epoch均沿用原路径。','',
'接口适配明确如下：当前小G浅层32通道、真实瓶颈64通道，故取[:7]/[:13]并聚合为32＋128；循环差异通过现有251→80投影接口进入浅层（分别投影后相减以消掉偏置），再加4个原始差异统计，融合输入共196维。旧版硬编码128/256通道及切片不能原样套用。','',
'生成器冻结并缓存特征。两类平衡到每类240份；重复不是新录制。旧80×80图像增强不适用于251行中混合的频谱/能量特征，本轮没有应用旧频率平移/增益增强，故不称trainb整套逐字复现。原始/循环ACC对照各用251通道的时间均值和标准差共502维，后接相同融合头与训练规则。','',
'## 注册和测试过程','',
'注册模板是10条mei归一化512向量的均值再归一化。仅用另10条mei＋30条ke校准余弦阈值，采用与既有评估一致的FAR/FRR平衡规则。六个头的checkpoint、模板和阈值先写入heads_locked.json并保存SHA256，此后才导出最终30mei/118ke并评分；评估时再次核对权重哈希。AUC是最终分数的描述性排序指标，不用于改阈值。','',
'置信区间为每个头在30/118条样本上的Wilson区间，详见summary.json；没有把两种子当作翻倍样本。模型权重、注册模板、源级划分及哈希在私有delivery.zip/models中，个人信号/逐条分数/图像在review中；公开仓库只放源码与汇总。','',
'## 生成检查口径','',
'另30个CG检查源不参与认证训练/注册/评分。该指标采用30条完整2秒201帧，和之前60源×180个65帧片段不同，不能直接同0.3071横比。原始图仍有模糊谐波，认证指标不等于语音重建成功。', '',
json.dumps(summary['generation_check']['macro'],ensure_ascii=False), '',
'最终结论限于：当前特征允许在这两个用户之间训练出有区分能力的认证头；多源头低误接受伴随较高本人拒绝，循环转换对身份认证的额外价值尚未建立。']
(root/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print(json.dumps(summary['aggregate'],indent=2))
