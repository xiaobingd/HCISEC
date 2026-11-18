"""Local comparison of adding input-ACC features to CycleGAN branches."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent.parent/'outputs/acc_feature_fusion_20261019';s=json.loads((root/'summary.json').read_text());routes=['raw_ACC','fusion','cycle_ACC','fusion_plus_ACC','cycle_plus_ACC'];names=dict(raw_ACC='输入ACC',fusion='CycleGAN多源',cycle_ACC='循环ACC',fusion_plus_ACC='输入ACC＋CycleGAN多源',cycle_plus_ACC='输入ACC＋循环ACC')
lines=['# 将输入ACC作为CycleGAN认证的融合特征','', '本轮新增两条认证路线，每条两次head初始化：输入ACC502维均值/标准差与CycleGAN多源196维拼接为698维；输入ACC502维与循环ACC502维拼接为1004维。拼接后使用原trainb核心512维融合头及128维投影头，逐身份SupCon训练，512维向量用于认证。输入ACC指同一预处理后的251通道时频/能量输入，不是未处理波形，也未新增时序编码器。', '', '## 控制条件','', '- CycleGAN与公共信号前端冻结，直接使用此前同一录制的缓存特征；没有新的逐用户滤波或归一化。','- 与上一轮identity_episode策略相同：最多1500次更新、25epochs、每轮60个32样本批次；相同优化器/学习率/温度/采样规则、两个种子42/43，五个身份标签。输入维度变大使首层参数增加，因此不是容量完全配平的特征因果实验。','- 使用原先20次验证集内部5模拟注册/5本人查询episode选模。全部4个checkpoint选定后才读取实际注册和阈值校准数据；全部模型/模板/阈值锁定后才读取最终特征。','- 最终同一30条本人、477条攻击（ke118、fanweibo120、fanweiqi120、fanweiqisis119）。三个单分支对照复用上一轮固定结果，采样规则相同不代表不同维度网络具有逐步相同梯度或随机序列。', '', '## 两个head种子的平均结果','', '| 特征 | 本人接受率↑ | FAR↓ | HTER↓ | AUC↑ |','|---|---:|---:|---:|---:|']
for route in routes:
    a=s['aggregate'][route];lines.append('|'+names[route]+'|'+'|'.join(f'{a[k]:.2%}' for k in ['TAR','FAR','HTER'])+f"|{a['AUC']:.4f}|")
lines+=['','## 逐攻击来源接受率','', '| 特征 | ke | fanweibo | fanweiqi | fanweiqisis |','|---|---:|---:|---:|---:|']
users=['chengkeyi','fanweibo','fanweiqi','fanweiqisis']
for route in routes:
    a=s['aggregate'][route];lines.append('|'+names[route]+'|'+'|'.join(f"{a['sources'][u]['FAR']:.2%}" for u in users)+'|')
lines+=['','## 两个新方案每次训练的实际计数','', '| 特征 | seed | 本人接受/30 | 攻击接受/477 | FAR | AUC | 选中更新数 |','|---|---:|---:|---:|---:|---:|---:|']
for r in s['results']:
    fp=sum(v['accepted'] for v in r['attack_sources'].values());lines.append('|'+names[r['route']]+'|'+str(r['seed'])+'|'+str(r['genuine']['accepted'])+'|'+str(fp)+f"|{r['FAR']:.2%}|{r['AUC']:.4f}|{r['best_epoch']*60}|")
lines+=['','## 与输入ACC对照的差异','']
base=s['aggregate']['raw_ACC']
for route in routes[-2:]:
    a=s['aggregate'][route];lines.append(f"- {names[route]}：本人接受率{base['TAR']:.2%}→{a['TAR']:.2%}，FAR{base['FAR']:.2%}→{a['FAR']:.2%}，HTER{base['HTER']:.2%}→{a['HTER']:.2%}，AUC{base['AUC']:.4f}→{a['AUC']:.4f}。")
lines+=['','需同时衡量误接受和本人误拒，不能根据FAR一个数字判断是否改善；两个种子与一个注册者不足以证明稳定收益。当前留出集已被历次诊断查看，所有攻击身份参与训练，属于探索性已见身份的录制留出，不能当成独立盲测、未见身份开集或跨日效果。没有根据最终结果重新选阈值或删除样本。','', '脚本 train_acc_feature_fusion_20261019.py 包含训练、锁定、验证与模型备份；report_acc_feature_fusion_20261019.py 生成报告。私有模型与锁文件保存在服务器 /root/imu_cyclegan_trial_20261016/acc_feature_fusion_20261019，备份 /autodl-fs/data/acc_feature_fusion_20261019。详细结果仅本地/私有保存。']
(root/'README.md').write_text('\n'.join(lines),encoding='utf-8')
fig,axs=plt.subplots(1,2,figsize=(11,4.5));x=np.arange(len(routes));short=['Input ACC','CycleGAN fusion','Cycle ACC','Input + GAN','Input + cycle']
for ax,k,title in [(axs[0],'TAR','Genuine acceptance TAR (%)'),(axs[1],'FAR','Attack acceptance FAR (%)')]:
    vals=[s['aggregate'][r][k]*100 for r in routes];bars=ax.bar(x,vals,color=['#808080','#5e81ac','#88a684','#d08b48','#a778a8']);ax.bar_label(bars,fmt='%.2f',fontsize=9,padding=3);ax.set_xticks(x,short,rotation=18);ax.set_ylim(0,110);ax.set_title(title)
fig.suptitle('Same30genuine/477attacks; identity contrastive +episodic selection');fig.tight_layout();fig.savefig(root/'feature_fusion_comparison.png',dpi=160);plt.close(fig);print(json.dumps(s['aggregate'],indent=2))
