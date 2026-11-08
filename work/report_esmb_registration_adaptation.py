import json
from pathlib import Path
import numpy as np
root=Path('outputs/esmb_registration_20261109')
assert json.loads((root/'audio_status.json').read_text())['state']=='complete'
assert json.loads((root/'full/status.json').read_text())['state']=='complete'
data=json.loads((root/'full/results_private.json').read_text());control=json.loads((root/'acc/results_private.json').read_text())
for e,ed in control.items():
    for n,nd in ed.items():
        for m,v in nd.items():assert v==data[e][n][m],(e,n,m)
summary={}
for e,ed in data.items():
    summary[e]={}
    for n,nd in ed.items():
        summary[e][n]={}
        for m,v in nd.items():
            summary[e][n][m]={}
            for scope in ['unseen_attacker','all_nine_sources']:
                summary[e][n][m][scope]={k:float(np.mean([z[scope][k] for z in v['targets'].values()])) for k in ['tar','far','hter','auc']}
    audit=json.loads((root/'full/audit_private.json').read_text())[e]
    for n,clients in audit.items():
        for u,entry in clients.items():
            assert len(set(entry['registration']))==int(n) and len(set(entry['negative']))==480 and len(set(entry['query']))==40
            assert not set(entry['registration']+entry['negative'])&set(entry['query'])
per_target_120={e:{m:{u:entry['unseen_attacker'] for u,entry in v['targets'].items()} for m,v in ed['120'].items()} for e,ed in data.items()}
Path('outputs/esmb_registration_adaptation_public_summary_20261109.json').write_text(json.dumps(dict(macro=summary,per_target_120=per_target_120,audio_parity=json.loads((root/'audio_status.json').read_text())),indent=2),encoding='utf-8')
text='''# ESMB：冻结编码器、个人注册适配及同步Audio辅助

日期2026-11-09。本轮已完成12/60/120条注册对照、两种既有冻结BC主干、两个未见目标。代码与协议在完整Audio结果产生前已上传。

## 主要结论

现有方法已迁移完成，但没有解决未见攻击问题。原BC编码器+120条余弦模板宏平均TAR95.00%、FAR28.75%、HTER16.88%；同注册量个人线性头为92.50%、66.25%、36.88%，残差+Audio为92.50%、67.50%、37.50%。Audio辅助未改善当前残差组。

同一原BC编码器下，余弦原型12→60→120条的HTER为23.13%→18.75%→16.88%，表明更多注册数据在本轮对模板有帮助；不能据此认为训练个人线性边界也会改善。现有Audio辅助BC主干beta0.5的120条余弦FAR更高，未优于原BC。

## 数据与技术路线

复用现有10身份、2000组2秒时间窗，每3秒取一窗。channel0真实麦克风，channel1骨传导BC，不是VR硬件ACC。4 Base为0005/0006/0007/0021，4 Cal为0022/0023/0026/0027，未见注册目标0024/0025相互攻击。Base/Cal/目标身份互斥；两个目标不参与通用编码器或Audio映射器训练。

既有BC ResNet18只用4Base前80窗训练、接着20窗选模，192维，seed42。两种主干分别不加Audio蒸馏beta0和旧Audio辅助beta0.5；本轮都冻结。不是本轮重新训练大主干，也不是3seed正式复现。

每目标12/60/120条注册：前120窗划为12个十窗时间块，各取前1/5/10条，嵌套集合。背景负类始终4Base×120=480条，所有方法/注册数量共享最终每人40窗（160..199）。每目标40本人、40未见攻击查询；全部其他来源口径360攻击，含4个已见负类。注册和查询原始波形时间区间不重叠。

五组方法：余弦原型、193参数个人线性头、增加192参数通道残差、残差加ACC自一致性、残差加注册Audio。适配器通道增益±10%，固定200步、沿用上一轮损失和优化器。注册用户只用自己注册正类和允许的Base背景负类适配。

Audio教师是冻结官方VoxCeleb ECAPA192，不是本地九人微调教师。新Audio→BC映射器192→128→192只用4Base前80窗/人，200步，Adam1e-3，配对余弦+.1跨模态身份相似监督。目标/Cal只访问前120窗注册Audio，最终查询Audio不输入教师。局部与归档Base Audio向量8例的最大差约3.3e-7，保证提取链路一致。计算在本地CPU完成，未中止服务器既有大模型训练。

四Cal身份各自个人适配后互相攻击，确定Cal-EER或Cal-FAR≤5%阈值。两规则均预定，最终测试不选阈值、人数、参数、方法；无Audio组完整重跑与前置ACC阶段逐指标精确一致。

## 两位目标的120条结果：Cal-EER阈值

|冻结主干|目标|方法|TAR|FAR|HTER|AUC|
|---|---|---|---:|---:|---:|---:|
'''
for e in data:
    for u in ['0024','0025']:
        for m in ['cosine','linear','residual','residual_self','residual_audio']:
            v=data[e]['120'][m+'_eer']['targets'][u]['unseen_attacker'];text+='|'+ '|'.join([e,u,m]+[f'{v[k]*100:.2f}' for k in ['tar','far','hter','auc']])+'|\n'
text+='\n## 完整宏平均对照\n\n所有指标为百分比，两目标等权平均；仅两位身份，不能外推ESMB总体。HTER不是最终测试EER。\n'
for e in data:
    for rule in ['eer','cal_far05']:
        for scope in ['unseen_attacker','all_nine_sources']:
            text+=f'\n### {e} / {rule} / {scope}\n\n|注册数|方法|TAR|FAR|HTER|AUC|\n|---:|---|---:|---:|---:|---:|\n'
            for n in ['12','60','120']:
                for m in ['cosine','linear','residual','residual_self','residual_audio']:
                    v=summary[e][n][m+'_'+rule][scope];text+='|'+ '|'.join([n,m]+[f'{v[k]*100:.2f}' for k in ['tar','far','hter','auc']])+'|\n'
text+='''
## 解释与边界

原BC beta0的120条余弦原型：0024 TAR97.5%、FAR30%；0025 TAR92.5%、FAR27.5%。个人线性头：0024 97.5%/32.5%，0025 87.5%/100%。0025线性头AUC约0.247，明显低于余弦原型约0.924，表明问题不只是阈值数值；当前个人判别方向对未见0024攻击的排序本身失败。训练背景可分并不保证未知攻击可分。需要更多独立未见身份验证，不能凭两位用户确认唯一机制。

Cal-FAR≤5%不保证测试未知攻击FAR≤5%。原BC120条余弦：0024 TAR67.5%/FAR7.5%，0025 65%/5%；相对EER工作点降低误接受，但拒绝更多本人。

ESMB缺少已核实逐句文本，当前只能证明时间片段隔离，不能声称严格同词、跨词或内容无关。单连续录制、相邻时间窗相关、两个目标此前已被探索，均限制统计和外推。该实验验证现有方法在骨传导模态上的探索性迁移，不能直接证明VR加速度上的结论。

旧ESMB100条+RBF较好的结果里，多位注册者参与过共享编码器训练；本轮目标从未参与通用模型训练，认证器和阈值也不同，不能直接相减归因于方法变差。全部两目标和攻击窗口保留，不挑容易来源、不删误接受样本。

私有注册参数、逐窗口清单及Audio信号不发布。公开代码、协议与聚合结果。
'''
for e in data:
    v=summary[e]['120']['residual_audio_eer']['unseen_attacker'];b=summary[e]['120']['residual_eer']['unseen_attacker']
    text+=f"\n{e} Audio辅助相对残差：HTER {100*b['hter']:.2f}%→{100*v['hter']:.2f}%，TAR {100*b['tar']:.2f}%→{100*v['tar']:.2f}%，FAR {100*b['far']:.2f}%→{100*v['far']:.2f}%；需并列判断，不以单项变化认定收益。\n"
Path('docs/experiments/2026-11-09-esmb-registration-adaptation-results.md').write_text(text,encoding='utf-8')
print('FULL REPORT AND PRIVACY-SAFE SUMMARY WRITTEN')
for e in data:
    print(e)
    for m in ['cosine','linear','residual','residual_audio']:
        print(m,{u:{k:round(v['unseen_attacker'][k]*100,2) for k in ['tar','far','hter','auc']} for u,v in data[e]['120'][m+'_eer']['targets'].items()})