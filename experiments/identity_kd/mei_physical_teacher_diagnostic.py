"""Training-only leave-word-out observability and identity diagnostics."""
import os
for k in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[k]='1'
import json
from pathlib import Path
import numpy as np
from scipy import signal
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
def norm(x):return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-8)
def grid(v,n):return np.interp(np.linspace(0,len(v)-1,n),np.arange(len(v)),v)
d=np.load('outputs/mei_physical_train_private.npz');user=d['user'];word=d['word'];b=d['spectrum'];features={k:[] for k in ['periodicity','low_spectrum','envelope']};quality=[]
for wave in d['wave']:
 w=wave.astype(float);w-=w.mean();frames=np.lib.stride_tricks.sliding_window_view(w,640)[::160];frames=frames-frames.mean(1,keepdims=True);eng=np.mean(frames**2,axis=1);ac=signal.fftconvolve(frames,frames[:,::-1],axes=1,mode='full')[:,639:];ac/=np.maximum(ac[:,:1],1e-12);lag=40+np.argmax(ac[:,40:201],axis=1);strength=ac[np.arange(len(ac)),lag];voiced=(strength>.5)&(eng>max(eng.max()*1e-4,1e-12));f0=16000/lag;vv=f0[voiced];features['periodicity'].append([np.median(vv) if len(vv) else 0,np.subtract(*np.percentile(vv,[75,25])) if len(vv) else 0,voiced.mean(),strength.mean()]);quality.append(float(voiced.mean()))
 f,_,z=signal.stft(w,16000,nperseg=4096,noverlap=3072,nfft=16384,boundary=None,padded=False);lp=np.log(abs(z)**2+1e-20);low=np.stack([np.interp(np.geomspace(10,300,40),f,col) for col in lp.T],1);m=low.mean(1);features['low_spectrum'].append(np.r_[m-m.mean(),low.std(1)]);env=grid(np.log(eng+1e-20),28);features['envelope'].append((env-env.mean())/max(env.std(),1e-6))
x=np.c_[b.mean(-1).reshape(len(b),-1),b.std(-1).reshape(len(b),-1),b.mean(2).reshape(len(b),-1)];rows=[]
for family,target in features.items():
 target=np.array(target);err=np.zeros(len(x));den=np.zeros(len(x));shuffle_err=np.zeros(len(x));scores=[];oracle=[];labels=[]
 for held in sorted(set(word)):
  tr=word!=held;te=~tr;sx=StandardScaler().fit(x[tr]);st=StandardScaler().fit(target[tr]);xx=sx.transform(x[tr]);xt=sx.transform(x[te]);tt=st.transform(target[tr]);truth=st.transform(target[te]);model=Ridge(alpha=100).fit(xx,tt);pred=model.predict(xt);sh=tt.copy();rng=np.random.RandomState(42+int(held))
  for u in range(1,10):jj=np.flatnonzero(user[tr]==u);sh[jj]=tt[rng.permutation(jj)]
  shuffled=Ridge(alpha=100).fit(xx,sh).predict(xt);err[te]=np.mean((truth-pred)**2,1);den[te]=np.mean(truth**2,1);shuffle_err[te]=np.mean((truth-shuffled)**2,1);centers=norm(np.stack([tt[user[tr]==u].mean(0) for u in range(1,10)]));scores.extend(norm(pred)@centers.T);oracle.extend(norm(truth)@centers.T);labels.extend(user[te])
 scores=np.array(scores);oracle=np.array(oracle);labels=np.array(labels)
 for u in range(1,10):
  j=user==u;r2=float(1-err[j].sum()/max(den[j].sum(),1e-9));sr2=float(1-shuffle_err[j].sum()/max(den[j].sum(),1e-9));auc=float(roc_auc_score(labels==u,oracle[:,u-1]));pa=float(roc_auc_score(labels==u,scores[:,u-1]));rows.append(dict(user=f'U{u}',family=family,records=int(j.sum()),crossword_prediction_R2=r2,same_user_shuffle_R2=sr2,audio_identity_auc=auc,predicted_identity_auc=pa,candidate=r2>0 and r2>sr2 and auc>=.7 and pa>=.7))
out=Path('outputs/mei_physical_teacher_diagnostic_20261114');out.mkdir(exist_ok=True);(out/'per_user_public.json').write_text(json.dumps(rows,indent=2));np.savez_compressed(out/'features_private.npz',**{k:np.array(v) for k,v in features.items()});lines=['# Mei9逐用户：Audio物理教师候选诊断','','仅使用合法训练/注册144条，每人16条，8个名义词。逐词留出：训练标准化、Ridge100固定、Audio身份原型均只用其余词。不访问最终校准/查询。与同人训练Audio打乱比较，排除仅靠身份均值的部分预测。','','周期性采用40ms自相关80–400Hz候选、峰值>.5与能量门控；不是人工核验F0。低频谱为候选10–300Hz，匹配现有ACC前端范围，但未做原始波形相干性证明。包络为相对时长归一化28点log能量，不是逐点同步验证。','','跨词Audio原型AUC诊断身份稳定/区分；R2相对训练均值判断预测。候选规则预设：R2>0且超过同人打乱，真实和预测身份AUC均≥.7。阈值是探索门槛，不是显著性检验。一个样本/词和少量重复不足以强证生理稳定性；无跨天或重新佩戴。','','|用户|目标|预测R2|同人打乱R2|Audio身份AUC|预测身份AUC|候选|','|---|---|---:|---:|---:|---:|---|']
for r in rows:lines.append(f'|{r["user"]}|{r["family"]}|{r["crossword_prediction_R2"]:.3f}|{r["same_user_shuffle_R2"]:.3f}|{r["audio_identity_auc"]:.3f}|{r["predicted_identity_auc"]:.3f}|{r["candidate"]}|')
lines+=['','## 逐人结论','','|用户|通过初筛的教师目标|暂未通过的主要原因|','|---|---|---|','|U1|10–300Hz候选谱形|周期性和包络跨词预测未超过训练均值|','|U2|低频谱形、包络|周期性虽能预测，但未超过同人打乱对照|','|U3|暂无|低频谱不超过同人打乱；包络预测后的身份区分未达初筛线|','|U4|包络|周期性自身身份AUC偏低，低频谱预测R2为负|','|U5|暂无|三类跨词预测R2均为负|','|U6|周期性|低频谱和包络预测R2均为负|','|U7|暂无|周期性和低频谱预测R2为负，包络身份区分偏弱|','|U8|暂无|三类跨词预测R2均为负|','|U9|低频谱形|周期性、包络预测R2为负|','','本诊断不等于认证增益。候选需再经同预算辅助头训练与无Audio基线对照；没有候选不证明更强预测器无效。本轮输入为tracking速度频谱，不是硬件原始ACC。Audio缓存为已有16kHz左声道训练片段，词标签未逐条转写核验。']
(out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf8');print(json.dumps(rows,indent=2))
