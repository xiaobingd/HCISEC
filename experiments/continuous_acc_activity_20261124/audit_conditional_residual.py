"""Train-cache-only conditional Audio relation prediction; no neural training."""
import json, itertools, hashlib
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/vr261125_conditional_identity_residual'
BASE=ROOT/'outputs/vr261119_10x40_raw_acc'
def unit(x): return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)
def fit_predict(x,y,q):
    sc=StandardScaler().fit(x)
    return Ridge(alpha=10).fit(sc.transform(x),y).predict(sc.transform(q))
def main():
    OUT.mkdir(exist_ok=True)
    p=BASE/'conditional_correspondence/raw_common_dynamics/local_power_private.npz'
    c=np.load(p);a=np.load(BASE/'conditional_correspondence/audio_conditioned_verifier/fit_ecapa_private.npz')
    d=np.load(BASE/'phrase_auth/frontend_private.npz')
    assert np.array_equal(c['indices'],a['indices']) and np.array_equal(c['sample_id'],a['sample_id'])
    for k in ['sample_id','user','word','repeat']:assert np.array_equal(c[k],d[k][c['indices']])
    u,w,r=[c[k] for k in ['user','word','repeat']];words=sorted(set(map(int,w)));users=np.arange(1,11)
    assert len(set(c['sample_id']))==1280 and len(words)==16
    assert all(np.sum((u==v)&(w==z))==8 for v in users for z in words)
    # No selected activity mask; complete-window ACC shape and energy statistics.
    x=c['acc'].astype(float); x=np.c_[x.mean(1),x.std(1),x[:,:73].mean(1),x[:,73:].mean(1)]
    # Pair cosine is insensitive to common log-power offset only after train-word scaling.
    trainwords=words[:12]; valwords=words[12:]
    act=c['audio_activity'].astype(float)
    assert np.isfinite(act).all()
    act=np.c_[act.mean(1),act.std(1)]
    pairs=list(itertools.combinations(range(10),2))
    protocol=dict(training_words=trainwords,validation_words=valwords,cache_only=True,
      question='Does same-word Audio relation improve prediction of cross-word ACC confusion beyond same-word ACC and activity covariates?',
      target='mean bidirectional ACC cross-word cosine of distinct-user prototypes; next cyclic word within each partition; even repeats support, odd repeats query',
      features='baseline same-word ACC pair cosine, per-endpoint Audio energy-proxy mean/std sum and absolute difference; added same-word ECAPA pair cosine',
      controls='same word for all identity pairs; word means removed within word over all pairs; activity regressors; no hard VAD; ridge10 fixed',
      folds='leave BOTH endpoint identities out of regression training (28 remaining pairs), validation four disjoint cache words; no identity onehot',
      permutations=199,seed=20261125,primary='validation macro endpoint-held-pair MSE reduction; positive is better; identity permutation refits every regression',
      hard='top-quartile same-word ACC similarity cutoff determined only training words per endpoint fold; no selection by validation target',
      scope='raw ACC statistics not learned E0; no encoder retraining or neural authentication claim; shared cohort and historic words, exploratory; energy proxy does not fully control activity/content',
      no_final_test_Audio=True,no_authentication_gain_claim=True)
    (OUT/'protocol_public.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
    def proto(z,ww,parity):return unit(np.stack([z[(u==us)&(w==ww)&(r%2==parity)].mean(0) for us in users]))
    # Scaling excludes held endpoint users AND held words separately in each fold.
    folds=[]
    for held,(i,j) in enumerate(pairs):
      baseusers=users[~np.isin(np.arange(10),[i,j])]
      mask=np.isin(w,trainwords)&np.isin(u,baseusers)
      sc=StandardScaler().fit(x[mask]);xx=sc.transform(x)
      features=[];targets=[];audios=[];pairids=[];wordids=[]
      for group in [trainwords,valwords]:
       for wi,ww in enumerate(group):
        nxt=group[(wi+1)%len(group)]
        xs=proto(xx,ww,0);xq=proto(xx,nxt,1)
        ap=proto(a['embedding'],ww,0)
        ac=np.stack([act[(u==us)&(w==ww)&(r%2==0)].mean(0) for us in users])
        ff=[];tt=[]
        for pi,(v,t) in enumerate(pairs):
          ff.append(np.r_[xs[v]@xs[t],ac[v]+ac[t],abs(ac[v]-ac[t])])
          tt.append(.5*(xs[v]@xq[t]+xs[t]@xq[v]))
          pairids.append(pi);wordids.append(ww)
        ff=np.array(ff);tt=np.array(tt)
        # Remove per-word average effects, without assigning identity labels.
        ref=np.array([i not in pairs[t] and j not in pairs[t] for t in range(len(pairs))])
        features.extend(ff-ff[ref].mean(0));targets.extend(tt-tt[ref].mean());audios.append(ap)
      f=np.array(features);y=np.array(targets);pid=np.array(pairids);wid=np.array(wordids)
      train=np.isin(wid,trainwords)&np.array([i not in pairs[t] and j not in pairs[t] for t in pid])
      test=np.isin(wid,valwords)&(pid==held)
      pb=fit_predict(f[train],y[train],f[test])
      hard=f[test,0]>=np.quantile(f[train,0],.75)
      folds.append((held,i,j,f,y,train,test,pb,np.array(audios),hard))
    rng=np.random.default_rng(20261125)
    permutations=[np.arange(10)]+[rng.permutation(10) for _ in range(199)]
    results=[];correct_rows=[]
    for it,perm in enumerate(permutations):
      ys=[];bs=[];ps=[];hs=[]
      for held,i,j,f,y,tr,te,pb,ap,hard in folds:
        z=np.array([[(v[perm[s]]@v[perm[t]]) for s,t in pairs] for v in ap])
        ref=np.array([i not in pp and j not in pp for pp in pairs]);z-=z[:,ref].mean(1,keepdims=True);z=z.ravel()
        fq=np.c_[f,z];pred=fit_predict(fq[tr],y[tr],fq[te])
        ys.extend(y[te]);bs.extend(pb);ps.extend(pred);hs.extend(hard)
        if it==0:
          correct_rows.append(dict(user_pair=[i+1,j+1],baseline_mse=float(np.mean((y[te]-pb)**2)),audio_mse=float(np.mean((y[te]-pred)**2)),validation_tasks=int(te.sum()),hard_tasks=int(hard.sum())))
      ys,bs,ps,hs=map(np.array,[ys,bs,ps,hs]);delta=(ys-bs)**2-(ys-ps)**2
      results.append(dict(mse_reduction=float(delta.mean()),hard_mse_reduction=float(delta[hs].mean()) if hs.any() else None,
        baseline_mse=float(np.mean((ys-bs)**2)),audio_mse=float(np.mean((ys-ps)**2)),baseline_spearman=float(spearmanr(ys,bs).statistic),audio_spearman=float(spearmanr(ys,ps).statistic)))
    null=np.array([v['mse_reduction'] for v in results[1:]]);real=results[0]
    summary=dict(correct=real,identity_permutation_p=float((1+np.sum(null>=real['mse_reduction']))/200),null_mean=float(null.mean()),null95=np.quantile(null,[.025,.975]).tolist(),
      cache_alignment='PASS',pairs=45,validation_pair_tasks=180,independent_users=10,final_Audio_read=False,
      limitation='Pair tasks share identities, not 180 independent subjects; permutation acts on ten user labels consistently across all words. Audio and ACC have different temporal window origins from newest continuous experiment.')
    summary['hard_identity_permutation_p']=float((1+sum(v['hard_mse_reduction']>=real['hard_mse_reduction'] for v in results[1:]))/200)
    per_user=[]
    for us in users:
      rows=[v for v in correct_rows if int(us) in v['user_pair']]
      per_user.append(dict(user=int(us),baseline_mse=float(np.mean([v['baseline_mse'] for v in rows])),audio_mse=float(np.mean([v['audio_mse'] for v in rows])),incident_pairs=len(rows)))
    (OUT/'per_user_public.json').write_text(json.dumps(per_user,indent=2),encoding='utf-8')
    (OUT/'summary_public.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'per_pair_public.json').write_text(json.dumps(correct_rows,indent=2),encoding='utf-8')
    text=f'''# 条件化 Audio 身份残差审计

本轮只读取已有16个训练词缓存，未读取最终测试 Audio，也未训练新神经网络。先控制同词内容与能量活动代理，检验 Audio 是否能增量预测 ACC 跨词身份混淆。

## 实际结果

```json
{json.dumps(summary,ensure_ascii=False,indent=2)}
```

MSE reduction 正值代表加入 Audio 更好。身份置换将整个用户的 Audio 一致换给另一个用户，199次都重新拟合预测器；以用户标签为置换单位，不能把重复录制/身份对当独立受试者。

本轮整体预测未改善，困难子集微小正值亦未超过身份置换。该观察不支持继续用完整 ECAPA 同词相似度作为本协议的 ACC 跨词身份监督。它不排除局部周期性、谐波形状或发声能量轨迹含有其他可迁移信息；这些应作为下一轮独立、预先固定的教师候选，不得用当前验证词挑选后再声称验证成功。

## 方法与解释边界

12词建模、4词验证，所有45个身份对依次留出，训练预测器排除这两位用户。每词所有用户使用相同词；偶数遍构建注册原型、奇数遍构建不同词查询原型。基线包含 ACC 同词关系与两用户活动代理统计，Audio 组仅增加同词 Audio 身份相似度。固定标准化＋Ridge(alpha=10)，不调参。难样本由训练词 ACC 相似度四分位数定义，不使用验证目标筛选。所有词级关系先减去该词平均值。

这是原始 ACC 统计特征的条件预测检验，不是神经 E0 的认证实验。活动代理不等于人工 VAD，均值控制不等于完全剔除内容或活动；不能把结果称为纯生理身份残差。历史16训练词与最新连续实验词划分不同，仅在自己的缓存内划分，不能借用其结果宣布最新协议泛化。缓存编码器是否记住身份也未被这个回归对照完全排除。

下一步只在真实对应明显超过身份置换时，才考虑将有支持的关系送入新词/新身份验证；若未超过，先检查原始时频发声特征而非扩大 ECAPA 蒸馏。本轮相关/预测结果均不能直接说明认证 EER 改善。

匿名逐身份对误差见 per_pair_public.json，逐用户关联身份对误差见 per_user_public.json（同一身份对会计入两位用户，不能作为独立样本）。本轮难子集由训练词决定，其结果亦须超过身份置换，不能只挑困难子集的正值作为成功依据。每词中心化均值只用八位基础身份之间的28个关系，不包含留出身份对；验证词背景用于描述性去词均值，不参与回归拟合。后续还需要新词与新身份确认。

不发布原始音频、embedding、身份映射或逐样本分数。
'''
    (OUT/'report.md').write_text(text,encoding='utf-8');print(json.dumps(summary,ensure_ascii=False))
if __name__=='__main__':
    with threadpool_limits(limits=1): main()
