"""Fixed local Audio descriptors, endpoint/word holdouts, identity permutation."""
import json,itertools
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from audit_conditional_residual import unit

ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'outputs/vr261119_10x40_raw_acc'
OUT=ROOT/'outputs/vr261125_local_identity_residual'
def descriptors(audio,energy):
    shape=audio-audio.mean(2,keepdims=True)
    shape=np.c_[shape.mean(1),shape.std(1)]
    en=energy-energy.mean(1,keepdims=True)
    en/=np.maximum(en.std(1,keepdims=True),1e-8)
    bins=np.stack([v.mean(1) for v in np.array_split(en,10,axis=1)],1)
    autoc=np.stack([(en[:,:-lag]*en[:,lag:]).mean(1) for lag in [1,2,5,10,20]],1)
    return {'shape':shape,'envelope':np.c_[bins,autoc]}
def predict(x,y,q):
    m=x.mean(0);s=np.maximum(x.std(0),1e-10);z=(x-m)/s;zz=(q-m)/s
    ym=y.mean();b=np.linalg.solve(z.T@z+10*np.eye(z.shape[1]),z.T@(y-ym))
    return zz@b+ym
def holm(p):
    order=np.argsort(p);q=np.zeros(len(p));last=0
    for rank,i in enumerate(order):last=max(last,min(1,p[i]*(len(p)-rank)));q[i]=last
    return q
def main():
    OUT.mkdir(exist_ok=True)
    c=np.load(BASE/'conditional_correspondence/raw_common_dynamics/local_power_private.npz')
    d=np.load(BASE/'phrase_auth/frontend_private.npz')
    for key in ['sample_id','user','word','repeat']:assert np.array_equal(c[key],d[key][c['indices']])
    u,w,r=[c[k] for k in ['user','word','repeat']];words=sorted(set(map(int,w)));assert len(words)==16
    trainwords=words[:12];valwords=words[12:];pairs=list(itertools.combinations(range(10),2))
    assert len(set(c['sample_id']))==1280 and all(np.sum((u==us)&(w==wo))==8 for us in range(1,11) for wo in words)
    audio=c['audio'].astype(float);energy=c['audio_activity'].astype(float);local=descriptors(audio,energy)
    acc=c['acc'].astype(float);acc=np.c_[acc.mean(1),acc.std(1),acc[:,:73].mean(1),acc[:,73:].mean(1)]
    act=np.c_[energy.mean(1),energy.std(1)]
    protocol=dict(train_words=trainwords,validation_words=valwords,users=10,samples=1280,
      candidates=['shape','envelope','combined'],shape='14D: per-frame 7band logpowers minus frequency mean; time mean/std; NOT F0 or resolved harmonics',
      envelope='15D: within-window standardized 300-3500Hz logenergy,10timebins,autocorrelation at20/40/100/200/400ms; NOT vocal-fold periodicity',
      target='bidirectional different-user cross-word raw ACC cosine; even repeats support, odd repeats nextword query',
      baseline='same-word ACC cosine + summed/absolute-difference energy mean/std; offline diagnostic uses Audio activity in botharms',
      fit='standardized Ridge10 fixed;45 endpointheldpair folds,8baseusers only;wordcentering28basepairs only;validationword background used for centering,notregression',
      permutation='199 fixedseed20261125 global Audio user permutations sharedacrosswords/candidates; refit everypredictor; candidatefeature scalers fittedbase/trainonly beforepermutation',
      multiplicity='Holm3 forprimaryMSEimprovement;hardquartile secondary,separateHolm3',
      hard='topquartile ACC samewordrelationship ontrain28basepairs',
      scope='historical16fitwordcache;fourvalidationwords alreadyseen;exploratorynotblind;physicalsync unverified;windowsmayoverlap;noF0claim;noauthenticationgainclaim',
      repeat_stability='Spearman same-word pairrelations even vs odd prototypes perword,descriptiveonly',
      no_final_test_audio=True,no_neural_training=True)
    (OUT/'protocol_public.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
    def proto(z,wo,parity):return unit(np.stack([z[(u==us)&(w==wo)&(r%2==parity)].mean(0) for us in range(1,11)]))
    folds=[]
    for held,(i,j) in enumerate(pairs):
      ref=np.array([i not in pp and j not in pp for pp in pairs]);base=np.isin(w,trainwords)&~np.isin(u,[i+1,j+1])
      xx=StandardScaler().fit(acc[base]).transform(acc)
      ll={k:StandardScaler().fit(v[base]).transform(v) for k,v in local.items()}
      fs=[];ys=[];aps={k:[] for k in ll};pid=[];wid=[]
      for group in [trainwords,valwords]:
       for wi,wo in enumerate(group):
        xs=proto(xx,wo,0);xq=proto(xx,group[(wi+1)%len(group)],1)
        ac=np.stack([act[(u==us)&(w==wo)&(r%2==0)].mean(0) for us in range(1,11)])
        f=np.array([np.r_[xs[v]@xs[t],ac[v]+ac[t],abs(ac[v]-ac[t])] for v,t in pairs])
        y=np.array([.5*(xs[v]@xq[t]+xs[t]@xq[v]) for v,t in pairs]);fs.extend(f-f[ref].mean(0));ys.extend(y-y[ref].mean())
        for k,z in ll.items():aps[k].append(proto(z,wo,0))
        pid.extend(range(45));wid.extend([wo]*45)
      f=np.array(fs);y=np.array(ys);pid=np.array(pid);wid=np.array(wid)
      tr=np.isin(wid,trainwords)&ref[pid];te=np.isin(wid,valwords)&(pid==held)
      pb=predict(f[tr],y[tr],f[te]);hard=f[te,0]>=np.quantile(f[tr,0],.75)
      folds.append((i,j,ref,f,y,tr,te,pb,aps,hard))
    rng=np.random.default_rng(20261125);perms=[np.arange(10)]+[rng.permutation(10) for _ in range(199)]
    results={k:[] for k in ['shape','envelope','combined']};rows=[]
    for it,perm in enumerate(perms):
      pool={k:[] for k in results}
      for i,j,ref,f,y,tr,te,pb,aps,hard in folds:
        zs={}
        for k,ap in aps.items():
          z=np.array([[v[perm[s]]@v[perm[t]] for s,t in pairs] for v in ap]);z-=z[:,ref].mean(1,keepdims=True);zs[k]=z.ravel()[:,None]
        zs['combined']=np.c_[zs['shape'],zs['envelope']]
        for k,z in zs.items():
          fq=np.c_[f,z];pr=predict(fq[tr],y[tr],fq[te]);delta=(y[te]-pb)**2-(y[te]-pr)**2
          pool[k].append((y[te],pb,pr,hard,delta))
          if it==0:rows.append(dict(user_pair=[i+1,j+1],candidate=k,baseline_mse=float(np.mean((y[te]-pb)**2)),audio_mse=float(np.mean((y[te]-pr)**2))))
      for k,v in pool.items():
        y,b,p,h,dlt=[np.concatenate([q[a] for q in v]) for a in range(5)]
        results[k].append(dict(baseline_mse=float(np.mean((y-b)**2)),audio_mse=float(np.mean((y-p)**2)),mse_reduction=float(dlt.mean()),hard_mse_reduction=float(dlt[h].mean()),baseline_spearman=float(spearmanr(y,b).statistic),audio_spearman=float(spearmanr(y,p).statistic)))
      if it%50==0:print('PERMUTATIONS',it,flush=True)
    summary=[]
    for k,res in results.items():
      v=dict(candidate=k,**res[0]);v['permutation_p']=(1+sum(q['mse_reduction']>=v['mse_reduction'] for q in res[1:]))/200
      v['hard_permutation_p']=(1+sum(q['hard_mse_reduction']>=v['hard_mse_reduction'] for q in res[1:]))/200;summary.append(v)
    for v,p,h in zip(summary,holm([v['permutation_p'] for v in summary]),holm([v['hard_permutation_p'] for v in summary])):v['holm_p']=float(p);v['hard_holm_p']=float(h)
    peruser=[]
    for us in range(1,11):
      for k in results:
        rr=[v for v in rows if us in v['user_pair'] and v['candidate']==k]
        peruser.append(dict(user=us,candidate=k,baseline_mse=float(np.mean([v['baseline_mse'] for v in rr])),audio_mse=float(np.mean([v['audio_mse'] for v in rr]))))
    for name,obj in [('summary_public.json',summary),('per_user_public.json',peruser),('per_pair_public.json',rows)]:
      (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# 局部 Audio 特征的条件身份残差检验','','三类候选预先固定；无最终测试 Audio、无神经模型训练。使用历史16训练词缓存12/4划分，四个验证词已被历史实验查看，属于探索性复核。','','|候选|基线MSE|加Audio MSE|整体改善|置换p|Holm p|难子集改善|难子集Holm p|','|---|---:|---:|---:|---:|---:|---:|---:|']
    for v in summary:lines.append('|'+v['candidate']+'|'+ '|'.join(f'{v[k]:.6f}' for k in ['baseline_mse','audio_mse','mse_reduction','permutation_p','holm_p','hard_mse_reduction','hard_holm_p'])+'|')
    lines+=['','正改善表示预测更好；只有正改善并且校正后的p<0.05才作为本轮有支持的候选。不得只报告最好一组。','','## 方法','','同词不同用户建立关系；控制同词ACC关系与能量均值/标准差。低频谱形状保留20–450Hz七个宽频带的相对形状及变化，能量节奏保留归一化时间轨迹与慢调制。两者都不是F0或精确谐波。45身份对轮流留出双方，训练只用其余8人；12词拟合、4词预测。每折标准化只用8人训练词，词级中心化只用28基础身份对。固定Ridge10，199次全身份Audio置换每次重拟合，三组做Holm校正。困难子集按训练ACC关系四分位数定义。','','## 解释边界','','这是原始ACC统计的关系预测，不是神经E0认证、蒸馏或EER收益。能量指标只是活动代理，不能认为完全控制了发声内容或活动。验证词基础身份数据用于描述性中心化，不是无背景的新用户部署。偶/奇遍来自原有切窗，可能存在重叠；物理同步未验证。45身份对和180任务共享10人，不能当180独立受试者。身份置换在用户层面进行。','','逐用户与身份对匿名误差另见JSON。未公开原始信号、embedding、模型、身份映射或逐样本分数。']
    supported=[v['candidate'] for v in summary if v['mse_reduction']>0 and v['holm_p']<.05]
    lines+=['','## 本轮判断','',f'通过整体检验的候选：{supported}。能量节奏虽然出现微小正改善，但整体和困难子集均未通过多重校正；谱形状与组合未改善。不能据此进入身份蒸馏。下一步尚未检验的是原始波形的真实F0/浊音周期性及分辨谐波；现有宽频带缓存无法回答这部分。', '', '偶/奇遍关系稳定性见 repeat_stability_public.json，仅是全队列训练词标准化后的描述性分析，不参与候选选择；汇总核验见 verification_public.json。']
    (OUT/'report.md').write_text('\n'.join(lines),encoding='utf-8');print(json.dumps(summary))
if __name__=='__main__':
    with threadpool_limits(limits=1):main()
