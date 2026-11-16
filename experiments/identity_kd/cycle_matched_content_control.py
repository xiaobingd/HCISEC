"""Frozen innovation-1 CycleGAN, identical queries, changed registration word.

Historical personal GAN adaptation is fixed in BOTH conditions. New K4
registration only fits the RBF head; no new Audio or generator updates.
"""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='2'
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np
import joblib
import torch
from sklearn.svm import SVC
from mei9_closed_personal_cycle import Generator
from run_mei9_shape_v4 import Student,encode
from vibravox_cycle_supplement import pooled
from run_crossword_neural_verifier import norm,metrics,threshold,dump

CAL=[21,22,23]
FINAL=[24,25,26,27,28,29]
PRIMARY=[1,2,3,4,6,7]
USERS=list(range(1,10))
METHODS=['statistics720','baseline312','selected_input256','cycle256_fused','cycle256_only']


def main(a):
    torch.set_num_threads(2)
    root=Path(a.assets);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    if not (root/'source_hashes_private.json').exists():
        root.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(a.bundle) as archive:
            for member in archive.infolist():
                assert (root/member.filename).resolve().is_relative_to(root.resolve())
            archive.extractall(root)
    hashes=json.loads((root/'source_hashes_private.json').read_text())
    for name,expected in hashes.items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected,name
    d=np.load(a.frontend);n=len(d['user'])
    original={k:np.array(v) for k,v in json.loads((root/'split_private.json').read_text()).items()}
    choice=json.loads((root/'selection_public.json').read_text())['selected']
    assert choice['step']==100 and choice['branch']=='input256' and choice['alpha']==.5
    # Verify original nominal same-word claim and its published scores.
    assert set(d['word'][original['register']])==set(d['word'][original['sameword']])==set(range(20,24))
    assert not set(original['register'])&set(original['sameword'])
    assert not set(d['word'][original['register']])&set(d['word'][original['crossword']])
    old=json.loads((root/'scores_private.json').read_text())
    published=json.loads((root/'per_user_public.json').read_text())
    old_replayed=0;old_error=0.
    for u in USERS:
        for method in ['statistics720','baseline312','selected_cycle']:
            cs=old[str(u)][method]['calibration']
            tau=threshold(np.array(cs['labels'],bool),np.array(cs['scores']))
            for cond in ['sameword','crossword']:
                q=old[str(u)][method][cond]
                m=metrics(np.array(q['labels'],bool),np.array(q['scores']),tau)
                saved=next(r for r in published if r['user']==f'Mei9-U{u:02}' and r['method']==method and r['condition']==cond and r['point']=='Cal_EER')
                err=max(abs(m[k]-saved[k]) for k in ['tar','far','frr','hter','auc'])
                assert err<1e-12,(u,method,cond,err)
                old_error=max(old_error,err);old_replayed+=1
    lookup={(u,w):np.flatnonzero((d['user']==u)&(d['word']==w)) for u in USERS for w in CAL+FINAL}
    assert all(len(j)>=8 for j in lookup.values())
    def reg(word,fold,u):return lookup[u,word][fold*4:fold*4+4]
    def queries(word,fold):return np.concatenate([lookup[u,word][(1-fold)*4:(1-fold)*4+4] for u in USERS])
    final_ids=np.unique(np.concatenate([lookup[u,w][:8] for u in USERS for w in FINAL]))
    calq=np.unique(np.concatenate([queries(w,0) for w in CAL]))
    assert not set(calq)&set(np.r_[original['fit'],original['dev'],original['register']])
    assert not set(final_ids)&set(np.r_[original['fit'],original['dev'],original['register']])
    selected=np.unique(np.r_[final_ids,np.concatenate([lookup[u,w][:8] for u in USERS for w in CAL])])
    aa=np.load(root/'prepared/a.npy',mmap_mode='r')
    shape=np.load(root/'prepared/shape.npy');raw=np.load(root/'prepared/stats.npy')
    net=Student();net.load_state_dict(torch.load(root/'student_private.pt',map_location='cpu',weights_only=False))
    base=np.c_[shape,encode(net,np.load(root/'prepared/x.npy'))]/np.sqrt(2)
    scalers=joblib.load(root/'gan_scalers_private.joblib')
    dump(out/'protocol_public.json',dict(state='running',primary_targets=PRIMARY,all_targets=USERS,
        primary_scope='same six targets as preceding report;all nine trained and all eight attackers retained',
        source_selection=choice,personal_generator='original owner100,old12 registration records,unchanged across comparisons',
        methods=METHODS,K=4,calibration_words=CAL,final_words=FINAL,ordered_crossword_pairs=30,
        final_folds='register0:4/query4:8 then reverse',
        calibration='fixed register0:4/query4:8 in both folds;avoids old personal training records in Cal queries',
        threshold='one per target/method,balanced pooled same/cross Cal;shared all final words/folds/conditions',
        no_new_training=True,new_registration='RBF K4 only,not GAN adaptation',
        original_sameword_is_nominally_valid=True,
        original_comparison_query_words_differ=True,
        limits=['K4 differs from originalK12','frozen historical personal adaptation is not new K4 adaptation',
                'nominal word labels,not transcript-verified','single-session explored cohort','pairs reuse queries']))
    dump(out/'original_result_audit_public.json',dict(state='PASS',workpoints=old_replayed,max_error=old_error,
        sameword_register_query_word_sets_equal=True,register_query_record_disjoint=True,
        cross_query_words_disjoint_from_registration=True,
        conclusion='original same-word labeling valid by nominal IDs;original same/cross comparison uses different query words'))
    rows=[];saved_scores={};saved_thresholds={};parity={}
    for owner in USERS:
        ck=torch.load(root/f'owner{owner}/gan100_private.pt',map_location='cpu',weights_only=False)
        assert ck['owner']==owner and ck['step']==100
        assert len(ck['train_indices'])==12 and set(d['user'][ck['train_indices']])=={owner}
        assert not set(ck['train_indices'])&set(np.r_[calq,final_ids])
        feature_path=out/f'owner{owner}_features_private.npz'
        if feature_path.exists():
            z=np.load(feature_path);assert np.array_equal(z['indices'],selected)
            input_raw=z['input256'];cycle_raw=z['cycle256']
        else:
            g=Generator(3,1).eval();f=Generator(1,3).eval()
            g.load_state_dict(ck['G']);f.load_state_dict(ck['F']);p=[];q=[]
            with torch.no_grad():
                for start in range(0,len(selected),8):
                    x=torch.tensor(np.array(aa[selected[start:start+8]]))
                    h=g.encoder(x);c=f(g.decoder(h))
                    p.append(pooled(h).numpy());q.append(pooled(g.encoder(c)).numpy())
            input_raw=np.concatenate(p);cycle_raw=np.concatenate(q)
            np.savez_compressed(feature_path,indices=selected,input256=input_raw,cycle256=cycle_raw)
            del g,f
        zz={}
        for name,value in [('input256',input_raw),('cycle256',cycle_raw)]:
            z=np.zeros((n,256),dtype='float32');z[selected]=norm(scalers[name].transform(value));zz[name]=z
        # Compare newly extracted raw features with the original canonical cache.
        reference=root/f'owner{owner}/features100_540_private.npz'
        if reference.exists():
            oldix=np.unique(np.r_[original['register'],original['calA'],original['calB'],original['sameword'],original['crossword']])
            overlap,ii,jj=np.intersect1d(selected,oldix,return_indices=True)
            prior=np.load(reference)
            error=max(float(np.max(np.abs(input_raw[ii]-prior['input256'][jj]))),float(np.max(np.abs(cycle_raw[ii]-prior['cycle256'][jj]))))
            assert error<1e-4,error
            parity[f'U{owner}']=dict(records=len(overlap),max_abs_error=error)
        banks=dict(statistics720=raw,baseline312=base,
            selected_input256=np.c_[base,.5*zz['input256']]/np.sqrt(1.25),
            cycle256_fused=np.c_[base,.5*zz['cycle256']]/np.sqrt(1.25),cycle256_only=zz['cycle256'])
        for method,v in banks.items():
            models={};cache={}
            def score(word,qword,fold):
                key=(word,qword,fold)
                if key in cache:return cache[key]
                if (word,fold) not in models:
                    ix=np.concatenate([reg(word,fold,u) for u in USERS])
                    models[word,fold]=SVC(C=10,gamma='scale',class_weight='balanced').fit(v[ix],d['user'][ix]==owner)
                cache[key]=models[word,fold].decision_function(v[queries(qword,fold)])
                return cache[key]
            cy=[];cs=[]
            for qw in CAL:
                yy=d['user'][queries(qw,0)]==owner
                for rw in CAL:
                    if rw==qw:continue
                    cy.extend([yy,yy]);cs.extend([score(qw,qw,0),score(rw,qw,0)])
            tau=threshold(np.concatenate(cy),np.concatenate(cs));saved_thresholds[f'{owner}_{method}']=tau
            for fold in [0,1]:
                for qw in FINAL:
                    yy=d['user'][queries(qw,fold)]==owner
                    for rw in FINAL:
                        if rw==qw:continue
                        for cond,word in [('sameword',qw),('crossword',rw)]:
                            qi=queries(qw,fold);ri=reg(word,fold,owner)
                            assert not set(qi)&set(ri)
                            ss=score(word,qw,fold)
                            saved_scores[f'{owner}_{method}_{fold}_{qw}_{rw}_{cond}']=ss
                            rows.append(dict(user=f'U{owner}',method=method,fold=fold,query_word=qw,
                                cross_registration_word=rw,condition=cond,**metrics(yy,ss,tau)))
        dump(out/'status.json',dict(stage='scoring',owner=owner,completed_owners=owner,total_owners=9))
        print('CYCLE_OWNER_COMPLETE',owner,flush=True)
    np.savez_compressed(out/'scores_private.npz',**saved_scores);dump(out/'thresholds_private.json',saved_thresholds)
    assert len(rows)==9*5*2*30*2
    error=0.
    for r in rows:
        owner=int(r['user'][1:]);method=r['method'];fold=r['fold'];qw=r['query_word'];rw=r['cross_registration_word']
        yy=d['user'][queries(qw,fold)]==owner
        ss=saved_scores[f'{owner}_{method}_{fold}_{qw}_{rw}_{r["condition"]}']
        mm=metrics(yy,ss,saved_thresholds[f'{owner}_{method}'])
        error=max(error,max(abs(mm[k]-r[k]) for k in mm))
    assert error==0
    per_user=[];summary=[]
    keys=['tar','far','frr','hter','eer','auc']
    for method in METHODS:
        for cond in ['sameword','crossword']:
            for u in USERS:
                rr=[r for r in rows if r['user']==f'U{u}' and r['method']==method and r['condition']==cond]
                per_user.append(dict(user=f'U{u}',method=method,condition=cond,
                                    **{k:float(np.mean([r[k] for r in rr])) for k in keys}))
            for scope,users in [('primary_six',PRIMARY),('all_nine',USERS)]:
                rr=[r for r in per_user if r['method']==method and r['condition']==cond and int(r['user'][1:]) in users]
                summary.append(dict(scope=scope,method=method,condition=cond,
                                    **{k:float(np.mean([r[k] for r in rr])) for k in keys}))
    dump(out/'per_user_public.json',per_user);dump(out/'results_public.json',dict(state='complete',summary=summary))
    dump(out/'verification_public.json',dict(state='PASS',matched_workpoints=len(rows),saved_score_replay_error=error,
         old_workpoints_replayed=old_replayed,old_metric_error=old_error,
         old_personal_training_disjoint_from_cal_queries_and_final=True,
         cal_queries_fixed_to_unseen_repeats=True,threshold_shared_same_cross_and_folds=True,
         source_payload_hashes_verified=True,source_feature_parity=parity,
         limits='score replay and source hash checks;not independent end-to-end implementation'))
    lines=['# 第一创新点CycleGAN：固定查询同词／跨词核查','',
        '复用定版个人CycleGAN100步、既有学生及Scaler；冻结融合权重0.5。原结果的同词名义标签有效：注册与同词查询均为词20–23的不同录制；其与跨词24–29的查询内容不同，不能直接隔离内容失配影响。',
        '', '新对照K4注册、K4查询、最终六词24–29全部30方向、两折交换重复。同词/跨词每个配对使用相同查询、攻击和阈值。个人GAN仍是旧本人12条微调，不根据新K4注册重训。Cal查询固定词21–23重复4–7，避免反向折把旧微调录制作为Cal查询。',
        '', 'input256为循环一致性训练的生成器输入编码；cycle256为ACC→Audio→ACC之后再编码。两者分开报告。selected_input256严格沿用旧选择；cycle256_fused为固定0.5诊断，不重新选权重。',
        '', '|范围|方法|条件|TAR%|FAR%|HTER%|EER%|AUC|','|---|---|---|---:|---:|---:|---:|---:|']
    for r in summary:lines.append(f"|{r['scope']}|{r['method']}|{r['condition']}|{r['tar']*100:.2f}|{r['far']*100:.2f}|{r['hter']*100:.2f}|{r['eer']*100:.2f}|{r['auc']:.4f}|")
    lines+=['','primary_six=U1/U2/U3/U4/U6/U7，沿用上一轮统计；all_nine提供原完整队列。所有范围保留全部八名攻击者，不删除被排除目标的攻击。',
        '', '## 逐用户完整结果','', '|用户|方法|条件|TAR%|FAR%|HTER%|EER%|AUC|','|---|---|---|---:|---:|---:|---:|---:|']
    for r in sorted(per_user,key=lambda r:(r['user'],r['method'],r['condition'])):lines.append(f"|{r['user']}|{r['method']}|{r['condition']}|{r['tar']*100:.2f}|{r['far']*100:.2f}|{r['hter']*100:.2f}|{r['eer']*100:.2f}|{r['auc']:.4f}|")
    lines+=['',f'原结果54项工作点重算误差{old_error:.3g}；新对照{len(rows)}项保存分数重算误差{error:.3g}。源模型/前端文件SHA校验通过，模型和融合权重未修改。',
        '', 'K4不同于旧K12，绝对数值不直接比较。共享G/F训练、历史个人微调、学生蒸馏均为原方案，不属于本轮新增训练。原始数据、特征、模型、分数及阈值私有。单种子、名义词序、已探索同会话数据；词对与两折复用录制，不构成独立新样本。']
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    dump(out/'status.json',dict(stage='complete'))
    print('CYCLE_MATCHED_CONTROL_COMPLETE',json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--frontend',required=True);p.add_argument('--assets',required=True)
    p.add_argument('--bundle',required=True);p.add_argument('--out',required=True);main(p.parse_args())
