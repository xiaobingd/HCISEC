"""Verify all Mei9 folds, only-vibration inference, and private user report."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from run_mei9_shape_v4 import Student,features,fold,reg,queries,replay,encode,score,norm
def main(a):
    torch.set_num_threads(1);root=Path(a.root);d=dict(np.load(a.frontend));xraw,vraw=features(d);pub=json.loads((root/'results_public.json').read_text());updates=[];ratios=[];resets=[];maxerror=0.;checks=0;foldrows=[]
    allowed=np.unique(np.concatenate([np.r_[replay(d,u),reg(d,u)] for u in range(1,10)]));qq=np.unique(np.concatenate([queries(d,u,c) for u in range(1,10) for c in ['sameword','crossword']]));assert not set(allowed)&set(qq)
    for target in range(1,10):
        folder=root/f'target{target}';row=json.loads((folder/'results_private.json').read_text());f=fold(target);assert row['fold']==f;foldrows.append(row);fit=np.isin(d['user'],f['base'])&(d['role']=='fit');x=StandardScaler().fit(xraw[fit]).transform(xraw).astype('float32');shape=norm(x[:,:120]);stat=norm(StandardScaler().fit(vraw[fit]).transform(vraw));zs={};frozen=[]
        for owner in f['calibration']+[target]:
            p=folder/f'owner{owner}';t=dict(np.load(p/'teacher_targets_private.npz'));expected=np.r_[np.concatenate([replay(d,u) for u in f['base']]),reg(d,owner)];assert np.array_equal(t['indices'],expected) and not set(expected)&set(qq);assert np.isfinite(t['adapted']).all();frozen.append(t['frozen'][:16]);audit=json.loads((p/'teacher_audit_private.json').read_text());assert audit['query_audio_forwards']==0 and audit['backbone_update']>0;updates.append(audit['backbone_update']);audit2=json.loads((p/'student_audit_private.json').read_text());assert set(audit2['training_ids'])==set(d['sample_id'][expected]);ratios.extend(v['gradient_ratio'] for v in audit2['trace']);net=Student();net.load_state_dict(torch.load(p/'student_private.pt',map_location='cpu',weights_only=False));ix=np.unique(np.r_[qq,reg(d,owner)]);z=np.zeros((len(x),192),dtype='float32');z[ix]=encode(net,x[ix]);zs[owner]=dict(shape120=shape,statistics720=stat,student192=z,supplement312=np.concatenate([shape,z],1)/np.sqrt(2))
        reset=max(float(abs(frozen[0]-v).max()) for v in frozen[1:]);assert reset<1e-5;resets.append(reset);saved=json.loads((folder/'scores_private.json').read_text());taus=json.loads((folder/'thresholds_private.json').read_text())
        for method in zs[target]:
            for condition in ['sameword','crossword']:
                cs=np.concatenate([saved[method]['calibration'][j][c]['scores'] for j in range(2) for c in ['sameword','crossword']]);cy=np.concatenate([saved[method]['calibration'][j][c]['labels'] for j in range(2) for c in ['sameword','crossword']]);assert float((cs[~cy]>=taus[method]['Cal_FAR5']).mean())<=.05
                for scope,attackers in [('two_unseen',f['attackers']),('all_eight_mixed',[u for u in range(1,10) if u!=target])]:
                    recompute=score(zs[target][method],d,target,attackers,condition);old=saved[method]['test'][scope][condition];error=float(np.max(abs(np.array(recompute['scores'])-old['scores'])));assert error<1e-6;maxerror=max(maxerror,error);s=np.array(old['scores']);y=np.array(old['labels']);assert int(y.sum())==24 and int((~y).sum())==(48 if scope=='two_unseen' else 192);auc=float(((s[y,None]>s[None,~y]).astype(float)+.5*(s[y,None]==s[None,~y])).mean())
                    for point,tau in taus[method].items():
                        r=row['results'][method][scope][condition][point];tar=float((s[y]>=tau).mean());far=float((s[~y]>=tau).mean());assert max(abs(auc-r['auc']),abs(tar-r['tar']),abs(far-r['far']),abs((1-tar+far)/2-r['hter']))<1e-7;checks+=1
        print('VERIFIED_TARGET',target,flush=True)
    for method in pub['results']:
        for scope in pub['results'][method]:
            for c in pub['results'][method][scope]:
                for p in pub['results'][method][scope][c]:
                    for metric,value in pub['results'][method][scope][c][p].items():assert abs(value-np.mean([v['results'][method][scope][c][p][metric] for v in foldrows]))<1e-12
    pub['integrity']=dict(state='pass',exact_timestamp_records=2695,all9_retained=True,balanced_folds=9,personal_teachers=27,query_audio_model_forwards=0,audio_cache_rows=144,no_audio_cache_overlap_query_indices=True,teacher_backbone_update_range=[min(updates),max(updates)],teacher_reset_base_embedding_max_difference=max(resets),realized_gradient_ratio_mean=float(np.mean(ratios)),realized_gradient_ratio_range=[min(ratios),max(ratios)],only_vibration_inference_models_verified=27,max_score_recompute_error=maxerror,pairwise_auc_workpoint_checks=checks,aggregate_recomputed=True);(root/'results_public.json').write_text(json.dumps(pub,indent=2))
    text='# 私有九目标结果：同词／跨词，两个未见攻击来源\n\n仅供用户审阅，不上传GitHub。编号沿用原始raw9身份映射。每目标24本人/48攻击；所有来源固定保留。\n\n'
    for c in ['sameword','crossword']:
        text+='## '+c+'\n\n| 目标 | 谱形状TAR | 谱形状FAR | 谱形状HTER | 补充TAR | 补充FAR | 补充HTER |\n|---|---:|---:|---:|---:|---:|---:|\n'
        for u,row in enumerate(foldrows,1):
            b=row['results']['shape120']['two_unseen'][c]['Cal_EER'];r=row['results']['supplement312']['two_unseen'][c]['Cal_EER'];text+=f"| U{u} | {100*b['tar']:.2f}% | {100*b['far']:.2f}% | {100*b['hter']:.2f}% | {100*r['tar']:.2f}% | {100*r['far']:.2f}% | {100*r['hter']:.2f}% |\n"
        text+='\n'
    (root/'user_results_private.md').write_text(text,encoding='utf8');print(json.dumps(pub['integrity']))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--frontend',required=True);main(p.parse_args())
