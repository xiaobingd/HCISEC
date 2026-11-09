"""ABCS V4 score/identity/Audio-boundary checks; private per-target table."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from run_abcs_shape_v4 import Student,norm,encode,fold,reg,replay,score
def main(a):
    torch.set_num_threads(1);root=Path(a.root);d=dict(np.load(root/'inputs_private.npz'));audio=dict(np.load(root/'audio_fit_register_private.npz'));pub=json.loads((root/'results_public.json').read_text());assert not np.isin(d['role'][audio['indices']],['calibration','query','develop']).any();cache=torch.load(root/'audio_frontend_private.pt',map_location='cpu',weights_only=False);assert np.array_equal(cache['indices'],audio['indices']) and np.array_equal(cache['lengths'].numpy(),audio['lengths']);rows=[];deltas=[];ratios=[];maxscore=0.;maxreset=0.;checks=0
    for u in range(1,9):
        f=fold(u);folder=root/f'target{u}';row=json.loads((folder/'results_private.json').read_text());assert row['fold']==f;rows.append(row);fit=np.isin(d['user'],f['base'])&(d['role']=='fit');x=StandardScaler().fit(d['x'][fit]).transform(d['x']).astype('float32');shape=norm(x[:,:40]);stats=norm(StandardScaler().fit(d['statistics'][fit]).transform(d['statistics']));vectors={};originals=[]
        for owner in f['calibration']+[u]:
            p=folder/f'owner{owner}';t=dict(np.load(p/'teacher_targets_private.npz'));expected=np.r_[np.concatenate([replay(d,v) for v in f['base']]),reg(d,owner)];assert np.array_equal(t['indices'],expected) and set(expected)<=set(audio['indices']);assert np.isfinite(t['adapted']).all();originals.append(t['frozen'][:16]);audit=json.loads((p/'teacher_audit_private.json').read_text());assert audit['query_audio_model_forwards']==0 and audit['backbone_update']>0;deltas.append(audit['backbone_update']);sa=json.loads((p/'student_audit_private.json').read_text());assert sa['training_indices']==expected.tolist();ratios.extend(v['gradient_ratio'] for v in sa['trace']);net=Student();net.load_state_dict(torch.load(p/'student_private.pt',map_location='cpu',weights_only=False));ix=np.flatnonzero(np.isin(d['role'],['registration','calibration','query']));z=np.zeros((992,192),dtype='float32');z[ix]=encode(net,x[ix]);joint=np.concatenate([shape,z],1)/np.sqrt(2);assert np.allclose(joint[:,:40],shape/np.sqrt(2),atol=1e-7);vectors[owner]=dict(shape40=shape,statistics240=stats,student192=z,supplement232=joint)
        reset=max(float(abs(originals[0]-v).max()) for v in originals[1:]);assert reset<1e-5;maxreset=max(maxreset,reset);saved=json.loads((folder/'scores_private.json').read_text());taus=json.loads((folder/'thresholds_private.json').read_text())
        for method in vectors[u]:
            cs=np.concatenate([v['scores'] for v in saved[method]['calibration']]);cy=np.concatenate([v['labels'] for v in saved[method]['calibration']]);assert int(cy.sum())==26 and int((~cy).sum())==26;assert float((cs[~cy]>=taus[method]['Cal_FAR5']).mean())<=.05
            for j,owner in enumerate(f['calibration']):
                q=score(vectors[owner][method],d,owner,[v for v in f['calibration'] if v!=owner],'calibration');assert np.max(abs(np.array(q['scores'])-saved[method]['calibration'][j]['scores']))<1e-6
            for scope,attacks in [('unseen_attacker',f['attackers']),('all_seven_mixed',[v for v in range(1,9) if v!=u])]:
                q=score(vectors[u][method],d,u,attacks,'query');old=saved[method]['test'][scope];error=float(np.max(abs(np.array(q['scores'])-old['scores'])));assert error<1e-6;maxscore=max(maxscore,error);s=np.array(old['scores']);y=np.array(old['labels']);assert int(y.sum())==24 and int((~y).sum())==(24 if scope=='unseen_attacker' else 168);auc=float(((s[y,None]>s[None,~y]).astype(float)+.5*(s[y,None]==s[None,~y])).mean())
                for point,tau in taus[method].items():
                    r=row['results'][method][scope][point];tar=float((s[y]>=tau).mean());far=float((s[~y]>=tau).mean());assert max(abs(auc-r['auc']),abs(tar-r['tar']),abs(far-r['far']),abs((1-tar+far)/2-r['hter']))<1e-7;checks+=1
        print('VERIFIED_TARGET',u,flush=True)
    for method in pub['results']:
        for scope in pub['results'][method]:
            for point in pub['results'][method][scope]:
                for k in ['tar','far','frr','hter','auc','eer']:assert abs(pub['results'][method][scope][point][k]-np.mean([v['results'][method][scope][point][k] for v in rows]))<1e-12
                pub['results'][method][scope][point]['worst_target_far']=max(v['results'][method][scope][point]['far'] for v in rows)
                pub['results'][method][scope][point]['lowest_target_tar']=min(v['results'][method][scope][point]['tar'] for v in rows)
    pub['integrity']=dict(state='pass',personal_teachers=24,teacher_backbone_update_range=[min(deltas),max(deltas)],teacher_reset_base_embedding_max_difference=maxreset,realized_gradient_ratio_mean=float(np.mean(ratios)),realized_gradient_ratio_range=[min(ratios),max(ratios)],query_audio_model_forwards=0,teacher_audio_indices=128,valid_short_clip_lengths_preserved=True,only_vibration_inference_models_verified=24,max_score_recompute_error=maxscore,pairwise_auc_workpoint_checks=checks,shape40_not_replaced=True,all_8_targets_retained=True,aggregate_recomputed=True);(root/'results_public.json').write_text(json.dumps(pub,indent=2));text='# 私有ABCS八目标结果\n\n通道0暂作振动，通道1暂作Audio，物理映射未确认。仅供本地审阅，不上传GitHub。每目标24本人/24未见攻击，12注册。\n\n| 目标 | 谱形状TAR | 谱形状FAR | 谱形状HTER | 补充TAR | 补充FAR | 补充HTER |\n|---|---:|---:|---:|---:|---:|---:|\n'
    for u,row in enumerate(rows,1):
        b=row['results']['shape40']['unseen_attacker']['Cal_EER'];v=row['results']['supplement232']['unseen_attacker']['Cal_EER'];text+=f"| ABCS-{u} | {100*b['tar']:.2f}% | {100*b['far']:.2f}% | {100*b['hter']:.2f}% | {100*v['tar']:.2f}% | {100*v['far']:.2f}% | {100*v['hter']:.2f}% |\n"
    (root/'user_results_private.md').write_text(text,encoding='utf8');print(json.dumps(pub['integrity']))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);main(p.parse_args())
