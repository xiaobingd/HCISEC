"""Registration RKD-02 integrity checks and identity-level paired summaries."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from run_registration_rkd02 import pool_ids,registration,score_owner
from run_rkd01_vibravox import Student,thresholds,metrics
def main(a):
    torch.set_num_threads(1);root=Path(a.root);source=Path(a.source);pub=json.loads((root/'results_public.json').read_text());rho=pub['rho'];d=dict(np.load(source/'inputs_private.npz'));old=np.load(source/'audio_targets_private.npy');teacher_updates=[];cosines=[];raw={m:{} for m in ['V0','V1','V2','V3','V4','V5','V6'] if m in pub['results']};grad={m:[] for m in raw};base=list(range(35));checks=0
    for u in range(35,59):
        folder=root/'outer'/f'owner{u}';t=dict(np.load(folder/'teacher_targets_private.npz'));pi=pool_ids(d,base);assert np.array_equal(t['indices'],np.r_[pi,registration(d,u)]);assert len(registration(d,u))==12;assert np.isfinite(t['adapted']).all();c=(t['frozen'][:len(pi)]*old[pi]).sum(1);assert c.min()>.9999;cosines.extend(c);audit=json.loads((folder/'teacher_audit_private.json').read_text());assert audit['query_audio_reads']==0 and audit['backbone_update']>0;teacher_updates.append(audit['backbone_update'])
        for m in raw:
            q=json.loads((folder/f'{m}_rho{rho}_private.json').read_text());raw[m][u]=q;grad[m].extend(q['trace']);y=np.array(q['labels']);assert int(y.sum())==24 and int((~y).sum())==264
    paired_rows={};workpoints={}
    for m in raw:
        cs=np.concatenate([raw[m][u]['scores'] for u in range(35,47)]);cy=np.concatenate([raw[m][u]['labels'] for u in range(35,47)]);taus=thresholds(cs,cy);workpoints[m]=taus;paired_rows[m]={}
        for point,tau in taus.items():
            if point=='Cal_FAR5':assert float((cs[~cy]>=tau).mean())<=.05
            vals=[]
            for u in range(47,59):
                q=raw[m][u];s=np.array(q['scores']);y=np.array(q['labels']);pos=s[y];neg=s[~y];auc=float(((pos[:,None]>neg).astype(float)+.5*(pos[:,None]==neg)).mean());r=metrics(s,y,tau);assert abs(r['auc']-auc)<1e-7;vals.append(r);checks+=1
            assert abs(np.mean([v['auc'] for v in vals])-pub['results'][m][point]['macro_auc'])<1e-7;assert abs(np.mean([v['hter'] for v in vals])-pub['results'][m][point]['hter'])<1e-7;paired_rows[m][point]=vals
    # Sampled full inference audit: only vibration tensors + saved student weights.
    fit=(d['user']<35)&(d['role']=='fit');x=StandardScaler().fit(d['x'][fit]).transform(d['x']).astype('float32');maxdiff=0
    for u in [35,47]:
        for m in raw:
            net=Student();net.load_state_dict(torch.load(root/'outer'/f'owner{u}'/f'{m}_rho{rho}_student_private.pt',map_location='cpu',weights_only=False));group=list(range(35,47)) if u<47 else list(range(47,59));r=score_owner(net,x,d,u,[v for v in group if v!=u]);diff=float(np.max(abs(np.array(r['scores'])-raw[m][u]['scores'])));assert diff<1e-6;maxdiff=max(maxdiff,diff)
    rng=np.random.RandomState(20261110);paired={}
    for treatment,control in [('V2','V1'),('V3','V1'),('V4','V2'),('V4','V3'),('V4','V0')]+([('V4','V5'),('V4','V6')] if 'V5' in raw else []):
        delta=np.array([t['hter']-c['hter'] for t,c in zip(paired_rows[treatment]['Cal_EER'],paired_rows[control]['Cal_EER'])]);boots=np.array([delta[rng.choice(12,12,replace=True)].mean() for _ in range(5000)]);paired[treatment+'_minus_'+control]=dict(hter_delta=float(delta.mean()),identity_bootstrap95=np.quantile(boots,[.025,.975]).tolist(),improved_targets=int((delta<0).sum()))
    pub['paired']=paired;pub['gradient_audit']={m:dict(lambda_mean=float(np.mean([v['lambda_rkd'] for v in vs])),realized_gradient_ratio_mean=float(np.mean([v['gradient_ratio'] for v in vs])),realized_gradient_ratio_range=[min(v['gradient_ratio'] for v in vs),max(v['gradient_ratio'] for v in vs)]) for m,vs in grad.items()};pub['integrity']=dict(state='pass',outer_personal_teachers=24,teacher_parameter_update_range=[min(teacher_updates),max(teacher_updates)],frozen_base_teacher_min_cosine_vs_original=float(min(cosines)),cal_far5_bound_verified=True,pairwise_auc_workpoint_checks=checks,inference_models_spot_checked=2*len(raw),max_spot_inference_score_difference=maxdiff,query_audio_reads=0);(root/'results_public.json').write_text(json.dumps(pub,indent=2));(root/'thresholds_private.json').write_text(json.dumps(workpoints));print(json.dumps(dict(state='verified',integrity=pub['integrity'],paired=paired,gradients=pub['gradient_audit']),indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--source',required=True);main(p.parse_args())
