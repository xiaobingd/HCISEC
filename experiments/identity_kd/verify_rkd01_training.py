"""Post-hoc Base-only implementation audit; never retune on final results."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.preprocessing import StandardScaler
from run_rkd01_vibravox import Student,supcon,rkd,prepared,comparisons,encode_subset
def main(a):
    torch.set_num_threads(1);root=Path(a.root);out=root/'training';d=dict(np.load(root/'inputs_private.npz'));t=np.load(root/'audio_targets_private.npy');selection=json.loads((out/'selection_private.json').read_text())['selected']['S3'];seed=42;steps=selection['step'];lam=selection['lam'];users=list(range(35));fit=np.isin(d['user'],users)&(d['role']=='fit');x=StandardScaler().fit(d['x'][fit]).transform(d['x']).astype('float32');lookup={u:np.flatnonzero((d['user']==u)&(d['role']=='fit')) for u in users}
    torch.manual_seed(seed);rng=np.random.RandomState(seed);net=Student();head=nn.Linear(192,len(users));opt=torch.optim.AdamW(list(net.parameters())+list(head.parameters()),lr=1e-4,weight_decay=1e-4);history=[];params=list(net.parameters())
    for step in range(1,steps+1):
        net.train();uu=rng.choice(users,8,replace=False);ix=np.concatenate([rng.choice(lookup[u],4,replace=False) for u in uu]);y=torch.tensor(d['user'][ix],dtype=torch.long);z=net(torch.tensor(x[ix]));base=F.cross_entropy(16*F.linear(z,F.normalize(head.weight,dim=1)),y)+.1*supcon(z,y);kd=rkd(z,torch.tensor(t[ix]),y)
        if step in [1,100,steps]:
            gb=torch.autograd.grad(base,params,retain_graph=True,allow_unused=True);gk=torch.autograd.grad(lam*kd,params,retain_graph=True,allow_unused=True);b=torch.cat([v.flatten() for v in gb if v is not None]);k=torch.cat([v.flatten() for v in gk if v is not None]);assert len(b)==len(k) and torch.isfinite(k).all();history.append(dict(step=step,identity_loss=float(base.detach()),rkd_loss=float(kd.detach()),identity_grad_norm=float(b.norm()),weighted_rkd_grad_norm=float(k.norm()),weighted_rkd_to_identity_grad_ratio=float(k.norm()/b.norm()),gradient_cosine=float(F.cosine_similarity(b[None],k[None]))))
        loss=base+lam*kd;opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(list(net.parameters())+list(head.parameters()),1);opt.step()
    saved=torch.load(out/'S3_seed42_private.pt',map_location='cpu',weights_only=False)['state'];diff=max(float((net.state_dict()[k]-saved[k]).abs().max()) for k in saved);assert diff<1e-6,diff
    baseline=torch.load(out/'S1_seed42_private.pt',map_location='cpu',weights_only=False)['state'];delta=max(float((saved[k]-baseline[k]).abs().max()) for k in saved);assert delta>0
    r=dict(state='complete',scope='post-hoc gradient audit on Base-fit only;separate frozen-score verification of final results;no tuning',seed=seed,lambda_rkd=lam,replayed_steps=steps,max_replay_checkpoint_difference=diff,max_rkd_vs_no_kd_parameter_difference=delta,gradient_diagnostic=history)
    # Recompute decisions directly from frozen thresholds; use pairwise AUC,
    # independent of sklearn's ROC/AUC implementation used during evaluation.
    selection_all=json.loads((out/'selection_private.json').read_text())['selected'];xx,full,shape,_=prepared(d,users);cal=list(range(35,47));test=list(range(47,59));fi=np.flatnonzero(np.isin(d['user'],cal+test)&np.isin(d['role'],['registration','query']));checks=0
    for sd in [42,43,44]:
        results=json.loads((out/f'seed{sd}_results_private.json').read_text());vectors={'S0_shape':shape,'S0_allstats':full}
        for m in ['S1','S2','S3','S4','S5','S6','S7']:
            ck=torch.load(out/f'{m}_seed{sd}_private.pt',map_location='cpu',weights_only=False);assert ck['base_users']==users;model=Student();model.load_state_dict(ck['state']);vectors[m+'_student']=encode_subset(model,xx,fi)
        for name,row in results.items():
            for k in [3,5,12]:
                if name.endswith('_fusion'):
                    m=name.split('_')[0];p=selection_all[m]
                    def get(us):
                        s,y,owner=comparisons(vectors[m+'_student'],d,us,k);b,_,_=comparisons(shape,d,us,k);mu=p['score_means'];st=p['score_stds'];alpha=p['alpha'];return (1-alpha)*(b-mu[0])/max(st[0],1e-8)+alpha*(s-mu[1])/max(st[1],1e-8),y,owner
                    cs,cy,_=get(cal);s,y,owner=get(test)
                else:cs,cy,_=comparisons(vectors[name],d,cal,k);s,y,owner=comparisons(vectors[name],d,test,k)
                assert int(y.sum())==288 and int((~y).sum())==3168
                for point,tau in row[str(k)]['thresholds'].items():
                    if point=='Cal_FAR5':assert float((cs[~cy]>=tau).mean())<=.05
                    for u in test:
                        pos=s[(owner==u)&y];neg=s[(owner==u)&~y];assert len(pos)==24 and len(neg)==264;tar=float((pos>=tau).mean());far=float((neg>=tau).mean());auc=float(((pos[:,None]>neg).astype(float)+.5*(pos[:,None]==neg)).mean());v=row[str(k)]['targets'][str(u)][point]
                        assert max(abs(tar-v['tar']),abs(far-v['far']),abs(auc-v['auc']),abs((1-tar+far)/2-v['hter']))<1e-7;checks+=1
    r['frozen_target_workpoint_checks']=checks;r['pairwise_auc_and_cal_far5_verified']=True;(root/'implementation_audit_public.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);main(p.parse_args())
