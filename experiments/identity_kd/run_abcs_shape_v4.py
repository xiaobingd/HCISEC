"""ABCS cross-sentence V4 only: fixed shape40 plus student192."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import argparse,copy,json,sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.preprocessing import StandardScaler
from run_rkd01_vibravox import Student,norm,supcon,rkd,encode,thresholds,metrics
from run_registration_rkd02 import dump

def fold(u):
    o=[1+(u-1+j)%8 for j in range(8)];return dict(target=u,attackers=o[1:2],calibration=o[2:4],base=o[4:])
def reg(d,u):return np.flatnonzero((d['user']==u)&(d['role']=='registration'))
def replay(d,u):return np.flatnonzero((d['user']==u)&(d['role']=='fit')&np.isin(d['text_index'],[0,15,30,45]))
def base_student(d,x,base,path):
    net=Student()
    if path.exists():net.load_state_dict(torch.load(path,map_location='cpu',weights_only=False));return net
    torch.manual_seed(42);rng=np.random.RandomState(42);net=Student();head=nn.Linear(192,4,bias=False);opt=torch.optim.AdamW(list(net.parameters())+list(head.parameters()),lr=1e-4,weight_decay=1e-4);lookup={(u,t):np.flatnonzero((d['user']==u)&(d['text_index']==t))[0] for u in base for t in range(60)};labels={u:j for j,u in enumerate(base)}
    for step in range(700):
        tt=rng.choice(60,8,replace=False);ii=np.array([lookup[u,t] for u in base for t in tt]);y=torch.tensor([labels[int(u)] for u in d['user'][ii]]);z=net(torch.tensor(x[ii]));loss=F.cross_entropy(16*F.linear(z,F.normalize(head.weight,dim=1)),y)+.1*supcon(z,y);opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(list(net.parameters())+list(head.parameters()),1);opt.step()
    torch.save(net.state_dict(),path);return net
def teacher(d,u,base,folder,model,initial,cache,slots):
    path=folder/'teacher_targets_private.npz'
    if path.exists():t=dict(np.load(path));return t['indices'],t['adapted']
    torch.manual_seed(4242+u);rng=np.random.RandomState(4242+u);ix=np.r_[np.concatenate([replay(d,v) for v in base]),reg(d,u)];assert len(ix)==28 and len(reg(d,u))==12;assert all((d['user'][i] in base and d['role'][i]=='fit') or (d['user'][i]==u and d['role'][i]=='registration') for i in ix)
    emb=model.mods.embedding_model;emb.load_state_dict(initial);emb.eval()
    for p in emb.parameters():p.requires_grad_(True)
    device=next(emb.parameters()).device;jj=torch.tensor([slots[int(i)] for i in ix]);feat=cache['features'][jj].to(device);lengths=cache['lengths'][jj].to(device);labels_map={v:j for j,v in enumerate(base+[u])};yall=torch.tensor([labels_map[int(v)] for v in d['user'][ix]],device=device)
    def forward(i):return F.normalize(emb(feat[i],lengths[i]).squeeze(1),dim=1)
    with torch.no_grad():frozen=forward(torch.arange(len(ix),device=device)).detach();proto=F.normalize(torch.stack([frozen[yall==j].mean(0) for j in range(5)]),dim=1)
    head=nn.Linear(192,5,bias=False).to(device)
    with torch.no_grad():head.weight.copy_(proto)
    opt=torch.optim.AdamW([dict(params=emb.parameters(),lr=1e-6),dict(params=head.parameters(),lr=1e-4)],weight_decay=1e-4);watch=next(p for p in emb.parameters() if p.ndim>1);before=watch.detach().clone()
    for step in range(30):
        batch=np.r_[rng.choice(np.flatnonzero(d['user'][ix]==u),4,replace=False),np.concatenate([rng.choice(np.flatnonzero(d['user'][ix]==v),2,replace=False) for v in base])];j=torch.tensor(batch,device=device);z=forward(j);y=yall[j];loss=F.cross_entropy(16*F.linear(z,F.normalize(head.weight,dim=1)),y)+.1*supcon(z.cpu(),y.cpu()).to(device)+.1*(1-F.cosine_similarity(z[4:],frozen[j[4:]],dim=1)).mean();opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(list(emb.parameters())+list(head.parameters()),1);opt.step()
    delta=float((watch-before).abs().max());assert delta>0
    for p in emb.parameters():p.requires_grad_(False)
    with torch.no_grad():adapted=forward(torch.arange(len(ix),device=device)).cpu().numpy()
    np.savez_compressed(path,indices=ix,frozen=frozen.cpu().numpy(),adapted=adapted);torch.save(emb.state_dict(),folder/'ecapa_private.pt');dump(folder/'teacher_audit_private.json',dict(owner=u,base=base,steps=30,backbone_update=delta,query_audio_model_forwards=0));return ix,adapted
def personal(d,x,initial,u,base,ix,t,folder):
    path=folder/'student_private.pt';net=copy.deepcopy(initial)
    if path.exists():net.load_state_dict(torch.load(path,map_location='cpu',weights_only=False));return net
    torch.manual_seed(42);rng=np.random.RandomState(42000+u);xx=x[ix];original=encode(initial,xx);labels={v:j for j,v in enumerate(base+[u])};yall=np.array([labels[int(v)] for v in d['user'][ix]]);lookup={v:np.flatnonzero(d['user'][ix]==v) for v in base+[u]};head=nn.Linear(192,5,bias=False)
    with torch.no_grad():head.weight.copy_(torch.tensor(norm(np.stack([original[yall==j].mean(0) for j in range(5)]))))
    params=list(net.parameters());opt=torch.optim.AdamW(params+list(head.parameters()),lr=1e-4,weight_decay=1e-4);ei=None;ek=None;trace=[]
    for step in range(1,101):
        batch=np.r_[rng.choice(lookup[u],8,replace=False),np.concatenate([lookup[v] for v in base])];y=torch.tensor(yall[batch]);z=net(torch.tensor(xx[batch]));identity=F.cross_entropy(16*F.linear(z,F.normalize(head.weight,dim=1)),y)+.1*supcon(z,y)+.1*(1-F.cosine_similarity(z[8:],torch.tensor(original[batch[8:]]),dim=1)).mean();kd=rkd(z,torch.tensor(t[batch]),y);gi=torch.autograd.grad(identity,params,retain_graph=True,allow_unused=True);gk=torch.autograd.grad(kd,params,retain_graph=True,allow_unused=True);ni=float(torch.sqrt(sum(v.square().sum() for v in gi if v is not None)));nk=float(torch.sqrt(sum(v.square().sum() for v in gk if v is not None)));ei=ni if ei is None else .9*ei+.1*ni;ek=nk if ek is None else .9*ek+.1*nk;lam=min(100.,.3*ei/max(ek,1e-8));loss=identity+lam*kd;opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(params+list(head.parameters()),1);opt.step()
        if step in [1,25,50,100]:trace.append(dict(step=step,lambda_rkd=lam,gradient_ratio=lam*nk/max(ni,1e-8)))
    torch.save(net.state_dict(),path);dump(folder/'student_audit_private.json',dict(trace=trace,training_indices=ix.tolist(),steps=100,rho=.3));return net
def score(z,d,u,attacks,role):
    en=reg(d,u);q=np.flatnonzero(np.isin(d['user'],[u]+attacks)&(d['role']==role));assert len(en)==12 and not set(en)&set(q);return dict(scores=(z[q]@norm(z[en].mean(0))).tolist(),labels=(d['user'][q]==u).tolist())
def evaluate(cal,q):
    cs=np.concatenate([v['scores'] for v in cal]);cy=np.concatenate([v['labels'] for v in cal]);taus=thresholds(cs,cy);s=np.array(q['scores']);y=np.array(q['labels']);assert float((cs[~cy]>=taus['Cal_FAR5']).mean())<=.05;return {p:metrics(s,y,t) for p,t in taus.items()},taus
def main(a):
    torch.set_num_threads(1);root=Path(a.root);d=dict(np.load(root/'inputs_private.npz'));audio=dict(np.load(root/'audio_fit_register_private.npz'));assert d['x'].shape==(992,246) and len(audio['indices'])==128;assert not np.isin(d['role'][audio['indices']],['develop','calibration','query']).any();sets=[set(d['text_index'][d['role']==r]) for r in ['fit','develop','registration','calibration','query']];assert all(not sets[i]&sets[j] for i in range(5) for j in range(i))
    sys.path.insert(0,'/root');from run_ecapa_pilot import teacher_load
    for name in ['model','deps']:
        p=root/name
        if not p.exists():p.symlink_to(Path(a.assets)/name,target_is_directory=True)
    model=teacher_load(root);original={k:v.detach().cpu().clone() for k,v in model.mods.embedding_model.state_dict().items()};expected=torch.load(root/'model'/'embedding_model.ckpt',map_location='cpu',weights_only=False);assert set(original)==set(expected) and all(torch.equal(original[k],expected[k]) for k in original);del expected;device=next(model.parameters()).device;cachepath=root/'audio_frontend_private.pt'
    if cachepath.exists():cache=torch.load(cachepath,map_location='cpu',weights_only=False)
    else:
        ff=[]
        with torch.no_grad():
            for s in range(0,len(audio['waves']),16):
                w=torch.tensor(audio['waves'][s:s+16],device=device);lens=torch.tensor(audio['lengths'][s:s+16],device=device);ff.append(model.mods.mean_var_norm(model.mods.compute_features(w),lens).cpu())
        cache=dict(features=torch.cat(ff),lengths=torch.tensor(audio['lengths']),indices=audio['indices']);torch.save(cache,cachepath)
    assert np.array_equal(cache['indices'],audio['indices']);slots={int(i):j for j,i in enumerate(audio['indices'])};rows={}
    for u in range(1,9):
        f=fold(u);assert len(set(f['base']+f['calibration']+f['attackers']+[u]))==8;folder=root/f'target{u}';folder.mkdir(exist_ok=True);done=folder/'results_private.json'
        if done.exists():rows[str(u)]=json.loads(done.read_text());continue
        fit=np.isin(d['user'],f['base'])&(d['role']=='fit');x=StandardScaler().fit(d['x'][fit]).transform(d['x']).astype('float32');shape=norm(x[:,:40]);stats=norm(StandardScaler().fit(d['statistics'][fit]).transform(d['statistics']));initial=base_student(d,x,f['base'],folder/'base_student_private.pt');zs={}
        for owner in f['calibration']+[u]:
            p=folder/f'owner{owner}';p.mkdir(exist_ok=True);ix,t=teacher(d,owner,f['base'],p,model,original,cache,slots);net=personal(d,x,initial,owner,f['base'],ix,t,p);z=encode(net,x);joint=np.concatenate([shape,z],1)/np.sqrt(2);assert np.allclose(joint[:,:40],shape/np.sqrt(2),atol=1e-7);zs[owner]=dict(shape40=shape,statistics240=stats,student192=z,supplement232=joint)
        results={};scores={};thresholds_private={}
        for m in zs[u]:
            cal=[score(zs[v][m],d,v,[w for w in f['calibration'] if w!=v],'calibration') for v in f['calibration']];scores[m]=dict(calibration=cal,test={});results[m]={}
            for scope,attacks in [('unseen_attacker',f['attackers']),('all_seven_mixed',[v for v in range(1,9) if v!=u])]:
                q=score(zs[u][m],d,u,attacks,'query');r,tau=evaluate(cal,q);results[m][scope]=r;scores[m]['test'][scope]=q;thresholds_private[m]=tau
        row=dict(fold=f,results=results);dump(folder/'scores_private.json',scores);dump(folder/'thresholds_private.json',thresholds_private);dump(done,row);rows[str(u)]=row;print('TARGET_COMPLETE',u,flush=True)
    public={m:{scope:{point:{metric:float(np.mean([r['results'][m][scope][point][metric] for r in rows.values()])) for metric in ['tar','far','frr','hter','auc','eer']} for point in ['Cal_EER','Cal_FAR5']} for scope in ['unseen_attacker','all_seven_mixed']} for m in ['shape40','statistics240','student192','supplement232']};rng=np.random.RandomState(20261110);delta=np.array([r['results']['supplement232']['unseen_attacker']['Cal_EER']['hter']-r['results']['shape40']['unseen_attacker']['Cal_EER']['hter'] for r in rows.values()]);boots=np.array([delta[rng.choice(8,8,replace=True)].mean() for _ in range(5000)])
    dump(root/'results_public.json',dict(state='complete',users=8,seed=42,K=12,alpha=1,rho=.3,results=public,paired_vs_shape=dict(hter_delta=float(delta.mean()),identity_bootstrap95=np.quantile(boots,[.025,.975]).tolist(),improved_targets=int((delta<0).sum())),protocol='8fold4Base2Cal1target1unseen;actual matched texts across users;12register13Calqueries24Testqueries;cross-sentence;V4 only',channels='ch0 provisional vibration;ch1 provisional Audio;official mapping not confirmed',limits=['partial corpus subset','no independent same-sentence repeats per identity,not same-content train/test experiment','single seed,alpha/rho transferred unchanged fromVibraVox','all_seven_mixed includes Base/Cal,not fully unseen','teacher center<=2sec with valid short-clip lengths,student whole-utterance statistics;no framewise alignment claim']));print('ALL_COMPLETE',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--assets',required=True);main(p.parse_args())
