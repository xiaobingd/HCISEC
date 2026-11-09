"""ABCS43 V4: 23 Base, 8 Cal, 12 unseen targets, immutable shape40 branch."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import argparse,json,sys,csv
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.preprocessing import StandardScaler
from run_rkd01_vibravox import Student,norm,supcon,encode,thresholds,metrics
from run_registration_rkd02 import adapt_student,dump

BASE=list(range(23));CAL=list(range(23,31));TEST=list(range(31,43))
def reg(d,u):return np.flatnonzero((d['user']==u)&(d['role']=='registration'))
def replay(d,u):return np.flatnonzero((d['user']==u)&d['replay'])
def base_student(d,x,path):
    net=Student()
    if path.exists():
        ck=torch.load(path,map_location='cpu',weights_only=False);assert ck['base_users']==BASE;net.load_state_dict(ck['state']);return net
    torch.manual_seed(42);rng=np.random.RandomState(42);net=Student();head=nn.Linear(192,23,bias=False);opt=torch.optim.AdamW(list(net.parameters())+list(head.parameters()),lr=1e-4,weight_decay=1e-4);lookup={u:{int(d['text_index'][i]):int(i) for i in np.flatnonzero((d['user']==u)&(d['role']=='fit'))} for u in BASE}
    for step in range(1,701):
        uu=rng.choice(BASE,8,replace=False);batch=[]
        for v,w in zip(uu[::2],uu[1::2]):
            shared=sorted(set(lookup[v])&set(lookup[w]));assert len(shared)>=4;tt=rng.choice(shared,4,replace=False);batch.extend(lookup[v][t] for t in tt);batch.extend(lookup[w][t] for t in tt)
        ii=np.array(batch);y=torch.tensor(d['user'][ii],dtype=torch.long);z=net(torch.tensor(x[ii]));loss=F.cross_entropy(16*F.linear(z,F.normalize(head.weight,dim=1)),y)+.1*supcon(z,y);opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(list(net.parameters())+list(head.parameters()),1);opt.step()
        if step%100==0:print('BASE_TRAIN',step,float(loss.detach()),flush=True)
    torch.save(dict(state=net.state_dict(),base_users=BASE,steps=700,seed=42),path);return net
def teacher(d,u,folder,model,original,cache,slots):
    path=folder/'teacher_targets_private.npz'
    if path.exists():t=dict(np.load(path));return t['indices'],t['adapted']
    torch.manual_seed(4242+u);rng=np.random.RandomState(4242+u);pi=np.concatenate([replay(d,v) for v in BASE]);ix=np.r_[pi,reg(d,u)];assert len(pi)==92 and len(ix)==104;assert all((d['user'][i] in BASE and d['role'][i]=='fit') or (d['user'][i]==u and d['role'][i]=='registration') for i in ix);emb=model.mods.embedding_model;emb.load_state_dict(original);emb.eval()
    for p in emb.parameters():p.requires_grad_(True)
    device=next(emb.parameters()).device;jj=torch.tensor([slots[int(i)] for i in ix]);feat=cache['features'][jj].to(device);lens=cache['lengths'][jj].to(device);yall=torch.tensor([int(v) if v<23 else 23 for v in d['user'][ix]],device=device)
    def forward(j):return F.normalize(emb(feat[j],lens[j]).squeeze(1),dim=1)
    with torch.no_grad():frozen=forward(torch.arange(len(ix),device=device)).detach();proto=F.normalize(torch.stack([frozen[yall==j].mean(0) for j in range(24)]),dim=1)
    head=nn.Linear(192,24,bias=False).to(device)
    with torch.no_grad():head.weight.copy_(proto)
    opt=torch.optim.AdamW([dict(params=emb.parameters(),lr=1e-6),dict(params=head.parameters(),lr=1e-4)],weight_decay=1e-4);watch=next(p for p in emb.parameters() if p.ndim>1);initial=watch.detach().clone()
    for step in range(30):
        uu=rng.choice(BASE,4,replace=False);batch=np.r_[rng.choice(np.flatnonzero(d['user'][ix]==u),4,replace=False),np.concatenate([rng.choice(np.flatnonzero(d['user'][ix]==v),2,replace=False) for v in uu])];j=torch.tensor(batch,device=device);z=forward(j);y=yall[j];loss=F.cross_entropy(16*F.linear(z,F.normalize(head.weight,dim=1)),y)+.1*supcon(z.cpu(),y.cpu()).to(device)+.1*(1-F.cosine_similarity(z[4:],frozen[j[4:]],dim=1)).mean();opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(list(emb.parameters())+list(head.parameters()),1);opt.step()
    delta=float((watch-initial).abs().max());assert delta>0
    for p in emb.parameters():p.requires_grad_(False)
    with torch.no_grad():adapted=forward(torch.arange(len(ix),device=device)).cpu().numpy()
    np.savez_compressed(path,indices=ix,frozen=frozen.cpu().numpy(),adapted=adapted);torch.save(emb.state_dict(),folder/'ecapa_private.pt');dump(folder/'teacher_audit_private.json',dict(owner=u,base=BASE,registration=12,base_replay=92,steps=30,backbone_update=delta,query_audio_model_forwards=0));print('TEACHER_COMPLETE',u,flush=True);return ix,adapted
def score(z,d,u,group,role):
    en=reg(d,u);q=np.flatnonzero(np.isin(d['user'],group)&(d['role']==role));assert len(en)==12 and not set(en)&set(q);return dict(scores=(z[q]@norm(z[en].mean(0))).tolist(),labels=(d['user'][q]==u).tolist())
def summarize(rows):
    results={};individual=[];thresholds_private={}
    for method in ['shape40','statistics240','student192','supplement232']:
        cal=[rows[u]['methods'][method] for u in CAL];cs=np.concatenate([v['scores'] for v in cal]);cy=np.concatenate([v['labels'] for v in cal]);tt=thresholds(cs,cy);assert float((cs[~cy]>=tt['Cal_FAR5']).mean())<=.05;thresholds_private[method]=tt;results[method]={}
        for point,tau in tt.items():
            scores=np.concatenate([rows[u]['methods'][method]['scores'] for u in TEST]);ys=np.concatenate([rows[u]['methods'][method]['labels'] for u in TEST]);r=metrics(scores,ys,tau);pp=[]
            for group,role in [(CAL,'Cal'),(TEST,'Test')]:
                for j,u in enumerate(group,1):
                    q=rows[u]['methods'][method];v=metrics(np.array(q['scores']),np.array(q['labels']),tau);individual.append(dict(user=f'ABCS43-{role}-{j:02}',role=role,method=method,point=point,**v))
                    if role=='Test':pp.append(v)
            r.update(macro_auc=float(np.mean([v['auc'] for v in pp])),macro_eer=float(np.mean([v['eer'] for v in pp])),worst_target_far=max(v['far'] for v in pp),lowest_target_tar=min(v['tar'] for v in pp));results[method][point]=r
    return results,individual,thresholds_private
def main(a):
    torch.set_num_threads(1);root=Path(a.root);d=dict(np.load(root/'inputs_private.npz'));audio=dict(np.load(root/'audio_fit_register_private.npz'));assert d['x'].shape==(2445,246) and len(audio['indices'])==332;assert set(d['user'][d['group']=='base'])==set(BASE) and set(d['user'][d['group']=='cal'])==set(CAL) and set(d['user'][d['group']=='test'])==set(TEST);assert not np.isin(d['role'][audio['indices']],['query','calibration','develop']).any();groups=[set(d['text_index'][d['role']==r]) for r in ['fit','develop','registration','calibration','query']];assert all(not groups[i]&groups[j] for i in range(5) for j in range(i));fit=(d['group']=='base')&(d['role']=='fit');x=StandardScaler().fit(d['x'][fit]).transform(d['x']).astype('float32');shape=norm(x[:,:40]);stats=norm(StandardScaler().fit(d['statistics'][fit]).transform(d['statistics']));initial=base_student(d,x,root/'base_student_private.pt');sys.path.insert(0,'/root');from run_ecapa_pilot import teacher_load
    for name in ['model','deps']:
        p=root/name
        if not p.exists():p.symlink_to(Path(a.assets)/name,target_is_directory=True)
    model=teacher_load(root);original={k:v.detach().cpu().clone() for k,v in model.mods.embedding_model.state_dict().items()};official=torch.load(root/'model'/'embedding_model.ckpt',map_location='cpu',weights_only=False);assert set(original)==set(official) and all(torch.equal(original[k],official[k]) for k in original);del official;device=next(model.parameters()).device;cachepath=root/'audio_frontend_private.pt'
    if cachepath.exists():cache=torch.load(cachepath,map_location='cpu',weights_only=False)
    else:
        ff=[]
        with torch.no_grad():
            for s in range(0,len(audio['waves']),16):
                w=torch.tensor(audio['waves'][s:s+16],device=device);lens=torch.tensor(audio['lengths'][s:s+16],device=device);ff.append(model.mods.mean_var_norm(model.mods.compute_features(w),lens).cpu())
        cache=dict(features=torch.cat(ff),lengths=torch.tensor(audio['lengths']),indices=audio['indices']);torch.save(cache,cachepath)
    assert np.array_equal(cache['indices'],audio['indices']);slots={int(i):j for j,i in enumerate(audio['indices'])};rows={}
    for u in CAL+TEST:
        folder=root/f'owner{u}';folder.mkdir(exist_ok=True);done=folder/'scores_private.json'
        if done.exists():rows[u]=json.loads(done.read_text());continue
        ix,t=teacher(d,u,folder,model,original,cache,slots);net,trace=adapt_student(d,x,initial,u,BASE,ix,t,'V4',.3);torch.save(net.state_dict(),folder/'student_private.pt');dump(folder/'student_audit_private.json',dict(training_indices=ix.tolist(),trace=trace,steps=100,rho=.3));group,role=(CAL,'calibration') if u in CAL else (TEST,'query');idx=np.flatnonzero(((d['user']==u)&(d['role']=='registration'))|(np.isin(d['user'],group)&(d['role']==role)));z=np.zeros((len(x),192),dtype='float32');z[idx]=encode(net,x[idx]);joint=np.concatenate([shape,z],1)/np.sqrt(2);assert np.allclose(joint[idx,:40],shape[idx]/np.sqrt(2),atol=1e-7);row=dict(owner=u,methods={m:score(v,d,u,group,role) for m,v in [('shape40',shape),('statistics240',stats),('student192',z),('supplement232',joint)]});dump(done,row);rows[u]=row;print('OWNER_COMPLETE',u,'Cal' if u in CAL else 'Test',flush=True)
    results,individual,tt=summarize(rows);dump(root/'thresholds_private.json',tt);dump(root/'per_user_public.json',individual);rng=np.random.RandomState(20261110);paired={}
    for control in ['shape40','student192']:
        delta=np.array([next(v['hter'] for v in individual if v['user']==f'ABCS43-Test-{j:02}' and v['method']=='supplement232' and v['point']=='Cal_EER')-next(v['hter'] for v in individual if v['user']==f'ABCS43-Test-{j:02}' and v['method']==control and v['point']=='Cal_EER') for j in range(1,13)]);boots=np.array([delta[rng.choice(12,12,replace=True)].mean() for _ in range(5000)]);paired[control]=dict(hter_delta=float(delta.mean()),identity_bootstrap95=np.quantile(boots,[.025,.975]).tolist(),improved_targets=int((delta<0).sum()))
    dump(root/'results_public.json',dict(state='complete',total_identities=43,base=23,calibration=8,test=12,K=12,seed=42,alpha=1,rho=.3,results=results,paired_vs=paired,protocol='fixed hash23/8/12;shared actual registration12/calquery24/testquery24,all text roles disjoint;Test each24g264att;V4 only;shape40+student192',channels='ch0 provisional vibration/ch1 provisional Audio,physical mapping unresolved',limits=['partial source archive;43 available users not59','single seed,explored corpus','no independent same-sentence repeated recording','no verified redon/crossdevice sessions']));print('ALL_COMPLETE',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--assets',required=True);main(p.parse_args())
