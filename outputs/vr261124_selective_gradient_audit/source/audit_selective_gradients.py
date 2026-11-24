"""Fit-only Audio relation/gradient audit; virtual updates are always rolled back.

No final query or calibration Audio, features, thresholds or authentication scores.
The historical checkpoint lacks its CE classifier: exact saved-model SupCon
component is the identity reference, not a reconstructed historical full CE loss.
"""
import argparse,copy,hashlib,importlib.util,json,sys
from pathlib import Path
import numpy as np
from scipy import signal
from scipy.stats import spearmanr
import torch
from torch.nn import functional as F

class Network(torch.nn.Module):
    """Exact historical ACC architecture, copied to avoid training-module side effects."""
    def __init__(self):
        super().__init__()
        self.body=torch.nn.Sequential(torch.nn.Conv1d(147,32,3,padding=1),torch.nn.GroupNorm(4,32),torch.nn.GELU(),
            torch.nn.Conv1d(32,32,3,padding=1),torch.nn.GroupNorm(4,32),torch.nn.GELU())
        self.identity=torch.nn.Linear(64,192)
        self.local=torch.nn.Conv1d(32,26,1)
    def forward(self,x):
        h=self.body(x)
        z=F.normalize(self.identity(torch.cat([h.mean(2),h.std(2)],1)),dim=1)
        return z,F.adaptive_avg_pool1d(self.local(h),12)

def unit(x):return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-8)
def cosine(g,h):
    g=torch.cat([v.reshape(-1) for v in g]);h=torch.cat([v.reshape(-1) for v in h])
    return float(F.cosine_similarity(g[None],h[None]))
def flatten(g):return torch.cat([v.reshape(-1) for v in g])
def supcon(z,y,words):
    logits=z@z.T/.1;pos=(y[:,None]==y[None,:])&(words[:,None]!=words[None,:])
    valid=(y[:,None]!=y[None,:])|pos
    assert pos.any(1).all()
    return -((logits-torch.logsumexp(logits.masked_fill(~valid,-torch.inf),1,keepdim=True))*pos).sum(1).div(pos.sum(1)).mean()
def local_kl(scores,teacher):
    return F.kl_div(F.log_softmax(scores/.1,dim=-1),F.softmax(teacher.detach()/.1,dim=-1),reduction='none').sum(-1)
def margin(scores):return (scores.diag()-scores.masked_fill(torch.eye(len(scores),dtype=torch.bool),-torch.inf).max(1).values).mean()
def main(root,episodes):
    torch.set_num_threads(2)
    out=root/'conditional_correspondence/selective_gradient_audit';out.mkdir(exist_ok=True)
    d=np.load(root/'conditional_correspondence/coarse_pilot/raw_inputs_private.npz')
    fit=d['fit'];idx=np.flatnonzero(fit);u,w,r,ids=[d[k][idx] for k in ('user','word','repeat','sample_id')]
    raw=d['acc'][idx];assert raw.shape==(1280,3,3000)
    cache=np.load(root/'conditional_correspondence/audio_conditioned_verifier/fit_ecapa_private.npz')
    assert np.array_equal(cache['indices'],idx) and np.array_equal(cache['sample_id'],ids)
    audio=cache['embedding'];assert audio.shape==(1280,192)
    fr,tt,st=signal.stft(raw,1000,nperseg=100,noverlap=80,nfft=100,boundary=None,padded=False)
    x=np.log(np.maximum(abs(st[:,:,1:50])**2,1e-20)).reshape(1280,147,len(tt)).astype('float32');del st,raw
    x=torch.tensor((x-x.mean((0,2),keepdims=True))/np.maximum(x.std((0,2),keepdims=True),1e-5))
    lookup={(int(a),int(b),int(c)):i for i,(a,b,c) in enumerate(zip(u,w,r))}
    words=sorted(map(int,set(w)));assert len(words)==16
    trainwords=words[:12];checkwords=words[12:]
    protocol=dict(records=1280,users=10,seeds=[42,43,44],episodes_per_seed=episodes,
      teacher='frozen official ECAPA192,fit-only cache',student='historical pure ACC A_ACC192 checkpoint,never saved over',
      identity_gradient='exact historical cross-word SupCon component only;historical CE classifier not saved,full CE+SupCon gradient unavailable',
      relation_loss='temperature .1 KL over genuine cross-word prototype and 3 ACC-hard impostor prototypes;teacher ranking soft target,not embedding regression',
      selection='ACC top3 hard identities;teacher positive margin stable under even/odd repeat Audio views;per-row BODY gradient cosine with global ACC SupCon >0',
      virtual_update='SGD lr.001,Audio gradient capped to .1 identity-gradient norm;rollback every episode;compare id-only,uniform-correct,uniform-wrong,selected-correct,selected-wrong;same correct-selection mask for wrong control',
      optimization_words=trainwords,virtual_check_words=checkwords,
      scope='all16words were seen by historical encoder;disjoint-word virtual check is an in-sample diagnostic,NOT unseen-word generalization or authentication gain',
      controls='wrong teacher globally permutes identity prototypes,query labels and ACC unchanged;selected-wrong shares correct-mask',
      forbidden='no REG/CAL/finalquery Audio or scores;no authentication thresholds;no user filtering;no new trained encoder committed')
    (out/'protocol_public.json').write_text(json.dumps(protocol,indent=2))
    ranking=[];gradients=[];steps=[];checks=[];baseline_errors=[]
    with np.load(root/'conditional_correspondence/stage3/views_private.npz') as zfile:
      references={s:zfile[f'{s}_A_ACC'][idx].copy() for s in (42,43,44)}
    y=torch.tensor(np.repeat(np.arange(10),2));batchwords=None
    for seed in (42,43,44):
      net=Network();cp=root/f'conditional_correspondence/stage3/A_ACC_{seed}_private.pt'
      original=cp.read_bytes();net.load_state_dict(torch.load(cp,map_location='cpu',weights_only=True));net.eval()
      params=list(net.body.parameters())+list(net.identity.parameters());body_count=len(list(net.body.parameters()))
      with torch.inference_mode():z=np.concatenate([net(x[i:i+80])[0].numpy() for i in range(0,1280,80)])
      err=float(abs(z-references[seed]).max());assert err<1e-4;baseline_errors.append(err)
      rng=np.random.default_rng(8000+seed)
      for ep in range(episodes):
        wa,wb=rng.choice(trainwords,2,replace=False);ra,rb=rng.integers(0,8,2)
        batch=np.array([[lookup[us,int(wa),int(ra)],lookup[us,int(wb),int(rb)]] for us in range(1,11)]).ravel()
        bw=torch.tensor(w[batch]);zs=net(x[batch])[0];ss=zs[::2]@zs[1::2].T
        aid=torch.tensor(audio[batch]);ts=aid[::2]@aid[1::2].T
        # Stability uses separate repeat parities with the same support/query words.
        ta=[]
        for parity in (0,1):
          pa=np.stack([unit(audio[(u==us)&(w==wa)&(r%2==parity)].mean(0)) for us in range(1,11)])
          pb=np.stack([unit(audio[(u==us)&(w==wb)&(r%2==parity)].mean(0)) for us in range(1,11)])
          ta.append(pa@pb.T)
        iid=torch.arange(10);hard=ss.detach().masked_fill(torch.eye(10,dtype=torch.bool),-torch.inf).topk(3,dim=1).indices
        choices=torch.cat([iid[:,None],hard],1);student_local=ss.gather(1,choices);teacher_local=ts.gather(1,choices)
        loss_id=supcon(zs,y,bw);gi=torch.autograd.grad(loss_id,params,retain_graph=True);gi_body=gi[:body_count]
        perm=rng.permutation(10)
        while np.any(perm==np.arange(10)):perm=rng.permutation(10)
        wrong=ts[perm][:,perm].gather(1,choices)
        lk=local_kl(student_local,teacher_local);lw=local_kl(student_local,wrong)
        gate=[]
        for owner in range(10):
          gs=torch.autograd.grad(lk[owner],params,retain_graph=True);alignment=cosine(gi_body,gs[:body_count])
          gw=torch.autograd.grad(lw[owner],params,retain_graph=True)
          negatives=hard[owner].numpy();stable=all(t[owner,owner]>max(t[owner,negatives]) for t in ta)
          choose=bool(stable and alignment>0);gate.append(choose)
          non=np.arange(10)!=owner;rho=float(spearmanr(ss[owner].detach().numpy()[non],ts[owner].numpy()[non]).statistic)
          topa=set(np.argsort(-ts[owner].numpy()[non])[:3]);topx=set(np.argsort(-ss[owner].detach().numpy()[non])[:3])
          ranking.append(dict(seed=seed,user=owner+1,episode=ep,spearman=rho,top3_overlap=len(topa&topx)/3,
              acc_margin=float(ss[owner,owner]-ss[owner,hard[owner,0]]),audio_margin_on_acc_hard=float(ts[owner,owner]-ts[owner,hard[owner,0]])))
          gradients.append(dict(seed=seed,user=owner+1,episode=ep,body_cosine=alignment,wrong_body_cosine=cosine(gi_body,gw[:body_count]),
              teacher_stable=bool(stable),selected=choose))
        mask=torch.tensor(gate,dtype=torch.float32)
        losses={'uniform_correct':lk.mean(),'uniform_wrong':lw.mean(),'selected_correct':(lk*mask).sum()/mask.sum().clamp_min(1),
                'selected_wrong':(lw*mask).sum()/mask.sum().clamp_min(1)}
        aux={name:torch.autograd.grad(loss,params,retain_graph=True) for name,loss in losses.items()}
        # Independent recordings and word IDs, all still within historical fit.
        va,vb=rng.choice(checkwords,2,replace=False);vr=rng.integers(0,8,2)
        check=np.array([[lookup[us,int(va),int(vr[0])],lookup[us,int(vb),int(vr[1])]] for us in range(1,11)]).ravel()
        assert not np.intersect1d(batch,check).size and not np.intersect1d(w[batch],w[check]).size
        cy=torch.tensor(w[check]);snapshot=[p.detach().clone() for p in params]
        with torch.inference_mode():
          zv=net(x[check])[0];pre=float(supcon(zv,y,cy));pre_margin=float(margin(zv[::2]@zv[1::2].T))
        for method in ['identity_only',*aux]:
          gaux=aux.get(method);scale=0. if gaux is None else min(10.,.1*float(flatten(gi).norm())/max(float(flatten(gaux).norm()),1e-12))
          with torch.no_grad():
            for i,p in enumerate(params):p.copy_(snapshot[i]-.001*(gi[i]+(0 if gaux is None else scale*gaux[i])))
            zv=net(x[check])[0];value=float(supcon(zv,y,cy));vscore=zv[::2]@zv[1::2].T
            for owner in range(10):
              neg=vscore[owner].clone();neg[owner]=-torch.inf
              steps.append(dict(seed=seed,episode=ep,user=owner+1,method=method,check_loss=value,check_margin=float(vscore[owner,owner]-neg.max()),pre_loss=pre,pre_margin=pre_margin,selected_rows=int(mask.sum())))
            for p,start in zip(params,snapshot):p.copy_(start)
        checks.append(dict(seed=seed,episode=ep,support_query_word_disjoint=True))
      assert original==cp.read_bytes()
      print('GRADIENT_AUDIT_SEED_DONE',seed,flush=True)
    # Only anonymous target-level aggregates are public; no pair/sample scores.
    aggregates=[]
    for us in range(1,11):
      rr=[a for a in ranking if a['user']==us];gg=[a for a in gradients if a['user']==us]
      aggregates.append(dict(user=us,ranking_spearman=float(np.mean([a['spearman'] for a in rr])),top3_overlap=float(np.mean([a['top3_overlap'] for a in rr])),
        audio_beats_acc_hard_rate=float(np.mean([a['audio_margin_on_acc_hard']>0 for a in rr])),gradient_positive_rate=float(np.mean([a['body_cosine']>0 for a in gg])),
        wrong_gradient_positive_rate=float(np.mean([a['wrong_body_cosine']>0 for a in gg])),gradient_cosine=float(np.mean([a['body_cosine'] for a in gg])),
        wrong_gradient_cosine=float(np.mean([a['wrong_body_cosine'] for a in gg])),stable_rate=float(np.mean([a['teacher_stable'] for a in gg])),selected_rate=float(np.mean([a['selected'] for a in gg]))))
    virtual=[]
    for method in ['uniform_correct','uniform_wrong','selected_correct','selected_wrong']:
      per=[]
      for us in range(1,11):
        diffs=[];md=[]
        for row in steps:
          if row['user']!=us or row['method']!=method:continue
          ref=next(v for v in steps if v['seed']==row['seed'] and v['episode']==row['episode'] and v['user']==us and v['method']=='identity_only')
          diffs.append(row['check_loss']-ref['check_loss']);md.append(row['check_margin']-ref['check_margin'])
        per.append(dict(user=us,check_loss_delta=float(np.mean(diffs)),check_margin_delta=float(np.mean(md)),margin_improvement_rate=float(np.mean(np.array(md)>0))))
      virtual.append(dict(method=method,per_user=per,mean_check_loss_delta=float(np.mean([a['check_loss_delta'] for a in per])),mean_check_margin_delta=float(np.mean([a['check_margin_delta'] for a in per]))))
    result=dict(per_user=aggregates,virtual_updates=virtual,verification=dict(status='PASS',fit_only=True,no_final_query_scoring=True,only_fit_features_used=True,encoder_rollback=True,checkpoints_byte_unchanged=True,baseline_max_error=max(baseline_errors),gradient_rows=len(gradients),virtual_workpoints=len(steps),CE_classifier_missing=True))
    (out/'summary_public.json').write_text(json.dumps(result,indent=2))
    (out/'diagnostics_private.json').write_text(json.dumps(dict(gradients=gradients,ranking=ranking,virtual=steps)))
    print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--episodes',type=int,default=20);a=p.parse_args();main(a.root,a.episodes)
