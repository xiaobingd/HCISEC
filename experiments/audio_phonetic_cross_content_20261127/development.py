"""Independent-word phone diagnostics and a pre-query immutable configuration lock."""
import json,hashlib,time
import numpy as np
import torch
from torch.nn import functional as F
from common import Student
from train import inputs,protocol,digest,ARMS
from prepare_acc import PRIVATE,OUT,DEV

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def run():
    torch.set_num_threads(2)
    data,x,valid,u,w,words,fit=inputs('development')
    cache=np.load(PRIVATE/'teacher_private.npz');indices=cache['indices'];pick=np.flatnonzero(np.isin(w[indices],DEV));ii=indices[pick]
    target=cache['posterior'][pick];mask=cache['valid'][pick]&valid[ii].numpy()
    assert not np.isin(w[ii],words).any() and mask.any()
    q=torch.tensor(target);mass=q[:,:,1:].sum(-1);phone=q[:,:,1:]/mass[:,:,None].clamp_min(1e-12)
    rows=[]
    for arm in ARMS:
        path=PRIVATE/f'development_42_{arm}_private.pt';saved=torch.load(path,weights_only=True)
        assert saved['steps']==200 and saved['protocol_sha']==digest(protocol())
        model=Student(392,12);model.load_state_dict(saved['model']);model.eval();pred=[]
        with torch.inference_mode():
            for batch in np.array_split(ii,max(1,int(np.ceil(len(ii)/40)))):
                pred.append(model(x[batch],valid[batch])[1].log_softmax(-1))
        logp=torch.cat(pred);full=F.kl_div(logp,q,reduction='none').sum(-1)
        conditional=F.log_softmax(logp[:,:,1:],-1);nonblank=F.kl_div(conditional,phone,reduction='none').sum(-1)
        effective=mass*torch.tensor(mask);confident=torch.tensor(mask)&(mass>.5)
        accuracy=((conditional.argmax(-1)==phone.argmax(-1))*effective).sum()/effective.sum().clamp_min(1e-12)
        row=dict(arm=arm,checkpoint_sha256=sha(path),dev_windows=len(ii),valid_frames=int(mask.sum()),
          full_posterior_KL=float(full[mask].mean()),teacher_nonblank_mass=float(mass[mask].mean()),
          nonblank_mass_weighted_conditional_KL=float((nonblank*effective).sum()/effective.sum().clamp_min(1e-12)),
          nonblank_mass_weighted_top1=float(accuracy),teacher_nonblank_majority_frames=int(confident.sum()),
          nonblank_majority_top1=float((conditional.argmax(-1)==phone.argmax(-1))[confident].float().mean()) if confident.any() else None,
          final_training=saved['trace'][-1],gradient=saved['gradient'])
        assert np.isfinite(row['full_posterior_KL']) and all(np.isfinite(v) for r in row['gradient'] for k,v in r.items())
        rows.append(row)
    result=dict(status='LOCKED',locked_at_unix=time.time(),protocol_sha256=digest(protocol()),
      teacher_cache_sha256=sha(PRIVATE/'teacher_private.npz'),teacher_audit_sha256=sha(OUT/'teacher_audit_public.json'),
      optimized_words=words,independent_development_words=DEV,models=rows,
      selected_settings=dict(lambda_phone=.1,lambda_content=.05,final_steps=1000,seeds=[42,43,44],registration_K=12),
      decision='Retain all predeclared settings: finite losses/gradients. No lambda scan, authentication queries or calibration used. Blank is not interpreted as silence.',
      query_scores_read=0,calibration_scores_read=0)
    path=OUT/'development_lock_public.json'
    if path.exists():
        old=json.loads(path.read_text(encoding='utf-8'));assert old['protocol_sha256']==result['protocol_sha256'];print('EXISTING_DEVELOPMENT_LOCK_RETAINED');return
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print('DEVELOPMENT_LOCKED',[(r['arm'],round(r['full_posterior_KL'],4),round(r['nonblank_mass_weighted_conditional_KL'],4)) for r in rows],flush=True)

if __name__=='__main__':run()
