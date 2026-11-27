"""Held-development-word phone learning diagnostic; no final query data or Audio."""
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from run import ROOT,SOURCE,PRIV,OUT,DEV,CONFIGS,inputs,model_from,write
import json
from conditional_loss import targets,loss

def main():
    torch.set_num_threads(2);data,x,valid,u,w,_=inputs('development');cache=np.load(SOURCE/'teacher_private.npz');ii=cache['indices']
    choose=np.flatnonzero(np.isin(w[ii],DEV));indices=ii[choose];assert np.isin(w[indices],DEV).all()
    q,weight,h=targets(cache['posterior'][choose],cache['valid'][choose]);assert weight.sum()>0
    assert int(cache['blank_id'])==0 and q.shape[-1]==391
    target=torch.tensor(q);ww=torch.tensor(weight);hh=torch.tensor(h);rows=[]
    vocab=json.loads((ROOT/'.runtime/models/wav2vec2-lv-60-espeak-cv-ft/vocab.json').read_text(encoding='utf-8'));symbols={int(v):k for k,v in vocab.items()}
    def modes(classes):
        totals=torch.bincount(classes.flatten(),weights=ww.flatten(),minlength=391);top=totals.topk(5)
        return [dict(token_id=int(k)+1,symbol=symbols[int(k)+1],weight_fraction=float(v/ww.sum())) for k,v in zip(top.indices,top.values)]
    for code in CONFIGS:
        model,_=model_from(PRIV/f'development_42_{code}_private.pt',12);logits=[]
        with torch.inference_mode():
            for batch in np.array_split(indices,5):logits.append(model(x[batch],valid[batch])[1])
        logits=torch.cat(logits);pred=logits[...,1:].argmax(-1)
        agreement=((pred==target.argmax(-1))*ww).sum()/ww.sum()
        soft=(target.gather(-1,pred[...,None]).squeeze(-1)*ww).sum()/ww.sum()
        rows.append(dict(arm=code,conditional_weighted_KL=float(loss(logits,target,ww,hh)),
          teacher_argmax_agreement=float(agreement),teacher_probability_at_student_argmax=float(soft),
          windows=len(indices),positive_weight_frames=int((ww>0).sum()),predicted_phone_modes=modes(pred),teacher_phone_modes=modes(target.argmax(-1))))
    write('phoneme_diagnostic_public.json',dict(status='COMPLETE',stage='400step12wordmodels;held4DEVwords',
      labels='teacher posterior agreement,not manually labeled phoneme accuracy',rows=rows,
      query_words_used=0,query_Audio_reads=0,used_for_hyperparameter_selection=False))
    print('CONDITIONAL_PHONE_DIAGNOSTIC',rows,flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
