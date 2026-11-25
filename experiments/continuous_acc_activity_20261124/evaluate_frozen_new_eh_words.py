"""Posthoc lexical-coverage follow-up, frozen models/thresholds, ACC only."""
import json,hashlib
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from train_phoneme_conditioned_identity import ROOT,BASE,OUT,PRIVATE,Net,embed,ARMS,TRAIN,QUERY
from train_acc_phoneme_locator import grid_descriptors
from audit_phoneme_identity import Decoder
from run_continuous import measure

def main():
    torch.set_num_threads(2)
    vocab=json.loads((ROOT/'outputs/vr261124_continuous_acc_activity/vocabulary_public.json').read_text(encoding='utf-8'))
    decoder=Decoder(loglevel='FATAL');lexicon={}
    for wo in range(35):
        text=vocab[str(wo)].lower()
        if wo in TRAIN+QUERY or ' ' in text or "'" in text:continue
        phones=decoder.lookup_word(text)
        if phones and 'EH' in phones.split():lexicon[wo]=dict(word=text,phones=phones)
    words=sorted(lexicon);assert words==[23,24,25]
    locks=json.loads((PRIVATE/'thresholds_private.json').read_text())
    hashes={str(seed)+'_'+arm:hashlib.sha256((PRIVATE/f'{seed}_{arm}_private.pt').read_bytes()).hexdigest() for seed in [42,43,44] for arm in ARMS}
    plan=dict(scope='posthocfollow-up motivatedbyNext;historicalwords notgloballyblind;noadditionaltrainingorcalibration',
      selection='allnotpreviouslyused Englishsinglewords whosefixed bundleddictionary firstpronunciationcontainsEH;selectedbyvocabonly beforeloadingACC/queryscores',
      words=words,lexicon=lexicon,model_sha256=hashes,threshold_sha256=hashlib.sha256((PRIVATE/'thresholds_private.json').read_bytes()).hexdigest(),
      no_Audio_opened=True,no_threshold_changes=True,no_model_changes=True)
    (OUT/'additional_EH_protocol_public.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf-8')
    d=np.load(BASE/'conditional_correspondence/coarse_pilot/raw_inputs_private.npz');u,w=d['user'],d['word'];raw=d['acc'];qi=np.flatnonzero(np.isin(w,words));assert len(qi)==240
    prepared=np.load(PRIVATE/'prepared_private.npz');base=np.load(PRIVATE/'inputs_private.npz')
    ri=np.flatnonzero(prepared['registration']);centers=np.arange(1.1,1.901,.02)
    xx=np.stack([grid_descriptors(raw[int(i)],centers) for i in qi]).astype('float32')
    xq=torch.tensor((xx-prepared['mean'])/prepared['scale']);xr=torch.tensor((base['x']-prepared['mean'])/prepared['scale'])
    rows=[];wordrows=[]
    for seed in [42,43,44]:
      for arm in ARMS:
        model=Net();model.load_state_dict(torch.load(PRIVATE/f'{seed}_{arm}_private.pt',weights_only=False)['model'])
        rz=embed(model,xr,ri);qz=embed(model,xq,np.arange(240));proto=np.stack([rz[prepared['u'][ri]==us].mean(0) for us in range(1,11)]);proto/=np.maximum(np.linalg.norm(proto,axis=1,keepdims=True),1e-12)
        for us in range(1,11):
          tau=next(v['tau'] for v in locks if v['seed']==seed and v['arm']==arm and v['user']==us);score=qz@proto[us-1]
          rows.append(dict(seed=seed,arm=arm,user=us,**measure(score,u[qi]==us,tau)))
          for wo in words:
            mask=w[qi]==wo;wordrows.append(dict(seed=seed,arm=arm,user=us,word=wo,**measure(score[mask],u[qi][mask]==us,tau)))
    keys=['eer','hter','far','frr','auc'];summary=[];users=[]
    for arm in ARMS:
      rr=[v for v in rows if v['arm']==arm];summary.append(dict(arm=arm,**{k:float(np.mean([v[k] for v in rr])) for k in keys}))
      for us in range(1,11):
        rr=[v for v in rows if v['arm']==arm and v['user']==us];users.append(dict(arm=arm,user=us,**{k:float(np.mean([v[k] for v in rr])) for k in keys}))
    for name,obj in [('additional_EH_summary_public.json',summary),('additional_EH_per_user_public.json',users),('additional_EH_per_seed_user_public.json',rows),('additional_EH_per_word_user_public.json',wordrows)]:
      (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
    assert hashes=={str(seed)+'_'+arm:hashlib.sha256((PRIVATE/f'{seed}_{arm}_private.pt').read_bytes()).hexdigest() for seed in [42,43,44] for arm in ARMS}
    assert plan['threshold_sha256']==hashlib.sha256((PRIVATE/'thresholds_private.json').read_bytes()).hexdigest()
    print(json.dumps(dict(words=words,summary=summary)))
if __name__=='__main__':
    with threadpool_limits(limits=2):main()
