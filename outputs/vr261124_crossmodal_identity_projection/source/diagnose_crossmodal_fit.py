"""Post-hoc fitting diagnosis only; never changes projections, alpha or thresholds."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from run_crossmodal_identity_projection import Pair,unit
def main(root):
    torch.set_num_threads(2);out=root/'conditional_correspondence/crossmodal_identity_projection'
    p=json.loads((out/'protocol_public.json').read_text());audit=json.loads((out/'episode_audit_public.json').read_text())
    d=np.load(root/'phrase_auth/frontend_private.npz');cache=np.load(root/'conditional_correspondence/audio_conditioned_verifier/fit_ecapa_private.npz')
    idx=cache['indices'];u,w=d['user'][idx],d['word'][idx];audio=cache['embedding'];assert np.array_equal(cache['sample_id'],d['sample_id'][idx])
    with np.load(root/'conditional_correspondence/stage3/views_private.npz') as zf:views={seed:zf[f'{seed}_A_ACC'][idx] for seed in (42,43,44)}
    rows=[]
    for a in audit:
      seed,held,dev=a['seed'],a['held_target'],a['inner_development_user'];net=Pair();net.load_state_dict(torch.load(out/f'crossmodal_seed{seed}_held{held}_private.pt',map_location='cpu',weights_only=True));net.eval()
      users=[us for us in range(1,11) if us not in (held,dev)]
      scopes=[('train_identity_fit_word_disjoint',users,users,p['train_words'][:6],p['train_words'][6:]),
        ('held_identity_fit_word_disjoint',list(range(1,11)),[held],p['train_words'][:6],p['train_words'][6:]),
        ('inner_dev_identity_inner_words',[us for us in range(1,11) if us!=held],[dev],p['inner_words'][:2],p['inner_words'][2:])]
      with torch.inference_mode():
        az=net.audio(torch.tensor(audio)).numpy();xz=net.acc(torch.tensor(views[seed])).numpy()
      for name,cohort,queryusers,supportwords,querywords in scopes:
        assert not set(supportwords)&set(querywords)
        proto=unit(np.stack([az[(u==us)&np.isin(w,supportwords)].mean(0) for us in cohort]))
        qi=np.flatnonzero(np.isin(u,queryusers)&np.isin(w,querywords));score=xz[qi]@proto.T
        prediction=np.array(cohort)[score.argmax(1)];accuracy=float(np.mean(prediction==u[qi]))
        rows.append(dict(seed=seed,user=held,scope=name,top1=accuracy,candidate_users=len(cohort),queries=len(qi),support_records_per_user=8*len(supportwords)))
    public=dict(posthoc=True,used_for_model_selection=False,rows=rows,macro=[dict(scope=name,top1=float(np.mean([r['top1'] for r in rows if r['scope']==name]))) for name in sorted({r['scope'] for r in rows})])
    (out/'fit_diagnostics_public.json').write_text(json.dumps(public,indent=2));print(json.dumps(public['macro']))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
