"""Audit whether fit-only Audio weights prioritize hard ACC impostors."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from sklearn.svm import SVC

OUT=Path('outputs/mei_audio_margin_residual_20261117')
USERS=[1,2,3,4,6,7,8,9]

def unit(x):return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-8)

def main():
    d=np.load('outputs/mei_crossword_representation_20261116/vectors_private.npz')
    t=np.load('outputs/mei_neural_verifier_20261116/redimnet2_train_embeddings_private.npz')
    assert np.array_equal(d['sample_id'],t['sample_id'])
    u,w=d['user'],d['word']
    fit=np.flatnonzero(w<18)
    assert np.array_equal(fit,t['indices']) and int(t['done'])==len(fit)
    assert not t['embedding'][w>=18].any()
    acc=d['E4'];audio=unit(t['embedding'])
    lookup={(i,j):np.flatnonzero((u==i)&(w==j)) for i in range(1,10) for j in range(30)}
    def center(z,part):
        return unit(np.stack([unit(z[np.concatenate([lookup[i,j][part] for j in range(18)])].mean(0)) for i in range(1,10)]))
    def weight(a,b):
        gap=np.diag(a@b.T)[:,None]-.5*(a@a.T+b@b.T)
        stable=np.abs(a@a.T-b@b.T)
        x=np.clip(gap,0,None)*np.exp(-5*stable)
        np.fill_diagonal(x,0)
        return x
    aa,ab=center(acc,slice(0,4)),center(acc,slice(4,8))
    ma,mb=center(audio,slice(0,4)),center(audio,slice(4,8))
    aw=weight(ma,mb);cw=weight(aa,ab)
    # Hardness: larger ACC target-prototype similarity indicates a harder source.
    acc_sim=.5*(aa@aa.T+ab@ab.T)
    audio_sim=.5*(ma@ma.T+mb@mb.T)
    fusion=json.loads(Path('outputs/mei_crossword_representation_20261116/fusion_selection_public.json').read_text(encoding='utf-8'))
    alpha=fusion['methods']['E4']['selected']['alpha']
    v=np.c_[d['raw'],alpha*acc]/np.sqrt(1+alpha**2)
    per=[];pairs=[]
    for owner in USERS:
        attacker=[i for i in USERS if i!=owner]
        k=owner-1;idx=np.array(attacker)-1
        # Independent diagnostic only: final-query scores are NOT used to choose any weights.
        attack_scores=np.zeros(len(attacker));n=0
        for fold in (0,1):
            for rw in range(24,30):
                ix=np.concatenate([lookup[i,rw][fold*4:fold*4+4] for i in USERS])
                clf=SVC(C=10,gamma='scale',class_weight='balanced').fit(v[ix],u[ix]==owner)
                for qw in range(24,30):
                    if qw==rw:continue
                    for p,i in enumerate(attacker):
                        q=lookup[i,qw][(1-fold)*4:(1-fold)*4+4]
                        attack_scores[p]+=clf.decision_function(v[q]).mean()
                    n+=1
        attack_scores/=n
        weight_values=aw[k,idx]
        acc_hard=acc_sim[k,idx]
        # Weight is proportional to Audio's separation, hence may favor easy impostors.
        train_rho=float(spearmanr(weight_values,acc_hard).statistic)
        final_rho=float(spearmanr(weight_values,attack_scores).statistic)
        acc_rho=float(spearmanr(cw[k,idx],attack_scores).statistic)
        top_weight=attacker[int(np.argmax(weight_values))]
        top_hard=attacker[int(np.argmax(attack_scores))]
        per.append(dict(user=f'U{owner}',train_acc_hardness_rho=train_rho,
                        final_attack_score_rho=final_rho,acc_weight_final_rho=acc_rho,
                        audio_top_matches_final_hardest=top_weight==top_hard,
                        audio_top_weight_rank_of_hardest=int(((-weight_values).argsort().argsort())[int(np.argmax(attack_scores))]+1),
                        audio_weight_zero_count=int(np.count_nonzero(weight_values==0))))
        for p,i in enumerate(attacker):
            pairs.append(dict(owner=f'U{owner}',attacker=f'U{i}',audio_weight=float(weight_values[p]),
                              acc_weight=float(cw[k,i-1]),train_acc_similarity=float(acc_hard[p]),
                              final_attack_score=float(attack_scores[p])))
    summary=dict(mean_train_acc_hardness_rho=float(np.mean([r['train_acc_hardness_rho'] for r in per])),
                 mean_final_attack_score_rho=float(np.mean([r['final_attack_score_rho'] for r in per])),
                 mean_acc_weight_final_rho=float(np.mean([r['acc_weight_final_rho'] for r in per])),
                 top1_matches=sum(r['audio_top_matches_final_hardest'] for r in per),targets=len(per),
                 zero_weight_pairs=sum(r['audio_weight_zero_count'] for r in per),total_pairs=len(pairs))
    result=dict(summary=summary,per_user=per)
    (OUT/'weight_audit_public.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'weight_audit_pairs_private.json').write_text(json.dumps(pairs,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
