import numpy as np

from sklearn.metrics import roc_curve, roc_auc_score

def cutoff(scores, labels, rule):
    scores, labels = np.asarray(scores), np.asarray(labels, dtype=bool)
    if rule == 'quantile_gap':
        # Fixed before query: bridge the low genuine and upper impostor tails.
        return float((np.quantile(scores[labels], .10) + np.quantile(scores[~labels], .95)) / 2)
    values = np.unique(scores)
    if rule == 'strict5':
        candidates = np.r_[values, values[-1] + max(1., abs(values[-1]) * 1e-6)]
    else:
        assert rule == 'midpoint5'
        candidates = np.r_[values[0] - 1., (values[:-1] + values[1:]) / 2, values[-1] + 1.]
    accept = scores[None, :] >= candidates[:, None]
    far = accept[:, ~labels].mean(1)
    frr = (~accept[:, labels]).mean(1)
    feasible = np.flatnonzero(far <= .05 + 1e-12)
    index = min(feasible, key=lambda i: (frr[i], far[i], -candidates[i]))
    return float(candidates[index])

def measure(scores, labels, tau):
    labels = np.asarray(labels, dtype=bool)
    accept = scores >= tau
    fa, fr = int(accept[~labels].sum()), int((~accept[labels]).sum())
    ng, ni = int(labels.sum()), int((~labels).sum())
    fp, tp, _ = roc_curve(labels, scores, drop_intermediate=False)
    ix = np.argmin(np.abs(fp - (1 - tp)))
    return dict(far=fa/ni, frr=fr/ng, hter=(fa/ni+fr/ng)/2,
                eer=float((fp[ix]+1-tp[ix])/2), auc=float(roc_auc_score(labels, scores)),
                genuine_trials=ng, attack_trials=ni, false_accepts=fa, false_rejects=fr)
