"""Fixed-budget closed-set enrollment on cached hardware ACC only."""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_curve
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

root = Path(__file__).resolve().parent
with np.load(root / "paired_raw_acc_private.npz") as data:
    ids = data["sample_id"]
    user, word, repeat = np.array([s.split("_") for s in ids], dtype=int).T
    assert len(set(ids)) == len(ids) == 3200
    assert set(zip(user, word, repeat)) == {(u,w,r) for u in range(1,11)
                                            for w in range(40) for r in range(8)}
    spectrum = data["spectrum"].reshape(len(ids), 120, -1)
    shape = spectrum.mean(axis=-1)
    amplitude = data["amplitude"]
with np.load(root / "stats27_private.npz") as data:
    assert np.array_equal(ids, data["sample_id"])
    features = np.column_stack([shape, spectrum.std(axis=-1),
                                *(s.mean(axis=-1) for s in np.array_split(spectrum, 4, axis=-1)),
                                amplitude, data["features"]])

if __name__ == '__main__':
    rows = []
    folds = np.random.default_rng(1122).permutation(40).reshape(5, 8)
    with threadpool_limits(limits=1):
        for i, held in enumerate(folds):
            pool = folds[(i+1)%5]
            dev = folds[(i+2)%5][:4]
            fit = ~np.isin(word, np.r_[held,pool,dev])
            cal_same = np.isin(word,pool[:4]) & (repeat >= 6)
            cal_cross = np.isin(word,dev) & (repeat < 2)
            cal = cal_same | cal_cross
            query = np.isin(word,held) & (repeat < 4)
            x = StandardScaler().fit(features[fit]).transform(features)
            # Balanced identities and repeats in each fit word.
            centers = np.stack([x[word == w].mean(0) for w in np.unique(word[fit])])
            basis = np.linalg.svd(centers-centers.mean(0), full_matrices=False)[2][:3]
            for width, reps in [(4,6),(8,3)]:
                reg = np.isin(word,pool[:width]) & (repeat < reps)
                assert reg.sum()==240 and cal.sum()==160 and query.sum()==320
                assert (fit.astype(int)+reg+cal+query).max()==1
                for shrink in [False,True]:
                    view = normalize(x - 0.5*(x@basis.T)@basis if shrink else x)
                    for u in range(1,10):
                        y = user==u
                        svm = SVC(C=10,class_weight="balanced").fit(view[reg],y[reg])
                        scores = svm.decision_function(view[cal])
                        f,t,h = roc_curve(y[cal],scores,drop_intermediate=False)
                        threshold = h[np.argmin(abs(f+t-1))]
                        scores = svm.decision_function(view[query])
                        labels = y[query]
                        far = np.mean(scores[~labels]>=threshold)
                        frr = np.mean(scores[labels]<threshold)
                        f,t,_ = roc_curve(labels,scores,drop_intermediate=False)
                        j = np.argmin(abs(f+t-1))
                        rows.append(dict(fold=i+1,user=u,words=width,shrink=shrink,
                                         eer=(f[j]+1-t[j])/2,far=far,frr=frr,hter=(far+frr)/2))
    (root/"enrollment_word_ablation_public.json").write_text(json.dumps(rows,indent=2))
    for width in [4,8]:
        for shrink in [False,True]:
            subset=[r for r in rows if r["words"]==width and r["shrink"]==shrink]
            print(width,shrink,[round(100*np.mean([r[k] for r in subset]),2)
                                for k in ("eer","hter","far","frr")],flush=True)

