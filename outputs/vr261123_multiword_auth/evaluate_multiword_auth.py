"""Frozen ACC multiword trials, registration templates and bounded quality retry."""
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.svm import SVC
from torch import nn
from torch.nn import functional as F
from threadpoolctl import threadpool_limits

root = Path("/root")
out = root / "full_run_10x40_raw_acc" / "multiword_auth"
out.mkdir(exist_ok=True)
torch.set_num_threads(2)
d = np.load(root / "paired_raw_acc_private.npz")
stats = np.load(root / "stats27_private.npz")
assert np.array_equal(d["sample_id"], stats["sample_id"])
u, w, r = np.array([s.split("_") for s in d["sample_id"]], dtype=int).T
fit = d["role"] == "fit"
reg = np.isin(w, [20, 21, 22, 23]) & (r < 6)
devwords = np.array([8, 9, 18, 19, 24])
testwords = np.array([25, 26, 27, 28, 29])
devmask = np.isin(w, devwords)
testmask = np.isin(w, testwords)
assert (fit.astype(int) + reg + devmask + testmask).max() == 1
assert [m.sum() for m in (reg, devmask, testmask)] == [240, 400, 400]
spec = d["spectrum"].reshape(len(u), 120, -1)
features = np.column_stack([
    spec.mean(-1), spec.std(-1),
    *(v.mean(-1) for v in np.array_split(spec, 4, axis=-1)),
    d["amplitude"], stats["features"],
])
raw = normalize(StandardScaler().fit(features[reg]).transform(features))
# Log-RMS entries of the original 3-axis amplitude table; higher is better.
energy = np.mean(d["amplitude"][:, [0, 5, 10]], axis=1)
floor = float(np.quantile(energy[fit], .05))
lookup = {(int(owner), int(word), int(rep)): int(i) for i, (owner, word, rep) in enumerate(zip(u, w, r))}
word_order = np.random.default_rng(1123).permutation(5)

def make_trials(words, count):
    ordered = words[word_order]
    base = [(int(word), rep) for rep in range(4) for word in ordered]
    groups = []
    for start in range(0, len(base), count):
        group = [base[(start + j) % len(base)] for j in range(count)]
        assert len({v[0] for v in group}) == count
        groups.append(group)
    first, retry, identities = [], [], []
    for owner in range(1, 11):
        for group in groups:
            first.append([lookup[owner, word, rep] for word, rep in group])
            retry.append([lookup[owner, word, rep + 4] for word, rep in group])
            identities.append(owner)
    first, retry = np.array(first), np.array(retry)
    assert not np.intersect1d(first, retry).size
    assert len(set(first.ravel())) == 200
    return first, retry, np.array(identities)

def aggregate(score, first, backup, quality):
    bad = energy[first] < floor
    if quality:
        chosen = np.where(bad, backup, first)
        reject = (energy[chosen] < floor).any(axis=1)
        extra = bad.sum(axis=1)
    else:
        chosen = first
        reject = np.zeros(len(first), dtype=bool)
        extra = np.zeros(len(first))
    return score[chosen].mean(axis=1), reject, extra

def threshold(score, rejected, labels):
    thresholds = np.r_[score.max() + max(1., abs(score.max()) * 1e-6), np.unique(score)]
    outcomes = []
    for tau in thresholds:
        accepted = (score >= tau) & ~rejected
        far = float(accepted[~labels].mean())
        frr = float((~accepted[labels]).mean())
        if far <= .05 + 1e-12:
            outcomes.append((frr, far, -float(tau), float(tau)))
    return min(outcomes)[-1]

def measures(score, rejected, extra, labels, tau):
    accepted = (score >= tau) & ~rejected
    far = float(accepted[~labels].mean())
    frr = float((~accepted[labels]).mean())
    ranking = score.copy()
    ranking[rejected] = score.min() - max(1., float(np.ptp(score)))
    fp, tp, _ = roc_curve(labels, ranking, drop_intermediate=False)
    ix = int(np.argmin(np.abs(fp - (1 - tp))))
    eer = float((fp[ix] + 1 - tp[ix]) / 2)
    return dict(far=far, frr=frr, hter=(far + frr) / 2,
                eer=eer, auc=float(roc_auc_score(labels, ranking)),
                retry_fraction=float((extra > 0).mean()),
                genuine_retry_fraction=float((extra[labels] > 0).mean()),
                attack_retry_fraction=float((extra[~labels] > 0).mean()),
                extra_recordings_per_trial=float(extra.mean()),
                genuine_quality_failure=float(rejected[labels].mean()),
                attack_quality_failure=float(rejected[~labels].mean()),
                genuine_trials=int(labels.sum()), attack_trials=int((~labels).sum()),
                false_rejects=int((~accepted[labels]).sum()), false_accepts=int(accepted[~labels].sum()))

protocol = dict(
    dataset="10 users; hardware ACC; closed known identities",
    encoder="frozen pure ACC CE 42/43/44 from prior matched episodic controls",
    registration_words=[20, 21, 22, 23], registration_per_user=24,
    development_words=devwords.tolist(), query_words=testwords.tolist(),
    source_records_per_identity_per_stage=20, backup_records_per_identity_per_stage=20,
    challenge_lengths=[1, 3, 5],
    trials_per_identity={1: 20, 3: 7, 5: 4},
    grouping="fixed cyclic word order; 3-word last trial reuses first record once to retain all20 records",
    quality="training 5th percentile log-RMS proxy; no measured SNR/dropout/clipping available",
    quality_floor=floor, retry="at most one backup recording per low-energy word; no score-based retry",
    policy="reject after low-quality backup; all trials retained",
    aggregation="mean per-record score",
    threshold="per-target/length/backend/policy development FAR<=5%, then minimum FRR; frozen before query",
    backends=["RawRBF", "RBF", "Cosine", "WordPrototypeTop2", "ShrinkageMahalanobis"],
    scope="historical words reassigned to allow5 unseen-in-encoder development and5 query prompts; exploratory",
    uncertainty="few non-overlapping groups; 3word one-record overlap; no independence-based confidence claim",
    ranking_metrics="EER nearest empirical FAR/FRR intersection average; quality rejects tied below all remaining scores; AUC includes all trials",
)
(out / "protocol_public.json").write_text(json.dumps(protocol, indent=2))
ck = torch.load(root / "adapted_acc_private.pt", map_location="cpu", weights_only=True)
x = torch.tensor((features - ck["mean"].numpy()) / ck["scale"].numpy(), dtype=torch.float32)
trials = {count: (make_trials(devwords, count), make_trials(testwords, count)) for count in (1, 3, 5)}
rows = []
locks = []
pending = []
with threadpool_limits(limits=2):
    for seed in (0, 42, 43, 44):
        if seed == 0:
            embedding = raw
            names = ["RawRBF"]
        else:
            net = nn.Sequential(nn.Linear(features.shape[1], 256), nn.LayerNorm(256),
                                nn.GELU(), nn.Linear(256, 192))
            state = torch.load(root / "full_run_10x40_raw_acc/strict_episodic_6x4" / f"CE_{seed}_private.pt",
                               map_location="cpu", weights_only=True)
            net.load_state_dict(state["encoder"])
            with torch.inference_mode():
                embedding = F.normalize(net.eval()(x), dim=1).numpy()
            names = ["RBF", "Cosine", "WordPrototypeTop2", "ShrinkageMahalanobis"]
        for owner in range(1, 11):
            labels = u == owner
            svm = SVC(C=10, class_weight="balanced").fit(embedding[reg], labels[reg])
            scores = {names[0]: svm.decision_function(embedding)}
            if seed:
                center = normalize(embedding[reg & labels].mean(0, keepdims=True))[0]
                scores["Cosine"] = embedding @ center
                prototypes = normalize(np.stack([embedding[reg & labels & (w == word)].mean(0)
                                                  for word in [20, 21, 22, 23]]))
                pair = embedding @ prototypes.T
                scores["WordPrototypeTop2"] = np.sort(pair, axis=1)[:, -2:].mean(1)
                covariance = LedoitWolf().fit(embedding[reg & labels])
                scores["ShrinkageMahalanobis"] = -covariance.mahalanobis(embedding)
            for backend in names:
                score = scores[backend]
                for count in (1, 3, 5):
                    dev, test = trials[count]
                    devlabels = dev[2] == owner
                    testlabels = test[2] == owner
                    for quality in (False, True):
                        ds, dr, de = aggregate(score, dev[0], dev[1], quality)
                        tau = threshold(ds, dr, devlabels)
                        lock = dict(seed=seed, user=owner, backend=backend, words=count,
                                    quality=quality, threshold=tau,
                                    development=measures(ds, dr, de, devlabels, tau))
                        locks.append(lock)
                        # Query scores are not used in any parameter/threshold choice.
                        ts, tr, te = aggregate(score, test[0], test[1], quality)
                        pending.append((dict(seed=seed, user=owner, backend=backend, words=count,
                                             quality=quality), ts, tr, te, testlabels, tau))
            print("USER_DONE", seed, owner, flush=True)
(out / "thresholds_public.json").write_text(json.dumps(locks, indent=2, allow_nan=False))
for config, ts, tr, te, labels, tau in pending:
    rows.append(dict(**config, **measures(ts, tr, te, labels, tau)))
(out / "results_public.json").write_text(json.dumps(rows, indent=2, allow_nan=False))
print("MULTIWORD_DONE", flush=True)

