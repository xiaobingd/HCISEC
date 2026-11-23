"""Matched closed-set cross-word 6-support/4-query metric-learning trial."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.svm import SVC
from torch import nn
from torch.nn import functional as F

root = Path("/root")
out = root / "full_run_10x40_raw_acc" / "strict_episodic_6x4"
out.mkdir(exist_ok=True)
torch.set_num_threads(2)
data = np.load(root / "paired_raw_acc_private.npz")
stats = np.load(root / "stats27_private.npz")
assert np.array_equal(data["sample_id"], stats["sample_id"])
users, words, repeats = np.array([s.split("_") for s in data["sample_id"]], dtype=int).T
fit = data["role"] == "fit"
register = np.isin(words, range(20, 24)) & (repeats < 3)
calibration = np.isin(words, range(20, 24)) & np.isin(repeats, [3, 4])
sameword = np.isin(words, range(20, 24)) & (repeats >= 5)
crossword = np.isin(words, range(24, 30)) & (repeats < 4)
assert (fit.astype(int) + register + calibration + sameword + crossword).max() == 1
assert [m.sum() for m in (fit, register, calibration, sameword, crossword)] == [2080, 120, 80, 120, 240]
spectrum = data["spectrum"].reshape(len(users), 120, -1)
features = np.column_stack([
    spectrum.mean(-1), spectrum.std(-1),
    *(v.mean(-1) for v in np.array_split(spectrum, 4, axis=-1)),
    data["amplitude"], stats["features"],
])
checkpoint_path = root / "adapted_acc_private.pt"
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
scaler_check = StandardScaler().fit(features[fit])
assert np.allclose(checkpoint["mean"].numpy(), scaler_check.mean_, atol=1e-5)
assert np.allclose(checkpoint["scale"].numpy(), scaler_check.scale_, rtol=1e-4, atol=1e-5)
x = torch.tensor((features - checkpoint["mean"].numpy()) / checkpoint["scale"].numpy(), dtype=torch.float32)
labels = torch.tensor(users - 1)
fitwords = np.unique(words[fit])
lookup = {(owner, word): np.flatnonzero(fit & (users == owner) & (words == word))
          for owner in range(1, 11) for word in fitwords}
assert len(fitwords) == 26 and all(len(v) == 8 for v in lookup.values())

def encoder():
    net = nn.Sequential(nn.Linear(features.shape[1], 256), nn.LayerNorm(256),
                        nn.GELU(), nn.Linear(256, 192))
    net.load_state_dict(checkpoint["encoder"])
    return net

def sample_episode(rng):
    chosen = rng.choice(fitwords, 5, replace=False)
    support_words, query_words = chosen[:3], chosen[3:]
    assert set(support_words).isdisjoint(set(query_words))
    support = np.array([np.concatenate([rng.choice(lookup[owner, word], 2, replace=False)
                                        for word in support_words]) for owner in range(1, 11)])
    query = np.array([np.concatenate([rng.choice(lookup[owner, word], 2, replace=False)
                                      for word in query_words]) for owner in range(1, 11)])
    assert support.shape == (10, 6) and query.shape == (10, 4)
    assert not np.intersect1d(support, query).size
    return np.concatenate([support, query], axis=1).ravel()

def supcon(z, y):
    sim = z @ z.T / .1
    diagonal = torch.eye(len(z), dtype=torch.bool)
    positive = (y[:, None] == y[None, :]) & ~diagonal
    logprob = sim - torch.logsumexp(sim.masked_fill(diagonal, -torch.inf), dim=1, keepdim=True)
    return -(logprob * positive).sum(1).div(positive.sum(1)).mean()

def metrics(score, mask, target, tau):
    y = users == target
    far = float(np.mean(score[mask & ~y] >= tau))
    frr = float(np.mean(score[mask & y] < tau))
    f, t, _ = roc_curve(y[mask], score[mask], drop_intermediate=False)
    j = np.argmin(abs(f + t - 1))
    return dict(eer=float((f[j] + 1 - t[j]) / 2), far=far, frr=frr,
                hter=(far + frr) / 2, auc=float(roc_auc_score(y[mask], score[mask])))

def evaluate(embeddings, method, seed):
    result = []
    for target in range(1, 11):
        y = users == target
        svm = SVC(C=10, class_weight="balanced").fit(embeddings[register], y[register])
        center = normalize(embeddings[register & y].mean(0, keepdims=True))[0]
        scores = dict(rbf=svm.decision_function(embeddings), cosine=embeddings @ center)
        for backend, score in scores.items():
            f, t, thresholds = roc_curve(y[calibration], score[calibration], drop_intermediate=False)
            j = min(range(len(thresholds)), key=lambda i: (abs(f[i] + t[i] - 1), f[i]))
            tau = thresholds[j]
            for condition, mask in (("sameword", sameword), ("crossword", crossword)):
                result.append(dict(method=method, seed=seed, user=target, backend=backend,
                                   condition=condition, **metrics(score, mask, target, tau)))
    return result

protocol = dict(
    methods=["CE", "CE_SupCon", "ProtoCosine", "ProtoLearnedScale"],
    seeds=[42, 43, 44], updates=400, samples_per_update=100,
    support="3 shared training words x 2 recordings per user",
    query="2 other shared training words x 2 recordings per user",
    scale="fixed16 or positive learned softplus scale initialized16",
    encoder="existing statistics MLP 762->256->192, pretrained pure ACC CE+SupCon",
    initialization_sha256=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
    fit_words=26, targets=list(range(1, 11)), attacks="other9 known identities, includesU10",
    registration_per_user=12, calibration_per_user=8,
    sameword_per_user=12, crossword_per_user=24,
    optimizer="AdamW lr1e-4 weight_decay=.01,clip_grad_norm1",
    selection="fixed final400; no test tuning; calibration EER threshold",
    primary_backend="RBF C10 balanced", secondary_backend="cosine prototype",
    provenance="all methods share existing700-step pure ACC initialization; historical query reuse is exploratory",
    interpretation="cosine ProtoNet adaptation and learned-scale variant; not canonical leave-one-out GE2E",
    goals=dict(macro_crossword_eer=.10, macro_crossword_hter=.125),
)
(out / "protocol_public.json").write_text(json.dumps(protocol, indent=2))
rows = []
training = []
with torch.inference_mode():
    initial = F.normalize(encoder().eval()(x), dim=1).numpy()
rows.extend(evaluate(initial, "initial", 0))
for seed in (42, 43, 44):
    for method in protocol["methods"]:
        final_path = out / f"{method}_{seed}_private.pt"
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        net = encoder()
        head = nn.Linear(192, 10, bias=False)
        with torch.no_grad():
            z = F.normalize(net(x), dim=1)
            head.weight.copy_(F.normalize(torch.stack([z[fit & (users == owner)].mean(0)
                                                       for owner in range(1, 11)]), dim=1))
        logscale = nn.Parameter(torch.tensor(float(np.log(np.expm1(16)))))
        parameters = list(net.parameters())
        if method in ("CE", "CE_SupCon"):
            parameters += list(head.parameters())
        if method == "ProtoLearnedScale":
            parameters += [logscale]
        optimizer = torch.optim.AdamW(parameters, lr=1e-4)
        losses = []
        if final_path.exists():
            saved = torch.load(final_path, map_location="cpu", weights_only=False)
            net.load_state_dict(saved["encoder"])
            logscale.data.copy_(saved["logscale"])
        else:
            for step in range(400):
                indices = sample_episode(rng)
                z = F.normalize(net(x[indices]), dim=1)
                y = labels[indices]
                if method in ("CE", "CE_SupCon"):
                    loss = F.cross_entropy(16 * F.linear(z, F.normalize(head.weight, dim=1)), y)
                    if method == "CE_SupCon":
                        loss = loss + .1 * supcon(z, y)
                else:
                    episode = z.reshape(10, 10, 192)
                    centers = F.normalize(episode[:, :6].mean(1), dim=1)
                    query = episode[:, 6:].reshape(40, 192)
                    query_labels = torch.arange(10).repeat_interleave(4)
                    scale = F.softplus(logscale) if method == "ProtoLearnedScale" else 16
                    loss = F.cross_entropy(scale * (query @ centers.T), query_labels)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(parameters, 1)
                optimizer.step()
                losses.append(float(loss.detach()))
            torch.save(dict(encoder=net.state_dict(), logscale=logscale.detach()), final_path)
        with torch.inference_mode():
            embeddings = F.normalize(net.eval()(x), dim=1).numpy()
        current = evaluate(embeddings, method, seed)
        rows.extend(current)
        training.append(dict(method=method, seed=seed,
                             final_loss=float(np.mean(losses[-50:])) if losses else None,
                             scale=float(F.softplus(logscale))))
        (out / "results_public.json").write_text(json.dumps(rows, indent=2))
        (out / "training_public.json").write_text(json.dumps(training, indent=2))
        for backend in ("rbf", "cosine"):
            v = [r for r in current if r["backend"] == backend and r["condition"] == "crossword"]
            print(seed, method, backend,
                  [round(100 * np.mean([r[k] for r in v]), 2) for k in ("eer", "hter", "far", "frr")],
                  flush=True)
print("STRICT_EPISODIC_DONE", flush=True)

