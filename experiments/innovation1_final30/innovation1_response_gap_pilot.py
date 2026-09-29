"""Pilot C0-C7 for user-specific CycleGAN response gaps, with N0-N2 inputs.

Uses the existing 150-CycleGAN / remaining-auth sample split. Every mismatch
model is evaluated only on the auth half of its own user's data. Classifier,
response standardization, and model-specific prototypes use auth_train only;
threshold uses target_dev_pool and impostor_calibration; final test results
are excluded from all model and threshold choices. Existing acc.npy is already normalized using
per-user full-data statistics, so this is exploratory rather than a final
leakage-free preprocessing experiment.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from modelb import create_multi_source_model
from target_auth_protocol_150_final30 import USER_NAMES, build_target_protocol, save_manifest, sha256_file

ROOT = "/root/autodl-fs/cyclegan_forward_aligned_20260928"
CHECKPOINTS = {u: f"{ROOT}/user{u}/cyclegan/model/checkpoint_epoch_50.pth" for u in (1, 4, 5)}
GROUPS = ("auth_train", "model_validation", "target_dev_pool",
          "impostor_calibration", "final_genuine", "final_impostor")
MODELS = (1, 4, 5)
TARGET = 4
C_VALUES = (0.01, 0.1, 1.0, 10.0)


def l2_rows(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def normalize_input(x, kind, reference):
    if kind == "N0_current":
        return x
    if kind == "N1_window_rms":
        ref = np.sqrt(np.mean(reference ** 2, axis=(1, 2))).mean()
        rms = np.sqrt(np.mean(x ** 2, axis=(1, 2), keepdims=True))
        return np.clip(x * ref / np.maximum(rms, 1e-6), -1, 1).astype(np.float32)
    if kind == "N2_window_mean_std":
        ref_mean = reference.mean()
        ref_std = reference.std()
        mean = x.mean(axis=(1, 2), keepdims=True)
        std = x.std(axis=(1, 2), keepdims=True)
        return np.clip((x - mean) * ref_std / np.maximum(std, 1e-6) + ref_mean,
                       -1, 1).astype(np.float32)
    raise ValueError(kind)


@torch.no_grad()
def extract_for_model(model, x, device, batch_size):
    output = {"shallow": [], "bottleneck": [], "absolute_cycle": [], "residual_stats": []}
    model.eval()
    for start in range(0, len(x), batch_size):
        batch = torch.as_tensor(x[start:start + batch_size], device=device,
                                dtype=torch.float32).unsqueeze(1)
        shallow = model.shallow_agg(model.shallow_encoder(batch))
        bottleneck = model.bottleneck_agg(model.bottleneck_encoder(batch))
        residual, scalar = model._compute_cycle_diff(batch)
        encoded_residual = model.diff_agg(model.diff_encoder(residual))
        absolute_cycle = torch.cat((encoded_residual, scalar), dim=1)
        abs_map = residual.abs().squeeze(1)
        flat = abs_map.reshape(len(abs_map), -1)
        phi = torch.stack((
            flat.mean(dim=1),
            flat.square().mean(dim=1).sqrt(),
            abs_map[:, :20].mean(dim=(1, 2)),
            abs_map[:, 20:50].mean(dim=(1, 2)),
            abs_map[:, 50:].mean(dim=(1, 2)),
        ), dim=1)
        for key, value in (("shallow", shallow), ("bottleneck", bottleneck),
                           ("absolute_cycle", absolute_cycle), ("residual_stats", phi)):
            output[key].append(value.cpu().numpy().astype(np.float32))
    return {key: np.concatenate(parts) for key, parts in output.items()}


def extract_all_for_norm(protocol, kind, reference, device, batch_size):
    by_model = {}
    for user in MODELS:
        model, _ = create_multi_source_model(CHECKPOINTS[user], device=device,
                                             freeze_generators=True, proj_dim=128)
        model.eval()
        by_group = {}
        for group in GROUPS:
            x = normalize_input(protocol.arrays[group], kind, reference)
            by_group[group] = extract_for_model(model, x, device, batch_size)
        by_model[user] = by_group
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"{kind}: extracted model user{user}", flush=True)
    return by_model


def fit_relative_features(by_model, labels):
    """Use only auth_train to calibrate model-specific response statistics."""
    stats = {}
    proto = {}
    score_cal = {}
    for user in MODELS:
        raw = by_model[user]["auth_train"]
        phi = raw["residual_stats"]
        std = phi.std(0).clip(1e-6)
        stats[user] = (phi.mean(0), std)
        owner_label = user - 1
        own = raw["bottleneck"][labels["auth_train"] == owner_label]
        proto[user] = l2_rows(own).mean(0)
        proto[user] /= max(np.linalg.norm(proto[user]), 1e-12)
        all_scores = l2_rows(raw["bottleneck"]) @ proto[user]
        score_cal[user] = (all_scores.mean(), max(all_scores.std(), 1e-8))
    groups = {}
    for group in GROUPS:
        p = {u: (by_model[u][group]["residual_stats"] - stats[u][0]) / stats[u][1]
             for u in MODELS}
        others = [u for u in MODELS if u != TARGET]
        delta4, delta5 = p[others[0]] - p[TARGET], p[others[1]] - p[TARGET]
        cycle_gap = np.concatenate((delta4, delta5,
                                    np.minimum(delta4, delta5), (delta4 + delta5) / 2), axis=1)
        matched = {}
        for user in MODELS:
            z = l2_rows(by_model[user][group]["bottleneck"])
            score = z @ proto[user]
            matched[user] = (score - score_cal[user][0]) / score_cal[user][1]
        st, sa, sb = matched[TARGET], matched[others[0]], matched[others[1]]
        response_gap = np.stack((st, sa, sb, st-sa, st-sb, st-np.maximum(sa,sb)), axis=1)
        target = by_model[TARGET][group]
        candidates = {
            "C0_shallow": target["shallow"],
            "C1_bottleneck": target["bottleneck"],
            "C2_absolute_cycle": target["absolute_cycle"],
            "C3_bottleneck_absolute": np.concatenate((target["bottleneck"], target["absolute_cycle"]), axis=1),
            "C4_cycle_gap": cycle_gap,
            "C5_response_gap": response_gap,
            "C6_relative_gaps": np.concatenate((cycle_gap, response_gap), axis=1),
            "C7_all": np.concatenate((target["shallow"], target["bottleneck"],
                                      target["absolute_cycle"], cycle_gap, response_gap), axis=1),
        }
        groups[group] = {k: v.astype(np.float32) for k, v in candidates.items()}
    return groups, {"response_score_train_mean_std": {str(u): list(map(float, score_cal[u])) for u in MODELS},
                    "response_prototype_owners": {str(u): USER_NAMES[u-1] for u in MODELS}}


def threshold(dev_pos, dev_neg):
    candidates = np.unique(np.r_[dev_pos, dev_neg])
    return min((float(((dev_pos < t).mean() + (dev_neg >= t).mean())/2),
                abs(float((dev_pos < t).mean() - (dev_neg >= t).mean())), float(t))
               for t in candidates)[2]


def eer(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    idx = np.argmin(np.abs(fpr-fnr))
    return float((fpr[idx] + fnr[idx])/2)


def evaluate(features, labels, name):
    best = None
    ytr, yval = labels["auth_train"], labels["model_validation"]
    for c in C_VALUES:
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=3000,
                                                                  class_weight="balanced"))
        clf.fit(features["auth_train"][name], ytr)
        val_acc = float(clf.score(features["model_validation"][name], yval))
        if best is None or val_acc > best[0]:
            best = (val_acc, c, clf)
    val_acc, c, clf = best
    target_idx = list(clf.classes_).index(TARGET - 1)
    score = lambda group: clf.predict_proba(features[group][name])[:, target_idx]
    pos_cal, neg_cal = score("target_dev_pool"), score("impostor_calibration")
    cutoff = threshold(pos_cal, neg_cal)
    positive, negative = score("final_genuine"), score("final_impostor")
    y = np.r_[np.ones(len(positive)), np.zeros(len(negative))]
    scores = np.r_[positive, negative]
    far = float((negative >= cutoff).mean())
    frr = float((positive < cutoff).mean())
    source_far = {}
    for lab in sorted(set(labels["final_impostor"])):
        mask = labels["final_impostor"] == lab
        source_far[USER_NAMES[int(lab)]] = {"n": int(mask.sum()),
                                          "accepted": int((negative[mask] >= cutoff).sum()),
                                          "far": float((negative[mask] >= cutoff).mean())}
    return {"C": c, "model_val_multiclass_acc": val_acc, "threshold": cutoff,
            "dev_far": float((neg_cal >= cutoff).mean()),
            "dev_frr": float((pos_cal < cutoff).mean()),
            "final_auc": float(roc_auc_score(y, scores)), "final_eer_descriptive": eer(y, scores),
            "final_far": far, "final_frr": frr, "final_hter": (far+frr)/2,
            "source_far": source_far}


def main():
    global TARGET
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="/root/autodl-fs/innovation1_response_gap_user4_20260929")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--target_epoch", type=int, default=50)
    parser.add_argument("--target_checkpoint", default=None)
    parser.add_argument("--norm", choices=("N0_current", "N1_window_rms", "N2_window_mean_std", "all"), default="all")
    parser.add_argument("--target_user", type=int, choices=(1, 4), default=4)
    parser.add_argument("--final_genuine", type=int, choices=(15, 30), default=15)
    args = parser.parse_args()
    TARGET = args.target_user
    CHECKPOINTS[TARGET] = CHECKPOINTS[TARGET].replace("checkpoint_epoch_50.pth", f"checkpoint_epoch_{args.target_epoch}.pth")
    if args.target_checkpoint:
        CHECKPOINTS[TARGET] = args.target_checkpoint
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    protocol = build_target_protocol(TARGET, split_seed=42, auth_seed=42, target_final_n=args.final_genuine)
    save_manifest(protocol, str(out / "protocol_manifest.json"))
    labels = protocol.labels
    target_train_mask = labels["auth_train"] == TARGET - 1
    reference = protocol.arrays["auth_train"][target_train_mask]
    result = {"protocol": protocol.manifest["protocol"], "target": USER_NAMES[TARGET-1],
              "mismatch_models": [u for u in MODELS if u != TARGET], "checkpoint_paths": CHECKPOINTS,
              "checkpoint_sha256": {str(u): sha256_file(CHECKPOINTS[u]) for u in MODELS},
              "counts": protocol.manifest["counts"], "normalizations": {},
              "note": "N1/N2 are post-hoc normalized-spectrogram stress tests for frozen generators. "
                      "Existing acc.npy already used per-user full-data scaling; N3 requires raw-data retraining."}
    norms = ("N0_current", "N1_window_rms", "N2_window_mean_std") if args.norm == "all" else (args.norm,)
    for kind in norms:
        by_model = extract_all_for_norm(protocol, kind, reference, device, args.batch_size)
        features, calibrations = fit_relative_features(by_model, labels)
        del by_model
        rows = {}
        for name in features["auth_train"]:
            rows[name] = evaluate(features, labels, name)
            print(kind, name, rows[name]["final_auc"], rows[name]["final_hter"], flush=True)
        result["normalizations"][kind] = {"calibration": calibrations, "candidates": rows}
        (out / f"{kind}.json").write_text(json.dumps(result["normalizations"][kind], indent=2), encoding="utf-8")
    (out / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("COMPLETE", out / "result.json", flush=True)


if __name__ == "__main__":
    main()

