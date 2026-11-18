"""trainb-compatible target binary head.

Keeps the original trainb frozen-feature cache, SupCon projection loss and
fusion model. Adds a target-user BCE head: target user is positive and every
other user is negative.
"""
import argparse
import copy
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score

from data import MultiUserSpectrogramDataset, SpectrogramAugment
from modelb import SupConLoss, create_multi_source_model
from target_auth_protocol_150 import USER_NAMES, build_target_protocol, save_manifest
from trainb_target_150 import precompute_features, head_projection


def choose_threshold(genuine, impostor):
    values = np.unique(np.r_[genuine, impostor])
    candidates = np.r_[values[0] - 1e-6, (values[:-1] + values[1:]) / 2,
                       values[-1] + 1e-6] if len(values) > 1 else values
    best = None
    for t in candidates:
        far = float(np.mean(impostor >= t)); frr = float(np.mean(genuine < t))
        key = (abs(far - frr), (far + frr) / 2)
        if best is None or key < best[0]: best = (key, float(t))
    return best[1]


@torch.no_grad()
def binary_values(model, binary, features, device):
    model.eval(); binary.eval(); out = []
    for part in features.split(64):
        out.append(binary(model.fusion(part.to(device))).squeeze(1).cpu().numpy())
    return np.concatenate(out)


def evaluate(model, binary, cached, protocol, target_user, final=False):
    device = next(model.parameters()).device
    values = {name: binary_values(model, binary, x, device) for name, x in cached.items()}
    target = target_user - 1
    cal_genuine = values["target_dev_pool"]
    cal_impostor = values["impostor_calibration"]
    t = choose_threshold(cal_genuine, cal_impostor)
    if final:
        genuine = values["final_genuine"]; impostor = values["final_impostor"]
    else:
        mask = protocol.labels["model_validation"] == target
        genuine = values["model_validation"][mask]
        impostor = values["model_validation"][~mask]
    far = float(np.mean(impostor >= t)); frr = float(np.mean(genuine < t))
    row = {"threshold": t, "far": far, "frr": frr, "hter": (far + frr) / 2,
           "auc": float(roc_auc_score(np.r_[np.ones(len(genuine)), np.zeros(len(impostor))],
                                       np.r_[genuine, impostor]))}
    if final:
        labels = protocol.labels["final_impostor"]
        row["attack_by_user"] = {}
        for label in range(9):
            if label == target: continue
            z = impostor[labels == label]
            row["attack_by_user"][USER_NAMES[label]] = {
                "accepted": int(np.sum(z >= t)), "n": int(len(z)),
                "far": float(np.mean(z >= t)), "mean_logit": float(z.mean())}
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target_user", type=int, required=True, choices=(1, 3, 4, 5))
    p.add_argument("--cyclegan_ckpt", required=True)
    p.add_argument("--save_dir", required=True)
    p.add_argument("--log_dir", required=True)
    p.add_argument("--num_epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--class_feature_copies", type=int, default=2)
    p.add_argument("--feature_batch_size", type=int, default=64)
    args = p.parse_args()
    os.makedirs(args.save_dir, exist_ok=True); os.makedirs(args.log_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    protocol = build_target_protocol(args.target_user, split_seed=42, auth_seed=42)
    manifest_path = os.path.join(args.save_dir, "split_manifest.json")
    save_manifest(protocol, manifest_path)
    model, info = create_multi_source_model(args.cyclegan_ckpt, freeze_generators=True,
                                             proj_dim=128, device=device)
    raw_x, raw_y = protocol.arrays["auth_train"], protocol.labels["auth_train"]
    parts, labs = [], []
    counts = {label: int((raw_y == label).sum()) for label in range(9)}
    desired = max(counts.values()) * args.class_feature_copies
    for label in range(9):
        mask = raw_y == label; repeat = 0; got = 0
        while got < desired:
            x, y = precompute_features(model, raw_x[mask], raw_y[mask], device,
                                        args.feature_batch_size,
                                        augment=None if repeat == 0 else SpectrogramAugment())
            take = min(desired - got, len(x)); parts.append(x[:take]); labs.append(y[:take])
            got += take; repeat += 1
    train_x = torch.cat(parts); train_labels = torch.cat(labs)
    train_target = (train_labels == args.target_user - 1).float()
    cached = {}
    for name in ("model_validation", "target_dev_pool", "impostor_calibration",
                 "final_genuine", "final_impostor"):
        x, _ = precompute_features(model, protocol.arrays[name], protocol.labels[name],
                                   device, args.feature_batch_size)
        cached[name] = x.to(device)
    loader = DataLoader(TensorDataset(train_x, train_labels, train_target),
                        batch_size=args.batch_size, shuffle=True, drop_last=True)
    binary = torch.nn.Linear(model.feat_dim, 1).to(device)
    params = [p for p in model.parameters() if p.requires_grad] + list(binary.parameters())
    optimizer = Adam(params, lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=1e-6)
    criterion = SupConLoss(0.07)
    pos_weight = torch.tensor([(train_target == 0).sum() / (train_target == 1).sum()], device=device)
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    best = None; best_state = None; history = []
    for epoch in range(args.num_epochs):
        model.train(); binary.train(); losses = []
        for features, labels, target in loader:
            features, labels, target = features.to(device), labels.to(device), target.to(device)
            fused = model.fusion(features)
            loss = criterion(model.projector(fused), labels) + args.beta * bce(binary(fused).squeeze(1), target)
            optimizer.zero_grad(); loss.backward();
            torch.nn.utils.clip_grad_norm_(params, 1.0); optimizer.step(); losses.append(float(loss.detach().cpu()))
        scheduler.step()
        if (epoch + 1) % 5 == 0:
            val = evaluate(model, binary, cached, protocol, args.target_user, False)
            row = {"epoch": epoch + 1, "loss": float(np.mean(losses)), "validation": val}
            history.append(row); print(row, flush=True)
            rank = (-val["hter"], val["auc"])
            if best is None or rank > best:
                best = rank; best_state = (epoch + 1, copy.deepcopy(model.state_dict()), copy.deepcopy(binary.state_dict()))
    best_epoch, model_state, binary_state = best_state
    model.load_state_dict(model_state); binary.load_state_dict(binary_state)
    result = {"protocol": "trainb_target_binary_v1", "target_user": args.target_user,
              "cyclegan_ckpt": args.cyclegan_ckpt, "head_framework": "trainb",
              "raw_auth_counts": counts, "target_positive_raw": counts[args.target_user - 1],
              "other_users_negative_raw": int(sum(v for k, v in counts.items() if k != args.target_user - 1)),
              "best_epoch": best_epoch,
              "validation": evaluate(model, binary, cached, protocol, args.target_user, False),
              "final_attack": evaluate(model, binary, cached, protocol, args.target_user, True),
              "counts": protocol.manifest["counts"], "history": history}
    with open(os.path.join(args.save_dir, "trainb_binary_results.json"), "w") as f: json.dump(result, f, indent=2)
    torch.save({"model_state_dict": model.state_dict(), "binary_state_dict": binary.state_dict(), "result": result},
               os.path.join(args.save_dir, "trainb_binary_auth_head.pth"))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__": main()
