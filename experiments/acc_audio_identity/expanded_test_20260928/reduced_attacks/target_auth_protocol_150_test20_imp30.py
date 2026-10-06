"""Disjoint 150-sample CycleGAN / remaining-sample auth protocol."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np

USER_NAMES = ["bo"] + [f"user{i}" for i in range(2, 10)]
USER_PATHS = [f"./meiy/{i}/acc.npy" for i in range(1, 10)]
GAN_SAMPLES = 150


def sha256_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _split_150(n_samples: int, user_label: int, seed: int = 42):
    if n_samples < GAN_SAMPLES + 5:
        raise ValueError(f"user {user_label + 1} has too few samples: {n_samples}")
    rng = np.random.RandomState(seed + user_label)
    indices = rng.permutation(n_samples)
    cyclegan = np.sort(indices[:GAN_SAMPLES])
    auth = np.sort(indices[GAN_SAMPLES:])
    return cyclegan, auth


def split_user_data(seed: int = 42):
    """Return per-user CycleGAN/auth indices and auditable source manifest."""
    result = {}
    for label, (name, path) in enumerate(zip(USER_NAMES, USER_PATHS)):
        data = np.load(path, mmap_mode="r")
        gan_idx, auth_idx = _split_150(len(data), label, seed)
        if set(gan_idx.tolist()) & set(auth_idx.tolist()):
            raise RuntimeError(f"split overlap for {name}")
        result[name] = {
            "path": path,
            "sha256": sha256_file(path),
            "n_total": int(len(data)),
            "cyclegan_indices": gan_idx.tolist(),
            "auth_indices": auth_idx.tolist(),
        }
    return result


def four_way_auth_split(indices: Sequence[int], seed: int, label: int):
    """Split auth half into train, model validation, dev, final test."""
    indices = np.asarray(indices, dtype=np.int64)
    rng = np.random.RandomState(seed + 1000 + label)
    indices = rng.permutation(indices)
    n = len(indices)
    n_train = int(round(0.50 * n))
    n_val = int(round(0.10 * n))
    n_dev = int(round(0.20 * n))
    train = indices[:n_train]
    model_val = indices[n_train:n_train + n_val]
    dev = indices[n_train + n_val:n_train + n_val + n_dev]
    final_test = indices[n_train + n_val + n_dev:]
    return train, model_val, dev, final_test


@dataclass
class ProtocolData:
    arrays: Dict[str, np.ndarray]
    labels: Dict[str, np.ndarray]
    sample_ids: Dict[str, np.ndarray]
    manifest: Dict


def build_target_protocol(target_user: int, split_seed: int = 42,
                          auth_seed: int = 42) -> ProtocolData:
    """Build one target run; target_user is 1-based and all users use auth halves."""
    if target_user not in (1, 3, 4, 5):
        raise ValueError("this experiment includes target users 1, 3, 4, 5")
    target_label = target_user - 1
    groups = {key: {"specs": [], "labels": [], "ids": []} for key in (
        "auth_train", "model_validation", "target_dev_pool", "final_genuine",
        "impostor_calibration", "final_impostor")}
    users = split_user_data(split_seed)
    assignments_manifest = {}

    for label, name in enumerate(USER_NAMES):
        source = users[name]
        data = np.load(source["path"]).astype(np.float32, copy=False)[:, :80, :80]
        gan_idx = np.asarray(source["cyclegan_indices"], dtype=np.int64)
        auth_idx = np.asarray(source["auth_indices"], dtype=np.int64)
        train, model_val, dev, final_test = four_way_auth_split(auth_idx, auth_seed, label)

        if label == target_label:
            assignments = {
                "auth_train": train,
                "model_validation": model_val,
                "target_dev_pool": dev,
                "final_genuine": final_test,
            }
        else:
            assignments = {
                "auth_train": train,
                "model_validation": model_val,
                "impostor_calibration": dev[:len(dev) // 2],
                "final_impostor": final_test,
            }

        assigned_ids = []
        for group, idx in assignments.items():
            groups[group]["specs"].append(data[idx])
            groups[group]["labels"].append(np.full(len(idx), label, dtype=np.int64))
            ids = [f"{name}:{int(i)}" for i in idx]
            groups[group]["ids"].extend(ids)
            assigned_ids.extend(ids)
        if set(assigned_ids) & {f"{name}:{int(i)}" for i in gan_idx}:
            raise RuntimeError(f"CycleGAN train data leaked into auth groups for {name}")
        assignments_manifest[name] = {
            "n_total": source["n_total"],
            "cyclegan_n": int(len(gan_idx)),
            "auth_n": int(len(auth_idx)),
            "cyclegan_indices": source["cyclegan_indices"],
            "auth_indices": source["auth_indices"],
            "auth_groups": {key: value.tolist() for key, value in assignments.items()},
        }

    arrays, labels, sample_ids = {}, {}, {}
    group_sets = {}
    for name, content in groups.items():
        arrays[name] = np.concatenate(content["specs"]) if content["specs"] else np.empty((0, 80, 80), np.float32)
        labels[name] = np.concatenate(content["labels"]) if content["labels"] else np.empty((0,), np.int64)
        sample_ids[name] = np.asarray(content["ids"])
        group_sets[name] = set(content["ids"])
    keys = list(group_sets)
    for i, left in enumerate(keys):
        for right in keys[i + 1:]:
            if group_sets[left] & group_sets[right]:
                raise RuntimeError(f"protocol group overlap: {left}/{right}")

    manifest = {
        "protocol": "150_cyclegan_plus_auth_test20_imp30_v1",
        "target_user": target_user,
        "target_name": USER_NAMES[target_label],
        "split_seed": split_seed,
        "auth_seed": auth_seed,
        "counts": {key: int(len(value)) for key, value in arrays.items()},
        "users": assignments_manifest,
        "assertions": {
            "each_user_cyclegan_auth_disjoint": True,
            "target_cyclegan_data_absent_from_auth_groups": True,
            "auth_groups_pairwise_disjoint": True,
        },
    }
    return ProtocolData(arrays, labels, sample_ids, manifest)


def save_manifest(protocol: ProtocolData, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(protocol.manifest, stream, ensure_ascii=False, indent=2)

