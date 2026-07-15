#!/usr/bin/env python3
"""Cross-region seed scores from the current PhysAE/GNN fold models."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data


ROOT = Path("/groups/icecube/holgerkc/ML-ICE-PROJECT")
MODEL_DIR = ROOT / "catalyst_pipeline/outputs"
FOLDS = MODEL_DIR / "physae_cross_region_folds_v4.npz"
CACHE = MODEL_DIR / "physae_cross_region_seed_scores_v4_current.npz"

EVAL_BATCH = 8192
NUM_WORKERS = 0

sys.path.insert(0, str(ROOT / "catalyst_pipeline/scripts"))
from physae_gnn_v4 import PhysGNN, load_graph, make_loader  # noqa: E402


def sigmoid(x: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float32, copy=False)


def model_path(fold: str) -> Path:
    return MODEL_DIR / f"physae_model_v4_{fold}.pt"


def cache_is_current(path: Path) -> bool:
    if os.environ.get("PHYS_AE_FORCE_SEED_SCORES") == "1":
        return False
    if not path.exists():
        return False
    try:
        with np.load(path) as data:
            return (
                int(data["model_A_mtime_ns"]) == model_path("A").stat().st_mtime_ns
                and int(data["model_B_mtime_ns"]) == model_path("B").stat().st_mtime_ns
                and int(data["model_A_size"]) == model_path("A").stat().st_size
                and int(data["model_B_size"]) == model_path("B").stat().st_size
            )
    except Exception:
        return False


def load_model(fold: str, f_in: int, edge_dim: int, dev: str):
    path = model_path(fold)
    st = torch.load(path, map_location=dev)
    cfg = st.get("config", {})
    model = PhysGNN(
        f_in,
        cfg.get("hidden", 384),
        cfg.get("latent", 192),
        cfg.get("layers", 3),
        edge_dim,
        cfg.get("dropout", 0.11136),
    ).to(dev)
    model.load_state_dict(st["model"])
    model.eval()
    temperature = float(st["temperature"])
    print(f"loaded fold {fold}: T={temperature:.4f}", flush=True)
    return model, temperature


def compute_seed_scores() -> dict[str, np.ndarray]:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    fanout = [16, 8, 8]
    t0 = time.time()
    n, f_in, x, ei, ea, _ = load_graph()
    data = Data(x=x, edge_index=ei, edge_attr=ea, y=torch.zeros(n, dtype=torch.int8))
    model_a, temp_a = load_model("A", f_in, ea.shape[1], dev)
    model_b, temp_b = load_model("B", f_in, ea.shape[1], dev)

    folds = np.load(FOLDS)
    out_a = folds["outliers_A"].astype(np.int64, copy=False)
    in_a = folds["inliers_A"].astype(np.int64, copy=False)
    out_b = folds["outliers_B"].astype(np.int64, copy=False)
    in_b = folds["inliers_B"].astype(np.int64, copy=False)

    rows = np.concatenate([out_a, in_a, out_b, in_b]).astype(np.int64, copy=False)
    target_model = np.concatenate(
        [
            np.ones(len(out_a) + len(in_a), dtype=np.int8),   # model B scores region A
            np.zeros(len(out_b) + len(in_b), dtype=np.int8),  # model A scores region B
        ]
    )
    if len(np.unique(rows)) != len(rows):
        raise RuntimeError("seed row ids are not unique")

    sort_idx = np.argsort(rows)
    rows_sorted = rows[sort_idx]
    calibrated_logits = np.empty(len(rows), dtype=np.float32)

    print(f"building seed NeighborLoader over {len(rows):,} seeds ...", flush=True)
    loader = make_loader(data, rows, fanout, EVAL_BATCH, False, NUM_WORKERS)
    print("loader ready; scoring cross-region seed logits ...", flush=True)

    done = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(dev)
            bs = batch.batch_size
            ids = batch.n_id[:bs].cpu().numpy()
            pos = np.searchsorted(rows_sorted, ids)
            if np.any(pos >= len(rows_sorted)) or np.any(rows_sorted[pos] != ids):
                raise RuntimeError("NeighborLoader returned an input node outside the seed set")
            original_pos = sort_idx[pos]
            target = target_model[original_pos]
            batch_logits = np.empty(bs, dtype=np.float32)

            mask_a = target == 0
            mask_b = target == 1
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
                if mask_a.any():
                    raw_a = model_a(batch)[:bs].float().cpu().numpy()
                    batch_logits[mask_a] = raw_a[mask_a] / temp_a
                if mask_b.any():
                    raw_b = model_b(batch)[:bs].float().cpu().numpy()
                    batch_logits[mask_b] = raw_b[mask_b] / temp_b

            calibrated_logits[original_pos] = batch_logits
            done += bs
            if done % (EVAL_BATCH * 20) < EVAL_BATCH:
                print(f"  scored {done:,}/{len(rows):,} seeds ({time.time() - t0:.0f}s)", flush=True)

    n_out_a, n_in_a, n_out_b, n_in_b = map(len, [out_a, in_a, out_b, in_b])
    i0 = 0
    i1 = i0 + n_out_a
    i2 = i1 + n_in_a
    i3 = i2 + n_out_b
    i4 = i3 + n_in_b
    assert i4 == len(calibrated_logits)

    return {
        "a_pos_logit": calibrated_logits[i0:i1],
        "a_neg_logit": calibrated_logits[i1:i2],
        "b_pos_logit": calibrated_logits[i2:i3],
        "b_neg_logit": calibrated_logits[i3:i4],
        "a_pos_score": sigmoid(calibrated_logits[i0:i1]),
        "a_neg_score": sigmoid(calibrated_logits[i1:i2]),
        "b_pos_score": sigmoid(calibrated_logits[i2:i3]),
        "b_neg_score": sigmoid(calibrated_logits[i3:i4]),
        "model_A_temperature": np.array(temp_a, dtype=np.float64),
        "model_B_temperature": np.array(temp_b, dtype=np.float64),
        "model_A_mtime_ns": np.array(model_path("A").stat().st_mtime_ns, dtype=np.int64),
        "model_B_mtime_ns": np.array(model_path("B").stat().st_mtime_ns, dtype=np.int64),
        "model_A_size": np.array(model_path("A").stat().st_size, dtype=np.int64),
        "model_B_size": np.array(model_path("B").stat().st_size, dtype=np.int64),
    }


def load_seed_scores() -> dict[str, np.ndarray]:
    if not cache_is_current(CACHE):
        scores = compute_seed_scores()
        tmp = CACHE.with_suffix(CACHE.suffix + ".tmp")
        with tmp.open("wb") as fh:
            np.savez_compressed(fh, **scores)
        tmp.replace(CACHE)
        print(f"wrote {CACHE}", flush=True)
    else:
        print(f"using cached {CACHE}", flush=True)
    with np.load(CACHE) as data:
        return {key: data[key] for key in data.files}
