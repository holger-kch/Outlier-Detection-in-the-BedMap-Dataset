#!/usr/bin/env python3
"""Step-4 GNN validation — the honest generalization gates.

For each cross-region fold (A trained on south, tested on the rest; B vice versa)
re-score the HELD-OUT region's outlier seeds + held-out inlier seeds and report:
  * overall cross-region ROC-AUC + recall@0.5/0.7  (prevalence-independent headline)
  * DEGREE-STRATIFIED metrics by the seed's own cross-track degree bucket
    {0, 1-2, 3-7, 8+}. The central-challenge gate: it must score the zero-cross-track
    seeds (the regime of the ~75% of all nodes) correctly, not only well-connected
    ones. cross-track degree is computed for EVALUATION ONLY (track_id is never a feature).
Writes outputs/physae_validation_v4.json.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch_geometric.data import Data

sys.path.insert(0, str(Path(__file__).resolve().parent))
from physae_gnn_v4 import PhysGNN, load_graph, make_loader, roc_auc  # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]
OUTDIR = PROJECT / "outputs"
SOURCE = "/groups/icecube/janikh/Bedmap/bedmap_clean.parquet"
DST_NPY = OUTDIR / "spatial_edges_v3_dst.npy"
K_FULL = 16
BUCKETS = [(0, 0), (1, 2), (3, 7), (8, 16)]
FANOUT = [16, 8, 8]
EVAL_BATCH = 8192
NUM_WORKERS, PREFETCH = 8, 4   # model dims are read from each checkpoint's saved config


def log(m):
    print(m, flush=True)


def load_track():
    pf = pq.ParquetFile(SOURCE); N = pf.metadata.num_rows
    tr = np.empty(N, np.int64); off = 0
    for rg in range(pf.metadata.num_row_groups):
        a = pf.read_row_group(rg, columns=["track_id"]).column("track_id").to_numpy(zero_copy_only=False)
        tr[off:off + len(a)] = a; off += len(a)
    return tr


def cross_degree(seed_ids, track):
    dst = np.load(DST_NPY, mmap_mode="r")
    N = len(dst) // K_FULL
    nbr16 = np.asarray(dst).reshape(N, K_FULL)
    out = np.empty(len(seed_ids), np.int64)
    for i, s in enumerate(seed_ids):
        out[i] = int((track[nbr16[s]] != track[s]).sum())
    return out


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    N, F_in, x, ei, ea, _ = load_graph()
    data = Data(x=x, edge_index=ei, edge_attr=ea, y=torch.zeros(N, dtype=torch.int8))

    folds = np.load(OUTDIR / "physae_cross_region_folds_v4.npz")
    outA, outB = folds["outliers_A"], folds["outliers_B"]
    inA, inB = folds["inliers_A"], folds["inliers_B"]
    rng = np.random.default_rng(20260606)
    log("loading track_id (evaluation only) ..."); track = load_track()

    def score(model, T, nodes):
        ld = make_loader(data, nodes, FANOUT, EVAL_BATCH, False, NUM_WORKERS, PREFETCH)
        out = []
        with torch.no_grad():
            for b in ld:
                b = b.to(dev); bs = b.batch_size
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
                    out.append(model(b)[:bs].float().cpu().numpy())
        return 1 / (1 + np.exp(-np.concatenate(out) / T))

    report = {"folds": {}}
    # fold A trained on region A -> tested on held-out region B (outliers + inliers), and vice versa
    for fold, test_pos, test_neg in (("A", outB, inB), ("B", outA, inA)):
        mp = OUTDIR / f"physae_model_v4_{fold}.pt"
        if not mp.exists():
            log(f"[{fold}] model missing -> skip"); continue
        st = torch.load(mp, map_location=dev)
        cfg = st.get("config", {})
        model = PhysGNN(F_in, cfg.get("hidden", 256), cfg.get("latent", 128),
                        cfg.get("layers", 3), ea.shape[1], cfg.get("dropout", 0.1)).to(dev)
        model.load_state_dict(st["model"]); model.eval()
        T = st["temperature"]
        inl_eval = test_neg if len(test_neg) <= 50000 else rng.choice(test_neg, 50000, replace=False)
        s_pos = score(model, T, test_pos)
        s_neg = score(model, T, inl_eval)
        yy = np.concatenate([np.ones(len(s_pos)), np.zeros(len(s_neg))])
        ss = np.concatenate([s_pos, s_neg])
        cd = cross_degree(test_pos, track)
        fr = {"n_test_pos": int(len(test_pos)), "auc": roc_auc(yy, ss),
              "recall@0.5": float((s_pos >= 0.5).mean()), "recall@0.7": float((s_pos >= 0.7).mean()),
              "by_cross_degree": {}}
        for lo, hi in BUCKETS:
            m = (cd >= lo) & (cd <= hi)
            if m.any():
                sp = s_pos[m]
                yb = np.concatenate([np.ones(int(m.sum())), np.zeros(len(s_neg))])
                sb = np.concatenate([sp, s_neg])
                fr["by_cross_degree"][f"{lo}-{hi}"] = {
                    "n": int(m.sum()), "recall@0.5": float((sp >= 0.5).mean()),
                    "recall@0.7": float((sp >= 0.7).mean()), "auc": roc_auc(yb, sb)}
        report["folds"][fold] = fr
        log(f"[{fold}] AUC={fr['auc']:.4f} recall@0.5={fr['recall@0.5']*100:.1f}%  "
            f"buckets={ {k: round(v['recall@0.5'], 3) for k, v in fr['by_cross_degree'].items()} }")
    (OUTDIR / "physae_validation_v4.json").write_text(json.dumps(report, indent=2))
    log(f"=== validation done === wrote physae_validation_v4.json ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
