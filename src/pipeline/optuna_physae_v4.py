#!/usr/bin/env python3
"""Optuna hyperparameter search for the Step-4 GNN (physae_gnn_v4).

CHEAP PROXY by design so it is fast: a fixed SUBSAMPLE of the training pool + FEW
epochs per trial. The objective is the CROSS-REGION generalization signal that
actually matters (train on region A, evaluate on the held-out region B) measured
by AVERAGE PRECISION (non-saturating, unlike AUC which sits at ~1.0 and cannot
rank trials). After the study, train the FULL folds with the best config.

Distributed: run several --worker tasks (slurm array) sharing one JournalStorage
on lustre; each grabs a GPU and runs trials against the same study.

Searched: pos_weight, pi, gamma_n, lr, weight_decay, hidden, dropout, fanout.
The outlier-dominant loss is preserved (pos_weight is the outlier-emphasis knob).
"""
import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

sys.path.insert(0, str(Path(__file__).resolve().parent))
from physae_gnn_v4 import PhysGNN, nnpu_loss, make_loader, load_graph  # noqa: E402
from torch_geometric.data import Data  # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]
OUTDIR = PROJECT / "outputs"

# proxy budget (kept small for speed)
PROXY_EPOCHS = 8
PROXY_U = 300_000        # region-A unlabeled sampled for the PU term
PROXY_INL = 100_000      # region-A inlier seeds sampled as trusted negatives
EVAL_NEG = 20_000        # region-B inlier sample used as negatives in the AP objective
BATCH = 4096
EVAL_FANOUT = [16, 8, 8]
NW = 0           # num_workers=0: NO DataLoader IPC -> uses ZERO /dev/shm (large k=16 batches overflow it).
                 # This lets concurrent jobs run without /dev/shm collisions; only RAM limits concurrency.
SEED = 20260606

_DEV = "cuda" if torch.cuda.is_available() else "cpu"
_STATE = {}


def log(m):
    print(m, flush=True)


def setup():
    """Load graph + fixed train/eval subsamples once per worker."""
    if _STATE:
        return
    N, F_in, x, ei, ea, _ = load_graph()
    folds = np.load(OUTDIR / "physae_cross_region_folds_v4.npz")
    outA, inA, outB, inB = folds["outliers_A"], folds["inliers_A"], folds["outliers_B"], folds["inliers_B"]
    region = np.asarray(np.load(OUTDIR / "physae_region_v4.npy", mmap_mode="r"))
    rng = np.random.default_rng(SEED)

    # fixed global labels: region-A outliers=1, region-A inliers=0, rest -1
    y = np.full(N, -1, np.int8); y[inA] = 0; y[outA] = 1
    data = Data(x=x, edge_index=ei, edge_attr=ea, y=torch.from_numpy(y))

    # fixed training subsample (same for every trial -> comparable objective)
    unl_A = np.where((y == -1) & (region == 0))[0]
    inl_tr = rng.choice(inA, size=min(PROXY_INL, len(inA)), replace=False)
    u_tr = rng.choice(unl_A, size=min(PROXY_U, len(unl_A)), replace=False)
    train_pool = np.concatenate([outA, inl_tr, u_tr]).astype(np.int64)

    # fixed eval set on the HELD-OUT region B (outliers vs inlier sample)
    eval_neg = rng.choice(inB, size=min(EVAL_NEG, len(inB)), replace=False)
    eval_ids = np.concatenate([outB, eval_neg]).astype(np.int64)
    eval_y = np.concatenate([np.ones(len(outB)), np.zeros(len(eval_neg))]).astype(np.int8)
    # NOTE: do NOT keep a persistent eval loader — its graph-CSC (~20 GB over 1.2B edges) plus a
    # trial's train-loader CSC = two CSCs at once -> exceeds the cgroup limit -> SIGBUS. We build the
    # eval loader only AFTER freeing the train loader (one CSC at a time).
    _STATE.update(dict(N=N, F_in=F_in, data=data, train_pool=train_pool,
                       eval_ids=eval_ids, eval_y=eval_y, n_eval=len(eval_ids)))
    log(f"setup: train_pool={len(train_pool):,} (outA={len(outA):,}+inl={len(inl_tr):,}+U={len(u_tr):,}) "
        f"eval={len(eval_ids):,} (posB={len(outB):,}) edges={ei.shape[1]:,}")


def eval_ap(model):
    s = _STATE
    ld = make_loader(s["data"], s["eval_ids"], EVAL_FANOUT, 8192, False, num_workers=NW, prefetch=4)
    out = []
    model.eval()
    with torch.no_grad():
        for b in ld:
            b = b.to(_DEV); bs = b.batch_size
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(_DEV == "cuda")):
                out.append(model(b)[:bs].float().cpu().numpy())
    del ld; gc.collect()
    scores = np.concatenate(out)
    return float(average_precision_score(s["eval_y"], scores))


def objective(trial):
    setup()
    s = _STATE
    hp = dict(
        pos_weight=trial.suggest_float("pos_weight", 2.0, 30.0, log=True),
        pi=trial.suggest_float("pi", 1e-3, 1.5e-2, log=True),
        gamma_n=trial.suggest_float("gamma_n", 0.05, 1.0, log=True),
        lr=trial.suggest_float("lr", 3e-4, 3e-3, log=True),
        wd=trial.suggest_float("wd", 1e-5, 1e-3, log=True),
        hidden=trial.suggest_categorical("hidden", [128, 256, 384]),
        dropout=trial.suggest_float("dropout", 0.0, 0.3),
        fanout=trial.suggest_categorical("fanout", ["16,8,8", "12,8,4"]),
    )
    fanout = [int(v) for v in hp["fanout"].split(",")]
    latent = max(64, hp["hidden"] // 2)
    torch.manual_seed(SEED)   # same init each trial -> fair comparison
    model = PhysGNN(s["F_in"], hp["hidden"], latent, 3, s["data"].edge_attr.shape[1], hp["dropout"]).to(_DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["wd"])
    train_loader = make_loader(s["data"], s["train_pool"], fanout, BATCH, True, num_workers=NW, prefetch=4)

    t0 = time.time()
    for ep in range(1, PROXY_EPOCHS + 1):
        model.train()
        for b in train_loader:
            b = b.to(_DEV); bs = b.batch_size
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(_DEV == "cuda")):
                logit = model(b)[:bs]
                loss = nnpu_loss(logit.float(), b.y[:bs], hp["pi"], hp["pos_weight"], hp["gamma_n"])
            loss.backward(); opt.step()
        if ep == 1 or ep == PROXY_EPOCHS:
            log(f"  trial {trial.number} ep {ep}/{PROXY_EPOCHS} ({time.time()-t0:.0f}s)")
    # free the train-loader's graph-CSC BEFORE building the eval loader: only ONE CSC alive at a
    # time, otherwise two ~20 GB CSCs over the 1.2B-edge graph blow the cgroup limit -> SIGBUS.
    del train_loader; gc.collect()
    if _DEV == "cuda":
        torch.cuda.empty_cache()
    ap = eval_ap(model)
    log(f"trial {trial.number} DONE ap={ap:.4f} ({time.time()-t0:.0f}s) {hp}")
    del model, opt; gc.collect()
    if _DEV == "cuda":
        torch.cuda.empty_cache()
    return ap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", default="physae_v4")
    ap.add_argument("--storage", default=str(OUTDIR / "optuna_physae_v4.journal"))
    ap.add_argument("--timeout", type=int, default=3600, help="seconds this worker runs trials")
    ap.add_argument("--n-trials", type=int, default=1000)
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--report", action="store_true", help="just print the best trial and exit")
    args = ap.parse_args()

    # JournalStorage is the NFS/lustre-safe shared backend for distributed workers
    try:
        from optuna.storages.journal import JournalFileBackend
        storage = optuna.storages.JournalStorage(JournalFileBackend(args.storage))
    except Exception as e:  # pragma: no cover - fallback for older layouts
        log(f"journal backend import fallback ({e}); using JournalFileStorage")
        storage = optuna.storages.JournalStorage(optuna.storages.JournalFileStorage(args.storage))

    if args.report:
        st = optuna.load_study(study_name=args.study, storage=storage)
        df_done = [t for t in st.trials if t.value is not None]
        log(f"trials: {len(st.trials)} (completed {len(df_done)})")
        log(f"BEST value(AP)={st.best_value:.4f}")
        log(f"BEST params={st.best_params}")
        return

    study = optuna.create_study(
        study_name=args.study, storage=storage, direction="maximize", load_if_exists=True,
        sampler=TPESampler(seed=SEED, n_startup_trials=10),
        pruner=MedianPruner(n_startup_trials=8, n_warmup_steps=3))
    log(f"worker on {_DEV} ({torch.cuda.get_device_name(0) if _DEV=='cuda' else 'cpu'}); "
        f"study={args.study} timeout={args.timeout}s")
    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout, catch=(RuntimeError,))
    log(f"worker done. study best AP={study.best_value:.4f} params={study.best_params}")


if __name__ == "__main__":
    main()
