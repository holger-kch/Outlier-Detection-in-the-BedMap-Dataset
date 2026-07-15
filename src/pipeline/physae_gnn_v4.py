#!/usr/bin/env python3
"""Step-4 GNN for Bedmap ice-thickness outlier detection.

A SEMI-SUPERVISED GraphSAGE node classifier. It is TRAINED DIRECTLY ON THE SEEDS
(38,881 outlier seeds = positives, 646,674 inlier seeds = trusted negatives) and
generalizes the score to ALL 74,747,031 points: because every node's prediction
is built by message passing over its physical neighbourhood, the learned function
applies to every node, including the ~75% with no cross-track neighbours and the
vast unlabeled mass.

The ~74M unlabeled points are handled with nnPU (positive-unlabeled) learning:
they are NEVER treated as negatives; instead they enter the PU risk estimator so
the model does not overfit the seed regions. Both seed classes set the decision
boundary (inlier seeds also pin real-but-extreme physics — deep troughs, shelves,
ice streams — as GOOD).

No BedMachine. No Step1+2 cone/octant features. Node features are fresh intrinsic
physics; coordinates build edges only. Graph = the full cached k=16 east/north map.

Modes:
  --fold {A,B,full}   train (A/B = cross-region generalization folds; full = all seeds)
  --infer             score all 74,747,031 points (full model + fold A/B agreement)
Crash-safe: checkpoints every --ckpt-every epochs; --resume is DEFAULT.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import rankdata
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import MessagePassing

PROJECT = Path(__file__).resolve().parents[1]
OUTDIR = PROJECT / "outputs"

# Full cached k=16 east/north map (reused directly; node-major [neighbour, node]).
EDGE_INDEX = OUTDIR / "spatial_edge_index_v3_k16.npy"
EDGE_ATTR = OUTDIR / "physae_edge_attr_v4_k16.npy"


def log(m):
    print(m, flush=True)


def write_json_atomic(path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


def torch_save_atomic(obj, path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


# ----------------------------------------------------------------- model
class EdgeGatedSAGE(MessagePassing):
    """SAGE-style conv with an edge-feature gate: message = MLP(x_j) * sigmoid(W_e . edge_attr).
    edge_attr = [log1p(distance), signed thickness gradient] -> lets a monotonic ramp
    (real trough/outlet) and a self-reversing spike (error) be weighted differently."""

    def __init__(self, in_ch, out_ch, edge_dim):
        super().__init__(aggr="mean")
        self.lin_msg = nn.Linear(in_ch, out_ch)
        self.lin_self = nn.Linear(in_ch, out_ch)
        self.edge_gate = nn.Linear(edge_dim, out_ch)

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr) + self.lin_self(x)

    def message(self, x_j, edge_attr):
        return self.lin_msg(x_j) * torch.sigmoid(self.edge_gate(edge_attr))


class PhysGNN(nn.Module):
    """Semi-supervised node classifier: EdgeGatedSAGE encoder + JumpingKnowledge + a head."""

    def __init__(self, f_in, hidden=128, latent=96, layers=3, edge_dim=2, drop=0.1):
        super().__init__()
        self.drop = drop
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        d = f_in
        dims = []
        for li in range(layers):
            out = hidden if li < layers - 1 else latent
            self.convs.append(EdgeGatedSAGE(d, out, edge_dim))
            self.norms.append(nn.LayerNorm(out))
            dims.append(out)
            d = out
        self.jk = nn.Linear(sum(dims), latent)             # JumpingKnowledge('cat')
        self.head = nn.Sequential(nn.Linear(latent, 64), nn.GELU(),
                                  nn.Dropout(drop), nn.Linear(64, 1))

    def forward(self, b):
        h = b.x.float(); ei = b.edge_index; ea = b.edge_attr.float()
        outs = []
        for li, (c, n) in enumerate(zip(self.convs, self.norms)):
            hn = F.dropout(F.gelu(n(c(h, ei, ea))), p=self.drop, training=self.training)
            if li > 0 and hn.shape == h.shape:
                hn = hn + h                                # residual where dims match
            outs.append(hn); h = hn
        z = self.jk(torch.cat(outs, dim=-1))
        return self.head(z).squeeze(-1)                    # logit per node


def nnpu_loss(logit, y, pi, pos_weight, gamma_n, share_lo=0.0, share_hi=1.0, return_share=False):
    """Outlier-favouring (cost-sensitive) nnPU with an OPTIONAL guaranteed outlier loss-SHARE band.
    P = outlier seeds, N = inlier seeds (trusted neg), U = unlabeled (PU mixture, never 'negative').
    If share_lo/share_hi are set (e.g. 0.5/0.8), the outlier term's effective weight is clamped each
    batch so that pos_term / total stays in [share_lo, share_hi] -> outliers stay dominant but never
    100%. The adaptive weight is DETACHED (a coefficient, not something the model optimises through)."""
    sp_pos = F.softplus(-logit)      # loss if treated positive  (l+)
    sp_neg = F.softplus(logit)       # loss if treated negative  (l-)
    P = y == 1; U = y == -1; Ng = y == 0
    Rp_plus = sp_pos[P].mean() if P.any() else logit.new_tensor(0.0)
    Rp_minus = sp_neg[P].mean() if P.any() else logit.new_tensor(0.0)
    Ru_minus = sp_neg[U].mean() if U.any() else logit.new_tensor(0.0)
    Rn_minus = sp_neg[Ng].mean() if Ng.any() else logit.new_tensor(0.0)
    neg = torch.clamp(Ru_minus - pi * Rp_minus, min=0.0) + gamma_n * Rn_minus   # all negative pressure
    w = logit.new_tensor(float(pos_weight))
    if share_lo > 0.0 or share_hi < 1.0:
        eps = 1e-8
        negd = neg.detach(); rpd = Rp_plus.detach().clamp(min=eps)
        wl = (share_lo / (1.0 - share_lo)) * (negd / rpd)   # weight giving share == share_lo
        wh = (share_hi / (1.0 - share_hi)) * (negd / rpd)   # weight giving share == share_hi
        w = torch.minimum(torch.maximum(w, wl), wh)         # detached coefficient -> share in [lo,hi]
    pos = w * Rp_plus
    loss = pos + neg
    if return_share:
        return loss, (pos.detach() / (loss.detach() + 1e-8))
    return loss


# ----------------------------------------------------------------- data
def load_graph():
    feat_stats = json.loads((OUTDIR / "physae_feature_stats_v4.json").read_text())
    N, F_in = feat_stats["N"], feat_stats["F_in"]
    x = torch.from_numpy(np.load(OUTDIR / "physae_node_features_v4.npy", mmap_mode="r"))
    ei = torch.from_numpy(np.load(EDGE_INDEX, mmap_mode="r"))   # int64 [neighbour, node], k=16
    ea = torch.from_numpy(np.load(EDGE_ATTR, mmap_mode="r"))
    log(f"graph: N={N:,} F_in={F_in} edges={ei.shape[1]:,} (k=16)")
    return N, F_in, x, ei, ea, feat_stats


def make_loader(data, nodes, fanout, bs, shuffle, num_workers=0, prefetch=4):
    kw = dict(num_neighbors=fanout, input_nodes=torch.as_tensor(nodes, dtype=torch.long),
              batch_size=bs, shuffle=shuffle, num_workers=num_workers)
    if num_workers > 0:                       # async prefetch so CPU sampling overlaps GPU compute (no GPU starvation)
        kw["persistent_workers"] = True
        kw["prefetch_factor"] = prefetch
        kw["pin_memory"] = True
    return NeighborLoader(data, **kw)


def roc_auc(y, s):
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def fit_temperature(logit, y, iters=200):
    lg = torch.tensor(logit, dtype=torch.float32); yt = torch.tensor(y, dtype=torch.float32)
    logT = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=iters)

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(lg / torch.exp(logT), yt)
        loss.backward(); return loss
    opt.step(closure)
    return float(torch.exp(logT).item())


def cosine_lr_at_epoch(base_lr, epoch, t_max, eta_min=0.0):
    return eta_min + (base_lr - eta_min) * (1.0 + np.cos(np.pi * epoch / t_max)) / 2.0


# ----------------------------------------------------------------- train
def train(args):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        log(f"GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory/1e9:.0f} GB)")
    fanout = [int(s) for s in args.fanout.split(",")]
    assert len(fanout) == args.layers
    t0 = time.time()
    N, F_in, x, ei, ea, _ = load_graph()

    folds = np.load(OUTDIR / "physae_cross_region_folds_v4.npz")
    outA, outB = folds["outliers_A"], folds["outliers_B"]
    inA, inB = folds["inliers_A"], folds["inliers_B"]
    rng = np.random.default_rng(args.seed)
    # 2-cross-fold: train on ONE region of Antarctica, predict on the OTHER (held-out region
    # fully unseen: its outliers, inliers AND unlabeled are all excluded from training).
    if args.fold == "A":
        pos_all, inl_all, test_pos, test_neg, train_region = outA, inA, outB, inB, 0
    elif args.fold == "B":
        pos_all, inl_all, test_pos, test_neg, train_region = outB, inB, outA, inA, 1
    else:  # full: train on all seeds (both regions), no held-out test -> used for final inference
        pos_all = np.concatenate([outA, outB]); inl_all = np.concatenate([inA, inB])
        test_pos = np.array([], np.int64); test_neg = np.array([], np.int64); train_region = None

    pperm = rng.permutation(len(pos_all)); ncal = max(1, len(pos_all) // 10)
    pos_cal = pos_all[pperm[:ncal]]; pos_tr = pos_all[pperm[ncal:]]
    iperm = rng.permutation(len(inl_all)); niv = max(1, len(inl_all) // 5)
    inl_cal = inl_all[iperm[:niv]]; inl_tr = inl_all[iperm[niv:]]

    # global labels: 1 = outlier seed, 0 = inlier seed, -1 = unlabeled.
    # Only pos_tr/inl_tr enter train_nodes; pos_cal/inl_cal are held out for validation loss.
    y = np.full(N, -1, np.int8)
    y[inl_tr] = 0
    y[pos_tr] = 1
    y[inl_cal] = 0
    y[pos_cal] = 1
    data = Data(x=x, edge_index=ei, edge_attr=ea, y=torch.from_numpy(y))
    unl_all = np.where(y == -1)[0]
    if train_region is not None:    # strict cross-region: unlabeled pool from the TRAINING region only
        region = np.load(OUTDIR / "physae_region_v4.npy", mmap_mode="r")
        unl_all = unl_all[np.asarray(region[unl_all]) == train_region]
    log(f"[{args.fold}] pos_tr={len(pos_tr):,} pos_cal={len(pos_cal):,} inl_tr={len(inl_tr):,} "
        f"inl_cal={len(inl_cal):,} test_pos={len(test_pos):,} test_neg={len(test_neg):,} U_pool={len(unl_all):,}")

    model = PhysGNN(F_in, args.hidden, args.latent, args.layers, ea.shape[1], args.dropout).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()}
    ckpt = OUTDIR / f"physae_ckpt_v4_{args.fold}.pt"
    best_ckpt = OUTDIR / f"physae_best_v4_{args.fold}.pt"
    hist_path = OUTDIR / f"physae_train_v4_{args.fold}_history.json"
    history = []
    best_val_loss = float("inf")
    best_epoch = 0
    no_improve = 0
    start = 0
    if args.resume and ckpt.exists():
        st = torch.load(ckpt, map_location=dev)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"]); ema = st["ema"]; start = st["epoch"]
        history = st.get("history", [])
        finite_val_rows = [r for r in history if r.get("val_loss") is not None]
        if finite_val_rows:
            best_row = min(finite_val_rows, key=lambda r: r["val_loss"])
            best_val_loss = float(best_row["val_loss"])
            best_epoch = int(best_row["epoch"])
            no_improve = max(0, start - best_epoch)
        log(f"[{args.fold}] resumed from epoch {start}")
        if args.epochs != sched.T_max:
            old_tmax = sched.T_max
            sched.T_max = args.epochs
            sched.base_lrs = [float(args.lr) for _ in opt.param_groups]
            sched.last_epoch = start
            for pg, base_lr in zip(opt.param_groups, sched.base_lrs):
                pg["initial_lr"] = base_lr
                pg["lr"] = float(cosine_lr_at_epoch(base_lr, start, args.epochs, sched.eta_min))
            sched._last_lr = [pg["lr"] for pg in opt.param_groups]
            log(f"[{args.fold}] extended scheduler T_max {old_tmax} -> {args.epochs}; "
                f"lr@epoch{start}={sched._last_lr[0]:.8g}")

    # Build the NeighborLoader ONCE: the CSC build over ~1.2B edges is expensive, so we must
    # NOT rebuild it per epoch. U is sampled once (a fixed pool); shuffle=True reshuffles order each epoch.
    u = rng.choice(unl_all, size=min(args.n_unlabeled, len(unl_all)), replace=False)
    train_nodes = np.concatenate([pos_tr, inl_tr, u])
    log(f"[{args.fold}] building NeighborLoader over {len(train_nodes):,} train seeds "
        f"(one-time CSC build over {data.edge_index.shape[1]:,} edges) ...")
    tb = time.time()
    train_loader = make_loader(data, train_nodes, fanout, args.batch_size, True, args.num_workers, args.prefetch)
    log(f"[{args.fold}] loader ready ({time.time()-tb:.0f}s, num_workers={args.num_workers})")

    val_loader = None
    if args.val_every > 0:
        val_nodes = np.concatenate([pos_cal, inl_cal]).astype(np.int64, copy=False)
        log(f"[{args.fold}] building calibration NeighborLoader over {len(val_nodes):,} validation seeds ...")
        tb = time.time()
        val_loader = make_loader(data, val_nodes, fanout, args.eval_batch, False, args.num_workers, args.prefetch)
        log(f"[{args.fold}] validation loader ready ({time.time()-tb:.0f}s)")

    def validation_loss():
        assert val_loader is not None
        model.eval()
        logits, labels = [], []
        with torch.no_grad():
            for b in val_loader:
                b = b.to(dev); bs = b.batch_size
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
                    logits.append(model(b)[:bs].float().cpu())
                labels.append(b.y[:bs].cpu())
        return float(nnpu_loss(torch.cat(logits), torch.cat(labels), args.pi, args.pos_weight,
                               args.gamma_n, args.share_lo, args.share_hi))

    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for ep in range(start + 1, args.epochs + 1):
        model.train()
        te = time.time(); tot = 0.0; nb = 0; shs = torch.zeros((), device=dev)
        for b in train_loader:
            b = b.to(dev); bs = b.batch_size
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
                logit = model(b)[:bs]
                loss, sh = nnpu_loss(logit.float(), b.y[:bs], args.pi, args.pos_weight,
                                     args.gamma_n, args.share_lo, args.share_hi, return_share=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            with torch.no_grad():
                for k, v in model.state_dict().items():
                    ema[k].mul_(args.ema).add_(v.detach(), alpha=1 - args.ema)
            tot += float(loss); shs += sh.detach(); nb += 1
        train_seconds = time.time() - te
        train_loss = tot / max(nb, 1)
        share = 100 * float(shs) / max(nb, 1)
        lr_used = opt.param_groups[0]["lr"]
        val_loss = validation_loss() if args.val_every > 0 and (ep % args.val_every == 0 or ep == args.epochs) else None
        val_loss_txt = f"{val_loss:.4f}" if val_loss is not None else "nan"
        sched.step()
        log(f"[{args.fold}] ep {ep}/{args.epochs} loss={train_loss:.4f} val_loss={val_loss_txt} "
            f"outlier_share={share:.0f}% ({train_seconds:.0f}s)")
        history.append({"epoch": ep, "train_loss": train_loss, "val_loss": val_loss,
                        "outlier_share": share, "lr": lr_used, "epoch_seconds": train_seconds})
        write_json_atomic(hist_path, history)
        if val_loss is not None:
            if val_loss < best_val_loss - args.early_stop_min_delta:
                best_val_loss = val_loss
                best_epoch = ep
                no_improve = 0
                torch_save_atomic({"model": model.state_dict(), "opt": opt.state_dict(),
                                   "sched": sched.state_dict(), "ema": ema, "epoch": ep,
                                   "config": vars(args), "history": history,
                                   "best_val_loss": best_val_loss, "best_epoch": best_epoch}, best_ckpt)
                log(f"[{args.fold}] best validation loss {best_val_loss:.4f} @ ep {best_epoch}")
            else:
                no_improve += 1
        if ep % args.ckpt_every == 0 or ep == args.epochs:
            torch_save_atomic({"model": model.state_dict(), "opt": opt.state_dict(),
                               "sched": sched.state_dict(), "ema": ema, "epoch": ep,
                               "config": vars(args), "history": history,
                               "best_val_loss": best_val_loss, "best_epoch": best_epoch}, ckpt)
            log(f"[{args.fold}] checkpoint @ ep {ep}")
        if args.early_stop_patience > 0 and val_loss is not None and no_improve >= args.early_stop_patience:
            log(f"[{args.fold}] early stopping after {no_improve} epochs without val_loss improvement "
                f"(best ep {best_epoch}, val_loss={best_val_loss:.4f})")
            break

    if best_ckpt.exists() and args.val_every > 0:
        st_best = torch.load(best_ckpt, map_location=dev)
        ema = st_best["ema"]
        log(f"[{args.fold}] using best validation checkpoint ep {st_best['epoch']} "
            f"(val_loss={st_best.get('best_val_loss', float('nan')):.4f})")
    model.load_state_dict(ema); model.eval()

    def score_ids(ids):
        ld = make_loader(data, ids, fanout, args.eval_batch, False, args.num_workers, args.prefetch)
        out = []
        with torch.no_grad():
            for b in ld:
                b = b.to(dev); bs = b.batch_size
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
                    out.append(model(b)[:bs].float().cpu())
        return torch.cat(out).numpy()

    # score every node needed for calibration + cross-region in ONE loader (one extra CSC build, not 3)
    neg_eval = (test_neg if len(test_neg) <= 100_000 else rng.choice(test_neg, 100_000, replace=False)) \
        if len(test_pos) else np.array([], np.int64)
    eval_ids = np.unique(np.concatenate([pos_cal, inl_cal, test_pos, neg_eval]).astype(np.int64))
    eval_log = score_ids(eval_ids)
    lmap = {int(i): float(v) for i, v in zip(eval_ids.tolist(), eval_log.tolist())}
    def g(ids):
        return np.array([lmap[int(i)] for i in ids], np.float32)

    T = fit_temperature(np.concatenate([g(pos_cal), g(inl_cal)]),
                        np.concatenate([np.ones(len(pos_cal)), np.zeros(len(inl_cal))]))
    log(f"[{args.fold}] temperature T={T:.4f}")

    result = {"fold": args.fold, "epochs": args.epochs, "k": 16, "fanout": fanout,
              "pi": args.pi, "pos_weight": args.pos_weight, "gamma_n": args.gamma_n, "temperature": T,
              "peak_gpu_gb": (torch.cuda.max_memory_allocated() / 1e9) if dev == "cuda" else 0.0,
              "history": history}
    if len(test_pos):   # cross-region generalization test on the HELD-OUT region (fully unseen)
        s_test = 1 / (1 + np.exp(-g(test_pos) / T))
        s_inl = 1 / (1 + np.exp(-g(neg_eval) / T))
        yy = np.concatenate([np.ones(len(s_test)), np.zeros(len(s_inl))])
        ss = np.concatenate([s_test, s_inl])
        for thr in (0.5, 0.7):
            pred = ss >= thr
            tp = int((pred & (yy == 1)).sum()); fn = int((~pred & (yy == 1)).sum())
            fp = int((pred & (yy == 0)).sum()); tn = int((~pred & (yy == 0)).sum())
            result[f"recall@{thr}"] = tp / max(tp + fn, 1)
            result[f"fpr@{thr}"] = fp / max(fp + tn, 1)
        result["cross_region_auc"] = roc_auc(yy, ss)
        result["n_test_pos"] = int(len(test_pos)); result["n_test_neg"] = int(len(neg_eval))
        log(f"[{args.fold}] CROSS-REGION AUC={result['cross_region_auc']:.4f} "
            f"recall@0.5={result['recall@0.5']*100:.1f}% recall@0.7={result['recall@0.7']*100:.1f}%")
    (OUTDIR / f"physae_train_v4_{args.fold}.json").write_text(json.dumps(result, indent=2))
    torch.save({"model": ema, "temperature": T, "config": vars(args)},
               OUTDIR / f"physae_model_v4_{args.fold}.pt")
    log(f"[{args.fold}] done; total {time.time()-t0:.0f}s")


# ----------------------------------------------------------------- infer (cross-fit / out-of-fold)
def infer(args):
    """CROSS-LABELLING: every point is scored by the fold model trained on the OTHER region,
    so no point is ever scored by a model that trained on its own region's labels (no
    training-on-final-labels). region A -> model B; region B -> model A."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    fanout = [int(s) for s in args.fanout.split(",")]
    t0 = time.time()
    N, F_in, x, ei, ea, _ = load_graph()
    data = Data(x=x, edge_index=ei, edge_attr=ea, y=torch.zeros(N, dtype=torch.int8))
    region = np.asarray(np.load(OUTDIR / "physae_region_v4.npy", mmap_mode="r"))   # 0=A, 1=B

    def load_fold(f):
        mp = OUTDIR / f"physae_model_v4_{f}.pt"
        assert mp.exists(), f"need {mp} — cross-fit inference requires BOTH fold A and B models"
        st = torch.load(mp, map_location=dev)
        cfg = st.get("config", {})
        m = PhysGNN(F_in, cfg.get("hidden", args.hidden), cfg.get("latent", args.latent),
                    cfg.get("layers", args.layers), ea.shape[1], cfg.get("dropout", args.dropout)).to(dev)
        m.load_state_dict(st["model"]); m.eval()
        log(f"loaded fold {f} (T={st['temperature']:.3f})")
        return m, st["temperature"]
    mA, TA = load_fold("A"); mB, TB = load_fold("B")

    p_oof = np.empty(N, np.float32)                 # out-of-fold cross-fitted score (HEADLINE)
    p_A = np.empty(N, np.float32); p_B = np.empty(N, np.float32)
    loader = make_loader(data, np.arange(N, dtype=np.int64), fanout, args.infer_batch, False, args.num_workers, args.prefetch)
    done = 0
    with torch.no_grad():
        for b in loader:
            b = b.to(dev); bs = b.batch_size
            ids = b.n_id[:bs].cpu().numpy()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
                lgA = mA(b)[:bs].float().cpu().numpy()
                lgB = mB(b)[:bs].float().cpu().numpy()
            sA = 1 / (1 + np.exp(-lgA / TA)); sB = 1 / (1 + np.exp(-lgB / TB))
            p_A[ids] = sA; p_B[ids] = sB
            reg = region[ids]
            p_oof[ids] = np.where(reg == 0, sB, sA)   # region A scored by B; region B scored by A
            done += bs
            if done % (args.infer_batch * 200) < args.infer_batch:
                log(f"  scored {done:,}/{N:,} ({time.time()-t0:.0f}s)")

    agreement = 1.0 - np.abs(p_A - p_B)
    tbl = pa.table({"row_id": np.arange(N, dtype=np.int64), "p_outlier": p_oof,
                    "score_A": p_A, "score_B": p_B, "agreement": agreement,
                    "region": region.astype(np.int8)})
    outp = OUTDIR / "physae_scores_v4.parquet"
    pq.write_table(tbl, outp)
    log(f"=== infer done (cross-fit) === wrote {outp}  ({time.time()-t0:.0f}s)")
    log(f"p_outlier (out-of-fold): mean={p_oof.mean():.4f} >0.5={int((p_oof>0.5).sum()):,} >0.7={int((p_oof>0.7).sum()):,}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", choices=["A", "B", "full"], default="full")
    ap.add_argument("--infer", action="store_true")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--latent", type=int, default=128)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--fanout", default="16,8,8")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--eval-batch", type=int, default=8192)
    ap.add_argument("--infer-batch", type=int, default=8192)
    ap.add_argument("--num-workers", type=int, default=8, help="NeighborLoader workers (async sampling so the GPU isn't starved)")
    ap.add_argument("--prefetch", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--pi", type=float, default=3e-3, help="positive class prior (only in the PU negative correction)")
    ap.add_argument("--pos-weight", type=float, default=8.0, help="OUTLIER emphasis: weight on the positive term so outliers dominate the loss")
    ap.add_argument("--gamma-n", type=float, default=0.3, help="weight on the trusted-negative (inlier) term; small = low inlier dependence")
    ap.add_argument("--share-lo", type=float, default=0.0, help="min outlier loss-share (e.g. 0.5); 0 = off")
    ap.add_argument("--share-hi", type=float, default=1.0, help="max outlier loss-share (e.g. 0.8); 1 = off")
    ap.add_argument("--grad-clip", type=float, default=5.0, help="max grad norm (stabilises the share-band lower clamp); 0 = off")
    ap.add_argument("--n-unlabeled", type=int, default=3_000_000, help="unlabeled nodes sampled per epoch for the PU U-term")
    ap.add_argument("--val-every", type=int, default=1, help="compute validation loss on calibration seeds every N epochs; 0 disables")
    ap.add_argument("--early-stop-patience", type=int, default=10, help="stop after this many validation checks without improvement; 0 disables")
    ap.add_argument("--early-stop-min-delta", type=float, default=1e-4, help="minimum validation-loss improvement counted by early stopping")
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--ckpt-every", type=int, default=5)
    ap.add_argument("--resume", dest="resume", action="store_true", default=True)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    ap.add_argument("--seed", type=int, default=20260606)
    args = ap.parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    if args.infer:
        infer(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
