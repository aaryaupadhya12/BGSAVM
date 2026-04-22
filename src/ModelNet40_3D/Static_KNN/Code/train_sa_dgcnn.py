"""
Train SA_DGCNN on ModelNet40.

Combines Anshull's DGCNN training pipeline (dynamic k-NN EdgeConv backbone,
AMP, cosine LR, label smoothing, grad-clip, per-batch augmentation) with the
spreading-activation pruner from ``sa_dgcnn.py``.

Key differences from ``train_spreading_activation_gcn.py``:

* Uses ``torch_cluster.knn_graph`` — dynamic graph rebuilt in feature space
  at every block (matches DGCNN's identity).
* Honest 90/10 stratified train/val split — model selection never touches
  the test set.  Test OA is evaluated once at the best-val epoch.
* AMP autocast + grad scaler, gradient clipping, cosine annealing,
  label smoothing, early stopping.
* Logs learnable SA hyperparameters each epoch so you can verify the module
  is actually adapting.
* Saves FLOPs / latency / throughput / memory at end of training if
  ``torch.profiler`` is available.

Usage (from ``Src/ModelNet40_3D/Static_KNN/Code/``):

    python train_sa_dgcnn.py --cache-path modelnet40_final.pt \\
        --epochs 200 --batch-size 32 --prune-ratio 0.7 --sa-learnable
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm.auto import tqdm, trange

from sa_dgcnn import SA_DGCNN

ROOT = Path(__file__).resolve().parent
DEFAULT_CACHE = ROOT / "modelnet40_final.pt"
DEFAULT_OUTPUT_DIR = ROOT / "runs" / "sa_dgcnn"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────── args ──
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--run-name", type=str, default=None)
    # training
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--weight-decay", type=float, default=5e-5)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--patience", type=int, default=40)
    p.add_argument("--eval-every", type=int, default=1)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    # model
    p.add_argument("--k-neighbors", type=int, default=20)
    p.add_argument("--in-channels", type=int, default=7,
                   help="7 = pos(3)+norm(3)+motif(1). Use 6 to drop motif.")
    p.add_argument("--embed-dims", type=int, default=1024)
    p.add_argument("--small", action="store_true")
    # spreading activation
    p.add_argument("--gate-mode", choices=("soft", "hard", "off"), default="soft",
                   help="soft=multiplicative gate (preserves multiscale concat); "
                        "hard=legacy top-k prune; off=baseline DGCNN (no SA).")
    p.add_argument("--gate-floor", type=float, default=0.1,
                   help="Minimum gate value in soft mode — low-salience nodes "
                        "are attenuated but not dropped.")
    p.add_argument("--gate-points", type=str, default="2",
                   help="Comma-separated conv indices after which to gate. "
                        "'2' = gate after conv2 (default); '2,3' = pyramidal. "
                        "Only soft mode supports pyramidal.")
    p.add_argument("--prune-ratio", type=float, default=0.7,
                   help="Only used when --gate-mode=hard.")
    p.add_argument("--prune-min-nodes", type=int, default=307,
                   help="Minimum surviving nodes per graph in hard-prune mode. "
                        "Default ~0.3 * 1024 — raise the floor so BN stats stay stable.")
    p.add_argument("--sa-steps", type=int, default=2)
    p.add_argument("--sa-learnable", action="store_true", default=True)
    p.add_argument("--no-sa-learnable", dest="sa_learnable", action="store_false")
    p.add_argument("--sa-reweight-each-step", action="store_true")
    p.add_argument("--saliency-scale", type=float, default=0.25,
                   help="Only used when --gate-mode=hard.")
    p.add_argument("--seed-mix-init", type=float, default=0.5)
    # test-time augmentation
    p.add_argument("--tta-rotations", type=int, default=1,
                   help="Number of evenly-spaced Y-axis rotations to average "
                        "at test time. 1 = no TTA; 12 is the classic setting.")
    return p.parse_args()


# ─────────────────────────────────────────── data loading / motifs ──
def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_cache(path: Path) -> tuple[list[Data], list[Data], list[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Cache not found: {path}")
    cache = torch.load(path, map_location="cpu", weights_only=False)
    return cache["train"], cache["test"], cache["classes"]


def _triangle_counts(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    # Dense for 1024-node clouds is acceptable; upgrade to torch_sparse
    # if scaling to larger graphs.
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    adj[edge_index[0], edge_index[1]] = 1.0
    adj[edge_index[1], edge_index[0]] = 1.0
    counts = (adj * (adj @ adj)).sum(dim=1) / 2.0
    m = counts.max()
    return counts / m if m > 0 else counts


def ensure_motif_attribute(data_list: list[Data], k: int) -> list[Data]:
    """Attach ``data.motif`` as a named attribute.  Idempotent."""
    from scipy.spatial import cKDTree
    for data in tqdm(data_list, desc="Motif (idempotent)", leave=False):
        if getattr(data, "motif", None) is not None and data.x.size(1) >= 7:
            continue
        pos = data.pos.cpu().numpy()
        tree = cKDTree(pos)
        _, nn_idx = tree.query(pos, k=k + 1)
        nn_idx = nn_idx[:, 1:]
        src = np.repeat(np.arange(pos.shape[0]), k)
        dst = nn_idx.flatten()
        ei = torch.tensor(np.stack([src, dst]), dtype=torch.long)
        motif = _triangle_counts(ei, pos.shape[0]).float()
        # Only append to x if not already present (x[:, 6] is the legacy slot).
        if data.x.size(1) < 7:
            data.x = torch.cat([data.x.float(), motif.unsqueeze(1)], dim=1)
        data.motif = motif
        data.edge_index = ei   # keep cached static graph for reference
    return data_list


def stratified_train_val_split(
    train_list: list[Data],
    val_frac: float,
    seed: int,
) -> tuple[list[Data], list[Data]]:
    rng = np.random.default_rng(seed)
    by_class: dict[int, list[int]] = defaultdict(list)
    for i, d in enumerate(train_list):
        by_class[int(d.y.item())].append(i)

    val_idx: list[int] = []
    for cls, idxs in by_class.items():
        n_val = max(1, int(round(len(idxs) * val_frac)))
        chosen = rng.choice(idxs, size=n_val, replace=False)
        val_idx.extend(chosen.tolist())
    val_set = set(val_idx)
    val = [train_list[i] for i in sorted(val_set)]
    trn = [d for i, d in enumerate(train_list) if i not in val_set]
    return trn, val


# ─────────────────────────────────────────────────── augmentation ──
def augment_batch(batch: Any) -> Any:
    """Per-graph Y-axis rotation + per-graph scale + per-node jitter."""
    device = batch.pos.device
    B = batch.num_graphs

    theta = torch.rand(B, device=device) * (2 * math.pi)
    cos_t, sin_t = torch.cos(theta), torch.sin(theta)
    zeros = torch.zeros(B, device=device)
    ones = torch.ones(B, device=device)
    R = torch.stack([
        torch.stack([cos_t,  zeros, sin_t], dim=1),
        torch.stack([zeros,  ones,  zeros], dim=1),
        torch.stack([-sin_t, zeros, cos_t], dim=1),
    ], dim=1)  # [B, 3, 3]

    R_per_node = R[batch.batch]
    pos = torch.bmm(batch.pos.unsqueeze(1), R_per_node.transpose(1, 2)).squeeze(1)

    scale = 0.8 + torch.rand(B, device=device) * 0.45
    pos = pos * scale[batch.batch].unsqueeze(1)
    pos = pos + (torch.randn_like(pos) * 0.02).clamp(-0.05, 0.05)

    batch.pos = pos
    # in_channels == 7 → [pos, norm, motif].  Rotate pos + normals; leave motif alone.
    if batch.x.size(1) >= 6:
        batch.x[:, :3] = pos
        batch.x[:, 3:6] = torch.bmm(
            batch.x[:, 3:6].unsqueeze(1), R_per_node.transpose(1, 2)
        ).squeeze(1)
    return batch


# ─────────────────────────────────────────────── train / eval loop ──
def train_one_epoch(
    model, loader, optimizer, scaler, device, *,
    label_smoothing: float, grad_clip: float, use_amp: bool, do_augment: bool,
) -> tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for batch in tqdm(loader, desc="  train", leave=False):
        batch = batch.to(device, non_blocking=True)
        if do_augment:
            batch = augment_batch(batch)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(batch)
            labels = batch.y.view(-1)
            loss = F.cross_entropy(logits, labels, label_smoothing=label_smoothing)

        if use_amp:
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += float(loss.item()) * batch.num_graphs
        correct    += int(logits.argmax(1).eq(labels).sum().item())
        total      += batch.num_graphs
    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, device, use_amp: bool) -> dict[str, float]:
    model.eval()
    preds, labels = [], []
    for batch in tqdm(loader, desc="  eval ", leave=False):
        batch = batch.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(batch)
        preds.append(logits.argmax(1).cpu())
        labels.append(batch.y.view(-1).cpu())
    p = torch.cat(preds).numpy()
    y = torch.cat(labels).numpy()
    return {
        "oa":       accuracy_score(y, p) * 100.0,
        "macc":     balanced_accuracy_score(y, p) * 100.0,
        "macro_f1": f1_score(y, p, average="macro", zero_division=0) * 100.0,
    }


def _rotate_batch_y(batch: Any, theta: float) -> Any:
    """Apply a fixed Y-axis rotation (theta radians) to a PyG batch in-place.

    Unlike ``augment_batch``, this is deterministic and rotation-only — no
    scale or jitter — so it's safe for test-time ensemble averaging.
    """
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    R = torch.tensor(
        [[cos_t, 0.0, sin_t], [0.0, 1.0, 0.0], [-sin_t, 0.0, cos_t]],
        device=batch.pos.device, dtype=batch.pos.dtype,
    )
    batch.pos = batch.pos @ R.T
    if batch.x.size(1) >= 6:
        batch.x[:, :3] = batch.pos
        batch.x[:, 3:6] = batch.x[:, 3:6] @ R.T
    return batch


@torch.no_grad()
def evaluate_tta(
    model, loader, device, use_amp: bool, n_rotations: int
) -> dict[str, float]:
    """Test-time augmentation via Y-axis rotation averaging.

    Runs ``n_rotations`` forward passes with evenly-spaced Y rotations, averages
    softmax probabilities, and returns the metrics.  ``n_rotations=1`` is
    equivalent to ``evaluate`` and is a no-op.
    """
    if n_rotations <= 1:
        return evaluate(model, loader, device, use_amp=use_amp)
    model.eval()
    preds, labels = [], []
    angles = [2 * math.pi * i / n_rotations for i in range(n_rotations)]
    for batch in tqdm(loader, desc=f"  eval x{n_rotations}", leave=False):
        batch = batch.to(device, non_blocking=True)
        labels.append(batch.y.view(-1).cpu())
        # Snapshot original pos/x so each rotation starts from the same input.
        pos0 = batch.pos.clone()
        x0 = batch.x.clone()
        prob_sum = None
        for theta in angles:
            batch.pos = pos0.clone()
            batch.x = x0.clone()
            batch = _rotate_batch_y(batch, theta)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(batch)
            probs = F.softmax(logits.float(), dim=1)
            prob_sum = probs if prob_sum is None else prob_sum + probs
        preds.append((prob_sum / n_rotations).argmax(1).cpu())
    p = torch.cat(preds).numpy()
    y = torch.cat(labels).numpy()
    return {
        "oa":       accuracy_score(y, p) * 100.0,
        "macc":     balanced_accuracy_score(y, p) * 100.0,
        "macro_f1": f1_score(y, p, average="macro", zero_division=0) * 100.0,
    }


# ───────────────────────────────────────────────────── efficiency ──
@torch.no_grad()
def measure_efficiency(model, sample: Data, device: torch.device) -> dict[str, Any]:
    model.eval()
    data = copy.deepcopy(sample).to(device)
    # Ensure a batch attribute for pooling
    if not hasattr(data, "batch") or data.batch is None:
        data.batch = torch.zeros(data.pos.size(0), dtype=torch.long, device=device)

    # Warm-up
    for _ in range(3):
        model(data)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Latency
    if device.type == "cuda":
        evts = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                for _ in range(50)]
        for s, e in evts:
            s.record(); model(data); e.record()
        torch.cuda.synchronize()
        times = np.array([s.elapsed_time(e) for s, e in evts])
    else:
        times = []
        for _ in range(50):
            t0 = time.perf_counter()
            model(data)
            times.append((time.perf_counter() - t0) * 1000.0)
        times = np.array(times)

    # Memory
    peak_mib = 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
        model(data)
        torch.cuda.synchronize()
        peak_mib = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    # FLOPs (best-effort — knn_graph C++ ops excluded)
    gflops = None
    try:
        act = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            act.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(activities=act, with_flops=True) as prof:
            for _ in range(10):
                model(data)
        total = sum(e.flops for e in prof.key_averages() if e.flops)
        gflops = (total / 10) / 1e9
    except Exception:
        pass

    return {
        "latency_mean_ms": float(times.mean()),
        "latency_p50_ms":  float(np.percentile(times, 50)),
        "latency_p95_ms":  float(np.percentile(times, 95)),
        "peak_memory_mib": float(peak_mib),
        "gflops_per_forward": gflops,
        "gflops_note": "knn_graph (C++) excluded from FLOP count",
    }


# ────────────────────────────────────────────────────────── main ──
def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.run_name:
        run_name = args.run_name
    elif args.gate_mode == "soft":
        run_name = (
            f"sa_dgcnn_k{args.k_neighbors}_softgate{args.gate_floor}"
            f"_steps{args.sa_steps}_{'learn' if args.sa_learnable else 'fixed'}"
        )
    elif args.gate_mode == "hard":
        run_name = (
            f"sa_dgcnn_k{args.k_neighbors}_hardprune{args.prune_ratio}"
            f"_steps{args.sa_steps}_{'learn' if args.sa_learnable else 'fixed'}"
        )
    else:  # off
        run_name = f"sa_dgcnn_k{args.k_neighbors}_baseline"
    out_dir = args.output_dir / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"→ cache:   {args.cache_path}")
    train_raw, test_list, classes = load_cache(args.cache_path)
    print(f"   {len(train_raw)} train + {len(test_list)} test | {len(classes)} classes")

    train_raw = ensure_motif_attribute(train_raw, k=args.k_neighbors)
    test_list = ensure_motif_attribute(test_list, k=args.k_neighbors)

    train_list, val_list = stratified_train_val_split(
        train_raw, val_frac=args.val_frac, seed=args.seed,
    )
    print(f"   train/val split: {len(train_list)} / {len(val_list)} ({args.val_frac:.0%} stratified)")

    def _loader(lst, shuffle):
        return DataLoader(
            lst, batch_size=args.batch_size, shuffle=shuffle,
            num_workers=args.num_workers, pin_memory=True,
            persistent_workers=args.num_workers > 0,
        )

    train_loader = _loader(train_list, True)
    val_loader   = _loader(val_list,   False)
    test_loader  = _loader(test_list,  False)

    gate_points = tuple(int(p.strip()) for p in args.gate_points.split(",") if p.strip())
    model = SA_DGCNN(
        in_channels=args.in_channels,
        num_classes=len(classes),
        k=args.k_neighbors,
        dropout=args.dropout,
        embed_dims=args.embed_dims,
        gate_mode=args.gate_mode,
        gate_floor=args.gate_floor,
        gate_points=gate_points,
        prune_ratio=args.prune_ratio,
        prune_min_nodes=args.prune_min_nodes,
        sa_steps=args.sa_steps,
        sa_learnable=args.sa_learnable,
        sa_reweight_each_step=args.sa_reweight_each_step,
        saliency_scale=args.saliency_scale,
        seed_mix_init=args.seed_mix_init,
        small=args.small,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"   model params: {n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5,
    )
    use_amp = (not args.no_amp) and DEVICE.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val_oa, best_epoch, patience_ctr = -math.inf, 0, 0
    best_ckpt = out_dir / "best.pt"
    history: list[dict[str, Any]] = []

    for epoch in trange(1, args.epochs + 1, desc=run_name):
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, scaler, DEVICE,
            label_smoothing=args.label_smoothing,
            grad_clip=args.grad_clip, use_amp=use_amp,
            do_augment=not args.no_augment,
        )
        scheduler.step()

        row: dict[str, Any] = {
            "epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc * 100.0,
            **{f"sa_{k}": v for k, v in model.current_hparams().items()},
        }

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            val = evaluate(model, val_loader, DEVICE, use_amp=use_amp)
            row.update({f"val_{k}": v for k, v in val.items()})
            if val["oa"] > best_val_oa:
                best_val_oa, best_epoch, patience_ctr = val["oa"], epoch, 0
                torch.save({"model": model.state_dict(), "epoch": epoch, "val": val}, best_ckpt)
            else:
                patience_ctr += 1

            if epoch % 10 == 0 or epoch == args.epochs:
                print(
                    f"  ep {epoch:3d} | tr_loss {tr_loss:.4f} tr_acc {tr_acc*100:.1f}% "
                    f"| val OA {val['oa']:.2f}% mAcc {val['macc']:.2f}% F1 {val['macro_f1']:.2f}% "
                    f"| best val OA {best_val_oa:.2f} @ ep {best_epoch}"
                )
                h = model.current_hparams()
                print(f"            SA: " + " ".join(f"{k}={v:.3f}" for k, v in h.items()))

        history.append(row)
        if patience_ctr >= args.patience:
            print(f"  early stop at epoch {epoch} (no val improvement for {args.patience} evals)")
            break

    # ── final test pass using the best val checkpoint ──
    ckpt = torch.load(best_ckpt, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model"])
    test = evaluate(model, test_loader, DEVICE, use_amp=use_amp)
    print(f"\nBest val epoch: {best_epoch}  | val OA {best_val_oa:.2f}%")
    print(f"Test  OA  {test['oa']:.2f}%  mAcc {test['macc']:.2f}%  macro_f1 {test['macro_f1']:.2f}%")

    test_tta: dict[str, float] | None = None
    if args.tta_rotations > 1:
        test_tta = evaluate_tta(
            model, test_loader, DEVICE,
            use_amp=use_amp, n_rotations=args.tta_rotations,
        )
        print(
            f"Test-TTA x{args.tta_rotations}: "
            f"OA  {test_tta['oa']:.2f}%  mAcc {test_tta['macc']:.2f}%  "
            f"macro_f1 {test_tta['macro_f1']:.2f}%"
        )

    efficiency = measure_efficiency(model, test_list[0], DEVICE)
    print("Efficiency:", {k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in efficiency.items()})

    # ── save artefacts ──
    import pandas as pd
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    with open(out_dir / "summary.json", "w") as f:
        json.dump({
            "run_name":       run_name,
            "args":           {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
            "best_val":       ckpt["val"],
            "best_epoch":     best_epoch,
            "test":           test,
            "test_tta":       test_tta,
            "efficiency":     efficiency,
            "final_hparams":  model.current_hparams(),
            "n_params":       n_params,
        }, f, indent=2, default=str)
    print(f"\nSaved artefacts to {out_dir}")


if __name__ == "__main__":
    main()
