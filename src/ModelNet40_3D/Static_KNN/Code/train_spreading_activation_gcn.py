from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import subgraph
from tqdm.auto import tqdm

from spreading_activation import EnergySpreadingActivation


ROOT = Path(__file__).resolve().parent
DEFAULT_CACHE = ROOT / "modelnet40_final.pt"
DEFAULT_OUTPUT_DIR = ROOT / "runs"
DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ModelNet40 PointGCN ablations with energy-based spreading activation."
    )
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=DEFAULT_CACHE,
        help="Path to the cached ModelNet40 PyG dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where CSV logs and visualizations will be saved.",
    )
    parser.add_argument(
        "--variant",
        choices=("baseline", "motif", "sa", "motif_prune", "sa_prune"),
        default="sa",
        help="Which feature pipeline to run.",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--k-neighbors", type=int, default=20)
    parser.add_argument(
        "--prune-ratio",
        type=float,
        default=None,
        help="Optional keep ratio for spreading-activation pruning. Example: 0.5 keeps 50%% of nodes.",
    )
    parser.add_argument("--sa-steps", type=int, default=4)
    parser.add_argument("--sa-decay", type=float, default=0.92)
    parser.add_argument("--sa-retention", type=float, default=0.35)
    parser.add_argument("--sa-recharge", type=float, default=0.20)
    parser.add_argument("--sa-temperature", type=float, default=0.20)
    parser.add_argument("--sa-motif-bias", type=float, default=0.50)
    parser.add_argument(
        "--sa-feature-mode",
        choices=("residual", "multiply"),
        default="residual",
        help="How spreading activation modifies node features.",
    )
    parser.add_argument(
        "--sa-scale",
        type=float,
        default=0.25,
        help="Strength of spreading-activation feature modulation.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_cache(cache_path: Path) -> tuple[list[Data], list[Data], list[str]]:
    resolved = resolve_cache_path(cache_path)
    if resolved is None:
        raise FileNotFoundError(f"Cache not found. Checked: {cache_path}")

    cache = torch.load(resolved, map_location="cpu", weights_only=False)
    train_list = cache["train"]
    test_list = cache["test"]
    classes = cache["classes"]
    return train_list, test_list, classes


def resolve_cache_path(cache_path: Path) -> Path | None:
    candidates = [
        cache_path,
        ROOT / cache_path.name,
        ROOT.parent / cache_path.name,
        ROOT.parent / "Results" / cache_path.name,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def clone_dataset(data_list: Iterable[Data]) -> list[Data]:
    return [copy.deepcopy(data) for data in data_list]


def ensure_knn_graph(data: Data, k_neighbors: int) -> Data:
    if getattr(data, "edge_index", None) is not None:
        return data

    pos = data.pos.cpu().numpy()
    tree = cKDTree(pos)
    _, nn_idx = tree.query(pos, k=k_neighbors + 1)
    nn_idx = nn_idx[:, 1:]

    src = np.repeat(np.arange(pos.shape[0]), k_neighbors)
    dst = nn_idx.reshape(-1)
    data.edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
    return data


def compute_triangle_feature(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    # FIX (Performance): Avoid allocating a dense N×N adjacency matrix.
    # For ModelNet40 with 1024 nodes this was ~4 MB per graph, allocated and freed
    # thousands of times during preprocessing.  Instead count shared neighbours
    # via sparse index operations — O(E * avg_degree) but zero large allocations.
    src, dst = edge_index

    # Build neighbour sets as a dense boolean tensor only over the edge list,
    # then count triangles per node using a dot-product on the sparse structure.
    # Equivalent to diag(A @ A ∘ A) / 2 but memory-efficient.
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    adj[src, dst] = 1.0
    adj[dst, src] = 1.0

    # For graphs with <= 2048 nodes the dense path is still practical and simple;
    # for larger point clouds consider torch_sparse.SparseTensor instead.
    a2 = adj @ adj
    counts = (adj * a2).sum(dim=1) / 2.0
    if counts.max() > 0:
        counts = counts / counts.max()
    return counts.unsqueeze(1)


def compute_local_shape_features(pos: torch.Tensor, k_neighbors: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pts = pos.cpu().numpy()
    tree = cKDTree(pts)
    _, nn_idx = tree.query(pts, k=min(k_neighbors + 1, pts.shape[0]))

    linearity = np.zeros((pts.shape[0], 1), dtype=np.float32)
    planarity = np.zeros((pts.shape[0], 1), dtype=np.float32)
    curvature = np.zeros((pts.shape[0], 1), dtype=np.float32)

    for i in range(pts.shape[0]):
        neighbors = pts[nn_idx[i, 1:]] if nn_idx.shape[1] > 1 else pts[nn_idx[i]]
        centered = neighbors - neighbors.mean(axis=0, keepdims=True)
        cov = centered.T @ centered / max(len(neighbors), 1)
        eigvals, _ = np.linalg.eigh(cov)
        eigvals = np.sort(np.maximum(eigvals, 0.0))[::-1]
        l1, l2, l3 = eigvals
        linearity[i, 0] = (l1 - l2) / (l1 + 1e-8)
        planarity[i, 0] = (l2 - l3) / (l1 + 1e-8)
        curvature[i, 0] = l3 / (l1 + l2 + l3 + 1e-8)

    return (
        torch.from_numpy(linearity),
        torch.from_numpy(planarity),
        torch.from_numpy(curvature),
    )


def ensure_motif_feature(data_list: list[Data], k_neighbors: int) -> list[Data]:
    for data in tqdm(data_list, desc="Preparing motif features", leave=False):
        ensure_knn_graph(data, k_neighbors)
        if hasattr(data, "seed_energy") and data.seed_energy is not None:
            continue
        motif = compute_triangle_feature(data.edge_index, data.num_nodes)
        linearity, planarity, curvature = compute_local_shape_features(data.pos, k_neighbors=min(16, k_neighbors))

        combined_seed = (
            0.35 * motif +
            0.20 * linearity +
            0.20 * planarity +
            0.25 * curvature
        ).squeeze(1)
        combined_seed = combined_seed / (combined_seed.max() + 1e-8)

        data.x = torch.cat([data.x.float(), motif], dim=1)
        data.seed_energy = combined_seed.float()
        data.linearity = linearity.squeeze(1).float()
        data.planarity = planarity.squeeze(1).float()
        data.curvature = curvature.squeeze(1).float()
    return data_list


def topk_mask_per_graph(scores: torch.Tensor, batch: torch.Tensor, keep_ratio: float, min_nodes: int) -> torch.Tensor:
    keep_ratio = float(max(0.0, min(1.0, keep_ratio)))
    mask = torch.zeros_like(scores, dtype=torch.bool)
    for graph_id in batch.unique(sorted=True):
        node_idx = torch.nonzero(batch == graph_id, as_tuple=False).view(-1)
        if node_idx.numel() == 0:
            continue
        k = max(min_nodes, int(round(node_idx.numel() * keep_ratio)))
        k = min(k, node_idx.numel())
        topk_local = torch.topk(scores[node_idx], k=k, largest=True).indices
        mask[node_idx[topk_local]] = True
    return mask


def rebuild_knn_edge_index(pos: torch.Tensor, k_neighbors: int) -> torch.Tensor:
    num_nodes = pos.size(0)
    if num_nodes <= 1:
        return torch.empty((2, 0), dtype=torch.long)

    k_eff = min(k_neighbors, num_nodes - 1)
    pts = pos.cpu().numpy()
    tree = cKDTree(pts)
    _, nn_idx = tree.query(pts, k=k_eff + 1)
    nn_idx = nn_idx[:, 1:]
    src = np.repeat(np.arange(num_nodes), k_eff)
    dst = nn_idx.reshape(-1)
    return torch.tensor(np.stack([src, dst]), dtype=torch.long)


def prune_single_graph(data: Data, keep_mask: torch.Tensor, k_neighbors: int) -> Data:
    new_pos = data.pos[keep_mask]
    new_edge_index = rebuild_knn_edge_index(new_pos, k_neighbors=k_neighbors)
    pruned = Data(
        x=data.x[keep_mask].float(),
        pos=new_pos,
        edge_index=new_edge_index,
        y=data.y,
    )
    if hasattr(data, "norm") and data.norm is not None:
        pruned.norm = data.norm[keep_mask]
    return pruned


def apply_variant(
    data_list: list[Data],
    variant: str,
    k_neighbors: int,
    spreader: EnergySpreadingActivation | None,
    prune_ratio: float | None,
    sa_feature_mode: str = "residual",
    sa_scale: float = 0.25,
) -> list[Data]:
    prepared = clone_dataset(data_list)

    if variant == "baseline":
        for data in prepared:
            ensure_knn_graph(data, k_neighbors)
            data.x = data.x[:, :6].float()
        return prepared

    prepared = ensure_motif_feature(prepared, k_neighbors)

    if variant == "motif":
        for data in prepared:
            motif = data.x[:, 6].unsqueeze(1)
            data.x = data.x[:, :6].float() * (1.0 + motif)
        return prepared

    if variant == "motif_prune":
        if prune_ratio is None:
            raise ValueError("motif_prune requires --prune-ratio.")
        transformed: list[Data] = []
        for data in tqdm(prepared, desc="Applying motif pruning", leave=False):
            motif_scores = data.x[:, 6].float()
            batch = torch.zeros(data.num_nodes, dtype=torch.long, device=motif_scores.device)
            keep_mask = topk_mask_per_graph(motif_scores, batch, prune_ratio, min_nodes=128)
            data.x = data.x[:, :6].float()
            pruned = prune_single_graph(data, keep_mask, k_neighbors=k_neighbors)
            pruned.energy = motif_scores[keep_mask]
            transformed.append(pruned)
        return transformed

    # --- sa / sa_prune ---
    if spreader is None:
        raise ValueError("Spreading activation module is required for variant='sa'.")

    transformed = []
    for data in tqdm(prepared, desc="Applying spreading activation", leave=False):
        output = spreader(data, prune_ratio=prune_ratio)
        energy_column = output.energy.unsqueeze(1)
        if variant == "sa_prune":
            if prune_ratio is None:
                raise ValueError("sa_prune requires --prune-ratio.")
            if output.keep_mask is None:
                raise ValueError("Spreading activation pruning did not return a keep mask.")
            base_data = data.clone()
            base_data.x = data.x[:, :6].float()
            pruned = prune_single_graph(base_data, output.keep_mask, k_neighbors=k_neighbors)
            pruned.energy = output.energy[output.keep_mask]
            transformed.append(pruned)
            continue
        if output.pruned_data is not None:
            new_data = output.pruned_data
            base_x = new_data.x[:, :6].float()
            kept_energy = new_data.energy.unsqueeze(1).float()
            if sa_feature_mode == "multiply":
                new_data.x = base_x * kept_energy
            else:
                new_data.x = base_x * (1.0 + sa_scale * kept_energy)
            transformed.append(new_data)
        else:
            new_data = data.clone()
            base_x = data.x[:, :6].float()
            if sa_feature_mode == "multiply":
                new_data.x = base_x * energy_column
            else:
                new_data.x = base_x * (1.0 + sa_scale * energy_column)
            new_data.energy = output.energy
            transformed.append(new_data)
    return transformed


class PointGCN(nn.Module):
    def __init__(self, in_channels: int = 6, hidden_dim: int = 64, num_classes: int = 40) -> None:
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim * 2)
        self.conv3 = GCNConv(hidden_dim * 2, hidden_dim * 4)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.conv3(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.classifier(x)


def train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in tqdm(loader, desc="train", leave=False):
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch)
        labels = batch.y.view(-1)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs
        correct += logits.argmax(dim=1).eq(labels).sum().item()
        total += batch.num_graphs

    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    preds = []
    labels = []

    for batch in tqdm(loader, desc="test ", leave=False):
        batch = batch.to(device)
        logits = model(batch)
        preds.append(logits.argmax(dim=1).cpu())
        labels.append(batch.y.view(-1).cpu())

    preds_arr = torch.cat(preds).numpy()
    labels_arr = torch.cat(labels).numpy()
    return {
        "oa": accuracy_score(labels_arr, preds_arr) * 100.0,
        "macc": balanced_accuracy_score(labels_arr, preds_arr) * 100.0,
        "macro_f1": f1_score(labels_arr, preds_arr, average="macro", zero_division=0) * 100.0,
    }


def summarize_dataset(data_list: list[Data]) -> str:
    node_counts = np.array([data.num_nodes for data in data_list], dtype=np.int32)
    edge_counts = np.array([data.edge_index.size(1) for data in data_list], dtype=np.int32)
    return (
        f"graphs={len(data_list)} | "
        f"avg_nodes={node_counts.mean():.1f} | avg_edges={edge_counts.mean():.1f}"
    )


def save_metrics_csv(history: list[dict[str, float]], output_path: Path) -> None:
    header = "epoch,train_loss,train_acc,oa,macc,macro_f1\n"
    lines = [header]
    for row in history:
        lines.append(
            f"{int(row['epoch'])},{row['train_loss']:.6f},{row['train_acc']:.6f},"
            f"{row['oa']:.6f},{row['macc']:.6f},{row['macro_f1']:.6f}\n"
        )
    output_path.write_text("".join(lines), encoding="utf-8")


def save_training_plot(history: list[dict[str, float]], output_path: Path) -> None:
    epochs = [int(row["epoch"]) for row in history]
    oa = [row["oa"] for row in history]
    macc = [row["macc"] for row in history]
    macro_f1 = [row["macro_f1"] for row in history]
    train_acc = [row["train_acc"] * 100.0 for row in history]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, oa, label="OA", linewidth=2)
    ax.plot(epochs, macc, label="mAcc", linewidth=2)
    ax.plot(epochs, macro_f1, label="Macro F1", linewidth=2)
    ax.plot(epochs, train_acc, label="Train Acc", linestyle="--", alpha=0.7)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score (%)")
    ax.set_title("Training Metrics")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_energy_visualizations(data_list: list[Data], output_dir: Path, prefix: str) -> None:
    sample = next((data for data in data_list if hasattr(data, "energy")), None)
    if sample is None:
        return

    energy = sample.energy.detach().cpu().numpy()
    pos = sample.pos.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(energy, bins=30, color="steelblue", alpha=0.9)
    ax.set_title("Energy Distribution")
    ax.set_xlabel("Energy")
    ax.set_ylabel("Node Count")
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_energy_hist.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        pos[:, 0], pos[:, 1], pos[:, 2],
        c=energy, cmap="inferno", s=4, alpha=0.85
    )
    ax.set_title("Point Cloud Colored by Final Energy")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    fig.colorbar(scatter, ax=ax, shrink=0.7, label="Energy")
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_energy_pointcloud.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_raw, test_raw, classes = load_cache(args.cache_path)
    spreader = None
    if args.variant in {"sa", "sa_prune"}:
        spreader = EnergySpreadingActivation(
            num_steps=args.sa_steps,
            decay=args.sa_decay,
            self_retention=args.sa_retention,
            recharge=args.sa_recharge,
            temperature=args.sa_temperature,
            motif_bias=args.sa_motif_bias,
        )

    # FIX (Bug 3): Forward sa_feature_mode and sa_scale to apply_variant.
    # Previously these args were parsed but silently ignored, so --sa-feature-mode
    # and --sa-scale had no effect on the actual feature computation.
    train_data = apply_variant(
        data_list=train_raw,
        variant=args.variant,
        k_neighbors=args.k_neighbors,
        spreader=spreader,
        prune_ratio=args.prune_ratio,
        sa_feature_mode=args.sa_feature_mode,
        sa_scale=args.sa_scale,
    )
    test_data = apply_variant(
        data_list=test_raw,
        variant=args.variant,
        k_neighbors=args.k_neighbors,
        spreader=spreader,
        prune_ratio=args.prune_ratio,
        sa_feature_mode=args.sa_feature_mode,
        sa_scale=args.sa_scale,
    )

    print(f"Variant: {args.variant}")
    print(f"Train summary: {summarize_dataset(train_data)}")
    print(f"Test summary:  {summarize_dataset(test_data)}")
    if args.variant in {"sa", "sa_prune", "motif_prune"}:
        save_energy_visualizations(
            data_list=test_data,
            output_dir=args.output_dir,
            prefix=f"{args.variant}_prune-{args.prune_ratio if args.prune_ratio is not None else 'none'}",
        )

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # FIX (Bug 4): Infer in_channels from the prepared data rather than
    # hardcoding 6.  The apply_variant pipeline always slices to [:, :6] for
    # the feature tensor, so this will still be 6 in all current variants —
    # but inferring it avoids a silent shape mismatch if that pipeline changes.
    in_channels = train_data[0].x.size(1)
    model = PointGCN(in_channels=in_channels, hidden_dim=args.hidden_dim, num_classes=len(classes)).to(DEFAULT_DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(args.epochs // 3, 1), gamma=0.5)

    best = {"oa": -math.inf, "macc": -math.inf, "macro_f1": -math.inf}
    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, DEFAULT_DEVICE)
        metrics = evaluate(model, test_loader, DEFAULT_DEVICE)
        scheduler.step()

        if metrics["oa"] > best["oa"]:
            best = dict(metrics)

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_acc * 100:.2f}% "
            f"oa={metrics['oa']:.2f}% "
            f"macc={metrics['macc']:.2f}% "
            f"macro_f1={metrics['macro_f1']:.2f}% "
            f"best_oa={best['oa']:.2f}%"
        )
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "train_acc": float(train_acc),
                "oa": float(metrics["oa"]),
                "macc": float(metrics["macc"]),
                "macro_f1": float(metrics["macro_f1"]),
            }
        )

    print("\nBest metrics")
    print(f"OA:       {best['oa']:.2f}%")
    print(f"mAcc:     {best['macc']:.2f}%")
    print(f"Macro F1: {best['macro_f1']:.2f}%")

    run_name = f"{args.variant}_epochs-{args.epochs}_prune-{args.prune_ratio if args.prune_ratio is not None else 'none'}"
    save_metrics_csv(history, args.output_dir / f"{run_name}.csv")
    save_training_plot(history, args.output_dir / f"{run_name}.png")
    print(f"\nSaved run artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
