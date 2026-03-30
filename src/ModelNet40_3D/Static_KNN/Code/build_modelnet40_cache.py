from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch_geometric.data import Data
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parent
DEFAULT_RAW_ROOT = ROOT.parents[3] / "data" / "ModelNet40"
DEFAULT_OUTPUT = ROOT / "modelnet40_final.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build local ModelNet40 cache for PointGCN experiments.")
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-points", type=int, default=1024)
    parser.add_argument("--graph-k", type=int, default=20)
    parser.add_argument("--normal-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_off(filepath: Path) -> np.ndarray | None:
    with filepath.open("r", encoding="utf-8", errors="ignore") as handle:
        header = handle.readline().strip()
        if "OFF" not in header:
            return None

        counts = header[3:].strip().split() if len(header) > 3 else handle.readline().strip().split()
        if not counts:
            return None

        n_verts = int(counts[0])
        verts = []
        for _ in range(n_verts):
            line = handle.readline().strip().split()
            if len(line) >= 3:
                verts.append([float(x) for x in line[:3]])

    return np.array(verts, dtype=np.float32) if verts else None


def process_pointcloud(
    verts: np.ndarray,
    n_points: int,
    normal_k: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if len(verts) >= n_points:
        idx = rng.choice(len(verts), n_points, replace=False)
    else:
        idx = rng.choice(len(verts), n_points, replace=True)

    pts = verts[idx]
    pts = pts - pts.mean(axis=0)
    pts = pts / (np.linalg.norm(pts, axis=1).max() + 1e-10)

    tree = cKDTree(pts)
    _, nn_idx = tree.query(pts, k=normal_k)

    normals = np.zeros_like(pts)
    for i in range(len(pts)):
        neighbors = pts[nn_idx[i]]
        centered = neighbors - neighbors.mean(axis=0)
        cov = centered.T @ centered
        _, eigvecs = np.linalg.eigh(cov)
        normals[i] = eigvecs[:, 0]

    normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-10)
    return pts.astype(np.float32), normals.astype(np.float32)


def build_edge_index(points: np.ndarray, k_neighbors: int) -> torch.Tensor:
    tree = cKDTree(points)
    _, idx = tree.query(points, k=k_neighbors + 1)
    idx = idx[:, 1:]

    src = np.repeat(np.arange(points.shape[0]), k_neighbors)
    dst = idx.reshape(-1)
    return torch.tensor(np.stack([src, dst]), dtype=torch.long)


def load_split(
    raw_root: Path,
    classes: list[str],
    class_to_idx: dict[str, int],
    split: str,
    n_points: int,
    graph_k: int,
    normal_k: int,
    rng: np.random.Generator,
) -> list[Data]:
    data_list: list[Data] = []
    all_files: list[tuple[Path, int]] = []

    for cls in classes:
        folder = raw_root / cls / split
        if not folder.exists():
            continue
        for off_file in folder.glob("*.off"):
            all_files.append((off_file, class_to_idx[cls]))

    for fpath, label in tqdm(all_files, desc=f"Loading {split}"):
        verts = read_off(fpath)
        if verts is None or len(verts) < 10:
            continue

        pts, nrm = process_pointcloud(verts, n_points=n_points, normal_k=normal_k, rng=rng)
        edge_index = build_edge_index(pts, k_neighbors=graph_k)

        pos = torch.tensor(pts)
        norm = torch.tensor(nrm)
        data = Data(
            pos=pos,
            norm=norm,
            x=torch.cat([pos, norm], dim=1),
            edge_index=edge_index,
            y=torch.tensor([label], dtype=torch.long),
        )
        data_list.append(data)

    return data_list


def main() -> None:
    args = parse_args()
    if not args.raw_root.exists():
        raise FileNotFoundError(f"Raw ModelNet40 path not found: {args.raw_root}")

    rng = np.random.default_rng(args.seed)
    classes = sorted([p.name for p in args.raw_root.iterdir() if p.is_dir()])
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

    print(f"Raw root: {args.raw_root}")
    print(f"Classes: {len(classes)}")
    print("Building train split...")
    train_list = load_split(
        raw_root=args.raw_root,
        classes=classes,
        class_to_idx=class_to_idx,
        split="train",
        n_points=args.num_points,
        graph_k=args.graph_k,
        normal_k=args.normal_k,
        rng=rng,
    )
    print("Building test split...")
    test_list = load_split(
        raw_root=args.raw_root,
        classes=classes,
        class_to_idx=class_to_idx,
        split="test",
        n_points=args.num_points,
        graph_k=args.graph_k,
        normal_k=args.normal_k,
        rng=rng,
    )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "train": train_list,
            "test": test_list,
            "classes": classes,
        },
        args.output_path,
    )

    size_mb = args.output_path.stat().st_size / 1e6
    print(f"\nSaved cache to: {args.output_path}")
    print(f"Train graphs: {len(train_list)}")
    print(f"Test graphs:  {len(test_list)}")
    print(f"Cache size:   {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
