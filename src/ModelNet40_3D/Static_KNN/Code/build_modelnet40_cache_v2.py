"""
Rebuild the ModelNet40 cache with proper geometry processing.

Differences vs. ``build_modelnet40_cache.py`` (v1):

1. Samples points from the **mesh surface**, weighted by face area, instead of
   reading only mesh vertices.  v1 heavily over-represented mesh-dense regions
   (e.g. dense legs of a chair) and ignored large flat faces.
2. Applies **Farthest Point Sampling** (FPS) to downsample oversample → 1024,
   giving uniform coverage.  v1 used `np.random.choice` which clustered.
3. Returns **oriented normals** straight from the sampled face's normal.  v1
   computed local-PCA normals with arbitrary sign flips.
4. **Skips** degenerate meshes (< 10 faces or < 100 vertices) instead of
   sampling with replacement, which created zero-distance duplicates.
5. Pre-computes the triangle-count motif and attaches it as `data.motif`
   (named attribute, not a column) — no more runtime motif pass.
6. Uses attribute name ``normals`` (not ``norm``) to avoid collision with
   PyG's reserved ``Data.norm``.
7. Stores ``x = [pos, normals, motif]`` (7 channels) for drop-in compatibility
   with the existing trainer.

Install dep:
    pip install trimesh

Run:
    python build_modelnet40_cache_v2.py \
        --raw-root ../../../../data/ModelNet40 \
        --output-path modelnet40_final_v2.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch_geometric.data import Data
from tqdm.auto import tqdm

try:
    import trimesh
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "build_modelnet40_cache_v2 needs `trimesh`. Install with: pip install trimesh"
    ) from e


ROOT = Path(__file__).resolve().parent
DEFAULT_RAW_ROOT = ROOT.parents[3] / "data" / "ModelNet40"
DEFAULT_OUTPUT = ROOT / "modelnet40_final_v2.pt"


# ─────────────────────────────────────────────────────────── args ──
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build ModelNet40 cache v2 (FPS + surface sampling + oriented normals).")
    p.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    p.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--num-points", type=int, default=1024,
                   help="Target number of points per cloud after FPS.")
    p.add_argument("--oversample", type=int, default=4096,
                   help="How many points to surface-sample before FPS.")
    p.add_argument("--graph-k", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ─────────────────────────────────────────────────────── geometry ──
def load_mesh(path: Path) -> trimesh.Trimesh | None:
    """Robust OFF loader.  Some ModelNet40 files have malformed headers."""
    try:
        mesh = trimesh.load(path, force="mesh", process=False)
    except Exception:
        # Fallback: fix the common "OFFn_verts ..." malformed header and retry.
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
            if text.startswith("OFF") and not text.startswith("OFF\n"):
                text = "OFF\n" + text[3:]
            mesh = trimesh.load(
                trimesh.util.wrap_as_stream(text),
                file_type="off",
                force="mesh",
                process=False,
            )
        except Exception:
            return None
    if not isinstance(mesh, trimesh.Trimesh):
        return None
    if len(mesh.vertices) < 100 or len(mesh.faces) < 10:
        return None
    return mesh


def surface_sample(mesh: trimesh.Trimesh, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Area-weighted surface sampling → (points [n,3], normals [n,3]).

    ``trimesh.sample.sample_surface`` returns the face index for each sample
    point; we index into the mesh's oriented face normals to get a per-point
    normal that is already outward-consistent across the mesh.
    """
    # trimesh's RNG is global; drive it from our local rng for determinism.
    seed = int(rng.integers(0, 2**31 - 1))
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        pts, face_idx = trimesh.sample.sample_surface(mesh, n)
    finally:
        np.random.set_state(state)
    normals = mesh.face_normals[face_idx]
    # trimesh's face_normals are unit-length; guard against degenerate faces.
    lens = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / (lens + 1e-10)
    return pts.astype(np.float32), normals.astype(np.float32)


def farthest_point_sample(
    pts: np.ndarray, k: int, rng: np.random.Generator
) -> np.ndarray:
    """Return indices of k FPS-sampled points from ``pts`` (CPU, numpy).

    O(N * k) — for N=4096, k=1024 this is ~30ms/cloud.  Good enough offline.
    """
    n = pts.shape[0]
    if n <= k:
        return np.arange(n)

    selected = np.empty(k, dtype=np.int64)
    start = int(rng.integers(0, n))
    selected[0] = start
    dists = np.linalg.norm(pts - pts[start], axis=1)
    for i in range(1, k):
        idx = int(np.argmax(dists))
        selected[i] = idx
        new_dists = np.linalg.norm(pts - pts[idx], axis=1)
        dists = np.minimum(dists, new_dists)
    return selected


def normalize_to_unit_sphere(pts: np.ndarray) -> np.ndarray:
    pts = pts - pts.mean(axis=0)
    radius = np.linalg.norm(pts, axis=1).max() + 1e-10
    return pts / radius


def build_edge_index(pts: np.ndarray, k: int) -> torch.Tensor:
    tree = cKDTree(pts)
    _, idx = tree.query(pts, k=k + 1)
    idx = idx[:, 1:]  # drop self
    src = np.repeat(np.arange(pts.shape[0]), k)
    dst = idx.reshape(-1)
    return torch.tensor(np.stack([src, dst]), dtype=torch.long)


def triangle_counts(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """A² self-count: number of length-2 paths back to each node = 2 × triangles."""
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    adj[edge_index[0], edge_index[1]] = 1.0
    adj[edge_index[1], edge_index[0]] = 1.0  # symmetrise (k-NN is directed)
    counts = (adj * (adj @ adj)).sum(dim=1) / 2.0
    m = counts.max()
    return counts / m if m > 0 else counts


# ───────────────────────────────────────────────────── per-split ──
def build_one(
    path: Path,
    label: int,
    n_points: int,
    oversample: int,
    graph_k: int,
    rng: np.random.Generator,
) -> Data | None:
    mesh = load_mesh(path)
    if mesh is None:
        return None

    # 1. Dense area-weighted surface sample, with oriented normals from faces.
    pts_dense, normals_dense = surface_sample(mesh, max(oversample, n_points), rng)

    # 2. FPS to a uniform `n_points` subset.
    keep = farthest_point_sample(pts_dense, n_points, rng)
    pts = pts_dense[keep]
    normals = normals_dense[keep]

    # 3. Unit-sphere normalise (pos only — normals stay unit-length).
    pts = normalize_to_unit_sphere(pts)

    # 4. Static k-NN edge index.
    ei = build_edge_index(pts, graph_k)

    # 5. Motif (normalised triangle counts) — attach as named attribute.
    motif = triangle_counts(ei, pts.shape[0]).float()

    pos_t = torch.from_numpy(pts).float()
    nrm_t = torch.from_numpy(normals).float()
    x = torch.cat([pos_t, nrm_t, motif.unsqueeze(1)], dim=1)  # [N, 7]

    data = Data(
        pos=pos_t,
        normals=nrm_t,     # not ``norm`` — avoids PyG reserved name
        motif=motif,
        x=x,
        edge_index=ei,
        y=torch.tensor([label], dtype=torch.long),
    )
    return data


def load_split(
    raw_root: Path,
    classes: list[str],
    class_to_idx: dict[str, int],
    split: str,
    *,
    n_points: int,
    oversample: int,
    graph_k: int,
    rng: np.random.Generator,
) -> tuple[list[Data], int]:
    all_files: list[tuple[Path, int]] = []
    for cls in classes:
        folder = raw_root / cls / split
        if not folder.exists():
            continue
        for off_file in sorted(folder.glob("*.off")):
            all_files.append((off_file, class_to_idx[cls]))

    out: list[Data] = []
    skipped = 0
    for path, label in tqdm(all_files, desc=f"  {split}"):
        data = build_one(
            path, label,
            n_points=n_points, oversample=oversample, graph_k=graph_k, rng=rng,
        )
        if data is None:
            skipped += 1
            continue
        out.append(data)
    return out, skipped


# ──────────────────────────────────────────────────────────── main ──
def main() -> None:
    args = parse_args()
    if not args.raw_root.exists():
        raise FileNotFoundError(f"Raw ModelNet40 path not found: {args.raw_root}")

    rng = np.random.default_rng(args.seed)
    classes = sorted([p.name for p in args.raw_root.iterdir() if p.is_dir()])
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

    print(f"Raw root:    {args.raw_root}")
    print(f"Classes:     {len(classes)}")
    print(f"Target pts:  {args.num_points}  (from {args.oversample} surface samples via FPS)")
    print(f"Graph k:     {args.graph_k}")

    # Separate RNGs per split so train/test don't share state.
    rng_train = np.random.default_rng(args.seed)
    rng_test  = np.random.default_rng(args.seed + 1)

    print("\nBuilding train split…")
    train_list, skipped_train = load_split(
        args.raw_root, classes, class_to_idx, "train",
        n_points=args.num_points, oversample=args.oversample,
        graph_k=args.graph_k, rng=rng_train,
    )
    print(f"  kept {len(train_list)} / skipped {skipped_train}")

    print("\nBuilding test split…")
    test_list, skipped_test = load_split(
        args.raw_root, classes, class_to_idx, "test",
        n_points=args.num_points, oversample=args.oversample,
        graph_k=args.graph_k, rng=rng_test,
    )
    print(f"  kept {len(test_list)} / skipped {skipped_test}")

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"train": train_list, "test": test_list, "classes": classes},
        args.output_path,
    )

    size_mb = args.output_path.stat().st_size / 1e6
    print(f"\nSaved cache to: {args.output_path}")
    print(f"Train graphs:   {len(train_list)}")
    print(f"Test graphs:    {len(test_list)}")
    print(f"Cache size:     {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
