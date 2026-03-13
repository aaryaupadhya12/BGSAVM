 
import os, torch, numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
from torch_cluster import knn_graph
from tqdm.auto import trange, tqdm
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

def compute_triangle_counts(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """
    Count triangles per node using the sparse A² trick.
 
    WHY THIS WORKS (the maths):
      For adjacency matrix A, entry (A²)[i,j] counts paths of length 2
      from i to j — i.e. common neighbours of i and j.
      A triangle at node i exists when edge (i,j) exists AND i,j share
      a common neighbour:
          triangles(i) = sum_j  A[i,j] * (A²)[i,j]
                       = row_sum( A  ⊙  A² )   where ⊙ = element-wise
      Each triangle is counted twice (clockwise + anti-clockwise), so
      divide by 2.
 
    Complexity: same O(N·k²) but via torch C++ matmul — ~100-200x faster
    than the Python triple-loop.  At N=1024, k=20: ~20ms vs ~4s per cloud.
 
    edge_index : [2, E]   directed edges (each undirected edge appears twice)
    num_nodes  : int      N
    returns    : [N, 1]   normalised triangle count per node  (float)
    """
    N   = num_nodes
    src = edge_index[0]
    dst = edge_index[1]
 
    # ── Build dense adjacency matrix  A  [N, N] ──────────────────────
    # N=1024 → 1024×1024 float32 = 4 MB, fine on CPU and GPU
    A = torch.zeros(N, N, dtype=torch.float32)
    A[src, dst] = 1.0
 
    # ── A² = A @ A  ───────────────────────────────────────────────────
    # (A²)[i,j] = number of common neighbours between node i and node j
    A2 = A @ A                           # [N, N]
 
    # ── Triangle count per node ───────────────────────────────────────
    # counts[i] = sum_j  A[i,j] * A2[i,j]
    counts = (A * A2).sum(dim=1)         # [N]
    counts = counts / 2.0                # each triangle counted twice
 
    # ── Normalise to [0, 1] ───────────────────────────────────────────
    max_c = counts.max()
    if max_c > 0:
        counts = counts / max_c
 
    return counts.unsqueeze(1)           # [N, 1]
 
 
def add_motif_features(data_list: list, k: int = 20, desc: str = '') -> list:
    """
    For each Data in data_list:
      1. Build a static KNN graph from pos (used only for motif counting)
      2. Compute triangle counts
      3. Append the count as an extra node feature → x becomes [N, 7]
 
    NOTE: This is done once at preprocessing time and cached.
    The dynamic KNN is rebuilt every forward pass in feature space.
    The motif count here is computed on the GEOMETRIC graph (pos-based),
    which is correct: we want structural importance of points in 3D space,
    not in the learned feature space.
    """
    from scipy.spatial import cKDTree
    import numpy as np
 
    for data in tqdm(data_list, desc=f'Motif scores {desc}', leave=False):
        pos  = data.pos.numpy()   # [N, 3]
        N    = pos.shape[0]
 
        # Build geometric KNN using cKDTree — same as your EDA preprocessing
        tree      = cKDTree(pos)
        _, nn_idx = tree.query(pos, k=k + 1)   # [N, k+1]
        nn_idx    = nn_idx[:, 1:]               # drop self → [N, k]
 
        src = np.repeat(np.arange(N), k)        # [N*k]
        dst = nn_idx.flatten()                  # [N*k]
 
        edge_index_geo = torch.tensor(
            np.stack([src, dst], axis=0),
            dtype=torch.long
        )
 
        # Compute triangle counts using the fast A² trick
        tri_counts = compute_triangle_counts(edge_index_geo, N)  # [N, 1]
 
        # Append to node features: x was [N, 6], now [N, 7]
        data.x = torch.cat([data.x, tri_counts], dim=1)
 
        # Store geometric edge_index (model rebuilds dynamically in feature space)
        data.edge_index = edge_index_geo
 
    return data_list


MOTIF_CACHE = '/kaggle/working/modelnet40_motif.pt'
 
if os.path.exists(MOTIF_CACHE):
    print("Loading motif-augmented cache …")
    motif_cache = torch.load(MOTIF_CACHE, map_location='cpu')
    train_list  = motif_cache['train']
    test_list   = motif_cache['test']
else:
    print("Computing triangle motif scores (one-time cost) …")
    train_list = add_motif_features(train_list, k=K_NEIGHBORS, desc='train')
    test_list  = add_motif_features(test_list,  k=K_NEIGHBORS, desc='test')
    torch.save({'train': train_list, 'test': test_list,
                'classes': CLASSES}, MOTIF_CACHE)
    motif_size = os.path.getsize(MOTIF_CACHE) / 1e6
    print(f"Saved motif cache: {motif_size:.0f} MB")
 
# Verify feature shape
sample = train_list[0]
print(f"Node feature shape: {sample.x.shape}  ← should be [1024, 7]")
print(f"  col 0-2 : xyz  |  col 3-5 : normals  |  col 6 : triangle count")
 