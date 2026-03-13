import os, torch, numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
from torch_cluster import knn_graph
from tqdm.auto import trange, tqdm
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


def build_dynamic_knn(x: torch.Tensor, k: int, batch: torch.Tensor) -> torch.Tensor:
    """
    Pure-torch GPU KNN — no external libraries, stays on P100 the whole time.
 
    Algorithm — pairwise squared distance via the identity:
        ||a - b||²  =  ||a||²  +  ||b||²  -  2 * a·b
    Written as a matrix op:
        D[i,j]  =  diag(X @ X.T)[i]  +  diag(X @ X.T)[j]  -  2 * (X @ X.T)[i,j]
    Then torch.topk finds the k smallest distances per row — all on GPU.
 
    x     : [N_total, F]   node features  (on GPU)
    k     : number of neighbours
    batch : [N_total]      batch assignment vector  (on GPU)
    returns edge_index [2, E]  on same device as x
 
    Processes each sample in the batch separately so no cross-sample
    edges are created. N=1024, F=128 → distance matrix is 1024×1024
    float32 = 4MB per sample, fine on P100 (16GB VRAM).
    """
    device     = x.device
    all_src, all_dst = [], []
 
    # unique_consecutive is faster than unique when batch is sorted
    # (PyG DataLoader always produces sorted batch vectors)
    for b_idx in batch.unique():
        mask = (batch == b_idx)                  # [N_total] bool
        pts  = x[mask]                           # [N, F]
        N    = pts.shape[0]
 
        # ── pairwise squared distances on GPU ─────────────────────
        # ||a||² broadcast trick — avoids materialising full [N,N,F]
        sq   = (pts * pts).sum(dim=1, keepdim=True)   # [N, 1]
        dist = sq + sq.T - 2.0 * (pts @ pts.T)        # [N, N]
        # Numerical noise can give tiny negatives → clamp
        dist = dist.clamp(min=0.0)
        # Zero out diagonal so self is never selected as a neighbour
        dist.fill_diagonal_(float('inf'))
 
        # ── k nearest neighbours ──────────────────────────────────
        # topk with largest=False → k smallest distances
        _, nn_idx = dist.topk(k, dim=1, largest=False)  # [N, k]
 
        # ── build local edge_index then map to global indices ─────
        global_idx = mask.nonzero(as_tuple=True)[0]     # [N] global positions
        src_local  = torch.arange(N, device=device).unsqueeze(1).expand(N, k)
        src_global = global_idx[src_local.reshape(-1)]  # [N*k]
        dst_global = global_idx[nn_idx.reshape(-1)]     # [N*k]
 
        all_src.append(src_global)
        all_dst.append(dst_global)
 
    edge_index = torch.stack([
        torch.cat(all_src),
        torch.cat(all_dst)
    ], dim=0)                                            # [2, E]
    return edge_index