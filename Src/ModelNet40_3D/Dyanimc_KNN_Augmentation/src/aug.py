import os, torch, numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
from torch_cluster import knn_graph
from tqdm.auto import trange, tqdm
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


# The derivation derived from Pointnett++ 
# rotate_y -> Modelnet objects have randomyae: the gravity axis is fixed 
# Yae -> means the rotation around the vertical acxis is random 
# jitter , scale , dropout the other 3 augmentation that where added in the paper 

def augment_pointcloud(pos: torch.Tensor) -> torch.Tensor:
    """
    pos -> [N , 3]
    return pos [N , 3 ] -> but augmented 
    """
    pos = pos.clone() 

    # Rotate_y 
    theta = torch.rand(1).item() *2 * np.pi # generate a number between 0 and 1 nad scales by 2pi
    cos_t , sine_t = np.cos(theta) , np.sine(theta)
    R = torch.tensor([[ cos_t, 0, sin_t],
                  [     0, 1,     0],
                  [-sin_t, 0, cos_t]], dtype=torch.float32)

    pos = pos @ R.T

    # ── Gaussian jitter ───────────────────────────────
    # N(0, 0.02) clipped to ±0.05  (PointNet exact numbers)
    noise = torch.clamp(torch.randn_like(pos) * 0.02, -0.05, 0.05)
    pos = pos + noise
 
    # ── Random uniform scale [0.8, 1.25] ─────────────
    scale = 0.8 + torch.rand(1).item() * 0.45    # uniform in [0.8, 1.25]
    pos   = pos * scale
 
    # ── Random point dropout (10–30 %) ───────────────
    # Drop fraction of points, then resample back to N.
    N          = pos.shape[0]
    drop_frac  = 0.10 + torch.rand(1).item() * 0.20   # 10–30 %
    n_keep     = int(N * (1.0 - drop_frac))
    keep_idx   = torch.randperm(N)[:n_keep]
    pos_kept   = pos[keep_idx]
    # Resample back to N  (replace=True is fine; it's just noise)
    resample   = torch.randint(0, n_keep, (N - n_keep,))
    pos        = torch.cat([pos_kept, pos_kept[resample]], dim=0)
 
    return pos


    








