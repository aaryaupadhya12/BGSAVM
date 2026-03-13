import os, torch, numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
from torch_cluster import knn_graph
from tqdm.auto import trange, tqdm
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CACHE_PATH = '/kaggle/working/modelnet40_final.pt'   # change as per the local path 
K_NEIGHBORS = 20          # neighbours for dynamic KNN
NUM_POINTS   = 1024
BATCH_SIZE   = 32
EPOCHS       = 50
LR           = 1e-3
NUM_CLASSES  = 40


# PyTorch 2.6 changed weights_only default to True, which blocks PyG
# Data objects (DataEdgeAttr etc.). Safe to set False for your own cache.
import torch_geometric.data.data          # ensure PyG classes are registered
torch.serialization.add_safe_globals([
    torch_geometric.data.data.DataEdgeAttr,
])
try:
    # Try weights_only=True first (safe path)
    cache = torch.load(CACHE_PATH, map_location='cpu', weights_only=False)
except Exception:
    # Fallback — should not be needed but keeps Kaggle across versions
    cache = torch.load(CACHE_PATH, map_location='cpu', weights_only=False)
 
train_list = cache['train']    # list of PyG Data objects
test_list  = cache['test']
CLASSES    = cache['classes']  # 40 class names from your EDA
print(f"  Train: {len(train_list)}   Test: {len(test_list)}")



