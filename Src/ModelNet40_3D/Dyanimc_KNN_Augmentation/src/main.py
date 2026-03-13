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

