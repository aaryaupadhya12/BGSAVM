# Motif-Grounded Uncertainty-Guided Spreading Activation for Efficient 3D Point Cloud Classification

## What This Is

A novel architecture for 3D point cloud classification that combines five independently published research ideas into a single pipeline. No existing paper combines all five. That gap is the contribution.

---

## The Five Theoretical Pillars

### 1. Dynamic Node Pruning
**Grounded in:** Rao et al., *DynamicViT*, NeurIPS 2021 (`arXiv: 2106.02034`)

DynamicViT prunes image tokens using attention scores. This work replaces that signal with motif participation and evidential uncertainty — adapting the pruning paradigm from 2D images to 3D point cloud graphs.

---

### 2. Motif-Based Graph Structure
**Grounded in:** Bodnar et al., *CW Networks*, NeurIPS 2021 (`arXiv: 2106.12575`)

CWN uses higher-order structures (triangles, rings) to increase GNN expressivity on full graphs. Here, motif participation scores are repurposed as a node selection criterion — nodes low in structural significance are pruned rather than retained with extra weight.

---

### 3. Uncertainty Estimation
**Grounded in:** Sensoy et al., *Evidential Deep Learning*, NeurIPS 2018 (`arXiv: 1806.01768`)

EDL fits a Dirichlet distribution over class probabilities in a single forward pass. Epistemic uncertainty per node (`u = K / S`) measures how little evidence the network has for any class. High uncertainty = uninformative node = pruning candidate.

---

### 4. Vision GNN Backbone
**Grounded in:** Han et al., *Vision GNN*, NeurIPS 2022 (`arXiv: 2206.00272`)

ViG treats images as graphs and processes them with alternating Grapher and FFN blocks. This work applies ViG to 3D point cloud graphs — the first such application — operating on the motif-pruned subgraph rather than the full 1024-node input.

---

### 5. Spreading Activation
**Grounded in:** Anderson, *A Spreading Activation Theory of Memory*, JVLB 1983 + Gasteiger et al., *APPNP*, ICLR 2019 (`arXiv: 1810.05997`)

Anderson's cognitive model describes activation propagating through associative networks weighted by link strength. APPNP implements this computationally via personalized PageRank. Here, edge weights are set by motif co-participation — activation spreads preferentially through structurally significant pathways.

---

## Baseline Comparisons

| Model | Paper | What It Does |
|-------|-------|-------------|
| PointNet++ | Qi et al., NeurIPS 2017 | Hierarchical point set learning, full input |
| DGCNN | Wang et al., ACM TOG 2019 | Dynamic KNN graphs on full point cloud |
| ViG | Han et al., NeurIPS 2022 | Graph backbone on full node set |

This work targets comparable accuracy to the above at significantly reduced node count and FLOPs.

---

## Benchmark

**Dataset:** ModelNet40 — 40-class 3D shape classification, 1024 points per cloud.
