"""
DGCNN backbone with mid-network spreading-activation pruning (Option A).

Pipeline:
    in (7 ch) ── conv1 ── conv2 ─┐
                                 │  seed = mix * motif + (1-mix) * learned(x2)
                                 │  energy = SA.diffuse(...)
                                 │  keep = per_graph_topk(energy, ratio)
                                 └──────► [prune nodes] ──► conv3 ── conv4 ── pool ── MLP
The score gradient reaches the backbone through a soft saliency re-weighting
on surviving nodes (``x2 *= 1 + saliency_scale * score``); the top-k selection
itself is hard (DynamicViT style).

Requires ``torch_cluster`` for dynamic k-NN in feature space (matches Anshull's
``BaselineDGCNN``).  Install the matching wheel from ``https://data.pyg.org/whl``.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import EdgeConv, global_max_pool, global_mean_pool

try:
    from torch_cluster import knn_graph
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "SA_DGCNN needs torch_cluster. Install the wheel matching your torch/cuda "
        "from https://data.pyg.org/whl — e.g. "
        "`pip install torch-cluster -f https://data.pyg.org/whl/torch-2.2.2+cu118.html`."
    ) from e

from sa_module import LearnableSpreadingActivation


def _edgeconv_block(in_ch: int, out_ch: int) -> EdgeConv:
    """Single-linear EdgeConv block matching Anshull's BaselineDGCNN layout."""
    return EdgeConv(
        nn.Sequential(
            nn.Linear(2 * in_ch, out_ch, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.LeakyReLU(0.2),
        ),
        aggr="max",
    )


class SA_DGCNN(nn.Module):
    """DGCNN + spreading-activation pruner between conv2 and conv3."""

    def __init__(
        self,
        in_channels: int = 7,
        num_classes: int = 40,
        k: int = 20,
        dropout: float = 0.5,
        embed_dims: int = 1024,
        prune_ratio: float = 0.7,
        prune_min_nodes: int = 64,
        sa_steps: int = 2,
        sa_learnable: bool = True,
        sa_reweight_each_step: bool = False,
        saliency_scale: float = 0.25,
        seed_mix_init: float = 0.5,
        small: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.k = int(k)
        self.prune_ratio = float(prune_ratio) if prune_ratio is not None else 1.0
        self.prune_min_nodes = int(prune_min_nodes)
        self.saliency_scale = float(saliency_scale)

        if small:
            ch1, ch2, ch3, ch4 = 32, 32, 64, 128
            embed_dims = min(embed_dims, 512)
        else:
            ch1, ch2, ch3, ch4 = 64, 64, 128, 256

        self.conv1 = _edgeconv_block(self.in_channels, ch1)
        self.conv2 = _edgeconv_block(ch1,              ch2)
        self.conv3 = _edgeconv_block(ch2,              ch3)
        self.conv4 = _edgeconv_block(ch3,              ch4)

        self.sa = LearnableSpreadingActivation(
            num_steps=sa_steps,
            learnable=sa_learnable,
            reweight_each_step=sa_reweight_each_step,
        )
        self.seed_head = nn.Linear(ch2, 1, bias=False)
        self.seed_mix_logit = nn.Parameter(
            torch.tensor(_inv_sigmoid(seed_mix_init)), requires_grad=True
        )

        total = ch1 + ch2 + ch3 + ch4
        self.global_mlp = nn.Sequential(
            nn.Linear(total * 2, embed_dims, bias=False),
            nn.BatchNorm1d(embed_dims),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
        )
        if small:
            self.classifier = nn.Sequential(
                nn.Linear(embed_dims, 256, bias=False), nn.BatchNorm1d(256),
                nn.LeakyReLU(0.2), nn.Dropout(dropout),
                nn.Linear(256, 128, bias=False), nn.BatchNorm1d(128),
                nn.LeakyReLU(0.2), nn.Dropout(dropout),
                nn.Linear(128, num_classes),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(embed_dims, 512, bias=False), nn.BatchNorm1d(512),
                nn.LeakyReLU(0.2), nn.Dropout(dropout),
                nn.Linear(512, 256, bias=False), nn.BatchNorm1d(256),
                nn.LeakyReLU(0.2), nn.Dropout(dropout),
                nn.Linear(256, num_classes),
            )

    # ── public introspection helpers ──
    @property
    def seed_mix(self) -> torch.Tensor:
        return torch.sigmoid(self.seed_mix_logit)

    def current_hparams(self) -> dict[str, float]:
        h = self.sa.current_hparams()
        h["seed_mix"] = float(self.seed_mix.detach())
        return h

    # ── forward ──
    def forward(self, data: Data) -> torch.Tensor:
        x = data.x[:, : self.in_channels]
        pos = data.pos
        batch = data.batch

        motif = _motif_prior(data, num_nodes=pos.size(0), device=pos.device)

        # --- Block 1 (positional k-NN) ---
        # knn_graph is a C++ op with float32/float64 dispatch — force fp32 even under AMP.
        ei = knn_graph(pos.float(), k=self.k, batch=batch, loop=False)
        x1 = self.conv1(x, ei)

        # --- Block 2 (feature-space k-NN) ---
        ei = knn_graph(x1.float(), k=self.k, batch=batch, loop=False)
        x2 = self.conv2(x1, ei)

        # --- SA pruning point ---
        if self.prune_ratio < 1.0:
            # Run SA in fp32 — softmax/scatter/min-max normalize are unstable in fp16.
            with torch.amp.autocast("cuda", enabled=False):
                learned = torch.sigmoid(self.seed_head(x2.float())).squeeze(-1)
                mix = self.seed_mix
                seed = mix * motif.float() + (1.0 - mix) * learned
                out = self.sa(
                    pos=pos.float(), edge_index=ei, batch=batch, seed=seed,
                    prune_ratio=self.prune_ratio, min_nodes=self.prune_min_nodes,
                )
            keep = out.keep_mask
            kept_score = out.energy[keep].to(x2.dtype)
            x1, x2 = x1[keep], x2[keep]
            pos, batch = pos[keep], batch[keep]
            # soft saliency reweighting so gradients flow through the score
            x2 = x2 * (1.0 + self.saliency_scale * kept_score.unsqueeze(-1))

        # --- Block 3 ---
        ei = knn_graph(x2.float(), k=self.k, batch=batch, loop=False)
        x3 = self.conv3(x2, ei)

        # --- Block 4 ---
        ei = knn_graph(x3.float(), k=self.k, batch=batch, loop=False)
        x4 = self.conv4(x3, ei)

        feats = torch.cat([x1, x2, x3, x4], dim=1)
        g = torch.cat(
            [global_max_pool(feats, batch), global_mean_pool(feats, batch)],
            dim=1,
        )
        return self.classifier(self.global_mlp(g))


def _motif_prior(data: Data, num_nodes: int, device: torch.device) -> torch.Tensor:
    if getattr(data, "motif", None) is not None:
        return data.motif.float()
    if data.x is not None and data.x.size(1) >= 7:
        return data.x[:, 6].float()
    return torch.full((num_nodes,), 0.5, device=device, dtype=torch.float32)


def _inv_sigmoid(p: float) -> float:
    p = min(max(p, 1e-4), 1.0 - 1e-4)
    return float(torch.logit(torch.tensor(p)).item())
