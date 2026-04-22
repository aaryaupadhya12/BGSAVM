"""
DGCNN backbone with mid-network spreading-activation gating (Option A).

Pipeline (soft-gate, default):
    in (7 ch) ── conv1 ── conv2 ─┐
                                 │  seed  = mix * motif + (1-mix) * MLP(x2)
                                 │  energy = SA.diffuse(...)          (in [0,1])
                                 │  gate   = gate_floor + (1-gate_floor) * energy
                                 │  x2    *= gate
                                 │  x2_in  = cat([x2, energy])          ⬅ feature
                                 └──► conv3(ch2+1 → ch3) ─ conv4 ─ pool ─ MLP

Pipeline (hard-prune, legacy / for ablation):
    ... ── conv2 ── top-k(energy) ── [drop pruned nodes] ── conv3 ── conv4 ...

Why soft gate by default:
    DGCNN's identity is the concat of *all four* EdgeConv outputs before pooling.
    Hard-pruning between conv2 and conv3 drops the conv1/conv2 contributions of
    30% of nodes from the pooled representation entirely — that's information
    loss the original DGCNN recipe never had.  A soft gate keeps every node in
    the graph, lets SA modulate how loudly each node speaks to conv3+, and is
    fully differentiable without DynamicViT-style hacks.

Why feed energy as a feature to conv3:
    Without it, SA only modulates magnitudes via the gate — it can attenuate,
    but it can't tell the downstream conv *which nodes are salient*.  Concating
    energy as an extra channel lets conv3 reason about saliency explicitly.
    The extra input channel (ch2 → ch2+1) is always present, even in off-mode
    (zero-filled), so the architecture is apples-to-apples across ablations.

Requires ``torch_cluster`` for dynamic k-NN in feature space.
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
    """DGCNN with a spreading-activation saliency module between conv2 and conv3.

    ``gate_mode``:
        - ``"soft"`` (default): every node survives; ``x2 *= gate`` where
          ``gate ∈ [gate_floor, 1]``.  Fully differentiable; preserves DGCNN's
          multiscale concat.
        - ``"hard"``: legacy DynamicViT-style behavior — top-k prune + soft
          saliency re-weight on survivors.  Kept for ablation.
        - ``"off"``: skip SA entirely (baseline DGCNN).
    """

    def __init__(
        self,
        in_channels: int = 7,
        num_classes: int = 40,
        k: int = 20,
        dropout: float = 0.5,
        embed_dims: int = 1024,
        gate_mode: str = "soft",
        gate_floor: float = 0.1,
        gate_points: tuple[int, ...] = (2,),   # which conv outputs to gate after; {2}, {3}, or {2,3}
        prune_ratio: float = 0.7,              # only used when gate_mode == "hard"
        prune_min_nodes: int = 307,            # ≈ 0.3 * 1024; old default 64 was too loose
        saliency_scale: float = 0.25,          # only used when gate_mode == "hard"
        sa_steps: int = 2,
        sa_learnable: bool = True,
        sa_reweight_each_step: bool = False,
        seed_mix_init: float = 0.5,
        small: bool = False,
    ) -> None:
        super().__init__()
        if gate_mode not in ("soft", "hard", "off"):
            raise ValueError(f"gate_mode must be one of soft/hard/off, got {gate_mode!r}")
        gate_points = tuple(sorted(set(int(p) for p in gate_points)))
        for p in gate_points:
            if p not in (2, 3):
                raise ValueError(f"gate_points must be a subset of (2, 3), got {gate_points}")
        if gate_mode == "hard" and gate_points != (2,):
            raise ValueError(
                "gate_mode='hard' only supports gate_points=(2,). "
                "Use gate_mode='soft' for pyramidal gating at (2, 3)."
            )
        self.in_channels = int(in_channels)
        self.k = int(k)
        self.gate_mode = gate_mode
        self.gate_floor = float(gate_floor)
        self.gate_points = gate_points
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
        # conv3 and conv4 both widened by 1 to accept the SA energy as an input
        # feature.  Channel is zero-filled when that gate point is disabled,
        # so the architecture is apples-to-apples across all ablations.
        self.conv3 = _edgeconv_block(ch2 + 1,          ch3)
        self.conv4 = _edgeconv_block(ch3 + 1,          ch4)

        self.sa = LearnableSpreadingActivation(
            num_steps=sa_steps,
            learnable=sa_learnable,
            reweight_each_step=sa_reweight_each_step,
        )
        # 2-layer MLP saliency heads — one per potential gate point.
        # Always instantiated for constant architecture (≈10K unused params in off-mode).
        def _seed_mlp(in_ch: int) -> nn.Sequential:
            hidden = max(in_ch // 2, 8)
            return nn.Sequential(
                nn.Linear(in_ch, hidden, bias=False),
                nn.BatchNorm1d(hidden),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden, 1),
            )
        self.seed_head_2 = _seed_mlp(ch2)
        self.seed_head_3 = _seed_mlp(ch3)

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
    def _apply_gate(
        self,
        feats: torch.Tensor,
        seed_head: nn.Module,
        motif: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        x1_for_hard: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Run SA at one gate point and return (gated_feats, energy_channel, pos, batch, x1_for_hard).

        `x1_for_hard` is only sliced in hard-prune mode; pass ``None`` at the
        second gate point where x1 has already been finalised.
        """
        with torch.amp.autocast("cuda", enabled=False):
            learned = torch.sigmoid(seed_head(feats.float())).squeeze(-1)
            mix = self.seed_mix
            seed = mix * motif.float() + (1.0 - mix) * learned
            out = self.sa(
                pos=pos.float(), edge_index=edge_index, batch=batch, seed=seed,
                prune_ratio=self.prune_ratio if self.gate_mode == "hard" else None,
                min_nodes=self.prune_min_nodes,
            )

        if self.gate_mode == "soft":
            energy = out.energy.to(feats.dtype)
            gate = self.gate_floor + (1.0 - self.gate_floor) * energy
            feats = feats * gate.unsqueeze(-1)
            return feats, energy, pos, batch, x1_for_hard

        # "hard" — legacy prune + saliency reweight
        keep = out.keep_mask
        kept_score = out.energy[keep].to(feats.dtype)
        feats = feats[keep] * (1.0 + self.saliency_scale * kept_score.unsqueeze(-1))
        pos, batch = pos[keep], batch[keep]
        if x1_for_hard is not None:
            x1_for_hard = x1_for_hard[keep]
        return feats, kept_score, pos, batch, x1_for_hard

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

        # --- Gate point #1 (after conv2) ---
        energy_ch_2 = torch.zeros(x2.size(0), device=x2.device, dtype=x2.dtype)
        if self.gate_mode != "off" and 2 in self.gate_points:
            x2, energy_ch_2, pos, batch, x1 = self._apply_gate(
                x2, self.seed_head_2, motif, pos, ei, batch, x1_for_hard=x1,
            )

        # --- Block 3 (feature-space k-NN on x2; conv3 gets [x2, energy_ch_2]) ---
        ei = knn_graph(x2.float(), k=self.k, batch=batch, loop=False)
        x3 = self.conv3(torch.cat([x2, energy_ch_2.unsqueeze(-1)], dim=1), ei)

        # --- Gate point #2 (after conv3).  Only valid with gate_mode='soft'. ---
        energy_ch_3 = torch.zeros(x3.size(0), device=x3.device, dtype=x3.dtype)
        if self.gate_mode == "soft" and 3 in self.gate_points:
            x3, energy_ch_3, pos, batch, _ = self._apply_gate(
                x3, self.seed_head_3, motif, pos, ei, batch, x1_for_hard=None,
            )

        # --- Block 4 (conv4 gets [x3, energy_ch_3]) ---
        ei = knn_graph(x3.float(), k=self.k, batch=batch, loop=False)
        x4 = self.conv4(torch.cat([x3, energy_ch_3.unsqueeze(-1)], dim=1), ei)

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
