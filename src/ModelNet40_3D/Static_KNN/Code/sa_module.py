"""
nn.Module wrappers around the core spreading-activation primitives.

Two entry points:

* ``LearnableSpreadingActivation`` — all five diffusion hyperparameters
  (decay, retention, recharge, temperature, seed_bias) are wrapped in
  sigmoid-mapped ``nn.Parameter`` s so the module can be trained end-to-end.
  Forward supports either a PyG ``Data`` object (preprocessing style) or
  explicit ``(pos, edge_index, batch, seed)`` tensors (in-network style).

* ``attach_spreading_activation`` — thin convenience helper that preserves
  the legacy API used by the notebook / static-kNN trainer.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn
from torch_geometric.data import Data
from torch_geometric.utils import subgraph

from sa_core import diffuse, per_graph_topk


@dataclass
class SpreadingActivationOutput:
    energy: Tensor
    keep_mask: Optional[Tensor] = None
    pruned_data: Optional[Data] = None


def _logit(p: float) -> float:
    p = min(max(p, 1e-4), 1.0 - 1e-4)
    return float(torch.logit(torch.tensor(p)).item())


def _infer_seed(data: Data) -> Tensor:
    """Prefer named attributes; fall back to motif column for legacy caches."""
    if getattr(data, "seed_energy", None) is not None:
        return data.seed_energy.float()
    if getattr(data, "motif", None) is not None:
        return data.motif.float()
    if data.x is not None and data.x.size(1) >= 7:
        warnings.warn(
            "Reading motif from data.x[:, 6] — attach data.motif during "
            "preprocessing to avoid relying on column order.",
            UserWarning, stacklevel=3,
        )
        return data.x[:, 6].float()
    raise ValueError(
        "No seed source found. Set data.motif / data.seed_energy or pass "
        "an explicit `seed` tensor."
    )


class LearnableSpreadingActivation(nn.Module):
    """Spreading-activation diffusion with optionally learnable hyperparameters.

    The five knobs are stored as logits and exposed via sigmoid-mapped
    properties, so the same code reads ``self.decay`` whether the module is
    frozen or trainable.  ``temperature`` lives in ``(0.01, 1.01)`` to avoid
    division-by-zero in edge-weight construction.
    """

    def __init__(
        self,
        num_steps: int = 4,
        init_decay: float = 0.92,
        init_retention: float = 0.35,
        init_recharge: float = 0.20,
        init_temperature: float = 0.20,
        init_seed_bias: float = 0.50,
        learnable: bool = True,
        reweight_each_step: bool = False,
    ) -> None:
        super().__init__()
        self.num_steps = int(num_steps)
        self.reweight_each_step = bool(reweight_each_step)

        def _p(x: float) -> nn.Parameter:
            return nn.Parameter(torch.tensor(_logit(x)), requires_grad=bool(learnable))

        self._decay_logit       = _p(init_decay)
        self._retention_logit   = _p(init_retention)
        self._recharge_logit    = _p(init_recharge)
        self._temperature_logit = _p(init_temperature)
        self._seed_bias_logit   = _p(init_seed_bias)

    @property
    def decay(self) -> Tensor:
        # Cap below 1 so the diffusion recurrence can't compound into NaN.
        return 0.99 * torch.sigmoid(self._decay_logit)
    @property
    def retention(self) -> Tensor:   return torch.sigmoid(self._retention_logit)
    @property
    def recharge(self) -> Tensor:    return torch.sigmoid(self._recharge_logit)
    @property
    def temperature(self) -> Tensor: return 0.01 + torch.sigmoid(self._temperature_logit)
    @property
    def seed_bias(self) -> Tensor:   return torch.sigmoid(self._seed_bias_logit)

    def current_hparams(self) -> dict[str, float]:
        return {
            "decay":       float(self.decay.detach()),
            "retention":   float(self.retention.detach()),
            "recharge":    float(self.recharge.detach()),
            "temperature": float(self.temperature.detach()),
            "seed_bias":   float(self.seed_bias.detach()),
        }

    def forward(
        self,
        data: Optional[Data] = None,
        *,
        pos: Optional[Tensor] = None,
        edge_index: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
        seed: Optional[Tensor] = None,
        prune_ratio: Optional[float] = None,
        min_nodes: int = 1,
    ) -> SpreadingActivationOutput:
        pos_t, ei_t, batch_t, seed_t = self._resolve_inputs(data, pos, edge_index, batch, seed)

        energy = diffuse(
            pos=pos_t, edge_index=ei_t, batch=batch_t, seed=seed_t,
            num_steps=self.num_steps,
            decay=self.decay, retention=self.retention, recharge=self.recharge,
            temperature=self.temperature, seed_bias=self.seed_bias,
            reweight_each_step=self.reweight_each_step,
        )

        keep_mask: Optional[Tensor] = None
        pruned_data: Optional[Data] = None
        if prune_ratio is not None and prune_ratio < 1.0:
            keep_mask = per_graph_topk(energy, batch_t, prune_ratio, min_nodes=min_nodes)
            if data is not None:
                pruned_data = self._build_pruned_data(data, keep_mask, batch_t, energy)

        return SpreadingActivationOutput(energy=energy, keep_mask=keep_mask, pruned_data=pruned_data)

    @staticmethod
    def _resolve_inputs(data, pos, edge_index, batch, seed):
        if data is not None:
            pos = data.pos if pos is None else pos
            edge_index = data.edge_index if edge_index is None else edge_index
            if batch is None:
                batch = getattr(data, "batch", None)
                if batch is None:
                    batch = torch.zeros(pos.size(0), dtype=torch.long, device=pos.device)
            if seed is None:
                seed = _infer_seed(data)
        else:
            if pos is None or edge_index is None or seed is None:
                raise ValueError("Either `data` or (pos, edge_index, seed) must be provided.")
            if batch is None:
                batch = torch.zeros(pos.size(0), dtype=torch.long, device=pos.device)
        return pos, edge_index, batch, seed

    @staticmethod
    def _build_pruned_data(data: Data, keep_mask: Tensor, batch: Tensor, energy: Tensor) -> Data:
        new_edge_index, _ = subgraph(keep_mask, data.edge_index, relabel_nodes=True)
        pruned = Data(
            x=data.x[keep_mask],
            pos=data.pos[keep_mask],
            edge_index=new_edge_index,
            y=data.y,
        )
        if getattr(data, "norm", None) is not None:
            pruned.norm = data.norm[keep_mask]
        for attr in ("motif", "linearity", "planarity", "curvature", "seed_energy"):
            v = getattr(data, attr, None)
            if v is not None:
                setattr(pruned, attr, v[keep_mask])
        pruned.batch = batch[keep_mask]
        pruned.energy = energy[keep_mask]
        return pruned


def attach_spreading_activation(
    data: Data,
    module: LearnableSpreadingActivation,
    prune_ratio: Optional[float] = None,
    min_nodes: int = 64,
    feature_scale: float = 0.25,
    feature_mode: str = "residual",
) -> Data:
    """Run the module and return a new ``Data`` with energy-aware features.

    ``feature_mode='residual'`` → ``x *= 1 + feature_scale * energy``  (safe default)
    ``feature_mode='multiply'`` → ``x *= energy``                       (legacy)
    """
    output = module(data, prune_ratio=prune_ratio, min_nodes=min_nodes)

    if output.pruned_data is not None:
        result = output.pruned_data
        kept_energy = output.energy[output.keep_mask].unsqueeze(-1)
    else:
        result = data.clone()
        result.energy = output.energy
        kept_energy = output.energy.unsqueeze(-1)

    base = result.x.float()
    if feature_mode == "multiply":
        result.x = base * kept_energy
    else:
        result.x = base * (1.0 + feature_scale * kept_energy)
    return result
