"""
Legacy ``EnergySpreadingActivation`` entry point.

Kept so the existing static-kNN trainer keeps importing:

    from spreading_activation import EnergySpreadingActivation

New code should import :class:`LearnableSpreadingActivation` and the core
primitives directly — see ``sa_module.py`` and ``sa_core.py``.

The implementation is now a thin wrapper around the shared core, with every
correctness bug from the original version fixed:

* Per-step normalisation removed — runs once, at the end, preserving the
  ``recharge`` signal through the loop.
* Zero-span graphs return 0.5 instead of collapsing to 0.
* ``per_graph_topk`` is vectorised and deterministic under ties.
* Hyperparameters can be learnable (via ``LearnableSpreadingActivation``).
* Edge weights can optionally be recomputed every step
  (``reweight_each_step=True``).
* ``attach_spreading_activation`` no longer double-applies energy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from torch import Tensor
from torch_geometric.data import Data

from sa_core import per_graph_topk  # re-exported for callers
from sa_module import (
    LearnableSpreadingActivation,
    SpreadingActivationOutput as _SAOut,
    attach_spreading_activation as _attach,
)


@dataclass
class SpreadingActivationOutput:
    """Legacy output struct — ``initial_energy`` / ``edge_weight`` / ``weighted_x`` are no longer populated.

    Callers of the new ``LearnableSpreadingActivation`` should use
    :class:`sa_module.SpreadingActivationOutput` instead, which only returns
    the fields that actually have a single canonical meaning.
    """
    energy: Tensor
    initial_energy: Optional[Tensor] = None
    edge_weight: Optional[Tensor] = None
    weighted_x: Optional[Tensor] = None
    pruned_data: Optional[Data] = None
    keep_mask: Optional[Tensor] = None


class EnergySpreadingActivation(LearnableSpreadingActivation):
    """Drop-in replacement for the original module.

    Backwards-compatible constructor signature (``num_steps``, ``decay``,
    ``self_retention``, ``recharge``, ``temperature``, ``motif_bias``).
    Hyperparameters are *frozen* by default — behaviour matches the old
    code.  Pass ``learnable=True`` to train them end-to-end.
    """

    def __init__(
        self,
        num_steps: int = 4,
        decay: float = 0.92,
        self_retention: float = 0.35,
        recharge: float = 0.20,
        temperature: float = 0.20,
        motif_bias: float = 0.50,
        learnable: bool = False,
        reweight_each_step: bool = False,
    ) -> None:
        super().__init__(
            num_steps=num_steps,
            init_decay=decay,
            init_retention=self_retention,
            init_recharge=recharge,
            init_temperature=temperature,
            init_seed_bias=motif_bias,
            learnable=learnable,
            reweight_each_step=reweight_each_step,
        )

    def forward(self, data, seed_energy=None, prune_ratio=None, min_nodes=64):
        out = super().forward(
            data=data, seed=seed_energy,
            prune_ratio=prune_ratio, min_nodes=min_nodes,
        )
        return SpreadingActivationOutput(
            energy=out.energy,
            keep_mask=out.keep_mask,
            pruned_data=out.pruned_data,
        )


def attach_spreading_activation(
    data: Data,
    module: EnergySpreadingActivation,
    prune_ratio: Optional[float] = None,
    min_nodes: int = 64,
) -> Data:
    """Legacy helper — residual-mode feature scaling (safe default)."""
    return _attach(data, module, prune_ratio=prune_ratio, min_nodes=min_nodes,
                   feature_mode="residual", feature_scale=0.25)
