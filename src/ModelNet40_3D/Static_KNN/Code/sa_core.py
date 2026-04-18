"""
Core spreading-activation primitives.

Pure tensor-in / tensor-out functions — no PyG ``Data`` handling, no Python
loops over graphs.  Autograd-friendly so they can be used either as a
preprocessing step *or* inside a differentiable model.

Design notes (fixes vs. the original ``EnergySpreadingActivation``):

1. Normalisation happens *once*, after the diffusion loop — the earlier code
   renormalised inside the loop, which stretched ``[min, max] -> [0, 1]``
   every step and killed the ``recharge`` signal.
2. ``_normalize_per_graph`` returns 0.5 on graphs with zero span instead of
   silently collapsing to zero.
3. ``per_graph_topk`` is vectorised (no Python loop) and breaks ties
   deterministically via node index.
4. Edge weights can be recomputed every step (``reweight_each_step=True``) to
   match Anderson's true spreading-activation dynamics, or kept fixed from the
   initial seed for an APPNP-style linear operator (the default, for speed).
5. Hyperparameters accept either plain floats *or* 0-dim learnable tensors,
   so the same core is reused by the learnable ``nn.Module`` wrapper.
"""
from __future__ import annotations

from typing import Union

import torch
from torch import Tensor
from torch_geometric.utils import scatter, softmax

EPS = 1e-8
Scalar = Union[float, Tensor]


def normalize_per_graph(values: Tensor, batch: Tensor) -> Tensor:
    """Min-max scale to [0, 1] per graph; return 0.5 where the span is zero."""
    graph_min = scatter(values, batch, dim=0, reduce="min")[batch]
    graph_max = scatter(values, batch, dim=0, reduce="max")[batch]
    span = graph_max - graph_min
    normed = (values - graph_min) / span.clamp_min(EPS)
    return torch.where(span > EPS, normed, torch.full_like(values, 0.5))


def build_edge_weights(
    pos: Tensor,
    batch: Tensor,
    edge_index: Tensor,
    seed_energy: Tensor,
    temperature: Scalar,
    seed_bias: Scalar,
) -> Tensor:
    """Softmax-over-destinations edge weights with distance locality + seed affinity.

    Weights are normalised per destination node (in-flow conservation): each
    node aggregates a weighted distribution of its senders rather than simply
    accumulating more from dense regions.  Changing this to per-source would
    turn the operator into an influence-amplification model instead.
    """
    src, dst = edge_index
    dist = torch.norm(pos[src] - pos[dst], dim=1)

    graph_max = scatter(pos, batch, dim=0, reduce="max")
    graph_min = scatter(pos, batch, dim=0, reduce="min")
    graph_diag = torch.norm(graph_max - graph_min, dim=1).clamp_min(EPS)

    temp = torch.as_tensor(temperature, device=dist.device, dtype=dist.dtype).clamp_min(EPS)
    edge_temperature = graph_diag[batch[dst]] * temp
    locality = torch.exp(-dist / edge_temperature)

    bias = torch.as_tensor(seed_bias, device=dist.device, dtype=dist.dtype)
    affinity = 1.0 + bias * 0.5 * (seed_energy[src] + seed_energy[dst])
    locality = locality * affinity

    return softmax(locality, dst)


def diffuse(
    pos: Tensor,
    edge_index: Tensor,
    batch: Tensor,
    seed: Tensor,
    num_steps: int,
    decay: Scalar,
    retention: Scalar,
    recharge: Scalar,
    temperature: Scalar,
    seed_bias: Scalar,
    reweight_each_step: bool = False,
) -> Tensor:
    """Run ``num_steps`` of spreading activation and return per-node energy in [0, 1]."""
    initial = normalize_per_graph(seed.float(), batch)

    edge_weight = build_edge_weights(
        pos=pos, batch=batch, edge_index=edge_index,
        seed_energy=initial, temperature=temperature, seed_bias=seed_bias,
    )

    energy = initial
    src, dst = edge_index
    num_nodes = pos.size(0)

    for _ in range(int(num_steps)):
        if reweight_each_step:
            edge_weight = build_edge_weights(
                pos=pos, batch=batch, edge_index=edge_index,
                seed_energy=energy, temperature=temperature, seed_bias=seed_bias,
            )
        transmitted = energy[src] * edge_weight
        incoming = scatter(transmitted, dst, dim=0, dim_size=num_nodes, reduce="sum")
        energy = decay * (retention * energy + (1.0 - retention) * incoming) + recharge * initial

    return normalize_per_graph(energy, batch)


def per_graph_topk(
    scores: Tensor,
    batch: Tensor,
    keep_ratio: float,
    min_nodes: int = 1,
) -> Tensor:
    """Vectorised per-graph top-k keep mask, deterministic under ties."""
    keep_ratio = float(max(0.0, min(1.0, keep_ratio)))
    n = scores.numel()
    device = scores.device

    idx = torch.arange(n, device=device, dtype=scores.dtype)
    s = scores + 1e-12 * idx

    # Sort by (batch asc, score desc) via two stable sorts.
    _, order = torch.sort(-s, stable=True)
    _, order2 = torch.sort(batch[order], stable=True)
    order = order[order2]

    graph_sizes = scatter(
        torch.ones(n, dtype=torch.long, device=device),
        batch, dim=0, reduce="sum",
    )
    offsets = torch.cumsum(graph_sizes, 0) - graph_sizes
    ranks = torch.arange(n, device=device) - offsets[batch[order]]

    k = torch.clamp(
        (graph_sizes.float() * keep_ratio).round().long(),
        min=int(min_nodes),
    ).clamp(max=graph_sizes)

    keep_sorted = ranks < k[batch[order]]

    mask = torch.zeros(n, dtype=torch.bool, device=device)
    mask[order[keep_sorted]] = True
    return mask
