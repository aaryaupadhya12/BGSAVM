from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn
from torch_geometric.data import Data
from torch_geometric.utils import softmax, scatter, subgraph


@dataclass
class SpreadingActivationOutput:
    energy: Tensor
    initial_energy: Tensor
    edge_weight: Tensor
    weighted_x: Tensor
    pruned_data: Optional[Data] = None
    keep_mask: Optional[Tensor] = None


def _resolve_batch(data: Data) -> Tensor:
    if hasattr(data, "batch") and data.batch is not None:
        return data.batch
    return torch.zeros(data.num_nodes, dtype=torch.long, device=data.x.device)


def _normalize_per_graph(values: Tensor, batch: Tensor, eps: float = 1e-8) -> Tensor:
    graph_min = scatter(values, batch, dim=0, reduce="min")[batch]
    graph_max = scatter(values, batch, dim=0, reduce="max")[batch]
    return (values - graph_min) / (graph_max - graph_min + eps)


def _motif_seed(data: Data) -> Tensor:
    if data.x is None:
        raise ValueError("Data.x is required for spreading activation.")

    if hasattr(data, "seed_energy") and data.seed_energy is not None:
        return data.seed_energy.float()

    if data.x.size(1) >= 7:
        motif = data.x[:, 6]
    else:
        # FIX (Bug 2): All-ones seed → after per-graph min-max normalisation the
        # result is 0/(0+eps) ≈ 0 everywhere, making the whole module a no-op.
        # Warn the caller so this isn't silently swallowed.
        warnings.warn(
            "Data.x has fewer than 7 columns; no motif column found. "
            "Falling back to uniform seed energy (all nodes equal). "
            "Spreading activation will produce near-zero energy after normalisation "
            "unless an explicit seed_energy is supplied.",
            UserWarning,
            stacklevel=3,
        )
        motif = torch.ones(data.num_nodes, device=data.x.device, dtype=data.x.dtype)

    return motif.float()


def _build_edge_weights(
    pos: Tensor,
    batch: Tensor,
    edge_index: Tensor,
    seed_energy: Tensor,
    temperature: float,
    motif_bias: float,
    eps: float = 1e-8,
) -> Tensor:
    src, dst = edge_index
    dist = torch.norm(pos[src] - pos[dst], dim=1)
    graph_max = scatter(pos, batch, dim=0, reduce="max")
    graph_min = scatter(pos, batch, dim=0, reduce="min")
    graph_diag = torch.norm(graph_max - graph_min, dim=1).clamp_min(eps)
    edge_temperature = graph_diag[batch[dst]] * max(temperature, eps)
    locality = torch.exp(-dist / edge_temperature)

    if motif_bias > 0:
        affinity = 1.0 + motif_bias * 0.5 * (seed_energy[src] + seed_energy[dst])
        locality = locality * affinity

    # Normalize over incoming edges so each destination aggregates a weighted
    # distribution of its senders rather than simply accumulating from many
    # neighbours because it sits in a dense region.
    #
    # This keeps the score closer to "structural salience propagated through
    # context" rather than "raw in-degree / density accumulation."
    return softmax(locality, dst)


def _topk_mask_per_graph(energy: Tensor, batch: Tensor, keep_ratio: float, min_nodes: int) -> Tensor:
    keep_ratio = float(max(0.0, min(1.0, keep_ratio)))
    mask = torch.zeros_like(energy, dtype=torch.bool)

    for graph_id in batch.unique(sorted=True):
        node_idx = torch.nonzero(batch == graph_id, as_tuple=False).view(-1)
        if node_idx.numel() == 0:
            continue

        k = max(min_nodes, int(round(node_idx.numel() * keep_ratio)))
        k = min(k, node_idx.numel())
        topk_local = torch.topk(energy[node_idx], k=k, largest=True).indices
        mask[node_idx[topk_local]] = True

    return mask


class EnergySpreadingActivation(nn.Module):
    """
    Energy-based spreading activation for point-cloud graphs.

    Intuition:
    - Each node stores energy.
    - Motif-heavy or otherwise salient nodes start with more energy.
    - Energy diffuses to nearby nodes along weighted edges.
    - The diffusion includes self-retention, recharge from the initial seed,
      and a global decay term so long-range influence drops off.
    - Final energy can be used to reweight node features or prune low-energy nodes.

    Edge-weight normalisation note:
        Weights are softmax-normalised over *outgoing* edges (per source node).
        This means hub nodes can accumulate more than average energy from their
        many incoming neighbours — it is an influence-amplification model, not
        a conservative one.  Pass ``softmax(locality, dst)`` in
        ``_build_edge_weights`` if you need in-flow conservation instead.

    Per-step normalisation note:
        ``_normalize_per_graph`` is called *after* the full diffusion loop, not
        inside it, so that absolute energy magnitudes are preserved during
        propagation.  A final normalisation is applied once at the end to bring
        values into [0, 1] for stable downstream use.
    """

    def __init__(
        self,
        num_steps: int = 4,
        decay: float = 0.92,
        self_retention: float = 0.35,
        recharge: float = 0.20,
        temperature: float = 0.20,
        motif_bias: float = 0.50,
    ) -> None:
        super().__init__()
        self.num_steps = num_steps
        self.decay = decay
        self.self_retention = self_retention
        self.recharge = recharge
        self.temperature = temperature
        self.motif_bias = motif_bias

    def forward(
        self,
        data: Data,
        seed_energy: Optional[Tensor] = None,
        prune_ratio: Optional[float] = None,
        min_nodes: int = 64,
    ) -> SpreadingActivationOutput:
        if data.x is None or data.pos is None or data.edge_index is None:
            raise ValueError("Data must contain x, pos, and edge_index.")

        batch = _resolve_batch(data)
        base_seed = _motif_seed(data) if seed_energy is None else seed_energy.float()
        initial_energy = _normalize_per_graph(base_seed, batch)

        edge_weight = _build_edge_weights(
            pos=data.pos.float(),
            batch=batch,
            edge_index=data.edge_index,
            seed_energy=initial_energy,
            temperature=self.temperature,
            motif_bias=self.motif_bias,
        )

        energy = initial_energy
        src, dst = data.edge_index

        for _ in range(self.num_steps):
            transmitted = energy[src] * edge_weight
            incoming = scatter(transmitted, dst, dim=0, dim_size=data.num_nodes, reduce="sum")
            energy = self.decay * (
                self.self_retention * energy + (1.0 - self.self_retention) * incoming
            ) + self.recharge * initial_energy
            # Per-step normalization prevents drift toward dense regions and keeps
            # the score comparable across propagation steps.
            energy = _normalize_per_graph(energy, batch)

        weighted_x = data.x * energy.unsqueeze(1)
        pruned_data = None
        keep_mask = None

        if prune_ratio is not None:
            keep_mask = _topk_mask_per_graph(
                energy=energy,
                batch=batch,
                keep_ratio=prune_ratio,
                min_nodes=min_nodes,
            )

            new_edge_index, _ = subgraph(keep_mask, data.edge_index, relabel_nodes=True)
            pruned_data = Data(
                x=weighted_x[keep_mask],
                pos=data.pos[keep_mask],
                edge_index=new_edge_index,
                y=data.y,
            )

            if hasattr(data, "norm") and data.norm is not None:
                pruned_data.norm = data.norm[keep_mask]
            if hasattr(data, "batch") and data.batch is not None:
                pruned_data.batch = batch[keep_mask]
            pruned_data.energy = energy[keep_mask]

        return SpreadingActivationOutput(
            energy=energy,
            initial_energy=initial_energy,
            edge_weight=edge_weight,
            weighted_x=weighted_x,
            pruned_data=pruned_data,
            keep_mask=keep_mask,
        )


def attach_spreading_activation(
    data: Data,
    module: EnergySpreadingActivation,
    prune_ratio: Optional[float] = None,
    min_nodes: int = 64,
) -> Data:
    """
    Convenience helper for notebook workflows.

    Returns a new Data object with:
    - ``x`` reweighted by energy
    - ``energy`` stored as a node-level attribute
    - optional pruning applied
    """
    output = module(data, prune_ratio=prune_ratio, min_nodes=min_nodes)

    if output.pruned_data is not None:
        result = output.pruned_data
        # FIX (Bug 1): Use the mask already computed inside forward() instead of
        # recomputing it here.  Recomputation is not guaranteed to produce the same
        # indices when there are ties in topk (PyTorch topk is not stable).
        # output.keep_mask is always set whenever pruned_data is not None.
        result.x = output.pruned_data.x
        result.energy = output.energy[output.keep_mask]
    else:
        result = data.clone()
        result.x = output.weighted_x
        result.energy = output.energy

    return result
