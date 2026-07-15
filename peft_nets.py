"""
Parameter-space compression (PEFT) for GNN-PE / GNN-UCB.

The GNN reward model (nets.py) is a Linear+ReLU stack with mean pooling. Full fine-tuning
trains ~4.2M params, and the GNN-PE posterior variance uses the NTK feature g = ∇_θ f0(G) with
a diagonal U over ALL θ — so uncertainty cost and memory scale with the trainable param count.

PEFT freezes the base weights and exposes only a small trainable set S:
  * 'lora'       : low-rank adapters (A, B) on the hidden Linear layers + a trainable head.
  * 'last_layer' : only the final Linear(width, 1) head (neural-linear / linear-probe).

Since algorithms.py sizes num_net_params / U and collects gradients by `requires_grad`, shrinking
S automatically shrinks BOTH the fine-tuning cost and the NTK dimension (dim of g and U) — the
same diagonal-NTK GP posterior on a parameter subnetwork.

NTK-init note: standard LoRA sets B = 0 (so W = W0), but then ∂f/∂A = 0 at init and half of each
adapter contributes nothing to g / U. A bandit needn't start at W0, so we init both A and B
small-nonzero to give the uncertainty model the full adapter dimension.
"""

from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Frozen base `nn.Linear` + trainable low-rank update s * (x A^T) B^T.

    A has shape (r, in_features), B has shape (out_features, r), scaling s = alpha / r.
    The base weight/bias are frozen; only A and B are trainable.
    """

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        assert isinstance(base, nn.Linear)
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.rank = int(rank)
        self.scaling = float(alpha) / float(rank)

        # Freeze the pretrained/base transform.
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        dev = base.weight.device
        dtype = base.weight.dtype
        # Trainable low-rank factors. BOTH init small-nonzero.
        self.lora_A = nn.Parameter(torch.empty(self.rank, self.in_features, device=dev, dtype=dtype))
        self.lora_B = nn.Parameter(torch.empty(self.out_features, self.rank, device=dev, dtype=dtype))
        # A ~ feature-scaled normal; B ~ small normal (nonzero so ∂f/∂A ≠ 0 at init).
        nn.init.normal_(self.lora_A, mean=0.0, std=1.0 / math.sqrt(self.in_features))
        nn.init.normal_(self.lora_B, mean=0.0, std=1e-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        lora = torch.nn.functional.linear(x, self.lora_A)          # (..., r)
        lora = torch.nn.functional.linear(lora, self.lora_B)       # (..., out)
        return out + self.scaling * lora


def _linear_indices(layers: nn.ModuleList) -> List[int]:
    return [i for i, m in enumerate(layers) if isinstance(m, nn.Linear)]


def apply_peft(net: nn.Module, method: str = "none", lora_rank: int = 8, lora_alpha: float = 16.0) -> nn.Module:
    """In-place-transform `net` for parameter-efficient fine-tuning and return it.

    Assumes `net.layers` is an `nn.ModuleList` of Linear/ReLU layers (see nets.py: NN / GNN),
    where the LAST Linear is the scalar output head.

    method:
      * 'none'       : unchanged (all parameters trainable) — identical to full fine-tuning.
      * 'lora'       : wrap every hidden Linear with LoRALinear (base frozen), keep the output
                       head fully trainable. Trainable S = adapters + head.
      * 'last_layer' : freeze all Linear layers except the output head. Trainable S = head only.
    """
    method = str(method).lower()
    if method == "none":
        return net
    if not hasattr(net, "layers") or not isinstance(net.layers, nn.ModuleList):
        raise ValueError("apply_peft expects a net with an `nn.ModuleList` attribute `layers`.")

    lin_idx = _linear_indices(net.layers)
    if len(lin_idx) < 1:
        raise ValueError("apply_peft found no nn.Linear layers to adapt.")
    head_idx = lin_idx[-1]          # final Linear(width, 1) is the scalar head
    hidden_idx = lin_idx[:-1]

    if method == "last_layer":
        # Freeze everything, then unfreeze the head only.
        for p in net.parameters():
            p.requires_grad_(False)
        for p in net.layers[head_idx].parameters():
            p.requires_grad_(True)
        return net

    if method == "lora":
        # Replace hidden Linear layers with LoRA-wrapped versions (base frozen, adapters trainable).
        for i in hidden_idx:
            base = net.layers[i]
            net.layers[i] = LoRALinear(base, rank=lora_rank, alpha=lora_alpha)
        # Keep the output head fully trainable (neural-linear head; negligible params).
        for p in net.layers[head_idx].parameters():
            p.requires_grad_(True)
        return net

    raise ValueError(f"Unknown PEFT method: {method}. Use 'none', 'lora', or 'last_layer'.")


def count_trainable_params(net: nn.Module) -> int:
    return int(sum(p.numel() for p in net.parameters() if p.requires_grad))
