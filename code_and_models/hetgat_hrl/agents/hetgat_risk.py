from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor, nn


class RiskMaskedHetGAT(nn.Module):
    """
    Minimal hetero graph attention with exponential risk masking:
        e'_ij = e_ij - beta * exp(clamp(risk_ij, max=2.0) * 2.0)
    """

    def __init__(self, in_dim: int, hidden_dim: int = 32, beta: float = 1.0):
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.beta = float(beta)
        self.w = nn.Linear(self.in_dim, self.hidden_dim, bias=False)
        self.a = nn.Linear(2 * self.hidden_dim, 1, bias=False)
        self.act = nn.LeakyReLU(0.2)

    def forward(
        self,
        x: Tensor,  # [N,F]
        edge_index: Tensor,  # [2,E]
        edge_risk: Optional[Tensor] = None,  # [E]
    ) -> Tuple[Tensor, Tensor]:
        n = x.shape[0]
        e = edge_index.shape[1]
        h = self.w(x)  # [N,H]
        src = edge_index[0]
        dst = edge_index[1]
        pair = torch.cat([h[src], h[dst]], dim=-1)
        logits = self.act(self.a(pair).squeeze(-1))  # [E]
        if edge_risk is not None:
            risk = torch.clamp(edge_risk, max=2.0)
            logits = logits - self.beta * torch.exp(risk * 2.0)

        # Segment softmax by src node.
        alpha = torch.zeros_like(logits)
        for i in range(n):
            mask = src == i
            if torch.any(mask):
                alpha[mask] = torch.softmax(logits[mask], dim=0)

        out = torch.zeros_like(h)
        for k in range(e):
            i = int(src[k].item())
            j = int(dst[k].item())
            out[i] = out[i] + alpha[k] * h[j]
        return out, alpha

