from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor, nn


class TaskAttentionModule(nn.Module):
    """
    Dynamic task attention:
    task feature = [norm_dx, norm_dy, norm_dist, emergency_flag, is_recommended]
    """

    def __init__(self, agent_dim: int, task_dim: int = 5, hidden_dim: int = 64):
        super().__init__()
        self.agent_proj = nn.Linear(agent_dim, hidden_dim)
        self.task_proj = nn.Linear(task_dim, hidden_dim)
        self.scale = hidden_dim ** 0.5

    def forward(
        self,
        agent_feat: Tensor,  # [B, A, F]
        task_feat: Tensor,  # [B, T, D_task]
        task_mask: Optional[Tensor] = None,  # [B, T], 1=valid
    ) -> Tuple[Tensor, Tensor]:
        q = self.agent_proj(agent_feat)  # [B, A, H]
        k = self.task_proj(task_feat)  # [B, T, H]
        logits = torch.einsum("bah,bth->bat", q, k) / self.scale
        if task_mask is not None:
            mask = task_mask.unsqueeze(1).expand_as(logits)
            logits = logits.masked_fill(mask <= 0, -1e9)
        weights = torch.softmax(logits, dim=-1)  # [B, A, T]
        focal = torch.einsum("bat,btd->bad", weights, task_feat)  # [B, A, D_task]
        return focal, weights
