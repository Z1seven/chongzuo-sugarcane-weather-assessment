from __future__ import annotations

import math

import torch
from torch import nn


class PGLiteTransformer(nn.Module):
    """Phenology-guided lightweight self-attention residual regressor."""

    def __init__(
        self,
        input_dim: int = 8,
        latent_dim: int = 8,
        trend_input_dim: int = 3,
        trend_hidden_dim: int = 6,
        head_hidden_dim: int = 10,
        dropout: float = 0.2,
        stage_count: int = 4,
    ) -> None:
        super().__init__()

        self.latent_dim = latent_dim
        self.embedding = nn.Linear(input_dim, latent_dim)
        self.stage_position = nn.Parameter(
            torch.zeros(1, stage_count, latent_dim)
        )

        self.query = nn.Linear(latent_dim, latent_dim, bias=False)
        self.key = nn.Linear(latent_dim, latent_dim, bias=False)
        self.value = nn.Linear(latent_dim, latent_dim, bias=False)
        self.layer_norm = nn.LayerNorm(latent_dim)

        self.trend_branch = nn.Sequential(
            nn.Linear(trend_input_dim, trend_hidden_dim),
            nn.GELU(),
        )

        self.residual_head = nn.Sequential(
            nn.Linear(latent_dim + trend_hidden_dim, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, 1),
        )

    def forward(
        self,
        stage_sequence: torch.Tensor,
        trend_features: torch.Tensor,
    ) -> torch.Tensor:
        embedded = self.embedding(stage_sequence) + self.stage_position

        q = self.query(embedded)
        k = self.key(embedded)
        v = self.value(embedded)

        attention_logits = torch.matmul(q, k.transpose(-1, -2))
        attention_logits = attention_logits / math.sqrt(self.latent_dim)
        attention_weights = torch.softmax(attention_logits, dim=-1)

        attended = torch.matmul(attention_weights, v)
        pooled = self.layer_norm(embedded + attended).mean(dim=1)

        trend_representation = self.trend_branch(trend_features)
        fused = torch.cat([pooled, trend_representation], dim=1)
        return self.residual_head(fused).squeeze(-1)
