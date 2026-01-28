"""
Hierarchical MLP architecture for PK/PD prediction.
Based on the successful old code with ResidualMLP blocks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Residual block with LayerNorm (from old successful code)."""

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def forward(self, x):
        """Residual forward: x + MLP(LN(x))"""
        h = self.ln(x)
        h = self.fc2(self.dropout(self.act(self.fc1(h))))
        return x + self.dropout(h)


class ResidualMLPEncoder(nn.Module):
    """Residual MLP encoder with configurable depth."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, n_blocks: int = 4, dropout: float = 0.3):
        super().__init__()

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Residual blocks
        self.blocks = nn.ModuleList([
            ResBlock(hidden_dim, dropout) for _ in range(n_blocks)
        ])

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Encode input to hidden representation."""
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        return h

    def enable_mc_dropout(self):
        """Enable dropout during inference for MC Dropout."""
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()


class HierarchicalPKPDMLP(nn.Module):
    """
    Hierarchical MLP for PK/PD prediction.

    Architecture modes:
    - separate: Independent PK and PD encoders
    - joint: PK pred detached before feeding to PD encoder
    - dual_stage: End-to-end training (PK gradients flow to PD)
    - shared: Single shared encoder for both PK and PD
    """

    def __init__(
        self,
        mode: str,
        pk_input_dim: int,
        pd_input_dim: int,
        hidden_dim: int = 256,
        n_blocks: int = 4,
        dropout: float = 0.3,
        head_hidden: int = 128
    ):
        super().__init__()
        self.mode = mode
        self.hidden_dim = hidden_dim

        if mode in ['separate', 'joint', 'dual_stage']:
            # Separate encoders for PK and PD
            self.pk_encoder = ResidualMLPEncoder(pk_input_dim, hidden_dim, n_blocks, dropout)
            self.pd_encoder = ResidualMLPEncoder(pd_input_dim + 1, hidden_dim, n_blocks, dropout)  # +1 for PK pred

        elif mode == 'shared':
            # Shared encoder
            self.encoder = ResidualMLPEncoder(max(pk_input_dim, pd_input_dim), hidden_dim, n_blocks, dropout)
            if pk_input_dim != pd_input_dim:
                self.pk_proj = nn.Linear(pk_input_dim, max(pk_input_dim, pd_input_dim))
                self.pd_proj = nn.Linear(pd_input_dim, max(pk_input_dim, pd_input_dim))
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Prediction heads
        self.pk_head = nn.Sequential(
            nn.Linear(hidden_dim, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1)
        )

        self.pd_head = nn.Sequential(
            nn.Linear(hidden_dim, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1)
        )

    def forward(self, x_pk=None, x_pd=None):
        """
        Forward pass supporting different modes.

        Args:
            x_pk: PK input features [batch, pk_features]
            x_pd: PD input features [batch, pd_features]

        Returns:
            dict with 'pk' and/or 'pd' predictions
        """
        if self.mode == 'separate':
            return self._forward_separate(x_pk, x_pd)
        elif self.mode == 'joint':
            return self._forward_joint(x_pk, x_pd)
        elif self.mode == 'dual_stage':
            return self._forward_dual_stage(x_pk, x_pd)
        elif self.mode == 'shared':
            return self._forward_shared(x_pk, x_pd)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _forward_separate(self, x_pk, x_pd):
        """Separate forward (independent PK and PD)."""
        results = {}

        if x_pk is not None:
            z_pk = self.pk_encoder(x_pk)
            pk_pred = self.pk_head(z_pk)
            results['pk'] = pk_pred

        if x_pd is not None:
            # For separate mode, PD encoder doesn't use PK prediction
            # Pad with zeros for the PK prediction dimension
            x_pd_padded = torch.cat([x_pd, torch.zeros(x_pd.size(0), 1, device=x_pd.device)], dim=1)
            z_pd = self.pd_encoder(x_pd_padded)
            pd_pred = self.pd_head(z_pd)
            results['pd'] = pd_pred

        return results

    def _forward_joint(self, x_pk, x_pd):
        """Joint forward (PK pred detached before feeding to PD)."""
        results = {}

        if x_pk is not None:
            z_pk = self.pk_encoder(x_pk)
            pk_pred = self.pk_head(z_pk)
            results['pk'] = pk_pred

        if x_pd is not None:
            # Use detached PK prediction
            pk_pred = results['pk'].detach() if 'pk' in results else torch.zeros(x_pd.size(0), 1, device=x_pd.device)
            x_pd_with_pk = torch.cat([x_pd, pk_pred], dim=1)
            z_pd = self.pd_encoder(x_pd_with_pk)
            pd_pred = self.pd_head(z_pd)
            results['pd'] = pd_pred

        return results

    def _forward_dual_stage(self, x_pk, x_pd):
        """Dual-stage forward (end-to-end, no detach)."""
        results = {}

        if x_pk is not None:
            z_pk = self.pk_encoder(x_pk)
            pk_pred = self.pk_head(z_pk)
            results['pk'] = pk_pred

        if x_pd is not None:
            # Use PK prediction (gradients flow)
            pk_pred = results.get('pk', torch.zeros(x_pd.size(0), 1, device=x_pd.device))
            x_pd_with_pk = torch.cat([x_pd, pk_pred], dim=1)
            z_pd = self.pd_encoder(x_pd_with_pk)
            pd_pred = self.pd_head(z_pd)
            results['pd'] = pd_pred

        return results

    def _forward_shared(self, x_pk, x_pd):
        """Shared encoder forward."""
        results = {}

        if x_pk is not None:
            x = self.pk_proj(x_pk) if hasattr(self, 'pk_proj') else x_pk
            z = self.encoder(x)
            pk_pred = self.pk_head(z)
            results['pk'] = pk_pred

        if x_pd is not None:
            x = self.pd_proj(x_pd) if hasattr(self, 'pd_proj') else x_pd
            z = self.encoder(x)
            pd_pred = self.pd_head(z)
            results['pd'] = pd_pred

        return results

    def enable_mc_dropout(self):
        """Enable MC Dropout for uncertainty estimation."""
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

        # Also enable for encoders
        if hasattr(self, 'pk_encoder'):
            self.pk_encoder.enable_mc_dropout()
        if hasattr(self, 'pd_encoder'):
            self.pd_encoder.enable_mc_dropout()
        if hasattr(self, 'encoder'):
            self.encoder.enable_mc_dropout()
