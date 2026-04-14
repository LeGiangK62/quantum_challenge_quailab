"""
Hierarchical LSTM architecture for PK/PD prediction.
Time-series approach for sequential PK/PD modeling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMEncoder(nn.Module):
    """LSTM encoder for sequence encoding."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # Layer norm
        self.layer_norm = nn.LayerNorm(hidden_dim * self.num_directions)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lengths=None):
        """
        Args:
            x: Input tensor [batch, seq_len, input_dim]
            lengths: Optional sequence lengths for packing

        Returns:
            outputs: LSTM outputs [batch, seq_len, hidden_dim * num_directions]
            hidden: Final hidden state
        """
        # Project input
        x = self.input_proj(x)
        x = F.relu(x)

        # Pack if lengths provided
        if lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )

        # LSTM forward
        outputs, (hidden, cell) = self.lstm(x)

        # Unpack if needed
        if lengths is not None:
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

        # Layer norm and dropout
        outputs = self.layer_norm(outputs)
        outputs = self.dropout(outputs)

        return outputs, hidden


class PKLSTMEncoder(nn.Module):
    """Stage 1: LSTM encoder for PK prediction."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        super().__init__()

        self.encoder = LSTMEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional
        )

        self.num_directions = 2 if bidirectional else 1
        output_dim = hidden_dim * self.num_directions

        # PK predictor head
        self.pk_predictor = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, lengths=None):
        """
        Args:
            x: Input features [batch, seq_len, input_dim]
            lengths: Sequence lengths

        Returns:
            embeddings: Sequence embeddings [batch, seq_len, hidden_dim]
            pk_predictions: PK predictions [batch, seq_len, 1]
        """
        embeddings, _ = self.encoder(x, lengths)
        pk_predictions = self.pk_predictor(embeddings)
        return embeddings, pk_predictions

    def enable_mc_dropout(self):
        """Enable dropout during inference for MC Dropout."""
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()


class PDLSTMDecoder(nn.Module):
    """Stage 2: LSTM decoder for PD prediction using PK predictions."""

    def __init__(
        self,
        input_dim: int,
        pk_embedding_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
        use_gating: bool = True
    ):
        super().__init__()

        self.use_gating = use_gating
        self.num_directions = 2 if bidirectional else 1

        # Combined input: original features + PK embeddings + PK predictions
        combined_dim = input_dim + pk_embedding_dim + 1

        # Gating mechanism - outputs same dim as combined for element-wise multiply
        if use_gating:
            self.gate = nn.Sequential(
                nn.Linear(combined_dim, combined_dim),
                nn.Sigmoid()
            )

        self.encoder = LSTMEncoder(
            input_dim=combined_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional
        )

        output_dim = hidden_dim * self.num_directions

        # PD predictor head
        self.pd_predictor = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

        # Residual connection from combined input
        self.residual_branch = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

        self.residual_weight = nn.Parameter(torch.tensor(0.1))

    def forward(self, x, pk_embeddings, pk_predictions, lengths=None):
        """
        Args:
            x: Original input features [batch, seq_len, input_dim]
            pk_embeddings: PK encoder embeddings [batch, seq_len, pk_embedding_dim]
            pk_predictions: PK predictions [batch, seq_len, 1]
            lengths: Sequence lengths

        Returns:
            pd_predictions: PD predictions [batch, seq_len, 1]
        """
        # Combine inputs
        combined = torch.cat([x, pk_embeddings, pk_predictions], dim=-1)

        # Apply gating
        if self.use_gating:
            gate_values = self.gate(combined)
            combined_gated = combined * gate_values
        else:
            combined_gated = combined

        # LSTM encoding
        outputs, _ = self.encoder(combined_gated, lengths)

        # Main PD prediction
        pd_main = self.pd_predictor(outputs)

        # Residual prediction
        pd_residual = self.residual_branch(combined)

        # Combine
        pd_predictions = pd_main + self.residual_weight * pd_residual

        return pd_predictions

    def enable_mc_dropout(self):
        """Enable dropout during inference for MC Dropout."""
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()


class HierarchicalPKPDLSTM(nn.Module):
    """
    Hierarchical LSTM for PK/PD prediction.

    Two-stage architecture:
    1. PK-LSTM: Predicts PK from patient time series
    2. PD-LSTM: Predicts PD using PK predictions + features

    Supports different training modes:
    - separate: Independent PK and PD encoders
    - joint: PK pred detached before feeding to PD encoder
    - dual_stage: End-to-end training (PK gradients flow to PD)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
        use_gating: bool = True,
        mode: str = 'dual_stage'
    ):
        super().__init__()

        self.mode = mode
        self.hidden_dim = hidden_dim
        self.num_directions = 2 if bidirectional else 1

        # Stage 1: PK encoder
        self.pk_encoder = PKLSTMEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional
        )

        pk_output_dim = hidden_dim * self.num_directions

        # Stage 2: PD decoder
        self.pd_decoder = PDLSTMDecoder(
            input_dim=input_dim,
            pk_embedding_dim=pk_output_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            use_gating=use_gating
        )

    def forward(self, x_pk=None, x_pd=None, lengths_pk=None, lengths_pd=None, return_pk=False):
        """
        Forward pass.

        Args:
            x_pk: PK input features [batch, seq_len, input_dim]
            x_pd: PD input features [batch, seq_len, input_dim]
            lengths_pk: PK sequence lengths
            lengths_pd: PD sequence lengths
            return_pk: Whether to return PK predictions

        Returns:
            dict with 'pk' and/or 'pd' predictions, or just pd_predictions if using GNN-style interface
        """
        results = {}

        # Stage 1: PK prediction
        if x_pk is not None:
            pk_embeddings, pk_predictions = self.pk_encoder(x_pk, lengths_pk)
            results['pk'] = pk_predictions
            results['pk_embeddings'] = pk_embeddings

        # Stage 2: PD prediction
        if x_pd is not None:
            batch_size, pd_seq_len, _ = x_pd.shape
            device = x_pd.device

            if 'pk_embeddings' in results:
                pk_emb = results['pk_embeddings']   # [batch, seq_pk, hidden]
                pk_pred = results['pk']             # [batch, seq_pk, 1]

                if self.mode == 'joint':
                    pk_emb = pk_emb.detach()
                    pk_pred = pk_pred.detach()

                # Use last valid PK hidden state as patient-level PK summary.
                # Broadcast to all PD timepoints.
                # This avoids spurious interpolation in embedding space and
                # correctly represents "cumulative PK history up to last observation".
                if lengths_pk is not None:
                    # Gather last valid timestep per patient
                    last_idx = (lengths_pk - 1).clamp(min=0)  # [batch]
                    pk_emb_last = pk_emb[torch.arange(batch_size, device=device), last_idx]   # [batch, hidden]
                    pk_pred_last = pk_pred[torch.arange(batch_size, device=device), last_idx] # [batch, 1]
                else:
                    pk_emb_last = pk_emb[:, -1, :]   # [batch, hidden]
                    pk_pred_last = pk_pred[:, -1, :]  # [batch, 1]

                # Expand to [batch, pd_seq_len, *]
                pk_emb  = pk_emb_last.unsqueeze(1).expand(-1, pd_seq_len, -1)
                pk_pred = pk_pred_last.unsqueeze(1).expand(-1, pd_seq_len, -1)
            else:
                # No PK available (placebo) - use zeros
                pk_emb = torch.zeros(batch_size, pd_seq_len, self.hidden_dim * self.num_directions, device=device)
                pk_pred = torch.zeros(batch_size, pd_seq_len, 1, device=device)

            pd_predictions = self.pd_decoder(x_pd, pk_emb, pk_pred, lengths_pd)
            results['pd'] = pd_predictions

        # Clean up intermediate results
        if 'pk_embeddings' in results:
            del results['pk_embeddings']

        if return_pk and 'pk' in results:
            return results['pd'], results['pk']

        return results

    def enable_mc_dropout(self):
        """Enable MC Dropout for uncertainty estimation."""
        self.pk_encoder.enable_mc_dropout()
        self.pd_decoder.enable_mc_dropout()


class SimpleLSTM(nn.Module):
    """
    Simple LSTM for direct PK or PD prediction (non-hierarchical).
    Useful as a baseline or for single-task prediction.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        super().__init__()

        self.encoder = LSTMEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional
        )

        num_directions = 2 if bidirectional else 1
        output_dim = hidden_dim * num_directions

        self.predictor = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, lengths=None):
        """
        Args:
            x: Input features [batch, seq_len, input_dim]
            lengths: Sequence lengths

        Returns:
            predictions: Predictions [batch, seq_len, 1]
        """
        outputs, _ = self.encoder(x, lengths)
        predictions = self.predictor(outputs)
        return predictions
