"""
Hierarchical GNN architecture for PK/PD prediction.
Graph-based approach leveraging temporal relationships.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn import LayerNorm


class PKGNNEncoder(nn.Module):
    """GNN encoder for PK prediction from patient graph."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 3,
                 dropout: float = 0.2, use_attention: bool = False):
        super().__init__()

        self.use_attention = use_attention
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # First layer
        if use_attention:
            self.convs.append(GATConv(input_dim, hidden_dim, heads=4, concat=False))
        else:
            self.convs.append(GCNConv(input_dim, hidden_dim))
        self.norms.append(LayerNorm(hidden_dim))

        # Hidden layers
        for _ in range(num_layers - 1):
            if use_attention:
                self.convs.append(GATConv(hidden_dim, hidden_dim, heads=4, concat=False))
            else:
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.norms.append(LayerNorm(hidden_dim))

        self.dropout = dropout

        # PK predictor head
        self.pk_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x, edge_index, edge_weight=None):
        """
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            edge_weight: Optional edge weights

        Returns:
            embeddings: Node embeddings [num_nodes, hidden_dim]
            pk_predictions: PK predictions [num_nodes, 1]
        """
        h = x
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            h_new = conv(h, edge_index, edge_weight=edge_weight)
            h_new = norm(h_new)
            h_new = torch.relu(h_new)

            if i < len(self.convs) - 1:
                h_new = torch.dropout(h_new, p=self.dropout, train=self.training)

            # Residual connection
            if i > 0 and h.size(-1) == h_new.size(-1):
                h = h + h_new
            else:
                h = h_new

        embeddings = h
        pk_predictions = self.pk_predictor(embeddings)

        return embeddings, pk_predictions

    def enable_mc_dropout(self):
        """Enable dropout during inference for MC Dropout."""
        for m in self.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()


class PDGNNDecoder(nn.Module):
    """GNN decoder for PD prediction using PK predictions + covariates."""

    def __init__(self, pk_embedding_dim: int, input_dim: int, hidden_dim: int = 64,
                 num_layers: int = 3, dropout: float = 0.2, use_attention: bool = False,
                 use_gating: bool = True):
        super().__init__()

        self.use_attention = use_attention
        self.use_gating = use_gating

        # Combine PK embeddings with input features
        combined_dim = pk_embedding_dim + input_dim + 1  # +1 for predicted PK value

        # Gating mechanism
        if use_gating:
            self.gate = nn.Sequential(
                nn.Linear(combined_dim, hidden_dim),
                nn.Sigmoid()
            )

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # First layer
        if use_attention:
            self.convs.append(GATConv(combined_dim, hidden_dim, heads=4, concat=False))
        else:
            self.convs.append(GCNConv(combined_dim, hidden_dim))
        self.norms.append(LayerNorm(hidden_dim))

        # Hidden layers
        for _ in range(num_layers - 1):
            if use_attention:
                self.convs.append(GATConv(hidden_dim, hidden_dim, heads=4, concat=False))
            else:
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.norms.append(LayerNorm(hidden_dim))

        self.dropout = dropout

        # PD predictor head
        self.pd_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x, pk_embeddings, pk_predictions, edge_index, edge_weight=None):
        """
        Args:
            x: Original node features [num_nodes, input_dim]
            pk_embeddings: PK encoder embeddings [num_nodes, pk_embedding_dim]
            pk_predictions: PK predictions [num_nodes, 1]
            edge_index: Edge indices [2, num_edges]
            edge_weight: Optional edge weights

        Returns:
            pd_predictions: PD predictions [num_nodes, 1]
        """
        # Concatenate PK info with original features
        h = torch.cat([x, pk_embeddings, pk_predictions], dim=-1)

        # Apply gating
        if self.use_gating:
            gate_values = self.gate(h)

        # GNN layers
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            h_new = conv(h, edge_index, edge_weight=edge_weight)
            h_new = norm(h_new)
            h_new = torch.relu(h_new)

            if i < len(self.convs) - 1:
                h_new = torch.dropout(h_new, p=self.dropout, train=self.training)

            # Apply gating
            if self.use_gating and i == 0:
                h_new = h_new * gate_values

            # Residual connection
            if i > 0 and h.size(-1) == h_new.size(-1):
                h = h + h_new
            else:
                h = h_new

        pd_predictions = self.pd_predictor(h)
        return pd_predictions

    def enable_mc_dropout(self):
        """Enable dropout during inference for MC Dropout."""
        for m in self.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()


class HierarchicalPKPDGNN(nn.Module):
    """
    Hierarchical GNN for PK/PD prediction.

    Two-stage architecture:
    1. PK-GNN: Predicts PK from patient graph
    2. PD-GNN: Predicts PD using PK predictions + graph structure
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 64,
        num_layers_pk: int = 3,
        num_layers_pd: int = 3,
        dropout: float = 0.2,
        use_attention: bool = False,
        use_gating: bool = True
    ):
        super().__init__()

        self.pk_encoder = PKGNNEncoder(
            input_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers_pk,
            dropout=dropout,
            use_attention=use_attention
        )

        self.pd_decoder = PDGNNDecoder(
            pk_embedding_dim=hidden_dim,
            input_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers_pd,
            dropout=dropout,
            use_attention=use_attention,
            use_gating=use_gating
        )

    def forward(self, batch, return_pk=False):
        """
        Forward pass through hierarchical GNN.

        Args:
            batch: PyG batch object with:
                - x: Node features
                - edge_index: Edge connectivity
                - edge_weight: Edge weights (optional)
                - pk_mask: Boolean mask for PK nodes
                - pd_mask: Boolean mask for PD nodes
            return_pk: Whether to return PK predictions

        Returns:
            pd_predictions: PD predictions for PD nodes
            pk_predictions (optional): PK predictions for PK nodes
        """
        x = batch.x
        edge_index = batch.edge_index
        edge_weight = batch.edge_weight if hasattr(batch, 'edge_weight') else None

        # Stage 1: PK prediction
        pk_embeddings, pk_predictions = self.pk_encoder(x, edge_index, edge_weight)

        # Stage 2: PD prediction
        pd_predictions = self.pd_decoder(
            x, pk_embeddings, pk_predictions,
            edge_index, edge_weight
        )

        if return_pk:
            return pd_predictions, pk_predictions
        return pd_predictions

    def enable_mc_dropout(self):
        """Enable MC Dropout for uncertainty estimation."""
        self.pk_encoder.enable_mc_dropout()
        self.pd_decoder.enable_mc_dropout()
