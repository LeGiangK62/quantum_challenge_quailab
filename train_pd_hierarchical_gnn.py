#!/usr/bin/env python3
"""
Hierarchical GNN for PK-PD Prediction with Residual Connections.

Architecture:
1. Stage 1: PK-GNN predicts PK values from covariates
2. Stage 2: PD-GNN predicts PD values using predicted PK + covariates
3. Residual connections for better gradient flow

Training Modes:
- Sequential: Train PK-GNN first, then train PD-GNN with frozen PK-GNN
- Joint: Train both stages end-to-end with weighted multi-task loss

Patient Selection Features:
- Evaluate specific patients: --eval_patient_ids 1 5 10
- Plot specific patients: --patient_ids 1 5 10
- Random patient selection: --random_patients --n_patients 5
- Default: First N patients

Example Usage:
  # Joint training with specific patient evaluation
  python train_pd_hierarchical_gnn.py --training_mode joint --eval_patient_ids 1 5 10

  # Sequential training, plot specific patients
  python train_pd_hierarchical_gnn.py --training_mode sequential --patient_ids 2 7 15

  # Joint training with random patients
  python train_pd_hierarchical_gnn.py --training_mode joint --random_patients --n_patients 5

  # With attention and custom hyperparameters
  python train_pd_hierarchical_gnn.py --use_attention --hidden_dim 128 --epochs 200
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, LayerNorm

# Set style
sns.set_style('whitegrid')
plt.rcParams['font.size'] = 10


class PKGNNEncoder(nn.Module):
    """Stage 1: GNN for PK prediction from covariates."""

    def __init__(self, input_dim, hidden_dim=64, num_layers=3, dropout=0.2, use_attention=False):
        super(PKGNNEncoder, self).__init__()

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
        # GNN layers with residual connections
        h = x
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            h_new = conv(h, edge_index, edge_weight=edge_weight)
            h_new = norm(h_new)
            h_new = torch.relu(h_new)

            if i < len(self.convs) - 1:
                h_new = torch.dropout(h_new, p=self.dropout, train=self.training)

            # Residual connection (except first layer)
            if i > 0 and h.size(-1) == h_new.size(-1):
                h = h + h_new
            else:
                h = h_new

        embeddings = h
        pk_predictions = self.pk_predictor(embeddings)

        return embeddings, pk_predictions


class PDGNNDecoder(nn.Module):
    """Stage 2: GNN for PD prediction using PK predictions + covariates."""

    def __init__(self, pk_embedding_dim, input_dim, hidden_dim=64, num_layers=3,
                 dropout=0.2, use_attention=False, use_gating=True):
        super(PDGNNDecoder, self).__init__()

        self.use_attention = use_attention
        self.use_gating = use_gating

        # Combine PK embeddings with input features
        combined_dim = pk_embedding_dim + input_dim + 1  # +1 for predicted PK value

        # Gating mechanism to control PK information flow
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

        # Residual branch - learns additional corrections
        self.residual_branch = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Learnable residual weight
        self.residual_weight = nn.Parameter(torch.tensor(0.1))

    def forward(self, x, pk_embeddings, pk_predictions, edge_index, edge_weight=None):
        """
        Args:
            x: Original node features [num_nodes, input_dim]
            pk_embeddings: PK embeddings from Stage 1 [num_nodes, pk_embedding_dim]
            pk_predictions: Predicted PK values [num_nodes, 1]
            edge_index: Edge indices [2, num_edges]
            edge_weight: Optional edge weights

        Returns:
            pd_predictions: PD predictions [num_nodes, 1]
        """
        # Combine all information
        combined = torch.cat([x, pk_embeddings, pk_predictions], dim=-1)

        # Apply gating if enabled
        if self.use_gating:
            gate_values = self.gate(combined)

        # GNN layers with residual connections
        h = combined
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            h_new = conv(h, edge_index, edge_weight=edge_weight)
            h_new = norm(h_new)
            h_new = torch.relu(h_new)

            # Apply gating to first layer
            if i == 0 and self.use_gating:
                h_new = h_new * gate_values

            if i < len(self.convs) - 1:
                h_new = torch.dropout(h_new, p=self.dropout, train=self.training)

            # Residual connection
            if i > 0 and h.size(-1) == h_new.size(-1):
                h = h + h_new
            else:
                h = h_new

        # Main PD prediction
        pd_main = self.pd_predictor(h)

        # Residual correction
        pd_residual = self.residual_branch(combined)

        # Final prediction with learnable residual weight
        pd_predictions = pd_main + self.residual_weight * pd_residual

        return pd_predictions


class HierarchicalPKPDGNN(nn.Module):
    """Hierarchical GNN: PK-GNN -> PD-GNN with residual connections."""

    def __init__(self, input_dim, hidden_dim=64, num_layers_pk=3, num_layers_pd=3,
                 dropout=0.2, use_attention=False, use_gating=True):
        super(HierarchicalPKPDGNN, self).__init__()

        self.pk_encoder = PKGNNEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers_pk,
            dropout=dropout,
            use_attention=use_attention
        )

        self.pd_decoder = PDGNNDecoder(
            pk_embedding_dim=hidden_dim,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers_pd,
            dropout=dropout,
            use_attention=use_attention,
            use_gating=use_gating
        )

    def forward(self, data, return_pk=False):
        """
        Args:
            data: PyG Data object with x, edge_index, edge_weight (optional)
            return_pk: Whether to return PK predictions

        Returns:
            pd_predictions: PD predictions [num_nodes, 1]
            pk_predictions: PK predictions [num_nodes, 1] (if return_pk=True)
        """
        x, edge_index = data.x, data.edge_index
        edge_weight = data.edge_weight if hasattr(data, 'edge_weight') else None

        # Stage 1: PK prediction
        pk_embeddings, pk_predictions = self.pk_encoder(x, edge_index, edge_weight)

        # Stage 2: PD prediction
        pd_predictions = self.pd_decoder(x, pk_embeddings, pk_predictions, edge_index, edge_weight)

        if return_pk:
            return pd_predictions, pk_predictions
        return pd_predictions

    def freeze_pk_encoder(self):
        """Freeze PK encoder for sequential training."""
        for param in self.pk_encoder.parameters():
            param.requires_grad = False

    def unfreeze_pk_encoder(self):
        """Unfreeze PK encoder for joint training."""
        for param in self.pk_encoder.parameters():
            param.requires_grad = True


def engineer_advanced_features(patient_data):
    """
    Enhanced feature engineering for PK/PD prediction.

    Features added:
    - Time transformations (log, sqrt, squared)
    - Rate of change features (ΔPK/Δt, ΔTime)
    - Cumulative dose over time
    - Dose-time interactions
    - Temporal patterns (sin/cos)
    - Dose per kg normalization
    - PK lag features
    """
    patient_data = patient_data.copy()

    # 1. Time transformations
    patient_data['TIME_log'] = np.log1p(patient_data['TIME'])
    patient_data['TIME_sqrt'] = np.sqrt(patient_data['TIME'])
    patient_data['TIME_squared'] = patient_data['TIME'] ** 2

    # 2. Cumulative dose
    if 'DOSE' in patient_data.columns:
        patient_data['CUMULATIVE_DOSE'] = patient_data['DOSE'].cumsum()
        patient_data['DOSE_per_kg'] = patient_data['DOSE'] / (patient_data['BW'] + 1e-8)

    # 3. Rate of change features (PK derivatives)
    pk_obs = patient_data[patient_data['DVID'] == 1].copy()
    if len(pk_obs) > 1:
        pk_obs['PK_rate'] = pk_obs['DV'].diff() / pk_obs['TIME'].diff()
        pk_obs['PK_rate'] = pk_obs['PK_rate'].fillna(0)
        patient_data = patient_data.merge(pk_obs[['TIME', 'PK_rate']], on='TIME', how='left')
        patient_data['PK_rate'] = patient_data['PK_rate'].fillna(0)
    else:
        patient_data['PK_rate'] = 0

    # 4. Time since last observation
    patient_data['TIME_diff'] = patient_data['TIME'].diff().fillna(0)

    # 5. Temporal patterns (circadian rhythm)
    patient_data['TIME_sin'] = np.sin(2 * np.pi * patient_data['TIME'] / 24)
    patient_data['TIME_cos'] = np.cos(2 * np.pi * patient_data['TIME'] / 24)

    # 6. Interaction features
    patient_data['TIME_x_DOSE'] = patient_data['TIME'] * patient_data.get('DOSE', 0)
    patient_data['BW_x_DOSE'] = patient_data['BW'] * patient_data.get('DOSE', 0)
    patient_data['COMED_x_TIME'] = patient_data['COMED'] * patient_data['TIME']

    return patient_data


def create_patient_graph_hierarchical(patient_data, feature_engineering=True):
    """
    Create a graph for one patient with enhanced features and edge weights.

    Returns:
        node_features: (num_nodes, feature_dim)
        edge_index: (2, num_edges)
        edge_weights: (num_edges,) - temporal distance-based weights
        node_types: (num_nodes,) - 0 for PK, 1 for PD
        pk_targets: PK values for PK nodes
        pd_targets: PD values for PD nodes
        pk_node_indices: Indices of PK nodes
        pd_node_indices: Indices of PD nodes
        times: Time values for each node
    """
    patient_data = patient_data.sort_values('TIME').reset_index(drop=True)

    # Apply feature engineering
    if feature_engineering:
        patient_data = engineer_advanced_features(patient_data)

    node_features = []
    node_types = []
    pk_targets = []
    pd_targets = []
    times = []
    edges = []
    edge_weights = []

    # Separate PK and PD observations
    pk_obs = patient_data[patient_data['DVID'] == 1].reset_index(drop=True)
    pd_obs = patient_data[patient_data['DVID'] == 2].reset_index(drop=True)

    node_idx = 0
    pk_node_map = {}
    pd_node_map = {}

    # Base features
    base_features = ['TIME', 'BW', 'DOSE', 'COMED']

    # Engineered features
    if feature_engineering:
        engineered_features = [
            'TIME_log', 'TIME_sqrt', 'TIME_squared',
            'CUMULATIVE_DOSE', 'DOSE_per_kg', 'PK_rate', 'TIME_diff',
            'TIME_sin', 'TIME_cos',
            'TIME_x_DOSE', 'BW_x_DOSE', 'COMED_x_TIME'
        ]
    else:
        engineered_features = []

    # Add PK nodes
    pk_node_indices = []
    for idx, row in pk_obs.iterrows():
        features = [row[f] for f in base_features]

        if feature_engineering:
            features.extend([row[f] for f in engineered_features])

        # Add PK value as input feature
        features.append(row['DV'])

        node_features.append(features)
        node_types.append(0)  # PK node
        pk_targets.append(row['DV'])
        pd_targets.append(0)  # Not predicting PD
        times.append(row['TIME'])
        pk_node_map[row['TIME']] = node_idx
        pk_node_indices.append(node_idx)
        node_idx += 1

    # Add PD nodes
    pd_node_indices = []
    for idx, row in pd_obs.iterrows():
        # Find most recent PK value and multiple lagged PK values
        recent_pk = pk_obs[pk_obs['TIME'] <= row['TIME']]

        if len(recent_pk) > 0:
            pk_value = recent_pk.iloc[-1]['DV']
            # PK-PD lag: difference between current time and last PK time
            pk_pd_lag = row['TIME'] - recent_pk.iloc[-1]['TIME']
        else:
            pk_value = 0.0
            pk_pd_lag = row['TIME']

        features = [row[f] for f in base_features]

        if feature_engineering:
            features.extend([row[f] for f in engineered_features])

        # Add PK value and lag as features
        features.append(pk_value)

        node_features.append(features)
        node_types.append(1)  # PD node
        pk_targets.append(0)  # Not predicting PK
        pd_targets.append(row['DV'])
        times.append(row['TIME'])
        pd_node_map[row['TIME']] = node_idx
        pd_node_indices.append(node_idx)
        node_idx += 1

    times_array = np.array(times)

    # Create edges with temporal-based weights
    # 1. Temporal edges within PK nodes
    pk_nodes = [i for i, t in enumerate(node_types) if t == 0]
    for i in range(len(pk_nodes) - 1):
        time_diff = abs(times_array[pk_nodes[i+1]] - times_array[pk_nodes[i]])
        weight = np.exp(-time_diff / 24.0)  # Decay with time (24h half-life)

        edges.append([pk_nodes[i], pk_nodes[i+1]])
        edge_weights.append(weight)
        edges.append([pk_nodes[i+1], pk_nodes[i]])
        edge_weights.append(weight)

    # 2. Temporal edges within PD nodes
    pd_nodes = [i for i, t in enumerate(node_types) if t == 1]
    for i in range(len(pd_nodes) - 1):
        time_diff = abs(times_array[pd_nodes[i+1]] - times_array[pd_nodes[i]])
        weight = np.exp(-time_diff / 24.0)

        edges.append([pd_nodes[i], pd_nodes[i+1]])
        edge_weights.append(weight)
        edges.append([pd_nodes[i+1], pd_nodes[i]])
        edge_weights.append(weight)

    # 3. PK-PD interaction edges (multi-hop: connect PD to recent PK values)
    for pd_idx in pd_nodes:
        pd_time = times_array[pd_idx]

        # Connect to multiple recent PK nodes (up to 3)
        connected_count = 0
        for pk_idx in reversed(pk_nodes):
            if times_array[pk_idx] <= pd_time and connected_count < 3:
                time_diff = abs(pd_time - times_array[pk_idx])
                weight = np.exp(-time_diff / 12.0)  # Stronger decay for PK-PD

                edges.append([pk_idx, pd_idx])
                edge_weights.append(weight)
                edges.append([pd_idx, pk_idx])
                edge_weights.append(weight)

                connected_count += 1

    node_features = np.array(node_features, dtype=np.float32)
    edge_index = np.array(edges, dtype=np.int64).T if len(edges) > 0 else np.array([[], []], dtype=np.int64)
    edge_weights = np.array(edge_weights, dtype=np.float32) if len(edge_weights) > 0 else np.array([], dtype=np.float32)

    return (node_features, edge_index, edge_weights, np.array(node_types),
            np.array(pk_targets), np.array(pd_targets),
            pk_node_indices, pd_node_indices, times_array)


def prepare_hierarchical_gnn_data(csv_path='Data/QIC2025-EstDat.csv',
                                   feature_engineering=True,
                                   test_size=0.2,
                                   random_seed=1712):
    """Prepare graph data for hierarchical GNN."""
    print("Loading data...")
    df = pd.read_csv(csv_path)

    # Filter to observations only
    df = df[df['EVID'] == 0].copy()
    if 'MDV' in df.columns:
        df = df[df['MDV'] == 0]

    print(f"Total observations: {len(df)}")

    if 'DVID' not in df.columns:
        raise ValueError("DVID column required")

    # Create graphs for each patient
    print("\nCreating patient graphs with enhanced features...")
    graphs = []
    all_node_features = []

    scaler_X = StandardScaler()

    for patient_id in df['ID'].unique():
        patient_data = df[df['ID'] == patient_id]

        (node_features, edge_index, edge_weights, node_types,
         pk_targets, pd_targets, pk_node_indices, pd_node_indices, times) = \
            create_patient_graph_hierarchical(patient_data, feature_engineering)

        if len(pd_node_indices) == 0 or len(pk_node_indices) == 0:
            continue  # Skip patients without both PK and PD

        all_node_features.append(node_features)

        graphs.append({
            'patient_id': patient_id,
            'node_features': node_features,
            'edge_index': edge_index,
            'edge_weights': edge_weights,
            'node_types': node_types,
            'pk_targets': pk_targets,
            'pd_targets': pd_targets,
            'pk_node_indices': pk_node_indices,
            'pd_node_indices': pd_node_indices,
            'times': times
        })

    print(f"Created {len(graphs)} patient graphs")

    # Fit scaler on all node features
    all_features_combined = np.vstack(all_node_features)
    scaler_X.fit(all_features_combined)

    # Scale node features
    for graph in graphs:
        graph['node_features'] = scaler_X.transform(graph['node_features'])

    # Split patients into train/test
    from sklearn.model_selection import train_test_split
    patient_indices = list(range(len(graphs)))
    train_indices, test_indices = train_test_split(patient_indices, test_size=test_size, random_state=random_seed)

    train_graphs = [graphs[i] for i in train_indices]
    test_graphs = [graphs[i] for i in test_indices]

    print(f"\n=== Data Split ===")
    print(f"Train patients: {len(train_graphs)}")
    print(f"Test patients: {len(test_graphs)}")

    # Calculate total predictions
    train_pk_count = sum(len(g['pk_node_indices']) for g in train_graphs)
    train_pd_count = sum(len(g['pd_node_indices']) for g in train_graphs)
    test_pk_count = sum(len(g['pk_node_indices']) for g in test_graphs)
    test_pd_count = sum(len(g['pd_node_indices']) for g in test_graphs)

    print(f"Train PK/PD predictions: {train_pk_count}/{train_pd_count}")
    print(f"Test PK/PD predictions: {test_pk_count}/{test_pd_count}")

    return {
        'train_graphs': train_graphs,
        'test_graphs': test_graphs,
        'scaler_X': scaler_X,
        'feature_dim': all_features_combined.shape[1]
    }


def create_pyg_data_list(graphs):
    """Convert graph dictionaries to PyG Data objects."""
    data_list = []

    for graph in graphs:
        x = torch.FloatTensor(graph['node_features'])
        edge_index = torch.LongTensor(graph['edge_index'])
        edge_weight = torch.FloatTensor(graph['edge_weights'])

        pk_targets = torch.FloatTensor(graph['pk_targets'])
        pd_targets = torch.FloatTensor(graph['pd_targets'])

        # Masks for PK and PD nodes
        pk_mask = torch.zeros(len(graph['node_types']), dtype=torch.bool)
        pk_mask[graph['pk_node_indices']] = True

        pd_mask = torch.zeros(len(graph['node_types']), dtype=torch.bool)
        pd_mask[graph['pd_node_indices']] = True

        times_tensor = torch.FloatTensor(graph['times'])

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_weight=edge_weight,
            pk_targets=pk_targets,
            pd_targets=pd_targets,
            pk_mask=pk_mask,
            pd_mask=pd_mask,
            patient_id=graph['patient_id'],
            times=times_tensor
        )

        data_list.append(data)

    return data_list


def train_sequential(train_graphs, val_graphs, feature_dim,
                     hidden_dim=64,
                     num_layers_pk=3,
                     num_layers_pd=3,
                     dropout=0.2,
                     use_attention=False,
                     use_gating=True,
                     learning_rate_pk=0.001,
                     learning_rate_pd=0.001,
                     epochs_pk=100,
                     epochs_pd=100,
                     batch_size=8,
                     device='cpu'):
    """
    Sequential training: Train PK-GNN first, freeze it, then train PD-GNN.
    """
    print("\n" + "="*60)
    print("SEQUENTIAL TRAINING MODE")
    print("="*60)

    train_data_list = create_pyg_data_list(train_graphs)
    val_data_list = create_pyg_data_list(val_graphs)

    train_loader = DataLoader(train_data_list, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data_list, batch_size=batch_size, shuffle=False)

    model = HierarchicalPKPDGNN(
        input_dim=feature_dim,
        hidden_dim=hidden_dim,
        num_layers_pk=num_layers_pk,
        num_layers_pd=num_layers_pd,
        dropout=dropout,
        use_attention=use_attention,
        use_gating=use_gating
    ).to(device)

    criterion = nn.MSELoss()

    # ==================== STAGE 1: Train PK-GNN ====================
    print("\n" + "="*60)
    print("STAGE 1: Training PK-GNN")
    print("="*60)

    optimizer_pk = optim.Adam(model.pk_encoder.parameters(), lr=learning_rate_pk)

    train_losses_pk = []
    val_losses_pk = []

    for epoch in range(epochs_pk):
        model.train()
        epoch_loss = 0
        total_pk_nodes = 0

        for batch in train_loader:
            batch = batch.to(device)

            # Forward pass (only PK encoder)
            _, pk_predictions = model.pk_encoder(
                batch.x, batch.edge_index,
                batch.edge_weight if hasattr(batch, 'edge_weight') else None
            )

            # Only compute loss on PK nodes
            pk_preds = pk_predictions[batch.pk_mask]
            pk_tgts = batch.pk_targets[batch.pk_mask].reshape(-1, 1)

            loss = criterion(pk_preds, pk_tgts)

            # Backward pass
            optimizer_pk.zero_grad()
            loss.backward()
            optimizer_pk.step()

            epoch_loss += loss.item() * len(pk_preds)
            total_pk_nodes += len(pk_preds)

        # Validation
        model.eval()
        val_loss = 0
        val_pk_nodes = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                _, pk_predictions = model.pk_encoder(
                    batch.x, batch.edge_index,
                    batch.edge_weight if hasattr(batch, 'edge_weight') else None
                )

                pk_preds = pk_predictions[batch.pk_mask]
                pk_tgts = batch.pk_targets[batch.pk_mask].reshape(-1, 1)

                loss = criterion(pk_preds, pk_tgts)
                val_loss += loss.item() * len(pk_preds)
                val_pk_nodes += len(pk_preds)

        train_losses_pk.append(epoch_loss / total_pk_nodes)
        val_losses_pk.append(val_loss / val_pk_nodes)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs_pk}] - PK Train Loss: {train_losses_pk[-1]:.4f}, Val Loss: {val_losses_pk[-1]:.4f}")

    # Freeze PK encoder
    print("\nFreezing PK-GNN encoder...")
    model.freeze_pk_encoder()

    # ==================== STAGE 2: Train PD-GNN ====================
    print("\n" + "="*60)
    print("STAGE 2: Training PD-GNN (PK-GNN frozen)")
    print("="*60)

    optimizer_pd = optim.Adam(model.pd_decoder.parameters(), lr=learning_rate_pd)

    train_losses_pd = []
    val_losses_pd = []

    for epoch in range(epochs_pd):
        model.train()
        epoch_loss = 0
        total_pd_nodes = 0

        for batch in train_loader:
            batch = batch.to(device)

            # Forward pass (full model)
            pd_predictions = model(batch)

            # Only compute loss on PD nodes
            pd_preds = pd_predictions[batch.pd_mask]
            pd_tgts = batch.pd_targets[batch.pd_mask].reshape(-1, 1)

            loss = criterion(pd_preds, pd_tgts)

            # Backward pass
            optimizer_pd.zero_grad()
            loss.backward()
            optimizer_pd.step()

            epoch_loss += loss.item() * len(pd_preds)
            total_pd_nodes += len(pd_preds)

        # Validation
        model.eval()
        val_loss = 0
        val_pd_nodes = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pd_predictions = model(batch)

                pd_preds = pd_predictions[batch.pd_mask]
                pd_tgts = batch.pd_targets[batch.pd_mask].reshape(-1, 1)

                loss = criterion(pd_preds, pd_tgts)
                val_loss += loss.item() * len(pd_preds)
                val_pd_nodes += len(pd_preds)

        train_losses_pd.append(epoch_loss / total_pd_nodes)
        val_losses_pd.append(val_loss / val_pd_nodes)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs_pd}] - PD Train Loss: {train_losses_pd[-1]:.4f}, Val Loss: {val_losses_pd[-1]:.4f}")

    return model, {
        'train_losses_pk': train_losses_pk,
        'val_losses_pk': val_losses_pk,
        'train_losses_pd': train_losses_pd,
        'val_losses_pd': val_losses_pd
    }


def train_joint(train_graphs, val_graphs, feature_dim,
                hidden_dim=64,
                num_layers_pk=3,
                num_layers_pd=3,
                dropout=0.2,
                use_attention=False,
                use_gating=True,
                learning_rate=0.001,
                epochs=150,
                batch_size=8,
                pk_loss_weight=0.3,
                pd_loss_weight=1.0,
                device='cpu'):
    """
    Joint training: Train both PK-GNN and PD-GNN end-to-end with multi-task loss.
    """
    print("\n" + "="*60)
    print("JOINT TRAINING MODE (End-to-End Multi-Task Learning)")
    print("="*60)
    print(f"Loss weights: PK={pk_loss_weight}, PD={pd_loss_weight}")

    train_data_list = create_pyg_data_list(train_graphs)
    val_data_list = create_pyg_data_list(val_graphs)

    train_loader = DataLoader(train_data_list, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data_list, batch_size=batch_size, shuffle=False)

    model = HierarchicalPKPDGNN(
        input_dim=feature_dim,
        hidden_dim=hidden_dim,
        num_layers_pk=num_layers_pk,
        num_layers_pd=num_layers_pd,
        dropout=dropout,
        use_attention=use_attention,
        use_gating=use_gating
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses_total = []
    train_losses_pk = []
    train_losses_pd = []
    val_losses_total = []
    val_losses_pk = []
    val_losses_pd = []

    for epoch in range(epochs):
        model.train()
        epoch_loss_total = 0
        epoch_loss_pk = 0
        epoch_loss_pd = 0
        total_pk_nodes = 0
        total_pd_nodes = 0

        for batch in train_loader:
            batch = batch.to(device)

            # Forward pass (both stages)
            pd_predictions, pk_predictions = model(batch, return_pk=True)

            # Compute losses for both PK and PD
            pk_preds = pk_predictions[batch.pk_mask]
            pk_tgts = batch.pk_targets[batch.pk_mask].reshape(-1, 1)
            loss_pk = criterion(pk_preds, pk_tgts)

            pd_preds = pd_predictions[batch.pd_mask]
            pd_tgts = batch.pd_targets[batch.pd_mask].reshape(-1, 1)
            loss_pd = criterion(pd_preds, pd_tgts)

            # Weighted multi-task loss
            loss_total = pk_loss_weight * loss_pk + pd_loss_weight * loss_pd

            # Backward pass
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

            epoch_loss_total += loss_total.item() * (len(pk_preds) + len(pd_preds))
            epoch_loss_pk += loss_pk.item() * len(pk_preds)
            epoch_loss_pd += loss_pd.item() * len(pd_preds)
            total_pk_nodes += len(pk_preds)
            total_pd_nodes += len(pd_preds)

        # Validation
        model.eval()
        val_loss_total = 0
        val_loss_pk = 0
        val_loss_pd = 0
        val_pk_nodes = 0
        val_pd_nodes = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pd_predictions, pk_predictions = model(batch, return_pk=True)

                pk_preds = pk_predictions[batch.pk_mask]
                pk_tgts = batch.pk_targets[batch.pk_mask].reshape(-1, 1)
                loss_pk = criterion(pk_preds, pk_tgts)

                pd_preds = pd_predictions[batch.pd_mask]
                pd_tgts = batch.pd_targets[batch.pd_mask].reshape(-1, 1)
                loss_pd = criterion(pd_preds, pd_tgts)

                loss_total = pk_loss_weight * loss_pk + pd_loss_weight * loss_pd

                val_loss_total += loss_total.item() * (len(pk_preds) + len(pd_preds))
                val_loss_pk += loss_pk.item() * len(pk_preds)
                val_loss_pd += loss_pd.item() * len(pd_preds)
                val_pk_nodes += len(pk_preds)
                val_pd_nodes += len(pd_preds)

        train_losses_total.append(epoch_loss_total / (total_pk_nodes + total_pd_nodes))
        train_losses_pk.append(epoch_loss_pk / total_pk_nodes)
        train_losses_pd.append(epoch_loss_pd / total_pd_nodes)

        val_losses_total.append(val_loss_total / (val_pk_nodes + val_pd_nodes))
        val_losses_pk.append(val_loss_pk / val_pk_nodes)
        val_losses_pd.append(val_loss_pd / val_pd_nodes)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] - "
                  f"Total: {train_losses_total[-1]:.4f}/{val_losses_total[-1]:.4f}, "
                  f"PK: {train_losses_pk[-1]:.4f}/{val_losses_pk[-1]:.4f}, "
                  f"PD: {train_losses_pd[-1]:.4f}/{val_losses_pd[-1]:.4f}")

    return model, {
        'train_losses_total': train_losses_total,
        'train_losses_pk': train_losses_pk,
        'train_losses_pd': train_losses_pd,
        'val_losses_total': val_losses_total,
        'val_losses_pk': val_losses_pk,
        'val_losses_pd': val_losses_pd
    }


def evaluate_hierarchical_gnn(model, graphs, device='cpu', selected_patients=None):
    """
    Evaluate hierarchical GNN model on both PK and PD predictions.

    Args:
        model: Trained hierarchical GNN model
        graphs: List of graph dictionaries
        device: Device to run evaluation on
        selected_patients: List of patient IDs to evaluate. If None, evaluates all patients.

    Returns:
        Dictionary with evaluation results including metrics and predictions
    """
    # Filter graphs if specific patients are selected
    if selected_patients is not None:
        graphs = [g for g in graphs if g['patient_id'] in selected_patients]
        if len(graphs) == 0:
            raise ValueError(f"No graphs found for selected patients: {selected_patients}")
        print(f"Evaluating {len(graphs)} selected patients: {selected_patients}")

    data_list = create_pyg_data_list(graphs)
    loader = DataLoader(data_list, batch_size=8, shuffle=False)

    model.eval()
    all_pk_predictions = []
    all_pk_targets = []
    all_pd_predictions = []
    all_pd_targets = []
    all_pk_metadata = []
    all_pd_metadata = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pd_predictions, pk_predictions = model(batch, return_pk=True)

            pd_predictions = pd_predictions.cpu().numpy().flatten()
            pk_predictions = pk_predictions.cpu().numpy().flatten()

            # Extract PK predictions and metadata
            pk_mask = batch.pk_mask.cpu().numpy()
            pk_preds = pk_predictions[pk_mask]
            pk_tgts = batch.pk_targets.cpu().numpy()[pk_mask]

            # Extract PD predictions and metadata
            pd_mask = batch.pd_mask.cpu().numpy()
            pd_preds = pd_predictions[pd_mask]
            pd_tgts = batch.pd_targets.cpu().numpy()[pd_mask]

            # Get patient IDs and times
            node_idx = 0
            for i in range(batch.num_graphs):
                num_nodes = (batch.batch == i).sum().item()
                graph_pk_mask = pk_mask[node_idx:node_idx+num_nodes]
                graph_pd_mask = pd_mask[node_idx:node_idx+num_nodes]
                graph_times = batch.times[node_idx:node_idx+num_nodes]

                for j, is_pk in enumerate(graph_pk_mask):
                    if is_pk:
                        all_pk_metadata.append((batch.patient_id[i], graph_times[j].item()))

                for j, is_pd in enumerate(graph_pd_mask):
                    if is_pd:
                        all_pd_metadata.append((batch.patient_id[i], graph_times[j].item()))

                node_idx += num_nodes

            all_pk_predictions.extend(pk_preds)
            all_pk_targets.extend(pk_tgts)
            all_pd_predictions.extend(pd_preds)
            all_pd_targets.extend(pd_tgts)

    all_pk_predictions = np.array(all_pk_predictions)
    all_pk_targets = np.array(all_pk_targets)
    all_pd_predictions = np.array(all_pd_predictions)
    all_pd_targets = np.array(all_pd_targets)

    # PK metrics
    pk_mse = np.mean((all_pk_targets - all_pk_predictions) ** 2)
    pk_rmse = np.sqrt(pk_mse)
    pk_mae = np.mean(np.abs(all_pk_targets - all_pk_predictions))
    pk_r2 = 1 - (np.sum((all_pk_targets - all_pk_predictions) ** 2) /
                 np.sum((all_pk_targets - np.mean(all_pk_targets)) ** 2))

    # PD metrics
    pd_mse = np.mean((all_pd_targets - all_pd_predictions) ** 2)
    pd_rmse = np.sqrt(pd_mse)
    pd_mae = np.mean(np.abs(all_pd_targets - all_pd_predictions))
    pd_r2 = 1 - (np.sum((all_pd_targets - all_pd_predictions) ** 2) /
                 np.sum((all_pd_targets - np.mean(all_pd_targets)) ** 2))

    return {
        'pk': {
            'mse': pk_mse,
            'rmse': pk_rmse,
            'mae': pk_mae,
            'r2': pk_r2,
            'predictions': all_pk_predictions,
            'targets': all_pk_targets,
            'metadata': all_pk_metadata
        },
        'pd': {
            'mse': pd_mse,
            'rmse': pd_rmse,
            'mae': pd_mae,
            'r2': pd_r2,
            'predictions': all_pd_predictions,
            'targets': all_pd_targets,
            'metadata': all_pd_metadata
        }
    }


def get_available_patients(graphs):
    """
    Get list of available patient IDs from graphs.

    Args:
        graphs: List of graph dictionaries

    Returns:
        List of unique patient IDs
    """
    return [g['patient_id'] for g in graphs]


def plot_hierarchical_results(train_results, test_results, losses_dict,
                              training_mode='sequential', save_dir='Results/PD_Hierarchical_GNN',
                              n_patients=3, patient_ids=None, random_patients=False, random_seed=1712):
    """
    Plot results for hierarchical GNN.

    Args:
        train_results: Training evaluation results
        test_results: Test evaluation results
        losses_dict: Dictionary of training losses
        training_mode: 'sequential' or 'joint'
        save_dir: Directory to save plots
        n_patients: Number of patients to plot (if patient_ids not specified)
        patient_ids: Specific patient IDs to plot (overrides n_patients)
        random_patients: Whether to randomly select patients
        random_seed: Random seed for patient selection
    """
    os.makedirs(save_dir, exist_ok=True)

    # 1. Training history
    if training_mode == 'sequential':
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))

        # PK training
        epochs_pk = range(1, len(losses_dict['train_losses_pk']) + 1)
        axes[0].plot(epochs_pk, losses_dict['train_losses_pk'], 'b-', label='Train', linewidth=2)
        axes[0].plot(epochs_pk, losses_dict['val_losses_pk'], 'r-', label='Validation', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss (MSE)')
        axes[0].set_title('Stage 1: PK-GNN Training History')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # PD training
        epochs_pd = range(1, len(losses_dict['train_losses_pd']) + 1)
        axes[1].plot(epochs_pd, losses_dict['train_losses_pd'], 'b-', label='Train', linewidth=2)
        axes[1].plot(epochs_pd, losses_dict['val_losses_pd'], 'r-', label='Validation', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss (MSE)')
        axes[1].set_title('Stage 2: PD-GNN Training History (PK frozen)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    else:  # joint
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        epochs = range(1, len(losses_dict['train_losses_total']) + 1)

        # Total loss
        axes[0].plot(epochs, losses_dict['train_losses_total'], 'b-', label='Train', linewidth=2)
        axes[0].plot(epochs, losses_dict['val_losses_total'], 'r-', label='Validation', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Weighted Loss')
        axes[0].set_title('Joint Training: Total Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # PK loss
        axes[1].plot(epochs, losses_dict['train_losses_pk'], 'b-', label='Train', linewidth=2)
        axes[1].plot(epochs, losses_dict['val_losses_pk'], 'r-', label='Validation', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss (MSE)')
        axes[1].set_title('PK-GNN Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # PD loss
        axes[2].plot(epochs, losses_dict['train_losses_pd'], 'b-', label='Train', linewidth=2)
        axes[2].plot(epochs, losses_dict['val_losses_pd'], 'r-', label='Validation', linewidth=2)
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Loss (MSE)')
        axes[2].set_title('PD-GNN Loss')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Scatter plots for PK and PD
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # PK - Train
    axes[0, 0].scatter(train_results['pk']['targets'], train_results['pk']['predictions'],
                       alpha=0.6, edgecolors='k', linewidth=0.5)
    min_val = min(train_results['pk']['targets'].min(), train_results['pk']['predictions'].min())
    max_val = max(train_results['pk']['targets'].max(), train_results['pk']['predictions'].max())
    axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual PK')
    axes[0, 0].set_ylabel('Predicted PK')
    axes[0, 0].set_title(f'PK Train (R²={train_results["pk"]["r2"]:.4f}, RMSE={train_results["pk"]["rmse"]:.4f})')
    axes[0, 0].grid(True, alpha=0.3)

    # PK - Test
    axes[0, 1].scatter(test_results['pk']['targets'], test_results['pk']['predictions'],
                       alpha=0.6, edgecolors='k', linewidth=0.5, color='orange')
    min_val = min(test_results['pk']['targets'].min(), test_results['pk']['predictions'].min())
    max_val = max(test_results['pk']['targets'].max(), test_results['pk']['predictions'].max())
    axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    axes[0, 1].set_xlabel('Actual PK')
    axes[0, 1].set_ylabel('Predicted PK')
    axes[0, 1].set_title(f'PK Test (R²={test_results["pk"]["r2"]:.4f}, RMSE={test_results["pk"]["rmse"]:.4f})')
    axes[0, 1].grid(True, alpha=0.3)

    # PD - Train
    axes[1, 0].scatter(train_results['pd']['targets'], train_results['pd']['predictions'],
                       alpha=0.6, edgecolors='k', linewidth=0.5)
    min_val = min(train_results['pd']['targets'].min(), train_results['pd']['predictions'].min())
    max_val = max(train_results['pd']['targets'].max(), train_results['pd']['predictions'].max())
    axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    axes[1, 0].set_xlabel('Actual PD')
    axes[1, 0].set_ylabel('Predicted PD')
    axes[1, 0].set_title(f'PD Train (R²={train_results["pd"]["r2"]:.4f}, RMSE={train_results["pd"]["rmse"]:.4f})')
    axes[1, 0].grid(True, alpha=0.3)

    # PD - Test
    axes[1, 1].scatter(test_results['pd']['targets'], test_results['pd']['predictions'],
                       alpha=0.6, edgecolors='k', linewidth=0.5, color='orange')
    min_val = min(test_results['pd']['targets'].min(), test_results['pd']['predictions'].min())
    max_val = max(test_results['pd']['targets'].max(), test_results['pd']['predictions'].max())
    axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    axes[1, 1].set_xlabel('Actual PD')
    axes[1, 1].set_ylabel('Predicted PD')
    axes[1, 1].set_title(f'PD Test (R²={test_results["pd"]["r2"]:.4f}, RMSE={test_results["pd"]["rmse"]:.4f})')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'predictions_scatter.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Time series plots for PD
    all_pd_metadata = train_results['pd']['metadata'] + test_results['pd']['metadata']
    all_pd_predictions = np.concatenate([train_results['pd']['predictions'], test_results['pd']['predictions']])
    all_pd_targets = np.concatenate([train_results['pd']['targets'], test_results['pd']['targets']])
    all_is_test = np.concatenate([
        np.zeros(len(train_results['pd']['predictions']), dtype=bool),
        np.ones(len(test_results['pd']['predictions']), dtype=bool)
    ])

    plot_df = pd.DataFrame({
        'ID': [m[0] for m in all_pd_metadata],
        'TIME': [m[1] for m in all_pd_metadata],
        'Actual': all_pd_targets,
        'Predicted': all_pd_predictions,
        'is_test': all_is_test
    })

    # Select patients based on criteria
    all_unique_patients = plot_df['ID'].unique()

    if patient_ids is not None:
        # Use specified patient IDs
        selected_patients = [pid for pid in patient_ids if pid in all_unique_patients]
        if len(selected_patients) == 0:
            print(f"Warning: None of the specified patients {patient_ids} found in data.")
            print(f"Available patients: {all_unique_patients[:10]}...")
            selected_patients = all_unique_patients[:n_patients]
        else:
            print(f"Plotting specified patients: {selected_patients}")
    elif random_patients:
        # Random selection
        np.random.seed(random_seed)
        n_to_select = min(n_patients, len(all_unique_patients))
        selected_patients = np.random.choice(all_unique_patients, size=n_to_select, replace=False)
        print(f"Plotting {n_to_select} random patients: {selected_patients}")
    else:
        # First N patients
        selected_patients = all_unique_patients[:n_patients]
        print(f"Plotting first {len(selected_patients)} patients: {selected_patients}")

    fig, axes = plt.subplots(len(selected_patients), 1, figsize=(12, 4*len(selected_patients)))
    if len(selected_patients) == 1:
        axes = [axes]

    for idx, patient_id in enumerate(selected_patients):
        patient_data = plot_df[plot_df['ID'] == patient_id].sort_values('TIME')

        axes[idx].plot(patient_data['TIME'], patient_data['Actual'],
                      'o-', label='Actual PD', markersize=6, linewidth=2, alpha=0.7, color='blue')

        train_data = patient_data[~patient_data['is_test']]
        test_data = patient_data[patient_data['is_test']]

        if len(train_data) > 0:
            axes[idx].plot(train_data['TIME'], train_data['Predicted'],
                          's', label='Predicted (Train)', markersize=6, alpha=0.6, color='orange')

        if len(test_data) > 0:
            axes[idx].plot(test_data['TIME'], test_data['Predicted'],
                          's', label='Predicted (Test)', markersize=7, alpha=0.9, color='red')

        axes[idx].set_xlabel('Time (hours)')
        axes[idx].set_ylabel('PD Value')
        axes[idx].set_title(f'Patient {int(patient_id)} - Hierarchical GNN ({training_mode.upper()})')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'timeseries_pd.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nPlots saved to: {save_dir}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Train Hierarchical GNN for PK-PD prediction')
    parser.add_argument('--csv_path', type=str, default='Data/QIC2025-EstDat.csv')
    parser.add_argument('--training_mode', type=str, default='joint', choices=['sequential', 'joint'])
    parser.add_argument('--feature_engineering', action='store_true', default=True)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--num_layers_pk', type=int, default=3)
    parser.add_argument('--num_layers_pd', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--use_attention', action='store_true', default=False)
    parser.add_argument('--use_gating', action='store_true', default=True)

    # Sequential training params
    parser.add_argument('--learning_rate_pk', type=float, default=0.001)
    parser.add_argument('--learning_rate_pd', type=float, default=0.001)
    parser.add_argument('--epochs_pk', type=int, default=100)
    parser.add_argument('--epochs_pd', type=int, default=100)

    # Joint training params
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--pk_loss_weight', type=float, default=0.3)
    parser.add_argument('--pd_loss_weight', type=float, default=1.0)

    # Common params
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--random_seed', type=int, default=1712)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--save_dir', type=str, default='Results/PD_Hierarchical_GNN')

    # Patient selection for evaluation and plotting
    parser.add_argument('--n_patients', type=int, default=3,
                       help='Number of patients to plot (if patient_ids not specified)')
    parser.add_argument('--patient_ids', type=int, nargs='+', default=None,
                       help='Specific patient IDs to plot (e.g., --patient_ids 1 5 10)')
    parser.add_argument('--random_patients', action='store_true', default=False,
                       help='Randomly select patients for plotting')
    parser.add_argument('--eval_patient_ids', type=int, nargs='+', default=None,
                       help='Specific patient IDs to evaluate separately (e.g., --eval_patient_ids 1 2 3)')

    args = parser.parse_args()

    # Prepare data
    data_dict = prepare_hierarchical_gnn_data(
        csv_path=args.csv_path,
        feature_engineering=args.feature_engineering,
        test_size=args.test_size,
        random_seed=args.random_seed
    )

    # Train based on mode
    if args.training_mode == 'sequential':
        model, losses_dict = train_sequential(
            data_dict['train_graphs'],
            data_dict['test_graphs'],
            data_dict['feature_dim'],
            hidden_dim=args.hidden_dim,
            num_layers_pk=args.num_layers_pk,
            num_layers_pd=args.num_layers_pd,
            dropout=args.dropout,
            use_attention=args.use_attention,
            use_gating=args.use_gating,
            learning_rate_pk=args.learning_rate_pk,
            learning_rate_pd=args.learning_rate_pd,
            epochs_pk=args.epochs_pk,
            epochs_pd=args.epochs_pd,
            batch_size=args.batch_size,
            device=args.device
        )
    else:  # joint
        model, losses_dict = train_joint(
            data_dict['train_graphs'],
            data_dict['test_graphs'],
            data_dict['feature_dim'],
            hidden_dim=args.hidden_dim,
            num_layers_pk=args.num_layers_pk,
            num_layers_pd=args.num_layers_pd,
            dropout=args.dropout,
            use_attention=args.use_attention,
            use_gating=args.use_gating,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            batch_size=args.batch_size,
            pk_loss_weight=args.pk_loss_weight,
            pd_loss_weight=args.pd_loss_weight,
            device=args.device
        )

    # Print available patients
    train_patient_ids = get_available_patients(data_dict['train_graphs'])
    test_patient_ids = get_available_patients(data_dict['test_graphs'])
    all_patient_ids = train_patient_ids + test_patient_ids

    print(f"\n=== Available Patients ===")
    print(f"Train patients ({len(train_patient_ids)}): {train_patient_ids[:10]}{'...' if len(train_patient_ids) > 10 else ''}")
    print(f"Test patients ({len(test_patient_ids)}): {test_patient_ids[:10]}{'...' if len(test_patient_ids) > 10 else ''}")

    # Evaluate
    print("\n=== Evaluation ===")
    print("\n--- Overall Evaluation (All Patients) ---")
    train_results = evaluate_hierarchical_gnn(model, data_dict['train_graphs'], args.device)
    test_results = evaluate_hierarchical_gnn(model, data_dict['test_graphs'], args.device)

    print(f"\nPK Metrics:")
    print(f"  Train - RMSE: {train_results['pk']['rmse']:.4f}, R²: {train_results['pk']['r2']:.4f}")
    print(f"  Test  - RMSE: {test_results['pk']['rmse']:.4f}, R²: {test_results['pk']['r2']:.4f}")

    print(f"\nPD Metrics:")
    print(f"  Train - RMSE: {train_results['pd']['rmse']:.4f}, R²: {train_results['pd']['r2']:.4f}")
    print(f"  Test  - RMSE: {test_results['pd']['rmse']:.4f}, R²: {test_results['pd']['r2']:.4f}")

    # Evaluate specific patients if requested
    if args.eval_patient_ids is not None:
        print("\n--- Specific Patient Evaluation ---")
        # Combine train and test graphs for patient-specific evaluation
        all_graphs = data_dict['train_graphs'] + data_dict['test_graphs']
        patient_results = evaluate_hierarchical_gnn(
            model, all_graphs, args.device,
            selected_patients=args.eval_patient_ids
        )

        print(f"\nPatient-Specific PK Metrics:")
        print(f"  RMSE: {patient_results['pk']['rmse']:.4f}, R²: {patient_results['pk']['r2']:.4f}")

        print(f"\nPatient-Specific PD Metrics:")
        print(f"  RMSE: {patient_results['pd']['rmse']:.4f}, R²: {patient_results['pd']['r2']:.4f}")

    # Plot
    save_dir = f"{args.save_dir}_{args.training_mode}"
    plot_hierarchical_results(train_results, test_results, losses_dict,
                             training_mode=args.training_mode,
                             save_dir=save_dir,
                             n_patients=args.n_patients,
                             patient_ids=args.patient_ids,
                             random_patients=args.random_patients,
                             random_seed=args.random_seed)

    # Save metrics
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'metrics.txt'), 'w') as f:
        f.write("="*60 + "\n")
        f.write(f"Hierarchical PK-PD GNN - {args.training_mode.upper()} Training\n")
        f.write("="*60 + "\n\n")
        f.write(f"Training Mode: {args.training_mode}\n")
        f.write(f"Feature Engineering: {args.feature_engineering}\n")
        f.write(f"Use Attention: {args.use_attention}\n")
        f.write(f"Use Gating: {args.use_gating}\n\n")

        f.write("PK Metrics:\n")
        f.write(f"  Train - RMSE: {train_results['pk']['rmse']:.4f}, MAE: {train_results['pk']['mae']:.4f}, R²: {train_results['pk']['r2']:.4f}\n")
        f.write(f"  Test  - RMSE: {test_results['pk']['rmse']:.4f}, MAE: {test_results['pk']['mae']:.4f}, R²: {test_results['pk']['r2']:.4f}\n\n")

        f.write("PD Metrics:\n")
        f.write(f"  Train - RMSE: {train_results['pd']['rmse']:.4f}, MAE: {train_results['pd']['mae']:.4f}, R²: {train_results['pd']['r2']:.4f}\n")
        f.write(f"  Test  - RMSE: {test_results['pd']['rmse']:.4f}, MAE: {test_results['pd']['mae']:.4f}, R²: {test_results['pd']['r2']:.4f}\n")

    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"Results saved to: {save_dir}")
    print('='*60)


if __name__ == "__main__":
    main()
