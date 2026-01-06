#!/usr/bin/env python3
"""
Train GNN to predict PD values.
Each patient is a graph with PK and PD nodes.
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
from torch_geometric.nn import GCNConv, global_mean_pool

# Set style
sns.set_style('whitegrid')
plt.rcParams['font.size'] = 10


class PDPredictionGNN(nn.Module):
    """GNN for PD prediction using patient graphs."""

    def __init__(self, input_dim, hidden_dim=64, num_layers=3, dropout=0.2):
        super(PDPredictionGNN, self).__init__()

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))

        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        self.dropout = dropout

        # Output layer for node-level prediction
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # GNN layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = torch.relu(x)
            if i < len(self.convs) - 1:
                x = torch.dropout(x, p=self.dropout, train=self.training)

        # Node-level prediction (only for PD nodes)
        node_predictions = self.predictor(x)

        return node_predictions


def create_patient_graph(patient_data, feature_engineering=False):
    """
    Create a graph for one patient.

    Nodes: Each observation (PK or PD) is a node
    Edges: Temporal connections + PK-PD interactions

    Args:
        patient_data: DataFrame for one patient
        feature_engineering: Whether to apply feature engineering

    Returns:
        node_features: (num_nodes, feature_dim)
        edge_index: (2, num_edges)
        node_types: (num_nodes,) - 0 for PK, 1 for PD
        pd_targets: (num_pd_nodes,) - PD values
        pd_node_indices: Indices of PD nodes
        times: Time values for each node
    """
    patient_data = patient_data.sort_values('TIME').reset_index(drop=True)

    node_features = []
    node_types = []
    node_targets = []
    times = []
    edges = []

    # Separate PK and PD observations
    pk_obs = patient_data[patient_data['DVID'] == 1].reset_index(drop=True)
    pd_obs = patient_data[patient_data['DVID'] == 2].reset_index(drop=True)

    node_idx = 0
    pk_node_map = {}  # time -> node_idx for PK
    pd_node_map = {}  # time -> node_idx for PD

    # Add PK nodes
    for idx, row in pk_obs.iterrows():
        features = [
            row['TIME'],
            row['BW'],
            row['DOSE'],
            row['COMED'],
            row['DV']  # PK value
        ]

        if feature_engineering:
            features.extend([
                np.log1p(row['TIME']),
                np.sqrt(row['TIME']),
                np.log1p(row['DV']),
                row['TIME'] * row['DV'],
                row['BW'] * row['DV'],
                row['DOSE'] / (row['BW'] + 1e-8),
                np.sin(2 * np.pi * row['TIME'] / 24),
                np.cos(2 * np.pi * row['TIME'] / 24)
            ])

        node_features.append(features)
        node_types.append(0)  # PK node
        node_targets.append(0)  # Not predicting PK
        times.append(row['TIME'])
        pk_node_map[row['TIME']] = node_idx
        node_idx += 1

    # Add PD nodes
    pd_node_indices = []
    for idx, row in pd_obs.iterrows():
        # Find most recent PK value
        recent_pk = pk_obs[pk_obs['TIME'] <= row['TIME']]
        pk_value = recent_pk.iloc[-1]['DV'] if len(recent_pk) > 0 else 0.0

        features = [
            row['TIME'],
            row['BW'],
            row['DOSE'],
            row['COMED'],
            pk_value  # Most recent PK
        ]

        if feature_engineering:
            features.extend([
                np.log1p(row['TIME']),
                np.sqrt(row['TIME']),
                np.log1p(pk_value),
                row['TIME'] * pk_value,
                row['BW'] * pk_value,
                row['DOSE'] / (row['BW'] + 1e-8),
                np.sin(2 * np.pi * row['TIME'] / 24),
                np.cos(2 * np.pi * row['TIME'] / 24)
            ])

        node_features.append(features)
        node_types.append(1)  # PD node
        node_targets.append(row['DV'])  # PD value to predict
        times.append(row['TIME'])
        pd_node_map[row['TIME']] = node_idx
        pd_node_indices.append(node_idx)
        node_idx += 1

    # Create edges
    # 1. Temporal edges within PK nodes
    pk_nodes = [i for i, t in enumerate(node_types) if t == 0]
    for i in range(len(pk_nodes) - 1):
        edges.append([pk_nodes[i], pk_nodes[i+1]])
        edges.append([pk_nodes[i+1], pk_nodes[i]])  # Bidirectional

    # 2. Temporal edges within PD nodes
    pd_nodes = [i for i, t in enumerate(node_types) if t == 1]
    for i in range(len(pd_nodes) - 1):
        edges.append([pd_nodes[i], pd_nodes[i+1]])
        edges.append([pd_nodes[i+1], pd_nodes[i]])  # Bidirectional

    # 3. PK-PD interaction edges
    for pd_idx in pd_nodes:
        pd_time = times[pd_idx]
        # Connect to most recent PK node
        for pk_idx in reversed(pk_nodes):
            if times[pk_idx] <= pd_time:
                edges.append([pk_idx, pd_idx])
                edges.append([pd_idx, pk_idx])
                break

    node_features = np.array(node_features, dtype=np.float32)
    edge_index = np.array(edges, dtype=np.int64).T if len(edges) > 0 else np.array([[], []], dtype=np.int64)

    return node_features, edge_index, np.array(node_types), np.array(node_targets), pd_node_indices, times


def prepare_pd_gnn_data(csv_path='Data/QIC2025-EstDat.csv',
                        feature_engineering=False,
                        test_size=0.2,
                        random_seed=1712):
    """
    Prepare graph data for GNN PD prediction.

    Each patient is a separate graph.
    """
    print("Loading data...")
    df = pd.read_csv(csv_path)

    # Filter to observations only
    df = df[df['EVID'] == 0].copy()
    if 'MDV' in df.columns:
        df = df[df['MDV'] == 0]

    print(f"Total observations: {len(df)}")

    if 'DVID' not in df.columns:
        raise ValueError("DVID column required for GNN approach")

    # Create graphs for each patient
    print("\nCreating patient graphs...")
    graphs = []
    all_metadata = []

    scaler_X = StandardScaler()
    all_node_features = []

    for patient_id in df['ID'].unique():
        patient_data = df[df['ID'] == patient_id]

        node_features, edge_index, node_types, node_targets, pd_node_indices, times = create_patient_graph(
            patient_data, feature_engineering
        )

        if len(pd_node_indices) == 0:
            continue  # Skip patients with no PD observations

        all_node_features.append(node_features)

        # Store graph data
        graphs.append({
            'patient_id': patient_id,
            'node_features': node_features,
            'edge_index': edge_index,
            'node_types': node_types,
            'node_targets': node_targets,
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

    # Calculate total PD predictions
    train_pd_count = sum(len(g['pd_node_indices']) for g in train_graphs)
    test_pd_count = sum(len(g['pd_node_indices']) for g in test_graphs)
    print(f"Train PD predictions: {train_pd_count}")
    print(f"Test PD predictions: {test_pd_count}")

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
        y = torch.FloatTensor(graph['node_targets'])

        # Mask for PD nodes (nodes we want to predict)
        pd_mask = torch.zeros(len(graph['node_types']), dtype=torch.bool)
        pd_mask[graph['pd_node_indices']] = True

        # Convert times to tensor for proper batching
        times_tensor = torch.FloatTensor(graph['times'])

        data = Data(
            x=x,
            edge_index=edge_index,
            y=y,
            pd_mask=pd_mask,
            patient_id=graph['patient_id'],
            times=times_tensor
        )

        data_list.append(data)

    return data_list


def train_gnn(train_graphs, val_graphs, feature_dim,
              hidden_dim=64,
              num_layers=3,
              dropout=0.2,
              learning_rate=0.001,
              epochs=100,
              batch_size=8,
              device='cpu'):
    """Train GNN model."""

    train_data_list = create_pyg_data_list(train_graphs)
    val_data_list = create_pyg_data_list(val_graphs)

    train_loader = DataLoader(train_data_list, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data_list, batch_size=batch_size, shuffle=False)

    model = PDPredictionGNN(feature_dim, hidden_dim, num_layers, dropout).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = []

    print(f"\n=== Training GNN ===")
    print(f"Hidden dim: {hidden_dim}, Layers: {num_layers}")
    print(f"Epochs: {epochs}, Batch size: {batch_size}, LR: {learning_rate}")
    print(f"Device: {device}")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        total_pd_nodes = 0

        for batch in train_loader:
            batch = batch.to(device)

            # Forward pass
            predictions = model(batch)

            # Only compute loss on PD nodes
            pd_predictions = predictions[batch.pd_mask]
            pd_targets = batch.y[batch.pd_mask].reshape(-1, 1)

            loss = criterion(pd_predictions, pd_targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(pd_predictions)
            total_pd_nodes += len(pd_predictions)

        # Validation
        model.eval()
        val_loss = 0
        val_pd_nodes = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                predictions = model(batch)
                pd_predictions = predictions[batch.pd_mask]
                pd_targets = batch.y[batch.pd_mask].reshape(-1, 1)

                loss = criterion(pd_predictions, pd_targets)
                val_loss += loss.item() * len(pd_predictions)
                val_pd_nodes += len(pd_predictions)

        train_losses.append(epoch_loss / total_pd_nodes)
        val_losses.append(val_loss / val_pd_nodes)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")

    return model, train_losses, val_losses


def evaluate_gnn(model, graphs, device='cpu'):
    """Evaluate GNN model."""
    data_list = create_pyg_data_list(graphs)
    loader = DataLoader(data_list, batch_size=8, shuffle=False)

    model.eval()
    all_predictions = []
    all_targets = []
    all_metadata = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            predictions = model(batch).cpu().numpy().flatten()

            # Extract PD predictions and metadata
            pd_mask = batch.pd_mask.cpu().numpy()
            pd_predictions = predictions[pd_mask]
            pd_targets = batch.y.cpu().numpy()[pd_mask]

            # Get patient IDs and times for PD nodes
            patient_ids = []
            times = []
            node_idx = 0
            for i in range(batch.num_graphs):
                num_nodes = (batch.batch == i).sum().item()
                graph_pd_mask = pd_mask[node_idx:node_idx+num_nodes]
                graph_times = batch.times[node_idx:node_idx+num_nodes]

                for j, is_pd in enumerate(graph_pd_mask):
                    if is_pd:
                        patient_ids.append(batch.patient_id[i])
                        times.append(graph_times[j])

                node_idx += num_nodes

            all_predictions.extend(pd_predictions)
            all_targets.extend(pd_targets)
            all_metadata.extend(list(zip(patient_ids, times)))

    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)

    mse = np.mean((all_targets - all_predictions) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(all_targets - all_predictions))
    r2 = 1 - (np.sum((all_targets - all_predictions) ** 2) / np.sum((all_targets - np.mean(all_targets)) ** 2))

    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'predictions': all_predictions,
        'targets': all_targets,
        'metadata': all_metadata
    }


def plot_results(train_results, test_results, train_losses, val_losses,
                 save_dir='Results/PD_GNN', n_patients=3, patient_ids=None,
                 random_patients=False, random_seed=1712):
    """Plot training results and patient-specific time series predictions."""

    os.makedirs(save_dir, exist_ok=True)

    # 1. Training history
    fig, ax = plt.subplots(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (MSE)')
    ax.set_title('Training History')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Scatter plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(train_results['targets'], train_results['predictions'], alpha=0.6, edgecolors='k', linewidth=0.5)
    min_val = min(train_results['targets'].min(), train_results['predictions'].min())
    max_val = max(train_results['targets'].max(), train_results['predictions'].max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    axes[0].set_xlabel('Actual PD')
    axes[0].set_ylabel('Predicted PD')
    axes[0].set_title(f'Train Set (R²={train_results["r2"]:.4f}, RMSE={train_results["rmse"]:.4f})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(test_results['targets'], test_results['predictions'], alpha=0.6, edgecolors='k', linewidth=0.5)
    min_val = min(test_results['targets'].min(), test_results['predictions'].min())
    max_val = max(test_results['targets'].max(), test_results['predictions'].max())
    axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    axes[1].set_xlabel('Actual PD')
    axes[1].set_ylabel('Predicted PD')
    axes[1].set_title(f'Test Set (R²={test_results["r2"]:.4f}, RMSE={test_results["rmse"]:.4f})')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'predictions_scatter.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Patient-specific time series
    all_metadata = train_results['metadata'] + test_results['metadata']
    all_predictions = np.concatenate([train_results['predictions'], test_results['predictions']])
    all_targets = np.concatenate([train_results['targets'], test_results['targets']])
    all_is_test = np.concatenate([
        np.zeros(len(train_results['predictions']), dtype=bool),
        np.ones(len(test_results['predictions']), dtype=bool)
    ])

    plot_df = pd.DataFrame({
        'ID': [m[0] for m in all_metadata],
        'TIME': [m[1] for m in all_metadata],
        'Actual': all_targets,
        'Predicted': all_predictions,
        'is_test': all_is_test
    })

    # Select patients
    all_unique_patients = plot_df['ID'].unique()

    if patient_ids is not None:
        unique_patients = [pid for pid in patient_ids if pid in all_unique_patients]
        if len(unique_patients) == 0:
            unique_patients = all_unique_patients[:n_patients]
    elif random_patients:
        np.random.seed(random_seed)
        n_to_select = min(n_patients, len(all_unique_patients))
        unique_patients = np.random.choice(all_unique_patients, size=n_to_select, replace=False)
    else:
        unique_patients = all_unique_patients[:n_patients]

    print(f"Plotting patients: {unique_patients}")

    fig, axes = plt.subplots(len(unique_patients), 1, figsize=(12, 4*len(unique_patients)))
    if len(unique_patients) == 1:
        axes = [axes]

    for idx, patient_id in enumerate(unique_patients):
        patient_data = plot_df[plot_df['ID'] == patient_id].sort_values('TIME')

        axes[idx].plot(patient_data['TIME'], patient_data['Actual'],
                      'o-', label='Actual PD', markersize=6, linewidth=2, alpha=0.7, color='blue')

        axes[idx].plot(patient_data['TIME'], patient_data['Predicted'],
                      '--', linewidth=1.5, alpha=0.4, color='gray', label='_nolegend_')

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
        axes[idx].set_title(f'Patient {int(patient_id)} - PD Prediction (GNN)')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)

    plt.suptitle('PD Predictions Over Time - GNN (Train and Test)', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'timeseries_predictions.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nPlots saved to: {save_dir}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Train GNN for PD prediction')
    parser.add_argument('--csv_path', type=str, default='Data/QIC2025-EstDat.csv')
    parser.add_argument('--feature_engineering', action='store_true', default=False)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--random_seed', type=int, default=1712)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--save_dir', type=str, default='Results/PD_GNN')
    parser.add_argument('--n_patients', type=int, default=3)
    parser.add_argument('--patient_ids', type=int, nargs='+', default=None)
    parser.add_argument('--random_patients', action='store_true', default=False)

    args = parser.parse_args()

    # Prepare data
    data_dict = prepare_pd_gnn_data(
        csv_path=args.csv_path,
        feature_engineering=args.feature_engineering,
        test_size=args.test_size,
        random_seed=args.random_seed
    )

    # Train
    model, train_losses, val_losses = train_gnn(
        data_dict['train_graphs'],
        data_dict['test_graphs'],
        data_dict['feature_dim'],
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device
    )

    # Evaluate
    print("\n=== Evaluation ===")
    train_results = evaluate_gnn(model, data_dict['train_graphs'], args.device)
    test_results = evaluate_gnn(model, data_dict['test_graphs'], args.device)

    print(f"\nTrain Metrics:")
    print(f"  RMSE: {train_results['rmse']:.4f}")
    print(f"  MAE:  {train_results['mae']:.4f}")
    print(f"  R²:   {train_results['r2']:.4f}")

    print(f"\nTest Metrics:")
    print(f"  RMSE: {test_results['rmse']:.4f}")
    print(f"  MAE:  {test_results['mae']:.4f}")
    print(f"  R²:   {test_results['r2']:.4f}")

    # Plot
    plot_results(train_results, test_results, train_losses, val_losses,
                args.save_dir, args.n_patients, args.patient_ids,
                args.random_patients, args.random_seed)

    # Save metrics
    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, 'metrics.txt'), 'w') as f:
        f.write("="*60 + "\n")
        f.write("PD Prediction - GNN - Evaluation Metrics\n")
        f.write("="*60 + "\n\n")
        f.write(f"Feature Engineering: {args.feature_engineering}\n")
        f.write(f"Hidden Dim: {args.hidden_dim}\n")
        f.write(f"Num Layers: {args.num_layers}\n\n")
        f.write("Train Metrics:\n")
        f.write(f"  RMSE: {train_results['rmse']:.4f}\n")
        f.write(f"  MAE:  {train_results['mae']:.4f}\n")
        f.write(f"  R²:   {train_results['r2']:.4f}\n\n")
        f.write("Test Metrics:\n")
        f.write(f"  RMSE: {test_results['rmse']:.4f}\n")
        f.write(f"  MAE:  {test_results['mae']:.4f}\n")
        f.write(f"  R²:   {test_results['r2']:.4f}\n")

    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"Results saved to: {args.save_dir}")
    print('='*60)


if __name__ == "__main__":
    main()
