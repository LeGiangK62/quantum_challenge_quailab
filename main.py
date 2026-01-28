"""
Unified main entry point for PK/PD prediction training.

Supports:
- MLP (hierarchical) with modes: separate, joint, dual_stage, shared
- GNN (hierarchical) with modes: joint, sequential

Usage:
    python main.py --model mlp --mode dual_stage --epochs 300
    python main.py --model gnn --mode joint --epochs 150
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader as TorchDataLoader

# Local imports
from Utils.args import get_args, print_args
from Utils.data_loader import prepare_pkpd_data
from Utils.log import calculate_metrics, log_metrics, plot_metrics, logger
from Models.mlp import HierarchicalPKPDMLP
from Models.gnn import HierarchicalPKPDGNN
from Models.quantum import HQCNN, HQGNN

# Set style
sns.set_style('whitegrid')
plt.rcParams['font.size'] = 10


# ============================================================
# Hierarchical HQCNN Wrapper
# ============================================================
class HierarchicalHQCNN(nn.Module):
    """
    Hierarchical HQCNN for PK/PD prediction.

    Uses separate HQCNN models for PK and PD prediction.
    PD model receives PK prediction as additional input.
    """

    def __init__(self, pk_input_dim, pd_input_dim, num_layers=1, mode='dual_stage'):
        super().__init__()
        self.mode = mode

        # Separate HQCNN for PK and PD
        self.pk_model = HQCNN(pk_input_dim, num_layers=num_layers)
        self.pd_model = HQCNN(pd_input_dim + 1, num_layers=num_layers)  # +1 for PK prediction

    def forward(self, x_pk=None, x_pd=None):
        """
        Forward pass.

        Args:
            x_pk: PK input features [batch, pk_features]
            x_pd: PD input features [batch, pd_features]

        Returns:
            dict with 'pk' and/or 'pd' predictions
        """
        results = {}

        if x_pk is not None:
            pk_pred = self.pk_model(x_pk)
            results['pk'] = pk_pred

        if x_pd is not None:
            if self.mode == 'dual_stage' and 'pk' in results:
                # Use PK prediction (gradients flow)
                pk_for_pd = results['pk']
            elif self.mode == 'joint' and 'pk' in results:
                # Detach PK prediction
                pk_for_pd = results['pk'].detach()
            else:
                # No PK available
                pk_for_pd = torch.zeros(x_pd.size(0), 1, device=x_pd.device)

            x_pd_with_pk = torch.cat([x_pd, pk_for_pd], dim=1)
            pd_pred = self.pd_model(x_pd_with_pk)
            results['pd'] = pd_pred

        return results


# ============================================================
# Dataset Classes
# ============================================================
class PKPDDataset(Dataset):
    """Dataset for PK/PD tabular data (MLP)."""

    def __init__(self, pk_data: dict, pd_data: dict):
        """
        Args:
            pk_data: dict with 'X', 'y', 'ids', 'times'
            pd_data: dict with 'X', 'y', 'ids', 'times'
        """
        self.pk_X = torch.FloatTensor(pk_data['X'])
        self.pk_y = torch.FloatTensor(pk_data['y']).unsqueeze(1)
        self.pk_ids = pk_data['ids']
        self.pk_times = pk_data['times']

        self.pd_X = torch.FloatTensor(pd_data['X'])
        self.pd_y = torch.FloatTensor(pd_data['y']).unsqueeze(1)
        self.pd_ids = pd_data['ids']
        self.pd_times = pd_data['times']

    def __len__(self):
        return max(len(self.pk_X), len(self.pd_X))

    def __getitem__(self, idx):
        pk_idx = idx % len(self.pk_X)
        pd_idx = idx % len(self.pd_X)

        return {
            'pk_x': self.pk_X[pk_idx],
            'pk_y': self.pk_y[pk_idx],
            'pk_id': self.pk_ids[pk_idx],
            'pk_time': self.pk_times[pk_idx],
            'pd_x': self.pd_X[pd_idx],
            'pd_y': self.pd_y[pd_idx],
            'pd_id': self.pd_ids[pd_idx],
            'pd_time': self.pd_times[pd_idx],
        }


def collate_pkpd(batch):
    """Collate function for PK/PD batch."""
    pk_x = torch.stack([item['pk_x'] for item in batch])
    pk_y = torch.stack([item['pk_y'] for item in batch])
    pd_x = torch.stack([item['pd_x'] for item in batch])
    pd_y = torch.stack([item['pd_y'] for item in batch])

    return {
        'pk_x': pk_x,
        'pk_y': pk_y,
        'pd_x': pd_x,
        'pd_y': pd_y,
    }


# ============================================================
# Loss Functions
# ============================================================
def compute_loss(pred, target, loss_type='mse', quantile_q=0.3, hybrid_lambda=0.5):
    """
    Compute regression loss.

    Args:
        pred: Predictions
        target: Ground truth
        loss_type: 'mse', 'mae', 'asymmetric', 'quantile', 'hybrid'
        quantile_q: Quantile parameter
        hybrid_lambda: Weight for MSE in hybrid loss
    """
    if loss_type == 'mse':
        return nn.functional.mse_loss(pred, target)
    elif loss_type == 'mae':
        return nn.functional.l1_loss(pred, target)
    elif loss_type == 'asymmetric':
        diff = pred - target
        loss = torch.where(diff > 0, 2.0 * diff**2, 1.0 * diff**2)
        return loss.mean()
    elif loss_type == 'quantile':
        diff = target - pred
        return torch.max(quantile_q * diff, (quantile_q - 1) * diff).mean()
    elif loss_type == 'hybrid':
        mse = nn.functional.mse_loss(pred, target)
        diff = target - pred
        quantile = torch.max(quantile_q * diff, (quantile_q - 1) * diff).mean()
        return hybrid_lambda * mse + (1 - hybrid_lambda) * quantile
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


# ============================================================
# Training Functions
# ============================================================
def train_mlp(model, train_loader, val_loader, args, device):
    """
    Train hierarchical MLP model.

    Returns:
        model: Trained model
        history: Training history dict
    """
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=25, factor=0.5)

    history = {
        'Epoch': [],
        'Train PK_MSE': [], 'Train PK_RMSE': [], 'Train PK_MAE': [], 'Train PK_R2': [],
        'Train PD_MSE': [], 'Train PD_RMSE': [], 'Train PD_MAE': [], 'Train PD_R2': [],
        'Val PK_MSE': [], 'Val PK_RMSE': [], 'Val PK_MAE': [], 'Val PK_R2': [],
        'Val PD_MSE': [], 'Val PD_RMSE': [], 'Val PD_MAE': [], 'Val PD_R2': [],
    }

    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    logger.info(f"Training MLP ({args.mode.upper()} mode)")
    logger.info(f"Epochs: {args.epochs}, Batch size: {args.batch_size}, LR: {args.learning_rate}")
    logger.info(f"PK loss: {args.loss_type_pk}, PD loss: {args.loss_type_pd}")

    for epoch in range(args.epochs):
        model.train()

        # Collect predictions for metrics
        train_pk_preds, train_pk_targets = [], []
        train_pd_preds, train_pd_targets = [], []

        for batch in train_loader:
            pk_x = batch['pk_x'].to(device)
            pk_y = batch['pk_y'].to(device)
            pd_x = batch['pd_x'].to(device)
            pd_y = batch['pd_y'].to(device)

            # Forward pass
            results = model(pk_x, pd_x)

            # Compute loss
            loss_pk = compute_loss(results['pk'], pk_y, args.loss_type_pk,
                                   args.quantile_q, args.hybrid_lambda)
            loss_pd = compute_loss(results['pd'], pd_y, args.loss_type_pd,
                                   args.quantile_q, args.hybrid_lambda)
            loss = args.pk_loss_weight * loss_pk + args.pd_loss_weight * loss_pd

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Collect predictions
            train_pk_preds.append(results['pk'].detach())
            train_pk_targets.append(pk_y.detach())
            train_pd_preds.append(results['pd'].detach())
            train_pd_targets.append(pd_y.detach())

        # Compute training metrics
        train_pk_preds = torch.cat(train_pk_preds)
        train_pk_targets = torch.cat(train_pk_targets)
        train_pd_preds = torch.cat(train_pd_preds)
        train_pd_targets = torch.cat(train_pd_targets)

        train_pk_metrics = calculate_metrics(train_pk_targets, train_pk_preds)
        train_pd_metrics = calculate_metrics(train_pd_targets, train_pd_preds)

        # Validation
        model.eval()
        val_pk_preds, val_pk_targets = [], []
        val_pd_preds, val_pd_targets = [], []

        with torch.no_grad():
            for batch in val_loader:
                pk_x = batch['pk_x'].to(device)
                pk_y = batch['pk_y'].to(device)
                pd_x = batch['pd_x'].to(device)
                pd_y = batch['pd_y'].to(device)

                results = model(pk_x, pd_x)

                val_pk_preds.append(results['pk'])
                val_pk_targets.append(pk_y)
                val_pd_preds.append(results['pd'])
                val_pd_targets.append(pd_y)

        val_pk_preds = torch.cat(val_pk_preds)
        val_pk_targets = torch.cat(val_pk_targets)
        val_pd_preds = torch.cat(val_pd_preds)
        val_pd_targets = torch.cat(val_pd_targets)

        val_pk_metrics = calculate_metrics(val_pk_targets, val_pk_preds)
        val_pd_metrics = calculate_metrics(val_pd_targets, val_pd_preds)

        # Update history
        history['Epoch'].append(epoch + 1)
        for k, v in train_pk_metrics.items():
            history[f'Train PK_{k}'].append(v)
        for k, v in train_pd_metrics.items():
            history[f'Train PD_{k}'].append(v)
        for k, v in val_pk_metrics.items():
            history[f'Val PK_{k}'].append(v)
        for k, v in val_pd_metrics.items():
            history[f'Val PD_{k}'].append(v)

        # Scheduler step
        scheduler.step(val_pd_metrics['MSE'])

        # Logging
        if (epoch + 1) % args.log_interval == 0:
            log_metrics(epoch + 1, "Train PK", train_pk_metrics)
            log_metrics(epoch + 1, "Train PD", train_pd_metrics)
            log_metrics(epoch + 1, "Val PK", val_pk_metrics)
            log_metrics(epoch + 1, "Val PD", val_pd_metrics)

        # Early stopping
        if not args.no_early_stopping:
            current_val_loss = val_pd_metrics['RMSE']
            if current_val_loss < best_val_loss - args.early_stopping_min_delta:
                best_val_loss = current_val_loss
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= args.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                if best_model_state is not None:
                    model.load_state_dict(best_model_state)
                break

    logger.info(f"Best Val PD RMSE: {best_val_loss:.4f}")
    return model, history


def train_gnn(model, train_data, val_data, args, device):
    """
    Train hierarchical GNN model.

    Returns:
        model: Trained model
        history: Training history dict
    """
    try:
        from torch_geometric.data import Data
        from torch_geometric.loader import DataLoader as PyGDataLoader
    except ImportError:
        raise ImportError("torch_geometric is required for GNN training. Install with: pip install torch-geometric")

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    history = {
        'Epoch': [],
        'Train PK_MSE': [], 'Train PK_RMSE': [], 'Train PK_MAE': [], 'Train PK_R2': [],
        'Train PD_MSE': [], 'Train PD_RMSE': [], 'Train PD_MAE': [], 'Train PD_R2': [],
        'Val PK_MSE': [], 'Val PK_RMSE': [], 'Val PK_MAE': [], 'Val PK_R2': [],
        'Val PD_MSE': [], 'Val PD_RMSE': [], 'Val PD_MAE': [], 'Val PD_R2': [],
    }

    train_loader = PyGDataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = PyGDataLoader(val_data, batch_size=args.batch_size, shuffle=False)

    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    criterion = nn.MSELoss()

    logger.info(f"Training GNN ({args.mode.upper()} mode)")
    logger.info(f"Epochs: {args.epochs}, Batch size: {args.batch_size}, LR: {args.learning_rate}")

    for epoch in range(args.epochs):
        model.train()

        train_pk_preds, train_pk_targets = [], []
        train_pd_preds, train_pd_targets = [], []

        for batch in train_loader:
            batch = batch.to(device)

            # Forward pass
            pd_predictions, pk_predictions = model(batch, return_pk=True)

            # Compute losses on masked nodes
            pk_preds = pk_predictions[batch.pk_mask]
            pk_tgts = batch.pk_targets[batch.pk_mask].reshape(-1, 1)
            pd_preds = pd_predictions[batch.pd_mask]
            pd_tgts = batch.pd_targets[batch.pd_mask].reshape(-1, 1)

            loss_pk = criterion(pk_preds, pk_tgts)
            loss_pd = criterion(pd_preds, pd_tgts)
            loss = args.pk_loss_weight * loss_pk + args.pd_loss_weight * loss_pd

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_pk_preds.append(pk_preds.detach().cpu())
            train_pk_targets.append(pk_tgts.detach().cpu())
            train_pd_preds.append(pd_preds.detach().cpu())
            train_pd_targets.append(pd_tgts.detach().cpu())

        # Compute training metrics
        train_pk_preds = torch.cat(train_pk_preds)
        train_pk_targets = torch.cat(train_pk_targets)
        train_pd_preds = torch.cat(train_pd_preds)
        train_pd_targets = torch.cat(train_pd_targets)

        train_pk_metrics = calculate_metrics(train_pk_targets, train_pk_preds)
        train_pd_metrics = calculate_metrics(train_pd_targets, train_pd_preds)

        # Validation
        model.eval()
        val_pk_preds, val_pk_targets = [], []
        val_pd_preds, val_pd_targets = [], []

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pd_predictions, pk_predictions = model(batch, return_pk=True)

                pk_preds = pk_predictions[batch.pk_mask]
                pk_tgts = batch.pk_targets[batch.pk_mask].reshape(-1, 1)
                pd_preds = pd_predictions[batch.pd_mask]
                pd_tgts = batch.pd_targets[batch.pd_mask].reshape(-1, 1)

                val_pk_preds.append(pk_preds.cpu())
                val_pk_targets.append(pk_tgts.cpu())
                val_pd_preds.append(pd_preds.cpu())
                val_pd_targets.append(pd_tgts.cpu())

        val_pk_preds = torch.cat(val_pk_preds)
        val_pk_targets = torch.cat(val_pk_targets)
        val_pd_preds = torch.cat(val_pd_preds)
        val_pd_targets = torch.cat(val_pd_targets)

        val_pk_metrics = calculate_metrics(val_pk_targets, val_pk_preds)
        val_pd_metrics = calculate_metrics(val_pd_targets, val_pd_preds)

        # Update history
        history['Epoch'].append(epoch + 1)
        for k, v in train_pk_metrics.items():
            history[f'Train PK_{k}'].append(v)
        for k, v in train_pd_metrics.items():
            history[f'Train PD_{k}'].append(v)
        for k, v in val_pk_metrics.items():
            history[f'Val PK_{k}'].append(v)
        for k, v in val_pd_metrics.items():
            history[f'Val PD_{k}'].append(v)

        # Logging
        if (epoch + 1) % args.log_interval == 0:
            log_metrics(epoch + 1, "Train PK", train_pk_metrics)
            log_metrics(epoch + 1, "Train PD", train_pd_metrics)
            log_metrics(epoch + 1, "Val PK", val_pk_metrics)
            log_metrics(epoch + 1, "Val PD", val_pd_metrics)

        # Early stopping
        if not args.no_early_stopping:
            current_val_loss = val_pd_metrics['RMSE']
            if current_val_loss < best_val_loss - args.early_stopping_min_delta:
                best_val_loss = current_val_loss
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= args.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                if best_model_state is not None:
                    model.load_state_dict(best_model_state)
                break

    logger.info(f"Best Val PD RMSE: {best_val_loss:.4f}")
    return model, history


# ============================================================
# Evaluation Functions
# ============================================================
def evaluate_mlp(model, data_loader, device):
    """
    Evaluate MLP model.

    Returns:
        dict with 'pk' and 'pd' metrics and predictions
    """
    model.eval()

    pk_preds, pk_targets = [], []
    pd_preds, pd_targets = [], []
    pk_metadata, pd_metadata = [], []

    with torch.no_grad():
        for batch in data_loader:
            pk_x = batch['pk_x'].to(device)
            pk_y = batch['pk_y'].to(device)
            pd_x = batch['pd_x'].to(device)
            pd_y = batch['pd_y'].to(device)

            results = model(pk_x, pd_x)

            pk_preds.append(results['pk'].cpu())
            pk_targets.append(pk_y.cpu())
            pd_preds.append(results['pd'].cpu())
            pd_targets.append(pd_y.cpu())

    pk_preds = torch.cat(pk_preds).numpy().flatten()
    pk_targets = torch.cat(pk_targets).numpy().flatten()
    pd_preds = torch.cat(pd_preds).numpy().flatten()
    pd_targets = torch.cat(pd_targets).numpy().flatten()

    pk_metrics = calculate_metrics(pk_targets, pk_preds)
    pd_metrics = calculate_metrics(pd_targets, pd_preds)

    return {
        'pk': {**pk_metrics, 'predictions': pk_preds, 'targets': pk_targets},
        'pd': {**pd_metrics, 'predictions': pd_preds, 'targets': pd_targets},
    }


def evaluate_gnn(model, data_list, device):
    """
    Evaluate GNN model.

    Returns:
        dict with 'pk' and 'pd' metrics and predictions
    """
    try:
        from torch_geometric.loader import DataLoader as PyGDataLoader
    except ImportError:
        raise ImportError("torch_geometric required")

    model.eval()
    loader = PyGDataLoader(data_list, batch_size=8, shuffle=False)

    pk_preds, pk_targets = [], []
    pd_preds, pd_targets = [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pd_predictions, pk_predictions = model(batch, return_pk=True)

            pk_preds.append(pk_predictions[batch.pk_mask].cpu())
            pk_targets.append(batch.pk_targets[batch.pk_mask].cpu())
            pd_preds.append(pd_predictions[batch.pd_mask].cpu())
            pd_targets.append(batch.pd_targets[batch.pd_mask].cpu())

    pk_preds = torch.cat(pk_preds).numpy().flatten()
    pk_targets = torch.cat(pk_targets).numpy().flatten()
    pd_preds = torch.cat(pd_preds).numpy().flatten()
    pd_targets = torch.cat(pd_targets).numpy().flatten()

    pk_metrics = calculate_metrics(pk_targets, pk_preds)
    pd_metrics = calculate_metrics(pd_targets, pd_preds)

    return {
        'pk': {**pk_metrics, 'predictions': pk_preds, 'targets': pk_targets},
        'pd': {**pd_metrics, 'predictions': pd_preds, 'targets': pd_targets},
    }


# ============================================================
# GNN Data Preparation
# ============================================================
def prepare_gnn_data(args):
    """
    Prepare graph data for GNN training.

    Returns:
        dict with train_data, val_data, test_data, feature_dim
    """
    try:
        from torch_geometric.data import Data
    except ImportError:
        raise ImportError("torch_geometric required for GNN")

    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    logger.info("Preparing GNN graph data...")

    df = pd.read_csv(args.csv_path)
    df.columns = [c.strip().upper() for c in df.columns]

    # Filter observations
    df = df[df['EVID'] == 0].copy()
    if 'MDV' in df.columns:
        df = df[df['MDV'] == 0]

    logger.info(f"Total observations: {len(df)}")

    # Feature engineering
    base_features = ['TIME', 'BW', 'DOSE', 'COMED']
    if args.add_decay:
        for hl in args.half_lives:
            df[f'DECAY_HL{hl}h'] = np.exp(-np.log(2) / hl * df['TIME'])
            base_features.append(f'DECAY_HL{hl}h')

    df['TIME_LOG'] = np.log1p(df['TIME'])
    df['TIME_SQUARED'] = df['TIME'] ** 2
    base_features.extend(['TIME_LOG', 'TIME_SQUARED'])

    # Create graphs per patient
    graphs = []
    all_features = []

    for patient_id in df['ID'].unique():
        patient_df = df[df['ID'] == patient_id].sort_values('TIME').reset_index(drop=True)

        pk_obs = patient_df[patient_df['DVID'] == 1]
        pd_obs = patient_df[patient_df['DVID'] == 2]

        if len(pk_obs) == 0 or len(pd_obs) == 0:
            continue

        # Node features and targets
        node_features = []
        pk_targets, pd_targets = [], []
        node_types = []
        times = []

        pk_indices, pd_indices = [], []
        node_idx = 0

        # Add PK nodes
        for _, row in pk_obs.iterrows():
            features = [row[f] for f in base_features]
            features.append(row['DV'])  # Add PK value
            node_features.append(features)
            pk_targets.append(row['DV'])
            pd_targets.append(0)
            node_types.append(0)
            times.append(row['TIME'])
            pk_indices.append(node_idx)
            node_idx += 1

        # Add PD nodes
        for _, row in pd_obs.iterrows():
            features = [row[f] for f in base_features]
            # Use most recent PK value
            recent_pk = pk_obs[pk_obs['TIME'] <= row['TIME']]
            pk_val = recent_pk.iloc[-1]['DV'] if len(recent_pk) > 0 else 0.0
            features.append(pk_val)
            node_features.append(features)
            pk_targets.append(0)
            pd_targets.append(row['DV'])
            node_types.append(1)
            times.append(row['TIME'])
            pd_indices.append(node_idx)
            node_idx += 1

        # Create edges
        edges = []
        edge_weights = []
        times_arr = np.array(times)

        # Temporal edges within PK nodes
        for i in range(len(pk_indices) - 1):
            src, dst = pk_indices[i], pk_indices[i + 1]
            time_diff = abs(times_arr[dst] - times_arr[src])
            weight = np.exp(-time_diff / 24.0)
            edges.extend([[src, dst], [dst, src]])
            edge_weights.extend([weight, weight])

        # Temporal edges within PD nodes
        for i in range(len(pd_indices) - 1):
            src, dst = pd_indices[i], pd_indices[i + 1]
            time_diff = abs(times_arr[dst] - times_arr[src])
            weight = np.exp(-time_diff / 24.0)
            edges.extend([[src, dst], [dst, src]])
            edge_weights.extend([weight, weight])

        # PK-PD edges
        for pd_idx in pd_indices:
            pd_time = times_arr[pd_idx]
            for pk_idx in pk_indices:
                if times_arr[pk_idx] <= pd_time:
                    time_diff = pd_time - times_arr[pk_idx]
                    weight = np.exp(-time_diff / 12.0)
                    edges.extend([[pk_idx, pd_idx], [pd_idx, pk_idx]])
                    edge_weights.extend([weight, weight])

        node_features = np.array(node_features, dtype=np.float32)
        all_features.append(node_features)

        graphs.append({
            'patient_id': patient_id,
            'node_features': node_features,
            'edge_index': np.array(edges, dtype=np.int64).T if edges else np.array([[], []], dtype=np.int64),
            'edge_weights': np.array(edge_weights, dtype=np.float32),
            'pk_targets': np.array(pk_targets, dtype=np.float32),
            'pd_targets': np.array(pd_targets, dtype=np.float32),
            'pk_indices': pk_indices,
            'pd_indices': pd_indices,
            'times': np.array(times, dtype=np.float32),
        })

    logger.info(f"Created {len(graphs)} patient graphs")

    # Scale features
    scaler = StandardScaler()
    all_features_concat = np.vstack(all_features)
    scaler.fit(all_features_concat)

    for g in graphs:
        g['node_features'] = scaler.transform(g['node_features'])

    # Split
    indices = list(range(len(graphs)))
    train_indices, test_indices = train_test_split(indices, test_size=args.test_size, random_state=args.random_seed)
    train_indices, val_indices = train_test_split(train_indices, test_size=args.val_size / (1 - args.test_size), random_state=args.random_seed)

    # Convert to PyG Data objects
    def to_pyg_data(graph):
        x = torch.FloatTensor(graph['node_features'])
        edge_index = torch.LongTensor(graph['edge_index'])
        edge_weight = torch.FloatTensor(graph['edge_weights'])

        pk_mask = torch.zeros(len(graph['node_features']), dtype=torch.bool)
        pk_mask[graph['pk_indices']] = True
        pd_mask = torch.zeros(len(graph['node_features']), dtype=torch.bool)
        pd_mask[graph['pd_indices']] = True

        return Data(
            x=x,
            edge_index=edge_index,
            edge_weight=edge_weight,
            pk_targets=torch.FloatTensor(graph['pk_targets']),
            pd_targets=torch.FloatTensor(graph['pd_targets']),
            pk_mask=pk_mask,
            pd_mask=pd_mask,
            patient_id=graph['patient_id'],
            times=torch.FloatTensor(graph['times']),
        )

    train_data = [to_pyg_data(graphs[i]) for i in train_indices]
    val_data = [to_pyg_data(graphs[i]) for i in val_indices]
    test_data = [to_pyg_data(graphs[i]) for i in test_indices]

    logger.info(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    return {
        'train_data': train_data,
        'val_data': val_data,
        'test_data': test_data,
        'feature_dim': all_features_concat.shape[1],
    }


# ============================================================
# Visualization
# ============================================================
def plot_predictions(train_results, test_results, save_dir, model_name):
    """Plot scatter plots and save."""
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # PK - Train
    axes[0, 0].scatter(train_results['pk']['targets'], train_results['pk']['predictions'],
                       alpha=0.6, edgecolors='k', linewidth=0.5)
    min_val = min(train_results['pk']['targets'].min(), train_results['pk']['predictions'].min())
    max_val = max(train_results['pk']['targets'].max(), train_results['pk']['predictions'].max())
    axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual PK')
    axes[0, 0].set_ylabel('Predicted PK')
    axes[0, 0].set_title(f'PK Train (R2={train_results["pk"]["R2"]:.4f}, RMSE={train_results["pk"]["RMSE"]:.4f})')
    axes[0, 0].grid(True, alpha=0.3)

    # PK - Test
    axes[0, 1].scatter(test_results['pk']['targets'], test_results['pk']['predictions'],
                       alpha=0.6, edgecolors='k', linewidth=0.5, color='orange')
    min_val = min(test_results['pk']['targets'].min(), test_results['pk']['predictions'].min())
    max_val = max(test_results['pk']['targets'].max(), test_results['pk']['predictions'].max())
    axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    axes[0, 1].set_xlabel('Actual PK')
    axes[0, 1].set_ylabel('Predicted PK')
    axes[0, 1].set_title(f'PK Test (R2={test_results["pk"]["R2"]:.4f}, RMSE={test_results["pk"]["RMSE"]:.4f})')
    axes[0, 1].grid(True, alpha=0.3)

    # PD - Train
    axes[1, 0].scatter(train_results['pd']['targets'], train_results['pd']['predictions'],
                       alpha=0.6, edgecolors='k', linewidth=0.5)
    min_val = min(train_results['pd']['targets'].min(), train_results['pd']['predictions'].min())
    max_val = max(train_results['pd']['targets'].max(), train_results['pd']['predictions'].max())
    axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    axes[1, 0].set_xlabel('Actual PD')
    axes[1, 0].set_ylabel('Predicted PD')
    axes[1, 0].set_title(f'PD Train (R2={train_results["pd"]["R2"]:.4f}, RMSE={train_results["pd"]["RMSE"]:.4f})')
    axes[1, 0].grid(True, alpha=0.3)

    # PD - Test
    axes[1, 1].scatter(test_results['pd']['targets'], test_results['pd']['predictions'],
                       alpha=0.6, edgecolors='k', linewidth=0.5, color='orange')
    min_val = min(test_results['pd']['targets'].min(), test_results['pd']['predictions'].min())
    max_val = max(test_results['pd']['targets'].max(), test_results['pd']['predictions'].max())
    axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    axes[1, 1].set_xlabel('Actual PD')
    axes[1, 1].set_ylabel('Predicted PD')
    axes[1, 1].set_title(f'PD Test (R2={test_results["pd"]["R2"]:.4f}, RMSE={test_results["pd"]["RMSE"]:.4f})')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'predictions_scatter.png'), dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved prediction plots to {save_dir}")


def plot_patient_timeseries(pk_data, pd_data, model, device, save_dir, model_name,
                            patient_ids=None):
    """
    Plot time series for selected patients.

    Args:
        pk_data: dict with 'X', 'y', 'ids', 'times' for PK
        pd_data: dict with 'X', 'y', 'ids', 'times' for PD
        model: trained model
        device: torch device
        save_dir: directory to save plots
        model_name: name for plot title
        patient_ids: list of patient IDs to plot (default: [9, 13, 26, 46])
    """
    os.makedirs(save_dir, exist_ok=True)

    if patient_ids is None:
        # Default: No dose (9), Dose 1 (13), Dose 3 (26), Dose 10 (46)
        patient_ids = [9, 13, 26, 46]

    # Filter to available patients (only need PD data, PK can be empty)
    available_pk_ids = set(pk_data['ids'])
    available_pd_ids = set(pd_data['ids'])
    patient_ids = [pid for pid in patient_ids if pid in available_pd_ids]

    if len(patient_ids) == 0:
        logger.warning("No specified patients found in data. Using first 4 available.")
        patient_ids = list(available_pd_ids)[:4]

    n_patients = len(patient_ids)
    if n_patients == 0:
        logger.warning("No patients available for plotting")
        return

    # Create 4x2 figure (patients x [PK, PD])
    fig, axes = plt.subplots(n_patients, 2, figsize=(14, 4 * n_patients))
    if n_patients == 1:
        axes = axes.reshape(1, -1)

    model.eval()

    for idx, patient_id in enumerate(patient_ids):
        # Get PK data for this patient
        pk_mask = pk_data['ids'] == patient_id
        pk_X = pk_data['X'][pk_mask]
        pk_y = pk_data['y'][pk_mask]
        pk_times = pk_data['times'][pk_mask]

        # Get PD data for this patient
        pd_mask = pd_data['ids'] == patient_id
        pd_X = pd_data['X'][pd_mask]
        pd_y = pd_data['y'][pd_mask]
        pd_times = pd_data['times'][pd_mask]

        # Plot PK (first column)
        ax_pk = axes[idx, 0]

        if len(pk_X) > 0:
            # Sort by time
            pk_order = np.argsort(pk_times)
            pk_times = pk_times[pk_order]
            pk_y = pk_y[pk_order]
            pk_X = pk_X[pk_order]

            # Get PK predictions
            with torch.no_grad():
                pk_X_tensor = torch.FloatTensor(pk_X).to(device)
                pk_results = model(pk_X_tensor, None)
                pk_pred = pk_results['pk'].cpu().numpy().flatten()

            ax_pk.plot(pk_times, pk_y, 'o-', label='Actual PK', markersize=6, linewidth=2, color='blue')
            ax_pk.plot(pk_times, pk_pred, 's--', label='Predicted PK', markersize=6, linewidth=2, color='orange')
            ax_pk.set_xlabel('Time (hours)')
            ax_pk.set_ylabel('PK Value')
            ax_pk.legend()
        else:
            # No PK data - leave blank with message
            ax_pk.text(0.5, 0.5, 'No PK data\n(No dose)', transform=ax_pk.transAxes,
                      ha='center', va='center', fontsize=14, color='gray')
            ax_pk.set_xlabel('Time (hours)')
            ax_pk.set_ylabel('PK Value')

        ax_pk.set_title(f'Patient {patient_id} - PK')
        ax_pk.grid(True, alpha=0.3)

        # Plot PD (second column)
        ax_pd = axes[idx, 1]

        if len(pd_X) > 0:
            # Sort by time
            pd_order = np.argsort(pd_times)
            pd_times = pd_times[pd_order]
            pd_y = pd_y[pd_order]
            pd_X = pd_X[pd_order]

            # Get PD predictions
            with torch.no_grad():
                pd_X_tensor = torch.FloatTensor(pd_X).to(device)
                pd_results = model(None, pd_X_tensor)
                pd_pred = pd_results['pd'].cpu().numpy().flatten()

            ax_pd.plot(pd_times, pd_y, 'o-', label='Actual PD', markersize=6, linewidth=2, color='blue')
            ax_pd.plot(pd_times, pd_pred, 's--', label='Predicted PD', markersize=6, linewidth=2, color='orange')
            ax_pd.legend()
        else:
            ax_pd.text(0.5, 0.5, 'No PD data', transform=ax_pd.transAxes,
                      ha='center', va='center', fontsize=14, color='gray')

        ax_pd.set_xlabel('Time (hours)')
        ax_pd.set_ylabel('PD Value')
        ax_pd.set_title(f'Patient {patient_id} - PD')
        ax_pd.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'patient_timeseries.png'), dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved patient time series plots to {save_dir}")


def plot_gnn_patient_timeseries(all_data, model, device, save_dir, model_name, patient_ids=None):
    """
    Plot time series for selected patients (GNN version).
    """
    os.makedirs(save_dir, exist_ok=True)

    if patient_ids is None:
        patient_ids = [9, 13, 26, 46]

    # Filter graphs
    graphs_to_plot = []
    for g in all_data:
        pid = g.patient_id
        if torch.is_tensor(pid):
            pid = pid.item()
        if pid in patient_ids:
            graphs_to_plot.append(g)

    if not graphs_to_plot:
        logger.warning("No specified patients found in GNN data for plotting.")
        return

    n_patients = len(graphs_to_plot)
    fig, axes = plt.subplots(n_patients, 2, figsize=(14, 4 * n_patients))
    if n_patients == 1:
        axes = axes.reshape(1, -1)

    model.eval()

    for idx, graph in enumerate(graphs_to_plot):
        pid = graph.patient_id
        if torch.is_tensor(pid):
            pid = pid.item()

        batch = graph.to(device)
        
        with torch.no_grad():
            pd_pred, pk_pred = model(batch, return_pk=True)

        # PK Plot
        ax_pk = axes[idx, 0]
        pk_mask = batch.pk_mask.cpu().numpy().astype(bool)
        
        if pk_mask.any():
            pk_times = batch.times[pk_mask].cpu().numpy()
            pk_y = batch.pk_targets[pk_mask].cpu().numpy()
            pk_p = pk_pred[pk_mask].cpu().numpy().flatten()
            
            sort_idx = np.argsort(pk_times)
            ax_pk.plot(pk_times[sort_idx], pk_y[sort_idx], 'o-', label='Actual PK', color='blue')
            ax_pk.plot(pk_times[sort_idx], pk_p[sort_idx], 's--', label='Predicted PK', color='orange')
            ax_pk.legend()
        else:
            ax_pk.text(0.5, 0.5, 'No PK data', transform=ax_pk.transAxes, ha='center')
            
        ax_pk.set_title(f'Patient {pid} - PK')
        ax_pk.set_xlabel('Time (hours)')
        ax_pk.set_ylabel('PK Value')
        ax_pk.grid(True, alpha=0.3)

        # PD Plot
        ax_pd = axes[idx, 1]
        pd_mask = batch.pd_mask.cpu().numpy().astype(bool)
        
        if pd_mask.any():
            pd_times = batch.times[pd_mask].cpu().numpy()
            pd_y = batch.pd_targets[pd_mask].cpu().numpy()
            pd_p = pd_pred[pd_mask].cpu().numpy().flatten()
            
            sort_idx = np.argsort(pd_times)
            ax_pd.plot(pd_times[sort_idx], pd_y[sort_idx], 'o-', label='Actual PD', color='blue')
            ax_pd.plot(pd_times[sort_idx], pd_p[sort_idx], 's--', label='Predicted PD', color='orange')
            ax_pd.legend()
        else:
            ax_pd.text(0.5, 0.5, 'No PD data', transform=ax_pd.transAxes, ha='center')

        ax_pd.set_title(f'Patient {pid} - PD')
        ax_pd.set_xlabel('Time (hours)')
        ax_pd.set_ylabel('PD Value')
        ax_pd.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'patient_timeseries.png'), dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved GNN patient time series plots to {save_dir}")


# ============================================================
# Main
# ============================================================
def main():
    # Get arguments
    args = get_args()
    print_args(args)

    # Create save directory
    timestamp = time.strftime('%y_%m_%d_%H_%M_%S')
    save_dir = os.path.join(args.save_dir, f"{args.experiment_name}_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    # Set device
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # Set random seed
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    # ==================== Model Selection ====================
    if args.model == 'mlp':
        logger.info("=" * 60)
        logger.info("TRAINING HIERARCHICAL MLP")
        logger.info("=" * 60)

        # Prepare MLP data
        if args.combine:
            # Use all data for training (no split)
            data = prepare_pkpd_data(
                csv_path=args.csv_path,
                test_size=args.test_size,
                val_size=args.val_size,
                random_state=args.random_seed,
                use_perkg=args.use_perkg,
                time_windows=args.time_windows,
                half_lives=args.half_lives,
                add_decay=args.add_decay,
                stratified_split=False,
            )
            logger.info("COMBINE MODE: Using ALL data for training")
            # Use same data for train/val/test
            data['val_pk'] = data['train_pk']
            data['val_pd'] = data['train_pd']
            data['test_pk'] = data['train_pk']
            data['test_pd'] = data['train_pd']
        else:
            data = prepare_pkpd_data(
                csv_path=args.csv_path,
                test_size=args.test_size,
                val_size=args.val_size,
                random_state=args.random_seed,
                use_perkg=args.use_perkg,
                time_windows=args.time_windows,
                half_lives=args.half_lives,
                add_decay=args.add_decay,
                stratified_split=args.stratified_split,
            )

        # Create datasets
        train_dataset = PKPDDataset(data['train_pk'], data['train_pd'])
        val_dataset = PKPDDataset(data['val_pk'], data['val_pd'])
        test_dataset = PKPDDataset(data['test_pk'], data['test_pd'])

        train_loader = TorchDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_pkpd)
        val_loader = TorchDataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_pkpd)
        test_loader = TorchDataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_pkpd)

        # Create model
        model = HierarchicalPKPDMLP(
            mode=args.mode,
            pk_input_dim=data['n_features'],
            pd_input_dim=data['n_features'],
            hidden_dim=args.hidden_dim,
            n_blocks=args.n_blocks,
            dropout=args.dropout,
            head_hidden=args.head_hidden,
        )

        logger.info(f"Model: {args.model.upper()}, Mode: {args.mode}")
        logger.info(f"Input dim: {data['n_features']}, Hidden dim: {args.hidden_dim}")
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")

        # Train
        model, history = train_mlp(model, train_loader, val_loader, args, device)

        # Evaluate
        logger.info("=" * 60)
        logger.info("EVALUATION")
        logger.info("=" * 60)

        train_results = evaluate_mlp(model, train_loader, device)
        test_results = evaluate_mlp(model, test_loader, device)

    elif args.model == 'gnn':
        logger.info("=" * 60)
        logger.info("TRAINING HIERARCHICAL GNN")
        logger.info("=" * 60)

        # Prepare GNN data
        data = prepare_gnn_data(args)

        # Create model
        model = HierarchicalPKPDGNN(
            feature_dim=data['feature_dim'],
            hidden_dim=args.gnn_hidden_dim,
            num_layers_pk=args.num_layers_pk,
            num_layers_pd=args.num_layers_pd,
            dropout=args.dropout,
            use_attention=args.use_attention,
            use_gating=args.use_gating,
        )


        logger.info(f"Model: {args.model.upper()}, Mode: {args.mode}")
        logger.info(f"Feature dim: {data['feature_dim']}, Hidden dim: {args.gnn_hidden_dim}")
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")

        # Train
        model, history = train_gnn(model, data['train_data'], data['val_data'], args, device)

        # Evaluate
        logger.info("=" * 60)
        logger.info("EVALUATION")
        logger.info("=" * 60)

        train_results = evaluate_gnn(model, data['train_data'], device)
        test_results = evaluate_gnn(model, data['test_data'], device)
    
    elif args.model == 'hqgnn':
        logger.info("=" * 60)
        logger.info("TRAINING HIERARCHICAL GNN")
        logger.info("=" * 60)

        # Prepare GNN data
        data = prepare_gnn_data(args)

        # Create model
        model = HQGNN(
            feature_dim=data['feature_dim'],
            hidden_dim=args.gnn_hidden_dim,
            num_layers_pk=args.num_layers_pk,
            num_layers_pd=args.num_layers_pd,
            dropout=args.dropout,
            use_attention=args.use_attention,
            use_gating=args.use_gating,
        )


        logger.info(f"Model: {args.model.upper()}, Mode: {args.mode}")
        logger.info(f"Feature dim: {data['feature_dim']}, Hidden dim: {args.gnn_hidden_dim}")
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")

        # Train
        model, history = train_gnn(model, data['train_data'], data['val_data'], args, device)

        # Evaluate
        logger.info("=" * 60)
        logger.info("EVALUATION")
        logger.info("=" * 60)

        train_results = evaluate_gnn(model, data['train_data'], device)
        test_results = evaluate_gnn(model, data['test_data'], device)

    elif args.model == 'hqcnn':
        logger.info("=" * 60)
        logger.info("TRAINING HIERARCHICAL HQCNN (Quantum)")
        logger.info("=" * 60)

        # Prepare data (same as MLP)
        if args.combine:
            data = prepare_pkpd_data(
                csv_path=args.csv_path,
                test_size=args.test_size,
                val_size=args.val_size,
                random_state=args.random_seed,
                use_perkg=args.use_perkg,
                time_windows=args.time_windows,
                half_lives=args.half_lives,
                add_decay=args.add_decay,
                stratified_split=False,
            )
            logger.info("COMBINE MODE: Using ALL data for training")
            data['val_pk'] = data['train_pk']
            data['val_pd'] = data['train_pd']
            data['test_pk'] = data['train_pk']
            data['test_pd'] = data['train_pd']
        else:
            data = prepare_pkpd_data(
                csv_path=args.csv_path,
                test_size=args.test_size,
                val_size=args.val_size,
                random_state=args.random_seed,
                use_perkg=args.use_perkg,
                time_windows=args.time_windows,
                half_lives=args.half_lives,
                add_decay=args.add_decay,
                stratified_split=args.stratified_split,
            )

        # Create datasets
        train_dataset = PKPDDataset(data['train_pk'], data['train_pd'])
        val_dataset = PKPDDataset(data['val_pk'], data['val_pd'])
        test_dataset = PKPDDataset(data['test_pk'], data['test_pd'])

        train_loader = TorchDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_pkpd)
        val_loader = TorchDataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_pkpd)
        test_loader = TorchDataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_pkpd)

        # Create HQCNN model
        model = HierarchicalHQCNN(
            pk_input_dim=data['n_features'],
            pd_input_dim=data['n_features'],
            num_layers=args.hqcnn_num_layers,
            mode=args.mode,
        )

        logger.info(f"Model: {args.model.upper()}, Mode: {args.mode}")
        logger.info(f"Input dim: {data['n_features']}, Quantum layers: {args.hqcnn_num_layers}")
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")

        # Train (reuse MLP training function)
        model, history = train_mlp(model, train_loader, val_loader, args, device)

        # Evaluate
        logger.info("=" * 60)
        logger.info("EVALUATION")
        logger.info("=" * 60)

        train_results = evaluate_mlp(model, train_loader, device)
        test_results = evaluate_mlp(model, test_loader, device)

    else:
        raise ValueError(f"Unsupported model: {args.model}")

    # ==================== Print Final Results ====================
    logger.info("=" * 60)
    logger.info("FINAL RESULTS")
    logger.info("=" * 60)

    logger.info("PK Metrics:")
    logger.info(f"  Train - MSE: {train_results['pk']['MSE']:.4f}, RMSE: {train_results['pk']['RMSE']:.4f}, MAE: {train_results['pk']['MAE']:.4f}, R2: {train_results['pk']['R2']:.4f}")
    logger.info(f"  Test  - MSE: {test_results['pk']['MSE']:.4f}, RMSE: {test_results['pk']['RMSE']:.4f}, MAE: {test_results['pk']['MAE']:.4f}, R2: {test_results['pk']['R2']:.4f}")

    logger.info("PD Metrics:")
    logger.info(f"  Train - MSE: {train_results['pd']['MSE']:.4f}, RMSE: {train_results['pd']['RMSE']:.4f}, MAE: {train_results['pd']['MAE']:.4f}, R2: {train_results['pd']['R2']:.4f}")
    logger.info(f"  Test  - MSE: {test_results['pd']['MSE']:.4f}, RMSE: {test_results['pd']['RMSE']:.4f}, MAE: {test_results['pd']['MAE']:.4f}, R2: {test_results['pd']['R2']:.4f}")

    # ==================== Save Results ====================

    # Save model
    if args.save_model:
        model_path = os.path.join(save_dir, 'model.pth')
        torch.save(model.state_dict(), model_path)
        logger.info(f"Saved model to {model_path}")

    # Save metrics to txt
    metrics_path = os.path.join(save_dir, 'metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write(f"EXPERIMENT: {args.experiment_name}\n")
        f.write(f"Model: {args.model.upper()}, Mode: {args.mode}\n")
        f.write("=" * 60 + "\n\n")

        f.write("Configuration:\n")
        f.write(f"  Hidden dim: {args.hidden_dim if args.model == 'mlp' else args.gnn_hidden_dim}\n")
        f.write(f"  Dropout: {args.dropout}\n")
        f.write(f"  Learning rate: {args.learning_rate}\n")
        f.write(f"  Batch size: {args.batch_size}\n")
        f.write(f"  Epochs: {args.epochs}\n\n")

        f.write("PK Metrics:\n")
        f.write(f"  Train - MSE: {train_results['pk']['MSE']:.4f}, RMSE: {train_results['pk']['RMSE']:.4f}, MAE: {train_results['pk']['MAE']:.4f}, R2: {train_results['pk']['R2']:.4f}\n")
        f.write(f"  Test  - MSE: {test_results['pk']['MSE']:.4f}, RMSE: {test_results['pk']['RMSE']:.4f}, MAE: {test_results['pk']['MAE']:.4f}, R2: {test_results['pk']['R2']:.4f}\n\n")

        f.write("PD Metrics:\n")
        f.write(f"  Train - MSE: {train_results['pd']['MSE']:.4f}, RMSE: {train_results['pd']['RMSE']:.4f}, MAE: {train_results['pd']['MAE']:.4f}, R2: {train_results['pd']['R2']:.4f}\n")
        f.write(f"  Test  - MSE: {test_results['pd']['MSE']:.4f}, RMSE: {test_results['pd']['RMSE']:.4f}, MAE: {test_results['pd']['MAE']:.4f}, R2: {test_results['pd']['R2']:.4f}\n")

    logger.info(f"Saved metrics to {metrics_path}")

    # Save plots
    if args.save_plots:
        # Training history
        plot_metrics(history, save_path=os.path.join(save_dir, 'training_history.png'))

        # Prediction scatter plots
        plot_predictions(train_results, test_results, save_dir, f"{args.model}_{args.mode}")

        # Patient time series (for MLP and HQCNN - need raw data with IDs)
        if args.model in ['mlp', 'hqcnn']:
            # Combine train and test data for patient plots
            all_pk_data = {
                'X': np.vstack([data['train_pk']['X'], data['val_pk']['X'], data['test_pk']['X']]),
                'y': np.concatenate([data['train_pk']['y'], data['val_pk']['y'], data['test_pk']['y']]),
                'ids': np.concatenate([data['train_pk']['ids'], data['val_pk']['ids'], data['test_pk']['ids']]),
                'times': np.concatenate([data['train_pk']['times'], data['val_pk']['times'], data['test_pk']['times']]),
            }
            all_pd_data = {
                'X': np.vstack([data['train_pd']['X'], data['val_pd']['X'], data['test_pd']['X']]),
                'y': np.concatenate([data['train_pd']['y'], data['val_pd']['y'], data['test_pd']['y']]),
                'ids': np.concatenate([data['train_pd']['ids'], data['val_pd']['ids'], data['test_pd']['ids']]),
                'times': np.concatenate([data['train_pd']['times'], data['val_pd']['times'], data['test_pd']['times']]),
            }
            plot_patient_timeseries(all_pk_data, all_pd_data, model, device,
                                    save_dir, f"{args.model}_{args.mode}",
                                    patient_ids=[9, 13, 26, 46])
        elif args.model in ['gnn', 'hqgnn']:
            all_gnn_data = data['train_data'] + data['val_data'] + data['test_data']
            plot_gnn_patient_timeseries(all_gnn_data, model, device,
                                        save_dir, f"{args.model}_{args.mode}",
                                        patient_ids=[9, 13, 26, 46])

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE!")
    logger.info(f"Results saved to: {save_dir}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
