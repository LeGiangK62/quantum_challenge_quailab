import os
import torch
import torch.nn as nn
import torch.optim as optim
from Utils.log import calculate_metrics, log_metrics, plot_metrics, logger

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
        if (epoch + 0) % args.log_interval == 0:
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
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=25, factor=0.5)

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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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

        # Step LR scheduler based on val PD RMSE
        scheduler.step(val_pd_metrics['RMSE'])

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
        if (epoch + 0) % args.log_interval == 0:
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


# ============================================================
# V2: Training & Evaluation with Target Transforms (log_pk, sqrt_pd)
# ============================================================
def train_mlp_v2(model, train_loader, val_loader, args, device, pk_transform='none', pd_transform='none'):
    """
    Train hierarchical MLP with target transforms.
    Loss is computed in transformed space; logged metrics are in original scale.
    """
    from Utils.data_loader import inverse_target_transform

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

    logger.info(f"Training MLP V2 ({args.mode.upper()} mode)")
    logger.info(f"PK transform: {pk_transform}, PD transform: {pd_transform}")
    logger.info(f"Epochs: {args.epochs}, Batch size: {args.batch_size}, LR: {args.learning_rate}")
    logger.info(f"PK loss: {args.loss_type_pk}, PD loss: {args.loss_type_pd}")

    def inv_pk(t):
        return torch.tensor(inverse_target_transform(t.numpy(), pk_transform), dtype=torch.float32)

    def inv_pd(t):
        return torch.tensor(inverse_target_transform(t.numpy(), pd_transform), dtype=torch.float32)

    for epoch in range(args.epochs):
        model.train()

        train_pk_preds, train_pk_targets = [], []
        train_pd_preds, train_pd_targets = [], []

        for batch in train_loader:
            pk_x = batch['pk_x'].to(device)
            pk_y = batch['pk_y'].to(device)
            pd_x = batch['pd_x'].to(device)
            pd_y = batch['pd_y'].to(device)

            results = model(pk_x, pd_x)

            loss_pk = compute_loss(results['pk'], pk_y, args.loss_type_pk,
                                   args.quantile_q, args.hybrid_lambda)
            loss_pd = compute_loss(results['pd'], pd_y, args.loss_type_pd,
                                   args.quantile_q, args.hybrid_lambda)
            loss = args.pk_loss_weight * loss_pk + args.pd_loss_weight * loss_pd

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_pk_preds.append(results['pk'].detach().cpu())
            train_pk_targets.append(pk_y.detach().cpu())
            train_pd_preds.append(results['pd'].detach().cpu())
            train_pd_targets.append(pd_y.detach().cpu())

        # Inverse transform for metrics in original scale
        train_pk_preds_orig = inv_pk(torch.cat(train_pk_preds))
        train_pk_targets_orig = inv_pk(torch.cat(train_pk_targets))
        train_pd_preds_orig = inv_pd(torch.cat(train_pd_preds))
        train_pd_targets_orig = inv_pd(torch.cat(train_pd_targets))

        train_pk_metrics = calculate_metrics(train_pk_targets_orig, train_pk_preds_orig)
        train_pd_metrics = calculate_metrics(train_pd_targets_orig, train_pd_preds_orig)

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

                val_pk_preds.append(results['pk'].cpu())
                val_pk_targets.append(pk_y.cpu())
                val_pd_preds.append(results['pd'].cpu())
                val_pd_targets.append(pd_y.cpu())

        val_pk_preds_orig = inv_pk(torch.cat(val_pk_preds))
        val_pk_targets_orig = inv_pk(torch.cat(val_pk_targets))
        val_pd_preds_orig = inv_pd(torch.cat(val_pd_preds))
        val_pd_targets_orig = inv_pd(torch.cat(val_pd_targets))

        val_pk_metrics = calculate_metrics(val_pk_targets_orig, val_pk_preds_orig)
        val_pd_metrics = calculate_metrics(val_pd_targets_orig, val_pd_preds_orig)

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

        scheduler.step(val_pd_metrics['MSE'])

        if (epoch + 0) % args.log_interval == 0:
            log_metrics(epoch + 1, "Train PK", train_pk_metrics)
            log_metrics(epoch + 1, "Train PD", train_pd_metrics)
            log_metrics(epoch + 1, "Val PK", val_pk_metrics)
            log_metrics(epoch + 1, "Val PD", val_pd_metrics)


        # Early stopping (on original scale RMSE)
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

    logger.info(f"Best Val PD RMSE (original scale): {best_val_loss:.4f}")
    return model, history


def evaluate_mlp_v2(model, data_loader, device, pk_transform='none', pd_transform='none'):
    """
    Evaluate MLP model with inverse-transform to original scale.
    """
    from Utils.data_loader import inverse_target_transform
    import numpy as np

    model.eval()

    pk_preds, pk_targets = [], []
    pd_preds, pd_targets = [], []

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

    # Inverse transform to original scale
    pk_preds = inverse_target_transform(torch.cat(pk_preds).numpy().flatten(), pk_transform)
    pk_targets = inverse_target_transform(torch.cat(pk_targets).numpy().flatten(), pk_transform)
    pd_preds = inverse_target_transform(torch.cat(pd_preds).numpy().flatten(), pd_transform)
    pd_targets = inverse_target_transform(torch.cat(pd_targets).numpy().flatten(), pd_transform)

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
# LSTM Training and Evaluation
# ============================================================
def train_lstm(model, data, args, device):
    """
    Train hierarchical LSTM model.
    """
    from Utils.data_loader import collate_lstm_batch
    import random

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

    train_seqs = data['train_sequences']
    val_seqs = data['val_sequences']

    logger.info(f"Training LSTM ({args.mode.upper()} mode)")
    logger.info(f"Epochs: {args.epochs}, Batch size: {args.batch_size}, LR: {args.learning_rate}")

    for epoch in range(args.epochs):
        model.train()
        shuffled_seqs = train_seqs.copy()
        random.shuffle(shuffled_seqs)

        train_pk_preds, train_pk_targets = [], []
        train_pd_preds, train_pd_targets = [], []

        for i in range(0, len(shuffled_seqs), args.batch_size):
            batch_seqs = shuffled_seqs[i:i + args.batch_size]
            batch = collate_lstm_batch(batch_seqs, device)

            if 'X_pk' in batch and 'X_pd' in batch:
                results = model(
                    x_pk=batch['X_pk'], x_pd=batch['X_pd'],
                    lengths_pk=batch['lengths_pk'], lengths_pd=batch['lengths_pd']
                )

                pk_pred = results['pk'].squeeze(-1)[batch['mask_pk']]
                pk_tgt = batch['y_pk'][batch['mask_pk']]
                pd_pred = results['pd'].squeeze(-1)[batch['mask_pd']]
                pd_tgt = batch['y_pd'][batch['mask_pd']]

                loss_pk = compute_loss(pk_pred, pk_tgt, args.loss_type_pk, args.quantile_q, args.hybrid_lambda)
                loss_pd = compute_loss(pd_pred, pd_tgt, args.loss_type_pd, args.quantile_q, args.hybrid_lambda)
                loss = args.pk_loss_weight * loss_pk + args.pd_loss_weight * loss_pd

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                train_pk_preds.append(pk_pred.detach().cpu())
                train_pk_targets.append(pk_tgt.detach().cpu())
                train_pd_preds.append(pd_pred.detach().cpu())
                train_pd_targets.append(pd_tgt.detach().cpu())

        if train_pk_preds:
            train_pk_metrics = calculate_metrics(torch.cat(train_pk_targets), torch.cat(train_pk_preds))
            train_pd_metrics = calculate_metrics(torch.cat(train_pd_targets), torch.cat(train_pd_preds))
        else:
            train_pk_metrics = train_pd_metrics = {'MSE': 0, 'RMSE': 0, 'MAE': 0, 'R2': 0}

        # Validation
        model.eval()
        val_pk_preds, val_pk_targets = [], []
        val_pd_preds, val_pd_targets = [], []

        with torch.no_grad():
            for i in range(0, len(val_seqs), args.batch_size):
                batch_seqs = val_seqs[i:i + args.batch_size]
                batch = collate_lstm_batch(batch_seqs, device)

                if 'X_pk' in batch and 'X_pd' in batch:
                    results = model(
                        x_pk=batch['X_pk'], x_pd=batch['X_pd'],
                        lengths_pk=batch['lengths_pk'], lengths_pd=batch['lengths_pd']
                    )
                    val_pk_preds.append(results['pk'].squeeze(-1)[batch['mask_pk']].cpu())
                    val_pk_targets.append(batch['y_pk'][batch['mask_pk']].cpu())
                    val_pd_preds.append(results['pd'].squeeze(-1)[batch['mask_pd']].cpu())
                    val_pd_targets.append(batch['y_pd'][batch['mask_pd']].cpu())

        if val_pk_preds:
            val_pk_metrics = calculate_metrics(torch.cat(val_pk_targets), torch.cat(val_pk_preds))
            val_pd_metrics = calculate_metrics(torch.cat(val_pd_targets), torch.cat(val_pd_preds))
        else:
            val_pk_metrics = val_pd_metrics = {'MSE': 0, 'RMSE': 0, 'MAE': 0, 'R2': 0}

        history['Epoch'].append(epoch + 1)
        for k, v in train_pk_metrics.items():
            history[f'Train PK_{k}'].append(v)
        for k, v in train_pd_metrics.items():
            history[f'Train PD_{k}'].append(v)
        for k, v in val_pk_metrics.items():
            history[f'Val PK_{k}'].append(v)
        for k, v in val_pd_metrics.items():
            history[f'Val PD_{k}'].append(v)

        scheduler.step(val_pd_metrics['MSE'])

        if (epoch + 0) % args.log_interval == 0:
            log_metrics(epoch + 1, "Train PK", train_pk_metrics)
            log_metrics(epoch + 1, "Train PD", train_pd_metrics)
            log_metrics(epoch + 1, "Val PK", val_pk_metrics)
            log_metrics(epoch + 1, "Val PD", val_pd_metrics)


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
                if best_model_state:
                    model.load_state_dict(best_model_state)
                break

    logger.info(f"Best Val PD RMSE: {best_val_loss:.4f}")
    return model, history


def evaluate_lstm(model, sequences, device):
    """Evaluate LSTM model."""
    from Utils.data_loader import collate_lstm_batch

    model.eval()
    pk_preds_list, pk_targets_list = [], []
    pd_preds_list, pd_targets_list = [], []

    with torch.no_grad():
        for i in range(0, len(sequences), 8):
            batch_seqs = sequences[i:i + 8]
            batch = collate_lstm_batch(batch_seqs, device)

            if 'X_pk' in batch and 'X_pd' in batch:
                results = model(
                    x_pk=batch['X_pk'], x_pd=batch['X_pd'],
                    lengths_pk=batch['lengths_pk'], lengths_pd=batch['lengths_pd']
                )
                pk_preds_list.append(results['pk'].squeeze(-1)[batch['mask_pk']].cpu())
                pk_targets_list.append(batch['y_pk'][batch['mask_pk']].cpu())
                pd_preds_list.append(results['pd'].squeeze(-1)[batch['mask_pd']].cpu())
                pd_targets_list.append(batch['y_pd'][batch['mask_pd']].cpu())

    if pk_preds_list:
        pk_preds_arr = torch.cat(pk_preds_list).numpy().flatten()
        pk_targets_arr = torch.cat(pk_targets_list).numpy().flatten()
        pd_preds_arr = torch.cat(pd_preds_list).numpy().flatten()
        pd_targets_arr = torch.cat(pd_targets_list).numpy().flatten()
        pk_metrics = calculate_metrics(pk_targets_arr, pk_preds_arr)
        pd_metrics = calculate_metrics(pd_targets_arr, pd_preds_arr)
    else:
        pk_preds_arr = pk_targets_arr = pd_preds_arr = pd_targets_arr = []
        pk_metrics = pd_metrics = {'MSE': 0, 'RMSE': 0, 'MAE': 0, 'R2': 0}

    return {
        'pk': {**pk_metrics, 'predictions': pk_preds_arr, 'targets': pk_targets_arr},
        'pd': {**pd_metrics, 'predictions': pd_preds_arr, 'targets': pd_targets_arr},
    }

