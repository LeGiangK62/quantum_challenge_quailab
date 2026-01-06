#!/usr/bin/env python3
"""
Train MLP to predict PD values using PK values and other features.
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Add Model to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'Model'))

# Set style
sns.set_style('whitegrid')
plt.rcParams['font.size'] = 10


class PDPredictionMLP(nn.Module):
    """MLP for PD prediction."""

    def __init__(self, input_dim, hidden_dims=[128, 64, 32], dropout=0.2):
        super(PDPredictionMLP, self).__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def prepare_pd_mlp_data(csv_path='Data/QIC2025-EstDat.csv',
                               feature_engineering=False,
                               test_size=0.2,
                               random_seed=1712):
    """
    Prepare data for MLP PD Prediction

    Input features: TIME, BW, PK_value, DOSE, COMED
    Output: PD value

    Args:
        csv_path: Path to CSV file
        feature_engineering: Whether to apply feature engineering
        test_size: Test set proportion
        random_seed: Random seed

    Returns:
        Dictionary with train/test data and metadata
    """
    print("Loading data...")
    df = pd.read_csv(csv_path)

    # Filter to observations only (EVID=0)
    df = df[df['EVID'] == 0].copy()

    # Filter out missing values
    if 'MDV' in df.columns:
        df = df[df['MDV'] == 0]

    print(f"Total observations: {len(df)}")

    # Create PK and PD columns
    # Pivot the data so each row has both PK and PD values at the same time point
    print("\nCreating PK-PD paired dataset...")

    # Separate PK and PD data
    if 'DVID' in df.columns:
        pk_data = df[df['DVID'] == 1][['ID', 'TIME', 'BW', 'DOSE', 'COMED', 'DV']].copy()
        pk_data = pk_data.rename(columns={'DV': 'PK'})

        pd_data = df[df['DVID'] == 2][['ID', 'TIME', 'BW', 'DOSE', 'COMED', 'DV']].copy()
        pd_data = pd_data.rename(columns={'DV': 'PD'})

        # Merge PK and PD on ID and TIME
        # For each PD observation, find the corresponding PK value
        # If no exact match, use the most recent PK value (forward fill)
        merged_data = []

        for patient_id in pd_data['ID'].unique():
            patient_pk = pk_data[pk_data['ID'] == patient_id].sort_values('TIME')
            patient_pd = pd_data[pd_data['ID'] == patient_id].sort_values('TIME')

            for _, pd_row in patient_pd.iterrows():
                time = pd_row['TIME']

                # Find most recent PK value (at or before this time)
                pk_at_time = patient_pk[patient_pk['TIME'] <= time]

                if len(pk_at_time) > 0:
                    pk_value = pk_at_time.iloc[-1]['PK']
                else:
                    # No PK value available yet, use 0
                    pk_value = 0.0

                merged_data.append({
                    'ID': patient_id,
                    'TIME': time,
                    'BW': pd_row['BW'],
                    'DOSE': pd_row['DOSE'],
                    'COMED': pd_row['COMED'],
                    'PK': pk_value,
                    'PD': pd_row['PD']
                })

        data = pd.DataFrame(merged_data)
    else:
        # No DVID column, use DV as target
        data = df[['ID', 'TIME', 'BW', 'DOSE', 'COMED', 'DV']].copy()
        data['PK'] = 0  # No PK data available
        data = data.rename(columns={'DV': 'PD'})

    print(f"PD observations with PK values: {len(data)}")
    print(f"\nFeature columns: TIME, BW, DOSE, COMED, PK")
    print(f"Target column: PD")

    # Basic features
    feature_cols = ['TIME', 'BW', 'DOSE', 'COMED', 'PK']

    if feature_engineering:
        print("\n=== Applying Feature Engineering ===")

        # Time transformations
        data['TIME_squared'] = data['TIME'] ** 2
        data['TIME_log'] = np.log1p(data['TIME'])
        data['TIME_sqrt'] = np.sqrt(data['TIME'])

        # PK transformations
        data['PK_squared'] = data['PK'] ** 2
        data['PK_log'] = np.log1p(data['PK'])

        # Interactions
        data['TIME_x_PK'] = data['TIME'] * data['PK']
        data['BW_x_PK'] = data['BW'] * data['PK']
        data['DOSE_x_PK'] = data['DOSE'] * data['PK']
        data['TIME_x_DOSE'] = data['TIME'] * data['DOSE']
        data['BW_x_DOSE'] = data['BW'] * data['DOSE']

        # Per-kg normalization
        data['DOSE_per_kg'] = data['DOSE'] / (data['BW'] + 1e-8)
        data['PK_per_kg'] = data['PK'] / (data['BW'] + 1e-8)

        # Trigonometric features (circadian rhythm)
        data['TIME_sin'] = np.sin(2 * np.pi * data['TIME'] / 24)
        data['TIME_cos'] = np.cos(2 * np.pi * data['TIME'] / 24)

        feature_cols = [col for col in data.columns if col not in ['ID', 'PD']]

        print(f"Total features after engineering: {len(feature_cols)}")
        print(f"Feature columns: {feature_cols}")

    # Prepare features and target
    X = data[feature_cols].values
    y = data['PD'].values

    # Keep track of patient IDs and TIME for plotting
    ids = data['ID'].values
    times = data['TIME'].values

    # Train-test split
    X_train, X_test, y_train, y_test, ids_train, ids_test, times_train, times_test = train_test_split(
        X, y, ids, times, test_size=test_size, random_state=random_seed
    )

    # Scale features
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    print(f"\n=== Data Split ===")
    print(f"Train samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Feature dimension: {X_train.shape[1]}")

    return {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'ids_train': ids_train,
        'ids_test': ids_test,
        'times_train': times_train,
        'times_test': times_test,
        'scaler_X': scaler_X,
        'feature_cols': feature_cols,
        'full_data': data
    }


def train_mlp(X_train, y_train, X_val, y_val,
              hidden_dims=[128, 64, 32],
              dropout=0.2,
              learning_rate=0.001,
              epochs=100,
              batch_size=32,
              device='cpu'):
    """Train MLP model."""

    input_dim = X_train.shape[1]

    # Create model
    model = PDPredictionMLP(input_dim, hidden_dims, dropout).to(device)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Create data loaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train).reshape(-1, 1)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Training history
    train_losses = []
    val_losses = []

    print(f"\n=== Training MLP ===")
    print(f"Model: {hidden_dims}")
    print(f"Epochs: {epochs}, Batch size: {batch_size}, LR: {learning_rate}")
    print(f"Device: {device}")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(torch.FloatTensor(X_val).to(device))
            val_loss = criterion(val_outputs, torch.FloatTensor(y_val).reshape(-1, 1).to(device))

        train_losses.append(epoch_loss / len(train_loader))
        val_losses.append(val_loss.item())

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")

    return model, train_losses, val_losses


def evaluate_model(model, X, y, device='cpu'):
    """Evaluate model."""
    model.eval()
    with torch.no_grad():
        predictions = model(torch.FloatTensor(X).to(device)).cpu().numpy().flatten()

    mse = np.mean((y - predictions) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y - predictions))
    r2 = 1 - (np.sum((y - predictions) ** 2) / np.sum((y - np.mean(y)) ** 2))

    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'predictions': predictions
    }


def plot_results(data_dict, train_metrics, test_metrics, train_losses, val_losses,
                 save_dir='Results/PD_MLP', n_patients=3, patient_ids=None, random_patients=False, random_seed=1712):
    """Plot training results and time series predictions.

    Args:
        data_dict: Dictionary with train/test data
        train_metrics: Training metrics
        test_metrics: Test metrics
        train_losses: Training losses
        val_losses: Validation losses
        save_dir: Directory to save plots
        n_patients: Number of patients to plot
        patient_ids: Specific patient IDs to plot (list of IDs)
        random_patients: Whether to randomly select patients
        random_seed: Random seed for patient selection
    """

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

    # 2. Scatter plot: Predicted vs Actual
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Train
    axes[0].scatter(data_dict['y_train'], train_metrics['predictions'], alpha=0.6, edgecolors='k', linewidth=0.5)
    min_val = min(data_dict['y_train'].min(), train_metrics['predictions'].min())
    max_val = max(data_dict['y_train'].max(), train_metrics['predictions'].max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    axes[0].set_xlabel('Actual PD')
    axes[0].set_ylabel('Predicted PD')
    axes[0].set_title(f'Train Set (R²={train_metrics["r2"]:.4f}, RMSE={train_metrics["rmse"]:.4f})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Test
    axes[1].scatter(data_dict['y_test'], test_metrics['predictions'], alpha=0.6, edgecolors='k', linewidth=0.5)
    min_val = min(data_dict['y_test'].min(), test_metrics['predictions'].min())
    max_val = max(data_dict['y_test'].max(), test_metrics['predictions'].max())
    axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    axes[1].set_xlabel('Actual PD')
    axes[1].set_ylabel('Predicted PD')
    axes[1].set_title(f'Test Set (R²={test_metrics["r2"]:.4f}, RMSE={test_metrics["rmse"]:.4f})')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'predictions_scatter.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Time series plots for individual patients (both train and test)
    # Combine train and test data
    all_ids = np.concatenate([data_dict['ids_train'], data_dict['ids_test']])
    all_times = np.concatenate([data_dict['times_train'], data_dict['times_test']])
    all_actuals = np.concatenate([data_dict['y_train'], data_dict['y_test']])
    all_predictions = np.concatenate([train_metrics['predictions'], test_metrics['predictions']])
    all_is_test = np.concatenate([np.zeros(len(data_dict['y_train']), dtype=bool),
                                   np.ones(len(data_dict['y_test']), dtype=bool)])

    # Create DataFrame
    plot_df = pd.DataFrame({
        'ID': all_ids,
        'TIME': all_times,
        'Actual': all_actuals,
        'Predicted': all_predictions,
        'is_test': all_is_test
    })

    # Select patients to plot
    all_unique_patients = np.unique(all_ids)

    if patient_ids is not None:
        # Use specific patient IDs provided
        unique_patients = [pid for pid in patient_ids if pid in all_unique_patients]
        if len(unique_patients) == 0:
            print(f"Warning: None of the specified patient IDs {patient_ids} found in data.")
            print(f"Available patient IDs: {all_unique_patients[:10]}...")
            unique_patients = all_unique_patients[:n_patients]
        else:
            print(f"Plotting specified patients: {unique_patients}")
    elif random_patients:
        # Randomly select patients
        np.random.seed(random_seed)
        n_to_select = min(n_patients, len(all_unique_patients))
        unique_patients = np.random.choice(all_unique_patients, size=n_to_select, replace=False)
        print(f"Randomly selected patients: {unique_patients}")
    else:
        # Use first n_patients
        unique_patients = all_unique_patients[:n_patients]
        print(f"Plotting first {len(unique_patients)} patients: {unique_patients}")

    fig, axes = plt.subplots(len(unique_patients), 1, figsize=(12, 4*len(unique_patients)))
    if len(unique_patients) == 1:
        axes = [axes]

    for idx, patient_id in enumerate(unique_patients):
        patient_data = plot_df[plot_df['ID'] == patient_id].sort_values('TIME')

        # Plot actual values
        axes[idx].plot(patient_data['TIME'], patient_data['Actual'],
                      'o-', label='Actual PD', markersize=6, linewidth=2, alpha=0.7, color='blue')

        # Plot ALL predictions connected with a line
        axes[idx].plot(patient_data['TIME'], patient_data['Predicted'],
                      '--', linewidth=1.5, alpha=0.4, color='gray', label='_nolegend_')

        # Plot predictions - distinguish train/test with markers
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
        axes[idx].set_title(f'Patient {int(patient_id)} - PD Prediction')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)

    plt.suptitle('PD Predictions Over Time (Train and Test)', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'timeseries_predictions.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nPlots saved to: {save_dir}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Train MLP for PD prediction')
    parser.add_argument('--csv_path', type=str, default='Data/QIC2025-EstDat.csv',
                       help='Path to CSV file')
    parser.add_argument('--feature_engineering', action='store_true', default=False,
                       help='Apply feature engineering')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[128, 64, 32],
                       help='Hidden layer dimensions')
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout rate')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Test set proportion')
    parser.add_argument('--random_seed', type=int, default=1712,
                       help='Random seed')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device (cpu or cuda)')
    parser.add_argument('--save_dir', type=str, default='Results/PD_MLP',
                       help='Directory to save results')

    # Plotting options
    parser.add_argument('--n_patients', type=int, default=3,
                       help='Number of patients to plot in time series')
    parser.add_argument('--patient_ids', type=int, nargs='+', default=None,
                       help='Specific patient IDs to plot (e.g., --patient_ids 16 36 46)')
    parser.add_argument('--random_patients', action='store_true', default=False,
                       help='Randomly select patients to plot')

    args = parser.parse_args()

    # Prepare data
    data_dict = prepare_pd_mlp_data(
        csv_path=args.csv_path,
        feature_engineering=args.feature_engineering,
        test_size=args.test_size,
        random_seed=args.random_seed
    )

    # Train model
    model, train_losses, val_losses = train_mlp(
        data_dict['X_train'],
        data_dict['y_train'],
        data_dict['X_test'],
        data_dict['y_test'],
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device
    )

    # Evaluate
    print("\n=== Evaluation ===")
    train_metrics = evaluate_model(model, data_dict['X_train'], data_dict['y_train'], args.device)
    test_metrics = evaluate_model(model, data_dict['X_test'], data_dict['y_test'], args.device)

    print(f"\nTrain Metrics:")
    print(f"  RMSE: {train_metrics['rmse']:.4f}")
    print(f"  MAE:  {train_metrics['mae']:.4f}")
    print(f"  R²:   {train_metrics['r2']:.4f}")

    print(f"\nTest Metrics:")
    print(f"  RMSE: {test_metrics['rmse']:.4f}")
    print(f"  MAE:  {test_metrics['mae']:.4f}")
    print(f"  R²:   {test_metrics['r2']:.4f}")

    # Plot results
    plot_results(data_dict, train_metrics, test_metrics, train_losses, val_losses,
                args.save_dir, args.n_patients, args.patient_ids, args.random_patients, args.random_seed)

    # Save metrics
    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, 'metrics.txt'), 'w') as f:
        f.write("="*60 + "\n")
        f.write("PD Prediction - Evaluation Metrics\n")
        f.write("="*60 + "\n\n")
        f.write(f"Feature Engineering: {args.feature_engineering}\n")
        f.write(f"Hidden Dims: {args.hidden_dims}\n")
        f.write(f"Epochs: {args.epochs}\n")
        f.write(f"Learning Rate: {args.learning_rate}\n\n")
        f.write("Train Metrics:\n")
        f.write(f"  RMSE: {train_metrics['rmse']:.4f}\n")
        f.write(f"  MAE:  {train_metrics['mae']:.4f}\n")
        f.write(f"  R²:   {train_metrics['r2']:.4f}\n\n")
        f.write("Test Metrics:\n")
        f.write(f"  RMSE: {test_metrics['rmse']:.4f}\n")
        f.write(f"  MAE:  {test_metrics['mae']:.4f}\n")
        f.write(f"  R²:   {test_metrics['r2']:.4f}\n")
        f.write("="*60 + "\n")

    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"Results saved to: {args.save_dir}")
    print('='*60)


if __name__ == "__main__":
    main()
