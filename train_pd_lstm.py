#!/usr/bin/env python3
"""
Train LSTM to predict PD values using temporal sequences.
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
from torch.utils.data import DataLoader, TensorDataset

# Set style
sns.set_style('whitegrid')
plt.rcParams['font.size'] = 10


class PDPredictionLSTM(nn.Module):
    """LSTM for PD prediction."""

    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2):
        super(PDPredictionLSTM, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        # Take the last time step
        last_output = lstm_out[:, -1, :]
        output = self.fc(last_output)
        return output


def create_sequences(data, sequence_length=10):
    """
    Create sequences for LSTM from patient data.

    For each patient, create overlapping sequences where:
    - Input: last sequence_length observations
    - Output: PD value at current time

    Args:
        data: DataFrame with patient data
        sequence_length: Length of input sequence

    Returns:
        X_sequences: (N, sequence_length, features)
        y_targets: (N,)
        metadata: List of (patient_id, time) for each sequence
    """
    sequences = []
    targets = []
    metadata = []

    feature_cols = [col for col in data.columns if col not in ['ID', 'PD']]

    for patient_id in data['ID'].unique():
        patient_data = data[data['ID'] == patient_id].sort_values('TIME').reset_index(drop=True)

        # Create sequences for this patient
        for i in range(sequence_length, len(patient_data)):
            # Get sequence of last sequence_length observations
            seq = patient_data.iloc[i-sequence_length:i][feature_cols].values
            target = patient_data.iloc[i]['PD']
            time = patient_data.iloc[i]['TIME']

            sequences.append(seq)
            targets.append(target)
            metadata.append((patient_id, time))

    return np.array(sequences), np.array(targets), metadata


def prepare_pd_lstm_data(csv_path='Data/QIC2025-EstDat.csv',
                         feature_engineering=False,
                         sequence_length=10,
                         test_size=0.2,
                         random_seed=1712):
    """
    Prepare sequential data for LSTM PD prediction.

    Args:
        csv_path: Path to CSV file
        feature_engineering: Whether to apply feature engineering
        sequence_length: Number of past observations to use
        test_size: Test set proportion
        random_seed: Random seed

    Returns:
        Dictionary with train/test data and metadata
    """
    print("Loading data...")
    df = pd.read_csv(csv_path)

    # Filter to observations only
    df = df[df['EVID'] == 0].copy()
    if 'MDV' in df.columns:
        df = df[df['MDV'] == 0]

    print(f"Total observations: {len(df)}")

    # Create PK-PD paired dataset (same as MLP)
    print("\nCreating PK-PD paired dataset...")

    if 'DVID' in df.columns:
        pk_data = df[df['DVID'] == 1][['ID', 'TIME', 'BW', 'DOSE', 'COMED', 'DV']].copy()
        pk_data = pk_data.rename(columns={'DV': 'PK'})

        pd_data = df[df['DVID'] == 2][['ID', 'TIME', 'BW', 'DOSE', 'COMED', 'DV']].copy()
        pd_data = pd_data.rename(columns={'DV': 'PD'})

        merged_data = []
        for patient_id in pd_data['ID'].unique():
            patient_pk = pk_data[pk_data['ID'] == patient_id].sort_values('TIME')
            patient_pd = pd_data[pd_data['ID'] == patient_id].sort_values('TIME')

            for _, pd_row in patient_pd.iterrows():
                time = pd_row['TIME']
                pk_at_time = patient_pk[patient_pk['TIME'] <= time]
                pk_value = pk_at_time.iloc[-1]['PK'] if len(pk_at_time) > 0 else 0.0

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
        data = df[['ID', 'TIME', 'BW', 'DOSE', 'COMED', 'DV']].copy()
        data['PK'] = 0
        data = data.rename(columns={'DV': 'PD'})

    print(f"PD observations with PK values: {len(data)}")

    # Basic features
    feature_cols = ['TIME', 'BW', 'DOSE', 'COMED', 'PK']

    if feature_engineering:
        print("\n=== Applying Feature Engineering ===")

        # Time transformations
        data['TIME_log'] = np.log1p(data['TIME'])
        data['TIME_sqrt'] = np.sqrt(data['TIME'])

        # PK transformations
        data['PK_log'] = np.log1p(data['PK'])

        # Interactions
        data['TIME_x_PK'] = data['TIME'] * data['PK']
        data['BW_x_PK'] = data['BW'] * data['PK']
        data['DOSE_x_PK'] = data['DOSE'] * data['PK']

        # Per-kg normalization
        data['DOSE_per_kg'] = data['DOSE'] / (data['BW'] + 1e-8)
        data['PK_per_kg'] = data['PK'] / (data['BW'] + 1e-8)

        # Trigonometric features
        data['TIME_sin'] = np.sin(2 * np.pi * data['TIME'] / 24)
        data['TIME_cos'] = np.cos(2 * np.pi * data['TIME'] / 24)

        feature_cols = [col for col in data.columns if col not in ['ID', 'PD']]
        print(f"Total features after engineering: {len(feature_cols)}")

    # Scale features
    scaler_X = StandardScaler()
    data[feature_cols] = scaler_X.fit_transform(data[feature_cols])

    # Create sequences
    print(f"\nCreating sequences (length={sequence_length})...")
    X_seq, y_seq, metadata = create_sequences(data, sequence_length)

    print(f"Total sequences created: {len(X_seq)}")
    print(f"Sequence shape: {X_seq.shape}")

    # Split by sequences (random split)
    from sklearn.model_selection import train_test_split
    indices = np.arange(len(X_seq))
    train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=random_seed)

    X_train = X_seq[train_idx]
    X_test = X_seq[test_idx]
    y_train = y_seq[train_idx]
    y_test = y_seq[test_idx]
    metadata_train = [metadata[i] for i in train_idx]
    metadata_test = [metadata[i] for i in test_idx]

    print(f"\n=== Data Split ===")
    print(f"Train sequences: {len(X_train)}")
    print(f"Test sequences: {len(X_test)}")
    print(f"Feature dimension: {X_train.shape[2]}")

    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'metadata_train': metadata_train,
        'metadata_test': metadata_test,
        'scaler_X': scaler_X,
        'feature_cols': feature_cols
    }


def train_lstm(X_train, y_train, X_val, y_val,
               hidden_dim=64,
               num_layers=2,
               dropout=0.2,
               learning_rate=0.001,
               epochs=100,
               batch_size=32,
               device='cpu'):
    """Train LSTM model."""

    input_dim = X_train.shape[2]

    model = PDPredictionLSTM(input_dim, hidden_dim, num_layers, dropout).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train).reshape(-1, 1)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    train_losses = []
    val_losses = []

    print(f"\n=== Training LSTM ===")
    print(f"Hidden dim: {hidden_dim}, Layers: {num_layers}")
    print(f"Epochs: {epochs}, Batch size: {batch_size}, LR: {learning_rate}")
    print(f"Device: {device}")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

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
                 save_dir='Results/PD_LSTM', n_patients=3, patient_ids=None,
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

    # 2. Scatter plot: Predicted vs Actual
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(data_dict['y_train'], train_metrics['predictions'], alpha=0.6, edgecolors='k', linewidth=0.5)
    min_val = min(data_dict['y_train'].min(), train_metrics['predictions'].min())
    max_val = max(data_dict['y_train'].max(), train_metrics['predictions'].max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    axes[0].set_xlabel('Actual PD')
    axes[0].set_ylabel('Predicted PD')
    axes[0].set_title(f'Train Set (R²={train_metrics["r2"]:.4f}, RMSE={train_metrics["rmse"]:.4f})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

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

    # 3. Patient-specific time series plots
    # Combine train and test data
    all_metadata = data_dict['metadata_train'] + data_dict['metadata_test']
    all_actuals = np.concatenate([data_dict['y_train'], data_dict['y_test']])
    all_predictions = np.concatenate([train_metrics['predictions'], test_metrics['predictions']])
    all_is_test = np.concatenate([
        np.zeros(len(data_dict['y_train']), dtype=bool),
        np.ones(len(data_dict['y_test']), dtype=bool)
    ])

    # Create DataFrame
    plot_df = pd.DataFrame({
        'ID': [m[0] for m in all_metadata],
        'TIME': [m[1] for m in all_metadata],
        'Actual': all_actuals,
        'Predicted': all_predictions,
        'is_test': all_is_test
    })

    # Select patients to plot
    all_unique_patients = plot_df['ID'].unique()

    if patient_ids is not None:
        unique_patients = [pid for pid in patient_ids if pid in all_unique_patients]
        if len(unique_patients) == 0:
            print(f"Warning: None of the specified patient IDs {patient_ids} found in data.")
            print(f"Available patient IDs: {all_unique_patients[:10]}...")
            unique_patients = all_unique_patients[:n_patients]
        else:
            print(f"Plotting specified patients: {unique_patients}")
    elif random_patients:
        np.random.seed(random_seed)
        n_to_select = min(n_patients, len(all_unique_patients))
        unique_patients = np.random.choice(all_unique_patients, size=n_to_select, replace=False)
        print(f"Randomly selected patients: {unique_patients}")
    else:
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
        axes[idx].set_title(f'Patient {int(patient_id)} - PD Prediction (LSTM)')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)

    plt.suptitle('PD Predictions Over Time - LSTM (Train and Test)', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'timeseries_predictions.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nPlots saved to: {save_dir}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Train LSTM for PD prediction')
    parser.add_argument('--csv_path', type=str, default='Data/QIC2025-EstDat.csv',
                       help='Path to CSV file')
    parser.add_argument('--feature_engineering', action='store_true', default=False,
                       help='Apply feature engineering')
    parser.add_argument('--sequence_length', type=int, default=10,
                       help='Length of input sequence')
    parser.add_argument('--hidden_dim', type=int, default=64,
                       help='LSTM hidden dimension')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='Number of LSTM layers')
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
    parser.add_argument('--save_dir', type=str, default='Results/PD_LSTM',
                       help='Directory to save results')

    # Plotting options
    parser.add_argument('--n_patients', type=int, default=3,
                       help='Number of patients to plot in time series')
    parser.add_argument('--patient_ids', type=int, nargs='+', default=None,
                       help='Specific patient IDs to plot')
    parser.add_argument('--random_patients', action='store_true', default=False,
                       help='Randomly select patients to plot')

    args = parser.parse_args()

    # Prepare data
    data_dict = prepare_pd_lstm_data(
        csv_path=args.csv_path,
        feature_engineering=args.feature_engineering,
        sequence_length=args.sequence_length,
        test_size=args.test_size,
        random_seed=args.random_seed
    )

    # Train model
    model, train_losses, val_losses = train_lstm(
        data_dict['X_train'],
        data_dict['y_train'],
        data_dict['X_test'],
        data_dict['y_test'],
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
        f.write("PD Prediction - LSTM - Evaluation Metrics\n")
        f.write("="*60 + "\n\n")
        f.write(f"Feature Engineering: {args.feature_engineering}\n")
        f.write(f"Sequence Length: {args.sequence_length}\n")
        f.write(f"Hidden Dim: {args.hidden_dim}\n")
        f.write(f"Num Layers: {args.num_layers}\n")
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
