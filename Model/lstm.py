import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from typing import Dict, Optional, List
import pandas as pd


class LSTMNetwork(nn.Module):
    """
    LSTM (Long Short-Term Memory) network for time-series PK/PD prediction.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64,
                 num_layers: int = 2, output_dim: int = 1,
                 dropout: float = 0.2, bidirectional: bool = False):
        """
        Initialize LSTM network.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden state dimension
            num_layers: Number of LSTM layers
            output_dim: Output dimension
            dropout: Dropout rate (applied between LSTM layers)
            bidirectional: Whether to use bidirectional LSTM
        """
        super(LSTMNetwork, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # Output layer
        fc_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(fc_input_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, sequence, features)

        Returns:
            Output predictions
        """
        # LSTM forward pass
        # lstm_out: (batch, sequence, hidden_dim * num_directions)
        # h_n: (num_layers * num_directions, batch, hidden_dim)
        # c_n: (num_layers * num_directions, batch, hidden_dim)
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Use the last output for prediction
        last_output = lstm_out[:, -1, :]

        # Apply dropout and linear layer
        out = self.dropout(last_output)
        out = self.fc(out)

        return out


class LSTMModel:
    """
    LSTM model wrapper for training and evaluation.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64,
                 num_layers: int = 2, output_dim: int = 1,
                 dropout: float = 0.2, bidirectional: bool = False,
                 learning_rate: float = 0.001,
                 device: Optional[str] = None):
        """
        Initialize LSTM model.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden state dimension
            num_layers: Number of LSTM layers
            output_dim: Output dimension
            dropout: Dropout rate
            bidirectional: Whether to use bidirectional LSTM
            learning_rate: Learning rate
            device: Device to use
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.learning_rate = learning_rate

        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Initialize network
        self.model = LSTMNetwork(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=output_dim,
            dropout=dropout,
            bidirectional=bidirectional
        ).to(self.device)

        # Loss and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.train_losses = []
        self.val_losses = []

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
             X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
             epochs: int = 100, batch_size: int = 32,
             verbose: bool = True) -> None:
        """
        Train the LSTM model.

        Args:
            X_train: Training sequences of shape (N, sequence_length, features)
            y_train: Training targets
            X_val: Validation sequences (optional)
            y_val: Validation targets (optional)
            epochs: Number of training epochs
            batch_size: Batch size
            verbose: Whether to print progress
        """
        # Convert to tensors
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values
        if isinstance(y_train, pd.Series):
            y_train = y_train.values

        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1).to(self.device)

        # Create data loader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Validation data
        if X_val is not None and y_val is not None:
            if isinstance(X_val, pd.DataFrame):
                X_val = X_val.values
            if isinstance(y_val, pd.Series):
                y_val = y_val.values

            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val).reshape(-1, 1).to(self.device)

        direction = "Bidirectional" if self.bidirectional else "Unidirectional"
        print(f"Training {direction} LSTM on {self.device}")
        print(f"Hidden dim: {self.hidden_dim}, Num layers: {self.num_layers}")
        print(f"Epochs: {epochs}, Batch size: {batch_size}")

        # Training loop
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0

            for batch_X, batch_y in train_loader:
                # Forward pass
                predictions = self.model(batch_X)
                loss = self.criterion(predictions, batch_y)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()

                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()

                epoch_loss += loss.item()

            avg_train_loss = epoch_loss / len(train_loader)
            self.train_losses.append(avg_train_loss)

            # Validation
            if X_val is not None and y_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_predictions = self.model(X_val_tensor)
                    val_loss = self.criterion(val_predictions, y_val_tensor).item()
                    self.val_losses.append(val_loss)

                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch [{epoch+1}/{epochs}], "
                          f"Train Loss: {avg_train_loss:.4f}, "
                          f"Val Loss: {val_loss:.4f}")
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch [{epoch+1}/{epochs}], "
                          f"Train Loss: {avg_train_loss:.4f}")

        print("Training completed.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Input sequences of shape (N, sequence_length, features)

        Returns:
            Predicted values
        """
        self.model.eval()

        if isinstance(X, pd.DataFrame):
            X = X.values

        X_tensor = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            predictions = self.model(X_tensor)

        return predictions.cpu().numpy().flatten()

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance.

        Args:
            X_test: Test sequences
            y_test: True test targets

        Returns:
            Dictionary with evaluation metrics
        """
        y_pred = self.predict(X_test)

        if isinstance(y_test, pd.Series):
            y_test = y_test.values

        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
        }

        direction = "Bidirectional" if self.bidirectional else "Unidirectional"
        print(f"\n=== {direction} LSTM Evaluation Metrics ===")
        for metric, value in metrics.items():
            print(f"{metric.upper()}: {value:.4f}")

        return metrics

    def save_model(self, path: str) -> None:
        """Save model state."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': {
                'input_dim': self.input_dim,
                'hidden_dim': self.hidden_dim,
                'num_layers': self.num_layers,
                'output_dim': self.output_dim,
                'dropout': self.dropout,
                'bidirectional': self.bidirectional,
                'learning_rate': self.learning_rate
            }
        }, path)
        print(f"Model saved to {path}")

    def load_model(self, path: str) -> None:
        """Load model state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        print(f"Model loaded from {path}")


def train_and_evaluate_lstm(X_train, X_test, y_train, y_test,
                            hidden_dims: Optional[List[int]] = None,
                            num_layers_list: Optional[List[int]] = None,
                            epochs: int = 100, batch_size: int = 32) -> Dict:
    """
    Train and evaluate LSTM models with different configurations.

    Args:
        X_train, X_test, y_train, y_test: Train and test data
        hidden_dims: List of hidden dimensions to try
        num_layers_list: List of number of layers to try
        epochs: Number of training epochs
        batch_size: Batch size

    Returns:
        Dictionary with all results
    """
    if hidden_dims is None:
        hidden_dims = [32, 64, 128]
    if num_layers_list is None:
        num_layers_list = [1, 2]

    input_dim = X_train.shape[2]  # Feature dimension
    results = {}

    for hidden_dim in hidden_dims:
        for num_layers in num_layers_list:
            for bidirectional in [False, True]:
                model_name = f"LSTM_h{hidden_dim}_l{num_layers}{'_bi' if bidirectional else ''}"
                print(f"\n{'='*60}")
                print(f"Training {model_name}")
                print('='*60)

                model = LSTMModel(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                    dropout=0.2,
                    bidirectional=bidirectional,
                    learning_rate=0.001
                )

                model.train(
                    X_train, y_train,
                    X_val=X_test, y_val=y_test,
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=True
                )

                metrics = model.evaluate(X_test, y_test)

                results[model_name] = {
                    'model': model,
                    'metrics': metrics,
                    'train_losses': model.train_losses,
                    'val_losses': model.val_losses
                }

    return results


if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.append('..')
    from Utils.pre_processing import PKPDDataProcessor

    # Load and preprocess data
    processor = PKPDDataProcessor()
    processor.load_data()

    # Prepare features and target
    X, y = processor.prepare_features_target()

    # Prepare sequences for LSTM
    print("\nPreparing sequence data for LSTM...")
    X_seq, y_seq = processor.prepare_sequence_data(X, y, sequence_length=10, step_size=5)

    # Split into train/test
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=0.2, random_state=42
    )

    print(f"\nTrain sequences: {X_train.shape}")
    print(f"Test sequences: {X_test.shape}")

    # Train and evaluate LSTM models
    results = train_and_evaluate_lstm(
        X_train, X_test, y_train, y_test,
        hidden_dims=[64],
        num_layers_list=[2],
        epochs=100,
        batch_size=32
    )

    # Compare models
    print("\n" + "="*60)
    print("LSTM Configuration Comparison")
    print("="*60)
    comparison_df = pd.DataFrame({
        model_name: results[model_name]['metrics']
        for model_name in results.keys()
    }).T
    print(comparison_df.to_string())
