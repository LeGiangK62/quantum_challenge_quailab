import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from typing import Dict, Optional, List
import pandas as pd


class MLPNetwork(nn.Module):
    """
    Multi-Layer Perceptron (MLP) neural network.
    """

    def __init__(self, input_dim: int, hidden_dims: List[int],
                 output_dim: int = 1, dropout: float = 0.2):
        """
        Initialize MLP network.

        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension (default: 1 for regression)
            dropout: Dropout rate
        """
        super(MLPNetwork, self).__init__()

        layers = []
        prev_dim = input_dim

        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass."""
        return self.network(x)


class MLPModel:
    """
    MLP model wrapper for training and evaluation.
    """

    def __init__(self, input_dim: int, hidden_dims: List[int] = [64, 32],
                 output_dim: int = 1, dropout: float = 0.2,
                 learning_rate: float = 0.001, device: Optional[str] = None):
        """
        Initialize MLP model.

        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension
            dropout: Dropout rate
            learning_rate: Learning rate for optimizer
            device: Device to use ('cuda' or 'cpu')
        """
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout = dropout
        self.learning_rate = learning_rate

        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Initialize network
        self.model = MLPNetwork(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            dropout=dropout
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
        Train the MLP model.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            epochs: Number of training epochs
            batch_size: Batch size
            verbose: Whether to print training progress
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

        print(f"Training MLP on {self.device}")
        print(f"Architecture: {self.input_dim} -> {' -> '.join(map(str, self.hidden_dims))} -> {self.output_dim}")
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
            X: Input features

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
            X_test: Test features
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

        print(f"\n=== MLP Evaluation Metrics ===")
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
                'hidden_dims': self.hidden_dims,
                'output_dim': self.output_dim,
                'dropout': self.dropout,
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


def train_and_evaluate_mlp(X_train, X_test, y_train, y_test,
                           hidden_dims_list: Optional[List[List[int]]] = None,
                           epochs: int = 100, batch_size: int = 32) -> Dict:
    """
    Train and evaluate MLP models with different architectures.

    Args:
        X_train, X_test, y_train, y_test: Train and test data
        hidden_dims_list: List of hidden layer configurations to try
        epochs: Number of training epochs
        batch_size: Batch size

    Returns:
        Dictionary with all results
    """
    if hidden_dims_list is None:
        hidden_dims_list = [
            [32],
            [64, 32],
            [128, 64, 32],
            [256, 128, 64]
        ]

    input_dim = X_train.shape[1]
    results = {}

    for i, hidden_dims in enumerate(hidden_dims_list):
        arch_name = f"MLP_{'-'.join(map(str, hidden_dims))}"
        print(f"\n{'='*60}")
        print(f"Training {arch_name}")
        print('='*60)

        model = MLPModel(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout=0.2,
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

        results[arch_name] = {
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
    data = processor.get_full_pipeline(scale_features=True)

    # Train and evaluate MLP models
    results = train_and_evaluate_mlp(
        data['X_train'], data['X_test'],
        data['y_train'], data['y_test'],
        epochs=100,
        batch_size=32
    )

    # Compare models
    print("\n" + "="*60)
    print("MLP Architecture Comparison")
    print("="*60)
    comparison_df = pd.DataFrame({
        arch: results[arch]['metrics']
        for arch in results.keys()
    }).T
    print(comparison_df.to_string())
