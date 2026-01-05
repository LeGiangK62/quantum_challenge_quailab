import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from typing import Dict, Optional, List
import pandas as pd


class CNN1DNetwork(nn.Module):
    """
    1D Convolutional Neural Network for time-series PK/PD prediction.
    """

    def __init__(self, input_channels: int, sequence_length: int,
                 conv_filters: List[int] = [64, 128, 64],
                 kernel_sizes: List[int] = [3, 3, 3],
                 fc_dims: List[int] = [64, 32],
                 output_dim: int = 1,
                 dropout: float = 0.2):
        """
        Initialize 1D CNN network.

        Args:
            input_channels: Number of input channels (features)
            sequence_length: Length of input sequence
            conv_filters: List of convolutional filter numbers
            kernel_sizes: List of kernel sizes for each conv layer
            fc_dims: List of fully connected layer dimensions
            output_dim: Output dimension
            dropout: Dropout rate
        """
        super(CNN1DNetwork, self).__init__()

        # Convolutional layers
        conv_layers = []
        in_channels = input_channels

        for filters, kernel_size in zip(conv_filters, kernel_sizes):
            conv_layers.extend([
                nn.Conv1d(in_channels, filters, kernel_size, padding=kernel_size//2),
                nn.ReLU(),
                nn.BatchNorm1d(filters),
                nn.Dropout(dropout)
            ])
            in_channels = filters

        self.conv_layers = nn.Sequential(*conv_layers)

        # Calculate flattened size after conv layers
        self.flatten_size = conv_filters[-1] * sequence_length

        # Fully connected layers
        fc_layers = []
        prev_dim = self.flatten_size

        for fc_dim in fc_dims:
            fc_layers.extend([
                nn.Linear(prev_dim, fc_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = fc_dim

        fc_layers.append(nn.Linear(prev_dim, output_dim))
        self.fc_layers = nn.Sequential(*fc_layers)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, channels, sequence_length)

        Returns:
            Output predictions
        """
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)
        return x


class CNNModel:
    """
    CNN model wrapper for training and evaluation.
    """

    def __init__(self, input_channels: int, sequence_length: int,
                 conv_filters: List[int] = [64, 128, 64],
                 kernel_sizes: List[int] = [3, 3, 3],
                 fc_dims: List[int] = [64, 32],
                 output_dim: int = 1,
                 dropout: float = 0.2,
                 learning_rate: float = 0.001,
                 device: Optional[str] = None):
        """
        Initialize CNN model.

        Args:
            input_channels: Number of input channels (features)
            sequence_length: Length of input sequence
            conv_filters: List of convolutional filter numbers
            kernel_sizes: List of kernel sizes
            fc_dims: List of fully connected layer dimensions
            output_dim: Output dimension
            dropout: Dropout rate
            learning_rate: Learning rate
            device: Device to use
        """
        self.input_channels = input_channels
        self.sequence_length = sequence_length
        self.conv_filters = conv_filters
        self.kernel_sizes = kernel_sizes
        self.fc_dims = fc_dims
        self.output_dim = output_dim
        self.dropout = dropout
        self.learning_rate = learning_rate

        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Initialize network
        self.model = CNN1DNetwork(
            input_channels=input_channels,
            sequence_length=sequence_length,
            conv_filters=conv_filters,
            kernel_sizes=kernel_sizes,
            fc_dims=fc_dims,
            output_dim=output_dim,
            dropout=dropout
        ).to(self.device)

        # Loss and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.train_losses = []
        self.val_losses = []

    def prepare_sequences(self, X: np.ndarray, y: np.ndarray,
                         sequence_length: int = 10,
                         step_size: int = 1) -> tuple:
        """
        Prepare sequential data for CNN.

        Args:
            X: Features array
            y: Target array
            sequence_length: Length of sequences
            step_size: Step size for sliding window

        Returns:
            Tuple of (X_sequences, y_sequences)
        """
        X_sequences = []
        y_sequences = []

        for i in range(0, len(X) - sequence_length + 1, step_size):
            X_sequences.append(X[i:i+sequence_length])
            y_sequences.append(y[i+sequence_length-1])  # Predict last value

        return np.array(X_sequences), np.array(y_sequences)

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
             X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
             epochs: int = 100, batch_size: int = 32,
             verbose: bool = True) -> None:
        """
        Train the CNN model.

        Args:
            X_train: Training sequences of shape (N, sequence_length, features)
            y_train: Training targets
            X_val: Validation sequences (optional)
            y_val: Validation targets (optional)
            epochs: Number of training epochs
            batch_size: Batch size
            verbose: Whether to print progress
        """
        # Convert to tensors and reshape for CNN: (batch, channels, sequence)
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values
        if isinstance(y_train, pd.Series):
            y_train = y_train.values

        # Reshape: (N, sequence, features) -> (N, features, sequence)
        if len(X_train.shape) == 3:
            X_train = np.transpose(X_train, (0, 2, 1))

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

            if len(X_val.shape) == 3:
                X_val = np.transpose(X_val, (0, 2, 1))

            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val).reshape(-1, 1).to(self.device)

        print(f"Training 1D CNN on {self.device}")
        print(f"Input shape: (batch, {self.input_channels}, {self.sequence_length})")
        print(f"Conv filters: {self.conv_filters}")
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
            X: Input sequences of shape (N, sequence_length, features)

        Returns:
            Predicted values
        """
        self.model.eval()

        if isinstance(X, pd.DataFrame):
            X = X.values

        # Reshape for CNN
        if len(X.shape) == 3:
            X = np.transpose(X, (0, 2, 1))

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

        print(f"\n=== CNN Evaluation Metrics ===")
        for metric, value in metrics.items():
            print(f"{metric.upper()}: {value:.4f}")

        return metrics


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

    # Prepare sequences for CNN
    print("\nPreparing sequence data for CNN...")
    X_seq, y_seq = processor.prepare_sequence_data(X, y, sequence_length=10, step_size=5)

    # Split into train/test
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=0.2, random_state=42
    )

    print(f"\nTrain sequences: {X_train.shape}")
    print(f"Test sequences: {X_test.shape}")

    # Initialize and train CNN
    input_channels = X_train.shape[2]  # Number of features
    sequence_length = X_train.shape[1]  # Sequence length

    model = CNNModel(
        input_channels=input_channels,
        sequence_length=sequence_length,
        conv_filters=[32, 64, 32],
        kernel_sizes=[3, 3, 3],
        fc_dims=[32, 16],
        dropout=0.2,
        learning_rate=0.001
    )

    model.train(
        X_train, y_train,
        X_val=X_test, y_val=y_test,
        epochs=100,
        batch_size=32,
        verbose=True
    )

    # Evaluate
    metrics = model.evaluate(X_test, y_test)
