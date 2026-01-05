import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from typing import Dict, Optional, List
import pandas as pd


class GCNNetwork(nn.Module):
    """
    Graph Convolutional Network (GCN) for PK/PD prediction.
    """

    def __init__(self, input_dim: int, hidden_dims: List[int] = [64, 32],
                 output_dim: int = 1, dropout: float = 0.2):
        """
        Initialize GCN network.

        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension
            dropout: Dropout rate
        """
        super(GCNNetwork, self).__init__()

        # GCN layers
        self.convs = nn.ModuleList()
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            self.convs.append(GCNConv(prev_dim, hidden_dim))
            prev_dim = hidden_dim

        # Output layer
        self.fc = nn.Linear(prev_dim, output_dim)
        self.dropout = dropout

    def forward(self, x, edge_index, batch=None):
        """
        Forward pass.

        Args:
            x: Node features (num_nodes, input_dim)
            edge_index: Edge indices (2, num_edges)
            batch: Batch assignment for each node (for batched graphs)

        Returns:
            Node-level predictions
        """
        # Apply GCN layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Output layer
        x = self.fc(x)

        return x


class GATNetwork(nn.Module):
    """
    Graph Attention Network (GAT) for PK/PD prediction.
    """

    def __init__(self, input_dim: int, hidden_dims: List[int] = [64, 32],
                 output_dim: int = 1, heads: int = 4, dropout: float = 0.2):
        """
        Initialize GAT network.

        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension
            heads: Number of attention heads
            dropout: Dropout rate
        """
        super(GATNetwork, self).__init__()

        # GAT layers
        self.convs = nn.ModuleList()
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            if i == len(hidden_dims) - 1:
                # Last GAT layer: concatenate heads
                self.convs.append(GATConv(prev_dim, hidden_dim, heads=1, dropout=dropout))
                prev_dim = hidden_dim
            else:
                # Intermediate layers: use multiple heads
                self.convs.append(GATConv(prev_dim, hidden_dim, heads=heads, dropout=dropout))
                prev_dim = hidden_dim * heads

        # Output layer
        self.fc = nn.Linear(prev_dim, output_dim)
        self.dropout = dropout

    def forward(self, x, edge_index, batch=None):
        """
        Forward pass.

        Args:
            x: Node features
            edge_index: Edge indices
            batch: Batch assignment

        Returns:
            Node-level predictions
        """
        # Apply GAT layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Output layer
        x = self.fc(x)

        return x


class GNNModel:
    """
    GNN model wrapper for training and evaluation.
    """

    def __init__(self, input_dim: int, model_type: str = 'gcn',
                 hidden_dims: List[int] = [64, 32],
                 output_dim: int = 1, dropout: float = 0.2,
                 learning_rate: float = 0.001,
                 device: Optional[str] = None):
        """
        Initialize GNN model.

        Args:
            input_dim: Input feature dimension
            model_type: Type of GNN ('gcn' or 'gat')
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension
            dropout: Dropout rate
            learning_rate: Learning rate
            device: Device to use
        """
        self.input_dim = input_dim
        self.model_type = model_type
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
        if model_type == 'gcn':
            self.model = GCNNetwork(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                output_dim=output_dim,
                dropout=dropout
            ).to(self.device)
        elif model_type == 'gat':
            self.model = GATNetwork(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                output_dim=output_dim,
                heads=4,
                dropout=dropout
            ).to(self.device)
        else:
            raise ValueError(f"Unknown model_type: {model_type}. Use 'gcn' or 'gat'.")

        # Loss and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.train_losses = []
        self.val_losses = []

    def prepare_graph_data(self, node_features: np.ndarray, edge_index: np.ndarray,
                          targets: np.ndarray, test_size: float = 0.2,
                          random_state: int = 42) -> tuple:
        """
        Prepare graph data for training.

        Args:
            node_features: Node feature matrix (num_nodes, num_features)
            edge_index: Edge index array (2, num_edges)
            targets: Target values for each node
            test_size: Test set proportion
            random_state: Random seed

        Returns:
            Tuple of (train_data, test_data)
        """
        # Split nodes into train/test
        num_nodes = len(node_features)
        indices = np.arange(num_nodes)
        train_idx, test_idx = train_test_split(
            indices, test_size=test_size, random_state=random_state
        )

        # Create train mask
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask[train_idx] = True

        # Create test mask
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask[test_idx] = True

        # Create PyG Data object
        data = Data(
            x=torch.FloatTensor(node_features),
            edge_index=torch.LongTensor(edge_index),
            y=torch.FloatTensor(targets).reshape(-1, 1),
            train_mask=train_mask,
            test_mask=test_mask
        ).to(self.device)

        print(f"Graph data prepared:")
        print(f"  Nodes: {data.num_nodes}")
        print(f"  Edges: {data.num_edges}")
        print(f"  Features: {data.num_node_features}")
        print(f"  Train nodes: {train_mask.sum().item()}")
        print(f"  Test nodes: {test_mask.sum().item()}")

        return data

    def train(self, data: Data, epochs: int = 100, verbose: bool = True) -> None:
        """
        Train the GNN model.

        Args:
            data: PyG Data object containing graph and targets
            epochs: Number of training epochs
            verbose: Whether to print progress
        """
        print(f"Training {self.model_type.upper()} on {self.device}")
        print(f"Hidden dims: {self.hidden_dims}")
        print(f"Epochs: {epochs}")

        # Training loop
        for epoch in range(epochs):
            self.model.train()

            # Forward pass
            self.optimizer.zero_grad()
            out = self.model(data.x, data.edge_index)

            # Compute loss only on training nodes
            train_loss = self.criterion(out[data.train_mask], data.y[data.train_mask])

            # Backward pass
            train_loss.backward()
            self.optimizer.step()

            self.train_losses.append(train_loss.item())

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_out = self.model(data.x, data.edge_index)
                val_loss = self.criterion(val_out[data.test_mask], data.y[data.test_mask])
                self.val_losses.append(val_loss.item())

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], "
                      f"Train Loss: {train_loss.item():.4f}, "
                      f"Val Loss: {val_loss.item():.4f}")

        print("Training completed.")

    def predict(self, data: Data, mask: Optional[torch.Tensor] = None) -> np.ndarray:
        """
        Make predictions.

        Args:
            data: PyG Data object
            mask: Optional mask to select specific nodes

        Returns:
            Predicted values
        """
        self.model.eval()

        with torch.no_grad():
            predictions = self.model(data.x, data.edge_index)

        if mask is not None:
            predictions = predictions[mask]

        return predictions.cpu().numpy().flatten()

    def evaluate(self, data: Data, mask: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Evaluate model performance.

        Args:
            data: PyG Data object
            mask: Mask for selecting nodes to evaluate (default: test_mask)

        Returns:
            Dictionary with evaluation metrics
        """
        if mask is None:
            mask = data.test_mask

        y_pred = self.predict(data, mask)
        y_true = data.y[mask].cpu().numpy().flatten()

        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
        }

        print(f"\n=== {self.model_type.upper()} Evaluation Metrics ===")
        for metric, value in metrics.items():
            print(f"{metric.upper()}: {value:.4f}")

        return metrics


def train_and_evaluate_gnn(node_features: np.ndarray, edge_index: np.ndarray,
                          targets: np.ndarray, model_types: Optional[List[str]] = None,
                          epochs: int = 100) -> Dict:
    """
    Train and evaluate different GNN models.

    Args:
        node_features: Node feature matrix
        edge_index: Edge index array
        targets: Target values
        model_types: List of model types to try
        epochs: Number of training epochs

    Returns:
        Dictionary with all results
    """
    if model_types is None:
        model_types = ['gcn', 'gat']

    input_dim = node_features.shape[1]
    results = {}

    for model_type in model_types:
        print(f"\n{'='*60}")
        print(f"Training {model_type.upper()} Model")
        print('='*60)

        model = GNNModel(
            input_dim=input_dim,
            model_type=model_type,
            hidden_dims=[64, 32],
            dropout=0.2,
            learning_rate=0.001
        )

        # Prepare graph data
        data = model.prepare_graph_data(node_features, edge_index, targets)

        # Train
        model.train(data, epochs=epochs, verbose=True)

        # Evaluate
        metrics = model.evaluate(data)

        results[model_type] = {
            'model': model,
            'metrics': metrics,
            'data': data,
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

    # Get graph data
    print("\nPreparing graph data for GNN...")
    graph_data = processor.get_graph_data()

    # Train and evaluate GNN models
    results = train_and_evaluate_gnn(
        node_features=graph_data['node_features'],
        edge_index=graph_data['edge_index'],
        targets=graph_data['targets'],
        model_types=['gcn', 'gat'],
        epochs=100
    )

    # Compare models
    print("\n" + "="*60)
    print("GNN Model Comparison")
    print("="*60)
    comparison_df = pd.DataFrame({
        model_type: results[model_type]['metrics']
        for model_type in results.keys()
    }).T
    print(comparison_df.to_string())
