import torch
import torch.nn as nn
import pennylane as qml
import numpy as np
from torch_geometric.nn import GCNConv, GATConv, LayerNorm
from .gnn import PKGNNEncoder


class QNN(nn.Module):
    """
    Basic Quantum Neural Network using StronglyEntanglingLayers PQC from PennyLane.

    Architecture:
        Input -> Classical Linear -> Angle Embedding -> StronglyEntanglingLayers -> Measurement -> Output
    """

    def __init__(self, input_features, n_qubits=4, n_layers=2):
        super(QNN, self).__init__()

        self.n_qubits = n_qubits
        self.n_layers = n_layers

        # Classical preprocessing layer to match input to n_qubits
        self.pre_net = nn.Linear(input_features, n_qubits)

        # Create quantum device
        self.dev = qml.device("default.qubit", wires=n_qubits)

        # Initialize quantum circuit weights
        # StronglyEntanglingLayers requires shape (n_layers, n_qubits, 3)
        weight_shape = (n_layers, n_qubits, 3)
        self.q_weights = nn.Parameter(torch.randn(weight_shape) * 0.1)

        # Create quantum node
        self.qnode = qml.QNode(self._circuit, self.dev, interface="torch", diff_method="backprop")

        # Classical post-processing layer
        self.post_net = nn.Linear(n_qubits, 1)

    def _circuit(self, inputs, weights):
        """
        Quantum circuit with angle embedding and strongly entangling layers.

        Args:
            inputs: Input features (n_qubits,)
            weights: Trainable parameters (n_layers, n_qubits, 3)

        Returns:
            Expectation values of PauliZ for each qubit
        """
        # Encode classical data into quantum state using angle embedding
        qml.AngleEmbedding(inputs, wires=range(self.n_qubits), rotation='Y')

        # Apply strongly entangling layers (parameterized quantum circuit)
        qml.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))

        # Measure expectation values
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def forward(self, x):
        """
        Forward pass through the hybrid quantum-classical network.

        Args:
            x: Input tensor of shape (batch_size, input_features)

        Returns:
            Output tensor of shape (batch_size, 1)
        """
        batch_size = x.shape[0]

        # Classical preprocessing
        x = torch.tanh(self.pre_net(x))  # tanh to bound inputs to [-1, 1]

        # Process each sample through quantum circuit
        q_outputs = []
        for i in range(batch_size):
            q_out = self.qnode(x[i], self.q_weights)
            q_outputs.append(torch.stack(q_out))

        q_outputs = torch.stack(q_outputs).float()  # Convert to float32 for PyTorch layers

        # Classical post-processing
        output = self.post_net(q_outputs)

        return output


class QNN_Amplitude(nn.Module):
    """
    Quantum Neural Network using Amplitude Embedding.

    Uses amplitude encoding to embed classical data into quantum state amplitudes,
    and measures probability distributions as output.

    Architecture:
        Input -> Normalize -> Amplitude Embedding -> StronglyEntanglingLayers -> Probs -> Output

    Note: Amplitude embedding requires input dimension to be 2^n_qubits.
    """

    def __init__(self, input_features, n_qubits=4, n_layers=2):
        super(QNN_Amplitude, self).__init__()

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.amplitude_dim = 2 ** n_qubits  # Amplitude embedding requires 2^n dimensions

        # Classical preprocessing to match amplitude embedding dimension
        self.pre_net = nn.Linear(input_features, self.amplitude_dim)

        # Create quantum device
        self.dev = qml.device("default.qubit", wires=n_qubits)

        # Initialize quantum circuit weights
        weight_shape = (n_layers, n_qubits, 3)
        self.q_weights = nn.Parameter(torch.randn(weight_shape) * 0.1)

        # Create quantum node
        self.qnode = qml.QNode(self._circuit, self.dev, interface="torch", diff_method="backprop")

        # Classical post-processing layer
        # Output from probs is 2^n_qubits dimensional
        self.post_net = nn.Linear(self.amplitude_dim, 1)

    def _normalize(self, x):
        """Normalize input vector for amplitude embedding (L2 norm = 1)."""
        norm = torch.norm(x, p=2, dim=-1, keepdim=True)
        # Avoid division by zero
        norm = torch.clamp(norm, min=1e-8)
        return x / norm

    def _circuit(self, inputs, weights):
        """
        Quantum circuit with amplitude embedding and strongly entangling layers.

        Args:
            inputs: Normalized input features (2^n_qubits,)
            weights: Trainable parameters (n_layers, n_qubits, 3)

        Returns:
            Probability distribution over computational basis states
        """
        # Encode classical data into quantum state amplitudes
        qml.AmplitudeEmbedding(inputs, wires=range(self.n_qubits), normalize=True)

        # Apply strongly entangling layers
        qml.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))

        # Return probabilities of all computational basis states
        return qml.probs(wires=range(self.n_qubits))

    def forward(self, x):
        """
        Forward pass through the quantum network.

        Args:
            x: Input tensor of shape (batch_size, input_features)

        Returns:
            Output tensor of shape (batch_size, 1)
        """
        batch_size = x.shape[0]

        # Classical preprocessing
        x = self.pre_net(x)
        x = self._normalize(x)

        # Process each sample through quantum circuit
        q_outputs = []
        for i in range(batch_size):
            q_out = self.qnode(x[i], self.q_weights)
            q_outputs.append(q_out)

        q_outputs = torch.stack(q_outputs).float()

        # Classical post-processing
        output = self.post_net(q_outputs)

        return output


# class QNN(nn.Module):
#     """
#     Enhanced Quantum Neural Network with Parallel Batch Processing 
#     and Data Re-uploading.

#     Architecture:
#         Input -> Classical Pre-net -> 
#         [Angle Embedding -> StronglyEntanglingLayers] x n_layers -> 
#         Measurement -> Classical Post-net -> Output
#     """

#     def __init__(self, input_features, n_qubits=4, n_layers=2, re_upload=True):
#         """
#         Args:
#             input_features (int): Dimension of input vector.
#             n_qubits (int): Number of qubits in the circuit.
#             n_layers (int): Number of quantum layers (depth).
#             re_upload (bool): If True, repeats the embedding before every layer.
#         """
#         super(QNN, self).__init__()

#         self.n_qubits = n_qubits
#         self.n_layers = n_layers
#         self.re_upload = re_upload

#         # 1. Classical Pre-processing
#         # Resizes input to match qubit count
#         self.pre_net = nn.Linear(input_features, n_qubits)

#         # 2. Quantum Device Configuration
#         # 'lightning.qubit' is faster (C++ backend). Fallback to 'default.qubit' if missing.
#         try:
#             self.dev = qml.device("lightning.qubit", wires=n_qubits)
#             diff_method = "adjoint" # Much faster for simulation
#         except:
#             self.dev = qml.device("default.qubit", wires=n_qubits)
#             diff_method = "backprop"
            
#         print(f"Using device: {self.dev.short_name} with diff_method: {diff_method}")

#         # 3. Initialize Weights
#         # Shape: (n_layers, n_qubits, 3) 
#         # We use uniform initialization for better convergence in QNNs
#         weight_shape = (n_layers, n_qubits, 3)
#         self.q_weights = nn.Parameter(torch.empty(weight_shape).uniform_(0, 2 * 3.1415))

#         # 4. Define QNode
#         self.qnode = qml.QNode(self._circuit, self.dev, interface="torch", diff_method=diff_method)

#         # 5. Classical Post-processing
#         self.post_net = nn.Linear(n_qubits, 1)


#     def _circuit(self, inputs, weights):
#         """
#         Quantum Circuit with Data Re-uploading support.
        
#         PennyLane automatically handles the batch dimension in 'inputs'.
#         """
#         # If re_upload is True, we interleave embedding and variational layers
#         if self.re_upload:
#             for i in range(self.n_layers):
#                 # Re-encode data
#                 qml.AngleEmbedding(inputs, wires=range(self.n_qubits), rotation='Y')
#                 # Apply one layer of ansatz
#                 # weights[i] has shape (1, n_qubits, 3), we need (1, n_qubits, 3) for the template
#                 # StronglyEntanglingLayers expects a 3D tensor of shape (L, M, 3). 
#                 # Since we iterate layer by layer, we unsqueeze the specific weight layer.
#                 w_layer = weights[i].unsqueeze(0) 
#                 qml.StronglyEntanglingLayers(w_layer, wires=range(self.n_qubits))
#         else:
#             # Standard single embedding (Original approach)
#             qml.AngleEmbedding(inputs, wires=range(self.n_qubits), rotation='Y')
#             qml.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))

#         # Measure all qubits in Z basis
#         return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]


#     def forward(self, x):
#         """
#         Vectorized Forward pass.
#         """
#         # Classical preprocessing (Bound to [-pi, pi] for AngleEmbedding)
#         x = torch.pi * torch.tanh(self.pre_net(x))

#         # Quantum Forward Pass
#         # NOTICE: No for-loop here. We pass the entire batch `x`.
#         # PennyLane returns shape (n_qubits, batch_size) or (n_qubits,) if batch=1
#         q_out = self.qnode(x, self.q_weights)
        
#         # Handle shape mismatch between PennyLane and PyTorch
#         if isinstance(q_out, tuple):
#              q_out = torch.stack(q_out) # (n_qubits, batch_size) -> Stack tuple to tensor

#         # If batching, we usually get (n_qubits, batch_size). We need (batch_size, n_qubits).
#         if q_out.ndim == 2:
#             q_out = q_out.T 
            
#         # Ensure float32 (sometimes QNodes return float64)
#         q_out = q_out.float()

#         # Classical post-processing
#         output = self.post_net(q_out)

#         return output


class HybridQNN(nn.Module):
    """
    Hybrid Quantum-Classical Neural Network with deeper classical layers.

    Architecture:
        Input -> Classical Encoder -> QNN Block -> Classical Decoder -> Output
    """

    def __init__(self, input_features, n_qubits=4, n_layers=2, hidden_dim=32):
        super(HybridQNN, self).__init__()

        self.n_qubits = n_qubits
        self.n_layers = n_layers

        # Classical encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, n_qubits),
            nn.Tanh()  # Bound to [-1, 1] for quantum encoding
        )

        # Quantum device and circuit
        self.dev = qml.device("default.qubit", wires=n_qubits)

        # Quantum weights
        weight_shape = (n_layers, n_qubits, 3)
        self.q_weights = nn.Parameter(torch.randn(weight_shape) * 0.1)

        # Create quantum node
        self.qnode = qml.QNode(self._circuit, self.dev, interface="torch", diff_method="backprop")

        # Classical decoder
        self.decoder = nn.Sequential(
            nn.Linear(n_qubits, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)
        )

    def _circuit(self, inputs, weights):
        """Quantum circuit with StronglyEntanglingLayers."""
        qml.AngleEmbedding(inputs, wires=range(self.n_qubits), rotation='Y')
        qml.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def forward(self, x):
        batch_size = x.shape[0]

        # Encode
        x = self.encoder(x)

        # Quantum processing
        q_outputs = []
        for i in range(batch_size):
            q_out = self.qnode(x[i], self.q_weights)
            q_outputs.append(torch.stack(q_out))

        q_outputs = torch.stack(q_outputs).float()  # Convert to float32 for PyTorch layers

        # Decode
        output = self.decoder(q_outputs)

        return output


class QNNClassifier(nn.Module):
    """
    QNN for binary classification tasks.

    Uses sigmoid activation on output for probability.
    """

    def __init__(self, input_features, n_qubits=4, n_layers=2):
        super(QNNClassifier, self).__init__()

        self.n_qubits = n_qubits
        self.n_layers = n_layers

        self.pre_net = nn.Linear(input_features, n_qubits)

        self.dev = qml.device("default.qubit", wires=n_qubits)

        weight_shape = (n_layers, n_qubits, 3)
        self.q_weights = nn.Parameter(torch.randn(weight_shape) * 0.1)

        self.qnode = qml.QNode(self._circuit, self.dev, interface="torch", diff_method="backprop")

        self.post_net = nn.Sequential(
            nn.Linear(n_qubits, 1),
            nn.Sigmoid()
        )

    def _circuit(self, inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(self.n_qubits), rotation='Y')
        qml.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def forward(self, x):
        batch_size = x.shape[0]

        x = torch.tanh(self.pre_net(x))

        q_outputs = []
        for i in range(batch_size):
            q_out = self.qnode(x[i], self.q_weights)
            q_outputs.append(torch.stack(q_out))

        q_outputs = torch.stack(q_outputs).float()  # Convert to float32 for PyTorch layers
        output = self.post_net(q_outputs)

        return output


# ============================================================================
# HQCNN - Hybrid Quantum Convolutional Neural Network
# ============================================================================

# QCNN helper functions
def _U_SU4(weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, wires):
    """SU(4) unitary gate for convolutional layer (15 params total)."""
    qml.U3(*weights_0, wires=wires[0])
    qml.U3(*weights_1, wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.RY(weights_2, wires=wires[0])
    qml.RZ(weights_3, wires=wires[1])
    qml.CNOT(wires=[wires[1], wires[0]])
    qml.RY(weights_4, wires=wires[0])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.U3(*weights_5, wires=wires[0])
    qml.U3(*weights_6, wires=wires[1])


def _Pooling_ansatz(weights_0, weights_1, wires):
    """Pooling ansatz circuit (2 params)."""
    qml.CRZ(weights_0, wires=[wires[0], wires[1]])
    qml.PauliX(wires=wires[0])
    qml.CRX(weights_1, wires=[wires[0], wires[1]])


# Default QCNN configuration for 8 qubits
_HQCNN_N_QUBITS = 8
_HQCNN_WEIGHT_SHAPES = {
    "weights_0": 3,
    "weights_1": 3,
    "weights_2": 1,
    "weights_3": 1,
    "weights_4": 1,
    "weights_5": 3,
    "weights_6": 3,
    "weights_7": 1,
    "weights_8": 1,
}
_HQCNN_POOLING_OUT = [1, 3, 5, 7]
_hqcnn_dev = qml.device("default.qubit", wires=_HQCNN_N_QUBITS)


@qml.qnode(_hqcnn_dev)
def _hqcnn_circuit(inputs, weights_0, weights_1, weights_2, weights_3, weights_4,
                   weights_5, weights_6, weights_7, weights_8):
    """
    QCNN circuit with convolutional and pooling layers.

    Architecture:
        - Angle Embedding (8 qubits)
        - Convolutional Layer 1 (U_SU4 gates)
        - Pooling Layer 1
        - Measurement on pooling output qubits [1, 3, 5, 7]
    """
    qml.AngleEmbedding(inputs, wires=range(_HQCNN_N_QUBITS))

    # Convolutional Layer 1
    _U_SU4(weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, wires=[0, 1])
    _U_SU4(weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, wires=[2, 3])
    _U_SU4(weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, wires=[4, 5])
    _U_SU4(weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, wires=[6, 7])

    _U_SU4(weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, wires=[1, 2])
    _U_SU4(weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, wires=[3, 4])
    _U_SU4(weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, wires=[5, 6])
    _U_SU4(weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, wires=[7, 0])

    # Pooling Layer 1
    _Pooling_ansatz(weights_7, weights_8, wires=[0, 1])
    _Pooling_ansatz(weights_7, weights_8, wires=[2, 3])
    _Pooling_ansatz(weights_7, weights_8, wires=[4, 5])
    _Pooling_ansatz(weights_7, weights_8, wires=[6, 7])

    return [qml.expval(qml.PauliZ(wires=i)) for i in _HQCNN_POOLING_OUT]


class HQCNN(nn.Module):
    """
    Hybrid Quantum Convolutional Neural Network (HQCNN).

    Uses a QCNN architecture with:
    - 8 qubits
    - Convolutional layers using U_SU4 gates
    - Pooling layers
    - Classical pre/post processing layers

    Architecture:
        Input -> Linear(input, 8) -> QCNN -> Linear(4, 1) -> Output
    """

    def __init__(self, input_features, num_layers=1):
        """
        Args:
            input_features: Number of input features
        """
        super(HQCNN, self).__init__()
        self.clayer_1 = nn.Linear(input_features, 8)
        self.qlayers = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.qlayers.append( qml.qnn.TorchLayer(_hqcnn_circuit, _HQCNN_WEIGHT_SHAPES))
        self.clayer_2 = nn.Linear(4, 1)

    def forward(self, x):
        x = self.clayer_1(x)
        for qlayer in self.qlayers:
            x = qlayer(x)
        x = self.clayer_2(x)
        return x


class QPDGNNDecoder(nn.Module):
    """Stage 2: GNN for PD prediction using PK predictions + covariates (Quantum version)."""

    def __init__(self, pk_embedding_dim, input_dim, hidden_dim=64, num_layers=3,
                 dropout=0.2, use_attention=False, use_gating=True, n_qubits=4, n_qlayers=1):
        super(QPDGNNDecoder, self).__init__()

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

        # PD predictor head (Quantum)
        self.pd_predictor = QNN(input_features=hidden_dim, n_qubits=n_qubits, n_layers=n_qlayers)

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


class HQGNN(nn.Module):
    """
    Hierarchical Quantum GNN for PK/PD prediction.
    
    Uses classical PK-GNN encoder and Quantum PD-GNN decoder.
    """

    def __init__(self, feature_dim, hidden_dim=64, num_layers_pk=3, num_layers_pd=3,
                 dropout=0.2, use_attention=False, use_gating=True, n_qubits=4, n_qlayers=1):
        super(HQGNN, self).__init__()

        self.pk_encoder = PKGNNEncoder(
            input_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers_pk,
            dropout=dropout,
            use_attention=use_attention
        )

        self.pd_decoder = QPDGNNDecoder(
            pk_embedding_dim=hidden_dim,
            input_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers_pd,
            dropout=dropout,
            use_attention=use_attention,
            use_gating=use_gating,
            n_qubits=n_qubits,
            n_qlayers=n_qlayers
        )

    def forward(self, data, return_pk=False):
        x, edge_index = data.x, data.edge_index
        edge_weight = data.edge_weight if hasattr(data, 'edge_weight') else None

        # Stage 1: PK prediction
        pk_embeddings, pk_predictions = self.pk_encoder(x, edge_index, edge_weight)

        # Stage 2: PD prediction
        pd_predictions = self.pd_decoder(x, pk_embeddings, pk_predictions, edge_index, edge_weight)

        if return_pk:
            return pd_predictions, pk_predictions
        return pd_predictions