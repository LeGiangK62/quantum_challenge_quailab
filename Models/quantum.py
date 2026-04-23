import torch
import torch.nn as nn
import torch.nn.functional as F
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

    def __init__(self, input_features, n_qubits=4, n_layers=2, q_dev=None):
        super(QNN, self).__init__()

        self.n_qubits = n_qubits
        self.n_layers = n_layers

        # Classical preprocessing layer to match input to n_qubits
        self.pre_net = nn.Linear(input_features, n_qubits)

        # Create quantum device (allow override for hardware/simulator backends)
        self.dev = q_dev if q_dev is not None else qml.device("default.qubit", wires=n_qubits)

        # Initialize quantum circuit weights
        # StronglyEntanglingLayers requires shape (n_layers, n_qubits, 3)
        weight_shape = (n_layers, n_qubits, 3)
        self.q_weights = nn.Parameter(torch.randn(weight_shape) * 0.1)

        # Create quantum node
        diff_method = "backprop" if q_dev is None else "best"
        self.qnode = qml.QNode(self._circuit, self.dev, interface="torch", diff_method=diff_method)

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

    def __init__(self, input_features, n_qubits=4, n_layers=2, q_dev=None):
        super(QNN_Amplitude, self).__init__()

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.amplitude_dim = 2 ** n_qubits  # Amplitude embedding requires 2^n dimensions

        # Classical preprocessing to match amplitude embedding dimension
        self.pre_net = nn.Linear(input_features, self.amplitude_dim)

        # Create quantum device (allow override for hardware/simulator backends)
        self.dev = q_dev if q_dev is not None else qml.device("default.qubit", wires=n_qubits)

        # Initialize quantum circuit weights
        weight_shape = (n_layers, n_qubits, 3)
        self.q_weights = nn.Parameter(torch.randn(weight_shape) * 0.1)

        # Create quantum node
        diff_method = "backprop" if q_dev is None else "best"
        self.qnode = qml.QNode(self._circuit, self.dev, interface="torch", diff_method=diff_method)

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


def _hqcnn_circuit_fn(inputs, weights_0, weights_1, weights_2, weights_3, weights_4,
                      weights_5, weights_6, weights_7, weights_8):
    """
    QCNN circuit body (no QNode decorator so it can be rebound to any device).

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


_hqcnn_circuit = qml.QNode(_hqcnn_circuit_fn, _hqcnn_dev)


class HQCNN(nn.Module):
    """
    Hybrid Quantum Convolutional Neural Network (HQCNN).

    Uses a QCNN architecture with:
    - 8 qubits
    - 8 qubits
    - Convolutional layers using U_SU4 gates
    - Pooling layers
    - Classical pre/post processing layers

    Architecture:
        Input -> Linear(input, 8) -> QCNN -> Linear(4, 1) -> Output
    """

    def __init__(self, input_features, num_layers=1, q_dev=None):
        """
        Args:
            input_features: Number of input features
            num_layers: Number of stacked QCNN layers
            q_dev: Optional PennyLane device. If None uses the module-level
                default (`default.qubit`). Pass a different device to run the
                same circuit on simulators or real hardware (e.g. qiskit.aer,
                qiskit.remote with an IBM backend).
        """
        super(HQCNN, self).__init__()
        self.clayer_1 = nn.Linear(input_features, 8)
        if q_dev is None:
            qnode = _hqcnn_circuit
        else:
            qnode = qml.QNode(_hqcnn_circuit_fn, q_dev, interface="torch")
        self.qlayers = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.qlayers.append(qml.qnn.TorchLayer(qnode, _HQCNN_WEIGHT_SHAPES))
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
                 dropout=0.2, use_attention=False, use_gating=True,
                 n_qlayers=1, n_qubits=4, using_hqcnn=False, q_dev=None):
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

        # PD predictor head
        if using_hqcnn:
            self.pd_predictor = HQCNN(input_features=hidden_dim, num_layers=n_qlayers, q_dev=q_dev)
        else:
            self.pd_predictor = QNN_Amplitude(
                input_features=hidden_dim,
                n_qubits=n_qubits,
                n_layers=n_qlayers,
                q_dev=q_dev,
            )

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


class LegacyQPDGNNDecoder(nn.Module):
    """
    Legacy Stage 2 decoder using QNN (for loading old checkpoints).
    Uses QNN with pre_net/q_weights/post_net structure.
    """

    def __init__(self, pk_embedding_dim, input_dim, hidden_dim=64, num_layers=3,
                 dropout=0.2, use_attention=False, use_gating=True):
        super(LegacyQPDGNNDecoder, self).__init__()

        self.use_attention = use_attention
        self.use_gating = use_gating

        combined_dim = pk_embedding_dim + input_dim + 1

        if use_gating:
            self.gate = nn.Sequential(
                nn.Linear(combined_dim, hidden_dim),
                nn.Sigmoid()
            )

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        if use_attention:
            self.convs.append(GATConv(combined_dim, hidden_dim, heads=4, concat=False))
        else:
            self.convs.append(GCNConv(combined_dim, hidden_dim))
        self.norms.append(LayerNorm(hidden_dim))

        for _ in range(num_layers - 1):
            if use_attention:
                self.convs.append(GATConv(hidden_dim, hidden_dim, heads=4, concat=False))
            else:
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.norms.append(LayerNorm(hidden_dim))

        self.dropout = dropout

        # Legacy QNN predictor (matches old checkpoint structure)
        self.pd_predictor = QNN(input_features=hidden_dim, n_qubits=4, n_layers=1)

        self.residual_branch = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

        self.residual_weight = nn.Parameter(torch.tensor(0.1))

    def forward(self, x, pk_embeddings, pk_predictions, edge_index, edge_weight=None):
        combined = torch.cat([x, pk_embeddings, pk_predictions], dim=-1)

        if self.use_gating:
            gate_values = self.gate(combined)

        h = combined
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            h_new = conv(h, edge_index, edge_weight=edge_weight)
            h_new = norm(h_new)
            h_new = torch.relu(h_new)

            if i == 0 and self.use_gating:
                h_new = h_new * gate_values

            if i < len(self.convs) - 1:
                h_new = torch.dropout(h_new, p=self.dropout, train=self.training)

            if i > 0 and h.size(-1) == h_new.size(-1):
                h = h + h_new
            else:
                h = h_new

        pd_main = self.pd_predictor(h)
        pd_residual = self.residual_branch(combined)
        pd_predictions = pd_main + self.residual_weight * pd_residual

        return pd_predictions


class LegacyHQGNN(nn.Module):
    """
    Legacy HQGNN for loading old checkpoints.
    Uses QNN instead of HQCNN for pd_predictor.
    """

    def __init__(self, feature_dim, hidden_dim=64, num_layers_pk=3, num_layers_pd=3,
                 dropout=0.2, use_attention=False, use_gating=True):
        super(LegacyHQGNN, self).__init__()

        self.pk_encoder = PKGNNEncoder(
            input_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers_pk,
            dropout=dropout,
            use_attention=use_attention
        )

        self.pd_decoder = LegacyQPDGNNDecoder(
            pk_embedding_dim=hidden_dim,
            input_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers_pd,
            dropout=dropout,
            use_attention=use_attention,
            use_gating=use_gating
        )

    def forward(self, data, return_pk=False):
        x, edge_index = data.x, data.edge_index
        edge_weight = data.edge_weight if hasattr(data, 'edge_weight') else None

        pk_embeddings, pk_predictions = self.pk_encoder(x, edge_index, edge_weight)
        pd_predictions = self.pd_decoder(x, pk_embeddings, pk_predictions, edge_index, edge_weight)

        if return_pk:
            return pd_predictions, pk_predictions
        return pd_predictions


class HQGNN(nn.Module):
    """
    Hierarchical Quantum GNN for PK/PD prediction.

    Uses classical PK-GNN encoder and Quantum PD-GNN decoder with HQCNN circuit.
    """

    def __init__(self, feature_dim, hidden_dim=64, num_layers_pk=3, num_layers_pd=3,
                 dropout=0.2, use_attention=False, use_gating=True,
                 n_qlayers=1, n_qubits=4, using_hqcnn=False, q_dev=None):
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
            n_qlayers=n_qlayers,
            n_qubits=n_qubits,
            using_hqcnn=using_hqcnn,
            q_dev=q_dev,
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
    

# ============================================================
# Hierarchical HQCNN Wrapper
# ============================================================
class HierarchicalHQCNN(nn.Module):
    """
    Hierarchical HQCNN for PK/PD prediction.

    Uses separate HQCNN models for PK and PD prediction.
    PD model receives PK prediction as additional input.
    """

    def __init__(self, pk_input_dim, pd_input_dim, num_layers=1, mode='dual_stage', q_dev=None):
        super().__init__()
        self.mode = mode

        # Separate HQCNN for PK and PD
        self.pk_model = HQCNN(pk_input_dim, num_layers=num_layers, q_dev=q_dev)
        self.pd_model = HQCNN(pd_input_dim + 1, num_layers=num_layers, q_dev=q_dev)  # +1 for PK prediction

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
# Hierarchical QNN (Amplitude Embedding) Wrapper
# ============================================================
class HierarchicalQNN(nn.Module):
    """
    Hierarchical QNN for PK/PD prediction.

    Uses QNN_Amplitude (amplitude embedding + StronglyEntanglingLayers)
    for both PK and PD prediction. Same structure as HierarchicalHQCNN
    but with QNN_Amplitude instead of HQCNN.
    """

    def __init__(self, pk_input_dim, pd_input_dim, n_qubits=4, n_qlayers=2, mode='dual_stage', q_dev=None):
        super().__init__()
        self.mode = mode

        self.pk_model = QNN_Amplitude(pk_input_dim, n_qubits=n_qubits, n_layers=n_qlayers, q_dev=q_dev)
        self.pd_model = QNN_Amplitude(pd_input_dim + 1, n_qubits=n_qubits, n_layers=n_qlayers, q_dev=q_dev)

    def forward(self, x_pk=None, x_pd=None):
        results = {}

        if x_pk is not None:
            pk_pred = self.pk_model(x_pk)
            results['pk'] = pk_pred

        if x_pd is not None:
            if self.mode == 'dual_stage' and 'pk' in results:
                pk_for_pd = results['pk']
            elif self.mode == 'joint' and 'pk' in results:
                pk_for_pd = results['pk'].detach()
            else:
                pk_for_pd = torch.zeros(x_pd.size(0), 1, device=x_pd.device)

            x_pd_with_pk = torch.cat([x_pd, pk_for_pd], dim=1)
            pd_pred = self.pd_model(x_pd_with_pk)
            results['pd'] = pd_pred

        return results


# ============================================================
# HQLSTM - Hierarchical Quantum LSTM
# ============================================================
class QLSTMEncoder(nn.Module):
    """LSTM encoder with quantum output layer."""

    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.3,
                 bidirectional=True, n_qlayers=1, n_qubits=4, using_hqcnn=False, q_dev=None):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        self.layer_norm = nn.LayerNorm(hidden_dim * self.num_directions)
        self.dropout = nn.Dropout(dropout)

        lstm_out_dim = hidden_dim * self.num_directions

        # Quantum predictor
        if using_hqcnn:
            self.predictor = HQCNN(lstm_out_dim, num_layers=n_qlayers, q_dev=q_dev)
        else:
            self.predictor = QNN_Amplitude(
                input_features=lstm_out_dim,
                n_qubits=n_qubits,
                n_layers=n_qlayers,
                q_dev=q_dev,
            )

        # Classical residual branch (same as PD decoder)
        self.residual_branch = nn.Sequential(
            nn.Linear(lstm_out_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.residual_weight = nn.Parameter(torch.tensor(0.1))

    def forward(self, x, lengths=None):
        """
        Args:
            x: [batch, seq_len, input_dim]
            lengths: Sequence lengths

        Returns:
            embeddings: [batch, seq_len, hidden_dim * num_directions]
            predictions: [batch, seq_len, 1]
        """
        x = torch.relu(self.input_proj(x))

        if lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )

        outputs, _ = self.lstm(x)

        if lengths is not None:
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

        outputs = self.layer_norm(outputs)
        outputs = self.dropout(outputs)

        # Apply quantum predictor to each timestep
        batch_size, seq_len, hidden_size = outputs.shape
        flat_outputs = outputs.reshape(-1, hidden_size)

        pk_main = self.predictor(flat_outputs)
        pk_main = pk_main.reshape(batch_size, seq_len, 1)

        # Classical residual
        pk_residual = self.residual_branch(flat_outputs)
        pk_residual = pk_residual.reshape(batch_size, seq_len, 1)

        predictions = pk_main + self.residual_weight * pk_residual

        return outputs, predictions


class QPDLSTMDecoder(nn.Module):
    """LSTM decoder with quantum output for PD prediction."""

    def __init__(self, input_dim, pk_embedding_dim, hidden_dim=128, num_layers=2,
                 dropout=0.3, bidirectional=True, use_gating=True, n_qlayers=1, n_qubits=4, using_hqcnn=False, q_dev=None):
        super().__init__()

        self.use_gating = use_gating
        self.num_directions = 2 if bidirectional else 1

        combined_dim = input_dim + pk_embedding_dim + 1

        if use_gating:
            self.gate = nn.Sequential(
                nn.Linear(combined_dim, combined_dim),
                nn.Sigmoid()
            )

        self.input_proj = nn.Linear(combined_dim, hidden_dim)

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        self.layer_norm = nn.LayerNorm(hidden_dim * self.num_directions)
        self.dropout = nn.Dropout(dropout)

        # Quantum predictor
        if using_hqcnn:
            self.predictor = HQCNN(hidden_dim * self.num_directions, num_layers=n_qlayers, q_dev=q_dev)
        else:
            self.predictor = QNN_Amplitude(
                input_features=hidden_dim * self.num_directions,
                n_qubits=n_qubits,
                n_layers=n_qlayers,
                q_dev=q_dev,
            )

        # Residual branch (classical)
        self.residual_branch = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.residual_weight = nn.Parameter(torch.tensor(0.1))

    def forward(self, x, pk_embeddings, pk_predictions, lengths=None):
        combined = torch.cat([x, pk_embeddings, pk_predictions], dim=-1)

        if self.use_gating:
            gate_values = self.gate(combined)
            combined = combined * gate_values

        h = torch.relu(self.input_proj(combined))

        if lengths is not None:
            h = nn.utils.rnn.pack_padded_sequence(
                h, lengths.cpu(), batch_first=True, enforce_sorted=False
            )

        outputs, _ = self.lstm(h)

        if lengths is not None:
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

        outputs = self.layer_norm(outputs)
        outputs = self.dropout(outputs)

        # Quantum prediction
        batch_size, seq_len, hidden_size = outputs.shape
        flat_outputs = outputs.reshape(-1, hidden_size)
        pd_main = self.predictor(flat_outputs)
        pd_main = pd_main.reshape(batch_size, seq_len, 1)

        # Residual
        flat_combined = combined.reshape(-1, combined.shape[-1])
        pd_residual = self.residual_branch(flat_combined)
        pd_residual = pd_residual.reshape(batch_size, seq_len, 1)

        return pd_main + self.residual_weight * pd_residual


class HQLSTM(nn.Module):
    """
    Hierarchical Quantum LSTM for PK/PD prediction.

    Uses classical LSTM with quantum (HQCNN) output layers.
    """

    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.3,
                 bidirectional=True, use_gating=True, mode='dual_stage', n_qlayers=1, n_qubits=4, using_hqcnn=False, q_dev=None):
        super().__init__()

        self.mode = mode
        self.hidden_dim = hidden_dim
        self.num_directions = 2 if bidirectional else 1

        # PK encoder
        self.pk_encoder = QLSTMEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            n_qlayers=n_qlayers,
            n_qubits=n_qubits,
            using_hqcnn=using_hqcnn,
            q_dev=q_dev,
        )

        pk_output_dim = hidden_dim * self.num_directions

        # PD decoder
        self.pd_decoder = QPDLSTMDecoder(
            input_dim=input_dim,
            pk_embedding_dim=pk_output_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            use_gating=use_gating,
            n_qlayers=n_qlayers,
            n_qubits=n_qubits,
            using_hqcnn=using_hqcnn,
            q_dev=q_dev,
        )

    def forward(self, x_pk=None, x_pd=None, lengths_pk=None, lengths_pd=None, return_pk=False):
        results = {}

        # PK prediction
        if x_pk is not None:
            pk_embeddings, pk_predictions = self.pk_encoder(x_pk, lengths_pk)
            results['pk'] = pk_predictions
            results['pk_embeddings'] = pk_embeddings

        # PD prediction
        if x_pd is not None:
            batch_size, pd_seq_len, _ = x_pd.shape
            device = x_pd.device

            if 'pk_embeddings' in results:
                pk_emb = results['pk_embeddings']
                pk_pred = results['pk']

                if self.mode == 'joint':
                    pk_emb = pk_emb.detach()
                    pk_pred = pk_pred.detach()

                # Align PK sequence length to PD sequence length if they differ
                pk_seq_len = pk_emb.shape[1]
                if pk_seq_len != pd_seq_len:
                    pk_emb = F.interpolate(
                        pk_emb.transpose(1, 2), size=pd_seq_len, mode='linear', align_corners=False
                    ).transpose(1, 2)
                    pk_pred = F.interpolate(
                        pk_pred.transpose(1, 2), size=pd_seq_len, mode='linear', align_corners=False
                    ).transpose(1, 2)
            else:
                pk_emb = torch.zeros(batch_size, pd_seq_len, self.hidden_dim * self.num_directions, device=device)
                pk_pred = torch.zeros(batch_size, pd_seq_len, 1, device=device)

            pd_predictions = self.pd_decoder(x_pd, pk_emb, pk_pred, lengths_pd)
            results['pd'] = pd_predictions

        if 'pk_embeddings' in results:
            del results['pk_embeddings']

        if return_pk and 'pk' in results:
            return results['pd'], results['pk']

        return results