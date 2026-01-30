"""
Dose prediction and optimization using trained hierarchical models.
Supports: MLP, GNN, HQCNN, HQGNN
"""

import argparse
import numpy as np
import pandas as pd
import torch
import os
from glob import glob
from sklearn.preprocessing import StandardScaler

# ============================================================
# SETTINGS
# ============================================================
DATA_CSV = "Data/QIC2025-EstDat.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Biomarker threshold - model-specific defaults
# MLP predicts lower absolute PD values, GNN predicts higher
# These are calibrated based on model outputs
BIOMARKER_THRESHOLD_MLP = 3.3    # For MLP/HQCNN
BIOMARKER_THRESHOLD_GNN = 6.0    # For GNN/HQGNN (higher scale)
BIOMARKER_THRESHOLD = 3.3        # Default fallback

DAILY_TAU_H = 24
WEEKLY_TAU_H = 168

DAILY_STEP_MG = 0.5
WEEKLY_STEP_MG = 5

# Simulation duration: 1200h (50 days)
# Check if PD ever reaches threshold within this duration
N_CYCLES_DAILY = 50   # 50 * 24 = 1200h
N_CYCLES_WEEKLY = 7   # 7 * 168 = 1176h (close to 1200h)

TIME_WINDOWS = [24, 48, 72, 96, 120, 144, 168]
HALF_LIVES = [24, 48, 72]


# ============================================================
# ARGUMENT PARSING
# ============================================================
def get_args():
    parser = argparse.ArgumentParser(description="Dose prediction with trained models")
    parser.add_argument("--model", type=str, required=True,
                        choices=["mlp", "gnn", "lstm", "hqcnn", "hqgnn", "hqlstm"],
                        help="Model type to use")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to model checkpoint (model.pth)")
    parser.add_argument("--results_dir", type=str, default="Results",
                        help="Directory containing experiment results")
    parser.add_argument("--experiment", type=str, default=None,
                        help="Experiment name prefix to find model")
    parser.add_argument("--hidden_dim", type=int, default=256,
                        help="Hidden dimension (for MLP/HQCNN)")
    parser.add_argument("--gnn_hidden_dim", type=int, default=64,
                        help="Hidden dimension (for GNN/HQGNN)")
    parser.add_argument("--lstm_hidden_dim", type=int, default=128,
                        help="Hidden dimension (for LSTM/HQLSTM)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Biomarker threshold (auto-selected per model if not specified: "
                             f"MLP/HQCNN={BIOMARKER_THRESHOLD_MLP}, GNN/HQGNN={BIOMARKER_THRESHOLD_GNN})")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode to show predictions")
    parser.add_argument("--criterion", type=str, default="ever",
                        choices=["ever", "trough", "average", "max"],
                        help="Evaluation criterion: ever (PD ever reaches threshold), "
                             "trough (Ctrough_ss), average (Cavg_ss), max (Cmax_ss). Default: 'ever'")
    return parser.parse_args()


# ============================================================
# MODEL LOADING
# ============================================================
def find_latest_model(results_dir, model_type, experiment=None):
    """Find the latest model checkpoint for the given model type."""
    if experiment:
        pattern = os.path.join(results_dir, f"{experiment}*/model.pth")
        matches = glob(pattern)
    else:
        # Try both naming conventions: "{model_type}_*" and "*_{model_type}_*"
        pattern1 = os.path.join(results_dir, f"{model_type}*/model.pth")
        pattern2 = os.path.join(results_dir, f"*_{model_type}_*/model.pth")
        matches = glob(pattern1) + glob(pattern2)

    if not matches:
        raise FileNotFoundError(f"No model found for {model_type} in {results_dir}")

    # Sort by modification time, get latest
    latest = max(matches, key=os.path.getmtime)
    print(f"Found model: {latest}")
    return latest


def prepare_scaler(df, model_type="mlp"):
    """
    Prepare feature scaler from data (matching training preprocessing).
    """
    if model_type in ["mlp", "hqcnn", "lstm", "hqlstm"]:
        from Utils.data_loader import engineer_dose_features, build_feature_list

        # Get dose events
        df_dose = df[df['EVID'] == 1].copy()
        df_obs = df[(df['EVID'] == 0) & (df['MDV'] == 0)].copy()

        # Engineer features
        df_final = engineer_dose_features(df_obs, df_dose, TIME_WINDOWS, HALF_LIVES)

        # Get feature list
        features = build_feature_list(TIME_WINDOWS, HALF_LIVES, add_decay=True, use_perkg=False)

        # Fit scaler on all data (for inference, we use all available data)
        scaler_X = StandardScaler()
        scaler_X.fit(df_final[features].values)

        print(f"Fitted MLP scaler on {len(df_final)} observations with {len(features)} features")
        return scaler_X, features

    else:  # gnn, hqgnn
        # GNN uses different features: TIME, BW, DOSE, COMED, DECAY_HL*, TIME_LOG, TIME_SQUARED
        df_obs = df[(df['EVID'] == 0)].copy()
        if 'MDV' in df_obs.columns:
            df_obs = df_obs[df_obs['MDV'] == 0]

        # Compute GNN features
        gnn_features = ['TIME', 'BW', 'DOSE', 'COMED']
        for hl in HALF_LIVES:
            col = f'DECAY_HL{hl}h'
            df_obs[col] = np.exp(-np.log(2) / hl * df_obs['TIME'])
            gnn_features.append(col)

        df_obs['TIME_LOG'] = np.log1p(df_obs['TIME'])
        df_obs['TIME_SQUARED'] = df_obs['TIME'] ** 2
        gnn_features.extend(['TIME_LOG', 'TIME_SQUARED'])

        # Fit scaler
        scaler_X = StandardScaler()
        scaler_X.fit(df_obs[gnn_features].values)

        print(f"Fitted GNN scaler on {len(df_obs)} observations with {len(gnn_features)} features")
        return scaler_X, gnn_features


def load_model(args):
    """Load model based on type."""
    # Determine model path
    if args.model_path:
        model_path = args.model_path
    else:
        model_path = find_latest_model(args.results_dir, args.model, args.experiment)

    # Get feature dimension (21 base features)
    n_features = 11 + len(TIME_WINDOWS) + len(HALF_LIVES)  # 21

    if args.model == "mlp":
        from Models.mlp import HierarchicalPKPDMLP
        model = HierarchicalPKPDMLP(
            pk_input_dim=n_features,
            pd_input_dim=n_features,
            hidden_dim=args.hidden_dim,
            mode='dual_stage'
        )
    elif args.model == "hqcnn":
        from main import HierarchicalHQCNN
        model = HierarchicalHQCNN(
            pk_input_dim=n_features,
            pd_input_dim=n_features,
            mode='dual_stage'
        )
    elif args.model == "gnn":
        from Models.gnn import HierarchicalPKPDGNN
        # GNN feature dim from saved model - look for pk_encoder first conv layer
        checkpoint = torch.load(model_path, map_location="cpu")
        # Find the pk_encoder's first conv layer weight to get feature_dim
        pk_conv_key = 'pk_encoder.convs.0.lin.weight'
        if pk_conv_key in checkpoint:
            # Shape is [hidden_dim, feature_dim]
            feature_dim = checkpoint[pk_conv_key].shape[1]
        else:
            # Fallback: try to find any convs.0 weight
            conv_keys = [k for k in checkpoint.keys() if 'convs.0' in k and 'weight' in k]
            if conv_keys:
                feature_dim = checkpoint[conv_keys[0]].shape[1]
            else:
                raise ValueError(f"Cannot infer feature_dim from checkpoint: {model_path}")

        print(f"Inferred feature_dim={feature_dim} from checkpoint")
        model = HierarchicalPKPDGNN(
            feature_dim=feature_dim,
            hidden_dim=args.gnn_hidden_dim
        )
    elif args.model == "hqgnn":
        from Models.quantum import HQGNN, LegacyHQGNN
        checkpoint = torch.load(model_path, map_location="cpu")
        pk_conv_key = 'pk_encoder.convs.0.lin.weight'
        if pk_conv_key in checkpoint:
            feature_dim = checkpoint[pk_conv_key].shape[1]
        else:
            conv_keys = [k for k in checkpoint.keys() if 'convs.0' in k and 'weight' in k]
            if conv_keys:
                feature_dim = checkpoint[conv_keys[0]].shape[1]
            else:
                raise ValueError(f"Cannot infer feature_dim from checkpoint: {model_path}")

        print(f"Inferred feature_dim={feature_dim} from checkpoint")

        # Detect legacy vs new architecture by checking pd_predictor keys
        is_legacy = 'pd_decoder.pd_predictor.q_weights' in checkpoint
        if is_legacy:
            print("Detected legacy HQGNN checkpoint (QNN-based), using LegacyHQGNN")
            model = LegacyHQGNN(
                feature_dim=feature_dim,
                hidden_dim=args.gnn_hidden_dim
            )
        else:
            model = HQGNN(
                feature_dim=feature_dim,
                hidden_dim=args.gnn_hidden_dim
            )
    elif args.model == "lstm":
        from Models.lstm import HierarchicalPKPDLSTM
        model = HierarchicalPKPDLSTM(
            input_dim=n_features,
            hidden_dim=args.lstm_hidden_dim,
            mode='dual_stage'
        )
    elif args.model == "hqlstm":
        from Models.quantum import HQLSTM
        model = HQLSTM(
            input_dim=n_features,
            hidden_dim=args.lstm_hidden_dim,
            mode='dual_stage'
        )

    # Load weights
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.to(DEVICE).eval()

    print(f"Loaded {args.model.upper()} model from {model_path}")
    return model


# ============================================================
# FEATURE ENGINEERING
# ============================================================
def compute_features_for_observation(
    time: float,
    bw: float,
    comed: float,
    dose_history: list,  # List of (time, amount) tuples
):
    """
    Compute features for a single observation point.

    Args:
        time: Current observation time
        bw: Body weight
        comed: Concomitant medication flag
        dose_history: List of (dose_time, dose_amount) prior to this time

    Returns:
        numpy array of features
    """
    # Filter doses before current time
    past_doses = [(t, amt) for t, amt in dose_history if t <= time]

    if len(past_doses) == 0:
        # No prior doses
        dose = 0.0
        tsld = 0.0
        last_dose_time = 0.0
        last_dose_amt = 0.0
        n_doses = 0
        cum_dose = 0.0
    else:
        last_t, last_amt = past_doses[-1]
        dose = last_amt  # Current dose level
        tsld = time - last_t
        last_dose_time = last_t
        last_dose_amt = last_amt
        n_doses = len(past_doses)
        cum_dose = sum(amt for _, amt in past_doses)

    # Time transformations
    time_squared = time ** 2
    time_log = np.log1p(time)

    # Window features
    window_features = []
    for window in TIME_WINDOWS:
        window_sum = sum(amt for t, amt in past_doses if time - window <= t <= time)
        window_features.append(window_sum)

    # Decay features
    decay_features = []
    for hl in HALF_LIVES:
        if tsld > 0:
            k = np.log(2) / hl
            decay = np.exp(-k * tsld)
        else:
            decay = 0.0
        decay_features.append(decay)

    # Combine all features (order must match training)
    features = [
        bw, comed, dose, time,
        tsld, last_dose_time, last_dose_amt,
        n_doses, cum_dose,
        time_squared, time_log
    ]
    features.extend(window_features)
    features.extend(decay_features)

    return np.array(features, dtype=np.float32)


# ============================================================
# SIMULATION FOR MLP/HQCNN
# ============================================================
def build_steady_state_features(
    bw: float,
    comed: float,
    dose_mg: float,
    tau_h: float,
    n_cycles: int,
    scaler=None
):
    """
    Build features for steady-state simulation.

    Args:
        bw: Body weight
        comed: Concomitant medication
        dose_mg: Dose amount in mg
        tau_h: Dosing interval in hours
        n_cycles: Number of dosing cycles
        scaler: Optional StandardScaler for normalization

    Returns:
        Tensor of features for the last cycle observation points
    """
    # Build dose history
    dose_times = np.arange(0.0, n_cycles * tau_h, tau_h)
    dose_history = [(t, dose_mg) for t in dose_times]

    # Observation times in last cycle
    t_start_last = (n_cycles - 1) * tau_h
    t_end = n_cycles * tau_h
    obs_times = np.arange(t_start_last + 0.5, t_end, 0.5)  # Every 0.5h

    # Compute features for each observation
    features_list = []
    for t in obs_times:
        feat = compute_features_for_observation(t, bw, comed, dose_history)
        features_list.append(feat)

    features = np.stack(features_list)

    if scaler is not None:
        features = scaler.transform(features)

    return torch.FloatTensor(features).to(DEVICE), obs_times


@torch.no_grad()
def simulate_subject_mlp(
    model,
    bw: float,
    comed: float,
    dose_mg: float,
    tau_h: float,
    n_cycles: int,
    scaler=None
):
    """Simulate PD for a subject using MLP/HQCNN model."""
    features, _ = build_steady_state_features(
        bw, comed, dose_mg, tau_h, n_cycles, scaler
    )

    # Model expects both PK and PD inputs
    # For PD prediction, we first get PK, then use it for PD
    results = model(features, features)

    pd_pred = results['pd'].cpu().numpy().flatten()
    return pd_pred


@torch.no_grad()
def simulate_subject_lstm(
    model,
    bw: float,
    comed: float,
    dose_mg: float,
    tau_h: float,
    n_cycles: int,
    scaler=None
):
    """Simulate PD for a subject using LSTM/HQLSTM model."""
    features, _ = build_steady_state_features(
        bw, comed, dose_mg, tau_h, n_cycles, scaler
    )

    # LSTM expects [batch, seq_len, features]
    features_seq = features.unsqueeze(0)  # [1, seq_len, n_features]
    lengths = torch.tensor([features.shape[0]])

    # Model forward
    results = model(
        x_pk=features_seq,
        x_pd=features_seq,
        lengths_pk=lengths,
        lengths_pd=lengths
    )

    pd_pred = results['pd'].cpu().numpy().flatten()
    return pd_pred


# ============================================================
# SIMULATION FOR GNN/HQGNN - Using Real Patient Data
# ============================================================

# Cache for patient graphs (built once, reused)
_patient_graphs_cache = None

def build_patient_graphs_from_data(df, half_lives=HALF_LIVES):
    """
    Build patient graphs from actual data (same as training).
    Returns dict: patient_id -> graph data
    """
    try:
        from torch_geometric.data import Data
    except ImportError:
        raise ImportError("torch_geometric required for GNN models")

    # Filter observations
    df = df[df['EVID'] == 0].copy()
    if 'MDV' in df.columns:
        df = df[df['MDV'] == 0]

    # Feature engineering (same as training)
    base_features = ['TIME', 'BW', 'DOSE', 'COMED']
    for hl in half_lives:
        df[f'DECAY_HL{hl}h'] = np.exp(-np.log(2) / hl * df['TIME'])
        base_features.append(f'DECAY_HL{hl}h')

    df['TIME_LOG'] = np.log1p(df['TIME'])
    df['TIME_SQUARED'] = df['TIME'] ** 2
    base_features.extend(['TIME_LOG', 'TIME_SQUARED'])

    patient_graphs = {}

    for patient_id in df['ID'].unique():
        patient_df = df[df['ID'] == patient_id].sort_values('TIME').reset_index(drop=True)

        pk_obs = patient_df[patient_df['DVID'] == 1]
        pd_obs = patient_df[patient_df['DVID'] == 2]

        if len(pd_obs) == 0:
            continue

        # Build nodes
        node_features = []
        pk_targets, pd_targets = [], []
        times = []
        pk_indices, pd_indices = [], []
        node_idx = 0

        # PK nodes
        for _, row in pk_obs.iterrows():
            features = [row[f] for f in base_features]
            node_features.append(features)
            pk_targets.append(row['DV'])
            pd_targets.append(0)
            times.append(row['TIME'])
            pk_indices.append(node_idx)
            node_idx += 1

        # PD nodes
        for _, row in pd_obs.iterrows():
            features = [row[f] for f in base_features]
            node_features.append(features)
            pk_targets.append(0)
            pd_targets.append(row['DV'])
            times.append(row['TIME'])
            pd_indices.append(node_idx)
            node_idx += 1

        # Build edges (same as training)
        edges = []
        edge_weights = []
        times_arr = np.array(times)

        # PK-PK temporal
        for i in range(len(pk_indices) - 1):
            src, dst = pk_indices[i], pk_indices[i + 1]
            time_diff = abs(times_arr[dst] - times_arr[src])
            weight = np.exp(-time_diff / 24.0)
            edges.extend([[src, dst], [dst, src]])
            edge_weights.extend([weight, weight])

        # PD-PD temporal
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

        patient_graphs[patient_id] = {
            'node_features': np.array(node_features, dtype=np.float32),
            'edge_index': np.array(edges, dtype=np.int64).T if edges else np.array([[], []], dtype=np.int64),
            'edge_weights': np.array(edge_weights, dtype=np.float32),
            'pk_indices': pk_indices,
            'pd_indices': pd_indices,
            'times': times_arr,
            'bw': patient_df['BW'].iloc[0],
            'comed': patient_df['COMED'].iloc[0],
            'original_dose': patient_df['DOSE'].iloc[0],
        }

    return patient_graphs, base_features


@torch.no_grad()
def simulate_subject_gnn(
    model,
    bw: float,
    comed: float,
    dose_mg: float,
    tau_h: float,
    n_cycles: int,
    scaler=None,
    patient_graph=None
):
    """
    Predict PD for a subject using GNN/HQGNN model with REAL patient graph.

    Uses actual patient graph structure, only modifies DOSE feature.
    """
    try:
        from torch_geometric.data import Data
    except ImportError:
        raise ImportError("torch_geometric required for GNN models")

    if patient_graph is None:
        return np.array([])

    # Copy node features and modify DOSE (index 2 in feature list)
    node_features = patient_graph['node_features'].copy()
    node_features[:, 2] = dose_mg  # Replace DOSE for all nodes

    if scaler is not None:
        node_features = scaler.transform(node_features)

    # Build PyG data
    n_nodes = len(node_features)
    x = torch.FloatTensor(node_features)
    edge_index = torch.LongTensor(patient_graph['edge_index'])
    edge_weight = torch.FloatTensor(patient_graph['edge_weights'])

    pk_mask = torch.zeros(n_nodes, dtype=torch.bool)
    pd_mask = torch.zeros(n_nodes, dtype=torch.bool)
    for idx in patient_graph['pk_indices']:
        pk_mask[idx] = True
    for idx in patient_graph['pd_indices']:
        pd_mask[idx] = True

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_weight=edge_weight,
        pk_mask=pk_mask,
        pd_mask=pd_mask,
        pk_targets=torch.zeros(n_nodes),
        pd_targets=torch.zeros(n_nodes),
        times=torch.FloatTensor(patient_graph['times'])
    ).to(DEVICE)

    # Get predictions
    pd_pred = model(data, return_pk=False)

    return pd_pred[pd_mask].cpu().numpy().flatten()


# ============================================================
# EVALUATION FUNCTIONS
# ============================================================
def evaluate_dose_population(
    model,
    model_type: str,
    df: pd.DataFrame,
    dose_mg: float,
    tau_h: float,
    n_cycles: int,
    scaler=None,
    threshold: float = BIOMARKER_THRESHOLD,
    debug: bool = False,
    criterion: str = "ever",
    patient_graphs: dict = None
):
    """
    Evaluate a dose on the population.

    Args:
        criterion: "ever" (PD ever reaches below threshold in 1200h),
                   "trough" (last value), "average" (Cavg), "max" (Cmax)
        patient_graphs: Pre-built patient graphs for GNN (from build_patient_graphs_from_data)

    Returns the fraction of subjects with biomarker < threshold.
    """
    # Only use subjects with non-zero dose history
    dosed_subjects = df[df["DOSE"] > 0]["ID"].unique()

    success = 0
    total = 0
    all_values = []  # Store criterion values for debug

    for sid in dosed_subjects:
        sdf = df[df["ID"] == sid]
        bw = float(sdf["BW"].iloc[0])
        comed = float(sdf["COMED"].iloc[0])

        if model_type in ["mlp", "hqcnn"]:
            biomarker = simulate_subject_mlp(
                model, bw, comed, dose_mg, tau_h, n_cycles, scaler
            )
        elif model_type in ["lstm", "hqlstm"]:
            biomarker = simulate_subject_lstm(
                model, bw, comed, dose_mg, tau_h, n_cycles, scaler
            )
        else:  # gnn, hqgnn
            # Use real patient graph
            patient_graph = patient_graphs.get(sid) if patient_graphs else None
            biomarker = simulate_subject_gnn(
                model, bw, comed, dose_mg, tau_h, n_cycles, scaler,
                patient_graph=patient_graph
            )

        if len(biomarker) == 0:
            continue

        # Calculate criterion value
        if criterion == "ever":
            # Check if PD EVER reaches below threshold (use min value)
            value = np.min(biomarker)
        elif criterion == "trough":
            # Trough = last value in the dosing interval (Ctrough_ss)
            value = biomarker[-1]
        elif criterion == "average":
            # Average = mean over the interval (Cavg_ss)
            value = np.mean(biomarker)
        elif criterion == "max":
            # Max = peak value (Cmax_ss)
            value = np.max(biomarker)
        else:
            value = np.min(biomarker)  # Default to ever

        all_values.append(value)
        total += 1
        if value < threshold:
            success += 1

    if debug and len(all_values) > 0:
        vals = np.array(all_values)
        print(f"    [DEBUG] {criterion.upper()}: min={vals.min():.2f}, max={vals.max():.2f}, "
              f"mean={vals.mean():.2f}, std={vals.std():.2f}")
        print(f"    [DEBUG] Below threshold ({threshold}): {(vals < threshold).sum()}/{len(vals)} "
              f"= {(vals < threshold).sum()/len(vals)*100:.1f}%")

    return success / total if total > 0 else 0.0


def find_min_dose(
    model,
    model_type: str,
    df: pd.DataFrame,
    tau_h: float,
    n_cycles: int,
    target_prob: float,
    dose_step: float,
    dose_min: float,
    dose_max: float,
    scaler=None,
    threshold: float = BIOMARKER_THRESHOLD,
    debug: bool = False,
    criterion: str = "trough",
    patient_graphs: dict = None
):
    """Find minimum dose to achieve target probability."""
    doses = np.arange(dose_min, dose_max + 1e-9, dose_step)

    for d in doses:
        rate = evaluate_dose_population(
            model, model_type, df, d, tau_h, n_cycles, scaler,
            threshold=threshold, debug=debug, criterion=criterion,
            patient_graphs=patient_graphs
        )
        print(f"  Dose {d:.2f} mg -> {rate*100:.1f}% subjects ({criterion}) < {threshold}")

        if rate >= target_prob:
            return d, rate

    return None, None


# ============================================================
# MAIN TASKS
# ============================================================
def run_all_tasks(model, model_type, df, scaler=None, threshold=BIOMARKER_THRESHOLD,
                   debug=False, criterion="trough"):
    """Run all 5 dose-finding tasks."""

    max_dose_data = float(pd.to_numeric(df["DOSE"], errors="coerce").max())
    if np.isnan(max_dose_data):
        max_dose_data = 10.0

    daily_max = max(10.0, max_dose_data * 2)
    weekly_max = max(70.0, max_dose_data * 10)

    results = {}

    # Build patient graphs for GNN/HQGNN models (using real data)
    patient_graphs = None
    if model_type in ["gnn", "hqgnn"]:
        print("\nBuilding patient graphs from real data...")
        patient_graphs, _ = build_patient_graphs_from_data(df)
        print(f"  Built {len(patient_graphs)} patient graphs")

    print(f"\nUsing biomarker threshold: {threshold}")
    criterion_desc = {
        "ever": "PD ever reaches below threshold in 1200h",
        "trough": "Ctrough_ss (last value)",
        "average": "Cavg_ss (mean value)",
        "max": "Cmax_ss (peak value)"
    }
    print(f"Evaluation criterion: {criterion} ({criterion_desc.get(criterion, criterion)})")

    # ==================== TASK 1 ====================
    print("\n" + "="*60)
    print("TASK 1: Once-daily dosing, target = 90%")
    print("="*60)

    d_daily_90, r_daily_90 = find_min_dose(
        model, model_type, df,
        tau_h=DAILY_TAU_H,
        n_cycles=N_CYCLES_DAILY,
        target_prob=0.90,
        dose_step=DAILY_STEP_MG,
        dose_min=DAILY_STEP_MG,
        dose_max=daily_max,
        scaler=scaler,
        threshold=threshold,
        debug=debug,
        criterion=criterion,
        patient_graphs=patient_graphs
    )
    results['daily_90'] = (d_daily_90, r_daily_90)

    # ==================== TASK 2 ====================
    print("\n" + "="*60)
    print("TASK 2: Once-weekly dosing, target = 90%")
    print("="*60)

    d_weekly_90, r_weekly_90 = find_min_dose(
        model, model_type, df,
        tau_h=WEEKLY_TAU_H,
        n_cycles=N_CYCLES_WEEKLY,
        target_prob=0.90,
        dose_step=WEEKLY_STEP_MG,
        dose_min=WEEKLY_STEP_MG,
        dose_max=weekly_max,
        scaler=scaler,
        threshold=threshold,
        debug=debug,
        criterion=criterion,
        patient_graphs=patient_graphs
    )
    results['weekly_90'] = (d_weekly_90, r_weekly_90)

    # ==================== TASK 3 ====================
    print("\n" + "="*60)
    print("TASK 3: Once-daily dosing, target = 75%")
    print("="*60)

    d_daily_75, r_daily_75 = find_min_dose(
        model, model_type, df,
        tau_h=DAILY_TAU_H,
        n_cycles=N_CYCLES_DAILY,
        target_prob=0.75,
        dose_step=DAILY_STEP_MG,
        dose_min=DAILY_STEP_MG,
        dose_max=daily_max,
        scaler=scaler,
        threshold=threshold,
        debug=debug,
        criterion=criterion,
        patient_graphs=patient_graphs
    )
    results['daily_75'] = (d_daily_75, r_daily_75)

    # ==================== TASK 4 ====================
    print("\n" + "="*60)
    print("TASK 4: Once-weekly dosing, target = 75%")
    print("="*60)

    d_weekly_75, r_weekly_75 = find_min_dose(
        model, model_type, df,
        tau_h=WEEKLY_TAU_H,
        n_cycles=N_CYCLES_WEEKLY,
        target_prob=0.75,
        dose_step=WEEKLY_STEP_MG,
        dose_min=WEEKLY_STEP_MG,
        dose_max=weekly_max,
        scaler=scaler,
        threshold=threshold,
        debug=debug,
        criterion=criterion,
        patient_graphs=patient_graphs
    )
    results['weekly_75'] = (d_weekly_75, r_weekly_75)

    # ==================== TASK 5 ====================
    print("\n" + "="*60)
    print("TASK 5: No COMED (COMED=0), daily dosing, target = 90%")
    print("="*60)

    df_nocomed = df.copy()
    df_nocomed["COMED"] = 0.0

    # Rebuild patient graphs with COMED=0 for GNN models
    patient_graphs_nocomed = None
    if model_type in ["gnn", "hqgnn"]:
        patient_graphs_nocomed, _ = build_patient_graphs_from_data(df_nocomed)

    d_nocomed_90, r_nocomed_90 = find_min_dose(
        model, model_type, df_nocomed,
        tau_h=DAILY_TAU_H,
        n_cycles=N_CYCLES_DAILY,
        target_prob=0.90,
        dose_step=DAILY_STEP_MG,
        dose_min=DAILY_STEP_MG,
        dose_max=daily_max,
        scaler=scaler,
        threshold=threshold,
        debug=debug,
        criterion=criterion,
        patient_graphs=patient_graphs_nocomed
    )
    results['nocomed_daily_90'] = (d_nocomed_90, r_nocomed_90)

    # ==================== SUMMARY ====================
    print("\n" + "="*60)
    print(f"SUMMARY (threshold={threshold})")
    print("="*60)
    print(f"Task 1 - Daily 90%     : {results['daily_90'][0]} mg (achieved: {results['daily_90'][1]*100 if results['daily_90'][1] else 'N/A'}%)")
    print(f"Task 2 - Weekly 90%    : {results['weekly_90'][0]} mg (achieved: {results['weekly_90'][1]*100 if results['weekly_90'][1] else 'N/A'}%)")
    print(f"Task 3 - Daily 75%     : {results['daily_75'][0]} mg (achieved: {results['daily_75'][1]*100 if results['daily_75'][1] else 'N/A'}%)")
    print(f"Task 4 - Weekly 75%    : {results['weekly_75'][0]} mg (achieved: {results['weekly_75'][1]*100 if results['weekly_75'][1] else 'N/A'}%)")
    print(f"Task 5 - No COMED 90%  : {results['nocomed_daily_90'][0]} mg (achieved: {results['nocomed_daily_90'][1]*100 if results['nocomed_daily_90'][1] else 'N/A'}%)")

    return results


# ============================================================
# MAIN
# ============================================================
def main():
    args = get_args()

    print(f"\n{'='*60}")
    print(f"DOSE PREDICTION WITH {args.model.upper()} MODEL")
    print(f"{'='*60}")

    # Load data
    df = pd.read_csv(DATA_CSV)
    df.columns = [c.strip().upper() for c in df.columns]
    print(f"Loaded {len(df['ID'].unique())} subjects from {DATA_CSV}")

    # Prepare scaler (matching training preprocessing)
    print("\nPreparing feature scaler...")
    scaler, _ = prepare_scaler(df, model_type=args.model)

    # Load model
    model = load_model(args)

    # Set default threshold based on model type
    # GNN/HQGNN predicts higher absolute PD values than MLP/HQCNN/LSTM/HQLSTM
    if args.threshold is None:
        if args.model in ["gnn", "hqgnn"]:
            args.threshold = BIOMARKER_THRESHOLD_GNN
            print(f"Note: Using threshold={args.threshold} for GNN (higher PD scale)")
        else:
            args.threshold = BIOMARKER_THRESHOLD_MLP
            print(f"Note: Using threshold={args.threshold} for {args.model.upper()}")

    # Default criterion is "ever" - check if PD ever reaches threshold in 1200h
    print(f"Criterion: '{args.criterion}' - {'PD ever reaches below threshold' if args.criterion == 'ever' else args.criterion}")

    # Show data statistics if debug
    if args.debug:
        pd_df = df[(df['EVID'] == 0) & (df['DVID'] == 2)]
        print(f"\n[DEBUG] PD value statistics:")
        print(f"  min={pd_df['DV'].min():.2f}, max={pd_df['DV'].max():.2f}, "
              f"mean={pd_df['DV'].mean():.2f}, std={pd_df['DV'].std():.2f}")

    # Run tasks
    run_all_tasks(model, args.model, df, scaler,
                  threshold=args.threshold, debug=args.debug,
                  criterion=args.criterion)

    print("\nDone!")


if __name__ == "__main__":
    main()
