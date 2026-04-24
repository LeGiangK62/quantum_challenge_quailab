"""
Dose prediction for the QIC2025 challenge.

Tasks:
  1. Daily dose (multiples of 0.1 mg) so 90% of population has PD > 10 ng/mL
     throughout a 24h interval at steady-state.
  2. Weekly dose (multiples of 1 mg) with same effect over 168h at steady-state.

Usage:
  python dose_prediction.py --model_dir Results/26_04_22_12_56_58_hqlstm_dual_stage_h128_combine
"""

import argparse
import os
import re
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from Utils.data_loader import (
    load_data, engineer_dose_features, build_feature_list,
    add_pk_patient_features, add_pk_cumulative_features,
)

# ============================================================
# CONSTANTS
# ============================================================
DEVICE = "cpu"
BIOMARKER_THRESHOLD = 10.0  # ng/mL as per challenge

TIME_WINDOWS = [24, 48, 72, 96, 120, 144, 168]
HALF_LIVES = [24, 48, 72]

# Steady-state: determined analytically from PK parameters
# N_ss = ceil(-ln(1 - f) / (ke * tau)),  f = 0.97 (97% of SS)
# ke is estimated PER SUBJECT from model's own PK predictions
POP_KE = 0.00457   # h^-1 fallback (population median, used if per-subject estimation fails)
SS_FRACTION = 0.97
KE_MIN = 0.001     # h^-1 minimum ke (cap: t½ < 693h = 29 days)
KE_MAX = 0.05      # h^-1 maximum ke (cap: t½ > 14h)


# ============================================================
# ARGUMENT PARSING
# ============================================================
def get_args():
    parser = argparse.ArgumentParser(description="QIC2025 Dose Prediction")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Path to experiment results directory (e.g. Results/26_04_22_...)")
    parser.add_argument("--csv_path", type=str, default="Data/UpdatedEstData.csv",
                        help="Path to data CSV")
    parser.add_argument("--threshold", type=float, default=BIOMARKER_THRESHOLD,
                        help="Biomarker threshold (ng/mL)")
    parser.add_argument("--n_population", type=int, default=200,
                        help="Number of virtual subjects to simulate")
    parser.add_argument("--n_ss_daily", type=int, default=None,
                        help="Fixed N_ss for daily dosing (overrides per-subject ke estimation). "
                             "E.g. 21 to match training data range")
    parser.add_argument("--n_ss_weekly", type=int, default=None,
                        help="Fixed N_ss for weekly dosing (overrides per-subject ke estimation)")
    parser.add_argument("--debug", action="store_true",
                        help="Show per-dose debug info")
    return parser.parse_args()


# ============================================================
# INFER MODEL TYPE FROM DIRECTORY NAME
# ============================================================
def infer_model_config(model_dir):
    """
    Infer model type and hyperparameters from the experiment directory name.
    E.g. '26_04_22_12_56_58_hqlstm_dual_stage_h128_combine'
    """
    dirname = os.path.basename(model_dir)

    # Detect model type
    model_types = ['hqlstm', 'hqgnn', 'hqcnn', 'lstm', 'gnn', 'mlp', 'resqnn', 'qnn']
    model_type = None
    for mt in model_types:
        if f'_{mt}_' in dirname or dirname.startswith(mt):
            model_type = mt
            break
    if model_type is None:
        raise ValueError(f"Cannot infer model type from directory: {dirname}")

    # Detect hidden dim
    m = re.search(r'_h(\d+)', dirname)
    hidden_dim = int(m.group(1)) if m else 128

    # Detect mode
    modes = ['dual_stage', 'joint', 'separate', 'shared']
    mode = 'dual_stage'
    for md in modes:
        if md in dirname:
            mode = md
            break

    # Detect flags
    combine = 'combine' in dirname
    stratified = 'stratified' in dirname

    print(f"Inferred: model={model_type}, hidden_dim={hidden_dim}, mode={mode}")
    return {
        'model_type': model_type,
        'hidden_dim': hidden_dim,
        'mode': mode,
        'combine': combine,
        'stratified': stratified,
    }


# ============================================================
# MODEL LOADING
# ============================================================
def load_model(model_dir, config, n_features):
    """Load model from saved checkpoint."""
    model_path = os.path.join(model_dir, "model.pth")
    checkpoint = torch.load(model_path, map_location="cpu")

    mt = config['model_type']
    hdim = config['hidden_dim']
    mode = config['mode']

    if mt == 'mlp':
        from Models.mlp import HierarchicalPKPDMLP
        # Infer n_blocks from checkpoint (prefix depends on mode:
        # separate/joint/dual_stage -> pk_encoder, shared -> encoder)
        n_blocks = 0
        for prefix in ('pk_encoder', 'encoder', 'shared_encoder'):
            n_blocks = sum(1 for k in checkpoint
                           if k.startswith(f'{prefix}.blocks.') and k.endswith('.fc1.weight'))
            if n_blocks > 0:
                break
        # Infer head_hidden from pk_head.0.weight shape [head_hidden, hidden_dim]
        head_w = checkpoint.get('pk_head.0.weight')
        head_hidden = int(head_w.shape[0]) if head_w is not None else 128
        model = HierarchicalPKPDMLP(
            pk_input_dim=n_features, pd_input_dim=n_features,
            hidden_dim=hdim, mode=mode,
            n_blocks=max(n_blocks, 1), head_hidden=head_hidden,
        )
    elif mt == 'hqcnn':
        from Models.quantum import HierarchicalHQCNN
        # Infer num_layers from checkpoint
        n_ql = sum(1 for k in checkpoint if 'pk_model.qlayers' in k and 'weights_0' in k)
        model = HierarchicalHQCNN(
            pk_input_dim=n_features, pd_input_dim=n_features,
            num_layers=max(n_ql, 1), mode=mode,
        )
    elif mt == 'qnn':
        from Models.quantum import HierarchicalQNN
        # Infer n_qubits and n_qlayers from checkpoint shape
        q_w = checkpoint.get('pk_model.q_weights')
        if q_w is not None:
            n_qlayers, n_qubits, _ = q_w.shape
        else:
            n_qlayers, n_qubits = 2, 4
        model = HierarchicalQNN(
            pk_input_dim=n_features, pd_input_dim=n_features,
            n_qubits=n_qubits, n_qlayers=n_qlayers, mode=mode,
        )
    elif mt == 'resqnn':
        from Models.quantum import HierarchicalResQNN
        # Infer n_blocks from checkpoint
        n_blocks = sum(1 for k in checkpoint
                       if k.startswith('pk_encoder.blocks.') and k.endswith('.q_weights'))
        # Infer n_qubits / n_qlayers from q_weights shape of first block
        q_w = checkpoint.get('pk_encoder.blocks.0.q_weights')
        if q_w is not None:
            n_qlayers, n_qubits, _ = q_w.shape
        else:
            n_qlayers, n_qubits = 1, 4
        head_w = checkpoint.get('pk_head.0.weight')
        head_hidden = int(head_w.shape[0]) if head_w is not None else 128
        model = HierarchicalResQNN(
            mode=mode,
            pk_input_dim=n_features, pd_input_dim=n_features,
            hidden_dim=hdim, n_blocks=max(n_blocks, 1), head_hidden=head_hidden,
            n_qubits=n_qubits, n_qlayers=n_qlayers,
        )
    elif mt == 'gnn':
        from Models.gnn import HierarchicalPKPDGNN
        # Infer feature_dim from checkpoint
        conv_keys = [k for k in checkpoint if 'pk_encoder.convs.0' in k and 'weight' in k]
        feature_dim = checkpoint[conv_keys[0]].shape[-1] if conv_keys else n_features
        model = HierarchicalPKPDGNN(feature_dim=feature_dim, hidden_dim=hdim)
    elif mt == 'hqgnn':
        from Models.quantum import HQGNN
        conv_keys = [k for k in checkpoint if 'pk_encoder.convs.0' in k and 'weight' in k]
        feature_dim = checkpoint[conv_keys[0]].shape[-1] if conv_keys else n_features
        # Detect if using QNN_Amplitude or HQCNN
        using_hqcnn = 'pd_decoder.pd_predictor.clayer_1.weight' in checkpoint
        q_w = checkpoint.get('pd_decoder.pd_predictor.q_weights')
        if q_w is not None and not using_hqcnn:
            n_qlayers, n_qubits, _ = q_w.shape
        else:
            n_qlayers, n_qubits = 1, 4
        model = HQGNN(
            feature_dim=feature_dim, hidden_dim=hdim,
            n_qlayers=n_qlayers, n_qubits=n_qubits, using_hqcnn=using_hqcnn,
        )
    elif mt == 'lstm':
        from Models.lstm import HierarchicalPKPDLSTM
        model = HierarchicalPKPDLSTM(
            input_dim=n_features, hidden_dim=hdim, mode=mode,
        )
    elif mt == 'hqlstm':
        from Models.quantum import HQLSTM
        # Detect using_hqcnn vs QNN_Amplitude
        using_hqcnn = 'pk_encoder.predictor.clayer_1.weight' in checkpoint
        q_w = checkpoint.get('pk_encoder.predictor.q_weights')
        if q_w is not None and not using_hqcnn:
            n_qlayers, n_qubits, _ = q_w.shape
        else:
            n_qlayers, n_qubits = 1, 4
        model = HQLSTM(
            input_dim=n_features, hidden_dim=hdim, mode=mode,
            n_qlayers=n_qlayers, n_qubits=n_qubits, using_hqcnn=using_hqcnn,
        )
    else:
        raise ValueError(f"Unknown model type: {mt}")

    model.load_state_dict(checkpoint)
    model.to(DEVICE).eval()
    print(f"Loaded {mt.upper()} from {model_path}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")
    return model


# ============================================================
# FEATURE ENGINEERING FOR SIMULATION
# ============================================================
def compute_features_for_timepoint(
    time, bw, dose_mg, dose_history, features_list,
    add_pk_summary=False, add_pk_cumulative=False,
):
    """
    Compute feature vector for a single observation timepoint.
    Must match the feature order from build_feature_list().
    """
    past_doses = [(t, amt) for t, amt in dose_history if t <= time]

    if len(past_doses) == 0:
        tsld = 0.0
        last_dose_time = 0.0
        last_dose_amt = 0.0
        n_doses = 0
        cum_dose = 0.0
    else:
        last_t, last_amt = past_doses[-1]
        tsld = time - last_t
        last_dose_time = last_t
        last_dose_amt = last_amt
        n_doses = len(past_doses)
        cum_dose = sum(amt for _, amt in past_doses)

    time_squared = time ** 2
    time_log = np.log1p(time)

    # Window sums
    window_sums = []
    for w in TIME_WINDOWS:
        s = sum(amt for t, amt in past_doses if time - w <= t <= time)
        window_sums.append(s)

    # Decay features
    decay_feats = []
    for hl in HALF_LIVES:
        if tsld > 0:
            decay_feats.append(np.exp(-np.log(2) / hl * tsld))
        else:
            decay_feats.append(0.0)

    # Base features (must match build_feature_list order)
    feat = {
        'BW': bw,
        'DOSE': dose_mg,
        'TIME': time,
        'TSLD': tsld,
        'LAST_DOSE_TIME': last_dose_time,
        'LAST_DOSE_AMT': last_dose_amt,
        'N_DOSES_UP_TO_T': n_doses,
        'CUM_DOSE_UP_TO_T': cum_dose,
        'TIME_SQUARED': time_squared,
        'TIME_LOG': time_log,
    }
    for i, w in enumerate(TIME_WINDOWS):
        feat[f'DOSE_SUM_PREV{w}H'] = window_sums[i]
    for i, hl in enumerate(HALF_LIVES):
        feat[f'DECAY_HL{hl}h'] = decay_feats[i]

    # PK summary/cumulative features → zero for simulation (no real PK data)
    if add_pk_summary:
        for col in ['PK_PATIENT_MAX', 'PK_PATIENT_MEAN', 'PK_PATIENT_AUC',
                     'PK_PATIENT_TMAX', 'PK_PATIENT_LAST', 'PK_PATIENT_CMAX_RATIO']:
            feat[col] = 0.0
    if add_pk_cumulative:
        for col in ['PK_CUM_MAX', 'PK_CUM_MEAN', 'PK_CUM_AUC', 'PK_CUM_LAST', 'PK_CUM_COUNT']:
            feat[col] = 0.0

    return np.array([feat.get(f, 0.0) for f in features_list], dtype=np.float32)


def calc_n_ss(ke, tau, fraction=SS_FRACTION):
    """
    Calculate number of dosing cycles to reach steady-state.

    From 1-compartment PK:
        fraction_reached(N) = 1 - e^(-N * ke * tau)
        N = -ln(1 - fraction) / (ke * tau)
    """
    return int(np.ceil(-np.log(1 - fraction) / (ke * tau)))


def _predict_model(model, model_type, features, obs_times):
    """
    Run model forward pass, return (pk_pred, pd_pred) as numpy arrays.
    """
    if model_type in ['mlp', 'hqcnn', 'qnn', 'resqnn']:
        results = model(features, features)
        pk = results['pk'].cpu().numpy().flatten()
        pd_ = results['pd'].cpu().numpy().flatten()
        return pk, pd_

    elif model_type in ['lstm', 'hqlstm']:
        features_seq = features.unsqueeze(0)
        lengths = torch.tensor([features.shape[0]])
        results = model(
            x_pk=features_seq, x_pd=features_seq,
            lengths_pk=lengths, lengths_pd=lengths,
        )
        pk = results['pk'].cpu().numpy().flatten()
        pd_ = results['pd'].cpu().numpy().flatten()
        return pk, pd_

    elif model_type in ['gnn', 'hqgnn']:
        from torch_geometric.data import Data
        n = features.shape[0]
        edges = []
        weights = []
        for i in range(n):
            edges.append([i, i])
            weights.append(1.0)
        for i in range(n - 1):
            td = abs(obs_times[i + 1] - obs_times[i])
            w = np.exp(-td / 24.0)
            edges.extend([[i, i + 1], [i + 1, i]])
            weights.extend([w, w])

        data = Data(
            x=features,
            edge_index=torch.LongTensor(np.array(edges).T),
            edge_weight=torch.FloatTensor(weights),
            pk_mask=torch.ones(n, dtype=torch.bool),
            pd_mask=torch.ones(n, dtype=torch.bool),
            pk_targets=torch.zeros(n), pd_targets=torch.zeros(n),
            times=torch.FloatTensor(obs_times),
        ).to(DEVICE)

        pd_pred, pk_pred = model(data, return_pk=True)
        return pk_pred.cpu().numpy().flatten(), pd_pred.cpu().numpy().flatten()

    return np.array([]), np.array([])


def _build_features(bw, dose_mg, tau_h, cycle_idx, dose_history,
                    features_list, scaler, add_pk_summary, add_pk_cumulative, offsets):
    """Build features for one cycle and return (features_tensor, obs_times)."""
    t_start = cycle_idx * tau_h
    obs_times = np.array([t_start + o for o in offsets if o <= tau_h])

    features = []
    for t in obs_times:
        feat = compute_features_for_timepoint(
            t, bw, dose_mg, dose_history, features_list,
            add_pk_summary, add_pk_cumulative,
        )
        features.append(feat)

    features = np.stack(features)
    if scaler is not None:
        features = scaler.transform(features)
    return torch.FloatTensor(features).to(DEVICE), obs_times


# ============================================================
# STEP 1: Estimate ke from model's PK predictions
# ============================================================
@torch.no_grad()
def estimate_ke_from_model(
    model, model_type, bw, dose_mg,
    features_list, scaler, add_pk_summary, add_pk_cumulative,
):
    """
    Estimate elimination rate constant (ke) from model's own PK predictions.

    Approach:
        1. Give 2 doses at t=0 and t=24h (to have some drug onboard)
        2. Predict PK at several timepoints AFTER last dose (decay phase)
        3. Fit log-linear: ln(PK) = ln(C0) - ke * t  →  ke = -slope

    Returns:
        ke: estimated elimination rate constant (h^-1), clamped to [KE_MIN, KE_MAX]
    """
    # Give 2 doses, then observe decay from t=48h onward
    dose_history = [(0.0, dose_mg), (24.0, dose_mg)]

    # Observation times during decay phase (after last dose at 24h)
    # Offsets from last dose: 1, 4, 12, 24, 48, 72, 120, 168h
    decay_offsets = [25, 28, 36, 48, 72, 96, 144, 192]
    obs_times = np.array(decay_offsets, dtype=float)

    features = []
    for t in obs_times:
        feat = compute_features_for_timepoint(
            t, bw, dose_mg, dose_history, features_list,
            add_pk_summary, add_pk_cumulative,
        )
        features.append(feat)

    features = np.stack(features)
    if scaler is not None:
        features = scaler.transform(features)
    features = torch.FloatTensor(features).to(DEVICE)

    pk_pred, _ = _predict_model(model, model_type, features, obs_times)

    # Fit log-linear to positive PK values
    # Use time relative to last dose (24h)
    t_rel = obs_times - 24.0  # time since last dose
    mask = pk_pred > 0
    if mask.sum() < 3:
        return POP_KE  # fallback

    t_fit = t_rel[mask]
    c_fit = pk_pred[mask]
    log_c = np.log(c_fit)

    # Linear regression: log(C) = intercept - ke * t
    slope, _ = np.polyfit(t_fit, log_c, 1)
    ke = -slope

    # Clamp to reasonable range
    ke = np.clip(ke, KE_MIN, KE_MAX)
    return ke


# ============================================================
# STEP 2: Simulate PD at steady-state using per-subject ke
# ============================================================
@torch.no_grad()
def simulate_ss_pd(
    model, model_type, bw, dose_mg, tau_h,
    features_list, scaler, add_pk_summary, add_pk_cumulative,
    n_ss_fixed=None,
):
    """
    Predict PD at steady-state for a single subject.

    Approach:
        1. Estimate ke from model's PK predictions for this (bw, dose)
        2. Calculate N_ss = ceil(-ln(0.03) / (ke * tau))
           OR use n_ss_fixed if provided
        3. Build dose history with N_ss doses
        4. Predict PD at the last cycle = steady-state

    Returns:
        pd_pred: PD predictions at SS cycle (array over timepoints)
        n_ss: number of cycles
        ke: estimated ke for this subject
    """
    # Step 1: estimate ke
    ke = estimate_ke_from_model(
        model, model_type, bw, dose_mg,
        features_list, scaler, add_pk_summary, add_pk_cumulative,
    )

    # Step 2: calculate N_ss (or use fixed)
    if n_ss_fixed is not None:
        n_ss = n_ss_fixed
    else:
        n_ss = calc_n_ss(ke, tau_h)

    # Step 3: build dose history
    dose_history = [(i * tau_h, dose_mg) for i in range(n_ss)]

    # Step 4: predict at last cycle
    if tau_h <= 24:
        offsets = [0, 0.5, 1, 2, 4, 6, 8, 12, 16, 20, 24]
    else:
        offsets = [0, 0.5, 1, 2, 4, 8, 12, 24, 36, 48, 72, 96, 120, 144, 168]

    features, obs_times = _build_features(
        bw, dose_mg, tau_h, n_ss - 1, dose_history,
        features_list, scaler, add_pk_summary, add_pk_cumulative, offsets,
    )

    _, pd_pred = _predict_model(model, model_type, features, obs_times)

    return pd_pred, n_ss, ke


# ============================================================
# DOSE FINDING
# ============================================================
def evaluate_dose(model, model_type, dose_mg, tau_h,
                  population_bws, features_list, scaler,
                  add_pk_summary, add_pk_cumulative, threshold,
                  n_ss_fixed=None):
    """
    Evaluate what fraction of population has PD > threshold
    THROUGHOUT the dosing interval at steady-state.

    Per subject:
        1. Estimate ke from model's PK prediction for this (bw, dose)
        2. N_ss = ceil(-ln(0.03) / (ke * tau))  OR use n_ss_fixed
        3. Predict PD at cycle N_ss → check min(PD) >= threshold
    """
    n_above = 0
    n_total = len(population_bws)
    min_pds = []
    ke_list = []
    nss_list = []

    for bw in population_bws:
        ss_pd, n_ss, ke = simulate_ss_pd(
            model, model_type, bw, dose_mg, tau_h,
            features_list, scaler, add_pk_summary, add_pk_cumulative,
            n_ss_fixed=n_ss_fixed,
        )
        if len(ss_pd) == 0:
            continue
        min_pd = np.min(ss_pd)
        min_pds.append(min_pd)
        ke_list.append(ke)
        nss_list.append(n_ss)
        if min_pd >= threshold:
            n_above += 1

    fraction = n_above / n_total if n_total > 0 else 0.0
    return fraction, np.array(min_pds), np.array(ke_list), np.array(nss_list)


def find_dose(model, model_type, tau_h, dose_step, dose_max,
              population_bws, features_list, scaler,
              add_pk_summary, add_pk_cumulative, threshold,
              target_fraction=0.90, debug=False, n_ss_fixed=None):
    """
    Find minimum dose (in multiples of dose_step) achieving target fraction.
    """
    if n_ss_fixed is not None:
        print(f"  N_ss fixed: {n_ss_fixed} cycles (user override)")
    else:
        print(f"  N_ss: per-subject (estimated from model PK predictions)")

    dose = dose_step
    while dose <= dose_max:
        frac, min_pds, ke_arr, nss_arr = evaluate_dose(
            model, model_type, dose, tau_h,
            population_bws, features_list, scaler,
            add_pk_summary, add_pk_cumulative, threshold,
            n_ss_fixed=n_ss_fixed,
        )
        status = "<<< FOUND" if frac >= target_fraction else ""
        print(f"  Dose {dose:6.1f} mg -> {frac*100:5.1f}% above {threshold} ng/mL {status}")

        if debug and len(min_pds) > 0:
            t_half_arr = np.log(2) / ke_arr
            print(f"       min_PD: min={min_pds.min():.2f}, mean={min_pds.mean():.2f}, "
                  f"max={min_pds.max():.2f}")
            print(f"       ke: mean={ke_arr.mean():.5f}, t1/2: mean={t_half_arr.mean():.0f}h "
                  f"({t_half_arr.mean()/24:.1f}d)")
            print(f"       N_ss: mean={nss_arr.mean():.0f}, range=[{nss_arr.min()}, {nss_arr.max()}]")

        if frac >= target_fraction:
            return dose, frac

        dose = round(dose + dose_step, 4)

    return None, None


# ============================================================
# MAIN
# ============================================================
def main():
    args = get_args()

    print(f"\n{'='*60}")
    print("QIC2025 DOSE PREDICTION")
    print(f"{'='*60}")

    # Infer model config
    config = infer_model_config(args.model_dir)
    mt = config['model_type']

    # Load and prepare data for scaler fitting
    print(f"\nLoading data from {args.csv_path}...")
    df_all, df_obs, df_dose = load_data(args.csv_path)

    df_final = engineer_dose_features(df_obs, df_dose, TIME_WINDOWS, HALF_LIVES)

    # Detect if model was trained with pk_summary / pk_cumulative
    # by checking metrics.txt or just try both (check checkpoint keys)
    model_path = os.path.join(args.model_dir, "model.pth")
    checkpoint = torch.load(model_path, map_location="cpu")

    # Heuristic: count features from first layer weight
    if mt in ['mlp', 'hqcnn', 'qnn', 'resqnn']:
        # Find input dimension from first layer
        for k, v in checkpoint.items():
            if 'weight' in k and v.dim() == 2:
                n_features_model = v.shape[-1]
                break
    elif mt in ['lstm', 'hqlstm']:
        for k, v in checkpoint.items():
            if 'input_proj.weight' in k:
                n_features_model = v.shape[-1]
                break
    elif mt in ['gnn', 'hqgnn']:
        for k, v in checkpoint.items():
            if 'pk_encoder.convs.0' in k and 'weight' in k:
                n_features_model = v.shape[-1]
                break

    # Try feature combinations to match
    add_pk_summary = False
    add_pk_cumulative = False

    base_n = len(build_feature_list(TIME_WINDOWS, HALF_LIVES))
    if n_features_model == base_n:
        pass
    elif n_features_model == base_n + 5:
        add_pk_cumulative = True
    elif n_features_model == base_n + 6:
        add_pk_summary = True
    elif n_features_model == base_n + 11:
        add_pk_summary = True
        add_pk_cumulative = True
    else:
        print(f"WARNING: Cannot auto-detect features. Model expects {n_features_model}, base={base_n}")
        # Try with both
        add_pk_summary = True
        add_pk_cumulative = True

    if add_pk_summary:
        df_final = add_pk_patient_features(df_final)
    if add_pk_cumulative:
        df_final = add_pk_cumulative_features(df_final)

    features_list = build_feature_list(
        TIME_WINDOWS, HALF_LIVES, add_decay=True,
        add_pk_summary=add_pk_summary, add_pk_cumulative=add_pk_cumulative,
    )
    print(f"Features: {len(features_list)} (pk_summary={add_pk_summary}, pk_cumulative={add_pk_cumulative})")

    # Fit scaler on all observed data
    scaler = StandardScaler()
    scaler.fit(df_final[features_list].values)
    print(f"Scaler fitted on {len(df_final)} observations")

    # Load model
    model = load_model(args.model_dir, config, len(features_list))

    # Population BWs
    rng = np.random.RandomState(42)
    population_bws = rng.uniform(50, 130, size=args.n_population)
    print(f"\nPopulation: {args.n_population} virtual subjects, BW ~ U(50, 130) kg")
    print(f"Threshold: {args.threshold} ng/mL")

    # ==================== TASK 1: Daily dose ====================
    print(f"\n{'='*60}")
    print("TASK 1: Daily dose (multiples of 0.1 mg)")
    print(f"  90% of population must have PD > {args.threshold} throughout 24h at steady-state")
    print(f"{'='*60}")

    daily_dose, daily_frac = find_dose(
        model, mt, tau_h=24,
        dose_step=0.1, dose_max=20.0,
        population_bws=population_bws, features_list=features_list,
        scaler=scaler, add_pk_summary=add_pk_summary,
        add_pk_cumulative=add_pk_cumulative,
        threshold=args.threshold, target_fraction=0.90, debug=args.debug,
        n_ss_fixed=args.n_ss_daily,
    )

    # ==================== TASK 2: Weekly dose ====================
    print(f"\n{'='*60}")
    print("TASK 2: Weekly dose (multiples of 1 mg)")
    print(f"  90% of population must have PD > {args.threshold} throughout 168h at steady-state")
    print(f"{'='*60}")

    weekly_dose, weekly_frac = find_dose(
        model, mt, tau_h=168,
        dose_step=1.0, dose_max=100.0,
        population_bws=population_bws, features_list=features_list,
        scaler=scaler, add_pk_summary=add_pk_summary,
        add_pk_cumulative=add_pk_cumulative,
        threshold=args.threshold, target_fraction=0.90, debug=args.debug,
        n_ss_fixed=args.n_ss_weekly,
    )

    # ==================== SUMMARY ====================
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Model: {args.model_dir}")
    print(f"Threshold: {args.threshold} ng/mL")
    print(f"Population: {args.n_population} subjects, BW ~ U(50, 130) kg")
    print()
    if daily_dose is not None:
        print(f"  Task 1 (Daily,  90%): {daily_dose:.1f} mg  (achieved {daily_frac*100:.1f}%)")
    else:
        print(f"  Task 1 (Daily,  90%): NOT FOUND (max dose tested)")
    if weekly_dose is not None:
        print(f"  Task 2 (Weekly, 90%): {weekly_dose:.0f} mg  (achieved {weekly_frac*100:.1f}%)")
    else:
        print(f"  Task 2 (Weekly, 90%): NOT FOUND (max dose tested)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
