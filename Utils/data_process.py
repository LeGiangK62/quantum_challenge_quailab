import torch 
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from Utils.log import logger


# ============================================================
# Dataset Classes
# ============================================================
class PKPDDataset(Dataset):
    """
    Dataset for PK/PD tabular data (MLP).

    Each sample pairs a PD observation with the most recent PK observation
    from the SAME patient (time <= pd_time), ensuring dual_stage/joint modes
    receive the correct patient's PK prediction when encoding PD.

    Placebo patients (no PK) are included with zero-filled PK features
    so their PD observations still contribute to PD training.
    """

    def __init__(self, pk_data: dict, pd_data: dict):
        pk_X = pk_data['X']
        pk_y = pk_data['y']
        pk_ids = pk_data['ids']
        pk_times = pk_data['times']

        pd_X = pd_data['X']
        pd_y = pd_data['y']
        pd_ids = pd_data['ids']
        pd_times = pd_data['times']

        n_features = pk_X.shape[1]

        # Group PK observations by patient
        pk_by_patient = {}
        for i, pid in enumerate(pk_ids):
            pk_by_patient.setdefault(pid, []).append(i)

        # Build aligned pairs: for each PD obs find closest past PK from same patient
        paired_pk_X, paired_pk_y = [], []
        paired_pd_X, paired_pd_y = [], []
        n_zero_pk = 0

        for pd_i in range(len(pd_ids)):
            pid = pd_ids[pd_i]
            t_pd = pd_times[pd_i]

            if pid in pk_by_patient:
                idxs = np.array(pk_by_patient[pid])
                times = pk_times[idxs]

                # Prefer most recent PK at or before pd_time
                past = idxs[times <= t_pd]
                if len(past) > 0:
                    pk_i = past[np.argmax(pk_times[past])]
                else:
                    # No past PK — use closest overall (early timepoints)
                    pk_i = idxs[np.argmin(np.abs(times - t_pd))]

                paired_pk_X.append(pk_X[pk_i])
                paired_pk_y.append(pk_y[pk_i])
            else:
                # Placebo: no PK — use zeros so PD still trains
                paired_pk_X.append(np.zeros(n_features, dtype=np.float32))
                paired_pk_y.append(np.float32(0.0))
                n_zero_pk += 1

            paired_pd_X.append(pd_X[pd_i])
            paired_pd_y.append(pd_y[pd_i])

        if n_zero_pk > 0:
            print(f"  PKPDDataset: {n_zero_pk} placebo PD obs paired with zero PK")
        print(f"  PKPDDataset: {len(paired_pd_X)} aligned (patient-matched) PK-PD pairs")

        self.pk_X = torch.FloatTensor(np.array(paired_pk_X))
        self.pk_y = torch.FloatTensor(np.array(paired_pk_y)).unsqueeze(1)
        self.pd_X = torch.FloatTensor(np.array(paired_pd_X))
        self.pd_y = torch.FloatTensor(np.array(paired_pd_y)).unsqueeze(1)

    def __len__(self):
        return len(self.pd_X)

    def __getitem__(self, idx):
        return {
            'pk_x': self.pk_X[idx],
            'pk_y': self.pk_y[idx],
            'pd_x': self.pd_X[idx],
            'pd_y': self.pd_y[idx],
        }


def collate_pkpd(batch):
    """Collate function for PK/PD batch."""
    pk_x = torch.stack([item['pk_x'] for item in batch])
    pk_y = torch.stack([item['pk_y'] for item in batch])
    pd_x = torch.stack([item['pd_x'] for item in batch])
    pd_y = torch.stack([item['pd_y'] for item in batch])

    return {
        'pk_x': pk_x,
        'pk_y': pk_y,
        'pd_x': pd_x,
        'pd_y': pd_y,
    }


# ============================================================
# GNN Data Preparation
# ============================================================
def prepare_gnn_data(args):
    """
    Prepare graph data for GNN training.

    Uses the same feature engineering pipeline as MLP (engineer_dose_features),
    stratified dose splitting, and proper scaler fitting (train only).

    Returns:
        dict with train_data, val_data, test_data, feature_dim
    """
    try:
        from torch_geometric.data import Data
    except ImportError:
        raise ImportError("torch_geometric required for GNN")

    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from Utils.data_loader import (
        load_data, engineer_dose_features, build_feature_list,
        stratified_dose_split, add_perkg_features,
        add_pk_patient_features, add_pk_cumulative_features,
    )

    logger.info("Preparing GNN graph data...")
    csv_data_path = 'Data/' + args.csv_path + '.csv'

    # --- Shared pipeline: load + engineer features (same as MLP) ---
    df_all, df_obs, df_dose = load_data(csv_data_path)

    df = engineer_dose_features(
        df_obs, df_dose,
        time_windows=args.time_windows,
        half_lives=args.half_lives,
        add_decay=args.add_decay,
    )

    if getattr(args, 'use_perkg', False):
        df = add_perkg_features(df, args.time_windows)

    if getattr(args, 'add_pk_summary', False):
        df = add_pk_patient_features(df)

    if getattr(args, 'add_pk_cumulative', False):
        df = add_pk_cumulative_features(df)

    # Exclude placebo patients
    if getattr(args, 'no_placebo', False):
        all_ids = df['ID'].unique()
        dosed_ids = df_dose['ID'].unique()
        placebo_ids = np.setdiff1d(all_ids, dosed_ids)
        n_before = len(all_ids)
        df = df[~df['ID'].isin(placebo_ids)]
        n_after = df['ID'].nunique()
        print(f"\nExcluded {n_before - n_after} placebo patients: {sorted(placebo_ids)}")

    features = build_feature_list(
        time_windows=args.time_windows,
        half_lives=args.half_lives,
        add_decay=args.add_decay,
        use_perkg=getattr(args, 'use_perkg', False),
        add_pk_summary=getattr(args, 'add_pk_summary', False),
        add_pk_cumulative=getattr(args, 'add_pk_cumulative', False),
    )
    logger.info(f"Total features (shared with MLP): {len(features)}")

    # --- Stratified dose split ---
    if getattr(args, 'stratified_split', True):
        split_ids = stratified_dose_split(
            df, df_dose,
            test_size=args.test_size,
            val_size=args.val_size,
            random_state=args.random_seed,
            combine=getattr(args, 'combine', False),
        )
    else:
        all_ids = df['ID'].unique()
        rng = np.random.RandomState(args.random_seed)
        shuffled = rng.permutation(all_ids)
        n_test = int(len(all_ids) * args.test_size)
        n_val = int(len(all_ids) * args.val_size)
        combine = getattr(args, 'combine', False)
        split_ids = {
            'test': shuffled[:n_test],
            'val': shuffled[n_test:n_test + n_val],
            'train': shuffled if combine else shuffled[n_test + n_val:],
        }

    # --- Fit scaler on train data only (no leakage) ---
    train_mask = df['ID'].isin(split_ids['train'])
    if getattr(args, 'normalize_data', False):
        scaler = MinMaxScaler(feature_range=(0, 1))
        logger.info("Using MinMaxScaler [0, 1] for input features")
    else:
        scaler = StandardScaler()
    scaler.fit(df.loc[train_mask, features].values)

    # --- Dose node setup ---
    use_dose_nodes = getattr(args, 'gnn_dose_nodes', False)
    df_dose_events = df_all[df_all['EVID'] == 1].copy()
    if use_dose_nodes:
        # Dose nodes need the same features; fill missing with 0
        df_dose_events['DOSE'] = df_dose_events['AMT']
        for f in features:
            if f not in df_dose_events.columns:
                df_dose_events[f] = 0.0
        logger.info(f"Dosing events (will be added as nodes): {len(df_dose_events)}")

    cross_decay = getattr(args, 'gnn_edge_decay', 12.0)

    # --- Build per-patient graphs ---
    graphs = {}  # patient_id -> graph dict

    for patient_id in df['ID'].unique():
        patient_df = df[df['ID'] == patient_id].sort_values('TIME').reset_index(drop=True)

        pk_obs = patient_df[patient_df['DVID'] == 1]
        pd_obs = patient_df[patient_df['DVID'] == 2]

        if len(pd_obs) == 0:
            continue

        node_features = []
        pk_targets, pd_targets = [], []
        times = []
        pk_indices, pd_indices, dose_indices = [], [], []
        node_idx = 0

        # Add DOSE nodes (if enabled)
        if use_dose_nodes:
            patient_doses = df_dose_events[df_dose_events['ID'] == patient_id].sort_values('TIME')
            for _, row in patient_doses.iterrows():
                feat = [row[f] if f in row.index else 0.0 for f in features]
                node_features.append(feat)
                pk_targets.append(0)
                pd_targets.append(0)
                times.append(row['TIME'])
                dose_indices.append(node_idx)
                node_idx += 1

        # Add PK nodes
        for _, row in pk_obs.iterrows():
            node_features.append([row[f] for f in features])
            pk_targets.append(row['DV'])
            pd_targets.append(0)
            times.append(row['TIME'])
            pk_indices.append(node_idx)
            node_idx += 1

        # Add PD nodes
        for _, row in pd_obs.iterrows():
            node_features.append([row[f] for f in features])
            pk_targets.append(0)
            pd_targets.append(row['DV'])
            times.append(row['TIME'])
            pd_indices.append(node_idx)
            node_idx += 1

        # Create edges
        edges = []
        edge_weights = []
        times_arr = np.array(times)
        n_nodes = len(times_arr)
        k_hop = getattr(args, 'gnn_k_hop', 3)  # skip-edge reach

        # Self-loops (critical for GNN stability)
        for i in range(n_nodes):
            edges.append([i, i])
            edge_weights.append(1.0)

        def add_temporal_edges(indices, decay_const=24.0):
            """Add sequential + k-hop skip edges within a node group."""
            for gap in range(1, k_hop + 1):
                for i in range(len(indices) - gap):
                    src, dst = indices[i], indices[i + gap]
                    time_diff = abs(times_arr[dst] - times_arr[src])
                    weight = np.exp(-time_diff / decay_const)
                    edges.extend([[src, dst], [dst, src]])
                    edge_weights.extend([weight, weight])

        # Temporal edges (sequential + skip) within each node type
        add_temporal_edges(dose_indices)
        add_temporal_edges(pk_indices)
        add_temporal_edges(pd_indices)

        # DOSE → PK edges
        for pk_idx in pk_indices:
            pk_time = times_arr[pk_idx]
            for d_idx in dose_indices:
                if times_arr[d_idx] <= pk_time:
                    time_diff = pk_time - times_arr[d_idx]
                    weight = np.exp(-time_diff / cross_decay)
                    edges.extend([[d_idx, pk_idx], [pk_idx, d_idx]])
                    edge_weights.extend([weight, weight])

        # DOSE → PD edges
        for pd_idx in pd_indices:
            pd_time = times_arr[pd_idx]
            for d_idx in dose_indices:
                if times_arr[d_idx] <= pd_time:
                    time_diff = pd_time - times_arr[d_idx]
                    weight = np.exp(-time_diff / cross_decay)
                    edges.extend([[d_idx, pd_idx], [pd_idx, d_idx]])
                    edge_weights.extend([weight, weight])

        # PK → PD edges
        for pd_idx in pd_indices:
            pd_time = times_arr[pd_idx]
            for pk_idx in pk_indices:
                if times_arr[pk_idx] <= pd_time:
                    time_diff = pd_time - times_arr[pk_idx]
                    weight = np.exp(-time_diff / cross_decay)
                    edges.extend([[pk_idx, pd_idx], [pd_idx, pk_idx]])
                    edge_weights.extend([weight, weight])

        # Scale node features
        node_features = np.array(node_features, dtype=np.float32)
        node_features = scaler.transform(node_features)

        graphs[patient_id] = {
            'patient_id': patient_id,
            'node_features': node_features,
            'edge_index': np.array(edges, dtype=np.int64).T if edges else np.array([[], []], dtype=np.int64),
            'edge_weights': np.array(edge_weights, dtype=np.float32),
            'pk_targets': np.array(pk_targets, dtype=np.float32),
            'pd_targets': np.array(pd_targets, dtype=np.float32),
            'pk_indices': pk_indices,
            'pd_indices': pd_indices,
            'dose_indices': dose_indices,
            'times': np.array(times, dtype=np.float32),
        }

    logger.info(f"Created {len(graphs)} patient graphs")

    # --- Convert to PyG Data and split ---
    def to_pyg_data(graph):
        x = torch.FloatTensor(graph['node_features'])
        edge_index = torch.LongTensor(graph['edge_index'])
        edge_weight = torch.FloatTensor(graph['edge_weights'])

        pk_mask = torch.zeros(len(graph['node_features']), dtype=torch.bool)
        pk_mask[graph['pk_indices']] = True
        pd_mask = torch.zeros(len(graph['node_features']), dtype=torch.bool)
        pd_mask[graph['pd_indices']] = True

        return Data(
            x=x,
            edge_index=edge_index,
            edge_weight=edge_weight,
            pk_targets=torch.FloatTensor(graph['pk_targets']),
            pd_targets=torch.FloatTensor(graph['pd_targets']),
            pk_mask=pk_mask,
            pd_mask=pd_mask,
            patient_id=graph['patient_id'],
            times=torch.FloatTensor(graph['times']),
        )

    train_data = [to_pyg_data(graphs[pid]) for pid in split_ids['train'] if pid in graphs]
    val_data = [to_pyg_data(graphs[pid]) for pid in split_ids['val'] if pid in graphs]
    test_data = [to_pyg_data(graphs[pid]) for pid in split_ids['test'] if pid in graphs]

    logger.info(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    return {
        'train_data': train_data,
        'val_data': val_data,
        'test_data': test_data,
        'feature_dim': len(features),
    }

