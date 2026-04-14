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

    Returns:
        dict with train_data, val_data, test_data, feature_dim
    """
    try:
        from torch_geometric.data import Data
    except ImportError:
        raise ImportError("torch_geometric required for GNN")

    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.model_selection import train_test_split

    logger.info("Preparing GNN graph data...")
    csv_data_path = 'Data/' + args.csv_path + '.csv'

    df = pd.read_csv(csv_data_path)
    df.columns = [c.strip().upper() for c in df.columns]

    # Filter observations
    df = df[df['EVID'] == 0].copy()
    if 'MDV' in df.columns:
        df = df[df['MDV'] == 0]

    logger.info(f"Total observations: {len(df)}")

    # Feature engineering
    base_features = ['TIME', 'BW', 'DOSE', 
                    #  'COMED'
                     ]
    if args.add_decay:
        for hl in args.half_lives:
            df[f'DECAY_HL{hl}h'] = np.exp(-np.log(2) / hl * df['TIME'])
            base_features.append(f'DECAY_HL{hl}h')

    df['TIME_LOG'] = np.log1p(df['TIME'])
    df['TIME_SQUARED'] = df['TIME'] ** 2
    base_features.extend(['TIME_LOG', 'TIME_SQUARED'])

    # Add patient-level PK summary features
    if getattr(args, 'add_pk_summary', False):
        pk_all = df[df['DVID'] == 1]
        for pid in df['ID'].unique():
            pk_p = pk_all[pk_all['ID'] == pid].sort_values('TIME')
            if len(pk_p) > 0:
                dv = pk_p['DV'].values
                times_p = pk_p['TIME'].values
                df.loc[df['ID'] == pid, 'PK_PATIENT_MAX'] = dv.max()
                df.loc[df['ID'] == pid, 'PK_PATIENT_MEAN'] = dv.mean()
                df.loc[df['ID'] == pid, 'PK_PATIENT_AUC'] = np.trapz(dv, times_p) if len(dv) > 1 else 0.0
                df.loc[df['ID'] == pid, 'PK_PATIENT_TMAX'] = times_p[np.argmax(dv)]
                df.loc[df['ID'] == pid, 'PK_PATIENT_LAST'] = dv[-1]
                df.loc[df['ID'] == pid, 'PK_PATIENT_CMAX_RATIO'] = dv.max() / dv.mean() if dv.mean() > 0 else 0.0
            else:
                for col in ['PK_PATIENT_MAX', 'PK_PATIENT_MEAN', 'PK_PATIENT_AUC',
                            'PK_PATIENT_TMAX', 'PK_PATIENT_LAST', 'PK_PATIENT_CMAX_RATIO']:
                    df.loc[df['ID'] == pid, col] = 0.0
        base_features.extend(['PK_PATIENT_MAX', 'PK_PATIENT_MEAN', 'PK_PATIENT_AUC',
                              'PK_PATIENT_TMAX', 'PK_PATIENT_LAST', 'PK_PATIENT_CMAX_RATIO'])
        logger.info("Added patient-level PK summary features for GNN")

    # Add cumulative PK features (causal, per-observation)
    if getattr(args, 'add_pk_cumulative', False):
        pk_all = df[df['DVID'] == 1]
        cum_cols = ['PK_CUM_MAX', 'PK_CUM_MEAN', 'PK_CUM_AUC', 'PK_CUM_LAST', 'PK_CUM_COUNT']
        for col in cum_cols:
            df[col] = 0.0

        pk_by_patient = {}
        for pid in df['ID'].unique():
            pk_p = pk_all[pk_all['ID'] == pid].sort_values('TIME')
            if len(pk_p) > 0:
                pk_by_patient[pid] = {'times': pk_p['TIME'].values, 'dv': pk_p['DV'].values}

        for idx, row in df.iterrows():
            pid, t = row['ID'], row['TIME']
            if pid not in pk_by_patient:
                continue
            pk_data = pk_by_patient[pid]
            mask = pk_data['times'] <= t
            if not mask.any():
                continue
            dv_up = pk_data['dv'][mask]
            times_up = pk_data['times'][mask]
            df.at[idx, 'PK_CUM_MAX'] = dv_up.max()
            df.at[idx, 'PK_CUM_MEAN'] = dv_up.mean()
            df.at[idx, 'PK_CUM_LAST'] = dv_up[-1]
            df.at[idx, 'PK_CUM_COUNT'] = len(dv_up)
            if len(dv_up) > 1:
                df.at[idx, 'PK_CUM_AUC'] = np.trapz(dv_up, times_up)

        base_features.extend(cum_cols)
        logger.info("Added cumulative PK features for GNN")

    # Create graphs per patient
    graphs = []
    all_features = []

    for patient_id in df['ID'].unique():
        patient_df = df[df['ID'] == patient_id].sort_values('TIME').reset_index(drop=True)

        pk_obs = patient_df[patient_df['DVID'] == 1]
        pd_obs = patient_df[patient_df['DVID'] == 2]

        # Only skip if no PD data (allow patients with only PD, no PK - e.g., placebo)
        if len(pd_obs) == 0:
            continue

        # Node features and targets
        node_features = []
        pk_targets, pd_targets = [], []
        node_types = []
        times = []

        pk_indices, pd_indices = [], []
        node_idx = 0

        # Add PK nodes
        for _, row in pk_obs.iterrows():
            features = [row[f] for f in base_features]
            node_features.append(features)
            pk_targets.append(row['DV'])
            pd_targets.append(0)
            node_types.append(0)
            times.append(row['TIME'])
            pk_indices.append(node_idx)
            node_idx += 1

        # Add PD nodes
        for _, row in pd_obs.iterrows():
            features = [row[f] for f in base_features]
            node_features.append(features)
            pk_targets.append(0)
            pd_targets.append(row['DV'])
            node_types.append(1)
            times.append(row['TIME'])
            pd_indices.append(node_idx)
            node_idx += 1

        # Create edges
        edges = []
        edge_weights = []
        times_arr = np.array(times)

        # Temporal edges within PK nodes
        for i in range(len(pk_indices) - 1):
            src, dst = pk_indices[i], pk_indices[i + 1]
            time_diff = abs(times_arr[dst] - times_arr[src])
            weight = np.exp(-time_diff / 24.0)
            edges.extend([[src, dst], [dst, src]])
            edge_weights.extend([weight, weight])

        # Temporal edges within PD nodes
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

        node_features = np.array(node_features, dtype=np.float32)
        all_features.append(node_features)

        graphs.append({
            'patient_id': patient_id,
            'node_features': node_features,
            'edge_index': np.array(edges, dtype=np.int64).T if edges else np.array([[], []], dtype=np.int64),
            'edge_weights': np.array(edge_weights, dtype=np.float32),
            'pk_targets': np.array(pk_targets, dtype=np.float32),
            'pd_targets': np.array(pd_targets, dtype=np.float32),
            'pk_indices': pk_indices,
            'pd_indices': pd_indices,
            'times': np.array(times, dtype=np.float32),
        })

    logger.info(f"Created {len(graphs)} patient graphs")

    # Scale features
    if getattr(args, 'normalize_data', False):
        scaler = MinMaxScaler(feature_range=(0, 1))
        logger.info("Using MinMaxScaler [0, 1] for input features")
    else:
        scaler = StandardScaler()
    all_features_concat = np.vstack(all_features)
    scaler.fit(all_features_concat)

    for g in graphs:
        g['node_features'] = scaler.transform(g['node_features'])

    # Split
    indices = list(range(len(graphs)))
    train_indices, test_indices = train_test_split(indices, test_size=args.test_size, random_state=args.random_seed)
    train_indices, val_indices = train_test_split(train_indices, test_size=args.val_size / (1 - args.test_size), random_state=args.random_seed)

    # Convert to PyG Data objects
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

    train_data = [to_pyg_data(graphs[i]) for i in train_indices]
    val_data = [to_pyg_data(graphs[i]) for i in val_indices]
    test_data = [to_pyg_data(graphs[i]) for i in test_indices]

    logger.info(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    return {
        'train_data': train_data,
        'val_data': val_data,
        'test_data': test_data,
        'feature_dim': all_features_concat.shape[1],
    }

