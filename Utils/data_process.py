import torch 
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from Utils.log import logger


# ============================================================
# Dataset Classes
# ============================================================
class PKPDDataset(Dataset):
    """Dataset for PK/PD tabular data (MLP)."""

    def __init__(self, pk_data: dict, pd_data: dict):
        """
        Args:
            pk_data: dict with 'X', 'y', 'ids', 'times'
            pd_data: dict with 'X', 'y', 'ids', 'times'
        """
        self.pk_X = torch.FloatTensor(pk_data['X'])
        self.pk_y = torch.FloatTensor(pk_data['y']).unsqueeze(1)
        self.pk_ids = pk_data['ids']
        self.pk_times = pk_data['times']

        self.pd_X = torch.FloatTensor(pd_data['X'])
        self.pd_y = torch.FloatTensor(pd_data['y']).unsqueeze(1)
        self.pd_ids = pd_data['ids']
        self.pd_times = pd_data['times']

    def __len__(self):
        return max(len(self.pk_X), len(self.pd_X))

    def __getitem__(self, idx):
        pk_idx = idx % len(self.pk_X)
        pd_idx = idx % len(self.pd_X)

        return {
            'pk_x': self.pk_X[pk_idx],
            'pk_y': self.pk_y[pk_idx],
            'pk_id': self.pk_ids[pk_idx],
            'pk_time': self.pk_times[pk_idx],
            'pd_x': self.pd_X[pd_idx],
            'pd_y': self.pd_y[pd_idx],
            'pd_id': self.pd_ids[pd_idx],
            'pd_time': self.pd_times[pd_idx],
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

    from sklearn.preprocessing import StandardScaler
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

