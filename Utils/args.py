"""
Command-line arguments for unified PK/PD training.
"""

import argparse


def get_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Unified PK/PD Prediction Training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # ==================== Model Architecture ====================
    parser.add_argument('--model', type=str, default='mlp',
                       choices=['mlp', 'gnn', 'lstm', 'hqcnn', 'hqgnn', 'hqlstm'],
                       help='Model architecture')
    parser.add_argument('--mode', type=str, default='dual_stage',
                       choices=['separate', 'joint', 'dual_stage', 'shared'],
                       help='Training mode for hierarchical models')

    # ==================== MLP Hyperparameters ====================
    parser.add_argument('--hidden_dim', type=int, default=256,
                       help='Hidden dimension for MLP')
    parser.add_argument('--n_blocks', type=int, default=4,
                       help='Number of ResBlocks for MLP')
    parser.add_argument('--head_hidden', type=int, default=128,
                       help='Hidden dimension for prediction heads')

    # ==================== GNN Hyperparameters ====================
    parser.add_argument('--gnn_hidden_dim', type=int, default=64,
                       help='Hidden dimension for GNN')
    parser.add_argument('--num_layers_pk', type=int, default=3,
                       help='Number of GNN layers for PK encoder')
    parser.add_argument('--num_layers_pd', type=int, default=3,
                       help='Number of GNN layers for PD decoder')
    parser.add_argument('--use_attention', action='store_true', default=False,
                       help='Use GAT instead of GCN')
    parser.add_argument('--use_gating', action='store_true', default=True,
                       help='Use gating mechanism in PD decoder')
    parser.add_argument('--gnn_dose_nodes', action='store_true', default=False,
                       help='Add dosing events as nodes in GNN graph')
    parser.add_argument('--gnn_edge_decay', type=float, default=12.0,
                       help='Decay constant (hours) for cross-type edge weights')

    # ==================== LSTM Hyperparameters ====================
    parser.add_argument('--lstm_hidden_dim', type=int, default=128,
                       help='Hidden dimension for LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=2,
                       help='Number of LSTM layers')
    parser.add_argument('--lstm_bidirectional', action='store_true', default=True,
                       help='Use bidirectional LSTM')

    # ==================== Quantum Hyperparameters ====================
    parser.add_argument('--hqcnn_num_layers', type=int, default=1,
                       help='Number of quantum layers for HQCNN')
    parser.add_argument('--using_hqcnn', action='store_true', default=False,
                       help='Use HQCNN circuit instead of QNN_Amplitude in HQLSTM')
    parser.add_argument('--n_qubits', type=int, default=4,
                       help='Number of qubits for QNN_Amplitude')
    parser.add_argument('--n_qlayers', type=int, default=2,
                       help='Number of StronglyEntanglingLayers for QNN_Amplitude')

    # ==================== Data ====================
    parser.add_argument('--csv_path', type=str, default='QIC2025-EstDat',
                       help='Path to data CSV file')
    parser.add_argument('--test_size', type=float, default=0.1,
                       help='Test set fraction')
    parser.add_argument('--val_size', type=float, default=0.1,
                       help='Validation set fraction')
    parser.add_argument('--random_seed', type=int, default=1712,
                       help='Random seed for reproducibility')
    parser.add_argument('--stratified_split', action='store_true', default=True,
                       help='Use dose-stratified splitting (CRITICAL for performance)')
    parser.add_argument('--use_perkg', action='store_true', default=False,
                       help='Add per-kg normalized features')
    parser.add_argument('--combine', action='store_true', default=False,
                       help='Use all data for training (no train/val/test split)')
    parser.add_argument('--normalize_data', action='store_true', default=False,
                       help='Normalize input features to [0,1] range (MinMaxScaler) instead of StandardScaler')

    # ==================== Feature Engineering ====================
    parser.add_argument('--time_windows', type=int, nargs='+',
                       default=[24, 48, 72, 96, 120, 144, 168],
                       help='Time windows for dose history (hours)')
    parser.add_argument('--half_lives', type=int, nargs='+',
                       default=[24, 48, 72],
                       help='Half-lives for decay features (hours)')
    parser.add_argument('--add_decay', action='store_true', default=True,
                       help='Add exponential decay features')
    parser.add_argument('--add_pk_summary', action='store_true', default=False,
                       help='Add patient-level PK summary features (PK_MAX, PK_MEAN, PK_AUC, etc.)')
    parser.add_argument('--add_pk_cumulative', action='store_true', default=False,
                       help='Add cumulative PK features up to each timepoint (causal, no leakage)')

    # ==================== Training ====================
    parser.add_argument('--epochs', type=int, default=300,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate')
    parser.add_argument('--pk_loss_weight', type=float, default=0.3,
                       help='Weight for PK loss')
    parser.add_argument('--pd_loss_weight', type=float, default=1.0,
                       help='Weight for PD loss')

    # ==================== Loss Functions ====================
    parser.add_argument('--loss_type_pk', type=str, default='mse',
                       choices=['mse', 'mae', 'asymmetric', 'quantile', 'hybrid'],
                       help='Loss function for PK')
    parser.add_argument('--loss_type_pd', type=str, default='hybrid',
                       choices=['mse', 'mae', 'asymmetric', 'quantile', 'hybrid'],
                       help='Loss function for PD')
    parser.add_argument('--quantile_q', type=float, default=0.3,
                       help='Quantile parameter for quantile/hybrid loss')
    parser.add_argument('--hybrid_lambda', type=float, default=0.5,
                       help='Weight for MSE in hybrid loss')

    # ==================== Target Transforms ====================
    parser.add_argument('--log_pk', action='store_true', default=False,
                       help='Apply log1p transform to PK targets')
    parser.add_argument('--sqrt_pd', action='store_true', default=False,
                       help='Apply sqrt transform to PD targets')

    # ==================== Data Augmentation ====================
    parser.add_argument('--use_augmentation', action='store_true', default=False,
                       help='Enable data augmentation')
    parser.add_argument('--aug_method', type=str, default='jitter_mixup',
                       choices=['jitter', 'mixup', 'jitter_mixup'],
                       help='Augmentation method')
    parser.add_argument('--aug_prob', type=float, default=0.5,
                       help='Probability of applying augmentation')

    # ==================== Regularization ====================
    parser.add_argument('--no_early_stopping', action='store_true', default=False,
                       help='Disable early stopping (train for all epochs)')
    parser.add_argument('--early_stopping_patience', type=int, default=100,
                       help='Early stopping patience (epochs)')
    parser.add_argument('--early_stopping_min_delta', type=float, default=0.0001,
                       help='Minimum improvement for early stopping')

    # ==================== MC Dropout ====================
    parser.add_argument('--use_mc_dropout', action='store_true', default=False,
                       help='Use MC Dropout for uncertainty estimation')
    parser.add_argument('--mc_samples', type=int, default=10,
                       help='Number of MC Dropout samples')

    # ==================== Output ====================
    parser.add_argument('--save_dir', type=str, default='Results',
                       help='Directory to save results')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Experiment name (default: auto-generated)')

    # ==================== Device ====================
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda', 'mps'],
                       help='Device to use for training')

    # ==================== Logging ====================
    parser.add_argument('--log_interval', type=int, default=10,
                       help='Print logs every N epochs')
    parser.add_argument('--save_model', action='store_true', default=True,
                       help='Save trained model')
    parser.add_argument('--save_plots', action='store_true', default=True,
                       help='Save training plots')

    args = parser.parse_args()

    # Auto-generate experiment name if not provided
    if args.experiment_name is None:
        if args.model in ['mlp', 'hqcnn']:
            hdim = args.hidden_dim
        elif args.model in ['lstm', 'hqlstm']:
            hdim = args.lstm_hidden_dim
        else:
            hdim = args.gnn_hidden_dim
        args.experiment_name = f"{args.model}_{args.mode}_h{hdim}"
        if args.use_augmentation:
            args.experiment_name += f"_aug{args.aug_method}"
        if args.combine:
            args.experiment_name += "_combine"
        elif args.stratified_split:
            args.experiment_name += "_stratified"

    return args


def print_args(args):
    """Pretty print arguments."""
    print("\n" + "="*60)
    print("EXPERIMENT CONFIGURATION")
    print("="*60)

    # Group arguments by category
    categories = {
        'Model': ['model', 'mode', 'hidden_dim', 'n_blocks', 'gnn_hidden_dim',
                 'num_layers_pk', 'num_layers_pd', 'lstm_hidden_dim', 'lstm_num_layers',
                 'lstm_bidirectional', 'dropout'],
        'Data': ['csv_path', 'test_size', 'val_size', 'random_seed',
                'stratified_split', 'use_perkg', 'combine', 'normalize_data'],
        'Features': ['time_windows', 'half_lives', 'add_decay'],
        'Training': ['epochs', 'batch_size', 'learning_rate',
                    'pk_loss_weight', 'pd_loss_weight'],
        'Loss': ['loss_type_pk', 'loss_type_pd', 'quantile_q', 'hybrid_lambda'],
        'Augmentation': ['use_augmentation', 'aug_method', 'aug_prob'],
        'Regularization': ['early_stopping_patience', 'early_stopping_min_delta'],
        'Uncertainty': ['use_mc_dropout', 'mc_samples'],
        'Output': ['save_dir', 'experiment_name', 'device']
    }

    for category, keys in categories.items():
        print(f"\n{category}:")
        for key in keys:
            if hasattr(args, key):
                value = getattr(args, key)
                # Format lists nicely
                if isinstance(value, list) and len(value) > 5:
                    value = f"{value[:3]} ... {value[-2:]}"
                print(f"  {key:.<30} {value}")

    print("\n" + "="*60 + "\n")
