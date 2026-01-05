import argparse
import os


def get_args():
    """
    Parse command-line arguments for PK/PD model training.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Train PK/PD prediction models',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model selection
    parser.add_argument('--model', type=str, required=True,
                       choices=['linear', 'ridge', 'lasso', 'svm', 'mlp', 'cnn', 'lstm', 'gcn', 'gat'],
                       help='Model type to train')

    # Data parameters
    parser.add_argument('--data_path', type=str,
                       default='Data/QIC2025-EstDat.csv',
                       help='Path to CSV data file')
    parser.add_argument('--target_col', type=str, default='DV',
                       help='Target column name')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Test set proportion (0.0-1.0)')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--scale_features', action='store_true', default=True,
                       help='Scale features using StandardScaler')
    parser.add_argument('--scale_target', action='store_true', default=False,
                       help='Scale target using StandardScaler')

    # SVM parameters
    parser.add_argument('--svm_kernel', type=str, default='rbf',
                       choices=['linear', 'rbf', 'poly', 'sigmoid'],
                       help='SVM kernel type')
    parser.add_argument('--svm_C', type=float, default=1.0,
                       help='SVM regularization parameter')
    parser.add_argument('--svm_epsilon', type=float, default=0.1,
                       help='SVM epsilon parameter')
    parser.add_argument('--svm_grid_search', action='store_true', default=False,
                       help='Use grid search for SVM hyperparameters')

    # Linear model parameters
    parser.add_argument('--linear_alpha', type=float, default=1.0,
                       help='Regularization strength for Ridge/Lasso')

    # Neural network general parameters
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate for optimizer')
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout rate')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use for training')

    # MLP parameters
    parser.add_argument('--mlp_hidden_dims', type=int, nargs='+',
                       default=[64, 32],
                       help='MLP hidden layer dimensions (space-separated)')

    # CNN/LSTM sequence parameters
    parser.add_argument('--sequence_length', type=int, default=10,
                       help='Sequence length for CNN/LSTM')
    parser.add_argument('--step_size', type=int, default=5,
                       help='Step size for sliding window')

    # CNN parameters
    parser.add_argument('--cnn_filters', type=int, nargs='+',
                       default=[32, 64, 32],
                       help='CNN filter sizes (space-separated)')
    parser.add_argument('--cnn_kernel_sizes', type=int, nargs='+',
                       default=[3, 3, 3],
                       help='CNN kernel sizes (space-separated)')
    parser.add_argument('--cnn_fc_dims', type=int, nargs='+',
                       default=[32, 16],
                       help='CNN fully connected layer dimensions')

    # LSTM parameters
    parser.add_argument('--lstm_hidden_dim', type=int, default=64,
                       help='LSTM hidden dimension')
    parser.add_argument('--lstm_num_layers', type=int, default=2,
                       help='Number of LSTM layers')
    parser.add_argument('--lstm_bidirectional', action='store_true', default=False,
                       help='Use bidirectional LSTM')

    # GNN parameters
    parser.add_argument('--gnn_hidden_dims', type=int, nargs='+',
                       default=[64, 32],
                       help='GNN hidden layer dimensions')

    # Results and visualization
    parser.add_argument('--results_dir', type=str, default='Results',
                       help='Directory to save results')
    parser.add_argument('--save_model', action='store_true', default=True,
                       help='Save trained model')
    parser.add_argument('--plot_results', action='store_true', default=True,
                       help='Generate and save plots')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='Print training progress')

    # Experiment naming
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Custom experiment name (auto-generated if not provided)')

    args = parser.parse_args()

    # Post-processing
    if args.device == 'auto':
        import torch
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Generate experiment name if not provided
    if args.experiment_name is None:
        args.experiment_name = generate_experiment_name(args)

    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)

    return args


def generate_experiment_name(args):
    """
    Generate experiment name based on model and parameters.

    Args:
        args: Parsed arguments

    Returns:
        Experiment name string
    """
    import datetime

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    if args.model in ['linear', 'ridge', 'lasso']:
        name = f"{args.model}"
        if args.model in ['ridge', 'lasso']:
            name += f"_alpha{args.linear_alpha}"

    elif args.model == 'svm':
        name = f"svm_{args.svm_kernel}_C{args.svm_C}"
        if args.svm_grid_search:
            name += "_gridsearch"

    elif args.model == 'mlp':
        hidden_str = '-'.join(map(str, args.mlp_hidden_dims))
        name = f"mlp_{hidden_str}_lr{args.learning_rate}"

    elif args.model == 'cnn':
        filters_str = '-'.join(map(str, args.cnn_filters))
        name = f"cnn_f{filters_str}_seq{args.sequence_length}"

    elif args.model == 'lstm':
        direction = 'bi' if args.lstm_bidirectional else 'uni'
        name = f"lstm_{direction}_h{args.lstm_hidden_dim}_l{args.lstm_num_layers}"

    elif args.model in ['gcn', 'gat']:
        hidden_str = '-'.join(map(str, args.gnn_hidden_dims))
        name = f"{args.model}_{hidden_str}"

    else:
        name = args.model

    return f"{name}_{timestamp}"


def print_args(args):
    """
    Print parsed arguments in a formatted way.

    Args:
        args: Parsed arguments
    """
    print("\n" + "="*60)
    print("Experiment Configuration")
    print("="*60)

    for arg in vars(args):
        print(f"{arg:.<30} {getattr(args, arg)}")

    print("="*60 + "\n")


if __name__ == "__main__":
    # Test argument parsing
    args = get_args()
    print_args(args)
