#!/usr/bin/env python3
"""
Main training script for PK/PD prediction models.
Supports multiple model types with command-line argument configuration.
"""

import sys
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Add Utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'Utils'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'Model'))

from Utils.args import get_args, print_args
from Utils.pre_processing import PKPDDataProcessor
from Utils.plotting import create_experiment_report

# Import models
from Model.linear_regression import LinearRegressionModel
from Model.svm import SVMModel, SVMGridSearch
from Model.mlp import MLPModel
from Model.cnn import CNNModel
from Model.lstm import LSTMModel
from Model.gnn import GNNModel


def train_linear_model(args, X_train, X_test, y_train, y_test):
    """Train linear regression models (OLS, Ridge, Lasso)."""
    print(f"\n{'='*60}")
    print(f"Training {args.model.upper()} Model")
    print('='*60)

    model = LinearRegressionModel(
        model_type=args.model,
        alpha=args.linear_alpha
    )

    model.train(X_train, y_train)
    metrics = model.evaluate(X_test, y_test)
    y_pred = model.predict(X_test)

    # Get feature importance
    feature_importance = model.get_feature_importance(X_train.columns.tolist())

    return model, metrics, y_pred, {'feature_importance': feature_importance}


def train_svm_model(args, X_train, X_test, y_train, y_test):
    """Train SVM model."""
    print(f"\n{'='*60}")
    print(f"Training SVM Model")
    print('='*60)

    if args.svm_grid_search:
        model = SVMGridSearch(kernel=args.svm_kernel, cv=3)
        model.train(X_train, y_train)
    else:
        model = SVMModel(
            kernel=args.svm_kernel,
            C=args.svm_C,
            epsilon=args.svm_epsilon
        )
        model.train(X_train, y_train)

    metrics = model.evaluate(X_test, y_test)
    y_pred = model.predict(X_test)

    return model, metrics, y_pred, {}


def train_mlp_model(args, X_train, X_test, y_train, y_test):
    """Train MLP model."""
    print(f"\n{'='*60}")
    print(f"Training MLP Model")
    print('='*60)

    input_dim = X_train.shape[1]

    model = MLPModel(
        input_dim=input_dim,
        hidden_dims=args.mlp_hidden_dims,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        device=args.device
    )

    model.train(
        X_train, y_train,
        X_val=X_test, y_val=y_test,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=args.verbose
    )

    metrics = model.evaluate(X_test, y_test)
    y_pred = model.predict(X_test)

    extra_info = {
        'train_losses': model.train_losses,
        'val_losses': model.val_losses
    }

    return model, metrics, y_pred, extra_info


def train_cnn_model(args, processor):
    """Train CNN model with sequence data."""
    print(f"\n{'='*60}")
    print(f"Training CNN Model")
    print('='*60)

    # Prepare sequence data
    X, y = processor.prepare_features_target()
    print("\nPreparing sequence data for CNN...")
    X_seq, y_seq = processor.prepare_sequence_data(
        X, y,
        sequence_length=args.sequence_length,
        step_size=args.step_size
    )

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=args.test_size, random_state=args.random_seed
    )

    input_channels = X_train.shape[2]
    sequence_length = X_train.shape[1]

    model = CNNModel(
        input_channels=input_channels,
        sequence_length=sequence_length,
        conv_filters=args.cnn_filters,
        kernel_sizes=args.cnn_kernel_sizes,
        fc_dims=args.cnn_fc_dims,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        device=args.device
    )

    model.train(
        X_train, y_train,
        X_val=X_test, y_val=y_test,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=args.verbose
    )

    metrics = model.evaluate(X_test, y_test)
    y_pred = model.predict(X_test)

    extra_info = {
        'train_losses': model.train_losses,
        'val_losses': model.val_losses
    }

    return model, metrics, y_pred, y_test, extra_info


def train_lstm_model(args, processor):
    """Train LSTM model with sequence data."""
    print(f"\n{'='*60}")
    print(f"Training LSTM Model")
    print('='*60)

    # Prepare sequence data
    X, y = processor.prepare_features_target()
    print("\nPreparing sequence data for LSTM...")
    X_seq, y_seq = processor.prepare_sequence_data(
        X, y,
        sequence_length=args.sequence_length,
        step_size=args.step_size
    )

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=args.test_size, random_state=args.random_seed
    )

    input_dim = X_train.shape[2]

    model = LSTMModel(
        input_dim=input_dim,
        hidden_dim=args.lstm_hidden_dim,
        num_layers=args.lstm_num_layers,
        dropout=args.dropout,
        bidirectional=args.lstm_bidirectional,
        learning_rate=args.learning_rate,
        device=args.device
    )

    model.train(
        X_train, y_train,
        X_val=X_test, y_val=y_test,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=args.verbose
    )

    metrics = model.evaluate(X_test, y_test)
    y_pred = model.predict(X_test)

    extra_info = {
        'train_losses': model.train_losses,
        'val_losses': model.val_losses
    }

    return model, metrics, y_pred, y_test, extra_info


def train_gnn_model(args, processor):
    """Train GNN model (GCN or GAT)."""
    print(f"\n{'='*60}")
    print(f"Training {args.model.upper()} Model")
    print('='*60)

    # Get graph data
    print("\nPreparing graph data for GNN...")
    graph_data = processor.get_graph_data()

    input_dim = graph_data['node_features'].shape[1]

    model = GNNModel(
        input_dim=input_dim,
        model_type=args.model,
        hidden_dims=args.gnn_hidden_dims,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        device=args.device
    )

    # Prepare graph data
    data = model.prepare_graph_data(
        graph_data['node_features'],
        graph_data['edge_index'],
        graph_data['targets'],
        test_size=args.test_size,
        random_state=args.random_seed
    )

    # Train
    model.train(data, epochs=args.epochs, verbose=args.verbose)

    # Evaluate
    metrics = model.evaluate(data)

    # Get predictions
    y_pred = model.predict(data, mask=data.test_mask)
    y_test = data.y[data.test_mask].cpu().numpy().flatten()

    extra_info = {
        'train_losses': model.train_losses,
        'val_losses': model.val_losses
    }

    return model, metrics, y_pred, y_test, extra_info


def main():
    """Main training function."""
    # Parse arguments
    args = get_args()
    print_args(args)

    # Initialize data processor
    processor = PKPDDataProcessor(data_path=args.data_path, verbose=False)

    # Load data
    processor.load_data()

    # Train model based on type
    if args.model in ['linear', 'ridge', 'lasso']:
        # Standard tabular models
        data = processor.get_full_pipeline(
            target_col=args.target_col,
            test_size=args.test_size,
            random_state=args.random_seed,
            scale_features=args.scale_features,
            scale_target=args.scale_target
        )
        X_train, X_test = data['X_train'], data['X_test']
        y_train, y_test = data['y_train'], data['y_test']

        model, metrics, y_pred, extra_info = train_linear_model(
            args, X_train, X_test, y_train, y_test
        )

    elif args.model == 'svm':
        # SVM model
        data = processor.get_full_pipeline(
            target_col=args.target_col,
            test_size=args.test_size,
            random_state=args.random_seed,
            scale_features=args.scale_features,
            scale_target=args.scale_target
        )
        X_train, X_test = data['X_train'], data['X_test']
        y_train, y_test = data['y_train'], data['y_test']

        model, metrics, y_pred, extra_info = train_svm_model(
            args, X_train, X_test, y_train, y_test
        )

    elif args.model == 'mlp':
        # MLP model
        data = processor.get_full_pipeline(
            target_col=args.target_col,
            test_size=args.test_size,
            random_state=args.random_seed,
            scale_features=args.scale_features,
            scale_target=args.scale_target
        )
        X_train, X_test = data['X_train'], data['X_test']
        y_train, y_test = data['y_train'], data['y_test']

        model, metrics, y_pred, extra_info = train_mlp_model(
            args, X_train, X_test, y_train, y_test
        )

    elif args.model == 'cnn':
        # CNN model (requires sequence data)
        model, metrics, y_pred, y_test, extra_info = train_cnn_model(args, processor)

    elif args.model == 'lstm':
        # LSTM model (requires sequence data)
        model, metrics, y_pred, y_test, extra_info = train_lstm_model(args, processor)

    elif args.model in ['gcn', 'gat']:
        # GNN models
        model, metrics, y_pred, y_test, extra_info = train_gnn_model(args, processor)

    else:
        raise ValueError(f"Unknown model type: {args.model}")

    # Convert to numpy if needed
    if hasattr(y_test, 'values'):
        y_test = y_test.values
    if isinstance(y_test, pd.Series):
        y_test = y_test.values

    # Create experiment report
    if args.plot_results:
        exp_dir = create_experiment_report(
            experiment_name=args.experiment_name,
            args=args,
            metrics=metrics,
            y_true=y_test,
            y_pred=y_pred,
            train_losses=extra_info.get('train_losses'),
            val_losses=extra_info.get('val_losses'),
            feature_importance=extra_info.get('feature_importance'),
            results_dir=args.results_dir
        )

        # Save model if requested
        if args.save_model and hasattr(model, 'save_model'):
            model_path = os.path.join(exp_dir, 'model.pt')
            model.save_model(model_path)

    # Final summary
    print(f"\n{'='*60}")
    print("Training Complete!")
    print('='*60)
    print(f"Experiment: {args.experiment_name}")
    print(f"\nFinal Test Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric.upper():.<20} {value:.6f}")
    print('='*60)


if __name__ == "__main__":
    main()
