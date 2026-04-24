"""
Unified main entry point for PK/PD prediction training.

Supports:
- MLP (hierarchical) with modes: separate, joint, dual_stage, shared
- GNN (hierarchical) with modes: joint, sequential

Usage:
    python main.py --model mlp --mode dual_stage --epochs 300
    python main.py --model gnn --mode joint --epochs 150
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader as TorchDataLoader

# Local imports
from Utils.args import get_args, print_args
from Utils.data_loader import prepare_pkpd_data, prepare_lstm_sequences, collate_lstm_batch
from Utils.data_process import PKPDDataset, prepare_gnn_data, collate_pkpd
from Utils.training import train_gnn, train_mlp, train_lstm, evaluate_gnn, evaluate_mlp, evaluate_lstm, train_mlp_v2, evaluate_mlp_v2
from Utils.plot import plot_gnn_patient_timeseries, plot_patient_timeseries, plot_predictions, plot_lstm_patient_timeseries
from Utils.log import plot_metrics, logger
from Models.mlp import HierarchicalPKPDMLP
from Models.gnn import HierarchicalPKPDGNN
from Models.lstm import HierarchicalPKPDLSTM
from Models.quantum import HQGNN, HierarchicalHQCNN, HierarchicalQNN, HierarchicalResQNN, HQLSTM

# Set style
sns.set_style('whitegrid')
plt.rcParams['font.size'] = 10


plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "mathtext.fontset": "cm",  # 'cm' = Computer Modern (the LaTeX font)
})



def main():
    # Get arguments
    args = get_args()
    print_args(args)

    start_time = time.time()
    logger.info(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    csv_data_path = 'Data/' + args.csv_path + '.csv'
    # Create save directory
    timestamp = time.strftime('%y_%m_%d_%H_%M_%S')
    save_dir = os.path.join(args.save_dir, f"{timestamp}_{args.experiment_name}")
    os.makedirs(save_dir, exist_ok=True)

    # Set device
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # Set random seed
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    # Default transforms (overridden per model below)
    pk_transform = 'none'
    pd_transform = 'none'

    # ==================== Model Selection ====================
    if args.model == 'mlp':
        logger.info("=" * 60)
        logger.info("TRAINING HIERARCHICAL MLP")
        logger.info("=" * 60)

        # Determine target transforms
        pk_transform = 'log' if args.log_pk else 'none'
        pd_transform = 'sqrt' if args.sqrt_pd else 'none'
        use_v2 = args.log_pk or args.sqrt_pd

        # Prepare MLP data
        if args.combine:
            # Use all data for training (no split)
            data = prepare_pkpd_data(
                csv_path=csv_data_path,
                test_size=args.test_size,
                val_size=args.val_size,
                random_state=args.random_seed,
                use_perkg=args.use_perkg,
                time_windows=args.time_windows,
                half_lives=args.half_lives,
                add_decay=args.add_decay,
                stratified_split=False,
                pk_transform=pk_transform,
                pd_transform=pd_transform,
                normalize_data=args.normalize_data,
                add_pk_summary=args.add_pk_summary,
                add_pk_cumulative=args.add_pk_cumulative,
                no_placebo=args.no_placebo,
            )
            # logger.info("COMBINE MODE: Using ALL data for training")
            # Use same data for train/val/test
            data['val_pk'] = data['train_pk']
            data['val_pd'] = data['train_pd']
            data['test_pk'] = data['train_pk']
            data['test_pd'] = data['train_pd']
        else:
            data = prepare_pkpd_data(
                csv_path=csv_data_path,
                test_size=args.test_size,
                val_size=args.val_size,
                random_state=args.random_seed,
                use_perkg=args.use_perkg,
                time_windows=args.time_windows,
                half_lives=args.half_lives,
                add_decay=args.add_decay,
                stratified_split=args.stratified_split,
                pk_transform=pk_transform,
                pd_transform=pd_transform,
                normalize_data=args.normalize_data,
                add_pk_summary=args.add_pk_summary,
                add_pk_cumulative=args.add_pk_cumulative,
                no_placebo=args.no_placebo,
            )

        # Create datasets
        train_dataset = PKPDDataset(data['train_pk'], data['train_pd'])
        val_dataset = PKPDDataset(data['val_pk'], data['val_pd'])
        test_dataset = PKPDDataset(data['test_pk'], data['test_pd'])

        train_loader = TorchDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_pkpd)
        val_loader = TorchDataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_pkpd)
        test_loader = TorchDataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_pkpd)

        # Create model
        model = HierarchicalPKPDMLP(
            mode=args.mode,
            pk_input_dim=data['n_features'],
            pd_input_dim=data['n_features'],
            hidden_dim=args.hidden_dim,
            n_blocks=args.n_blocks,
            dropout=args.dropout,
            head_hidden=args.head_hidden,
        )

        logger.info(f"Model: {args.model.upper()}, Mode: {args.mode}")
        logger.info(f"Input dim: {data['n_features']}, Hidden dim: {args.hidden_dim}")
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")

        # Train & Evaluate
        if use_v2:
            model, history = train_mlp_v2(model, train_loader, val_loader, args, device,
                                          pk_transform=pk_transform, pd_transform=pd_transform)
            logger.info("=" * 60)
            logger.info("EVALUATION (original scale)")
            logger.info("=" * 60)
            train_results = evaluate_mlp_v2(model, train_loader, device,
                                            pk_transform=pk_transform, pd_transform=pd_transform)
            test_results = evaluate_mlp_v2(model, test_loader, device,
                                           pk_transform=pk_transform, pd_transform=pd_transform)
        else:
            model, history = train_mlp(model, train_loader, val_loader, args, device)
            logger.info("=" * 60)
            logger.info("EVALUATION")
            logger.info("=" * 60)
            train_results = evaluate_mlp(model, train_loader, device)
            test_results = evaluate_mlp(model, test_loader, device)

    elif args.model == 'gnn':
        logger.info("=" * 60)
        logger.info("TRAINING HIERARCHICAL GNN")
        logger.info("=" * 60)

        # Prepare GNN data
        data = prepare_gnn_data(args)

        if args.combine:
            data['val_data'] = data['train_data']
            data['test_data'] = data['train_data']

        # Create model
        model = HierarchicalPKPDGNN(
            feature_dim=data['feature_dim'],
            hidden_dim=args.gnn_hidden_dim,
            num_layers_pk=args.num_layers_pk,
            num_layers_pd=args.num_layers_pd,
            dropout=args.dropout,
            use_attention=args.use_attention,
            use_gating=args.use_gating,
        )


        logger.info(f"Model: {args.model.upper()}, Mode: {args.mode}")
        logger.info(f"Feature dim: {data['feature_dim']}, Hidden dim: {args.gnn_hidden_dim}")
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")

        # Train
        model, history = train_gnn(model, data['train_data'], data['val_data'], args, device)

        # Evaluate
        logger.info("=" * 60)
        logger.info("EVALUATION")
        logger.info("=" * 60)

        train_results = evaluate_gnn(model, data['train_data'], device)
        test_results = evaluate_gnn(model, data['test_data'], device)
    
    elif args.model == 'hqgnn':
        logger.info("=" * 60)
        logger.info("TRAINING HIERARCHICAL GNN")
        logger.info("=" * 60)

        # Prepare GNN data
        data = prepare_gnn_data(args)

        if args.combine:
            data['val_data'] = data['train_data']
            data['test_data'] = data['train_data']

        # Create model
        model = HQGNN(
            feature_dim=data['feature_dim'],
            hidden_dim=args.gnn_hidden_dim,
            num_layers_pk=args.num_layers_pk,
            num_layers_pd=args.num_layers_pd,
            dropout=args.dropout,
            use_attention=args.use_attention,
            use_gating=args.use_gating,
            n_qlayers=args.n_qlayers,
            n_qubits=args.n_qubits,
            using_hqcnn=args.using_hqcnn,
        )


        logger.info(f"Model: {args.model.upper()}, Mode: {args.mode}")
        logger.info(f"Feature dim: {data['feature_dim']}, Hidden dim: {args.gnn_hidden_dim}")
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")

        # Train
        model, history = train_gnn(model, data['train_data'], data['val_data'], args, device)

        # Evaluate
        logger.info("=" * 60)
        logger.info("EVALUATION")
        logger.info("=" * 60)

        train_results = evaluate_gnn(model, data['train_data'], device)
        test_results = evaluate_gnn(model, data['test_data'], device)

    elif args.model == 'hqcnn':
        logger.info("=" * 60)
        logger.info("TRAINING HIERARCHICAL HQCNN (Quantum)")
        logger.info("=" * 60)

        # Prepare data (same as MLP)
        if args.combine:
            data = prepare_pkpd_data(
                csv_path=csv_data_path,
                test_size=args.test_size,
                val_size=args.val_size,
                random_state=args.random_seed,
                use_perkg=args.use_perkg,
                time_windows=args.time_windows,
                half_lives=args.half_lives,
                add_decay=args.add_decay,
                stratified_split=False,
                normalize_data=args.normalize_data,
                add_pk_summary=args.add_pk_summary,
                add_pk_cumulative=args.add_pk_cumulative,
                no_placebo=args.no_placebo,
            )
            # logger.info("COMBINE MODE: Using ALL data for training")
            data['val_pk'] = data['train_pk']
            data['val_pd'] = data['train_pd']
            data['test_pk'] = data['train_pk']
            data['test_pd'] = data['train_pd']
        else:
            data = prepare_pkpd_data(
                csv_path=csv_data_path,
                test_size=args.test_size,
                val_size=args.val_size,
                random_state=args.random_seed,
                use_perkg=args.use_perkg,
                time_windows=args.time_windows,
                half_lives=args.half_lives,
                add_decay=args.add_decay,
                stratified_split=args.stratified_split,
                normalize_data=args.normalize_data,
                add_pk_summary=args.add_pk_summary,
                add_pk_cumulative=args.add_pk_cumulative,
                no_placebo=args.no_placebo,
            )

        # Create datasets
        train_dataset = PKPDDataset(data['train_pk'], data['train_pd'])
        val_dataset = PKPDDataset(data['val_pk'], data['val_pd'])
        test_dataset = PKPDDataset(data['test_pk'], data['test_pd'])

        train_loader = TorchDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_pkpd)
        val_loader = TorchDataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_pkpd)
        test_loader = TorchDataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_pkpd)

        # Create HQCNN model
        model = HierarchicalHQCNN(
            pk_input_dim=data['n_features'],
            pd_input_dim=data['n_features'],
            num_layers=args.hqcnn_num_layers,
            mode=args.mode,
        )

        logger.info(f"Model: {args.model.upper()}, Mode: {args.mode}")
        logger.info(f"Input dim: {data['n_features']}, Quantum layers: {args.hqcnn_num_layers}")
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")

        # Train (reuse MLP training function)
        model, history = train_mlp(model, train_loader, val_loader, args, device)

        # Evaluate
        logger.info("=" * 60)
        logger.info("EVALUATION")
        logger.info("=" * 60)

        train_results = evaluate_mlp(model, train_loader, device)
        test_results = evaluate_mlp(model, test_loader, device)

    elif args.model == 'qnn':
        logger.info("=" * 60)
        logger.info("TRAINING HIERARCHICAL QNN (Amplitude Embedding)")
        logger.info("=" * 60)

        # Prepare data (same as MLP)
        if args.combine:
            data = prepare_pkpd_data(
                csv_path=csv_data_path,
                test_size=args.test_size,
                val_size=args.val_size,
                random_state=args.random_seed,
                use_perkg=args.use_perkg,
                time_windows=args.time_windows,
                half_lives=args.half_lives,
                add_decay=args.add_decay,
                stratified_split=False,
                normalize_data=args.normalize_data,
                add_pk_summary=args.add_pk_summary,
                add_pk_cumulative=args.add_pk_cumulative,
                no_placebo=args.no_placebo,
            )
            data['val_pk'] = data['train_pk']
            data['val_pd'] = data['train_pd']
            data['test_pk'] = data['train_pk']
            data['test_pd'] = data['train_pd']
        else:
            data = prepare_pkpd_data(
                csv_path=csv_data_path,
                test_size=args.test_size,
                val_size=args.val_size,
                random_state=args.random_seed,
                use_perkg=args.use_perkg,
                time_windows=args.time_windows,
                half_lives=args.half_lives,
                add_decay=args.add_decay,
                stratified_split=args.stratified_split,
                normalize_data=args.normalize_data,
                add_pk_summary=args.add_pk_summary,
                add_pk_cumulative=args.add_pk_cumulative,
                no_placebo=args.no_placebo,
            )

        train_dataset = PKPDDataset(data['train_pk'], data['train_pd'])
        val_dataset = PKPDDataset(data['val_pk'], data['val_pd'])
        test_dataset = PKPDDataset(data['test_pk'], data['test_pd'])

        train_loader = TorchDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_pkpd)
        val_loader = TorchDataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_pkpd)
        test_loader = TorchDataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_pkpd)

        model = HierarchicalQNN(
            pk_input_dim=data['n_features'],
            pd_input_dim=data['n_features'],
            n_qubits=args.n_qubits,
            n_qlayers=args.n_qlayers,
            mode=args.mode,
        )

        logger.info(f"Model: {args.model.upper()}, Mode: {args.mode}")
        logger.info(f"Input dim: {data['n_features']}, Qubits: {args.n_qubits}, Q-layers: {args.n_qlayers}")
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")

        model, history = train_mlp(model, train_loader, val_loader, args, device)

        logger.info("=" * 60)
        logger.info("EVALUATION")
        logger.info("=" * 60)

        train_results = evaluate_mlp(model, train_loader, device)
        test_results = evaluate_mlp(model, test_loader, device)

    elif args.model == 'resqnn':
        logger.info("=" * 60)
        logger.info("TRAINING HIERARCHICAL RESIDUAL QNN (MLP scaffold + quantum core)")
        logger.info("=" * 60)

        # Reuse MLP data pipeline
        if args.combine:
            data = prepare_pkpd_data(
                csv_path=csv_data_path,
                test_size=args.test_size,
                val_size=args.val_size,
                random_state=args.random_seed,
                use_perkg=args.use_perkg,
                time_windows=args.time_windows,
                half_lives=args.half_lives,
                add_decay=args.add_decay,
                stratified_split=False,
                normalize_data=args.normalize_data,
                add_pk_summary=args.add_pk_summary,
                add_pk_cumulative=args.add_pk_cumulative,
                no_placebo=args.no_placebo,
            )
            data['val_pk'] = data['train_pk']
            data['val_pd'] = data['train_pd']
            data['test_pk'] = data['train_pk']
            data['test_pd'] = data['train_pd']
        else:
            data = prepare_pkpd_data(
                csv_path=csv_data_path,
                test_size=args.test_size,
                val_size=args.val_size,
                random_state=args.random_seed,
                use_perkg=args.use_perkg,
                time_windows=args.time_windows,
                half_lives=args.half_lives,
                add_decay=args.add_decay,
                stratified_split=args.stratified_split,
                normalize_data=args.normalize_data,
                add_pk_summary=args.add_pk_summary,
                add_pk_cumulative=args.add_pk_cumulative,
                no_placebo=args.no_placebo,
            )

        train_dataset = PKPDDataset(data['train_pk'], data['train_pd'])
        val_dataset = PKPDDataset(data['val_pk'], data['val_pd'])
        test_dataset = PKPDDataset(data['test_pk'], data['test_pd'])

        train_loader = TorchDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_pkpd)
        val_loader = TorchDataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_pkpd)
        test_loader = TorchDataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_pkpd)

        model = HierarchicalResQNN(
            mode=args.mode,
            pk_input_dim=data['n_features'],
            pd_input_dim=data['n_features'],
            hidden_dim=args.hidden_dim,
            n_blocks=args.n_blocks,
            dropout=args.dropout,
            head_hidden=args.head_hidden,
            n_qubits=args.n_qubits,
            n_qlayers=args.n_qlayers,
            q_dev=None,
        )

        logger.info(f"Model: {args.model.upper()}, Mode: {args.mode}")
        logger.info(f"Input dim: {data['n_features']}, Hidden: {args.hidden_dim}, Blocks: {args.n_blocks}, "
                    f"Qubits: {args.n_qubits}, Q-layers: {args.n_qlayers}")
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")

        model, history = train_mlp(model, train_loader, val_loader, args, device)

        logger.info("=" * 60)
        logger.info("EVALUATION")
        logger.info("=" * 60)

        train_results = evaluate_mlp(model, train_loader, device)
        test_results = evaluate_mlp(model, test_loader, device)

    elif args.model in ['lstm', 'hqlstm']:
        logger.info("=" * 60)
        logger.info(f"TRAINING HIERARCHICAL {'HQLSTM (Quantum)' if args.model == 'hqlstm' else 'LSTM'}")
        logger.info("=" * 60)

        # Prepare LSTM sequence data
        data = prepare_lstm_sequences(
            csv_path=csv_data_path,
            test_size=args.test_size,
            val_size=args.val_size,
            random_state=args.random_seed,
            use_perkg=args.use_perkg,
            time_windows=args.time_windows,
            half_lives=args.half_lives,
            add_decay=args.add_decay,
            stratified_split=args.stratified_split,
            combine=args.combine,
            normalize_data=args.normalize_data,
            add_pk_summary=args.add_pk_summary,
            add_pk_cumulative=args.add_pk_cumulative,
            no_placebo=args.no_placebo,
        )

        if args.combine:
            # logger.info("COMBINE MODE: Using ALL data for training")
            data['val_sequences'] = data['train_sequences']
            data['test_sequences'] = data['train_sequences']

        # Create model
        if args.model == 'lstm':
            model = HierarchicalPKPDLSTM(
                input_dim=data['n_features'],
                hidden_dim=args.lstm_hidden_dim,
                num_layers=args.lstm_num_layers,
                dropout=args.dropout,
                bidirectional=args.lstm_bidirectional,
                use_gating=args.use_gating,
                mode=args.mode,
            )
        else:  # hqlstm
            model = HQLSTM(
                input_dim=data['n_features'],
                hidden_dim=args.lstm_hidden_dim,
                num_layers=args.lstm_num_layers,
                dropout=args.dropout,
                bidirectional=args.lstm_bidirectional,
                use_gating=args.use_gating,
                mode=args.mode,
                n_qlayers=args.n_qlayers,
                n_qubits=args.n_qubits,
                using_hqcnn=args.using_hqcnn,
            )

        logger.info(f"Model: {args.model.upper()}, Mode: {args.mode}")
        logger.info(f"Input dim: {data['n_features']}, Hidden dim: {args.lstm_hidden_dim}")
        logger.info(f"LSTM layers: {args.lstm_num_layers}, Bidirectional: {args.lstm_bidirectional}")
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")

        # Train
        model, history = train_lstm(model, data, args, device)

        # Evaluate
        logger.info("=" * 60)
        logger.info("EVALUATION")
        logger.info("=" * 60)

        train_results = evaluate_lstm(model, data['train_sequences'], device)
        test_results = evaluate_lstm(model, data['test_sequences'], device)

    else:
        raise ValueError(f"Unsupported model: {args.model}")

    # ==================== Print Final Results ====================
    logger.info("=" * 60)
    logger.info("FINAL RESULTS")
    logger.info("=" * 60)

    logger.info("PK Metrics:")
    logger.info(f"  Train - MSE: {train_results['pk']['MSE']:.4f}, RMSE: {train_results['pk']['RMSE']:.4f}, MAE: {train_results['pk']['MAE']:.4f}, R2: {train_results['pk']['R2']:.4f}")
    logger.info(f"  Test  - MSE: {test_results['pk']['MSE']:.4f}, RMSE: {test_results['pk']['RMSE']:.4f}, MAE: {test_results['pk']['MAE']:.4f}, R2: {test_results['pk']['R2']:.4f}")

    logger.info("PD Metrics:")
    logger.info(f"  Train - MSE: {train_results['pd']['MSE']:.4f}, RMSE: {train_results['pd']['RMSE']:.4f}, MAE: {train_results['pd']['MAE']:.4f}, R2: {train_results['pd']['R2']:.4f}")
    logger.info(f"  Test  - MSE: {test_results['pd']['MSE']:.4f}, RMSE: {test_results['pd']['RMSE']:.4f}, MAE: {test_results['pd']['MAE']:.4f}, R2: {test_results['pd']['R2']:.4f}")

    # ==================== Save Results ====================

    # Save model
    if args.save_model:
        model_path = os.path.join(save_dir, 'model.pth')
        torch.save(model.state_dict(), model_path)
        logger.info(f"Saved model to {model_path}")

    # Save metrics to txt
    metrics_path = os.path.join(save_dir, 'metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write(f"EXPERIMENT: {args.experiment_name}\n")
        f.write(f"Model: {args.model.upper()}, Mode: {args.mode}\n")
        f.write("=" * 60 + "\n\n")

        f.write("Configuration:\n")
        f.write(f"  Hidden dim: {args.hidden_dim if args.model == 'mlp' else args.gnn_hidden_dim}\n")
        f.write(f"  Dropout: {args.dropout}\n")
        f.write(f"  Learning rate: {args.learning_rate}\n")
        f.write(f"  Batch size: {args.batch_size}\n")
        f.write(f"  Epochs: {args.epochs}\n\n")

        f.write("PK Metrics:\n")
        f.write(f"  Train - MSE: {train_results['pk']['MSE']:.4f}, RMSE: {train_results['pk']['RMSE']:.4f}, MAE: {train_results['pk']['MAE']:.4f}, R2: {train_results['pk']['R2']:.4f}\n")
        f.write(f"  Test  - MSE: {test_results['pk']['MSE']:.4f}, RMSE: {test_results['pk']['RMSE']:.4f}, MAE: {test_results['pk']['MAE']:.4f}, R2: {test_results['pk']['R2']:.4f}\n\n")

        f.write("PD Metrics:\n")
        f.write(f"  Train - MSE: {train_results['pd']['MSE']:.4f}, RMSE: {train_results['pd']['RMSE']:.4f}, MAE: {train_results['pd']['MAE']:.4f}, R2: {train_results['pd']['R2']:.4f}\n")
        f.write(f"  Test  - MSE: {test_results['pd']['MSE']:.4f}, RMSE: {test_results['pd']['RMSE']:.4f}, MAE: {test_results['pd']['MAE']:.4f}, R2: {test_results['pd']['R2']:.4f}\n")

    logger.info(f"Saved metrics to {metrics_path}")

    # Save plots
    if args.save_plots:
        # Training history
        plot_metrics(history, save_path=os.path.join(save_dir, 'training_history.png'))

        # Prediction scatter plots
        plot_predictions(train_results, test_results, save_dir, f"{args.model}_{args.mode}")

        # Patient time series (for MLP and HQCNN - need raw data with IDs)
        if args.model in ['mlp', 'hqcnn', 'qnn', 'resqnn']:
            # Combine train and test data for patient plots
            all_pk_data = {
                'X': np.vstack([data['train_pk']['X'], data['val_pk']['X'], data['test_pk']['X']]),
                'y': np.concatenate([data['train_pk']['y'], data['val_pk']['y'], data['test_pk']['y']]),
                'ids': np.concatenate([data['train_pk']['ids'], data['val_pk']['ids'], data['test_pk']['ids']]),
                'times': np.concatenate([data['train_pk']['times'], data['val_pk']['times'], data['test_pk']['times']]),
            }
            all_pd_data = {
                'X': np.vstack([data['train_pd']['X'], data['val_pd']['X'], data['test_pd']['X']]),
                'y': np.concatenate([data['train_pd']['y'], data['val_pd']['y'], data['test_pd']['y']]),
                'ids': np.concatenate([data['train_pd']['ids'], data['val_pd']['ids'], data['test_pd']['ids']]),
                'times': np.concatenate([data['train_pd']['times'], data['val_pd']['times'], data['test_pd']['times']]),
            }
            plot_patient_timeseries(all_pk_data, all_pd_data, model, device,
                                    save_dir, f"{args.model}_{args.mode}",
                                    patient_ids=[9, 13, 26, 46],
                                    pk_transform=pk_transform,
                                    pd_transform=pd_transform)
        elif args.model in ['gnn', 'hqgnn']:
            # Deduplicate by patient_id (combine mode causes overlap)
            seen_ids = set()
            all_gnn_data = []
            for d in data['train_data'] + data['val_data'] + data['test_data']:
                if d.patient_id not in seen_ids:
                    seen_ids.add(d.patient_id)
                    all_gnn_data.append(d)
            plot_gnn_patient_timeseries(all_gnn_data, model, device,
                                        save_dir, f"{args.model}_{args.mode}",
                                        patient_ids=[9, 13, 26, 46])
        elif args.model in ['lstm', 'hqlstm']:
            # Deduplicate by patient id (combine mode causes overlap)
            seen_ids = set()
            all_sequences = []
            for s in data['train_sequences'] + data['val_sequences'] + data['test_sequences']:
                if s['id'] not in seen_ids:
                    seen_ids.add(s['id'])
                    all_sequences.append(s)
            plot_lstm_patient_timeseries(all_sequences, model, device,
                                         save_dir, f"{args.model}_{args.mode}",
                                         patient_ids=[9, 13, 26, 46])

    elapsed = time.time() - start_time
    hours, remainder = divmod(int(elapsed), 3600)
    minutes, seconds = divmod(remainder, 60)

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE!")
    logger.info(f"Total time: {hours}h {minutes}m {seconds}s")
    logger.info(f"Results saved to: {save_dir}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
