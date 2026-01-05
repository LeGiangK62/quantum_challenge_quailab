# Usage Guide - PK/PD Prediction Models

This guide provides examples for running all available models.

## Quick Start

### Basic Command Structure
```bash
python main.py --model MODEL_NAME [OPTIONS]
```

## Model Examples

### 1. Linear Regression Models

**Ordinary Least Squares (OLS)**
```bash
python main.py --model linear
```

**Ridge Regression**
```bash
python main.py --model ridge --linear_alpha 1.0
```

**Lasso Regression**
```bash
python main.py --model lasso --linear_alpha 0.5
```

### 2. Support Vector Machine (SVM)

**SVM with RBF kernel**
```bash
python main.py --model svm --svm_kernel rbf --svm_C 1.0
```

**SVM with Grid Search**
```bash
python main.py --model svm --svm_kernel rbf --svm_grid_search
```

**SVM with Linear kernel**
```bash
python main.py --model svm --svm_kernel linear
```

### 3. Multi-Layer Perceptron (MLP)

**Basic MLP**
```bash
python main.py --model mlp --epochs 100 --batch_size 32
```

**MLP with custom architecture**
```bash
python main.py --model mlp \
    --mlp_hidden_dims 128 64 32 \
    --epochs 150 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --dropout 0.3
```

**MLP with different learning rate**
```bash
python main.py --model mlp \
    --mlp_hidden_dims 64 32 \
    --learning_rate 0.0001 \
    --epochs 200
```

### 4. Convolutional Neural Network (CNN)

**Basic CNN**
```bash
python main.py --model cnn \
    --sequence_length 10 \
    --step_size 5 \
    --epochs 100
```

**CNN with custom architecture**
```bash
python main.py --model cnn \
    --cnn_filters 64 128 64 \
    --cnn_kernel_sizes 3 3 3 \
    --cnn_fc_dims 64 32 \
    --sequence_length 15 \
    --step_size 5 \
    --epochs 150 \
    --batch_size 32
```

### 5. Long Short-Term Memory (LSTM)

**Basic LSTM**
```bash
python main.py --model lstm \
    --sequence_length 10 \
    --step_size 5 \
    --epochs 100
```

**Bidirectional LSTM**
```bash
python main.py --model lstm \
    --lstm_hidden_dim 128 \
    --lstm_num_layers 3 \
    --lstm_bidirectional \
    --sequence_length 15 \
    --epochs 150
```

**LSTM with custom configuration**
```bash
python main.py --model lstm \
    --lstm_hidden_dim 64 \
    --lstm_num_layers 2 \
    --sequence_length 10 \
    --dropout 0.3 \
    --learning_rate 0.001 \
    --epochs 100
```

### 6. Graph Neural Networks (GNN)

**Graph Convolutional Network (GCN)**
```bash
python main.py --model gcn \
    --gnn_hidden_dims 64 32 \
    --epochs 100
```

**Graph Attention Network (GAT)**
```bash
python main.py --model gat \
    --gnn_hidden_dims 64 32 \
    --epochs 150 \
    --learning_rate 0.001
```

## Common Parameters

### Data Parameters
- `--data_path`: Path to CSV file (default: `Data/QIC2025-EstDat.csv`)
- `--target_col`: Target column name (default: `DV`)
- `--test_size`: Test set proportion (default: `0.2`)
- `--random_seed`: Random seed (default: `42`)
- `--scale_features`: Scale features (default: `True`)
- `--scale_target`: Scale target (default: `False`)

### Training Parameters (Neural Networks)
- `--epochs`: Number of training epochs (default: `100`)
- `--batch_size`: Batch size (default: `32`)
- `--learning_rate`: Learning rate (default: `0.001`)
- `--dropout`: Dropout rate (default: `0.2`)
- `--device`: Device to use (`auto`, `cpu`, `cuda`) (default: `auto`)

### Results Parameters
- `--results_dir`: Directory to save results (default: `Results`)
- `--save_model`: Save trained model (default: `True`)
- `--plot_results`: Generate plots (default: `True`)
- `--experiment_name`: Custom experiment name (default: auto-generated)

## Advanced Examples

### Custom Experiment Name
```bash
python main.py --model mlp \
    --experiment_name "mlp_final_experiment" \
    --epochs 200
```

### Training on CPU only
```bash
python main.py --model mlp \
    --device cpu \
    --epochs 100
```

### Disable plotting and model saving
```bash
python main.py --model linear \
    --plot_results False \
    --save_model False
```

### Custom data path
```bash
python main.py --model svm \
    --data_path "path/to/your/data.csv" \
    --target_col "TARGET_COLUMN"
```

## Output Structure

After running an experiment, results are saved in:
```
Results/
└── {experiment_name}_{timestamp}/
    ├── config.txt
    ├── model.pt (if applicable)
    ├── evaluation/
    │   ├── summary.png
    │   ├── predictions.png
    │   ├── distributions.png
    │   ├── metrics.txt
    │   └── metrics.png
    └── training/
        ├── training_history.png
        └── feature_importance.png (if applicable)
```

## Viewing Results

All plots are saved as PNG files. Metrics are also saved in a text file for easy reference.

## Tips

1. **Start simple**: Begin with linear models to establish a baseline
2. **Grid search**: Use `--svm_grid_search` for SVM to find optimal hyperparameters
3. **Monitor training**: Neural network models print progress every 10 epochs
4. **Experiment**: Try different architectures and hyperparameters
5. **Compare**: Run multiple experiments and compare results in the `Results/` directory
