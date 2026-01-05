# PK/PD Prediction Models - Complete Guide

## Project Overview

This project implements multiple machine learning models for Pharmacokinetic/Pharmacodynamic (PK/PD) prediction, including traditional ML models and deep learning approaches.

## 📁 Project Structure

```
quantum_challenge_quailab/
├── Data/
│   └── QIC2025-EstDat.csv          # PK/PD dataset
├── Utils/
│   ├── pre_processing.py           # Data preprocessing utilities
│   ├── args.py                     # Command-line argument parser
│   └── plotting.py                 # Visualization utilities
├── Model/
│   ├── linear_regression.py        # Linear models (OLS, Ridge, Lasso)
│   ├── svm.py                      # Support Vector Machines
│   ├── mlp.py                      # Multi-Layer Perceptron
│   ├── cnn.py                      # 1D Convolutional Neural Network
│   ├── lstm.py                     # Long Short-Term Memory
│   └── gnn.py                      # Graph Neural Networks (GCN, GAT)
├── Results/                        # Experiment results (auto-generated)
├── main.py                         # Main training script
├── run_examples.sh                 # Quick example runner
├── requirements.txt                # Python dependencies
├── environment.yml                 # Conda environment
├── INSTALL_INSTRUCTIONS.md         # Installation guide
├── USAGE_GUIDE.md                  # Detailed usage examples
└── QUICK_REFERENCE.md              # Quick command reference
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Using conda (recommended)
conda env create -f environment.yml
conda activate quantum_challenge

# OR using pip
pip install torch==2.0.1 torchvision>=0.15.0
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
pip install -r requirements.txt
```

### 2. Train a Model

**Train MLP (Multi-Layer Perceptron):**
```bash
python main.py --model mlp --epochs 100 --batch_size 32
```

**Or use the convenience script:**
```bash
./run_examples.sh mlp
```

### 3. View Results

Results are automatically saved in `Results/{experiment_name}_{timestamp}/`

## 📊 Available Models

| Model | Type | Command | Key Parameters |
|-------|------|---------|----------------|
| Linear Regression | Traditional ML | `--model linear` | - |
| Ridge Regression | Traditional ML | `--model ridge` | `--linear_alpha` |
| Lasso Regression | Traditional ML | `--model lasso` | `--linear_alpha` |
| SVM | Traditional ML | `--model svm` | `--svm_kernel`, `--svm_C` |
| MLP | Deep Learning | `--model mlp` | `--mlp_hidden_dims`, `--epochs` |
| CNN | Deep Learning | `--model cnn` | `--cnn_filters`, `--sequence_length` |
| LSTM | Deep Learning | `--model lstm` | `--lstm_hidden_dim`, `--lstm_num_layers` |
| GCN | Graph Neural Network | `--model gcn` | `--gnn_hidden_dims` |
| GAT | Graph Neural Network | `--model gat` | `--gnn_hidden_dims` |

## 💡 Example Commands

### MLP Examples

**Basic MLP:**
```bash
python main.py --model mlp --epochs 100 --batch_size 32
```

**Custom Architecture:**
```bash
python main.py --model mlp \
    --mlp_hidden_dims 128 64 32 \
    --epochs 150 \
    --learning_rate 0.001 \
    --dropout 0.3
```

**Different Configurations:**
```bash
# Small network, more epochs
python main.py --model mlp --mlp_hidden_dims 32 16 --epochs 200

# Large network, higher learning rate
python main.py --model mlp --mlp_hidden_dims 256 128 64 --learning_rate 0.01

# With custom experiment name
python main.py --model mlp --experiment_name "mlp_final_v1" --epochs 100
```

### Other Models

**Linear Models:**
```bash
python main.py --model linear
python main.py --model ridge --linear_alpha 1.0
python main.py --model lasso --linear_alpha 0.5
```

**SVM:**
```bash
python main.py --model svm --svm_kernel rbf
python main.py --model svm --svm_kernel linear --svm_grid_search
```

**CNN:**
```bash
python main.py --model cnn \
    --sequence_length 10 \
    --cnn_filters 32 64 32 \
    --epochs 100
```

**LSTM:**
```bash
python main.py --model lstm \
    --lstm_hidden_dim 64 \
    --lstm_num_layers 2 \
    --sequence_length 10 \
    --epochs 100
```

**GNN:**
```bash
python main.py --model gcn --gnn_hidden_dims 64 32 --epochs 100
python main.py --model gat --gnn_hidden_dims 64 32 --epochs 100
```

## 📈 Output and Results

Each experiment creates a directory with:

```
Results/{experiment_name}_{timestamp}/
├── config.txt                      # All hyperparameters
├── model.pt                        # Saved model (deep learning)
├── evaluation/
│   ├── summary.png                 # Comprehensive results figure
│   ├── predictions.png             # Prediction vs actual plots
│   ├── distributions.png           # Distribution comparison
│   ├── metrics.txt                 # Numerical metrics
│   └── metrics.png                 # Metrics visualization
└── training/
    ├── training_history.png        # Loss curves
    └── feature_importance.png      # Feature importance (if applicable)
```

### Metrics Computed

- **MSE** (Mean Squared Error)
- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **R²** (R-squared coefficient)

## 🔧 Common Parameters

### Data Parameters
- `--data_path`: Path to CSV file
- `--test_size`: Test set proportion (default: 0.2)
- `--random_seed`: Random seed (default: 42)
- `--scale_features`: Scale features (default: True)

### Neural Network Parameters
- `--epochs`: Training epochs (default: 100)
- `--batch_size`: Batch size (default: 32)
- `--learning_rate`: Learning rate (default: 0.001)
- `--dropout`: Dropout rate (default: 0.2)
- `--device`: Device (`auto`, `cpu`, `cuda`)

### Results Parameters
- `--results_dir`: Results directory (default: `Results`)
- `--save_model`: Save trained model (default: True)
- `--plot_results`: Generate plots (default: True)
- `--experiment_name`: Custom name (default: auto-generated)

## 📚 Documentation Files

- **INSTALL_INSTRUCTIONS.md** - Installation guide
- **USAGE_GUIDE.md** - Detailed usage examples
- **QUICK_REFERENCE.md** - Quick command reference
- **Results/README.md** - Results directory structure

## 🎯 Tips for MLP Training

1. **Start with default settings:**
   ```bash
   python main.py --model mlp --epochs 100
   ```

2. **Monitor training:** Progress is printed every 10 epochs

3. **Adjust architecture:** Try different hidden layer configurations
   ```bash
   python main.py --model mlp --mlp_hidden_dims 32 16        # Small
   python main.py --model mlp --mlp_hidden_dims 64 32        # Medium (default)
   python main.py --model mlp --mlp_hidden_dims 128 64 32    # Large
   ```

4. **Tune learning rate:** If loss doesn't decrease, try lower learning rate
   ```bash
   python main.py --model mlp --learning_rate 0.0001
   ```

5. **Increase epochs:** For better convergence
   ```bash
   python main.py --model mlp --epochs 200
   ```

## 🆘 Getting Help

View all available options:
```bash
python main.py --help
```

## 📝 Example Workflow

```bash
# 1. Install dependencies
conda env create -f environment.yml
conda activate quantum_challenge

# 2. Test with simple model
python main.py --model linear

# 3. Train MLP
python main.py --model mlp --epochs 100

# 4. Check results
ls -lh Results/

# 5. View plots and metrics
# Open the PNG files and metrics.txt in the latest experiment folder
```

## 🎓 Model Selection Guide

- **Linear/Ridge/Lasso**: Fastest, good baseline, interpretable
- **SVM**: Good for non-linear relationships, moderate speed
- **MLP**: Flexible, learns complex patterns, requires tuning
- **CNN**: Best for temporal patterns in sequences
- **LSTM**: Best for long-term dependencies in time series
- **GCN/GAT**: Best for graph-structured data with relationships

---

**Ready to train MLP?** Run:
```bash
python main.py --model mlp --epochs 100 --batch_size 32
```
