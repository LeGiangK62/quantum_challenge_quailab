# Hierarchical GNN for PK-PD Prediction - Usage Guide

## Overview

This hierarchical (two-stage) GNN architecture predicts PD values using a novel approach:
1. **Stage 1 (PK-GNN)**: Predicts PK values from patient covariates
2. **Stage 2 (PD-GNN)**: Uses predicted PK values + embeddings to predict PD
3. **Residual Connections**: Learnable residual branch for improved predictions

## Key Features

### 1. Two Training Modes

#### **Joint Training** (Recommended, Default)
- Trains both PK-GNN and PD-GNN simultaneously end-to-end
- Multi-task loss: `Loss = λ_PK * Loss_PK + λ_PD * Loss_PD`
- Better gradient flow and feature learning
- Default weights: PK=0.3, PD=1.0

```bash
python train_pd_hierarchical_gnn.py --training_mode joint --epochs 150
```

#### **Sequential Training**
- Stage 1: Train PK-GNN for 100 epochs
- Freeze PK-GNN weights
- Stage 2: Train PD-GNN for 100 epochs with frozen PK encoder

```bash
python train_pd_hierarchical_gnn.py --training_mode sequential \
    --epochs_pk 100 --epochs_pd 100
```

### 2. Enhanced Feature Engineering (36+ Features)

Automatically applied features include:
- **Time transformations**: log, sqrt, squared
- **Rate of change**: ΔPK/Δt (derivative modeling)
- **Cumulative dose**: Total drug accumulation over time
- **PK-PD lag**: Time difference between PK and PD measurements
- **Temporal patterns**: sin/cos for circadian rhythms
- **Dose normalization**: Per-kg dosing
- **Interaction features**: TIME×DOSE, BW×DOSE, COMED×TIME

### 3. Patient Selection Features

#### **Evaluate Specific Patients**
Compute metrics for specific patient IDs:
```bash
python train_pd_hierarchical_gnn.py --eval_patient_ids 1 5 10 25
```

#### **Plot Specific Patients**
Generate time-series plots for specific patients:
```bash
python train_pd_hierarchical_gnn.py --patient_ids 2 7 15 30
```

#### **Random Patient Selection**
Randomly select patients for visualization:
```bash
python train_pd_hierarchical_gnn.py --random_patients --n_patients 5 --random_seed 42
```

#### **Default Behavior**
Plots first N patients (default: 3):
```bash
python train_pd_hierarchical_gnn.py --n_patients 5
```

### 4. Advanced Architecture Options

#### **Attention Mechanism**
Use Graph Attention Networks (GAT) instead of GCN:
```bash
python train_pd_hierarchical_gnn.py --use_attention
```

#### **Gating Control**
Enable gating to control PK information flow to PD-GNN (default: enabled):
```bash
python train_pd_hierarchical_gnn.py --use_gating
```

#### **Custom Architecture**
```bash
python train_pd_hierarchical_gnn.py \
    --hidden_dim 128 \
    --num_layers_pk 4 \
    --num_layers_pd 3 \
    --dropout 0.3
```

### 5. Graph Construction Enhancements

- **Edge weights**: Temporal distance-based (exponential decay)
- **Multi-hop PK-PD connections**: Each PD node connects to 3 recent PK nodes
- **Bidirectional edges**: Better information flow
- **Temporal attention**: Time-weighted connections

## Complete Example Commands

### Example 1: Quick Test with Joint Training
```bash
python train_pd_hierarchical_gnn.py \
    --training_mode joint \
    --epochs 150 \
    --patient_ids 1 5 10
```

### Example 2: Sequential Training with Evaluation
```bash
python train_pd_hierarchical_gnn.py \
    --training_mode sequential \
    --epochs_pk 100 \
    --epochs_pd 100 \
    --eval_patient_ids 2 7 15 \
    --patient_ids 2 7 15
```

### Example 3: Advanced Configuration with Attention
```bash
python train_pd_hierarchical_gnn.py \
    --training_mode joint \
    --use_attention \
    --use_gating \
    --hidden_dim 128 \
    --num_layers_pk 4 \
    --num_layers_pd 4 \
    --dropout 0.25 \
    --epochs 200 \
    --learning_rate 0.0005 \
    --pk_loss_weight 0.4 \
    --pd_loss_weight 1.0 \
    --random_patients \
    --n_patients 10
```

### Example 4: Production Run with GPU
```bash
python train_pd_hierarchical_gnn.py \
    --training_mode joint \
    --device cuda \
    --epochs 300 \
    --batch_size 16 \
    --hidden_dim 256 \
    --save_dir Results/Production_Run
```

## Command-Line Arguments Reference

### Training Mode
- `--training_mode {sequential,joint}`: Training strategy (default: joint)

### Architecture
- `--hidden_dim INT`: Hidden dimension size (default: 64)
- `--num_layers_pk INT`: Number of GNN layers in PK encoder (default: 3)
- `--num_layers_pd INT`: Number of GNN layers in PD decoder (default: 3)
- `--dropout FLOAT`: Dropout rate (default: 0.2)
- `--use_attention`: Use GAT instead of GCN
- `--use_gating`: Enable gating mechanism (default: True)

### Sequential Training
- `--learning_rate_pk FLOAT`: Learning rate for PK-GNN (default: 0.001)
- `--learning_rate_pd FLOAT`: Learning rate for PD-GNN (default: 0.001)
- `--epochs_pk INT`: Epochs for PK-GNN training (default: 100)
- `--epochs_pd INT`: Epochs for PD-GNN training (default: 100)

### Joint Training
- `--learning_rate FLOAT`: Learning rate (default: 0.001)
- `--epochs INT`: Total training epochs (default: 150)
- `--pk_loss_weight FLOAT`: Weight for PK loss (default: 0.3)
- `--pd_loss_weight FLOAT`: Weight for PD loss (default: 1.0)

### Data & Evaluation
- `--csv_path PATH`: Path to data CSV (default: 'Data/QIC2025-EstDat.csv')
- `--test_size FLOAT`: Test set proportion (default: 0.2)
- `--batch_size INT`: Batch size (default: 8)
- `--random_seed INT`: Random seed (default: 1712)

### Patient Selection
- `--n_patients INT`: Number of patients to plot (default: 3)
- `--patient_ids INT [INT ...]`: Specific patient IDs to plot
- `--random_patients`: Randomly select patients for plotting
- `--eval_patient_ids INT [INT ...]`: Patient IDs for separate evaluation

### Output
- `--save_dir PATH`: Directory for results (default: 'Results/PD_Hierarchical_GNN')
- `--device {cpu,cuda}`: Device to use (default: 'cpu')

## Output Files

The script generates the following outputs in the save directory:

1. **training_history.png**: Loss curves during training
2. **predictions_scatter.png**: Scatter plots for PK and PD predictions
3. **timeseries_pd.png**: Time-series plots for selected patients
4. **metrics.txt**: Detailed evaluation metrics

## Model Architecture Details

### PK-GNN Encoder
```
Input Features → GNN Layers (with LayerNorm & Residual) → PK Predictions + Embeddings
```

### PD-GNN Decoder
```
[Original Features + PK Embeddings + Predicted PK]
    → Gating Mechanism
    → GNN Layers (with LayerNorm & Residual)
    → Main PD Prediction

Residual Branch: [Combined Features] → Linear → Residual Correction

Final PD = Main PD + α * Residual
```

### Learnable Components
- PK embeddings from Stage 1
- Gating weights for information flow control
- Residual weight α (learned during training)
- All GNN layer parameters

## Tips for Best Performance

1. **Start with joint training**: Usually gives better results than sequential
2. **Tune loss weights**: Adjust `--pk_loss_weight` and `--pd_loss_weight` based on relative importance
3. **Use attention for complex patterns**: `--use_attention` if your data has complex temporal dependencies
4. **Increase model capacity**: Try `--hidden_dim 128` or `--num_layers_pd 4` for better PD predictions
5. **Patient-specific evaluation**: Use `--eval_patient_ids` to check performance on specific patient cohorts
6. **Random seed**: Set `--random_seed` for reproducibility

## Troubleshooting

### Poor PK predictions?
- Increase `--pk_loss_weight` (try 0.5 or 0.7)
- Add more PK-GNN layers: `--num_layers_pk 4`
- Increase training epochs: `--epochs 200`

### Poor PD predictions?
- Ensure PK predictions are good first
- Try attention: `--use_attention`
- Increase PD-GNN depth: `--num_layers_pd 4`
- Adjust `--pd_loss_weight` to 1.5

### Overfitting?
- Increase dropout: `--dropout 0.3`
- Reduce model size: `--hidden_dim 32`
- Add more training data or use data augmentation

### Memory issues?
- Reduce batch size: `--batch_size 4`
- Reduce hidden dimension: `--hidden_dim 32`
- Use fewer layers: `--num_layers_pk 2 --num_layers_pd 2`
