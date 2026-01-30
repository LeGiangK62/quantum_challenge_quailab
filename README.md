# Hierarchical PK/PD Prediction

Repository for Quantum Challenge 2025 - Team PNU of QUAILAB



## Team PNU
### Supervisor
* **Won-Joo Hwang** ([wjhwang@pusan.ac.kr](mailto:wjhwang@pusan.ac.kr))

### Team Members
| Name | Email|
| :--- | :--- |
| **Seon Geun Jeong** | wjdtjsrms11@pusan.ac.kr |
| **Le Tung Giang**   | giang.lt2399144@pusan.ac.kr |
| **Nguyen Doan Hieu**| hieu.nguyendoan@pusan.ac.kr |
| **Mai Dinh Cong**| Cong.md204521@gmail.com |



## Models

| Model   | Description |
|---------|-------------|
| `mlp`   | Hierarchical MLP with ResBlocks |
| `gnn`   | Hierarchical Graph Neural Network (GCN/GAT) |
| `lstm`  | Hierarchical Bidirectional LSTM |
| `hqcnn` | Hierarchical Quantum CNN |
| `hqgnn` | Hierarchical Quantum GNN |
| `hqlstm`| Hierarchical Quantum LSTM |

## Arguments

### Model Architecture
```
--model         Model type: mlp, gnn, lstm, hqcnn, hqgnn, hqlstm (default: mlp)
--mode          Training mode: separate, joint, dual_stage, shared (default: dual_stage)
```

### MLP Hyperparameters
```
--hidden_dim    Hidden dimension (default: 256)
--n_blocks      Number of ResBlocks (default: 4)
--head_hidden   Hidden dimension for prediction heads (default: 128)
```

### GNN Hyperparameters
```
--gnn_hidden_dim    Hidden dimension (default: 64)
--num_layers_pk     Number of GNN layers for PK encoder (default: 3)
--num_layers_pd     Number of GNN layers for PD decoder (default: 3)
--use_attention     Use GAT instead of GCN (default: False)
--use_gating        Use gating mechanism in PD decoder (default: True)
```

### LSTM Hyperparameters
```
--lstm_hidden_dim       Hidden dimension (default: 128)
--lstm_num_layers       Number of LSTM layers (default: 2)
--lstm_bidirectional    Use bidirectional LSTM (default: True)
```

### Quantum Hyperparameters
```
--hqcnn_num_layers    Number of quantum layers (default: 1)
```

### Data
```
--csv_path          Path to data CSV file (default: Data/QIC2025-EstDat.csv)
--test_size         Test set fraction (default: 0.1)
--val_size          Validation set fraction (default: 0.1)
--random_seed       Random seed (default: 1712)
--stratified_split  Use dose-stratified splitting (default: True)
--use_perkg         Add per-kg normalized features (default: False)
```

### Feature Engineering
```
--time_windows    Time windows for dose history in hours (default: [24,48,72,96,120,144,168])
--half_lives      Half-lives for decay features in hours (default: [24,48,72])
--add_decay       Add exponential decay features (default: True)
```

### Training
```
--epochs          Number of training epochs (default: 300)
--batch_size      Batch size (default: 16)
--learning_rate   Learning rate (default: 0.001)
```

---

## Sample Commands

### Training

```bash
# MLP 
python main.py --model mlp --epochs 300 --

# GNN
python main.py --model gnn --epochs 300 --gnn_hidden_dim 64 

# GNN with attention (GAT)
python main.py --model gnn --use_attention --epochs 300 

# LSTM
python main.py --model lstm --epochs 300 --lstm_hidden_dim 128 

# HQCNN (Quantum MLP)
python main.py --model hqcnn --epochs 300 --hqcnn_num_layers 1 

# HQGNN (Quantum GNN)
python main.py --model hqgnn --epochs 300 --gnn_hidden_dim 64 

# HQLSTM (Quantum LSTM)
python main.py --model hqlstm --epochs 300 --lstm_hidden_dim 128 
```


---

## Dose Prediction

After training, use `dose_prediction.py` to find optimal doses.

### Arguments
```
--model         Model type: mlp, gnn, lstm, hqcnn, hqgnn, hqlstm
--model_path    Path to model checkpoint (optional, auto-finds latest)
--results_dir   Directory containing results (default: Results)
```

### Sample Commands

```bash
# MLP dose prediction
python dose_prediction.py --model mlp

# GNN dose prediction
python dose_prediction.py --model gnn

# LSTM dose prediction
python dose_prediction.py --model lstm

# HQCNN dose prediction
python dose_prediction.py --model hqcnn

# HQGNN dose prediction
python dose_prediction.py --model hqgnn

# HQLSTM dose prediction
python dose_prediction.py --model hqlstm
```

---

## Project Structure

```
.
├── main.py                 # Training entry point
├── dose_prediction.py      # Dose optimization
├── Data/
│   └── QIC2025-EstDat.csv  # Dataset
├── Models/
│   ├── mlp.py              # MLP architecture
│   ├── gnn.py              # GNN architecture
│   ├── lstm.py             # LSTM architecture
│   └── quantum.py          # Quantum models (HQCNN, HQGNN, HQLSTM)
├── Utils/
│   ├── args.py             # Argument parsing
│   ├── data_loader.py      # Data loading and preprocessing
│   ├── training.py         # Training loops
│   └── log.py              # Logging utilities
└── Results/                # Saved models and outputs
```

---
# Contact

QUAILAB

Le Tung Giang - giang.lt2399144@pusan.ac.kr