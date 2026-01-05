# Time Series Visualization - Why It Works Differently for Each Model

## The Problem

Time series plots require mapping predictions back to specific **time points** and **patients**. However, different models make predictions at different granularities:

## Model Types and Their Prediction Granularity

### ✅ Works Perfectly: Tabular Models (Linear, Ridge, Lasso, SVM, MLP)

**How they work:**
- Each input row = one observation at one time point
- Each prediction = one time point
- Direct 1:1 mapping between predictions and time points

**Example:**
```
Input Row:   [BW=58, COMED=0, DOSE=0, TIME=72, ID=1] → Prediction: 18.12
                                                     ↓
                                        Time point 72h for Patient 1
```

**Why time series works:**
- We know exactly which time point and patient each prediction belongs to
- Can directly plot prediction vs time for each patient
- **Current Status: ✅ WORKING**

---

### ⚠️ Complex: Sequence Models (CNN, LSTM)

**How they work:**
- Input = **sequence** of time points (e.g., 10 consecutive observations)
- Output = **one prediction** for the entire sequence
- Mapping is **one prediction to many time points**

**Example:**
```
Input Sequence (10 time points):
[
  [58, 0, 0, 0],     # t=0
  [58, 0, 0, 1],     # t=1
  [58, 0, 0, 2],     # t=2
  ...
  [58, 0, 0, 72]     # t=72
]
         ↓
Prediction: 18.12 (for t=72, the last time point)
```

**The Challenge:**
1. Each prediction corresponds to a **sequence**, not a single time point
2. Sequences may **overlap** (sliding window with step_size < sequence_length)
3. Need to decide: Does the prediction represent:
   - The last time point? ✓ (most common)
   - The average of all time points?
   - Some specific time point?

**Current Status: ⚠️ DISABLED**
- Time series plots are currently disabled for CNN/LSTM
- Reason: Requires careful mapping of sequences to time points
- TODO: Implement proper sequence-to-timepoint mapping

---

### ⚠️ Complex: Graph Neural Networks (GCN, GAT)

**How they work:**
- Input = **graph** with nodes (observations) and edges (connections)
- Each node = one observation = one time point
- Predictions are at node level
- BUT: Train/test split is at **node level**, not at **patient level**

**Example:**
```
Graph for all patients:
Node 0 (P1, t=0)  →  Node 1 (P1, t=1)  →  Node 2 (P1, t=2)
Node 3 (P2, t=0)  →  Node 4 (P2, t=1)  →  Node 5 (P2, t=2)
...

Test mask: [Node 1, Node 4, Node 5, ...]  # Random nodes selected
Predictions: [18.1, 10.3, 11.5, ...]       # One per test node
```

**The Challenge:**
1. Test nodes are **randomly scattered** across patients and time points
2. Not all time points for a patient may be in the test set
3. Need to track which original time points were used for testing

**Current Status: ⚠️ PARTIALLY WORKING**
- Basic implementation added
- Uses approximate index matching
- May not be 100% accurate due to random node sampling

---

## Why the Difference Matters

### Tabular Models
```
Data flow:
Original Data → Split by rows → Test indices preserved → Easy mapping
Row 100 (P1, t=72) → Test → Prediction → Row 100 (P1, t=72, pred=18.1)
                                       ↓
                              Perfect time series plot!
```

### Sequence Models
```
Data flow:
Original Data → Create sequences → Split sequences → Lost row indices!
Rows 0-9 → Sequence 1 → Test → Prediction 1 → Which row does this represent?
                                              ↓
                                    Need to reconstruct mapping
```

### Graph Models
```
Data flow:
Original Data → Build graph → Random node mask → Node indices != Row indices
Row 100 → Node 50 → Test → Prediction → Need to map Node 50 back to Row 100
                                       ↓
                            Complex index tracking required
```

## Solutions Implemented

### ✅ Tabular Models (Working)
```python
# Direct index matching
test_indices = y_test.index
test_data = original_df.loc[test_indices]
test_data['predictions'] = y_pred
# ✅ Perfect mapping!
```

### ⚠️ Sequence Models (Disabled)
```python
# Currently disabled with informative message
if args.model in ['cnn', 'lstm']:
    print("Note: Time series plots not available for sequence models")
    test_data_with_predictions = None
```

**What needs to be done:**
1. Track which original rows went into each sequence
2. Map predictions back to the **last time point** of each sequence
3. Handle overlapping sequences properly

### ⚠️ Graph Models (Basic Implementation)
```python
# Approximate matching by length
test_size = len(y_pred)
sampled_indices = original_df.index[-test_size:]  # Approximate!
test_data = original_df.loc[sampled_indices]
test_data['predictions'] = y_pred
```

**Limitations:**
- Assumes test nodes are from the end of the dataset
- May not match exactly if nodes are randomly selected
- Works reasonably well but not perfect

**What needs to be done:**
1. Track exact test node indices during GNN training
2. Map node indices back to original DataFrame rows
3. Return proper index mapping from train_gnn_model()

## Summary Table

| Model Type | Time Series Support | Reason |
|------------|-------------------|--------|
| Linear/Ridge/Lasso | ✅ Full support | Direct row-to-prediction mapping |
| SVM | ✅ Full support | Direct row-to-prediction mapping |
| MLP | ✅ Full support | Direct row-to-prediction mapping |
| CNN | ❌ Disabled | Sequence-level predictions, mapping complex |
| LSTM | ❌ Disabled | Sequence-level predictions, mapping complex |
| GCN | ⚠️ Basic support | Node-level predictions, approximate mapping |
| GAT | ⚠️ Basic support | Node-level predictions, approximate mapping |

## Workarounds for CNN/LSTM

If you want to see temporal predictions for CNN/LSTM models:

### Option 1: Use Non-Overlapping Sequences
```python
# Set step_size = sequence_length
python main.py --model lstm \
    --sequence_length 10 \
    --step_size 10  # No overlap
```
This makes mapping clearer (each prediction to exactly one sequence).

### Option 2: Export Predictions and Create Custom Plots
```python
# After training, load predictions
import pandas as pd
import matplotlib.pyplot as plt

# Manually reconstruct time series from sequence predictions
# (Requires tracking which sequences correspond to which patients)
```

### Option 3: Use MLP Instead
```python
# For time series analysis, use MLP on tabular data
python main.py --model mlp --epochs 100
```
MLP still captures temporal patterns through the TIME feature.

## Future Improvements

### For CNN/LSTM:
1. Modify training functions to return:
   - Original row indices for each sequence
   - Mapping from sequence index to time point
2. Implement proper sequence-to-timepoint mapping
3. Handle overlapping sequences appropriately

### For GNN:
1. Return test node indices from `train_gnn_model()`
2. Create proper node-to-row index mapping
3. Track exact test nodes instead of approximating

### General:
1. Add a `metadata` return value from all training functions
2. Include index mappings in metadata
3. Use metadata for accurate time series reconstruction

## Recommendation

**For now, if you need time series analysis:**
1. ✅ Use **MLP, SVM, or Linear models** (fully supported)
2. ⚠️ Use **GCN/GAT** with caution (approximate mapping)
3. ❌ Avoid expecting time series plots from **CNN/LSTM** (disabled)

**The core metrics (MSE, RMSE, MAE, R²) work perfectly for all models** - only the time series visualization is affected!
