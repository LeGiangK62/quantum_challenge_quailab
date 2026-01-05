# Quick Reference Card

## Train MLP Model

### Basic Command
```bash
python main.py --model mlp --epochs 100 --batch_size 32
```

### With Custom Architecture
```bash
python main.py --model mlp \
    --mlp_hidden_dims 64 32 \
    --epochs 100 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --dropout 0.2
```

### Using the Shell Script
```bash
./run_examples.sh mlp
```

## All Available Models - One-Liners

```bash
# Linear Models
python main.py --model linear
python main.py --model ridge --linear_alpha 1.0
python main.py --model lasso --linear_alpha 0.5

# SVM
python main.py --model svm --svm_kernel rbf

# Neural Networks
python main.py --model mlp --epochs 100
python main.py --model cnn --sequence_length 10 --epochs 100
python main.py --model lstm --lstm_hidden_dim 64 --epochs 100

# Graph Neural Networks
python main.py --model gcn --epochs 100
python main.py --model gat --epochs 100
```

## Common Parameter Adjustments

### Change Learning Rate
```bash
python main.py --model mlp --learning_rate 0.0001
```

### Increase Epochs
```bash
python main.py --model mlp --epochs 200
```

### Change Batch Size
```bash
python main.py --model mlp --batch_size 64
```

### Adjust Dropout
```bash
python main.py --model mlp --dropout 0.3
```

### Custom Experiment Name
```bash
python main.py --model mlp --experiment_name "my_mlp_experiment"
```

## Output Location

Results are saved in:
```
Results/{experiment_name}_{timestamp}/
```

## View Results

After training, check:
- `Results/{experiment_name}/evaluation/summary.png` - Main results
- `Results/{experiment_name}/evaluation/metrics.txt` - Numerical metrics
- `Results/{experiment_name}/training/training_history.png` - Loss curves

## Help

For all available options:
```bash
python main.py --help
```
