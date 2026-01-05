# Quantum-Enhanced PK/PD Environment Setup

## Quick Start with Conda (Recommended)

### Option 1: Create environment from YAML file
```bash
# Create the conda environment
conda env create -f environment.yml

# Activate the environment
conda activate quantum_challenge

# Verify installation
python -c "import pennylane as qml; print(f'PennyLane version: {qml.__version__}')"
```

### Option 2: Manual setup
```bash
# Create new conda environment with Python 3.10
conda create -n quantum_challenge python=3.10 -y

# Activate environment
conda activate quantum_challenge

# Install packages from conda-forge
conda install -c conda-forge numpy scipy pandas scikit-learn matplotlib seaborn jupyter -y

# Install PennyLane and other pip packages
pip install -r requirements.txt
```

## Package Overview

### Quantum Computing Stack
- **PennyLane**: Main quantum ML framework
- **PennyLane-Lightning**: Fast CPU-based quantum simulator (no GPU needed)
- **PennyLane-Torch**: Integration layer for PyTorch + PennyLane

### Deep Learning
- **PyTorch**: Deep learning framework (CPU version)
- **TorchVision**: Image utilities (useful for data augmentation)

### Classical ML & Scientific Computing
- **NumPy/SciPy**: Numerical computing
- **Pandas**: Data manipulation
- **Scikit-learn**: Classical ML models (for comparison/hybrid approaches)

### PK/PD Specific
- **Diffrax**: Modern ODE solver for compartmental models
- **Optuna**: Hyperparameter optimization

### Visualization
- **Matplotlib/Seaborn**: Plotting concentration-time curves, dose-response

## Why No GPU?

For this project, CPU is sufficient because:
1. **Dataset size**: 48 subjects, ~2800 observations is small
2. **PennyLane-Lightning**: Highly optimized CPU simulator
3. **Quantum circuits**: Small-scale variational circuits run fast on CPU
4. **PK/PD ODEs**: Solved efficiently with CPU-based integrators

## Optional GPU Setup (if needed later)

If you want GPU acceleration for larger experiments:
```bash
# For NVIDIA GPUs
pip install pennylane-lightning-gpu

# Or use Qiskit GPU simulator
pip install pennylane-qiskit qiskit-aer-gpu
```

## Verify Installation

```bash
# Test PennyLane
python -c "import pennylane as qml; dev = qml.device('lightning.qubit', wires=2); print('PennyLane OK')"

# Test all imports
python -c "import numpy, scipy, pandas, sklearn, matplotlib, torch, pennylane; print('All packages OK')"

# Test PyTorch + PennyLane integration
python -c "import torch; import pennylane as qml; print(f'PyTorch {torch.__version__}, PennyLane {qml.__version__}')"
```

## Environment Management

```bash
# List conda environments
conda env list

# Deactivate environment
conda deactivate

# Remove environment (if needed)
conda env remove -n quantum_challenge

# Update environment from YAML
conda env update -f environment.yml --prune
```

## Troubleshooting

### Issue: PennyLane installation fails
```bash
pip install --upgrade pip
pip install pennylane --no-cache-dir
```

### Issue: Import errors
```bash
# Reinstall in editable mode if developing custom modules
pip install -e .
```

## Next Steps

1. Activate environment: `conda activate quantum_challenge`
2. Run exploratory analysis: `python main.py --explore`
3. Train quantum model: `python main.py --model quantum --train`
