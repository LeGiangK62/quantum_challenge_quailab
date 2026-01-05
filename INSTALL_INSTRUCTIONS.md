# Installation Instructions

This guide will help you set up the environment for the PK/PD prediction project.

## Option 1: Using Conda Environment (Recommended)

The conda environment will automatically install all dependencies including PyTorch Geometric with the correct CPU wheels.

```bash
# Create the conda environment
conda env create -f environment.yml

# Activate the environment
conda activate quantum_challenge
```

## Option 2: Using pip with requirements.txt

If you're using pip directly (without conda), you need to install PyTorch Geometric dependencies in the correct order:

```bash
# 1. First, install PyTorch (if not already installed)
pip install torch==2.0.1 torchvision>=0.15.0

# 2. Install PyG dependencies with CPU wheels
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cpu.html

# 3. Install all other dependencies
pip install -r requirements.txt
```

**Important**: The order matters! Install torch-scatter and torch-sparse from the PyG wheel repository **before** installing other dependencies.

## Verification

After installation, verify that all packages are installed correctly:

```python
import torch
import torch_geometric
import torch_scatter
import torch_sparse
import pennylane
import sklearn
import pandas
import numpy

print("All packages imported successfully!")
print(f"PyTorch version: {torch.__version__}")
print(f"PyG version: {torch_geometric.__version__}")
```

## Troubleshooting

### Issue: torch-scatter installation fails

**Solution**: Make sure you're using the correct wheel URL for your PyTorch version:
```bash
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
```

### Issue: CUDA vs CPU mismatch

This project uses **CPU-only** versions. If you have CUDA installed, make sure to:
1. Use the `+cpu` wheel URLs for PyG packages
2. Install `cpuonly` package in conda (already in environment.yml)

### Issue: diffrax installation fails

diffrax requires JAX. Install it separately if needed:
```bash
pip install "jax[cpu]"
pip install diffrax
```

## Quick Test

Run a quick test of the preprocessing pipeline:

```bash
cd Utils
python pre_processing.py
```

This should load the data and display basic statistics without errors.
