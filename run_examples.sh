#!/bin/bash

# Example training scripts for PK/PD prediction models
# Make this file executable with: chmod +x run_examples.sh

echo "PK/PD Model Training Examples"
echo "=============================="
echo ""

# Function to run a model and wait
run_model() {
    echo "Running: $1"
    echo "Command: $2"
    echo "----------------------------------------"
    eval $2
    echo ""
    echo "Completed: $1"
    echo "========================================"
    echo ""
}

# Check if specific model is requested
if [ $# -eq 0 ]; then
    echo "Usage: ./run_examples.sh [model_name]"
    echo ""
    echo "Available models:"
    echo "  linear  - Linear Regression (OLS)"
    echo "  ridge   - Ridge Regression"
    echo "  lasso   - Lasso Regression"
    echo "  svm     - Support Vector Machine"
    echo "  mlp     - Multi-Layer Perceptron"
    echo "  cnn     - Convolutional Neural Network"
    echo "  lstm    - Long Short-Term Memory"
    echo "  gcn     - Graph Convolutional Network"
    echo "  gat     - Graph Attention Network"
    echo "  all     - Run all models (sequential)"
    echo ""
    echo "Example: ./run_examples.sh mlp"
    exit 1
fi

MODEL=$1

case $MODEL in
    linear)
        run_model "Linear Regression" "python main.py --model linear"
        ;;

    ridge)
        run_model "Ridge Regression" "python main.py --model ridge --linear_alpha 1.0"
        ;;

    lasso)
        run_model "Lasso Regression" "python main.py --model lasso --linear_alpha 1.0"
        ;;

    svm)
        run_model "SVM with RBF kernel" "python main.py --model svm --svm_kernel rbf --svm_C 1.0"
        ;;

    mlp)
        run_model "Multi-Layer Perceptron" "python main.py --model mlp --mlp_hidden_dims 64 32 --epochs 100 --batch_size 32 --learning_rate 0.001"
        ;;

    cnn)
        run_model "CNN" "python main.py --model cnn --cnn_filters 32 64 32 --sequence_length 10 --step_size 5 --epochs 100"
        ;;

    lstm)
        run_model "LSTM" "python main.py --model lstm --lstm_hidden_dim 64 --lstm_num_layers 2 --sequence_length 10 --step_size 5 --epochs 100"
        ;;

    gcn)
        run_model "Graph Convolutional Network" "python main.py --model gcn --gnn_hidden_dims 64 32 --epochs 100"
        ;;

    gat)
        run_model "Graph Attention Network" "python main.py --model gat --gnn_hidden_dims 64 32 --epochs 100"
        ;;

    all)
        echo "Running all models sequentially..."
        echo ""

        run_model "Linear Regression" "python main.py --model linear"
        run_model "Ridge Regression" "python main.py --model ridge"
        run_model "Lasso Regression" "python main.py --model lasso"
        run_model "SVM" "python main.py --model svm"
        run_model "MLP" "python main.py --model mlp --epochs 100"
        run_model "CNN" "python main.py --model cnn --epochs 100"
        run_model "LSTM" "python main.py --model lstm --epochs 100"
        run_model "GCN" "python main.py --model gcn --epochs 100"
        run_model "GAT" "python main.py --model gat --epochs 100"

        echo "All models completed!"
        ;;

    *)
        echo "Unknown model: $MODEL"
        echo "Run './run_examples.sh' without arguments to see available models."
        exit 1
        ;;
esac

echo ""
echo "Results saved in: Results/"
echo "Check the latest experiment directory for plots and metrics."
