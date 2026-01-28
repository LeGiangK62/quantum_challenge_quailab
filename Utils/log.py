import logging
import numpy as np
import torch
import matplotlib.pyplot as plt
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='|%(levelname)s - %(message)s')
# logging.basicConfig(level=logging.INFO, format='\t - %(levelname)s - %(message)s')
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_metrics(y_true, y_pred):
    """
    Calculates MSE, RMSE, MAE, and R2.
    Args:
        y_true: Ground truth (Tensor or numpy array)
        y_pred: Predictions (Tensor or numpy array)
    Returns:
        dict: Dictionary of metrics
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
        
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    if ss_tot < 1e-8:
        r2 = 0.0
    else:
        r2 = 1 - (ss_res / ss_tot)
        
    return {
        "MSE": float(mse),
        "RMSE": float(rmse),
        "MAE": float(mae),
        "R2": float(r2)
    }

def log_metrics(epoch, stage, metrics):
    metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
    logger.info(f"Epoch {epoch} [{stage}] | {metrics_str}")

def plot_metrics(history, save_path="metrics_plot.png"):
    """
    Plots training and evaluation metrics (MSE, RMSE, MAE, R2) in subplots.
    Separates PK and PD into different figures.

    Args:
        history (dict): Dictionary containing lists of metric values per epoch.
                        Keys should be in format "{Stage}_{Metric}" (e.g., "Train PK_MSE").
        save_path (str): Base path to save the figures. Suffixes _PK and _PD will be added.
    """
    metrics = ["MSE", "RMSE", "MAE", "R2"]
    domains = ["PK", "PD"]
    epochs = history.get("Epoch", None)

    if save_path:
        base_path, ext = os.path.splitext(save_path)

    for domain in domains:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        fig.suptitle(f"{domain} Metrics")

        for i, metric in enumerate(metrics):
            ax = axes[i]
            # Filter keys for this domain and metric
            relevant_keys = [k for k in history.keys() if domain in k and k.endswith(metric)]
            
            # Exclude RMSE from MSE plots
            if metric == "MSE":
                relevant_keys = [k for k in relevant_keys if not k.endswith("RMSE")]

            for key in relevant_keys:
                # Clean label: remove metric suffix and domain name
                label = key.replace(f"_{metric}", "").replace(domain, "").strip()
                if epochs and len(epochs) == len(history[key]):
                    ax.plot(epochs, history[key], label=label)
                else:
                    ax.plot(history[key], label=label)

            ax.set_title(metric)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(metric)
            ax.legend()
            ax.grid(True)

        plt.tight_layout()
        if save_path:
            plt.savefig(f"{base_path}_{domain}{ext}")
        plt.close()