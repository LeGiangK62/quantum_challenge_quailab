import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import os


# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10


def plot_predictions(y_true, y_pred, save_path=None, title='Predictions vs Actual'):
    """
    Plot predicted vs actual values.

    Args:
        y_true: True values
        y_pred: Predicted values
        save_path: Path to save the plot
        title: Plot title
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter plot
    axes[0].scatter(y_true, y_pred, alpha=0.6, edgecolors='k', linewidth=0.5)

    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')

    axes[0].set_xlabel('Actual Values')
    axes[0].set_ylabel('Predicted Values')
    axes[0].set_title(f'{title} - Scatter Plot')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Residual plot
    residuals = y_true - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.6, edgecolors='k', linewidth=0.5)
    axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel('Predicted Values')
    axes[1].set_ylabel('Residuals')
    axes[1].set_title(f'{title} - Residual Plot')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.close()


def plot_training_history(train_losses, val_losses=None, save_path=None, title='Training History'):
    """
    Plot training and validation loss curves.

    Args:
        train_losses: List of training losses
        val_losses: List of validation losses (optional)
        save_path: Path to save the plot
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)

    if val_losses is not None:
        ax.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.close()


def plot_metrics_comparison(metrics_dict, save_path=None, title='Model Performance Metrics'):
    """
    Plot comparison of multiple metrics.

    Args:
        metrics_dict: Dictionary of metric names and values
        save_path: Path to save the plot
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    metrics = list(metrics_dict.keys())
    values = list(metrics_dict.values())

    bars = ax.bar(metrics, values, color='steelblue', alpha=0.7, edgecolor='black')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.4f}',
               ha='center', va='bottom', fontsize=10)

    ax.set_ylabel('Value')
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.close()


def plot_feature_importance(importance_df, top_n=None, save_path=None, title='Feature Importance'):
    """
    Plot feature importance.

    Args:
        importance_df: DataFrame with 'feature' and importance columns
        top_n: Number of top features to show (None for all)
        save_path: Path to save the plot
        title: Plot title
    """
    if top_n is not None:
        importance_df = importance_df.head(top_n)

    fig, ax = plt.subplots(figsize=(10, max(6, len(importance_df) * 0.3)))

    # Sort by absolute coefficient
    if 'abs_coefficient' in importance_df.columns:
        importance_df = importance_df.sort_values('abs_coefficient', ascending=True)
        values = importance_df['coefficient'].values
    else:
        importance_df = importance_df.sort_values(importance_df.columns[1], ascending=True)
        values = importance_df.iloc[:, 1].values

    features = importance_df['feature'].values

    # Color bars based on positive/negative
    colors = ['green' if x > 0 else 'red' for x in values]

    ax.barh(features, values, color=colors, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Coefficient Value')
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='x')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.close()


def plot_distribution_comparison(y_true, y_pred, save_path=None, title='Distribution Comparison'):
    """
    Plot distribution comparison between actual and predicted values.

    Args:
        y_true: True values
        y_pred: Predicted values
        save_path: Path to save the plot
        title: Plot title
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    axes[0].hist(y_true, bins=30, alpha=0.6, label='Actual', color='blue', edgecolor='black')
    axes[0].hist(y_pred, bins=30, alpha=0.6, label='Predicted', color='red', edgecolor='black')
    axes[0].set_xlabel('Value')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'{title} - Histogram')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Box plot
    data_to_plot = [y_true, y_pred]
    axes[1].boxplot(data_to_plot, labels=['Actual', 'Predicted'], patch_artist=True,
                   boxprops=dict(facecolor='lightblue', alpha=0.7),
                   medianprops=dict(color='red', linewidth=2))
    axes[1].set_ylabel('Value')
    axes[1].set_title(f'{title} - Box Plot')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.close()


def create_results_summary(metrics, y_true, y_pred, save_path=None):
    """
    Create a comprehensive results summary figure.

    Args:
        metrics: Dictionary of evaluation metrics
        y_true: True values
        y_pred: Predicted values
        save_path: Path to save the plot
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Scatter plot (top-left, larger)
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    ax1.scatter(y_true, y_pred, alpha=0.6, edgecolors='k', linewidth=0.5)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    ax1.set_xlabel('Actual Values', fontsize=12)
    ax1.set_ylabel('Predicted Values', fontsize=12)
    ax1.set_title('Predictions vs Actual', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Residual plot (top-right)
    ax2 = fig.add_subplot(gs[0, 2])
    residuals = y_true - y_pred
    ax2.scatter(y_pred, residuals, alpha=0.6, edgecolors='k', linewidth=0.5)
    ax2.axhline(y=0, color='r', linestyle='--', lw=2)
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Residuals')
    ax2.set_title('Residual Plot')
    ax2.grid(True, alpha=0.3)

    # 3. Metrics bar plot (middle-right)
    ax3 = fig.add_subplot(gs[1, 2])
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())
    bars = ax3.barh(metric_names, metric_values, color='steelblue', alpha=0.7, edgecolor='black')
    for i, (bar, val) in enumerate(zip(bars, metric_values)):
        ax3.text(val, bar.get_y() + bar.get_height()/2, f'{val:.4f}',
                ha='left', va='center', fontsize=9, fontweight='bold')
    ax3.set_xlabel('Value')
    ax3.set_title('Performance Metrics')
    ax3.grid(True, alpha=0.3, axis='x')

    # 4. Distribution comparison (bottom-left)
    ax4 = fig.add_subplot(gs[2, 0:2])
    ax4.hist(y_true, bins=30, alpha=0.6, label='Actual', color='blue', edgecolor='black')
    ax4.hist(y_pred, bins=30, alpha=0.6, label='Predicted', color='red', edgecolor='black')
    ax4.set_xlabel('Value')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Distribution Comparison')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. Error distribution (bottom-right)
    ax5 = fig.add_subplot(gs[2, 2])
    ax5.hist(residuals, bins=30, alpha=0.7, color='purple', edgecolor='black')
    ax5.axvline(x=0, color='r', linestyle='--', lw=2, label='Zero Error')
    ax5.set_xlabel('Residual')
    ax5.set_ylabel('Frequency')
    ax5.set_title('Error Distribution')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Summary plot saved to {save_path}")

    plt.close()


def plot_time_series_predictions(df_with_predictions, save_path=None,
                                  n_patients=5, title='Time Series Predictions'):
    """
    Plot predictions vs actual values over time for individual patients.

    Args:
        df_with_predictions: DataFrame with columns ['ID', 'TIME', 'DV', 'predictions', 'DVID']
        save_path: Path to save the plot
        n_patients: Number of patients to plot
        title: Plot title
    """
    # Get unique patient IDs
    unique_patients = df_with_predictions['ID'].unique()[:n_patients]

    # Calculate grid dimensions
    n_cols = min(2, len(unique_patients))
    n_rows = (len(unique_patients) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, patient_id in enumerate(unique_patients):
        ax = axes[idx]

        # Get patient data
        patient_data = df_with_predictions[df_with_predictions['ID'] == patient_id].sort_values('TIME')

        # Plot actual values
        ax.plot(patient_data['TIME'], patient_data['DV'],
               'o-', label='Actual', markersize=6, linewidth=2, alpha=0.7)

        # Plot predictions
        ax.plot(patient_data['TIME'], patient_data['predictions'],
               's--', label='Predicted', markersize=6, linewidth=2, alpha=0.7)

        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Concentration/Effect')
        ax.set_title(f'Patient {int(patient_id)}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(len(unique_patients), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Time series plot saved to {save_path}")

    plt.close()


def plot_pk_pd_separately(df_with_predictions, save_path=None,
                          n_patients=3, title='PK and PD Predictions'):
    """
    Plot PK (DVID=1) and PD (DVID=2) predictions separately.

    Args:
        df_with_predictions: DataFrame with columns ['ID', 'TIME', 'DV', 'predictions', 'DVID']
        save_path: Path to save the plot
        n_patients: Number of patients to plot
        title: Plot title
    """
    if 'DVID' not in df_with_predictions.columns:
        print("Warning: DVID column not found. Skipping PK/PD separation.")
        return

    # Get unique patient IDs
    unique_patients = df_with_predictions['ID'].unique()[:n_patients]

    fig, axes = plt.subplots(n_patients, 2, figsize=(16, 4*n_patients))
    if n_patients == 1:
        axes = axes.reshape(1, -1)

    for idx, patient_id in enumerate(unique_patients):
        # Get patient data
        patient_data = df_with_predictions[df_with_predictions['ID'] == patient_id].sort_values('TIME')

        # PK data (DVID = 1)
        pk_data = patient_data[patient_data['DVID'] == 1]
        if len(pk_data) > 0:
            axes[idx, 0].plot(pk_data['TIME'], pk_data['DV'],
                            'o-', label='Actual PK', markersize=6, linewidth=2,
                            color='blue', alpha=0.7)
            axes[idx, 0].plot(pk_data['TIME'], pk_data['predictions'],
                            's--', label='Predicted PK', markersize=6, linewidth=2,
                            color='red', alpha=0.7)
            axes[idx, 0].set_xlabel('Time (hours)')
            axes[idx, 0].set_ylabel('Drug Concentration (PK)')
            axes[idx, 0].set_title(f'Patient {int(patient_id)} - Pharmacokinetics')
            axes[idx, 0].legend()
            axes[idx, 0].grid(True, alpha=0.3)
        else:
            axes[idx, 0].text(0.5, 0.5, 'No PK data', ha='center', va='center')
            axes[idx, 0].set_title(f'Patient {int(patient_id)} - No PK Data')

        # PD data (DVID = 2)
        pd_data = patient_data[patient_data['DVID'] == 2]
        if len(pd_data) > 0:
            axes[idx, 1].plot(pd_data['TIME'], pd_data['DV'],
                            'o-', label='Actual PD', markersize=6, linewidth=2,
                            color='green', alpha=0.7)
            axes[idx, 1].plot(pd_data['TIME'], pd_data['predictions'],
                            's--', label='Predicted PD', markersize=6, linewidth=2,
                            color='orange', alpha=0.7)
            axes[idx, 1].set_xlabel('Time (hours)')
            axes[idx, 1].set_ylabel('Drug Effect (PD)')
            axes[idx, 1].set_title(f'Patient {int(patient_id)} - Pharmacodynamics')
            axes[idx, 1].legend()
            axes[idx, 1].grid(True, alpha=0.3)
        else:
            axes[idx, 1].text(0.5, 0.5, 'No PD data', ha='center', va='center')
            axes[idx, 1].set_title(f'Patient {int(patient_id)} - No PD Data')

    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"PK/PD plot saved to {save_path}")

    plt.close()


def plot_pk_pd_comparison(df_with_predictions, save_path=None):
    """
    Plot overall PK vs PD predictions comparison.

    Args:
        df_with_predictions: DataFrame with predictions
        save_path: Path to save the plot
    """
    if 'DVID' not in df_with_predictions.columns:
        print("Warning: DVID column not found. Skipping PK/PD comparison.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # PK data
    pk_data = df_with_predictions[df_with_predictions['DVID'] == 1]
    # PD data
    pd_data = df_with_predictions[df_with_predictions['DVID'] == 2]

    if len(pk_data) > 0:
        # PK scatter
        axes[0, 0].scatter(pk_data['DV'], pk_data['predictions'],
                          alpha=0.6, edgecolors='k', linewidth=0.5, color='blue')
        min_val = min(pk_data['DV'].min(), pk_data['predictions'].min())
        max_val = max(pk_data['DV'].max(), pk_data['predictions'].max())
        axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual PK (Concentration)')
        axes[0, 0].set_ylabel('Predicted PK (Concentration)')
        axes[0, 0].set_title('Pharmacokinetics (PK) Predictions')
        axes[0, 0].grid(True, alpha=0.3)

        # PK residuals
        pk_residuals = pk_data['DV'] - pk_data['predictions']
        axes[1, 0].scatter(pk_data['predictions'], pk_residuals,
                          alpha=0.6, edgecolors='k', linewidth=0.5, color='blue')
        axes[1, 0].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[1, 0].set_xlabel('Predicted PK')
        axes[1, 0].set_ylabel('Residuals')
        axes[1, 0].set_title('PK Residuals')
        axes[1, 0].grid(True, alpha=0.3)

    if len(pd_data) > 0:
        # PD scatter
        axes[0, 1].scatter(pd_data['DV'], pd_data['predictions'],
                          alpha=0.6, edgecolors='k', linewidth=0.5, color='green')
        min_val = min(pd_data['DV'].min(), pd_data['predictions'].min())
        max_val = max(pd_data['DV'].max(), pd_data['predictions'].max())
        axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        axes[0, 1].set_xlabel('Actual PD (Effect)')
        axes[0, 1].set_ylabel('Predicted PD (Effect)')
        axes[0, 1].set_title('Pharmacodynamics (PD) Predictions')
        axes[0, 1].grid(True, alpha=0.3)

        # PD residuals
        pd_residuals = pd_data['DV'] - pd_data['predictions']
        axes[1, 1].scatter(pd_data['predictions'], pd_residuals,
                          alpha=0.6, edgecolors='k', linewidth=0.5, color='green')
        axes[1, 1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[1, 1].set_xlabel('Predicted PD')
        axes[1, 1].set_ylabel('Residuals')
        axes[1, 1].set_title('PD Residuals')
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"PK/PD comparison plot saved to {save_path}")

    plt.close()


def save_metrics_to_file(metrics, save_path):
    """
    Save metrics to a text file.

    Args:
        metrics: Dictionary of metrics
        save_path: Path to save the file
    """
    with open(save_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("Evaluation Metrics\n")
        f.write("="*60 + "\n\n")

        for metric, value in metrics.items():
            f.write(f"{metric.upper():.<30} {value:.6f}\n")

        f.write("\n" + "="*60 + "\n")

    print(f"Metrics saved to {save_path}")


def create_experiment_report(experiment_name, args, metrics, y_true, y_pred,
                            train_losses=None, val_losses=None,
                            feature_importance=None, results_dir='Results',
                            test_data_with_predictions=None):
    """
    Create a comprehensive experiment report with all plots and metrics.

    Args:
        experiment_name: Name of the experiment
        args: Experiment arguments
        metrics: Evaluation metrics
        y_true: True values
        y_pred: Predicted values
        train_losses: Training losses (optional)
        val_losses: Validation losses (optional)
        feature_importance: Feature importance DataFrame (optional)
        results_dir: Base results directory
        test_data_with_predictions: DataFrame with test data and predictions including TIME, ID, DVID (optional)
    """
    # Create experiment directory structure
    exp_dir = os.path.join(results_dir, experiment_name)
    os.makedirs(exp_dir, exist_ok=True)

    # Create subdirectories
    eval_dir = os.path.join(exp_dir, 'evaluation')
    train_dir = os.path.join(exp_dir, 'training')
    timeseries_dir = os.path.join(exp_dir, 'timeseries')
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(timeseries_dir, exist_ok=True)

    print(f"\nCreating experiment report in: {exp_dir}")

    # 1. Save comprehensive summary
    summary_path = os.path.join(eval_dir, 'summary.png')
    create_results_summary(metrics, y_true, y_pred, save_path=summary_path)

    # 2. Save predictions plot
    pred_path = os.path.join(eval_dir, 'predictions.png')
    plot_predictions(y_true, y_pred, save_path=pred_path)

    # 3. Save distribution comparison
    dist_path = os.path.join(eval_dir, 'distributions.png')
    plot_distribution_comparison(y_true, y_pred, save_path=dist_path)

    # 4. Save metrics
    metrics_path = os.path.join(eval_dir, 'metrics.txt')
    save_metrics_to_file(metrics, metrics_path)

    # 5. Save metrics plot
    metrics_plot_path = os.path.join(eval_dir, 'metrics.png')
    plot_metrics_comparison(metrics, save_path=metrics_plot_path)

    # 6. Save training history if available
    if train_losses is not None:
        history_path = os.path.join(train_dir, 'training_history.png')
        plot_training_history(train_losses, val_losses, save_path=history_path)

    # 7. Save feature importance if available
    if feature_importance is not None:
        importance_path = os.path.join(train_dir, 'feature_importance.png')
        plot_feature_importance(feature_importance, top_n=15, save_path=importance_path)

    # 8. Time series plots if test data is available
    if test_data_with_predictions is not None:
        # General time series for all patients
        ts_path = os.path.join(timeseries_dir, 'time_series_predictions.png')
        plot_time_series_predictions(test_data_with_predictions, save_path=ts_path, n_patients=5)

        # PK/PD separated plots
        pk_pd_path = os.path.join(timeseries_dir, 'pk_pd_separate.png')
        plot_pk_pd_separately(test_data_with_predictions, save_path=pk_pd_path, n_patients=3)

        # PK/PD comparison
        pk_pd_comp_path = os.path.join(timeseries_dir, 'pk_pd_comparison.png')
        plot_pk_pd_comparison(test_data_with_predictions, save_path=pk_pd_comp_path)

        # Save predictions to CSV
        csv_path = os.path.join(exp_dir, 'predictions.csv')
        test_data_with_predictions.to_csv(csv_path, index=False)
        print(f"Predictions saved to CSV: {csv_path}")

    # 9. Save configuration
    config_path = os.path.join(exp_dir, 'config.txt')
    with open(config_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write(f"Experiment: {experiment_name}\n")
        f.write("="*60 + "\n\n")

        f.write("Configuration:\n")
        f.write("-"*60 + "\n")
        for arg in sorted(vars(args)):
            f.write(f"{arg:.<30} {getattr(args, arg)}\n")

        f.write("\n" + "="*60 + "\n")

    print(f"\nExperiment report completed!")
    print(f"Results saved in: {exp_dir}")
    print(f"  - Evaluation plots: {eval_dir}")
    print(f"  - Training plots: {train_dir}")
    if test_data_with_predictions is not None:
        print(f"  - Time series plots: {timeseries_dir}")
    print(f"  - Configuration: {config_path}")

    return exp_dir
