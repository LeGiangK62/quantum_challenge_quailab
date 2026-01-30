import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader as TorchDataLoader
from Utils.log import calculate_metrics, log_metrics, plot_metrics, logger


# ============================================================
# Visualization
# ============================================================
def plot_predictions(train_results, test_results, save_dir, model_name):
    """Plot scatter plots and save. Also saves raw data for reproducibility."""
    os.makedirs(save_dir, exist_ok=True)

    # Save raw data for reproducibility
    np.savez(
        os.path.join(save_dir, 'predictions_scatter_data.npz'),
        train_pk_targets=train_results['pk']['targets'],
        train_pk_predictions=train_results['pk']['predictions'],
        train_pk_R2=train_results['pk']['R2'],
        train_pk_RMSE=train_results['pk']['RMSE'],
        test_pk_targets=test_results['pk']['targets'],
        test_pk_predictions=test_results['pk']['predictions'],
        test_pk_R2=test_results['pk']['R2'],
        test_pk_RMSE=test_results['pk']['RMSE'],
        train_pd_targets=train_results['pd']['targets'],
        train_pd_predictions=train_results['pd']['predictions'],
        train_pd_R2=train_results['pd']['R2'],
        train_pd_RMSE=train_results['pd']['RMSE'],
        test_pd_targets=test_results['pd']['targets'],
        test_pd_predictions=test_results['pd']['predictions'],
        test_pd_R2=test_results['pd']['R2'],
        test_pd_RMSE=test_results['pd']['RMSE'],
    )
    logger.info(f"Saved predictions scatter data to {save_dir}/predictions_scatter_data.npz")

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # PK - Train
    axes[0, 0].scatter(train_results['pk']['targets'], train_results['pk']['predictions'],
                       alpha=0.6, edgecolors='k', linewidth=0.5)
    min_val = min(train_results['pk']['targets'].min(), train_results['pk']['predictions'].min())
    max_val = max(train_results['pk']['targets'].max(), train_results['pk']['predictions'].max())
    axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual PK')
    axes[0, 0].set_ylabel('Predicted PK')
    axes[0, 0].set_title(f'PK Train (R2={train_results["pk"]["R2"]:.4f}, RMSE={train_results["pk"]["RMSE"]:.4f})')
    axes[0, 0].grid(True, alpha=0.3)

    # PK - Test
    axes[0, 1].scatter(test_results['pk']['targets'], test_results['pk']['predictions'],
                       alpha=0.6, edgecolors='k', linewidth=0.5, color='orange')
    min_val = min(test_results['pk']['targets'].min(), test_results['pk']['predictions'].min())
    max_val = max(test_results['pk']['targets'].max(), test_results['pk']['predictions'].max())
    axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    axes[0, 1].set_xlabel('Actual PK')
    axes[0, 1].set_ylabel('Predicted PK')
    axes[0, 1].set_title(f'PK Test (R2={test_results["pk"]["R2"]:.4f}, RMSE={test_results["pk"]["RMSE"]:.4f})')
    axes[0, 1].grid(True, alpha=0.3)

    # PD - Train
    axes[1, 0].scatter(train_results['pd']['targets'], train_results['pd']['predictions'],
                       alpha=0.6, edgecolors='k', linewidth=0.5)
    min_val = min(train_results['pd']['targets'].min(), train_results['pd']['predictions'].min())
    max_val = max(train_results['pd']['targets'].max(), train_results['pd']['predictions'].max())
    axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    axes[1, 0].set_xlabel('Actual PD')
    axes[1, 0].set_ylabel('Predicted PD')
    axes[1, 0].set_title(f'PD Train (R2={train_results["pd"]["R2"]:.4f}, RMSE={train_results["pd"]["RMSE"]:.4f})')
    axes[1, 0].grid(True, alpha=0.3)

    # PD - Test
    axes[1, 1].scatter(test_results['pd']['targets'], test_results['pd']['predictions'],
                       alpha=0.6, edgecolors='k', linewidth=0.5, color='orange')
    min_val = min(test_results['pd']['targets'].min(), test_results['pd']['predictions'].min())
    max_val = max(test_results['pd']['targets'].max(), test_results['pd']['predictions'].max())
    axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    axes[1, 1].set_xlabel('Actual PD')
    axes[1, 1].set_ylabel('Predicted PD')
    axes[1, 1].set_title(f'PD Test (R2={test_results["pd"]["R2"]:.4f}, RMSE={test_results["pd"]["RMSE"]:.4f})')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'predictions_scatter.png'), dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved prediction plots to {save_dir}")


def plot_patient_timeseries(pk_data, pd_data, model, device, save_dir, model_name,
                            patient_ids=None):
    """
    Plot time series for selected patients.
    Also saves raw data for reproducibility.

    Args:
        pk_data: dict with 'X', 'y', 'ids', 'times' for PK
        pd_data: dict with 'X', 'y', 'ids', 'times' for PD
        model: trained model
        device: torch device
        save_dir: directory to save plots
        model_name: name for plot title
        patient_ids: list of patient IDs to plot (default: [9, 13, 26, 46])
    """
    os.makedirs(save_dir, exist_ok=True)

    if patient_ids is None:
        # Default: No dose (9), Dose 1 (13), Dose 3 (26), Dose 10 (46)
        patient_ids = [9, 13, 26, 46]

    # Filter to available patients (only need PD data, PK can be empty)
    available_pk_ids = set(pk_data['ids'])
    available_pd_ids = set(pd_data['ids'])
    patient_ids = [pid for pid in patient_ids if pid in available_pd_ids]

    if len(patient_ids) == 0:
        logger.warning("No specified patients found in data. Using first 4 available.")
        patient_ids = list(available_pd_ids)[:4]

    n_patients = len(patient_ids)
    if n_patients == 0:
        logger.warning("No patients available for plotting")
        return

    # Create 4x2 figure (patients x [PK, PD])
    fig, axes = plt.subplots(n_patients, 2, figsize=(14, 4 * n_patients))
    if n_patients == 1:
        axes = axes.reshape(1, -1)

    model.eval()

    # Collect data for saving
    timeseries_data = {}

    for idx, patient_id in enumerate(patient_ids):
        # Get PK data for this patient
        pk_mask = pk_data['ids'] == patient_id
        pk_X = pk_data['X'][pk_mask]
        pk_y = pk_data['y'][pk_mask]
        pk_times = pk_data['times'][pk_mask]

        # Get PD data for this patient
        pd_mask = pd_data['ids'] == patient_id
        pd_X = pd_data['X'][pd_mask]
        pd_y = pd_data['y'][pd_mask]
        pd_times = pd_data['times'][pd_mask]

        # Plot PK (first column)
        ax_pk = axes[idx, 0]

        if len(pk_X) > 0:
            # Sort by time
            pk_order = np.argsort(pk_times)
            pk_times = pk_times[pk_order]
            pk_y = pk_y[pk_order]
            pk_X = pk_X[pk_order]

            # Get PK predictions
            with torch.no_grad():
                pk_X_tensor = torch.FloatTensor(pk_X).to(device)
                pk_results = model(pk_X_tensor, None)
                pk_pred = pk_results['pk'].cpu().numpy().flatten()

            # Store data for saving
            timeseries_data[f'patient_{patient_id}_pk_times'] = pk_times
            timeseries_data[f'patient_{patient_id}_pk_actual'] = pk_y
            timeseries_data[f'patient_{patient_id}_pk_predicted'] = pk_pred

            ax_pk.plot(pk_times, pk_y, 'o-', label='Actual PK', markersize=6, linewidth=2, color='blue')
            ax_pk.plot(pk_times, pk_pred, 's--', label='Predicted PK', markersize=6, linewidth=2, color='orange')
            ax_pk.set_xlabel('Time (hours)')
            ax_pk.set_ylabel('PK Value')
            ax_pk.legend()
        else:
            # No PK data - leave blank with message
            ax_pk.text(0.5, 0.5, 'No PK data\n(No dose)', transform=ax_pk.transAxes,
                      ha='center', va='center', fontsize=14, color='gray')
            ax_pk.set_xlabel('Time (hours)')
            ax_pk.set_ylabel('PK Value')

        ax_pk.set_title(f'Patient {patient_id} - PK')
        ax_pk.grid(True, alpha=0.3)

        # Plot PD (second column)
        ax_pd = axes[idx, 1]

        if len(pd_X) > 0:
            # Sort by time
            pd_order = np.argsort(pd_times)
            pd_times = pd_times[pd_order]
            pd_y = pd_y[pd_order]
            pd_X = pd_X[pd_order]

            # Get PD predictions
            with torch.no_grad():
                pd_X_tensor = torch.FloatTensor(pd_X).to(device)
                pd_results = model(None, pd_X_tensor)
                pd_pred = pd_results['pd'].cpu().numpy().flatten()

            # Store data for saving
            timeseries_data[f'patient_{patient_id}_pd_times'] = pd_times
            timeseries_data[f'patient_{patient_id}_pd_actual'] = pd_y
            timeseries_data[f'patient_{patient_id}_pd_predicted'] = pd_pred

            ax_pd.plot(pd_times, pd_y, 'o-', label='Actual PD', markersize=6, linewidth=2, color='blue')
            ax_pd.plot(pd_times, pd_pred, 's--', label='Predicted PD', markersize=6, linewidth=2, color='orange')
            ax_pd.legend()
        else:
            ax_pd.text(0.5, 0.5, 'No PD data', transform=ax_pd.transAxes,
                      ha='center', va='center', fontsize=14, color='gray')

        ax_pd.set_xlabel('Time (hours)')
        ax_pd.set_ylabel('PD Value')
        ax_pd.set_title(f'Patient {patient_id} - PD')
        ax_pd.grid(True, alpha=0.3)

    # Save timeseries data for reproducibility
    timeseries_data['patient_ids'] = np.array(patient_ids)
    np.savez(os.path.join(save_dir, 'patient_timeseries_data.npz'), **timeseries_data)
    logger.info(f"Saved patient timeseries data to {save_dir}/patient_timeseries_data.npz")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'patient_timeseries.png'), dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved patient time series plots to {save_dir}")


def plot_gnn_patient_timeseries(all_data, model, device, save_dir, model_name, patient_ids=None):
    """
    Plot time series for selected patients (GNN version).
    Also saves raw data for reproducibility.
    """
    os.makedirs(save_dir, exist_ok=True)

    if patient_ids is None:
        patient_ids = [9, 13, 26, 46]

    # Filter graphs and sort by patient ID
    graphs_to_plot = []
    for g in all_data:
        pid = g.patient_id
        if torch.is_tensor(pid):
            pid = pid.item()
        if pid in patient_ids:
            graphs_to_plot.append((pid, g))

    # Sort by patient ID
    graphs_to_plot.sort(key=lambda x: x[0])
    graphs_to_plot = [g for _, g in graphs_to_plot]

    if not graphs_to_plot:
        logger.warning("No specified patients found in GNN data for plotting.")
        return

    n_patients = len(graphs_to_plot)
    fig, axes = plt.subplots(n_patients, 2, figsize=(14, 4 * n_patients))
    if n_patients == 1:
        axes = axes.reshape(1, -1)

    model.eval()

    # Collect data for saving
    timeseries_data = {}
    plotted_patient_ids = []

    for idx, graph in enumerate(graphs_to_plot):
        pid = graph.patient_id
        if torch.is_tensor(pid):
            pid = pid.item()
        plotted_patient_ids.append(pid)

        batch = graph.to(device)

        with torch.no_grad():
            pd_pred, pk_pred = model(batch, return_pk=True)

        # PK Plot
        ax_pk = axes[idx, 0]
        pk_mask = batch.pk_mask.cpu().numpy().astype(bool)

        if pk_mask.any():
            pk_times = batch.times[pk_mask].cpu().numpy()
            pk_y = batch.pk_targets[pk_mask].cpu().numpy()
            pk_p = pk_pred[pk_mask].cpu().numpy().flatten()

            sort_idx = np.argsort(pk_times)

            # Store data for saving
            timeseries_data[f'patient_{pid}_pk_times'] = pk_times[sort_idx]
            timeseries_data[f'patient_{pid}_pk_actual'] = pk_y[sort_idx]
            timeseries_data[f'patient_{pid}_pk_predicted'] = pk_p[sort_idx]

            ax_pk.plot(pk_times[sort_idx], pk_y[sort_idx], 'o-', label='Actual PK', color='blue')
            ax_pk.plot(pk_times[sort_idx], pk_p[sort_idx], 's--', label='Predicted PK', color='orange')
            ax_pk.legend()
        else:
            ax_pk.text(0.5, 0.5, 'No PK data\n(No dose)', transform=ax_pk.transAxes,
                      ha='center', va='center', fontsize=14, color='gray')

        ax_pk.set_title(f'Patient {pid} - PK')
        ax_pk.set_xlabel('Time (hours)')
        ax_pk.set_ylabel('PK Value')
        ax_pk.grid(True, alpha=0.3)

        # PD Plot
        ax_pd = axes[idx, 1]
        pd_mask = batch.pd_mask.cpu().numpy().astype(bool)

        if pd_mask.any():
            pd_times = batch.times[pd_mask].cpu().numpy()
            pd_y = batch.pd_targets[pd_mask].cpu().numpy()
            pd_p = pd_pred[pd_mask].cpu().numpy().flatten()

            sort_idx = np.argsort(pd_times)

            # Store data for saving
            timeseries_data[f'patient_{pid}_pd_times'] = pd_times[sort_idx]
            timeseries_data[f'patient_{pid}_pd_actual'] = pd_y[sort_idx]
            timeseries_data[f'patient_{pid}_pd_predicted'] = pd_p[sort_idx]

            ax_pd.plot(pd_times[sort_idx], pd_y[sort_idx], 'o-', label='Actual PD', color='blue')
            ax_pd.plot(pd_times[sort_idx], pd_p[sort_idx], 's--', label='Predicted PD', color='orange')
            ax_pd.legend()
        else:
            ax_pd.text(0.5, 0.5, 'No PD data', transform=ax_pd.transAxes,
                      ha='center', va='center', fontsize=14, color='gray')

        ax_pd.set_title(f'Patient {pid} - PD')
        ax_pd.set_xlabel('Time (hours)')
        ax_pd.set_ylabel('PD Value')
        ax_pd.grid(True, alpha=0.3)

    # Save timeseries data for reproducibility
    timeseries_data['patient_ids'] = np.array(plotted_patient_ids)
    np.savez(os.path.join(save_dir, 'patient_timeseries_data.npz'), **timeseries_data)
    logger.info(f"Saved GNN patient timeseries data to {save_dir}/patient_timeseries_data.npz")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'patient_timeseries.png'), dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved GNN patient time series plots to {save_dir}")