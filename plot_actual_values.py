#!/usr/bin/env python3
"""
Plot actual DV values from the CSV file.
Shows PK and PD values separately over time for individual patients.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
sns.set_style('whitegrid')
plt.rcParams['font.size'] = 10

def plot_actual_values(csv_path='Data/QIC2025-EstDat.csv',
                       n_patients=3,
                       save_path='actual_values_plot.png'):
    """
    Plot actual PK and PD values separately over time for selected patients.

    Args:
        csv_path: Path to CSV file
        n_patients: Number of patients to plot
        save_path: Path to save the plot
    """
    # Load data
    df = pd.read_csv(csv_path)

    # Filter to observations only (EVID=0)
    df = df[df['EVID'] == 0].copy()

    # Filter out missing values if MDV column exists
    if 'MDV' in df.columns:
        df = df[df['MDV'] == 0]

    # Get unique patient IDs
    unique_patients = df['ID'].unique()[:n_patients]

    print(f"Plotting actual values for {len(unique_patients)} patients")
    print(f"Patients: {unique_patients}")

    # Check if DVID column exists
    has_dvid = 'DVID' in df.columns

    if has_dvid:
        # Create subplots: 2 columns (PK and PD) x n_patients rows
        fig, axes = plt.subplots(n_patients, 2, figsize=(14, 4*n_patients))
        if n_patients == 1:
            axes = axes.reshape(1, -1)

        for idx, patient_id in enumerate(unique_patients):
            # Get patient data
            patient_data = df[df['ID'] == patient_id].sort_values('TIME')

            # Get metadata
            bw = patient_data['BW'].iloc[0]
            dose = patient_data['DOSE'].iloc[0]

            # PK data (DVID = 1)
            pk_data = patient_data[patient_data['DVID'] == 1]
            if len(pk_data) > 0:
                axes[idx, 0].plot(pk_data['TIME'], pk_data['DV'],
                                'o-', markersize=6, linewidth=2,
                                alpha=0.7, color='blue')
                axes[idx, 0].set_xlabel('Time (hours)')
                axes[idx, 0].set_ylabel('Drug Concentration (PK)')
                axes[idx, 0].set_title(f'ID: {int(patient_id)} | DOSE: {int(dose)} | BW: {int(bw)} - Pharmacokinetics')
                axes[idx, 0].grid(True, alpha=0.3)
            else:
                axes[idx, 0].text(0.5, 0.5, 'No PK data', ha='center', va='center',
                                transform=axes[idx, 0].transAxes)
                axes[idx, 0].set_title(f'ID: {int(patient_id)} - No PK Data')

            # PD data (DVID = 2)
            pd_data = patient_data[patient_data['DVID'] == 2]
            if len(pd_data) > 0:
                axes[idx, 1].plot(pd_data['TIME'], pd_data['DV'],
                                'o-', markersize=6, linewidth=2,
                                alpha=0.7, color='green')
                axes[idx, 1].set_xlabel('Time (hours)')
                axes[idx, 1].set_ylabel('Drug Effect (PD)')
                axes[idx, 1].set_title(f'ID: {int(patient_id)} | DOSE: {int(dose)} | BW: {int(bw)} - Pharmacodynamics')
                axes[idx, 1].grid(True, alpha=0.3)
            else:
                axes[idx, 1].text(0.5, 0.5, 'No PD data', ha='center', va='center',
                                transform=axes[idx, 1].transAxes)
                axes[idx, 1].set_title(f'ID: {int(patient_id)} - No PD Data')

        plt.suptitle('Actual PK and PD Values Over Time', fontsize=16, fontweight='bold', y=1.00)

    else:
        # No DVID column, plot all values together
        n_cols = min(3, len(unique_patients))
        n_rows = (len(unique_patients) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
        if len(unique_patients) == 1:
            axes = np.array([axes])
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()

        for idx, patient_id in enumerate(unique_patients):
            ax = axes[idx]

            # Get patient data
            patient_data = df[df['ID'] == patient_id].sort_values('TIME')

            # Get metadata
            bw = patient_data['BW'].iloc[0]
            dose = patient_data['DOSE'].iloc[0]

            # Plot actual values
            ax.plot(patient_data['TIME'], patient_data['DV'],
                   'o-', markersize=6, linewidth=2,
                   alpha=0.7, color='blue')

            ax.set_xlabel('Time (hours)')
            ax.set_ylabel('DV Value')
            ax.set_title(f'ID: {int(patient_id)} | DOSE: {int(dose)} | BW: {int(bw)}')
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(len(unique_patients), len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle('Actual Values Over Time', fontsize=14, fontweight='bold', y=1.00)

    plt.tight_layout()

    # Save plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {save_path}")

    plt.show()

    # Print summary statistics
    print("\n=== Summary Statistics ===")
    print(f"Total observations: {len(df)}")
    print(f"Number of patients: {len(df['ID'].unique())}")
    print(f"Time range: {df['TIME'].min()} - {df['TIME'].max()} hours")
    print(f"DV range: {df['DV'].min():.2f} - {df['DV'].max():.2f}")

    if has_dvid:
        pk_count = len(df[df['DVID'] == 1])
        pd_count = len(df[df['DVID'] == 2])
        print(f"PK observations (DVID=1): {pk_count}")
        print(f"PD observations (DVID=2): {pd_count}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Plot actual values from CSV file')
    parser.add_argument('--csv_path', type=str, default='Data/QIC2025-EstDat.csv',
                       help='Path to CSV file')
    parser.add_argument('--n_patients', type=int, default=3,
                       help='Number of patients to plot')
    parser.add_argument('--save_path', type=str, default='actual_values_plot.png',
                       help='Path to save the plot')

    args = parser.parse_args()

    plot_actual_values(
        csv_path=args.csv_path,
        n_patients=args.n_patients,
        save_path=args.save_path
    )
