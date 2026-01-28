"""
Data loader and preprocessing utilities for PK/PD prediction.
Based on the successful old code architecture with stratified dose-based splitting.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler


# =========================================================
# Data Loading
# =========================================================
def load_data(path: str = "Data/QIC2025-EstDat.csv") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load PK/PD data and separate into observations and dosing events.

    Returns:
        df_all: Full dataset
        df_obs: Observation records (EVID==0 & MDV==0)
        df_dose: Dosing records (EVID==1)
    """
    df = pd.read_csv(path)
    df.columns = [c.strip().upper() for c in df.columns]

    df_obs = df[(df['EVID'] == 0) & (df['MDV'] == 0)].copy()
    df_dose = df[df['EVID'] == 1].copy()

    print(f"Loaded data: {len(df)} total records")
    print(f"  Observations: {len(df_obs)}")
    print(f"  Dosing events: {len(df_dose)}")

    return df, df_obs, df_dose


# =========================================================
# Feature Engineering (from old successful code)
# =========================================================
def engineer_dose_features(
    df_obs: pd.DataFrame,
    df_dose: pd.DataFrame,
    time_windows: list = None,
    half_lives: list = None,
    add_decay: bool = True
) -> pd.DataFrame:
    """
    Apply leakage-safe feature engineering based on dose history.

    Features added:
    - TSLD: Time Since Last Dose
    - LAST_DOSE_AMT, LAST_DOSE_TIME
    - N_DOSES_UP_TO_T: Number of doses received
    - CUM_DOSE_UP_TO_T: Cumulative dose
    - DOSE_SUM_PREV{window}H: Dose sum in time windows
    - DECAY_HL{hl}h: Exponential decay features
    - TIME_SQUARED, TIME_LOG: Time transformations
    """
    if time_windows is None:
        time_windows = [24, 48, 72, 96, 120, 144, 168]
    if half_lives is None:
        half_lives = [24, 48, 72]

    df = df_obs.copy()

    # Initialize features
    df['TSLD'] = np.nan
    df['LAST_DOSE_AMT'] = 0.0
    df['LAST_DOSE_TIME'] = np.nan
    df['N_DOSES_UP_TO_T'] = 0
    df['CUM_DOSE_UP_TO_T'] = 0.0
    df['TIME_SQUARED'] = df['TIME'] ** 2
    df['TIME_LOG'] = np.log1p(df['TIME'])

    # Initialize window features
    for window in time_windows:
        df[f'DOSE_SUM_PREV{window}H'] = 0.0

    # Initialize decay features
    if add_decay:
        for hl in half_lives:
            df[f'DECAY_HL{hl}h'] = 0.0

    # Pre-group doses per subject
    dose_by_id = {}
    for subject_id, subject_doses in df_dose.groupby('ID'):
        dose_data = subject_doses.sort_values('TIME')[['TIME', 'AMT']].values
        dose_by_id[subject_id] = dose_data

    # Process each subject
    for subject_id in df['ID'].unique():
        subject_mask = df['ID'] == subject_id
        subject_times = df.loc[subject_mask, 'TIME'].values
        subject_indices = df[subject_mask].index

        if subject_id not in dose_by_id:
            continue

        dose_data = dose_by_id[subject_id]
        dose_times = dose_data[:, 0]
        dose_amounts = dose_data[:, 1]
        cumsum_doses = np.cumsum(dose_amounts)

        for idx, curr_time in zip(subject_indices, subject_times):
            # Find last dose before current time
            past_mask = dose_times <= curr_time
            if not past_mask.any():
                continue

            last_idx = np.where(past_mask)[0][-1]
            last_time = dose_times[last_idx]
            last_amt = dose_amounts[last_idx]
            tsld = curr_time - last_time
            n_doses = np.sum(past_mask)
            cum_dose = cumsum_doses[last_idx]

            df.loc[idx, 'TSLD'] = tsld
            df.loc[idx, 'LAST_DOSE_AMT'] = last_amt
            df.loc[idx, 'LAST_DOSE_TIME'] = last_time
            df.loc[idx, 'N_DOSES_UP_TO_T'] = n_doses
            df.loc[idx, 'CUM_DOSE_UP_TO_T'] = cum_dose

            # Decay features
            if add_decay and not np.isnan(tsld):
                for hl in half_lives:
                    k = np.log(2) / hl
                    df.loc[idx, f'DECAY_HL{hl}h'] = np.exp(-k * tsld)

            # Window sums
            for window in time_windows:
                window_mask = (dose_times < curr_time) & (dose_times >= curr_time - window)
                if window_mask.any():
                    df.loc[idx, f'DOSE_SUM_PREV{window}H'] = dose_amounts[window_mask].sum()

    # Fill NaN
    df.fillna(0, inplace=True)

    print(f"Feature engineering complete:")
    print(f"  Time windows: {time_windows}")
    if add_decay:
        print(f"  Decay half-lives: {half_lives}")

    return df


def build_feature_list(
    time_windows: list = None,
    half_lives: list = None,
    add_decay: bool = True,
    use_perkg: bool = False
) -> List[str]:
    """Build the feature list matching old code."""
    if time_windows is None:
        time_windows = [24, 48, 72, 96, 120, 144, 168]
    if half_lives is None:
        half_lives = [24, 48, 72]

    base_features = [
        'BW', 'COMED', 'DOSE', 'TIME',
        'TSLD', 'LAST_DOSE_TIME', 'LAST_DOSE_AMT',
        'N_DOSES_UP_TO_T', 'CUM_DOSE_UP_TO_T',
        'TIME_SQUARED', 'TIME_LOG'
    ]

    # Window features
    window_features = [f'DOSE_SUM_PREV{w}H' for w in time_windows]
    base_features.extend(window_features)

    # Decay features
    if add_decay:
        decay_features = [f'DECAY_HL{hl}h' for hl in half_lives]
        base_features.extend(decay_features)

    # Per-kg features
    if use_perkg:
        perkg_features = [
            'DOSE_PER_KG', 'LAST_DOSE_AMT_PER_KG',
            'CUM_DOSE_UP_TO_T_PER_KG'
        ]
        perkg_features.extend([f'DOSE_SUM_PREV{w}H_PER_KG' for w in time_windows])
        base_features.extend(perkg_features)

    return base_features


def add_perkg_features(df: pd.DataFrame, time_windows: list = None) -> pd.DataFrame:
    """Add per-kg normalized dose features."""
    if time_windows is None:
        time_windows = [24, 48, 72, 96, 120, 144, 168]

    df = df.copy()
    bw = df['BW'].replace(0, np.nan)

    df['DOSE_PER_KG'] = (df['DOSE'] / bw).fillna(0.0)
    df['LAST_DOSE_AMT_PER_KG'] = (df['LAST_DOSE_AMT'] / bw).fillna(0.0)
    df['CUM_DOSE_UP_TO_T_PER_KG'] = (df['CUM_DOSE_UP_TO_T'] / bw).fillna(0.0)

    for window in time_windows:
        col = f'DOSE_SUM_PREV{window}H'
        if col in df.columns:
            df[f'{col}_PER_KG'] = (df[col] / bw).fillna(0.0)

    return df


# =========================================================
# Dose-Stratified Splitting (CRITICAL for old code performance)
# =========================================================
def get_subject_primary_dose(df_dose: pd.DataFrame) -> pd.Series:
    """
    Calculate representative dose per subject.
    - If subject has DOSE==0 -> 0 (placebo)
    - Else use mode of non-zero doses
    """
    def per_subject(doses):
        doses = doses.values
        if (doses == 0).any():
            return 0.0
        nz = doses[doses > 0]
        if nz.size:
            unique, counts = np.unique(nz, return_counts=True)
            return float(unique[np.argmax(counts)])
        return 0.0

    return df_dose.groupby('ID')['AMT'].apply(per_subject)


def stratified_dose_split(
    df: pd.DataFrame,
    df_dose: pd.DataFrame,
    test_size: float = 0.1,
    val_size: float = 0.1,
    random_state: int = 42,
    n_bins: int = 4,
    combine: bool = True
) -> Dict[str, np.ndarray]:
    """
    Stratified splitting by dose levels (stratify_dose_even strategy).

    This ensures balanced dose distribution across train/val/test sets.
    CRITICAL: This was the key to old code's success!
    """
    # Get subject-level primary dose
    subject_dose = get_subject_primary_dose(df_dose)
    all_subjects = np.array(sorted(df['ID'].unique()))

    # Create dose bins
    subject_dose = subject_dose.reindex(all_subjects, fill_value=0.0)
    zero_mask = subject_dose == 0.0

    bins = pd.Series(index=subject_dose.index, dtype=object)
    bins[zero_mask] = 'placebo'

    # Bin non-zero doses
    nonzero = subject_dose[~zero_mask]
    if nonzero.size:
        n_bins_actual = min(n_bins, nonzero.nunique())
        try:
            qbins = pd.qcut(nonzero, q=n_bins_actual, duplicates='drop')
            bins.loc[nonzero.index] = qbins.astype(str)
        except ValueError:
            # If qcut fails, use single bin
            bins.loc[nonzero.index] = 'dose>0'

    # Split within each bin
    rng = np.random.RandomState(random_state)
    train_ids, val_ids, test_ids = [], [], []

    print(f"\nDose-stratified split (n_bins={n_bins}):")
    for bin_name, bin_subjects in bins.groupby(bins):
        bin_ids = np.array(sorted(bin_subjects.index))
        n = len(bin_ids)

        if n < 3:
            # Too few subjects - add to train
            train_ids.extend(bin_ids)
            print(f"  {bin_name}: {n} subjects -> train")
            continue

        # Calculate split sizes
        n_test = max(1, int(n * test_size))
        n_val = max(1, int(n * val_size))
        n_train = n - n_test - n_val 

        # Shuffle and split
        shuffled = rng.permutation(bin_ids)
        if combine:
            train_ids.extend(shuffled)
        else:
            train_ids.extend(shuffled[:n_train])
        val_ids.extend(shuffled[n_train:n_train + n_val])
        test_ids.extend(shuffled[n_train + n_val:])

        print(f"  {bin_name}: {n} subjects -> {n_train} train, {n_val} val, {n_test} test")

    return {
        'train': np.array(sorted(train_ids)),
        'val': np.array(sorted(val_ids)),
        'test': np.array(sorted(test_ids))
    }


# =========================================================
# Complete Data Preparation Pipeline
# =========================================================
def prepare_pkpd_data(
    csv_path: str = 'Data/QIC2025-EstDat.csv',
    test_size: float = 0.1,
    val_size: float = 0.1,
    random_state: int = 42,
    use_perkg: bool = False,
    time_windows: list = None,
    half_lives: list = None,
    add_decay: bool = True,
    stratified_split: bool = True,
    combine: bool = True
) -> Dict:
    """
    Complete data preparation pipeline matching old code.

    Returns dict with:
        - train_pk, val_pk, test_pk: PK datasets
        - train_pd, val_pd, test_pd: PD datasets
        - pk_features, pd_features: Feature lists
        - scaler_X, scaler_y_pk, scaler_y_pd: Fitted scalers
    """
    # Load data
    df_all, df_obs, df_dose = load_data(csv_path)

    # Feature engineering
    df_final = engineer_dose_features(
        df_obs, df_dose,
        time_windows=time_windows,
        half_lives=half_lives,
        add_decay=add_decay
    )

    # Add per-kg features if requested
    if use_perkg:
        df_final = add_perkg_features(df_final, time_windows)
        print("  Added per-kg features")

    # Build feature list
    features = build_feature_list(
        time_windows=time_windows,
        half_lives=half_lives,
        add_decay=add_decay,
        use_perkg=use_perkg
    )

    print(f"\nTotal features: {len(features)}")

    # Stratified split by dose
    if stratified_split:
        split_ids = stratified_dose_split(
            df_final, df_dose,
            test_size=test_size,
            val_size=val_size,
            random_state=random_state,
            combine=combine
        )
    else:
        # Simple random split
        all_ids = df_final['ID'].unique()
        rng = np.random.RandomState(random_state)
        shuffled = rng.permutation(all_ids)
        n_test = int(len(all_ids) * test_size)
        n_val = int(len(all_ids) * val_size)
        split_ids = {
            'test': shuffled[:n_test],
            'val': shuffled[n_test:n_test + n_val],
            'train': shuffled if combine else shuffled[n_test + n_val:]
        }

    # Separate PK and PD
    pk_df = df_final[df_final['DVID'] == 1].copy()
    pd_df = df_final[df_final['DVID'] == 2].copy()

    print(f"\nPK observations: {len(pk_df)}")
    print(f"PD observations: {len(pd_df)}")

    # Fit scaler on training data only
    train_mask_pk = pk_df['ID'].isin(split_ids['train'])
    train_mask_pd = pd_df['ID'].isin(split_ids['train'])

    scaler_X = StandardScaler()
    scaler_X.fit(pk_df.loc[train_mask_pk, features].values)

    scaler_y_pk = StandardScaler()
    scaler_y_pk.fit(pk_df.loc[train_mask_pk, ['DV']].values)

    scaler_y_pd = StandardScaler()
    scaler_y_pd.fit(pd_df.loc[train_mask_pd, ['DV']].values)

    # Prepare datasets
    def prepare_split(df, split_name):
        mask = df['ID'].isin(split_ids[split_name])
        sub_df = df[mask].copy()

        X = scaler_X.transform(sub_df[features].values)
        y = sub_df['DV'].values
        ids = sub_df['ID'].values
        times = sub_df['TIME'].values

        return {'X': X, 'y': y, 'ids': ids, 'times': times}

    result = {
        'train_pk': prepare_split(pk_df, 'train'),
        'val_pk': prepare_split(pk_df, 'val'),
        'test_pk': prepare_split(pk_df, 'test'),
        'train_pd': prepare_split(pd_df, 'train'),
        'val_pd': prepare_split(pd_df, 'val'),
        'test_pd': prepare_split(pd_df, 'test'),
        'pk_features': features,
        'pd_features': features,
        'scaler_X': scaler_X,
        'scaler_y_pk': scaler_y_pk,
        'scaler_y_pd': scaler_y_pd,
        'n_features': len(features)
    }

    print(f"\nDatasets prepared:")
    print(f"  Train PK: {len(result['train_pk']['y'])}")
    print(f"  Val PK: {len(result['val_pk']['y'])}")
    print(f"  Test PK: {len(result['test_pk']['y'])}")
    print(f"  Train PD: {len(result['train_pd']['y'])}")
    print(f"  Val PD: {len(result['val_pd']['y'])}")
    print(f"  Test PD: {len(result['test_pd']['y'])}")

    return result
