import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Tuple, Optional, Dict
import os


class PKPDDataProcessor:
    """
    Data processor for PK/PD (Pharmacokinetic/Pharmacodynamic) prediction tasks.
    Handles loading, preprocessing, and preparing data for various ML models.
    """

    def __init__(self, data_path: str = "../Data/QIC2025-EstDat.csv", verbose: bool = False):
        """
        Initialize the data processor.

        Args:
            data_path: Path to the CSV data file
            verbose: Whether to print verbose output during processing
        """
        self.data_path = data_path
        self.df = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.verbose = verbose

    def load_data(self) -> pd.DataFrame:
        """
        Load data from CSV file.

        Returns:
            DataFrame containing the raw data
        """
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found at {self.data_path}")

        self.df = pd.read_csv(self.data_path)
        if self.verbose:
            print(f"Data loaded successfully. Shape: {self.df.shape}")
            print(f"Columns: {self.df.columns.tolist()}")
        return self.df

    def explore_data(self) -> Dict:
        """
        Perform basic data exploration.

        Returns:
            Dictionary with exploration statistics
        """
        if self.df is None:
            self.load_data()

        stats = {
            'shape': self.df.shape,
            'columns': self.df.columns.tolist(),
            'missing_values': self.df.isnull().sum().to_dict(),
            'data_types': self.df.dtypes.to_dict(),
            'basic_stats': self.df.describe().to_dict(),
            'unique_subjects': self.df['ID'].nunique() if 'ID' in self.df.columns else None
        }

        if self.verbose:
            print("\n=== Data Exploration ===")
            print(f"Shape: {stats['shape']}")
            print(f"Unique subjects: {stats['unique_subjects']}")
            print(f"\nMissing values:\n{pd.Series(stats['missing_values'])}")
            print(f"\nBasic statistics:\n{pd.DataFrame(stats['basic_stats'])}")

        return stats

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer advanced features for PK/PD prediction.
        Implements 36+ engineered features based on pharmacokinetic principles.

        Args:
            df: DataFrame with observations (must contain ID, TIME, DOSE, BW, COMED columns)

        Returns:
            DataFrame with engineered features
        """
        engineered_df = df.copy()

        if self.verbose:
            print("\n=== Feature Engineering ===")
            print("Creating 36+ advanced features...")

        # Sort by ID and TIME to ensure proper temporal ordering
        engineered_df = engineered_df.sort_values(['ID', 'TIME']).reset_index(drop=True)

        # 1. TIME TRANSFORMATIONS (3 features)
        if self.verbose:
            print("  1. Time transformations...")
        engineered_df['TIME_squared'] = engineered_df['TIME'] ** 2
        engineered_df['TIME_log'] = np.log1p(engineered_df['TIME'])  # log(1 + TIME) to handle TIME=0
        engineered_df['TIME_sqrt'] = np.sqrt(engineered_df['TIME'])

        # 2. DOSE HISTORY FEATURES (3 features)
        if self.verbose:
            print("  2. Dose history features...")
        # Get dosing events (EVID=1)
        if 'EVID' in df.columns and 'AMT' in df.columns:
            dose_df = df[df['EVID'] == 1][['ID', 'TIME', 'AMT']].copy()
            dose_df = dose_df.rename(columns={'AMT': 'DOSE_AMT', 'TIME': 'DOSE_TIME'})

            # For each observation, find time since last dose and last dose amount
            engineered_df['time_since_last_dose'] = 0.0
            engineered_df['last_dose_amount'] = 0.0
            engineered_df['cumulative_dose'] = 0.0

            for subject_id in engineered_df['ID'].unique():
                subject_mask = engineered_df['ID'] == subject_id
                subject_doses = dose_df[dose_df['ID'] == subject_id].copy()

                if len(subject_doses) > 0:
                    for idx in engineered_df[subject_mask].index:
                        obs_time = engineered_df.loc[idx, 'TIME']

                        # Find all doses before this observation
                        prior_doses = subject_doses[subject_doses['DOSE_TIME'] <= obs_time]

                        if len(prior_doses) > 0:
                            # Time since last dose
                            last_dose_time = prior_doses['DOSE_TIME'].max()
                            engineered_df.loc[idx, 'time_since_last_dose'] = obs_time - last_dose_time

                            # Last dose amount
                            last_dose = prior_doses[prior_doses['DOSE_TIME'] == last_dose_time]['DOSE_AMT'].iloc[0]
                            engineered_df.loc[idx, 'last_dose_amount'] = last_dose

                            # Cumulative dose
                            engineered_df.loc[idx, 'cumulative_dose'] = prior_doses['DOSE_AMT'].sum()
        else:
            # Fallback if dose events not available
            engineered_df['time_since_last_dose'] = engineered_df['TIME']
            engineered_df['last_dose_amount'] = engineered_df.get('DOSE', 0)
            engineered_df['cumulative_dose'] = engineered_df.get('DOSE', 0)

        # 3. ROLLING WINDOW FEATURES (7 windows)
        if self.verbose:
            print("  3. Rolling window features (7 windows)...")
        windows = [24, 48, 72, 96, 120, 144, 168]  # Hours

        for window in windows:
            # Rolling mean of DV within time window
            engineered_df[f'rolling_mean_{window}h'] = 0.0

            # Calculate per patient
            for subject_id in engineered_df['ID'].unique():
                subject_mask = engineered_df['ID'] == subject_id
                subject_data = engineered_df[subject_mask].copy()

                for idx in subject_data.index:
                    current_time = engineered_df.loc[idx, 'TIME']
                    time_window_mask = (subject_data['TIME'] >= current_time - window) & \
                                     (subject_data['TIME'] < current_time)

                    if 'DV' in subject_data.columns:
                        window_values = subject_data.loc[subject_data.index[time_window_mask], 'DV']
                        if len(window_values) > 0:
                            engineered_df.loc[idx, f'rolling_mean_{window}h'] = window_values.mean()

        # 4. EXPONENTIAL DECAY FEATURES (3 half-lives)
        if self.verbose:
            print("  4. Exponential decay features (drug elimination)...")
        half_lives = [24, 48, 72]  # Hours

        for hl in half_lives:
            decay_rate = np.log(2) / hl
            engineered_df[f'exp_decay_{hl}h'] = np.exp(-decay_rate * engineered_df['TIME'])

            # Dose-weighted exponential decay
            if 'last_dose_amount' in engineered_df.columns:
                engineered_df[f'dose_exp_decay_{hl}h'] = \
                    engineered_df['last_dose_amount'] * np.exp(-decay_rate * engineered_df['time_since_last_dose'])

        # 5. INTERACTION FEATURES (3 interactions)
        if self.verbose:
            print("  5. Interaction features...")
        if 'BW' in engineered_df.columns:
            engineered_df['BW_x_DOSE'] = engineered_df['BW'] * engineered_df.get('DOSE', 0)
            engineered_df['BW_x_TIME'] = engineered_df['BW'] * engineered_df['TIME']

        engineered_df['TIME_x_DOSE'] = engineered_df['TIME'] * engineered_df.get('DOSE', 0)

        if 'COMED' in engineered_df.columns:
            engineered_df['COMED_x_DOSE'] = engineered_df['COMED'] * engineered_df.get('DOSE', 0)
            engineered_df['COMED_x_TIME'] = engineered_df['COMED'] * engineered_df['TIME']

        # 6. PER-KG NORMALIZATION (dose features normalized by body weight)
        if self.verbose:
            print("  6. Per-kg dose normalization...")
        if 'BW' in engineered_df.columns:
            # Avoid division by zero
            bw_safe = engineered_df['BW'].replace(0, 1)

            if 'DOSE' in engineered_df.columns:
                engineered_df['dose_per_kg'] = engineered_df['DOSE'] / bw_safe

            if 'last_dose_amount' in engineered_df.columns:
                engineered_df['last_dose_per_kg'] = engineered_df['last_dose_amount'] / bw_safe

            if 'cumulative_dose' in engineered_df.columns:
                engineered_df['cumulative_dose_per_kg'] = engineered_df['cumulative_dose'] / bw_safe

        # 7. ADDITIONAL TEMPORAL FEATURES
        if self.verbose:
            print("  7. Additional temporal features...")
        # Time of day effects (assuming TIME is in hours from start)
        engineered_df['TIME_sin'] = np.sin(2 * np.pi * engineered_df['TIME'] / 24)
        engineered_df['TIME_cos'] = np.cos(2 * np.pi * engineered_df['TIME'] / 24)

        # Observation order per patient
        engineered_df['observation_number'] = engineered_df.groupby('ID').cumcount() + 1

        # Time from previous observation
        engineered_df['time_from_prev_obs'] = engineered_df.groupby('ID')['TIME'].diff().fillna(0)

        if self.verbose:
            print(f"Feature engineering complete! Created {len(engineered_df.columns)} total features.")
            print(f"New engineered features: {len(engineered_df.columns) - len(df.columns)}")

        return engineered_df

    def prepare_features_target(self, target_col: str = 'DV',
                                feature_cols: Optional[list] = None,
                                use_feature_engineering: bool = False) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Separate features and target variable.

        Args:
            target_col: Name of the target column (default: 'DV' for dependent variable)
            feature_cols: List of feature column names. If None, uses all except target and ID columns
            use_feature_engineering: Whether to apply advanced feature engineering (36+ features)

        Returns:
            Tuple of (features DataFrame, target Series)
        """
        if self.df is None:
            self.load_data()

        # Apply feature engineering if requested
        if use_feature_engineering:
            print("\n🔬 Applying advanced feature engineering (36+ features)...")
            # Engineer features on full dataset to get proper dose history context
            engineered_full = self.engineer_features(self.df)

            # Filter to observations only AFTER engineering
            if 'EVID' in engineered_full.columns:
                obs_df = engineered_full[engineered_full['EVID'] == 0].copy()
            else:
                obs_df = engineered_full.copy()

            # Remove rows where target is missing
            if 'MDV' in obs_df.columns:
                obs_df = obs_df[obs_df['MDV'] == 0]

        else:
            # Original behavior - no feature engineering
            # Filter to observations only (EVID=0 for observations)
            if 'EVID' in self.df.columns:
                obs_df = self.df[self.df['EVID'] == 0].copy()
            else:
                obs_df = self.df.copy()

            # Remove rows where target is missing
            if 'MDV' in obs_df.columns:
                obs_df = obs_df[obs_df['MDV'] == 0]

        # Default feature columns: exclude ID, target, and metadata columns
        if feature_cols is None:
            exclude_cols = ['ID', target_col, 'EVID', 'MDV', 'CMT', 'DVID', 'AMT']
            feature_cols = [col for col in obs_df.columns if col not in exclude_cols]

        X = obs_df[feature_cols]
        y = obs_df[target_col]

        print(f"\n📊 Features: {X.shape[1]} features, {X.shape[0]} samples")
        if self.verbose:
            print(f"Target shape: {y.shape}")
            if use_feature_engineering:
                print(f"Feature columns (first 15): {feature_cols[:15]}")

        return X, y

    def preprocess_data(self, X: pd.DataFrame, y: pd.Series,
                       test_size: float = 0.2,
                       random_state: int = 42,
                       scale_features: bool = True,
                       scale_target: bool = False) -> Tuple:
        """
        Preprocess data: train-test split and scaling.

        Args:
            X: Features DataFrame
            y: Target Series
            test_size: Proportion of test set
            random_state: Random seed for reproducibility
            scale_features: Whether to scale features
            scale_target: Whether to scale target

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        if self.verbose:
            print(f"\nTrain set size: {len(X_train)}")
            print(f"Test set size: {len(X_test)}")

        # Scale features
        if scale_features:
            X_train = pd.DataFrame(
                self.scaler_X.fit_transform(X_train.values),
                columns=X_train.columns,
                index=X_train.index
            )
            X_test = pd.DataFrame(
                self.scaler_X.transform(X_test.values),
                columns=X_test.columns,
                index=X_test.index
            )
            if self.verbose:
                print("Features scaled using StandardScaler")

        # Scale target if requested
        if scale_target:
            y_train = pd.Series(
                self.scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten(),
                index=y_train.index
            )
            y_test = pd.Series(
                self.scaler_y.transform(y_test.values.reshape(-1, 1)).flatten(),
                index=y_test.index
            )
            if self.verbose:
                print("Target scaled using StandardScaler")

        return X_train, X_test, y_train, y_test

    def prepare_sequence_data(self, X: pd.DataFrame, y: pd.Series,
                             sequence_length: int = 10,
                             step_size: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequential data for time-series models (LSTM, CNN).
        Groups data by subject ID and creates sequences.

        Args:
            X: Features DataFrame
            y: Target Series
            sequence_length: Length of each sequence
            step_size: Step size for sliding window

        Returns:
            Tuple of (X_sequences, y_sequences) as numpy arrays
        """
        if 'ID' not in self.df.columns:
            raise ValueError("ID column required for sequence data preparation")

        X_sequences = []
        y_sequences = []

        # Get subject IDs
        subject_ids = X.index.to_series().map(
            lambda idx: self.df.loc[self.df.index == idx, 'ID'].values[0]
        )

        # Group by subject
        for subject_id in subject_ids.unique():
            subject_mask = subject_ids == subject_id
            X_subject = X[subject_mask].values
            y_subject = y[subject_mask].values

            # Create sequences with sliding window
            for i in range(0, len(X_subject) - sequence_length + 1, step_size):
                X_sequences.append(X_subject[i:i+sequence_length])
                y_sequences.append(y_subject[i+sequence_length-1])  # Predict last value

        X_sequences = np.array(X_sequences)
        y_sequences = np.array(y_sequences)

        if self.verbose:
            print(f"\nSequence data prepared:")
            print(f"X_sequences shape: {X_sequences.shape}")
            print(f"y_sequences shape: {y_sequences.shape}")

        return X_sequences, y_sequences

    def get_graph_data(self) -> Dict:
        """
        Prepare data for Graph Neural Networks.
        Creates node features and edge indices based on temporal connections.

        Returns:
            Dictionary with graph data components
        """
        if self.df is None:
            self.load_data()

        # Filter to observations
        obs_df = self.df[self.df['EVID'] == 0].copy() if 'EVID' in self.df.columns else self.df.copy()

        # Node features: all covariates
        feature_cols = ['BW', 'COMED', 'DOSE', 'TIME']
        node_features = obs_df[feature_cols].values

        # Target values
        targets = obs_df['DV'].values

        # Create edge indices (connect sequential timepoints within same subject)
        edge_index = []
        node_idx = 0

        for subject_id in obs_df['ID'].unique():
            subject_mask = obs_df['ID'] == subject_id
            subject_indices = np.where(subject_mask)[0]

            # Connect consecutive timepoints
            for i in range(len(subject_indices) - 1):
                edge_index.append([subject_indices[i], subject_indices[i+1]])
                edge_index.append([subject_indices[i+1], subject_indices[i]])  # Bidirectional

        edge_index = np.array(edge_index).T if edge_index else np.array([[], []])

        graph_data = {
            'node_features': node_features,
            'edge_index': edge_index,
            'targets': targets,
            'num_nodes': len(node_features),
            'num_edges': edge_index.shape[1] if edge_index.size > 0 else 0
        }

        if self.verbose:
            print(f"\nGraph data prepared:")
            print(f"Number of nodes: {graph_data['num_nodes']}")
            print(f"Number of edges: {graph_data['num_edges']}")
            print(f"Node features shape: {graph_data['node_features'].shape}")

        return graph_data

    def get_full_pipeline(self, target_col: str = 'DV',
                         feature_cols: Optional[list] = None,
                         test_size: float = 0.2,
                         random_state: int = 42,
                         scale_features: bool = True,
                         scale_target: bool = False,
                         use_feature_engineering: bool = False) -> Dict:
        """
        Complete preprocessing pipeline.

        Args:
            target_col: Target column name
            feature_cols: Feature column names (None = auto-select)
            test_size: Test set proportion
            random_state: Random seed
            scale_features: Whether to scale features
            scale_target: Whether to scale target
            use_feature_engineering: Whether to apply advanced feature engineering (36+ features)

        Returns:
            Dictionary with all preprocessed data
        """
        # Load and explore
        self.load_data()
        stats = self.explore_data()

        # Prepare features and target (with optional feature engineering)
        X, y = self.prepare_features_target(target_col, feature_cols, use_feature_engineering)

        # Preprocess
        X_train, X_test, y_train, y_test = self.preprocess_data(
            X, y, test_size, random_state, scale_features, scale_target
        )

        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'stats': stats,
            'feature_cols': X.columns.tolist(),
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y if scale_target else None
        }


if __name__ == "__main__":
    # Example usage
    processor = PKPDDataProcessor()

    # Get preprocessed data
    data = processor.get_full_pipeline(scale_features=True, scale_target=False)

    print("\n=== Preprocessing Complete ===")
    print(f"Training samples: {len(data['X_train'])}")
    print(f"Testing samples: {len(data['X_test'])}")
