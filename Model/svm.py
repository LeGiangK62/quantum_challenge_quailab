import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from typing import Dict, Optional
import pandas as pd


class SVMModel:
    """
    Support Vector Machine (SVM) model for PK/PD prediction.
    Supports different kernels and hyperparameter tuning.
    """

    def __init__(self, kernel: str = 'rbf', C: float = 1.0,
                 epsilon: float = 0.1, gamma: str = 'scale'):
        """
        Initialize SVM model.

        Args:
            kernel: Kernel type ('linear', 'poly', 'rbf', 'sigmoid')
            C: Regularization parameter
            epsilon: Epsilon in epsilon-SVR model
            gamma: Kernel coefficient for 'rbf', 'poly', 'sigmoid'
        """
        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon
        self.gamma = gamma

        self.model = SVR(
            kernel=kernel,
            C=C,
            epsilon=epsilon,
            gamma=gamma
        )

        self.is_fitted = False

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train the SVM model.

        Args:
            X_train: Training features
            y_train: Training targets
        """
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values
        if isinstance(y_train, pd.Series):
            y_train = y_train.values

        print(f"Training SVM model with {self.kernel} kernel...")
        print(f"Parameters: C={self.C}, epsilon={self.epsilon}, gamma={self.gamma}")
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        print("Training completed.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Input features

        Returns:
            Predicted values
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before prediction.")

        if isinstance(X, pd.DataFrame):
            X = X.values

        return self.model.predict(X)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance.

        Args:
            X_test: Test features
            y_test: True test targets

        Returns:
            Dictionary with evaluation metrics
        """
        y_pred = self.predict(X_test)

        if isinstance(y_test, pd.Series):
            y_test = y_test.values

        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
        }

        print(f"\n=== SVM ({self.kernel}) Evaluation Metrics ===")
        for metric, value in metrics.items():
            print(f"{metric.upper()}: {value:.4f}")

        return metrics

    def get_support_vectors_info(self) -> Dict:
        """
        Get information about support vectors.

        Returns:
            Dictionary with support vector information
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained first.")

        info = {
            'n_support_vectors': len(self.model.support_vectors_),
            'support_vectors': self.model.support_vectors_,
            'dual_coef': self.model.dual_coef_,
        }

        print(f"\nNumber of support vectors: {info['n_support_vectors']}")

        return info


class SVMGridSearch:
    """
    SVM model with grid search for hyperparameter tuning.
    """

    def __init__(self, kernel: str = 'rbf', cv: int = 5):
        """
        Initialize SVM with grid search.

        Args:
            kernel: Kernel type
            cv: Number of cross-validation folds
        """
        self.kernel = kernel
        self.cv = cv
        self.best_model = None
        self.best_params = None
        self.grid_search = None

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
             param_grid: Optional[Dict] = None) -> None:
        """
        Train SVM with grid search.

        Args:
            X_train: Training features
            y_train: Training targets
            param_grid: Dictionary of parameters to search
        """
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values
        if isinstance(y_train, pd.Series):
            y_train = y_train.values

        # Default parameter grid
        if param_grid is None:
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'epsilon': [0.01, 0.1, 0.2],
                'gamma': ['scale', 'auto', 0.1, 0.01]
            }

        print(f"Starting grid search with {self.kernel} kernel...")
        print(f"Parameter grid: {param_grid}")
        print(f"Cross-validation folds: {self.cv}")

        base_model = SVR(kernel=self.kernel)
        self.grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=self.cv,
            scoring='neg_mean_squared_error',
            verbose=1,
            n_jobs=-1
        )

        self.grid_search.fit(X_train, y_train)
        self.best_model = self.grid_search.best_estimator_
        self.best_params = self.grid_search.best_params_

        print("\n=== Grid Search Complete ===")
        print(f"Best parameters: {self.best_params}")
        print(f"Best CV score (neg_MSE): {self.grid_search.best_score_:.4f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using best model.

        Args:
            X: Input features

        Returns:
            Predicted values
        """
        if self.best_model is None:
            raise ValueError("Model must be trained before prediction.")

        if isinstance(X, pd.DataFrame):
            X = X.values

        return self.best_model.predict(X)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate best model.

        Args:
            X_test: Test features
            y_test: True test targets

        Returns:
            Dictionary with evaluation metrics
        """
        y_pred = self.predict(X_test)

        if isinstance(y_test, pd.Series):
            y_test = y_test.values

        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
        }

        print(f"\n=== Best SVM ({self.kernel}) Evaluation Metrics ===")
        for metric, value in metrics.items():
            print(f"{metric.upper()}: {value:.4f}")

        return metrics


def train_and_evaluate_svm_models(X_train, X_test, y_train, y_test,
                                  use_grid_search: bool = False) -> Dict:
    """
    Train and evaluate SVM models with different kernels.

    Args:
        X_train, X_test, y_train, y_test: Train and test data
        use_grid_search: Whether to use grid search for hyperparameter tuning

    Returns:
        Dictionary with all results
    """
    results = {}
    kernels = ['linear', 'rbf', 'poly']

    for kernel in kernels:
        print(f"\n{'='*60}")
        print(f"Training SVM with {kernel.upper()} kernel")
        print('='*60)

        if use_grid_search:
            model = SVMGridSearch(kernel=kernel, cv=3)
            model.train(X_train, y_train)
        else:
            model = SVMModel(kernel=kernel)
            model.train(X_train, y_train)

        metrics = model.evaluate(X_test, y_test)

        results[kernel] = {
            'model': model,
            'metrics': metrics
        }

    return results


if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.append('..')
    from Utils.pre_processing import PKPDDataProcessor

    # Load and preprocess data
    processor = PKPDDataProcessor()
    data = processor.get_full_pipeline(scale_features=True)

    # Train and evaluate SVM models
    print("\n" + "="*60)
    print("Training SVM Models (without grid search)")
    print("="*60)
    results = train_and_evaluate_svm_models(
        data['X_train'], data['X_test'],
        data['y_train'], data['y_test'],
        use_grid_search=False
    )

    # Compare models
    print("\n" + "="*60)
    print("SVM Kernel Comparison")
    print("="*60)
    comparison_df = pd.DataFrame({
        kernel: results[kernel]['metrics']
        for kernel in results.keys()
    }).T
    print(comparison_df.to_string())

    # Optionally run grid search for best kernel
    print("\n" + "="*60)
    print("Running Grid Search for RBF kernel")
    print("="*60)
    grid_model = SVMGridSearch(kernel='rbf', cv=3)
    grid_model.train(data['X_train'], data['y_train'])
    grid_metrics = grid_model.evaluate(data['X_test'], data['y_test'])
