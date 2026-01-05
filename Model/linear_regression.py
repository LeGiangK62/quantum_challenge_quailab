import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from typing import Dict, Optional
import pandas as pd


class LinearRegressionModel:
    """
    Linear Regression model for PK/PD prediction.
    Supports standard OLS, Ridge, and Lasso regression.
    """

    def __init__(self, model_type: str = 'ols', alpha: float = 1.0):
        """
        Initialize Linear Regression model.

        Args:
            model_type: Type of linear model ('ols', 'ridge', 'lasso')
            alpha: Regularization strength for Ridge/Lasso
        """
        self.model_type = model_type
        self.alpha = alpha

        if model_type == 'ols':
            self.model = LinearRegression()
        elif model_type == 'ridge':
            self.model = Ridge(alpha=alpha)
        elif model_type == 'lasso':
            self.model = Lasso(alpha=alpha)
        else:
            raise ValueError(f"Unknown model_type: {model_type}. Use 'ols', 'ridge', or 'lasso'.")

        self.is_fitted = False

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train the linear regression model.

        Args:
            X_train: Training features
            y_train: Training targets
        """
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values
        if isinstance(y_train, pd.Series):
            y_train = y_train.values

        print(f"Training {self.model_type.upper()} model...")
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

        print(f"\n=== {self.model_type.upper()} Evaluation Metrics ===")
        for metric, value in metrics.items():
            print(f"{metric.upper()}: {value:.4f}")

        return metrics

    def get_coefficients(self) -> Dict:
        """
        Get model coefficients.

        Returns:
            Dictionary with intercept and coefficients
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained first.")

        return {
            'intercept': self.model.intercept_,
            'coefficients': self.model.coef_
        }

    def get_feature_importance(self, feature_names: Optional[list] = None) -> pd.DataFrame:
        """
        Get feature importance based on coefficient magnitudes.

        Args:
            feature_names: List of feature names

        Returns:
            DataFrame with feature importance
        """
        coeffs = self.get_coefficients()

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(coeffs['coefficients']))]

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'coefficient': coeffs['coefficients'],
            'abs_coefficient': np.abs(coeffs['coefficients'])
        })

        importance_df = importance_df.sort_values('abs_coefficient', ascending=False)

        print("\n=== Feature Importance (by coefficient magnitude) ===")
        print(importance_df.to_string(index=False))

        return importance_df


def train_and_evaluate_linear_models(X_train, X_test, y_train, y_test,
                                     feature_names: Optional[list] = None) -> Dict:
    """
    Train and evaluate all linear model variants.

    Args:
        X_train, X_test, y_train, y_test: Train and test data
        feature_names: Names of features

    Returns:
        Dictionary with all results
    """
    results = {}

    for model_type in ['ols', 'ridge', 'lasso']:
        print(f"\n{'='*60}")
        print(f"Training {model_type.upper()} Model")
        print('='*60)

        model = LinearRegressionModel(model_type=model_type, alpha=1.0)
        model.train(X_train, y_train)
        metrics = model.evaluate(X_test, y_test)

        if feature_names:
            importance = model.get_feature_importance(feature_names)
        else:
            importance = None

        results[model_type] = {
            'model': model,
            'metrics': metrics,
            'importance': importance
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

    # Train and evaluate linear models
    results = train_and_evaluate_linear_models(
        data['X_train'], data['X_test'],
        data['y_train'], data['y_test'],
        feature_names=data['feature_cols']
    )

    # Compare models
    print("\n" + "="*60)
    print("Model Comparison")
    print("="*60)
    comparison_df = pd.DataFrame({
        model_type: results[model_type]['metrics']
        for model_type in results.keys()
    }).T
    print(comparison_df.to_string())
