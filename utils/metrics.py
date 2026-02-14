"""
Reliable DOM Fluorescence Prediction via Solvent Sensitive Machine Learning and Domain Refinement
==================================================================================================
Evaluation Metrics Module

This module provides evaluation metrics for assessing model performance
in predicting maximum fluorescence emission wavelength (λem,max).
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, Tuple
import pandas as pd


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate regression metrics for model evaluation.
    
    Parameters
    ----------
    y_true : np.ndarray
        True target values
    y_pred : np.ndarray
        Predicted target values
    
    Returns
    -------
    Dict[str, float]
        Dictionary containing R², MAE, and RMSE
    """
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    return {
        "R2": r2,
        "MAE": mae,
        "RMSE": rmse
    }


def calculate_improvement(baseline_metrics: Dict[str, float], 
                         enhanced_metrics: Dict[str, float]) -> Dict[str, float]:
    """
    Calculate percentage improvement from baseline to enhanced model.
    
    Parameters
    ----------
    baseline_metrics : Dict[str, float]
        Metrics from baseline model
    enhanced_metrics : Dict[str, float]
        Metrics from enhanced model
    
    Returns
    -------
    Dict[str, float]
        Percentage improvement for each metric
    """
    improvements = {}
    
    # R² improvement (higher is better)
    improvements["R2_improvement"] = (
        (enhanced_metrics["R2"] - baseline_metrics["R2"]) / baseline_metrics["R2"] * 100
    )
    
    # MAE improvement (lower is better, so we invert)
    improvements["MAE_improvement"] = (
        (baseline_metrics["MAE"] - enhanced_metrics["MAE"]) / baseline_metrics["MAE"] * 100
    )
    
    # RMSE improvement (lower is better, so we invert)
    improvements["RMSE_improvement"] = (
        (baseline_metrics["RMSE"] - enhanced_metrics["RMSE"]) / baseline_metrics["RMSE"] * 100
    )
    
    return improvements


def compare_models(baseline_metrics: Dict[str, float],
                   enhanced_metrics: Dict[str, float],
                   dataset_name: str = "Test Set") -> pd.DataFrame:
    """
    Create a comparison table between baseline and enhanced models.
    
    Parameters
    ----------
    baseline_metrics : Dict[str, float]
        Metrics from baseline model
    enhanced_metrics : Dict[str, float]
        Metrics from enhanced model
    dataset_name : str, optional
        Name of the dataset being evaluated
    
    Returns
    -------
    pd.DataFrame
        Comparison table with metrics and improvements
    """
    improvements = calculate_improvement(baseline_metrics, enhanced_metrics)
    
    comparison_df = pd.DataFrame({
        "Metric": ["R²", "MAE", "RMSE"],
        "Baseline": [
            f"{baseline_metrics['R2']:.4f}",
            f"{baseline_metrics['MAE']:.2f}",
            f"{baseline_metrics['RMSE']:.2f}"
        ],
        "XGB-A(11)": [
            f"{enhanced_metrics['R2']:.4f}",
            f"{enhanced_metrics['MAE']:.2f}",
            f"{enhanced_metrics['RMSE']:.2f}"
        ],
        "Improvement (%)": [
            f"{improvements['R2_improvement']:+.2f}%",
            f"{improvements['MAE_improvement']:+.2f}%",
            f"{improvements['RMSE_improvement']:+.2f}%"
        ]
    })
    
    comparison_df.name = dataset_name
    return comparison_df


def evaluate_on_subsets(model, X: pd.DataFrame, y: pd.Series, 
                       solvent_mask: pd.Series) -> Tuple[Dict, Dict]:
    """
    Evaluate model performance on aqueous and non-aqueous subsets.
    
    Parameters
    ----------
    model : trained model
        Model with predict() method
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        True target values
    solvent_mask : pd.Series
        Boolean mask indicating aqueous samples (True = aqueous)
    
    Returns
    -------
    Tuple[Dict, Dict]
        (aqueous_metrics, non_aqueous_metrics)
    """
    y_pred = model.predict(X)
    
    # Aqueous subset
    aqueous_metrics = calculate_metrics(
        y[solvent_mask].values,
        y_pred[solvent_mask]
    )
    
    # Non-aqueous subset
    non_aqueous_metrics = calculate_metrics(
        y[~solvent_mask].values,
        y_pred[~solvent_mask]
    )
    
    return aqueous_metrics, non_aqueous_metrics


def print_metrics_summary(metrics: Dict[str, float], dataset_name: str = "Dataset"):
    """
    Print formatted metrics summary.
    
    Parameters
    ----------
    metrics : Dict[str, float]
        Dictionary containing evaluation metrics
    dataset_name : str, optional
        Name of the dataset
    """
    print(f"\n{'='*60}")
    print(f"Performance on {dataset_name}")
    print(f"{'='*60}")
    print(f"R² Score:  {metrics['R2']:.4f}")
    print(f"MAE:       {metrics['MAE']:.2f} nm")
    print(f"RMSE:      {metrics['RMSE']:.2f} nm")
    print(f"{'='*60}\n")


def calculate_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Calculate prediction residuals.
    
    Parameters
    ----------
    y_true : np.ndarray
        True target values
    y_pred : np.ndarray
        Predicted target values
    
    Returns
    -------
    np.ndarray
        Residuals (y_true - y_pred)
    """
    return y_true - y_pred


def calculate_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Calculate percentage error for each prediction.
    
    Parameters
    ----------
    y_true : np.ndarray
        True target values
    y_pred : np.ndarray
        Predicted target values
    
    Returns
    -------
    np.ndarray
        Percentage errors
    """
    return np.abs((y_true - y_pred) / y_true) * 100


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    y_true = np.random.uniform(400, 600, 100)
    y_pred_baseline = y_true + np.random.normal(0, 15, 100)
    y_pred_enhanced = y_true + np.random.normal(0, 12, 100)
    
    baseline_metrics = calculate_metrics(y_true, y_pred_baseline)
    enhanced_metrics = calculate_metrics(y_true, y_pred_enhanced)
    
    print_metrics_summary(baseline_metrics, "Baseline Model")
    print_metrics_summary(enhanced_metrics, "XGB-A(11) Model")
    
    comparison = compare_models(baseline_metrics, enhanced_metrics, "Test Set")
    print("\nModel Comparison:")
    print(comparison.to_string(index=False))
