"""
Reliable DOM Fluorescence Prediction via Solvent Sensitive Machine Learning and Domain Refinement
==================================================================================================
Model Evaluation Module

This module provides model evaluation functionality:
1. Load trained models and make predictions
2. Evaluate performance on different test sets (full vs aqueous-focused)
3. Generate performance comparison reports
4. Analyze prediction errors and residuals
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

import xgboost as xgb
from pathlib import Path
import json

import config
from utils.metrics import (calculate_metrics, print_metrics_summary, 
                          compare_models, calculate_residuals, 
                          calculate_percentage_error)


class ModelEvaluator:
    """
    Model evaluation and analysis.
    """
    
    def __init__(self, model_path: Optional[Path] = None):
        """
        Initialize the evaluator.
        
        Parameters
        ----------
        model_path : Path, optional
            Path to trained model file
        """
        self.model_path = model_path
        self.model = None
        
    def load_model(self, model_path: Optional[Path] = None) -> xgb.XGBRegressor:
        """
        Load trained XGBoost model from JSON file.
        
        Parameters
        ----------
        model_path : Path, optional
            Path to model file. If None, uses self.model_path
        
        Returns
        -------
        xgb.XGBRegressor
            Loaded model
        """
        if model_path is None:
            model_path = self.model_path
            
        if model_path is None:
            raise ValueError("No model path provided")
        
        print(f"Loading model from: {model_path}")
        self.model = xgb.XGBRegressor()
        self.model.load_model(str(model_path))
        print("Model loaded successfully")
        
        return self.model
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using loaded model.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
        
        Returns
        -------
        np.ndarray
            Predictions
        """
        if self.model is None:
            raise ValueError("Model not loaded. Run load_model() first.")
        
        return self.model.predict(X)
    
    def evaluate_on_dataset(self, X: pd.DataFrame, y: pd.Series,
                           dataset_name: str = "Test Set") -> Dict[str, float]:
        """
        Evaluate model on a dataset.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            True target values
        dataset_name : str
            Name of the dataset
        
        Returns
        -------
        Dict[str, float]
            Evaluation metrics
        """
        y_pred = self.predict(X)
        metrics = calculate_metrics(y.values, y_pred)
        
        print_metrics_summary(metrics, dataset_name)
        
        return metrics
    
    def evaluate_by_solvent_type(self, X: pd.DataFrame, y: pd.Series,
                                is_aqueous: pd.Series) -> Dict[str, Dict]:
        """
        Evaluate model separately on aqueous and non-aqueous samples.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            True target values
        is_aqueous : pd.Series
            Boolean mask for aqueous samples
        
        Returns
        -------
        Dict[str, Dict]
            Metrics for aqueous and non-aqueous subsets
        """
        print("\n" + "="*70)
        print("Evaluation by Solvent Type")
        print("="*70)
        
        y_pred = self.predict(X)
        
        # Aqueous subset
        aqueous_mask = is_aqueous.values
        aqueous_metrics = calculate_metrics(
            y[aqueous_mask].values,
            y_pred[aqueous_mask]
        )
        print_metrics_summary(aqueous_metrics, "Aqueous Samples")
        
        # Non-aqueous subset
        non_aqueous_metrics = calculate_metrics(
            y[~aqueous_mask].values,
            y_pred[~aqueous_mask]
        )
        print_metrics_summary(non_aqueous_metrics, "Non-Aqueous Samples")
        
        return {
            "aqueous": aqueous_metrics,
            "non_aqueous": non_aqueous_metrics
        }
    
    def analyze_residuals(self, y_true: np.ndarray, y_pred: np.ndarray,
                         dataset_name: str = "Test Set"):
        """
        Analyze prediction residuals.
        
        Parameters
        ----------
        y_true : np.ndarray
            True target values
        y_pred : np.ndarray
            Predicted values
        dataset_name : str
            Name of the dataset
        """
        residuals = calculate_residuals(y_true, y_pred)
        percentage_errors = calculate_percentage_error(y_true, y_pred)
        
        print("\n" + "="*70)
        print(f"Residual Analysis - {dataset_name}")
        print("="*70)
        print(f"Mean Residual:        {np.mean(residuals):.2f} nm")
        print(f"Std Residual:         {np.std(residuals):.2f} nm")
        print(f"Min Residual:         {np.min(residuals):.2f} nm")
        print(f"Max Residual:         {np.max(residuals):.2f} nm")
        print(f"Median Abs Residual:  {np.median(np.abs(residuals)):.2f} nm")
        print(f"\nMean Percentage Error: {np.mean(percentage_errors):.2f}%")
        print(f"Median Percentage Error: {np.median(percentage_errors):.2f}%")
        print("="*70)
    
    def generate_evaluation_report(self, X_test: pd.DataFrame, y_test: pd.Series,
                                  is_aqueous_test: pd.Series,
                                  save_dir: Optional[Path] = None) -> Dict:
        """
        Generate comprehensive evaluation report.
        
        Parameters
        ----------
        X_test : pd.DataFrame
            Test feature matrix
        y_test : pd.Series
            Test target values
        is_aqueous_test : pd.Series
            Boolean mask for aqueous samples
        save_dir : Path, optional
            Directory to save results
        
        Returns
        -------
        Dict
            Complete evaluation results
        """
        if save_dir is None:
            save_dir = config.RESULTS_DIR
        
        print("\n" + "="*70)
        print("GENERATING COMPREHENSIVE EVALUATION REPORT")
        print("="*70)
        
        # Make predictions
        y_pred = self.predict(X_test)
        
        # Overall metrics
        print("\n--- Overall Performance ---")
        overall_metrics = calculate_metrics(y_test.values, y_pred)
        print_metrics_summary(overall_metrics, "Full Test Set")
        
        # Residual analysis
        self.analyze_residuals(y_test.values, y_pred, "Full Test Set")
        
        # Solvent-specific metrics
        solvent_metrics = self.evaluate_by_solvent_type(X_test, y_test, is_aqueous_test)
        
        # Save metrics to JSON
        results = {
            "overall_metrics": overall_metrics,
            "aqueous_metrics": solvent_metrics["aqueous"],
            "non_aqueous_metrics": solvent_metrics["non_aqueous"]
        }
        
        results_path = save_dir / "evaluation_metrics.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"\nMetrics saved to: {results_path}")
        
        # Save predictions
        predictions_df = pd.DataFrame({
            'actual': y_test.values,
            'predicted': y_pred,
            'residual': y_test.values - y_pred,
            'percentage_error': calculate_percentage_error(y_test.values, y_pred),
            'is_aqueous': is_aqueous_test.values
        })
        predictions_path = save_dir / "predictions.csv"
        predictions_df.to_csv(predictions_path, index=False)
        print(f"Predictions saved to: {predictions_path}")
        
        # Generate summary statistics
        summary_stats = self._generate_summary_statistics(predictions_df)
        summary_path = save_dir / "evaluation_summary.txt"
        with open(summary_path, 'w') as f:
            f.write(summary_stats)
        print(f"Summary statistics saved to: {summary_path}")
        
        print("\n" + "="*70)
        print("EVALUATION REPORT COMPLETE")
        print("="*70)
        
        return results
    
    def _generate_summary_statistics(self, predictions_df: pd.DataFrame) -> str:
        """
        Generate summary statistics text.
        
        Parameters
        ----------
        predictions_df : pd.DataFrame
            DataFrame with predictions and residuals
        
        Returns
        -------
        str
            Formatted summary statistics
        """
        summary = []
        summary.append("="*70)
        summary.append("EVALUATION SUMMARY STATISTICS")
        summary.append("="*70)
        summary.append("")
        
        # Overall statistics
        summary.append("Overall Performance:")
        summary.append(f"  Total samples: {len(predictions_df)}")
        summary.append(f"  Mean absolute error: {predictions_df['residual'].abs().mean():.2f} nm")
        summary.append(f"  Std of residuals: {predictions_df['residual'].std():.2f} nm")
        summary.append(f"  Mean percentage error: {predictions_df['percentage_error'].mean():.2f}%")
        summary.append("")
        
        # Aqueous samples
        aqueous_df = predictions_df[predictions_df['is_aqueous']]
        summary.append("Aqueous Samples:")
        summary.append(f"  Count: {len(aqueous_df)}")
        summary.append(f"  Mean absolute error: {aqueous_df['residual'].abs().mean():.2f} nm")
        summary.append(f"  Mean percentage error: {aqueous_df['percentage_error'].mean():.2f}%")
        summary.append("")
        
        # Non-aqueous samples
        non_aqueous_df = predictions_df[~predictions_df['is_aqueous']]
        summary.append("Non-Aqueous Samples:")
        summary.append(f"  Count: {len(non_aqueous_df)}")
        summary.append(f"  Mean absolute error: {non_aqueous_df['residual'].abs().mean():.2f} nm")
        summary.append(f"  Mean percentage error: {non_aqueous_df['percentage_error'].mean():.2f}%")
        summary.append("")
        
        # Prediction range
        summary.append("Prediction Range:")
        summary.append(f"  Actual min: {predictions_df['actual'].min():.1f} nm")
        summary.append(f"  Actual max: {predictions_df['actual'].max():.1f} nm")
        summary.append(f"  Predicted min: {predictions_df['predicted'].min():.1f} nm")
        summary.append(f"  Predicted max: {predictions_df['predicted'].max():.1f} nm")
        summary.append("")
        
        summary.append("="*70)
        
        return "\n".join(summary)


def compare_baseline_and_enhanced(baseline_model_path: Path,
                                  enhanced_model_path: Path,
                                  X_test: pd.DataFrame,
                                  y_test: pd.Series,
                                  X_test_aqueous: pd.DataFrame,
                                  y_test_aqueous: pd.Series) -> pd.DataFrame:
    """
    Compare baseline and enhanced model performance.
    
    Parameters
    ----------
    baseline_model_path : Path
        Path to baseline model
    enhanced_model_path : Path
        Path to enhanced model
    X_test : pd.DataFrame
        Full test features
    y_test : pd.Series
        Full test target
    X_test_aqueous : pd.DataFrame
        Aqueous test features
    y_test_aqueous : pd.Series
        Aqueous test target
    
    Returns
    -------
    pd.DataFrame
        Comparison table
    """
    print("\n" + "="*70)
    print("COMPARING BASELINE AND ENHANCED MODELS")
    print("="*70)
    
    # Evaluate baseline model
    print("\n--- Baseline Model ---")
    baseline_evaluator = ModelEvaluator(baseline_model_path)
    baseline_evaluator.load_model()
    
    baseline_full = baseline_evaluator.evaluate_on_dataset(X_test, y_test, "Full Test Set")
    baseline_aqueous = baseline_evaluator.evaluate_on_dataset(X_test_aqueous, y_test_aqueous, 
                                                              "Aqueous Test Set")
    
    # Evaluate enhanced model
    print("\n--- Enhanced Model (XGB-A(11)) ---")
    enhanced_evaluator = ModelEvaluator(enhanced_model_path)
    enhanced_evaluator.load_model()
    
    enhanced_full = enhanced_evaluator.evaluate_on_dataset(X_test, y_test, "Full Test Set")
    enhanced_aqueous = enhanced_evaluator.evaluate_on_dataset(X_test_aqueous, y_test_aqueous,
                                                              "Aqueous Test Set")
    
    # Generate comparison tables
    print("\n" + "="*70)
    print("PERFORMANCE COMPARISON")
    print("="*70)
    
    print("\n--- Full Test Set ---")
    full_comparison = compare_models(baseline_full, enhanced_full, "Full Test Set")
    print(full_comparison.to_string(index=False))
    
    print("\n--- Aqueous-Focused Test Set ---")
    aqueous_comparison = compare_models(baseline_aqueous, enhanced_aqueous, 
                                       "Aqueous-Focused Test Set")
    print(aqueous_comparison.to_string(index=False))
    
    # Combine and save
    comparison_df = pd.concat([
        full_comparison.assign(Dataset="Full Test Set"),
        aqueous_comparison.assign(Dataset="Aqueous Test Set")
    ])
    
    comparison_path = config.RESULTS_DIR / "model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)
    print(f"\nComparison saved to: {comparison_path}")
    
    return comparison_df


if __name__ == "__main__":
    # Example usage
    import sys
    
    # Check if model exists
    model_path = config.TRAINED_MODEL_PATH
    
    if not model_path.exists():
        print(f"Model file not found: {model_path}")
        print("Please ensure the trained model is in the models/ directory")
        sys.exit(1)
    
    # Check if processed data exists
    processed_path = config.DATA_DIR / "processed_data.csv"
    
    if not processed_path.exists():
        print(f"Processed data not found: {processed_path}")
        print("Please run 1_data_preprocessing.py first")
        sys.exit(1)
    
    # Load data
    print("Loading processed data...")
    df = pd.read_csv(processed_path)
    
    X = df.drop(columns=[config.TARGET_COLUMN, 'is_aqueous'])
    y = df[config.TARGET_COLUMN]
    is_aqueous = df['is_aqueous']
    
    # Split into train/test (for evaluation purposes)
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test, _, is_aqueous_test = train_test_split(
        X, y, is_aqueous,
        test_size=config.TEST_SIZE,
        stratify=is_aqueous,
        random_state=config.RANDOM_STATE
    )
    
    # Evaluate model
    evaluator = ModelEvaluator(model_path)
    evaluator.load_model()
    
    results = evaluator.generate_evaluation_report(
        X_test, y_test, is_aqueous_test,
        save_dir=config.RESULTS_DIR
    )
    
    print("\nEvaluation complete!")
