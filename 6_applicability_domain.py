"""
Reliable DOM Fluorescence Prediction via Solvent Sensitive Machine Learning and Domain Refinement
==================================================================================================
Applicability Domain Analysis Module (AD-SAL)

This module implements the Applicability Domain based on Similarity-weighted Average Leverage (AD-SAL)
to identify reliable predictions and filter out unreliable ones.

Key features:
1. Calculate weighted similarity between samples
2. Compute similarity density index (ρ)
3. Compute local discontinuity index (δ)
4. Test different α values for AD threshold
5. Apply AD filtering to improve prediction reliability

Reference: Based on the AD-SAL methodology from the paper
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import xgboost as xgb
from pathlib import Path
import json
import matplotlib.pyplot as plt

import config
from utils.metrics import calculate_metrics, print_metrics_summary


class ADSALAnalyzer:
    """
    Applicability Domain analysis using Similarity-weighted Average Leverage (AD-SAL).
    """
    
    def __init__(self, model_path: Optional[Path] = None, 
                 alpha: float = 15.0, 
                 epsilon: float = 1e-6):
        """
        Initialize AD-SAL analyzer.
        
        Parameters
        ----------
        model_path : Path, optional
            Path to trained model
        alpha : float
            Exponential transformation intensity (default: 15)
        epsilon : float
            Small value to prevent division by zero (default: 1e-6)
        """
        self.model_path = model_path
        self.model = None
        self.X_train = None
        self.y_train = None
        self.alpha = alpha
        self.epsilon = epsilon
        
    def load_model(self, model_path: Optional[Path] = None) -> xgb.XGBRegressor:
        """
        Load trained XGBoost model.
        
        Parameters
        ----------
        model_path : Path, optional
            Path to model file
        
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
    
    def set_training_data(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Set training data for AD calculation.
        
        Parameters
        ----------
        X_train : pd.DataFrame
            Training feature matrix
        y_train : pd.Series
            Training target values
        """
        self.X_train = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
        self.y_train = y_train.values if isinstance(y_train, pd.Series) else y_train
        print(f"Training data set: {self.X_train.shape[0]} samples, {self.X_train.shape[1]} features")
    
    def compute_weighted_similarity(self, x_query: np.ndarray, 
                                   feature_weights: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calculate weighted similarity between query sample and training set.
        
        Formula: sim = 1 / (1 + sqrt(Σ w_k * |x_query_k - x_train_k|^2))
        
        Parameters
        ----------
        x_query : np.ndarray
            Query sample features (1D array)
        feature_weights : np.ndarray, optional
            Feature weights (1D array)
        
        Returns
        -------
        np.ndarray
            Similarities with all training samples
        """
        if self.X_train is None:
            raise ValueError("Training data not set. Run set_training_data() first.")
        
        n_features = self.X_train.shape[1]
        
        if feature_weights is None:
            feature_weights = np.ones(n_features) / n_features
        
        # Calculate feature ranges for normalization
        feature_min = self.X_train.min(axis=0)
        feature_max = self.X_train.max(axis=0)
        feature_range = feature_max - feature_min
        feature_range[feature_range == 0] = 1  # Avoid division by zero
        
        # Normalize
        x_query_norm = (x_query - feature_min) / feature_range
        X_train_norm = (self.X_train - feature_min) / feature_range
        
        # Calculate weighted Euclidean distance
        diff = np.abs(X_train_norm - x_query_norm)
        weighted_diff = diff * feature_weights
        distances = np.sqrt(np.sum(weighted_diff ** 2, axis=1))
        
        # Convert to similarity (0-1)
        similarities = 1 / (1 + distances)
        
        return similarities
    
    def compute_density_index(self, similarities: np.ndarray) -> float:
        """
        Calculate similarity density index ρ_i.
        
        Formula: ρ_i = Σ exp(α * (sim_ij - 1)) * sim_ij / (sim_ij + ε)
        
        Parameters
        ----------
        similarities : np.ndarray
            Similarities with all training samples (1D array)
        
        Returns
        -------
        float
            Density index
        """
        weights = np.exp(self.alpha * (similarities - 1))
        weighted_sim = weights * similarities / (similarities + self.epsilon)
        density = np.sum(weighted_sim)
        
        return density
    
    def compute_discontinuity_index(self, similarities: np.ndarray, 
                                   y_query: float) -> float:
        """
        Calculate local discontinuity index δ_i.
        
        Used to evaluate activity cliffs (structurally similar but activity different).
        
        Formula: δ_i = Σ w_ij * |y_i - y_j| / (|y_i| + |y_j| + ε)
        
        Parameters
        ----------
        similarities : np.ndarray
            Similarities with all training samples
        y_query : float
            Query sample target value
        
        Returns
        -------
        float
            Discontinuity index
        """
        if self.y_train is None:
            raise ValueError("Training target values not set.")
        
        weights = np.exp(self.alpha * (similarities - 1))
        
        y_diff = np.abs(y_query - self.y_train)
        y_sum = np.abs(y_query) + np.abs(self.y_train) + self.epsilon
        
        weighted_discontinuity = weights * y_diff / y_sum
        discontinuity = np.sum(weighted_discontinuity)
        
        return discontinuity
    
    def compute_ad_indices(self, X_query: pd.DataFrame, 
                          y_query: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Calculate AD-SAL indices for query samples.
        
        Parameters
        ----------
        X_query : pd.DataFrame
            Query feature matrix
        y_query : pd.Series, optional
            Query target values
        
        Returns
        -------
        pd.DataFrame
            AD indices for each sample
        """
        print("\nCalculating AD-SAL indices...")
        
        X_query_array = X_query.values if isinstance(X_query, pd.DataFrame) else X_query
        y_query_array = y_query.values if y_query is not None and isinstance(y_query, pd.Series) else y_query
        
        results = []
        
        for i in range(len(X_query_array)):
            x = X_query_array[i]
            y = y_query_array[i] if y_query_array is not None else None
            
            # Calculate similarities
            similarities = self.compute_weighted_similarity(x)
            
            # Calculate density index
            density = self.compute_density_index(similarities)
            
            # Calculate discontinuity index (if labels available)
            if y is not None:
                discontinuity = self.compute_discontinuity_index(similarities, y)
            else:
                discontinuity = None
            
            results.append({
                'density': density,
                'discontinuity': discontinuity,
                'max_similarity': np.max(similarities),
                'mean_similarity': np.mean(similarities)
            })
        
        results_df = pd.DataFrame(results)
        
        print(f"AD indices calculated for {len(results_df)} samples")
        print(f"\nDensity index (ρ) statistics:")
        print(f"  Mean: {results_df['density'].mean():.4f}")
        print(f"  Std: {results_df['density'].std():.4f}")
        print(f"  Range: [{results_df['density'].min():.4f}, {results_df['density'].max():.4f}]")
        
        if results_df['discontinuity'].notna().any():
            print(f"\nDiscontinuity index (δ) statistics:")
            print(f"  Mean: {results_df['discontinuity'].mean():.4f}")
            print(f"  Std: {results_df['discontinuity'].std():.4f}")
        
        return results_df
    
    def determine_ad_threshold(self, ad_indices: pd.DataFrame, 
                              percentile: float = 95.0) -> float:
        """
        Determine AD threshold based on density index distribution.
        
        Parameters
        ----------
        ad_indices : pd.DataFrame
            AD indices from training set
        percentile : float
            Percentile for threshold (default: 95)
        
        Returns
        -------
        float
            AD threshold
        """
        threshold = np.percentile(ad_indices['density'], percentile)
        print(f"\nAD threshold (density ρ at {percentile}th percentile): {threshold:.4f}")
        
        return threshold
    
    def apply_ad_filter(self, X: pd.DataFrame, y: pd.Series,
                       ad_indices: pd.DataFrame,
                       threshold: float) -> Tuple[pd.DataFrame, pd.Series, np.ndarray]:
        """
        Filter samples within applicability domain.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target values
        ad_indices : pd.DataFrame
            AD indices
        threshold : float
            Density threshold
        
        Returns
        -------
        Tuple[pd.DataFrame, pd.Series, np.ndarray]
            (X_filtered, y_filtered, mask_within_ad)
        """
        mask_within_ad = ad_indices['density'] >= threshold
        
        X_filtered = X[mask_within_ad]
        y_filtered = y[mask_within_ad]
        
        n_total = len(X)
        n_within = mask_within_ad.sum()
        n_outside = n_total - n_within
        
        print(f"\nAD Filtering Results:")
        print(f"  Total samples: {n_total}")
        print(f"  Within AD: {n_within} ({n_within/n_total*100:.1f}%)")
        print(f"  Outside AD: {n_outside} ({n_outside/n_total*100:.1f}%)")
        
        return X_filtered, y_filtered, mask_within_ad
    
    def evaluate_with_ad(self, X_test: pd.DataFrame, y_test: pd.Series,
                        threshold: float,
                        dataset_name: str = "Test Set") -> Dict:
        """
        Evaluate model performance with AD filtering.
        
        Parameters
        ----------
        X_test : pd.DataFrame
            Test feature matrix
        y_test : pd.Series
            Test target values
        threshold : float
            AD threshold
        dataset_name : str
            Name of the dataset
        
        Returns
        -------
        Dict
            Evaluation results
        """
        if self.model is None:
            raise ValueError("Model not loaded. Run load_model() first.")
        
        print("\n" + "="*70)
        print(f"AD-SAL Evaluation - {dataset_name}")
        print("="*70)
        
        # Calculate AD indices
        ad_indices = self.compute_ad_indices(X_test, y_test)
        
        # Make predictions for all samples
        y_pred_all = self.model.predict(X_test)
        
        # Evaluate on all samples
        print("\n--- Performance on All Samples ---")
        metrics_all = calculate_metrics(y_test.values, y_pred_all)
        print_metrics_summary(metrics_all, "All Samples")
        
        # Filter by AD
        X_within, y_within, mask_within = self.apply_ad_filter(
            X_test, y_test, ad_indices, threshold
        )
        
        # Make predictions for samples within AD
        y_pred_within = self.model.predict(X_within)
        
        # Evaluate on samples within AD
        print("\n--- Performance Within AD ---")
        metrics_within = calculate_metrics(y_within.values, y_pred_within)
        print_metrics_summary(metrics_within, "Within AD")
        
        # Calculate improvement
        rmse_improvement = metrics_all['RMSE'] - metrics_within['RMSE']
        mae_improvement = metrics_all['MAE'] - metrics_within['MAE']
        r2_improvement = metrics_within['R2'] - metrics_all['R2']
        
        print("\n--- Performance Improvement ---")
        print(f"RMSE improvement: {rmse_improvement:.2f} nm ({rmse_improvement/metrics_all['RMSE']*100:.1f}%)")
        print(f"MAE improvement: {mae_improvement:.2f} nm ({mae_improvement/metrics_all['MAE']*100:.1f}%)")
        print(f"R² improvement: {r2_improvement:.4f}")
        
        results = {
            'threshold': threshold,
            'n_total': len(X_test),
            'n_within_ad': mask_within.sum(),
            'coverage': mask_within.sum() / len(X_test),
            'metrics_all': metrics_all,
            'metrics_within_ad': metrics_within,
            'rmse_improvement': rmse_improvement,
            'mae_improvement': mae_improvement,
            'r2_improvement': r2_improvement,
            'ad_indices': ad_indices
        }
        
        return results
    
    def test_multiple_alpha_values(self, X_test: pd.DataFrame, y_test: pd.Series,
                                   alpha_values: list = [5, 10, 15, 20, 25],
                                   percentile: float = 95.0) -> pd.DataFrame:
        """
        Test multiple α values and compare performance.
        
        Parameters
        ----------
        X_test : pd.DataFrame
            Test feature matrix
        y_test : pd.Series
            Test target values
        alpha_values : list
            List of α values to test
        percentile : float
            Percentile for threshold determination
        
        Returns
        -------
        pd.DataFrame
            Comparison table
        """
        print("\n" + "="*70)
        print("Testing Multiple α Values")
        print("="*70)
        
        results_list = []
        
        for alpha in alpha_values:
            print(f"\n--- Testing α = {alpha} ---")
            
            # Set alpha
            self.alpha = alpha
            
            # Calculate AD indices for training set to determine threshold
            ad_indices_train = self.compute_ad_indices(
                pd.DataFrame(self.X_train), 
                pd.Series(self.y_train)
            )
            threshold = self.determine_ad_threshold(ad_indices_train, percentile)
            
            # Evaluate on test set
            results = self.evaluate_with_ad(X_test, y_test, threshold, f"Test Set (α={alpha})")
            
            results_list.append({
                'alpha': alpha,
                'Threshold': f"{threshold:.4f}",
                'Coverage (%)': f"{results['coverage']*100:.1f}",
                'N_within_AD': results['n_within_ad'],
                'RMSE_all': f"{results['metrics_all']['RMSE']:.2f}",
                'RMSE_within': f"{results['metrics_within_ad']['RMSE']:.2f}",
                'RMSE_improvement': f"{results['rmse_improvement']:.2f}",
                'MAE_all': f"{results['metrics_all']['MAE']:.2f}",
                'MAE_within': f"{results['metrics_within_ad']['MAE']:.2f}",
                'MAE_improvement': f"{results['mae_improvement']:.2f}",
                'R2_all': f"{results['metrics_all']['R2']:.4f}",
                'R2_within': f"{results['metrics_within_ad']['R2']:.4f}",
                'R2_improvement': f"{results['r2_improvement']:.4f}"
            })
        
        comparison_df = pd.DataFrame(results_list)
        
        print("\n" + "="*70)
        print("Comparison of Different α Values")
        print("="*70)
        print(comparison_df.to_string(index=False))
        
        return comparison_df
    
    def plot_ad_distribution(self, ad_indices: pd.DataFrame, 
                            threshold: float,
                            save_path: Optional[Path] = None):
        """
        Plot AD indices distribution.
        
        Parameters
        ----------
        ad_indices : pd.DataFrame
            AD indices
        threshold : float
            AD threshold
        save_path : Path, optional
            Path to save figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Density distribution
        axes[0].hist(ad_indices['density'], bins=30, edgecolor='black', alpha=0.7)
        axes[0].axvline(threshold, color='r', linestyle='--', linewidth=2, label=f'Threshold = {threshold:.4f}')
        axes[0].set_xlabel('Density Index (ρ)', fontsize=12)
        axes[0].set_ylabel('Count', fontsize=12)
        axes[0].set_title('Distribution of Similarity Density', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Max similarity distribution
        axes[1].hist(ad_indices['max_similarity'], bins=30, edgecolor='black', alpha=0.7)
        axes[1].set_xlabel('Max Similarity', fontsize=12)
        axes[1].set_ylabel('Count', fontsize=12)
        axes[1].set_title('Distribution of Maximum Similarity', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
            print(f"AD distribution plot saved to: {save_path}")
        
        plt.show()
        plt.close()
    
    def apply_optimal_ad(self, X_test: pd.DataFrame, y_test: pd.Series,
                        alpha: float = 15.0,
                        percentile: float = 95.0,
                        save_dir: Optional[Path] = None) -> Dict:
        """
        Apply optimal AD filtering and save results.
        
        Parameters
        ----------
        X_test : pd.DataFrame
            Test feature matrix
        y_test : pd.Series
            Test target values
        alpha : float
            Optimal α value (default: 15)
        percentile : float
            Percentile for threshold (default: 95)
        save_dir : Path, optional
            Directory to save results
        
        Returns
        -------
        Dict
            Complete AD analysis results
        """
        if save_dir is None:
            save_dir = config.RESULTS_DIR
        
        print("\n" + "="*70)
        print(f"Applying Optimal AD-SAL (α={alpha})")
        print("="*70)
        
        # Set alpha
        self.alpha = alpha
        
        # Calculate AD indices for training set to determine threshold
        ad_indices_train = self.compute_ad_indices(
            pd.DataFrame(self.X_train), 
            pd.Series(self.y_train)
        )
        threshold = self.determine_ad_threshold(ad_indices_train, percentile)
        
        # Evaluate on test set
        results = self.evaluate_with_ad(X_test, y_test, threshold, "Test Set")
        
        # Plot distribution
        self.plot_ad_distribution(
            results['ad_indices'], 
            threshold,
            save_path=save_dir / "ad_distribution.png"
        )
        
        # Save results
        results_summary = {
            'alpha': alpha,
            'percentile': percentile,
            'threshold': threshold,
            'n_total': results['n_total'],
            'n_within_ad': results['n_within_ad'],
            'coverage': results['coverage'],
            'metrics_all': results['metrics_all'],
            'metrics_within_ad': results['metrics_within_ad']
        }
        
        results_path = save_dir / "ad_sal_results.json"
        with open(results_path, 'w') as f:
            json.dump(results_summary, f, indent=4)
        print(f"\nAD-SAL results saved to: {results_path}")
        
        # Save AD indices
        ad_indices_path = save_dir / "ad_indices.csv"
        results['ad_indices'].to_csv(ad_indices_path, index=False)
        print(f"AD indices saved to: {ad_indices_path}")
        
        # Save predictions with AD labels
        y_pred = self.model.predict(X_test)
        mask_within = results['ad_indices']['density'] >= threshold
        
        predictions_df = pd.DataFrame({
            'actual': y_test.values,
            'predicted': y_pred,
            'density': results['ad_indices']['density'].values,
            'discontinuity': results['ad_indices']['discontinuity'].values,
            'within_ad': mask_within,
            'residual': y_test.values - y_pred
        })
        predictions_path = save_dir / "predictions_with_ad.csv"
        predictions_df.to_csv(predictions_path, index=False)
        print(f"Predictions with AD labels saved to: {predictions_path}")
        
        return results


def perform_ad_analysis(model_path: Path,
                       X_train: pd.DataFrame,
                       y_train: pd.Series,
                       X_test: pd.DataFrame,
                       y_test: pd.Series,
                       test_alpha_values: bool = True) -> Dict:
    """
    Perform complete AD-SAL analysis.
    
    Parameters
    ----------
    model_path : Path
        Path to trained model
    X_train : pd.DataFrame
        Training feature matrix
    y_train : pd.Series
        Training target values
    X_test : pd.DataFrame
        Test feature matrix
    y_test : pd.Series
        Test target values
    test_alpha_values : bool
        Whether to test multiple α values
    
    Returns
    -------
    Dict
        Complete AD analysis results
    """
    print("\n" + "="*70)
    print("AD-SAL APPLICABILITY DOMAIN ANALYSIS")
    print("="*70)
    
    # Create analyzer
    analyzer = ADSALAnalyzer(model_path, alpha=15.0)
    analyzer.load_model()
    analyzer.set_training_data(X_train, y_train)
    
    # Test multiple α values if requested
    if test_alpha_values:
        print("\n--- Testing Multiple α Values ---")
        comparison_df = analyzer.test_multiple_alpha_values(
            X_test, y_test,
            alpha_values=[5, 10, 15, 20, 25],
            percentile=95.0
        )
        
        # Save comparison
        comparison_path = config.RESULTS_DIR / "ad_alpha_comparison.csv"
        comparison_df.to_csv(comparison_path, index=False)
        print(f"\nα value comparison saved to: {comparison_path}")
    
    # Apply optimal AD (α=15)
    print("\n--- Applying Optimal AD (α=15) ---")
    results = analyzer.apply_optimal_ad(
        X_test, y_test,
        alpha=15.0,
        percentile=95.0,
        save_dir=config.RESULTS_DIR
    )
    
    print("\n" + "="*70)
    print("AD-SAL ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nRecommended α value: 15")
    print(f"Coverage: {results['coverage']*100:.1f}%")
    print(f"RMSE improvement: {results['rmse_improvement']:.2f} nm")
    
    return results


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
    
    # Split into train/test
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        stratify=is_aqueous,
        random_state=config.RANDOM_STATE
    )
    
    # Perform AD analysis
    results = perform_ad_analysis(
        model_path=model_path,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        test_alpha_values=True  # Test α=5,10,15,20,25
    )
    
    print("\nAD-SAL analysis complete!")
    print(f"Results saved to: {config.RESULTS_DIR}")
