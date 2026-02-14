"""
Reliable DOM Fluorescence Prediction via Solvent Sensitive Machine Learning and Domain Refinement
==================================================================================================
SHAP Feature Analysis Module

This module performs SHAP (SHapley Additive exPlanations) analysis to interpret
the XGBoost model's predictions, focusing on aqueous solvent samples.

Key features:
1. SHAP value calculation for aqueous samples
2. Feature importance ranking (top 20)
3. Beeswarm plot visualization
4. Bar plot visualization
"""

import pandas as pd
import numpy as np
from typing import Optional
import warnings
warnings.filterwarnings('ignore')

import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from pathlib import Path

import config


class SHAPAnalyzer:
    """
    SHAP-based model interpretation for aqueous DOM fluorescence prediction.
    """
    
    def __init__(self, model_path: Optional[Path] = None):
        """
        Initialize SHAP analyzer.
        
        Parameters
        ----------
        model_path : Path, optional
            Path to trained XGBoost model
        """
        self.model_path = model_path
        self.model = None
        self.explainer = None
        self.shap_values = None
        
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
    
    def calculate_shap_values(self, X_aqueous: pd.DataFrame,
                             background_samples: Optional[int] = None) -> np.ndarray:
        """
        Calculate SHAP values for aqueous samples.
        
        Parameters
        ----------
        X_aqueous : pd.DataFrame
            Feature matrix for aqueous samples
        background_samples : int, optional
            Number of background samples for SHAP explainer
        
        Returns
        -------
        np.ndarray
            SHAP values
        """
        if self.model is None:
            raise ValueError("Model not loaded. Run load_model() first.")
        
        print("\n" + "="*70)
        print("Calculating SHAP Values for Aqueous Samples")
        print("="*70)
        print(f"Number of aqueous samples: {len(X_aqueous)}")
        
        # Create SHAP explainer
        if background_samples is None:
            background_samples = min(config.SHAP_SAMPLE_SIZE, len(X_aqueous))
        
        print(f"Using {background_samples} background samples for SHAP explainer")
        
        # Sample background data
        if len(X_aqueous) > background_samples:
            background_data = X_aqueous.sample(n=background_samples, random_state=config.RANDOM_STATE)
        else:
            background_data = X_aqueous
        
        # Create TreeExplainer
        print("Creating SHAP TreeExplainer...")
        self.explainer = shap.TreeExplainer(self.model, background_data)
        
        # Calculate SHAP values
        print("Computing SHAP values...")
        self.shap_values = self.explainer.shap_values(X_aqueous)
        
        print(f"SHAP values shape: {self.shap_values.shape}")
        print("SHAP calculation complete")
        
        return self.shap_values
    
    def get_top_features(self, X_aqueous: pd.DataFrame, 
                        top_n: int = 20) -> pd.DataFrame:
        """
        Get top N most important features based on mean absolute SHAP values.
        
        Parameters
        ----------
        X_aqueous : pd.DataFrame
            Feature matrix for aqueous samples
        top_n : int
            Number of top features to return
        
        Returns
        -------
        pd.DataFrame
            Top features with their importance scores
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not calculated. Run calculate_shap_values() first.")
        
        # Calculate mean absolute SHAP values
        mean_abs_shap = np.abs(self.shap_values).mean(axis=0)
        
        # Create DataFrame
        feature_importance = pd.DataFrame({
            'feature': X_aqueous.columns,
            'importance': mean_abs_shap
        })
        
        # Sort by importance
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        
        # Get top N
        top_features = feature_importance.head(top_n)
        
        print("\n" + "="*70)
        print(f"Top {top_n} Most Important Features (Aqueous Samples)")
        print("="*70)
        for idx, row in top_features.iterrows():
            print(f"{row['feature']:40s} {row['importance']:.6f}")
        print("="*70)
        
        return top_features
    
    def plot_beeswarm(self, X_aqueous: pd.DataFrame,
                     top_n: int = 20,
                     save_path: Optional[Path] = None):
        """
        Generate SHAP beeswarm plot for top N features.
        
        Parameters
        ----------
        X_aqueous : pd.DataFrame
            Feature matrix for aqueous samples
        top_n : int
            Number of top features to display
        save_path : Path, optional
            Path to save figure
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not calculated. Run calculate_shap_values() first.")
        
        print(f"\nGenerating SHAP beeswarm plot (top {top_n} features)...")
        
        # Get top features
        top_features = self.get_top_features(X_aqueous, top_n)
        top_feature_names = top_features['feature'].tolist()
        
        # Get indices of top features
        feature_indices = [X_aqueous.columns.get_loc(f) for f in top_feature_names]
        
        # Filter SHAP values and features
        shap_values_filtered = self.shap_values[:, feature_indices]
        X_filtered = X_aqueous[top_feature_names]
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Generate beeswarm plot
        shap.summary_plot(
            shap_values_filtered,
            X_filtered,
            plot_type="dot",
            max_display=top_n,
            show=False
        )
        
        plt.title(f'SHAP Beeswarm Plot - Top {top_n} Features (Aqueous Samples)', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('SHAP Value (impact on model output)', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
            print(f"Beeswarm plot saved to: {save_path}")
        
        plt.show()
        plt.close()
    
    def plot_bar(self, X_aqueous: pd.DataFrame,
                top_n: int = 20,
                save_path: Optional[Path] = None):
        """
        Generate SHAP bar plot for top N features.
        
        Parameters
        ----------
        X_aqueous : pd.DataFrame
            Feature matrix for aqueous samples
        top_n : int
            Number of top features to display
        save_path : Path, optional
            Path to save figure
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not calculated. Run calculate_shap_values() first.")
        
        print(f"\nGenerating SHAP bar plot (top {top_n} features)...")
        
        # Get top features
        top_features = self.get_top_features(X_aqueous, top_n)
        top_feature_names = top_features['feature'].tolist()
        
        # Get indices of top features
        feature_indices = [X_aqueous.columns.get_loc(f) for f in top_feature_names]
        
        # Filter SHAP values and features
        shap_values_filtered = self.shap_values[:, feature_indices]
        X_filtered = X_aqueous[top_feature_names]
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Generate bar plot
        shap.summary_plot(
            shap_values_filtered,
            X_filtered,
            plot_type="bar",
            max_display=top_n,
            show=False
        )
        
        plt.title(f'SHAP Feature Importance - Top {top_n} Features (Aqueous Samples)', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('Mean |SHAP Value|', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
            print(f"Bar plot saved to: {save_path}")
        
        plt.show()
        plt.close()
    
    def generate_shap_analysis(self, X_aqueous: pd.DataFrame,
                              top_n: int = 20,
                              save_dir: Optional[Path] = None) -> pd.DataFrame:
        """
        Complete SHAP analysis pipeline for aqueous samples.
        
        Parameters
        ----------
        X_aqueous : pd.DataFrame
            Feature matrix for aqueous samples
        top_n : int
            Number of top features to analyze
        save_dir : Path, optional
            Directory to save results
        
        Returns
        -------
        pd.DataFrame
            Top features with importance scores
        """
        if save_dir is None:
            save_dir = config.FIGURES_DIR
        
        print("\n" + "="*70)
        print("SHAP ANALYSIS FOR AQUEOUS SAMPLES")
        print("="*70)
        
        # Calculate SHAP values
        self.calculate_shap_values(X_aqueous)
        
        # Get top features
        top_features = self.get_top_features(X_aqueous, top_n)
        
        # Save feature importance
        importance_path = save_dir / "shap_feature_importance.csv"
        top_features.to_csv(importance_path, index=False)
        print(f"\nFeature importance saved to: {importance_path}")
        
        # Generate beeswarm plot
        beeswarm_path = save_dir / "shap_beeswarm_plot.png"
        self.plot_beeswarm(X_aqueous, top_n, beeswarm_path)
        
        # Generate bar plot
        bar_path = save_dir / "shap_bar_plot.png"
        self.plot_bar(X_aqueous, top_n, bar_path)
        
        print("\n" + "="*70)
        print("SHAP ANALYSIS COMPLETE")
        print("="*70)
        
        return top_features
    
    def save_shap_values(self, X_aqueous: pd.DataFrame, 
                        save_path: Optional[Path] = None):
        """
        Save SHAP values to CSV file.
        
        Parameters
        ----------
        X_aqueous : pd.DataFrame
            Feature matrix for aqueous samples
        save_path : Path, optional
            Path to save SHAP values
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not calculated. Run calculate_shap_values() first.")
        
        if save_path is None:
            save_path = config.RESULTS_DIR / "shap_values.csv"
        
        # Create DataFrame with SHAP values
        shap_df = pd.DataFrame(
            self.shap_values,
            columns=[f"SHAP_{col}" for col in X_aqueous.columns]
        )
        
        shap_df.to_csv(save_path, index=False)
        print(f"\nSHAP values saved to: {save_path}")


def analyze_aqueous_samples(model_path: Path, 
                           X: pd.DataFrame, 
                           is_aqueous: pd.Series,
                           top_n: int = 20) -> pd.DataFrame:
    """
    Perform SHAP analysis on all aqueous samples.
    
    Parameters
    ----------
    model_path : Path
        Path to trained model
    X : pd.DataFrame
        Full feature matrix
    is_aqueous : pd.Series
        Boolean mask for aqueous samples
    top_n : int
        Number of top features to analyze
    
    Returns
    -------
    pd.DataFrame
        Top features with importance scores
    """
    # Filter aqueous samples
    X_aqueous = X[is_aqueous]
    
    print(f"\nTotal samples: {len(X)}")
    print(f"Aqueous samples: {len(X_aqueous)} ({len(X_aqueous)/len(X)*100:.1f}%)")
    
    # Create analyzer
    analyzer = SHAPAnalyzer(model_path)
    analyzer.load_model()
    
    # Run analysis
    top_features = analyzer.generate_shap_analysis(
        X_aqueous, 
        top_n=top_n,
        save_dir=config.FIGURES_DIR
    )
    
    # Save SHAP values
    analyzer.save_shap_values(X_aqueous)
    
    return top_features


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
    is_aqueous = df['is_aqueous']
    
    # Perform SHAP analysis on all aqueous samples
    top_features = analyze_aqueous_samples(
        model_path=model_path,
        X=X,
        is_aqueous=is_aqueous,
        top_n=20
    )
    
    print("\nSHAP analysis complete!")
    print(f"Results saved to: {config.FIGURES_DIR}")
