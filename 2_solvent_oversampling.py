"""
Reliable DOM Fluorescence Prediction via Solvent Sensitive Machine Learning and Domain Refinement
==================================================================================================
Solvent-Directed Oversampling Module

This module implements the solvent-directed oversampling strategy to address
the imbalance between aqueous and non-aqueous measurements in photophysical datasets.

Key features:
- Aqueous data enhancement (XGB-A(11): 11x oversampling)
- Baseline model (no enhancement)
- Aqueous-focused test set construction
- Train-test split with stratification
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

import config


class SolventDirectedSampler:
    """
    Implements solvent-directed oversampling strategy for aqueous data enhancement.
    """
    
    def __init__(self, enhancement_factor: int = config.AQUEOUS_ENHANCEMENT_FACTOR,
                 random_state: int = config.RANDOM_STATE):
        """
        Initialize the sampler.
        
        Parameters
        ----------
        enhancement_factor : int
            Multiplication factor for aqueous samples (11 for XGB-A(11))
        random_state : int
            Random seed for reproducibility
        """
        self.enhancement_factor = enhancement_factor
        self.random_state = random_state
        
    def split_data(self, X: pd.DataFrame, y: pd.Series, 
                   is_aqueous: pd.Series,
                   test_size: float = config.TEST_SIZE) -> Tuple:
        """
        Split data into train and test sets with stratification.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target variable
        is_aqueous : pd.Series
            Boolean mask indicating aqueous samples
        test_size : float
            Proportion of data for testing
        
        Returns
        -------
        Tuple
            (X_train, X_test, y_train, y_test, 
             is_aqueous_train, is_aqueous_test)
        """
        print("\n" + "="*60)
        print("Splitting Data into Train and Test Sets")
        print("="*60)
        
        # Stratified split to maintain aqueous/non-aqueous ratio
        X_train, X_test, y_train, y_test, aq_train, aq_test = train_test_split(
            X, y, is_aqueous,
            test_size=test_size,
            stratify=is_aqueous,
            random_state=self.random_state
        )
        
        print(f"\nTrain set: {len(X_train)} samples")
        print(f"  - Aqueous: {aq_train.sum()} ({aq_train.sum()/len(aq_train)*100:.1f}%)")
        print(f"  - Non-aqueous: {(~aq_train).sum()} ({(~aq_train).sum()/len(aq_train)*100:.1f}%)")
        
        print(f"\nTest set: {len(X_test)} samples")
        print(f"  - Aqueous: {aq_test.sum()} ({aq_test.sum()/len(aq_test)*100:.1f}%)")
        print(f"  - Non-aqueous: {(~aq_test).sum()} ({(~aq_test).sum()/len(aq_test)*100:.1f}%)")
        
        return X_train, X_test, y_train, y_test, aq_train, aq_test
    
    def oversample_aqueous(self, X_train: pd.DataFrame, 
                          y_train: pd.Series,
                          is_aqueous_train: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Apply solvent-directed oversampling to aqueous training data.
        
        Parameters
        ----------
        X_train : pd.DataFrame
            Training feature matrix
        y_train : pd.Series
            Training target variable
        is_aqueous_train : pd.Series
            Boolean mask for aqueous samples in training set
        
        Returns
        -------
        Tuple[pd.DataFrame, pd.Series]
            (X_train_enhanced, y_train_enhanced) - Enhanced training data
        """
        print("\n" + "="*60)
        print(f"Applying Solvent-Directed Oversampling (Factor: {self.enhancement_factor})")
        print("="*60)
        
        # Separate aqueous and non-aqueous samples
        X_aqueous = X_train[is_aqueous_train]
        y_aqueous = y_train[is_aqueous_train]
        
        X_non_aqueous = X_train[~is_aqueous_train]
        y_non_aqueous = y_train[~is_aqueous_train]
        
        print(f"\nOriginal training set:")
        print(f"  - Aqueous: {len(X_aqueous)} samples")
        print(f"  - Non-aqueous: {len(X_non_aqueous)} samples")
        print(f"  - Ratio (aqueous/non-aqueous): {len(X_aqueous)/len(X_non_aqueous):.2f}")
        
        if self.enhancement_factor > 1:
            # Oversample aqueous data with replacement
            n_samples = self.enhancement_factor * len(X_aqueous)
            
            X_aqueous_enhanced = X_aqueous.sample(
                n=n_samples, 
                replace=True, 
                random_state=self.random_state
            )
            y_aqueous_enhanced = y_aqueous.loc[X_aqueous_enhanced.index]
            
            # Combine enhanced aqueous with original non-aqueous
            X_train_enhanced = pd.concat([X_aqueous_enhanced, X_non_aqueous], axis=0)
            y_train_enhanced = pd.concat([y_aqueous_enhanced, y_non_aqueous], axis=0)
            
            # Shuffle
            shuffle_idx = np.random.RandomState(self.random_state).permutation(len(X_train_enhanced))
            X_train_enhanced = X_train_enhanced.iloc[shuffle_idx].reset_index(drop=True)
            y_train_enhanced = y_train_enhanced.iloc[shuffle_idx].reset_index(drop=True)
            
            print(f"\nEnhanced training set:")
            print(f"  - Aqueous: {len(X_aqueous_enhanced)} samples (enhanced)")
            print(f"  - Non-aqueous: {len(X_non_aqueous)} samples (original)")
            print(f"  - Total: {len(X_train_enhanced)} samples")
            print(f"  - New ratio (aqueous/non-aqueous): {len(X_aqueous_enhanced)/len(X_non_aqueous):.2f}")
            
        else:
            # No enhancement (baseline model)
            X_train_enhanced = X_train.copy()
            y_train_enhanced = y_train.copy()
            print("\nNo enhancement applied (baseline model)")
        
        return X_train_enhanced, y_train_enhanced
    
    def create_aqueous_focused_test_set(self, X_test: pd.DataFrame,
                                       y_test: pd.Series,
                                       is_aqueous_test: pd.Series) -> Tuple:
        """
        Create an aqueous-focused test set for targeted evaluation.
        
        Parameters
        ----------
        X_test : pd.DataFrame
            Test feature matrix
        y_test : pd.Series
            Test target variable
        is_aqueous_test : pd.Series
            Boolean mask for aqueous samples in test set
        
        Returns
        -------
        Tuple
            (X_test_aqueous, y_test_aqueous) - Aqueous-only test set
        """
        X_test_aqueous = X_test[is_aqueous_test]
        y_test_aqueous = y_test[is_aqueous_test]
        
        print("\n" + "="*60)
        print("Aqueous-Focused Test Set")
        print("="*60)
        print(f"Aqueous test samples: {len(X_test_aqueous)}")
        print(f"Target range: [{y_test_aqueous.min():.1f}, {y_test_aqueous.max():.1f}] nm")
        
        return X_test_aqueous, y_test_aqueous
    
    def prepare_datasets(self, X: pd.DataFrame, y: pd.Series,
                        is_aqueous: pd.Series,
                        model_type: str = "enhanced") -> dict:
        """
        Complete pipeline to prepare train/test datasets with optional enhancement.
        
        Parameters
        ----------
        X : pd.DataFrame
            Full feature matrix
        y : pd.Series
            Full target variable
        is_aqueous : pd.Series
            Boolean mask for aqueous samples
        model_type : str
            "enhanced" for XGB-A(11) or "baseline" for no enhancement
        
        Returns
        -------
        dict
            Dictionary containing all dataset splits:
            - X_train, y_train: Enhanced/baseline training data
            - X_test, y_test: Full test set
            - X_test_aqueous, y_test_aqueous: Aqueous-focused test set
            - is_aqueous_test: Aqueous mask for test set
        """
        print("\n" + "="*70)
        print(f"Preparing Datasets for {model_type.upper()} Model")
        print("="*70)
        
        # Set enhancement factor based on model type
        if model_type.lower() == "baseline":
            self.enhancement_factor = config.BASELINE_ENHANCEMENT_FACTOR
        elif model_type.lower() == "enhanced":
            self.enhancement_factor = config.AQUEOUS_ENHANCEMENT_FACTOR
        else:
            raise ValueError(f"Unknown model_type: {model_type}. Use 'baseline' or 'enhanced'")
        
        # Split data
        X_train, X_test, y_train, y_test, aq_train, aq_test = self.split_data(
            X, y, is_aqueous
        )
        
        # Apply oversampling
        X_train_enhanced, y_train_enhanced = self.oversample_aqueous(
            X_train, y_train, aq_train
        )
        
        # Create aqueous-focused test set
        X_test_aqueous, y_test_aqueous = self.create_aqueous_focused_test_set(
            X_test, y_test, aq_test
        )
        
        datasets = {
            "X_train": X_train_enhanced,
            "y_train": y_train_enhanced,
            "X_test": X_test,
            "y_test": y_test,
            "X_test_aqueous": X_test_aqueous,
            "y_test_aqueous": y_test_aqueous,
            "is_aqueous_test": aq_test
        }
        
        print("\n" + "="*70)
        print("Dataset Preparation Complete")
        print("="*70)
        
        return datasets


def compare_sampling_strategies(X: pd.DataFrame, y: pd.Series, 
                               is_aqueous: pd.Series) -> dict:
    """
    Compare baseline and enhanced sampling strategies.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target variable
    is_aqueous : pd.Series
        Boolean mask for aqueous samples
    
    Returns
    -------
    dict
        Dictionary with baseline and enhanced datasets
    """
    print("\n" + "="*70)
    print("Comparing Sampling Strategies")
    print("="*70)
    
    # Baseline model (no enhancement)
    baseline_sampler = SolventDirectedSampler(enhancement_factor=1)
    baseline_datasets = baseline_sampler.prepare_datasets(X, y, is_aqueous, "baseline")
    
    # Enhanced model (XGB-A(11))
    enhanced_sampler = SolventDirectedSampler(enhancement_factor=11)
    enhanced_datasets = enhanced_sampler.prepare_datasets(X, y, is_aqueous, "enhanced")
    
    return {
        "baseline": baseline_datasets,
        "enhanced": enhanced_datasets
    }


if __name__ == "__main__":
    # Example usage
    from pathlib import Path
    
    # Load processed data
    processed_path = config.DATA_DIR / "processed_data.csv"
    
    if processed_path.exists():
        print("Loading processed data...")
        df = pd.read_csv(processed_path)
        
        # Separate features, target, and aqueous mask
        X = df.drop(columns=[config.TARGET_COLUMN, 'is_aqueous'])
        y = df[config.TARGET_COLUMN]
        is_aqueous = df['is_aqueous']
        
        # Prepare datasets for enhanced model
        sampler = SolventDirectedSampler(enhancement_factor=11)
        datasets = sampler.prepare_datasets(X, y, is_aqueous, model_type="enhanced")
        
        print("\n" + "="*70)
        print("Dataset Summary")
        print("="*70)
        print(f"Training set size: {len(datasets['X_train'])}")
        print(f"Full test set size: {len(datasets['X_test'])}")
        print(f"Aqueous test set size: {len(datasets['X_test_aqueous'])}")
        
        # Compare strategies
        comparison = compare_sampling_strategies(X, y, is_aqueous)
        
        print("\n" + "="*70)
        print("Training Set Size Comparison")
        print("="*70)
        print(f"Baseline: {len(comparison['baseline']['X_train'])} samples")
        print(f"Enhanced (XGB-A(11)): {len(comparison['enhanced']['X_train'])} samples")
        print(f"Increase: {len(comparison['enhanced']['X_train']) - len(comparison['baseline']['X_train'])} samples")
        
    else:
        print(f"Processed data not found: {processed_path}")
        print("Please run 1_data_preprocessing.py first")
