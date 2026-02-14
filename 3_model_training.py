"""
Reliable DOM Fluorescence Prediction via Solvent Sensitive Machine Learning and Domain Refinement
==================================================================================================
Model Training Module

This module implements:
1. XGBoost model training with Optuna hyperparameter optimization
2. Baseline model training (no aqueous enhancement)
3. Enhanced model training (XGB-A(11) with 11x aqueous oversampling)
4. Model saving and loading functionality
5. Cross-validation and performance tracking
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

import xgboost as xgb
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import json
from pathlib import Path

import config
from utils.metrics import calculate_metrics, print_metrics_summary


class XGBoostTrainer:
    """
    XGBoost model trainer with Optuna hyperparameter optimization.
    """
    
    def __init__(self, model_name: str = "xgboost_model",
                 random_state: int = config.RANDOM_STATE):
        """
        Initialize the trainer.
        
        Parameters
        ----------
        model_name : str
            Name for the model (used for saving)
        random_state : int
            Random seed for reproducibility
        """
        self.model_name = model_name
        self.random_state = random_state
        self.model = None
        self.best_params = None
        self.study = None
        
    def objective(self, trial: optuna.Trial, X_train: pd.DataFrame, 
                  y_train: pd.Series) -> float:
        """
        Optuna objective function for hyperparameter optimization.
        
        Parameters
        ----------
        trial : optuna.Trial
            Optuna trial object
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training target
        
        Returns
        -------
        float
            Cross-validation RMSE score
        """
        # Sample hyperparameters
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 
                                             config.XGBOOST_PARAM_SPACE['n_estimators'][0],
                                             config.XGBOOST_PARAM_SPACE['n_estimators'][1]),
            'max_depth': trial.suggest_int('max_depth',
                                          config.XGBOOST_PARAM_SPACE['max_depth'][0],
                                          config.XGBOOST_PARAM_SPACE['max_depth'][1]),
            'learning_rate': trial.suggest_float('learning_rate',
                                                config.XGBOOST_PARAM_SPACE['learning_rate'][0],
                                                config.XGBOOST_PARAM_SPACE['learning_rate'][1],
                                                log=True),
            'subsample': trial.suggest_float('subsample',
                                            config.XGBOOST_PARAM_SPACE['subsample'][0],
                                            config.XGBOOST_PARAM_SPACE['subsample'][1]),
            'colsample_bytree': trial.suggest_float('colsample_bytree',
                                                   config.XGBOOST_PARAM_SPACE['colsample_bytree'][0],
                                                   config.XGBOOST_PARAM_SPACE['colsample_bytree'][1]),
            'min_child_weight': trial.suggest_int('min_child_weight',
                                                 config.XGBOOST_PARAM_SPACE['min_child_weight'][0],
                                                 config.XGBOOST_PARAM_SPACE['min_child_weight'][1]),
            'gamma': trial.suggest_float('gamma',
                                        config.XGBOOST_PARAM_SPACE['gamma'][0],
                                        config.XGBOOST_PARAM_SPACE['gamma'][1]),
            'reg_alpha': trial.suggest_float('reg_alpha',
                                            config.XGBOOST_PARAM_SPACE['reg_alpha'][0],
                                            config.XGBOOST_PARAM_SPACE['reg_alpha'][1]),
            'reg_lambda': trial.suggest_float('reg_lambda',
                                             config.XGBOOST_PARAM_SPACE['reg_lambda'][0],
                                             config.XGBOOST_PARAM_SPACE['reg_lambda'][1]),
        }
        
        # Add fixed parameters
        params.update(config.XGBOOST_FIXED_PARAMS)
        
        # Create model
        model = xgb.XGBRegressor(**params)
        
        # Cross-validation
        kfold = KFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=self.random_state)
        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=kfold,
            scoring='neg_root_mean_squared_error',
            n_jobs=config.N_JOBS
        )
        
        # Return mean RMSE (negative because sklearn uses negative scores)
        return -cv_scores.mean()
    
    def optimize_hyperparameters(self, X_train: pd.DataFrame, 
                                y_train: pd.Series,
                                n_trials: int = config.N_TRIALS) -> Dict:
        """
        Optimize hyperparameters using Optuna.
        
        Parameters
        ----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training target
        n_trials : int
            Number of optimization trials
        
        Returns
        -------
        Dict
            Best hyperparameters
        """
        print("\n" + "="*70)
        print(f"Optimizing Hyperparameters with Optuna ({n_trials} trials)")
        print("="*70)
        
        # Create study
        sampler = TPESampler(seed=self.random_state)
        self.study = optuna.create_study(
            direction='minimize',
            sampler=sampler,
            study_name=self.model_name
        )
        
        # Optimize
        self.study.optimize(
            lambda trial: self.objective(trial, X_train, y_train),
            n_trials=n_trials,
            show_progress_bar=True
        )
        
        self.best_params = self.study.best_params
        self.best_params.update(config.XGBOOST_FIXED_PARAMS)
        
        print(f"\nBest CV RMSE: {self.study.best_value:.4f}")
        print("\nBest Hyperparameters:")
        for param, value in self.best_params.items():
            if param not in config.XGBOOST_FIXED_PARAMS:
                print(f"  {param}: {value}")
        
        return self.best_params
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              params: Optional[Dict] = None) -> xgb.XGBRegressor:
        """
        Train XGBoost model with given or optimized parameters.
        
        Parameters
        ----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training target
        params : Dict, optional
            Model parameters. If None, uses optimized parameters
        
        Returns
        -------
        xgb.XGBRegressor
            Trained model
        """
        print("\n" + "="*70)
        print("Training XGBoost Model")
        print("="*70)
        
        if params is None:
            if self.best_params is None:
                raise ValueError("No parameters provided. Run optimize_hyperparameters first.")
            params = self.best_params
        
        # Create and train model
        self.model = xgb.XGBRegressor(**params)
        self.model.fit(X_train, y_train)
        
        # Training performance
        y_train_pred = self.model.predict(X_train)
        train_metrics = calculate_metrics(y_train.values, y_train_pred)
        
        print_metrics_summary(train_metrics, "Training Set")
        
        return self.model
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series,
                dataset_name: str = "Test Set") -> Dict[str, float]:
        """
        Evaluate model on test set.
        
        Parameters
        ----------
        X_test : pd.DataFrame
            Test features
        y_test : pd.Series
            Test target
        dataset_name : str
            Name of the dataset for display
        
        Returns
        -------
        Dict[str, float]
            Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained. Run train() first.")
        
        y_pred = self.model.predict(X_test)
        metrics = calculate_metrics(y_test.values, y_pred)
        
        print_metrics_summary(metrics, dataset_name)
        
        return metrics
    
    def save_model(self, save_path: Optional[Path] = None):
        """
        Save trained model to JSON format.
        
        Parameters
        ----------
        save_path : Path, optional
            Path to save model. If None, uses default from config
        """
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")
        
        if save_path is None:
            save_path = config.MODEL_DIR / f"{self.model_name}.json"
        
        self.model.save_model(str(save_path))
        print(f"\nModel saved to: {save_path}")
        
        # Save hyperparameters
        params_path = save_path.parent / f"{self.model_name}_params.json"
        with open(params_path, 'w') as f:
            json.dump(self.best_params, f, indent=4)
        print(f"Parameters saved to: {params_path}")
    
    def load_model(self, load_path: Path) -> xgb.XGBRegressor:
        """
        Load trained model from JSON format.
        
        Parameters
        ----------
        load_path : Path
            Path to model file
        
        Returns
        -------
        xgb.XGBRegressor
            Loaded model
        """
        self.model = xgb.XGBRegressor()
        self.model.load_model(str(load_path))
        print(f"Model loaded from: {load_path}")
        
        return self.model


def train_baseline_model(X_train: pd.DataFrame, y_train: pd.Series,
                        X_test: pd.DataFrame, y_test: pd.Series,
                        X_test_aqueous: pd.DataFrame, y_test_aqueous: pd.Series,
                        optimize: bool = True) -> Tuple[xgb.XGBRegressor, Dict]:
    """
    Train baseline model (no aqueous enhancement).
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    X_test : pd.DataFrame
        Full test features
    y_test : pd.Series
        Full test target
    X_test_aqueous : pd.DataFrame
        Aqueous test features
    y_test_aqueous : pd.Series
        Aqueous test target
    optimize : bool
        Whether to optimize hyperparameters
    
    Returns
    -------
    Tuple[xgb.XGBRegressor, Dict]
        (trained_model, all_metrics)
    """
    print("\n" + "="*70)
    print("TRAINING BASELINE MODEL")
    print("="*70)
    
    trainer = XGBoostTrainer(model_name="baseline_model")
    
    # Optimize hyperparameters
    if optimize:
        trainer.optimize_hyperparameters(X_train, y_train)
    
    # Train model
    model = trainer.train(X_train, y_train)
    
    # Evaluate on full test set
    print("\n--- Full Test Set Evaluation ---")
    full_test_metrics = trainer.evaluate(X_test, y_test, "Full Test Set")
    
    # Evaluate on aqueous test set
    print("\n--- Aqueous Test Set Evaluation ---")
    aqueous_test_metrics = trainer.evaluate(X_test_aqueous, y_test_aqueous, 
                                           "Aqueous-Focused Test Set")
    
    # Save model
    trainer.save_model(config.BASELINE_MODEL_PATH)
    
    all_metrics = {
        "full_test": full_test_metrics,
        "aqueous_test": aqueous_test_metrics
    }
    
    return model, all_metrics


def train_enhanced_model(X_train: pd.DataFrame, y_train: pd.Series,
                        X_test: pd.DataFrame, y_test: pd.Series,
                        X_test_aqueous: pd.DataFrame, y_test_aqueous: pd.Series,
                        optimize: bool = True) -> Tuple[xgb.XGBRegressor, Dict]:
    """
    Train enhanced model (XGB-A(11) with 11x aqueous oversampling).
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Enhanced training features (with oversampling)
    y_train : pd.Series
        Enhanced training target
    X_test : pd.DataFrame
        Full test features
    y_test : pd.Series
        Full test target
    X_test_aqueous : pd.DataFrame
        Aqueous test features
    y_test_aqueous : pd.Series
        Aqueous test target
    optimize : bool
        Whether to optimize hyperparameters
    
    Returns
    -------
    Tuple[xgb.XGBRegressor, Dict]
        (trained_model, all_metrics)
    """
    print("\n" + "="*70)
    print("TRAINING ENHANCED MODEL (XGB-A(11))")
    print("="*70)
    
    trainer = XGBoostTrainer(model_name="xgb_a11_model")
    
    # Optimize hyperparameters
    if optimize:
        trainer.optimize_hyperparameters(X_train, y_train)
    
    # Train model
    model = trainer.train(X_train, y_train)
    
    # Evaluate on full test set
    print("\n--- Full Test Set Evaluation ---")
    full_test_metrics = trainer.evaluate(X_test, y_test, "Full Test Set")
    
    # Evaluate on aqueous test set
    print("\n--- Aqueous Test Set Evaluation ---")
    aqueous_test_metrics = trainer.evaluate(X_test_aqueous, y_test_aqueous,
                                           "Aqueous-Focused Test Set")
    
    # Save model
    trainer.save_model(config.ENHANCED_MODEL_PATH)
    
    all_metrics = {
        "full_test": full_test_metrics,
        "aqueous_test": aqueous_test_metrics
    }
    
    return model, all_metrics


def train_both_models(baseline_datasets: Dict, enhanced_datasets: Dict,
                     optimize: bool = True) -> Dict:
    """
    Train both baseline and enhanced models and compare performance.
    
    Parameters
    ----------
    baseline_datasets : Dict
        Datasets for baseline model
    enhanced_datasets : Dict
        Datasets for enhanced model
    optimize : bool
        Whether to optimize hyperparameters
    
    Returns
    -------
    Dict
        Dictionary containing both models and their metrics
    """
    print("\n" + "="*70)
    print("TRAINING BOTH BASELINE AND ENHANCED MODELS")
    print("="*70)
    
    # Train baseline model
    baseline_model, baseline_metrics = train_baseline_model(
        baseline_datasets['X_train'],
        baseline_datasets['y_train'],
        baseline_datasets['X_test'],
        baseline_datasets['y_test'],
        baseline_datasets['X_test_aqueous'],
        baseline_datasets['y_test_aqueous'],
        optimize=optimize
    )
    
    # Train enhanced model
    enhanced_model, enhanced_metrics = train_enhanced_model(
        enhanced_datasets['X_train'],
        enhanced_datasets['y_train'],
        enhanced_datasets['X_test'],
        enhanced_datasets['y_test'],
        enhanced_datasets['X_test_aqueous'],
        enhanced_datasets['y_test_aqueous'],
        optimize=optimize
    )
    
    # Compare performance
    print("\n" + "="*70)
    print("PERFORMANCE COMPARISON")
    print("="*70)
    
    from utils.metrics import compare_models
    
    print("\n--- Full Test Set ---")
    full_comparison = compare_models(
        baseline_metrics['full_test'],
        enhanced_metrics['full_test'],
        "Full Test Set"
    )
    print(full_comparison.to_string(index=False))
    
    print("\n--- Aqueous-Focused Test Set ---")
    aqueous_comparison = compare_models(
        baseline_metrics['aqueous_test'],
        enhanced_metrics['aqueous_test'],
        "Aqueous-Focused Test Set"
    )
    print(aqueous_comparison.to_string(index=False))
    
    # Save comparison results
    results_path = config.RESULTS_DIR / "model_comparison.csv"
    comparison_df = pd.concat([
        full_comparison.assign(Dataset="Full Test Set"),
        aqueous_comparison.assign(Dataset="Aqueous Test Set")
    ])
    comparison_df.to_csv(results_path, index=False)
    print(f"\nComparison results saved to: {results_path}")
    
    return {
        "baseline_model": baseline_model,
        "enhanced_model": enhanced_model,
        "baseline_metrics": baseline_metrics,
        "enhanced_metrics": enhanced_metrics
    }


if __name__ == "__main__":
    # Example usage
    from pathlib import Path
    import sys
    
    # Check if processed data exists
    processed_path = config.DATA_DIR / "processed_data.csv"
    
    if not processed_path.exists():
        print(f"Processed data not found: {processed_path}")
        print("Please run 1_data_preprocessing.py first")
        sys.exit(1)
    
    # Load processed data
    print("Loading processed data...")
    df = pd.read_csv(processed_path)
    
    X = df.drop(columns=[config.TARGET_COLUMN, 'is_aqueous'])
    y = df[config.TARGET_COLUMN]
    is_aqueous = df['is_aqueous']
    
    # Prepare datasets with oversampling
    from module_2_solvent_oversampling import compare_sampling_strategies
    
    datasets = compare_sampling_strategies(X, y, is_aqueous)
    
    # Train both models
    results = train_both_models(
        datasets['baseline'],
        datasets['enhanced'],
        optimize=True
    )
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Baseline model saved to: {config.BASELINE_MODEL_PATH}")
    print(f"Enhanced model saved to: {config.ENHANCED_MODEL_PATH}")
