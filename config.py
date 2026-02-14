"""
Reliable DOM Fluorescence Prediction via Solvent Sensitive Machine Learning and Domain Refinement
==================================================================================================
Global Configuration for Fluorescence Emission Wavelength Prediction

This module contains all configuration parameters for the machine learning
framework predicting maximum fluorescence emission wavelength (λem,max) of
dissolved organic matter (DOM) in aqueous environments.
"""

import os
from pathlib import Path

# ============================================================================
# Directory Configuration
# ============================================================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"

# Create directories if they don't exist
for directory in [DATA_DIR, MODEL_DIR, RESULTS_DIR, FIGURES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Data Processing Configuration
# ============================================================================
# Target variable
TARGET_COLUMN = "fluo_maxima"  # Maximum fluorescence emission wavelength (nm)

# Solvent descriptors (key features for solvent effect analysis)
SOLVENT_DESCRIPTORS = ["e30", "DIESab", "SPab"]

# Aqueous solvent identification
AQUEOUS_SOLVENTS = ["water", "h2o", "aqueous", "buffer", "pbs"]

# Feature selection parameters
VARIANCE_THRESHOLD = 0.01  # Remove low-variance features
CORRELATION_THRESHOLD = 0.95  # Remove highly correlated features

# ============================================================================
# Solvent-Directed Oversampling Configuration
# ============================================================================
# Enhancement factor for aqueous data (XGB-A(11) model)
AQUEOUS_ENHANCEMENT_FACTOR = 11

# Baseline model (no enhancement)
BASELINE_ENHANCEMENT_FACTOR = 1

# Train-test split
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ============================================================================
# Model Training Configuration
# ============================================================================
# XGBoost hyperparameter optimization (Optuna)
N_TRIALS = 100  # Number of Optuna trials
CV_FOLDS = 5    # Cross-validation folds

# XGBoost hyperparameter search space
XGBOOST_PARAM_SPACE = {
    "n_estimators": (100, 1000),
    "max_depth": (3, 10),
    "learning_rate": (0.01, 0.3),
    "subsample": (0.6, 1.0),
    "colsample_bytree": (0.6, 1.0),
    "min_child_weight": (1, 10),
    "gamma": (0, 5),
    "reg_alpha": (0, 1),
    "reg_lambda": (0, 1),
}

# Fixed XGBoost parameters
XGBOOST_FIXED_PARAMS = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

# ============================================================================
# Model Evaluation Configuration
# ============================================================================
# Evaluation metrics
METRICS = ["R2", "MAE", "RMSE"]

# Performance comparison groups
COMPARISON_GROUPS = {
    "aqueous_test": "Aqueous-focused test set",
    "full_test": "Full chemical space test set",
}

# ============================================================================
# SHAP Analysis Configuration
# ============================================================================
# SHAP parameters
SHAP_SAMPLE_SIZE = 1000  # Number of samples for SHAP background
SHAP_MAX_DISPLAY = 20    # Maximum features to display in SHAP plots

# Feature importance threshold
FEATURE_IMPORTANCE_THRESHOLD = 0.01

# ============================================================================
# AD-SAL Framework Configuration
# ============================================================================
# Applicability Domain parameters
SOLVENT_WEIGHT = 3  # Optimal weight for solvent descriptors in Tanimoto metric
STRUCTURAL_WEIGHT = 1  # Weight for structural descriptors

# AD threshold (distance-based)
AD_THRESHOLD_PERCENTILE = 95  # Use 95th percentile of training distances

# Tanimoto similarity calculation
TANIMOTO_METHOD = "weighted"  # Options: "standard", "weighted"

# Risk classification thresholds
RISK_LEVELS = {
    "reliable_interpolation": 0.7,  # Tanimoto similarity > 0.7
    "moderate_risk": 0.5,           # 0.5 < similarity <= 0.7
    "high_risk_extrapolation": 0.5, # similarity <= 0.5
}

# ============================================================================
# Visualization Configuration
# ============================================================================
# Figure settings
FIGURE_DPI = 300
FIGURE_FORMAT = "png"
FIGURE_SIZE = (10, 6)

# Color schemes
COLOR_AQUEOUS = "#1f77b4"
COLOR_NON_AQUEOUS = "#ff7f0e"
COLOR_BASELINE = "#2ca02c"
COLOR_ENHANCED = "#d62728"

# Plot style
PLOT_STYLE = "seaborn-v0_8-darkgrid"

# ============================================================================
# Logging Configuration
# ============================================================================
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = RESULTS_DIR / "training.log"

# ============================================================================
# Reproducibility
# ============================================================================
# Set random seeds for reproducibility
import random
import numpy as np

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# ============================================================================
# Model File Paths
# ============================================================================
TRAINED_MODEL_PATH = MODEL_DIR / "xgboost_model.json"  # XGB-A(11) trained model
SCALER_PATH = MODEL_DIR / "scaler.pkl"

# For training new models (if needed)
BASELINE_MODEL_PATH = MODEL_DIR / "baseline_model.json"  # Will be created during training
ENHANCED_MODEL_PATH = MODEL_DIR / "xgboost_model.json"  # XGB-A(11) model

# ============================================================================
# Feature Engineering Configuration
# ============================================================================
# Molecular descriptor calculation methods
DESCRIPTOR_METHODS = ["mordred", "rdkit", "padel"]

# Feature normalization
NORMALIZATION_METHOD = "standard"  # Options: "standard", "minmax", "robust"

# ============================================================================
# Computational Resources
# ============================================================================
N_JOBS = -1  # Use all available CPU cores
MEMORY_LIMIT = None  # No memory limit (set to int for GB limit)

# ============================================================================
# Validation
# ============================================================================
def validate_config():
    """Validate configuration parameters."""
    assert AQUEOUS_ENHANCEMENT_FACTOR > 0, "Enhancement factor must be positive"
    assert 0 < TEST_SIZE < 1, "Test size must be between 0 and 1"
    assert CV_FOLDS > 1, "CV folds must be greater than 1"
    assert 0 < SOLVENT_WEIGHT, "Solvent weight must be positive"
    assert N_TRIALS > 0, "Number of trials must be positive"
    print("✓ Configuration validated successfully")

if __name__ == "__main__":
    validate_config()
    print(f"Base directory: {BASE_DIR}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Model directory: {MODEL_DIR}")
    print(f"Results directory: {RESULTS_DIR}")
