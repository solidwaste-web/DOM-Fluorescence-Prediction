# DOM Fluorescence Prediction via Machine Learning

**Reliable DOM Fluorescence Prediction via Solvent Sensitive Machine Learning and Domain Refinement**

> **Note**: This repository contains the code and data for manuscript review purposes only. The code is provided to support the reproducibility and transparency of our research findings.

---

## Overview

This project implements a machine learning pipeline for predicting dissolved organic matter (DOM) fluorescence emission wavelengths, with a focus on aqueous solvent systems. The methodology combines:

- **XGBoost regression** with solvent-specific optimization
- **SHAP analysis** for feature importance interpretation
- **AD-SAL** (Applicability Domain based on Similarity-weighted Average Leverage) for prediction reliability assessment

## Project Structure

DOM_Fluorescence_Prediction/
├── data/ # Data directory
│ ├── raw/ # Raw input data (not included)
│ └── processed/ # Processed data (generated)
├── models/ # Trained models (generated)
├── results/ # Evaluation results (generated)
├── figures/ # Generated figures (generated)
├── utils/ # Utility functions
│ ├── init.py
│ └── metrics.py # Performance metrics
├── config.py # Configuration settings
├── 1_data_preprocessing.py # Data preprocessing module
├── 2_feature_engineering.py # Feature engineering module
├── 3_model_training.py # Model training module
├── 4_model_evaluation.py # Model evaluation module
├── 5_shap_analysis.py # SHAP feature analysis module
├── 6_applicability_domain.py # AD-SAL analysis module
├── main.py # Main execution script
├── requirements.txt # Python dependencies
└── README.md # This file


## Requirements

### Python Version
- Python 3.8 - 3.10 (recommended: 3.9)

### Core Dependencies
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0
xgboost==1.7.6
shap==0.42.1
matplotlib==3.7.2
seaborn==0.12.2
joblib==1.3.2
tqdm==4.66.1


### Optional Dependencies
jupyter==1.0.0 # For interactive notebooks
ipykernel==6.25.0 # Jupyter kernel support


## Installation

### Method 1: Using pip (Recommended)

1. **Clone or download this repository**

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
Install dependencies

pip install -r requirements.txt
Method 2: Using conda
Create conda environment

conda create -n dom_prediction python=3.9
conda activate dom_prediction
Install dependencies

pip install -r requirements.txt
Verify Installation
python -c "import numpy, pandas, sklearn, xgboost, shap; print('All dependencies installed successfully!')"
Data Preparation
For Reviewers: Due to data privacy and manuscript review policies, the raw data is not included in this repository. The data structure and format are described below for reproducibility verification.

Expected Data Format
The raw data file should be placed in data/raw/ directory with the following structure:

Column Name	Description	Type
Solvent	Solvent name	string
Emission_Wavelength	Fluorescence emission wavelength (nm)	float
Feature_1 to Feature_N	Molecular descriptors	float
Data Requirements
CSV format

No missing values in critical columns

Emission wavelength in nanometers (nm)

Solvent information for aqueous/non-aqueous classification

Usage
Quick Start
Run the complete pipeline:

python main.py
This will execute all steps:

Data preprocessing and feature engineering

Model training (baseline and enhanced XGB-A(11))

Model evaluation

SHAP feature importance analysis

AD-SAL applicability domain analysis

Advanced Usage
Skip specific steps (useful when re-running analysis):

# Skip training (use existing model)
python main.py --skip-training

# Skip SHAP analysis
python main.py --skip-shap

# Skip AD-SAL analysis
python main.py --skip-ad

# Combine multiple flags
python main.py --skip-training --skip-shap
Run Individual Modules
Each module can be run independently:

# Data preprocessing
python 1_data_preprocessing.py

# Feature engineering
python 2_feature_engineering.py

# Model training
python 3_model_training.py

# Model evaluation
python 4_model_evaluation.py

# SHAP analysis
python 5_shap_analysis.py

# AD-SAL analysis
python 6_applicability_domain.py
Configuration
Key parameters can be modified in config.py:

# Data paths
RAW_DATA_PATH = Path("data/raw/your_data.csv")
TARGET_COLUMN = "Emission_Wavelength"

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2

# XGBoost hyperparameters
XGBOOST_PARAMS = {
    'n_estimators': 500,
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
}

# SHAP parameters
SHAP_SAMPLE_SIZE = 100

# AD-SAL parameters
AD_ALPHA = 15.0
AD_PERCENTILE = 95.0

# Figure settings
FIGURE_DPI = 300
Output Files
After running the pipeline, the following files will be generated:

Data
data/processed_data.csv - Processed dataset with engineered features

Models
models/xgboost_model.json - Trained XGBoost model

models/baseline_model.json - Baseline model (optional)

Results
results/evaluation_metrics.json - Model performance metrics

results/predictions.csv - Predictions on test set

results/evaluation_summary.txt - Text summary of evaluation

results/shap_feature_importance.csv - SHAP feature importance scores

results/shap_values.csv - SHAP values for all samples

results/ad_sal_results.json - AD-SAL analysis results

results/ad_indices.csv - AD indices for all samples

results/predictions_with_ad.csv - Predictions with AD labels

results/ad_alpha_comparison.csv - Comparison of different α values

Figures
figures/shap_beeswarm_plot.png - SHAP beeswarm plot (top 20 features)

figures/shap_bar_plot.png - SHAP feature importance bar plot

figures/ad_distribution.png - AD indices distribution

Methodology
1. Data Preprocessing
Missing value handling

Outlier detection and removal

Feature scaling and normalization

Solvent type classification (aqueous vs. non-aqueous)

2. Feature Engineering
Molecular descriptor calculation

Solvent-specific feature encoding

Feature selection based on correlation and importance

3. Model Training
Baseline Model: Standard XGBoost with default parameters

Enhanced Model (XGB-A(11)): Optimized XGBoost with:

Aqueous-focused training strategy

Hyperparameter tuning via grid search

5-fold cross-validation

Early stopping to prevent overfitting

4. Model Evaluation
Performance metrics: R², RMSE, MAE, MAPE

Separate evaluation for aqueous and non-aqueous samples

Residual analysis and error distribution

Prediction vs. actual scatter plots

5. SHAP Analysis
TreeExplainer for XGBoost interpretation

Feature importance ranking (top 20 features)

Beeswarm plot: feature value vs. SHAP value

Bar plot: mean absolute SHAP values

Analysis focused on aqueous samples only

6. AD-SAL Analysis
Similarity-weighted average leverage calculation

Density index (ρ): measures local similarity density

Discontinuity index (δ): detects activity cliffs

Testing multiple α values (5, 10, 15, 20, 25)

Optimal threshold determination (95th percentile)

Prediction reliability assessment and filtering

Key Features
Solvent-Specific Optimization
Separate handling of aqueous and non-aqueous systems

Enhanced performance on aqueous DOM samples

Stratified train/test splitting to maintain solvent distribution

Interpretability
SHAP values for global and local feature importance

Clear identification of key molecular descriptors

Visualization of feature contributions to predictions

Reliability Assessment
AD-SAL for identifying reliable predictions

Coverage vs. accuracy trade-off analysis

Filtering of out-of-domain predictions

Confidence scoring for each prediction

Performance Metrics
The model is evaluated using:

R² (Coefficient of Determination): Model fit quality (0-1, higher is better)

RMSE (Root Mean Square Error): Average prediction error in nm (lower is better)

MAE (Mean Absolute Error): Average absolute error in nm (lower is better)

MAPE (Mean Absolute Percentage Error): Relative error percentage (lower is better)

Separate metrics are reported for:

Full test set

Aqueous samples only

Non-aqueous samples only

Samples within applicability domain (AD)

Reproducibility
To ensure reproducibility:

All random seeds are fixed (RANDOM_STATE = 42)

Data splitting is stratified by solvent type

Model hyperparameters are explicitly defined in config.py

Complete pipeline is documented and automated

Dependency versions are pinned in requirements.txt

Cross-validation uses fixed folds

Troubleshooting
Common Issues
1. Import errors

# Solution: Reinstall dependencies
pip install --upgrade -r requirements.txt
2. SHAP calculation is slow

# Solution: Reduce sample size in config.py
SHAP_SAMPLE_SIZE = 50  # Default is 100
3. Memory errors during training

# Solution: Reduce XGBoost parameters
XGBOOST_PARAMS = {
    'n_estimators': 300,  # Reduce from 500
    'max_depth': 4,       # Reduce from 6
}
4. File not found errors

# Solution: Check data path in config.py
RAW_DATA_PATH = Path("data/raw/your_actual_filename.csv")
Limitations
Model performance depends on data quality and completeness

AD-SAL coverage may vary with threshold selection (trade-off between coverage and accuracy)

SHAP analysis is computationally intensive for large datasets (>10,000 samples)

Predictions outside the applicability domain should be interpreted with caution

Model is trained on specific DOM types and may not generalize to all organic matter

System Requirements
Minimum Requirements
CPU: 2 cores

RAM: 4 GB

Storage: 1 GB free space

Recommended Requirements
CPU: 4+ cores

RAM: 8+ GB

Storage: 5 GB free space

OS: Windows 10/11, macOS 10.15+, or Linux (Ubuntu 20.04+)

Estimated Runtime
Data preprocessing: 1-2 minutes

Model training: 5-10 minutes

SHAP analysis: 10-20 minutes

AD-SAL analysis: 5-10 minutes

Total pipeline: ~30-45 minutes (depending on dataset size)

Citation
If you use this code or methodology, please cite our manuscript:

[Citation information will be added upon publication]
Contact
For questions regarding this code or the manuscript, please contact:

Corresponding Author: [Name and Email]

First Author: [Name and Email]

License
This code is provided for manuscript review purposes only.

The code may be used by reviewers to verify the reproducibility of our results

Redistribution or commercial use is not permitted without explicit permission

Upon publication, a more permissive license may be applied

Acknowledgments
This work was supported by [Funding Information].

We thank the developers of the open-source libraries used in this project:

XGBoost team for the gradient boosting framework

SHAP team for the interpretability tools

scikit-learn contributors for machine learning utilities

Last Updated: December 2024

Manuscript Status: Under Review

Code Version: 1.0.0

Python Version: 3.9

Platform: Cross-platform (Windows/macOS/Linux)