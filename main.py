"""
Reliable DOM Fluorescence Prediction via Solvent Sensitive Machine Learning and Domain Refinement
==================================================================================================
Main Execution Script

This script orchestrates the complete machine learning pipeline:
1. Data preprocessing
2. Solvent-directed oversampling
3. Model training (baseline and enhanced XGB-A(11))
4. Model evaluation
5. SHAP feature importance analysis
6. AD-SAL applicability domain analysis

Usage:
    python main.py [--skip-training] [--skip-shap] [--skip-ad]
"""

import argparse
import sys
import importlib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import pandas as pd

# Import config normally
import config

# Dynamically import modules with numeric prefixes
data_preprocessing_module = importlib.import_module('1_data_preprocessing')
DataPreprocessor = data_preprocessing_module.DataPreprocessor

solvent_oversampling_module = importlib.import_module('2_solvent_oversampling')
SolventDirectedSampler = solvent_oversampling_module.SolventDirectedSampler

model_training_module = importlib.import_module('3_model_training')
ModelTrainer = model_training_module.ModelTrainer

model_evaluation_module = importlib.import_module('4_model_evaluation')
ModelEvaluator = model_evaluation_module.ModelEvaluator

shap_analysis_module = importlib.import_module('5_shap_analysis')
analyze_aqueous_samples = shap_analysis_module.analyze_aqueous_samples

applicability_domain_module = importlib.import_module('6_applicability_domain')
perform_ad_analysis = applicability_domain_module.perform_ad_analysis


def setup_directories():
    """Create necessary directories if they don't exist."""
    directories = [
        config.DATA_DIR,
        config.MODEL_DIR,
        config.RESULTS_DIR,
        config.FIGURES_DIR
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    
    print("Directory structure verified")


def check_raw_data():
    """Check if raw data file exists."""
    if not config.RAW_DATA_PATH.exists():
        print(f"ERROR: Raw data file not found: {config.RAW_DATA_PATH}")
        print("Please place your raw data file in the data/ directory")
        return False
    return True


def run_preprocessing():
    """Run data preprocessing."""
    print("\n" + "="*70)
    print("STEP 1: DATA PREPROCESSING")
    print("="*70)
    
    # Check if processed data already exists
    processed_path = config.DATA_DIR / "processed_data.csv"
    
    if processed_path.exists():
        print(f"\nProcessed data already exists: {processed_path}")
        response = input("Do you want to reprocess? (y/n): ").lower()
        if response != 'y':
            print("Skipping preprocessing, loading existing data...")
            return pd.read_csv(processed_path)
    
    # Load raw data
    print("\nLoading raw data...")
    df_raw = pd.read_csv(config.RAW_DATA_PATH)
    print(f"Raw data loaded: {df_raw.shape[0]} samples, {df_raw.shape[1]} columns")
    
    # Preprocess data
    print("\nPreprocessing data...")
    preprocessor = DataPreprocessor()
    df_clean = preprocessor.preprocess(df_raw)
    
    # Save processed data
    df_clean.to_csv(processed_path, index=False)
    print(f"\nProcessed data saved to: {processed_path}")
    
    return df_clean


def run_oversampling(X, y, is_aqueous):
    """Run solvent-directed oversampling."""
    print("\n" + "="*70)
    print("STEP 2: SOLVENT-DIRECTED OVERSAMPLING")
    print("="*70)
    
    # Create sampler
    sampler = SolventDirectedSampler(
        enhancement_factor=config.AQUEOUS_ENHANCEMENT_FACTOR
    )
    
    # Prepare datasets
    datasets = sampler.prepare_datasets(X, y, is_aqueous, model_type="enhanced")
    
    print("\nOversampling complete!")
    
    return datasets


def run_training(X_train, y_train):
    """Run model training."""
    print("\n" + "="*70)
    print("STEP 3: MODEL TRAINING")
    print("="*70)
    
    # Check if model already exists
    model_path = config.TRAINED_MODEL_PATH
    
    if model_path.exists():
        print(f"\nTrained model already exists: {model_path}")
        response = input("Do you want to retrain? (y/n): ").lower()
        if response != 'y':
            print("Skipping training, using existing model...")
            return
    
    # Train model
    trainer = ModelTrainer()
    
    # Train enhanced model (XGB-A(11))
    print("\n--- Training Enhanced Model (XGB-A(11)) ---")
    enhanced_model = trainer.train_enhanced_model(X_train, y_train)
    
    print("\nModel training complete!")


def run_evaluation(X_test, y_test, is_aqueous_test):
    """Run model evaluation."""
    print("\n" + "="*70)
    print("STEP 4: MODEL EVALUATION")
    print("="*70)
    
    # Check if model exists
    model_path = config.TRAINED_MODEL_PATH
    
    if not model_path.exists():
        print(f"ERROR: Model file not found: {model_path}")
        print("Please run training first")
        return None
    
    # Evaluate model
    evaluator = ModelEvaluator(model_path)
    evaluator.load_model()
    
    results = evaluator.generate_evaluation_report(
        X_test, y_test, is_aqueous_test,
        save_dir=config.RESULTS_DIR
    )
    
    print("\nModel evaluation complete!")
    
    return results


def run_shap_analysis(X, is_aqueous):
    """Run SHAP feature importance analysis."""
    print("\n" + "="*70)
    print("STEP 5: SHAP FEATURE IMPORTANCE ANALYSIS")
    print("="*70)
    
    # Check if model exists
    model_path = config.TRAINED_MODEL_PATH
    
    if not model_path.exists():
        print(f"ERROR: Model file not found: {model_path}")
        print("Please run training first")
        return None
    
    # Run SHAP analysis on aqueous samples
    top_features = analyze_aqueous_samples(
        model_path=model_path,
        X=X,
        is_aqueous=is_aqueous,
        top_n=20
    )
    
    print("\nSHAP analysis complete!")
    
    return top_features


def run_ad_analysis(X_train, y_train, X_test, y_test):
    """Run AD-SAL applicability domain analysis."""
    print("\n" + "="*70)
    print("STEP 6: AD-SAL APPLICABILITY DOMAIN ANALYSIS")
    print("="*70)
    
    # Check if model exists
    model_path = config.TRAINED_MODEL_PATH
    
    if not model_path.exists():
        print(f"ERROR: Model file not found: {model_path}")
        print("Please run training first")
        return None
    
    # Run AD analysis
    results = perform_ad_analysis(
        model_path=model_path,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        test_alpha_values=True
    )
    
    print("\nAD-SAL analysis complete!")
    
    return results


def print_summary(eval_results, shap_results, ad_results):
    """Print final summary of all analyses."""
    print("\n" + "="*70)
    print("PIPELINE EXECUTION SUMMARY")
    print("="*70)
    
    if eval_results:
        print("\n--- Model Performance ---")
        print(f"Overall R²: {eval_results['overall_metrics']['R2']:.4f}")
        print(f"Overall RMSE: {eval_results['overall_metrics']['RMSE']:.2f} nm")
        print(f"Overall MAE: {eval_results['overall_metrics']['MAE']:.2f} nm")
        
        print(f"\nAqueous R²: {eval_results['aqueous_metrics']['R2']:.4f}")
        print(f"Aqueous RMSE: {eval_results['aqueous_metrics']['RMSE']:.2f} nm")
        print(f"Aqueous MAE: {eval_results['aqueous_metrics']['MAE']:.2f} nm")
    
    if shap_results is not None:
        print("\n--- SHAP Analysis ---")
        print(f"Top 20 features identified")
        print(f"Results saved to: {config.FIGURES_DIR}")
    
    if ad_results:
        print("\n--- AD-SAL Analysis ---")
        print(f"Coverage: {ad_results['coverage']*100:.1f}%")
        print(f"RMSE improvement: {ad_results['rmse_improvement']:.2f} nm")
        print(f"Recommended α: 15")
    
    print("\n--- Output Files ---")
    print(f"Processed data: {config.DATA_DIR / 'processed_data.csv'}")
    print(f"Trained model: {config.TRAINED_MODEL_PATH}")
    print(f"Evaluation results: {config.RESULTS_DIR}")
    print(f"Figures: {config.FIGURES_DIR}")
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)


def main():
    """Main execution function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="DOM Fluorescence Prediction Pipeline"
    )
    parser.add_argument(
        '--skip-training',
        action='store_true',
        help='Skip model training (use existing model)'
    )
    parser.add_argument(
        '--skip-shap',
        action='store_true',
        help='Skip SHAP analysis'
    )
    parser.add_argument(
        '--skip-ad',
        action='store_true',
        help='Skip AD-SAL analysis'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("DOM FLUORESCENCE PREDICTION PIPELINE")
    print("Reliable DOM Fluorescence Prediction via Solvent Sensitive")
    print("Machine Learning and Domain Refinement")
    print("="*70)
    
    # Setup
    setup_directories()
    
    if not check_raw_data():
        sys.exit(1)
    
    # Step 1: Preprocessing
    df_processed = run_preprocessing()
    
    # Prepare data
    X = df_processed.drop(columns=[config.TARGET_COLUMN, 'is_aqueous'])
    y = df_processed[config.TARGET_COLUMN]
    is_aqueous = df_processed['is_aqueous']
    
    # Step 2: Solvent-directed oversampling
    datasets = run_oversampling(X, y, is_aqueous)
    
    X_train = datasets['X_train']
    y_train = datasets['y_train']
    X_test = datasets['X_test']
    y_test = datasets['y_test']
    is_aqueous_test = datasets['is_aqueous_test']
    
    # Step 3: Model Training
    if not args.skip_training:
        run_training(X_train, y_train)
    else:
        print("\n[SKIPPED] Model training")
    
    # Step 4: Model Evaluation
    eval_results = run_evaluation(X_test, y_test, is_aqueous_test)
    
    # Step 5: SHAP Analysis
    if not args.skip_shap:
        shap_results = run_shap_analysis(X, is_aqueous)
    else:
        print("\n[SKIPPED] SHAP analysis")
        shap_results = None
    
    # Step 6: AD-SAL Analysis
    if not args.skip_ad:
        ad_results = run_ad_analysis(X_train, y_train, X_test, y_test)
    else:
        print("\n[SKIPPED] AD-SAL analysis")
        ad_results = None
    
    # Print summary
    print_summary(eval_results, shap_results, ad_results)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
