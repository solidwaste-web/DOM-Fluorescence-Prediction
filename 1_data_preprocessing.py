"""
Reliable DOM Fluorescence Prediction via Solvent Sensitive Machine Learning and Domain Refinement
==================================================================================================
Data Preprocessing Module

This module handles:
1. Raw data collection and cleaning
2. Molecular descriptor calculation (Mordred, RDKit, PaDEL)
3. Solvent descriptor calculation (e30, DIESab, SPab)
4. Aqueous/non-aqueous solvent classification
5. Feature engineering and selection
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from mordred import Calculator, descriptors
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
import joblib

import config


class DataPreprocessor:
    """
    Main class for data preprocessing and feature engineering.
    """
    
    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize the preprocessor.
        
        Parameters
        ----------
        data_path : str, optional
            Path to raw data CSV file
        """
        self.data_path = data_path
        self.data = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def load_data(self, data_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load raw data from CSV file.
        
        Parameters
        ----------
        data_path : str, optional
            Path to CSV file. If None, uses self.data_path
        
        Returns
        -------
        pd.DataFrame
            Loaded data
        """
        if data_path is None:
            data_path = self.data_path
            
        if data_path is None:
            raise ValueError("No data path provided")
        
        print(f"Loading data from {data_path}...")
        self.data = pd.read_csv(data_path)
        print(f"Loaded {len(self.data)} samples")
        
        return self.data
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean raw data: remove duplicates, handle missing values.
        
        Parameters
        ----------
        df : pd.DataFrame
            Raw data
        
        Returns
        -------
        pd.DataFrame
            Cleaned data
        """
        print("\nCleaning data...")
        initial_count = len(df)
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Remove rows with missing target values
        df = df.dropna(subset=[config.TARGET_COLUMN])
        
        # Remove rows with missing SMILES
        if 'SMILES' in df.columns:
            df = df.dropna(subset=['SMILES'])
        
        print(f"Removed {initial_count - len(df)} samples during cleaning")
        print(f"Remaining samples: {len(df)}")
        
        return df
    
    def classify_solvents(self, df: pd.DataFrame, 
                         solvent_column: str = 'solvent') -> pd.DataFrame:
        """
        Classify solvents as aqueous or non-aqueous.
        
        Parameters
        ----------
        df : pd.DataFrame
            Data with solvent information
        solvent_column : str
            Name of the solvent column
        
        Returns
        -------
        pd.DataFrame
            Data with 'is_aqueous' column added
        """
        print("\nClassifying solvents...")
        
        def is_aqueous(solvent_name):
            if pd.isna(solvent_name):
                return False
            solvent_lower = str(solvent_name).lower()
            return any(aq in solvent_lower for aq in config.AQUEOUS_SOLVENTS)
        
        df['is_aqueous'] = df[solvent_column].apply(is_aqueous)
        
        n_aqueous = df['is_aqueous'].sum()
        n_non_aqueous = len(df) - n_aqueous
        
        print(f"Aqueous samples: {n_aqueous} ({n_aqueous/len(df)*100:.1f}%)")
        print(f"Non-aqueous samples: {n_non_aqueous} ({n_non_aqueous/len(df)*100:.1f}%)")
        
        return df
    
    def calculate_molecular_descriptors(self, smiles_list: List[str]) -> pd.DataFrame:
        """
        Calculate molecular descriptors using Mordred and RDKit.
        
        Parameters
        ----------
        smiles_list : List[str]
            List of SMILES strings
        
        Returns
        -------
        pd.DataFrame
            Molecular descriptors
        """
        print("\nCalculating molecular descriptors...")
        
        # Convert SMILES to RDKit molecules
        mols = [Chem.MolFromSmiles(smi) for smi in smiles_list]
        
        # Mordred descriptors
        print("Computing Mordred descriptors...")
        calc = Calculator(descriptors, ignore_3D=True)
        mordred_df = calc.pandas(mols)
        
        # Convert to numeric and handle errors
        mordred_df = mordred_df.apply(pd.to_numeric, errors='coerce')
        
        # RDKit descriptors
        print("Computing RDKit descriptors...")
        rdkit_descriptors = []
        for mol in mols:
            if mol is not None:
                desc_dict = {}
                for name, func in Descriptors.descList:
                    try:
                        desc_dict[f"RDKit_{name}"] = func(mol)
                    except:
                        desc_dict[f"RDKit_{name}"] = np.nan
                rdkit_descriptors.append(desc_dict)
            else:
                rdkit_descriptors.append({})
        
        rdkit_df = pd.DataFrame(rdkit_descriptors)
        
        # Combine descriptors
        descriptors_df = pd.concat([mordred_df, rdkit_df], axis=1)
        
        print(f"Calculated {descriptors_df.shape[1]} molecular descriptors")
        
        return descriptors_df
    
    def calculate_solvent_descriptors(self, df: pd.DataFrame,
                                     solvent_column: str = 'solvent') -> pd.DataFrame:
        """
        Calculate or extract solvent descriptors (e30, DIESab, SPab).
        
        Parameters
        ----------
        df : pd.DataFrame
            Data with solvent information
        solvent_column : str
            Name of the solvent column
        
        Returns
        -------
        pd.DataFrame
            Data with solvent descriptors added
        """
        print("\nProcessing solvent descriptors...")
        
        # Check if solvent descriptors already exist
        if all(desc in df.columns for desc in config.SOLVENT_DESCRIPTORS):
            print("Solvent descriptors already present in data")
            return df
        
        # If not present, you would need a solvent database or calculation method
        # For now, we assume they are provided in the input data
        print("Warning: Solvent descriptors not found. Please ensure e30, DIESab, SPab are in input data")
        
        return df
    
    def remove_low_variance_features(self, X: pd.DataFrame, 
                                    threshold: float = None) -> pd.DataFrame:
        """
        Remove features with low variance.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
        threshold : float, optional
            Variance threshold. If None, uses config.VARIANCE_THRESHOLD
        
        Returns
        -------
        pd.DataFrame
            Feature matrix with low-variance features removed
        """
        if threshold is None:
            threshold = config.VARIANCE_THRESHOLD
        
        print(f"\nRemoving features with variance < {threshold}...")
        initial_features = X.shape[1]
        
        selector = VarianceThreshold(threshold=threshold)
        X_selected = selector.fit_transform(X)
        
        selected_features = X.columns[selector.get_support()]
        X_filtered = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
        
        print(f"Removed {initial_features - X_filtered.shape[1]} low-variance features")
        print(f"Remaining features: {X_filtered.shape[1]}")
        
        return X_filtered
    
    def remove_correlated_features(self, X: pd.DataFrame,
                                  threshold: float = None) -> pd.DataFrame:
        """
        Remove highly correlated features.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
        threshold : float, optional
            Correlation threshold. If None, uses config.CORRELATION_THRESHOLD
        
        Returns
        -------
        pd.DataFrame
            Feature matrix with correlated features removed
        """
        if threshold is None:
            threshold = config.CORRELATION_THRESHOLD
        
        print(f"\nRemoving features with correlation > {threshold}...")
        initial_features = X.shape[1]
        
        # Calculate correlation matrix
        corr_matrix = X.corr().abs()
        
        # Select upper triangle of correlation matrix
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find features with correlation greater than threshold
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        
        X_filtered = X.drop(columns=to_drop)
        
        print(f"Removed {len(to_drop)} correlated features")
        print(f"Remaining features: {X_filtered.shape[1]}")
        
        return X_filtered
    
    def normalize_features(self, X: pd.DataFrame, 
                          fit: bool = True) -> pd.DataFrame:
        """
        Normalize features using StandardScaler.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
        fit : bool
            If True, fit the scaler. If False, use existing scaler
        
        Returns
        -------
        pd.DataFrame
            Normalized feature matrix
        """
        print("\nNormalizing features...")
        
        if fit:
            X_normalized = self.scaler.fit_transform(X)
        else:
            X_normalized = self.scaler.transform(X)
        
        X_normalized_df = pd.DataFrame(
            X_normalized, 
            columns=X.columns, 
            index=X.index
        )
        
        return X_normalized_df
    
    def preprocess_pipeline(self, data_path: str,
                           save_processed: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Complete preprocessing pipeline.
        
        Parameters
        ----------
        data_path : str
            Path to raw data CSV
        save_processed : bool
            If True, save processed data
        
        Returns
        -------
        Tuple[pd.DataFrame, pd.Series]
            (X, y) - Feature matrix and target variable
        """
        print("="*60)
        print("Starting Data Preprocessing Pipeline")
        print("="*60)
        
        # Load and clean data
        df = self.load_data(data_path)
        df = self.clean_data(df)
        
        # Classify solvents
        df = self.classify_solvents(df)
        
        # Calculate molecular descriptors
        if 'SMILES' in df.columns:
            mol_descriptors = self.calculate_molecular_descriptors(df['SMILES'].tolist())
            df = pd.concat([df, mol_descriptors], axis=1)
        
        # Process solvent descriptors
        df = self.calculate_solvent_descriptors(df)
        
        # Separate features and target
        feature_cols = [col for col in df.columns 
                       if col not in [config.TARGET_COLUMN, 'SMILES', 'solvent', 'is_aqueous']]
        
        X = df[feature_cols].copy()
        y = df[config.TARGET_COLUMN].copy()
        
        # Handle missing values in features
        X = X.fillna(X.median())
        
        # Feature selection
        X = self.remove_low_variance_features(X)
        X = self.remove_correlated_features(X)
        
        # Normalize features
        X = self.normalize_features(X, fit=True)
        
        # Store metadata
        self.feature_names = X.columns.tolist()
        
        # Save processed data
        if save_processed:
            processed_path = config.DATA_DIR / "processed_data.csv"
            processed_df = pd.concat([X, y, df['is_aqueous']], axis=1)
            processed_df.to_csv(processed_path, index=False)
            print(f"\nSaved processed data to {processed_path}")
            
            # Save scaler
            scaler_path = config.SCALER_PATH
            joblib.dump(self.scaler, scaler_path)
            print(f"Saved scaler to {scaler_path}")
        
        print("\n" + "="*60)
        print("Preprocessing Complete")
        print("="*60)
        print(f"Final feature matrix shape: {X.shape}")
        print(f"Target variable shape: {y.shape}")
        
        return X, y, df['is_aqueous']


if __name__ == "__main__":
    # Example usage
    preprocessor = DataPreprocessor()
    
    # Replace with your actual data path
    data_path = config.DATA_DIR / "raw_data.csv"
    
    if data_path.exists():
        X, y, is_aqueous = preprocessor.preprocess_pipeline(str(data_path))
        
        print("\nPreprocessing Summary:")
        print(f"Total samples: {len(X)}")
        print(f"Total features: {X.shape[1]}")
        print(f"Aqueous samples: {is_aqueous.sum()}")
        print(f"Target range: [{y.min():.1f}, {y.max():.1f}] nm")
    else:
        print(f"Data file not found: {data_path}")
        print("Please place your raw data CSV in the data/ directory")
