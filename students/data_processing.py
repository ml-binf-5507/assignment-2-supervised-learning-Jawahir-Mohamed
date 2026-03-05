"""
Data loading and preprocessing functions for heart disease dataset.
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



def load_heart_disease_data(filepath='data/heart_disease_uci.csv'):
    """
    Load the heart disease dataset from CSV.
    
    Parameters
    ----------
    filepath : str
        Path to the heart disease CSV file
        
    Returns
    -------
    pd.DataFrame
        Raw dataset with all features and targets
        
    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist
    ValueError
        If the CSV is empty or malformed
        
    Examples
    --------
    >>> df = load_heart_disease_data('data/heart_disease_uci.csv')
    >>> df.shape
    (270, 15)
    """
    # Hint: Use pd.read_csv()
    # Hint: Check if file exists and raise helpful error if not
    # TODO: Implement data loading

    
    # Check if file exists
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"CSV file not found at path: {filepath}")
    
    # Read the CSV file
    try:
        df = pd.read_csv(filepath)
    except pd.errors.EmptyDataError:
        raise ValueError(f"The CSV file at {filepath} is empty")
    
    # Check if DataFrame is empty
    if df.empty:
        raise ValueError(f"The CSV file at {filepath} contains no data")
    
    return df
    
    


def preprocess_data(df):
    """
    Handle missing values, encode categorical variables, and clean data.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset
        
    Returns
    -------
    pd.DataFrame
        Cleaned and preprocessed dataset
    """
    # TODO: Implement preprocessing
    # - Handle missing values
    # - Encode categorical variables (e.g., sex, cp, fbs, etc.)
    # - Ensure all columns are numeric
    #pass
    df_clean = df.copy()
    # Common missing value representations
    missing_values = ["?", "NA", "N/A", "na", "n/a", "NaN", "nan", "", "null","Nan"]
    df_clean = df_clean.replace(missing_values, np.nan)
    
    # Fill missing values
    for col in df_clean.columns:
        if df_clean[col].dtype in ['int64', 'float64']:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        else:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
    
    # Encode sex
    if 'sex' in df_clean.columns:
        sex_map = {'male': 1, 'female': 0, 'M': 1, 'F': 0, 'Male': 1, 'Female': 0}
        df_clean['sex'] = df_clean['sex'].map(sex_map)
    
    # Convert remaining object columns to numeric
#for col in df_clean.columns:
        #if df_clean[col].dtype == 'object':
            #df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            #df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    return df_clean
    

def prepare_regression_data(df, target='chol'):
    """
    Prepare data for linear regression (predicting serum cholesterol).
    
    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed dataset
    target : str
        Target column name (default: 'chol')
        
    Returns
    -------
    tuple
        (X, y) feature matrix and target vector
    """
    # TODO: Implement regression data preparation
    # - Remove rows with missing chol values
    # - Exclude chol from features
    # - Return X (features) and y (target)
    
    # Remove rows with missing target values
    df_clean = df.dropna(subset=[target])
    
    # Separate features (X) and target (y)
    X = df_clean.drop(columns=[target])
    y = df_clean[target]
    
    return X, y


def prepare_classification_data(df, target='num'):
    """
    Prepare data for classification (predicting heart disease presence).
    
    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed dataset
    target : str
        Target column name (default: 'num')
        
    Returns
    -------
    tuple
        (X, y) feature matrix and target vector (binary)
    """
    # TODO: Implement classification data preparation
    # - Binarize target variable
    # - Exclude target from features
    # - Exclude chol from features
    # - Return X (features) and y (target)
    #pass

    df_clean = df.copy()
    
    # Determine which target column actually exists
    if target in df_clean.columns:
        target_col = target
    elif 'num' in df_clean.columns:
        target_col = 'num'
    else:
        raise KeyError("No valid target column found")
    
    # Drop rows with missing target values
    df_clean = df_clean.dropna(subset=[target_col])
    
    # Binary target (Series)
    y = (df_clean[target_col] > 0).astype(int)
    
    # Features: drop target and chol
    X = df_clean.drop(columns=[c for c in [target_col, 'chol'] if c in df_clean.columns])
    
    return X, y


def split_and_scale(X, y, test_size=0.2, random_state=42):
    """
    Split data into train/test sets and scale features.
    
    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Feature matrix
    y : pd.Series or np.ndarray
        Target vector
    test_size : float
        Proportion of data to use for testing
    random_state : int
        Random seed for reproducibility
        
    Returns
    -------
    tuple
        (X_train_scaled, X_test_scaled, y_train, y_test, scaler)
        where scaler is the fitted StandardScaler
    """
    # TODO: Implement train/test split and scaling
    # - Use train_test_split with provided parameters
    # - Fit StandardScaler on training data only
    # - Transform both train and test data
    # - Return scaled data and scaler object
    #pass
     # If X contains 'target', drop it before scaling
    if isinstance(X, pd.DataFrame) and 'target' in X.columns:
        X = X.drop(columns=['target'])
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler