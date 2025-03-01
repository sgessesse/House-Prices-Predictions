"""
Data Preprocessor Module

This module provides functions to preprocess the house prices dataset.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def identify_missing_values(df):
    """
    Identify missing values in the dataset.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        
    Returns:
        pandas.DataFrame: DataFrame with missing value counts and percentages
    """
    # Calculate missing values
    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / len(df)) * 100
    
    # Create a DataFrame with missing value information
    missing_info = pd.DataFrame({
        'Missing Values': missing_values,
        'Missing Percentage': missing_percentage.round(2)
    })
    
    # Filter to only show features with missing values
    missing_info = missing_info[missing_info['Missing Values'] > 0].sort_values('Missing Values', ascending=False)
    
    return missing_info

def handle_missing_values(df, strategy='advanced'):
    """
    Handle missing values in the dataset.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        strategy (str): Strategy for handling missing values ('simple' or 'advanced')
        
    Returns:
        pandas.DataFrame: DataFrame with missing values handled
    """
    df_processed = df.copy()
    
    if strategy == 'simple':
        # Simple imputation strategy
        numeric_cols = df_processed.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = df_processed.select_dtypes(include=['object']).columns
        
        # For numeric columns, use median imputation
        if len(numeric_cols) > 0:
            numeric_imputer = SimpleImputer(strategy='median')
            df_processed[numeric_cols] = numeric_imputer.fit_transform(df_processed[numeric_cols])
        
        # For categorical columns, use most frequent imputation
        if len(categorical_cols) > 0:
            categorical_imputer = SimpleImputer(strategy='most_frequent')
            df_processed[categorical_cols] = categorical_imputer.fit_transform(df_processed[categorical_cols])
    
    elif strategy == 'advanced':
        # Advanced imputation strategy based on domain knowledge
        
        # NA in these columns likely means the house doesn't have these features
        na_means_none_features = [
            'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 
            'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
            'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'
        ]
        
        for feature in na_means_none_features:
            if feature in df_processed.columns:
                df_processed[feature] = df_processed[feature].fillna('None')
        
        # LotFrontage can be imputed based on neighborhood
        if 'LotFrontage' in df_processed.columns:
            # Group by neighborhood and fill with median LotFrontage of each neighborhood
            df_processed['LotFrontage'] = df_processed.groupby('Neighborhood')['LotFrontage'].transform(
                lambda x: x.fillna(x.median())
            )
            # If still missing, fill with overall median
            df_processed['LotFrontage'] = df_processed['LotFrontage'].fillna(df_processed['LotFrontage'].median())
        
        # GarageYrBlt - if garage doesn't exist, use house YearBuilt
        if 'GarageYrBlt' in df_processed.columns and 'YearBuilt' in df_processed.columns:
            no_garage_mask = df_processed['GarageType'] == 'None'
            df_processed.loc[no_garage_mask, 'GarageYrBlt'] = 0
            df_processed.loc[~no_garage_mask & df_processed['GarageYrBlt'].isna(), 'GarageYrBlt'] = \
                df_processed.loc[~no_garage_mask & df_processed['GarageYrBlt'].isna(), 'YearBuilt']
        
        # For remaining numeric features, use median
        numeric_cols = df_processed.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_cols:
            df_processed[col] = df_processed[col].fillna(df_processed[col].median())
        
        # For remaining categorical features, use most frequent value
        categorical_cols = df_processed.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0])
    
    return df_processed

def encode_categorical_features(df, encoding_type='onehot'):
    """
    Encode categorical features in the dataset.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        encoding_type (str): Type of encoding ('onehot' or 'ordinal')
        
    Returns:
        pandas.DataFrame: DataFrame with encoded categorical features
    """
    df_processed = df.copy()
    
    # Identify categorical columns
    categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
    
    if encoding_type == 'onehot':
        # One-hot encode categorical features
        for col in categorical_cols:
            # Get dummies and drop the first category to avoid multicollinearity
            dummies = pd.get_dummies(df_processed[col], prefix=col, drop_first=True)
            # Add the dummies to the dataframe
            df_processed = pd.concat([df_processed, dummies], axis=1)
            # Drop the original column
            df_processed = df_processed.drop(col, axis=1)
    
    elif encoding_type == 'ordinal':
        # Define ordinal features and their order
        ordinal_features = {
            'ExterQual': ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
            'ExterCond': ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
            'BsmtQual': ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
            'BsmtCond': ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
            'BsmtExposure': ['None', 'No', 'Mn', 'Av', 'Gd'],
            'BsmtFinType1': ['None', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'],
            'BsmtFinType2': ['None', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'],
            'HeatingQC': ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
            'KitchenQual': ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
            'FireplaceQu': ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
            'GarageQual': ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
            'GarageCond': ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
            'PoolQC': ['None', 'Fa', 'TA', 'Gd', 'Ex'],
            'LotShape': ['IR3', 'IR2', 'IR1', 'Reg'],
            'LandContour': ['Low', 'HLS', 'Bnk', 'Lvl'],
            'Utilities': ['ELO', 'NoSeWa', 'NoSewr', 'AllPub'],
            'LandSlope': ['Sev', 'Mod', 'Gtl'],
            'Functional': ['Sal', 'Sev', 'Maj2', 'Maj1', 'Mod', 'Min2', 'Min1', 'Typ']
        }
        
        # Apply ordinal encoding for specified features
        for feature, ordering in ordinal_features.items():
            if feature in df_processed.columns:
                # Create a mapping dictionary
                mapping = {val: i for i, val in enumerate(ordering)}
                # Apply the mapping
                df_processed[feature] = df_processed[feature].map(mapping)
        
        # For remaining categorical features, use one-hot encoding
        remaining_cat_cols = [col for col in categorical_cols if col not in ordinal_features]
        for col in remaining_cat_cols:
            dummies = pd.get_dummies(df_processed[col], prefix=col, drop_first=True)
            df_processed = pd.concat([df_processed, dummies], axis=1)
            df_processed = df_processed.drop(col, axis=1)
    
    return df_processed

def handle_outliers(df, method='iqr', columns=None):
    """
    Handle outliers in the dataset.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        method (str): Method for handling outliers ('iqr', 'zscore', or 'cap')
        columns (list): List of columns to check for outliers. If None, all numeric columns are used.
        
    Returns:
        pandas.DataFrame: DataFrame with outliers handled
    """
    df_processed = df.copy()
    
    # If no columns specified, use all numeric columns
    if columns is None:
        columns = df_processed.select_dtypes(include=['int64', 'float64']).columns.tolist()
        # Exclude ID columns and target variable if present
        if 'Id' in columns:
            columns.remove('Id')
        if 'SalePrice' in columns:
            columns.remove('SalePrice')
    
    if method == 'iqr':
        # IQR method
        for col in columns:
            Q1 = df_processed[col].quantile(0.25)
            Q3 = df_processed[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Cap the outliers
            df_processed[col] = df_processed[col].clip(lower=lower_bound, upper=upper_bound)
    
    elif method == 'zscore':
        # Z-score method
        for col in columns:
            mean = df_processed[col].mean()
            std = df_processed[col].std()
            
            # Identify outliers (Z-score > 3 or < -3)
            outliers = (df_processed[col] - mean).abs() > 3 * std
            
            # Replace outliers with mean
            df_processed.loc[outliers, col] = mean
    
    elif method == 'cap':
        # Percentile capping
        for col in columns:
            lower_percentile = df_processed[col].quantile(0.01)
            upper_percentile = df_processed[col].quantile(0.99)
            
            df_processed[col] = df_processed[col].clip(lower=lower_percentile, upper=upper_percentile)
    
    return df_processed

def normalize_features(df, method='standard', columns=None):
    """
    Normalize numeric features in the dataset.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        method (str): Method for normalization ('standard', 'minmax', or 'log')
        columns (list): List of columns to normalize. If None, all numeric columns are used.
        
    Returns:
        pandas.DataFrame: DataFrame with normalized features
    """
    df_processed = df.copy()
    
    # If no columns specified, use all numeric columns
    if columns is None:
        columns = df_processed.select_dtypes(include=['int64', 'float64']).columns.tolist()
        # Exclude ID columns and target variable if present
        if 'Id' in columns:
            columns.remove('Id')
        if 'SalePrice' in columns:
            columns.remove('SalePrice')
    
    if method == 'standard':
        # Standardization (z-score normalization)
        scaler = StandardScaler()
        df_processed[columns] = scaler.fit_transform(df_processed[columns])
    
    elif method == 'minmax':
        # Min-max normalization
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        df_processed[columns] = scaler.fit_transform(df_processed[columns])
    
    elif method == 'log':
        # Log transformation
        for col in columns:
            # Add a small constant to handle zeros
            df_processed[col] = np.log1p(df_processed[col])
    
    return df_processed

def create_preprocessing_pipeline(categorical_features, numeric_features):
    """
    Create a scikit-learn preprocessing pipeline.
    
    Args:
        categorical_features (list): List of categorical feature names
        numeric_features (list): List of numeric feature names
        
    Returns:
        sklearn.pipeline.Pipeline: Preprocessing pipeline
    """
    # Define preprocessing for categorical features
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Define preprocessing for numeric features
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor

def preprocess_data(train_df, test_df=None, target_column='SalePrice'):
    """
    Preprocess the data for model training.
    
    Args:
        train_df (pandas.DataFrame): Training dataframe
        test_df (pandas.DataFrame, optional): Test dataframe
        target_column (str): Name of the target column
        
    Returns:
        tuple: Processed X_train, y_train, X_test, preprocessor
    """
    # Make a copy of the dataframes
    train = train_df.copy()
    
    # Extract target variable
    y_train = None
    if target_column in train.columns:
        y_train = train[target_column]
        train = train.drop(target_column, axis=1)
    
    # Identify feature types
    categorical_features = train.select_dtypes(include=['object']).columns.tolist()
    numeric_features = train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Remove 'Id' from features if present
    if 'Id' in numeric_features:
        numeric_features.remove('Id')
        train = train.drop('Id', axis=1)
        if test_df is not None:
            test_df = test_df.drop('Id', axis=1)
    
    # Remove 'LogSalePrice' from features if present
    if 'LogSalePrice' in numeric_features:
        numeric_features.remove('LogSalePrice')
        train = train.drop('LogSalePrice', axis=1)
    
    # Replace infinite values with NaN
    train = train.replace([np.inf, -np.inf], np.nan)
    
    # Create and fit the preprocessing pipeline
    preprocessor = create_preprocessing_pipeline(categorical_features, numeric_features)
    X_train = preprocessor.fit_transform(train)
    
    # Process test data if provided
    X_test = None
    if test_df is not None:
        test = test_df.copy()
        # Replace infinite values with NaN in test data
        test = test.replace([np.inf, -np.inf], np.nan)
        X_test = preprocessor.transform(test)
    
    # Get feature names after preprocessing
    cat_feature_names = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
    feature_names = np.concatenate([numeric_features, cat_feature_names])
    
    return X_train, y_train, X_test, preprocessor, feature_names 