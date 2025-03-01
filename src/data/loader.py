"""
Data Loader Module

This module provides functions to load the house prices dataset.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

def get_data_path():
    """
    Returns the path to the data directory.
    
    Returns:
        Path: Path object pointing to the data directory
    """
    # Get the current file's directory
    current_dir = Path(__file__).resolve().parent
    
    # Navigate to the data directory (2 levels up + Data)
    data_dir = current_dir.parent.parent / "Data"
    
    return data_dir

def load_train_data():
    """
    Load the training dataset.
    
    Returns:
        pandas.DataFrame: The training dataset
    """
    data_dir = get_data_path()
    train_path = data_dir / "train.csv"
    
    print(f"Loading training data from {train_path}")
    return pd.read_csv(train_path)

def load_test_data():
    """
    Load the test dataset.
    
    Returns:
        pandas.DataFrame: The test dataset
    """
    data_dir = get_data_path()
    test_path = data_dir / "test.csv"
    
    print(f"Loading test data from {test_path}")
    return pd.read_csv(test_path)

def load_data_description():
    """
    Load the data description text file.
    
    Returns:
        str: The content of the data description file
    """
    data_dir = get_data_path()
    description_path = data_dir / "data_description.txt"
    
    with open(description_path, 'r') as file:
        description = file.read()
    
    return description

def get_feature_groups():
    """
    Group features by their type based on domain knowledge.
    
    Returns:
        dict: Dictionary with feature groups
    """
    return {
        'location': ['MSSubClass', 'MSZoning', 'Neighborhood', 'Condition1', 'Condition2'],
        'lot': ['LotFrontage', 'LotArea', 'LotShape', 'LandContour', 'LotConfig', 'LandSlope'],
        'house_type': ['BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl'],
        'utilities': ['Utilities', 'Heating', 'CentralAir', 'Electrical'],
        'quality': ['OverallQual', 'OverallCond', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 
                   'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC'],
        'size': ['1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFinSF1', 'BsmtFinSF2', 
                'BsmtUnfSF', 'TotalBsmtSF', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', 
                '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal'],
        'rooms': ['BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 
                 'TotRmsAbvGrd', 'Functional', 'Fireplaces'],
        'garage': ['GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageCars'],
        'year': ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'MoSold', 'YrSold'],
        'exterior': ['Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea', 'ExterQual', 'ExterCond'],
        'basement': ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'],
        'sale': ['SaleType', 'SaleCondition']
    }

if __name__ == "__main__":
    # Test the data loading functions
    train_data = load_train_data()
    test_data = load_test_data()
    
    print(f"Train data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    
    # Display the first few rows of the training data
    print("\nFirst 5 rows of training data:")
    print(train_data.head()) 