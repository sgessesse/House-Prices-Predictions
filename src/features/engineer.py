"""
Feature Engineering Module

This module provides functions to create and transform features for the house prices dataset.
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

def create_total_area(df):
    """
    Create a total area feature by summing various area features.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        
    Returns:
        pandas.DataFrame: DataFrame with new total area feature
    """
    df_new = df.copy()
    
    # Create total area feature
    area_cols = [
        'TotalBsmtSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF', 
        'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch'
    ]
    
    # Only use columns that exist in the dataframe
    available_cols = [col for col in area_cols if col in df_new.columns]
    
    # Fill NAs with 0 for area columns
    for col in available_cols:
        df_new[col] = df_new[col].fillna(0)
    
    # Create total area feature
    df_new['TotalArea'] = df_new[available_cols].sum(axis=1)
    
    return df_new

def create_age_features(df):
    """
    Create age-related features.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        
    Returns:
        pandas.DataFrame: DataFrame with new age features
    """
    df_new = df.copy()
    
    # Create house age at time of sale
    if 'YearBuilt' in df_new.columns and 'YrSold' in df_new.columns:
        df_new['HouseAge'] = df_new['YrSold'] - df_new['YearBuilt']
    
    # Create years since remodeling
    if 'YearRemodAdd' in df_new.columns and 'YrSold' in df_new.columns:
        df_new['RemodAge'] = df_new['YrSold'] - df_new['YearRemodAdd']
    
    # Create garage age
    if 'GarageYrBlt' in df_new.columns and 'YrSold' in df_new.columns:
        # Replace 0 (no garage) with YrSold to get 0 age
        df_new['GarageAge'] = df_new.apply(
            lambda row: 0 if row['GarageYrBlt'] == 0 else row['YrSold'] - row['GarageYrBlt'], 
            axis=1
        )
    
    return df_new

def create_quality_features(df):
    """
    Create overall quality features by combining various quality indicators.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        
    Returns:
        pandas.DataFrame: DataFrame with new quality features
    """
    df_new = df.copy()
    
    # Map quality values to numeric scores
    quality_map = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
    
    # List of quality features
    quality_features = [
        'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 
        'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond'
    ]
    
    # Convert quality features to numeric
    for feature in quality_features:
        if feature in df_new.columns:
            df_new[feature] = df_new[feature].map(quality_map).fillna(0)
    
    # Create overall exterior quality
    exterior_quality_features = ['ExterQual', 'ExterCond']
    available_exterior = [col for col in exterior_quality_features if col in df_new.columns]
    if available_exterior:
        df_new['OverallExteriorQuality'] = df_new[available_exterior].mean(axis=1)
    
    # Create overall basement quality
    basement_quality_features = ['BsmtQual', 'BsmtCond']
    available_basement = [col for col in basement_quality_features if col in df_new.columns]
    if available_basement:
        df_new['OverallBasementQuality'] = df_new[available_basement].mean(axis=1)
    
    # Create overall garage quality
    garage_quality_features = ['GarageQual', 'GarageCond']
    available_garage = [col for col in garage_quality_features if col in df_new.columns]
    if available_garage:
        df_new['OverallGarageQuality'] = df_new[available_garage].mean(axis=1)
    
    return df_new

def create_bathroom_features(df):
    """
    Create combined bathroom features.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        
    Returns:
        pandas.DataFrame: DataFrame with new bathroom features
    """
    df_new = df.copy()
    
    # Create total bathrooms feature
    bathroom_cols = ['FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath']
    available_bathroom_cols = [col for col in bathroom_cols if col in df_new.columns]
    
    if available_bathroom_cols:
        # Fill NAs with 0
        for col in available_bathroom_cols:
            df_new[col] = df_new[col].fillna(0)
        
        # Calculate total bathrooms (half baths count as 0.5)
        df_new['TotalBathrooms'] = df_new['FullBath'] + 0.5 * df_new['HalfBath']
        
        if 'BsmtFullBath' in df_new.columns:
            df_new['TotalBathrooms'] += df_new['BsmtFullBath']
        
        if 'BsmtHalfBath' in df_new.columns:
            df_new['TotalBathrooms'] += 0.5 * df_new['BsmtHalfBath']
    
    return df_new

def create_neighborhood_features(df):
    """
    Create neighborhood-based features.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        
    Returns:
        pandas.DataFrame: DataFrame with new neighborhood features
    """
    df_new = df.copy()
    
    if 'Neighborhood' in df_new.columns and 'SalePrice' in df_new.columns:
        # Calculate neighborhood average price
        neighborhood_avg_price = df_new.groupby('Neighborhood')['SalePrice'].transform('mean')
        df_new['NeighborhoodAvgPrice'] = neighborhood_avg_price
        
        # Calculate price relative to neighborhood average
        df_new['PriceToNeighborhoodRatio'] = df_new['SalePrice'] / neighborhood_avg_price
    
    return df_new

def create_price_per_sqft(df):
    """
    Create price per square foot feature.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        
    Returns:
        pandas.DataFrame: DataFrame with new price per sqft feature
    """
    df_new = df.copy()
    
    if 'GrLivArea' in df_new.columns and 'SalePrice' in df_new.columns:
        # Calculate price per square foot
        df_new['PricePerSqFt'] = df_new['SalePrice'] / df_new['GrLivArea']
    
    return df_new

def create_has_features(df):
    """
    Create binary features indicating presence of certain house features.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        
    Returns:
        pandas.DataFrame: DataFrame with new binary features
    """
    df_new = df.copy()
    
    # Create has_pool feature
    if 'PoolArea' in df_new.columns:
        df_new['HasPool'] = (df_new['PoolArea'] > 0).astype(int)
    
    # Create has_garage feature
    if 'GarageArea' in df_new.columns:
        df_new['HasGarage'] = (df_new['GarageArea'] > 0).astype(int)
    
    # Create has_basement feature
    if 'TotalBsmtSF' in df_new.columns:
        df_new['HasBasement'] = (df_new['TotalBsmtSF'] > 0).astype(int)
    
    # Create has_fireplace feature
    if 'Fireplaces' in df_new.columns:
        df_new['HasFireplace'] = (df_new['Fireplaces'] > 0).astype(int)
    
    # Create has_second_floor feature
    if '2ndFlrSF' in df_new.columns:
        df_new['HasSecondFloor'] = (df_new['2ndFlrSF'] > 0).astype(int)
    
    return df_new

def create_interaction_features(df):
    """
    Create interaction features between important variables.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        
    Returns:
        pandas.DataFrame: DataFrame with new interaction features
    """
    df_new = df.copy()
    
    # Quality * Area interaction
    if 'OverallQual' in df_new.columns and 'GrLivArea' in df_new.columns:
        df_new['QualityByArea'] = df_new['OverallQual'] * df_new['GrLivArea']
    
    # Age * Quality interaction
    if 'OverallQual' in df_new.columns and 'HouseAge' in df_new.columns:
        df_new['QualityByAge'] = df_new['OverallQual'] * (1 / (df_new['HouseAge'] + 1))
    
    # Total bathrooms * Total rooms interaction
    if 'TotalBathrooms' in df_new.columns and 'TotRmsAbvGrd' in df_new.columns:
        df_new['BathToRoomRatio'] = df_new['TotalBathrooms'] / df_new['TotRmsAbvGrd']
    
    return df_new

def create_polynomial_features(df, columns, degree=2):
    """
    Create polynomial features for specified columns.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        columns (list): List of columns to create polynomial features for
        degree (int): Degree of polynomial features
        
    Returns:
        pandas.DataFrame: DataFrame with new polynomial features
    """
    df_new = df.copy()
    
    # Create polynomial features
    for col in columns:
        if col in df_new.columns:
            for d in range(2, degree + 1):
                df_new[f'{col}^{d}'] = df_new[col] ** d
    
    return df_new

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    A transformer class for feature engineering.
    
    This class implements the scikit-learn transformer interface for feature engineering.
    """
    
    def __init__(self, create_area=True, create_age=True, create_quality=True, 
                 create_bathroom=True, create_has=True, create_interaction=True):
        """
        Initialize the FeatureEngineer.
        
        Args:
            create_area (bool): Whether to create area features
            create_age (bool): Whether to create age features
            create_quality (bool): Whether to create quality features
            create_bathroom (bool): Whether to create bathroom features
            create_has (bool): Whether to create binary 'has' features
            create_interaction (bool): Whether to create interaction features
        """
        self.create_area = create_area
        self.create_age = create_age
        self.create_quality = create_quality
        self.create_bathroom = create_bathroom
        self.create_has = create_has
        self.create_interaction = create_interaction
    
    def fit(self, X, y=None):
        """
        Fit the transformer (no-op for this transformer).
        
        Args:
            X (pandas.DataFrame): Input dataframe
            y (pandas.Series, optional): Target variable
            
        Returns:
            self: The fitted transformer
        """
        return self
    
    def transform(self, X):
        """
        Transform the input dataframe by creating engineered features.
        
        Args:
            X (pandas.DataFrame): Input dataframe
            
        Returns:
            pandas.DataFrame: Transformed dataframe with engineered features
        """
        df = X.copy()
        
        # Apply feature engineering functions based on configuration
        if self.create_area:
            df = create_total_area(df)
        
        if self.create_age:
            df = create_age_features(df)
        
        if self.create_quality:
            df = create_quality_features(df)
        
        if self.create_bathroom:
            df = create_bathroom_features(df)
        
        if self.create_has:
            df = create_has_features(df)
        
        if self.create_interaction:
            df = create_interaction_features(df)
        
        return df 