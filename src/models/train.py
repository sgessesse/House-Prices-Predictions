"""
Model Training Module

This module provides functions to train various regression models for house price prediction.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score, KFold
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import time
import joblib
import os
from pathlib import Path

def get_models_directory():
    """
    Returns the path to the models directory.
    
    Returns:
        Path: Path object pointing to the models directory
    """
    # Get the current file's directory
    current_dir = Path(__file__).resolve().parent
    
    # Create models directory if it doesn't exist
    models_dir = current_dir / "saved_models"
    os.makedirs(models_dir, exist_ok=True)
    
    return models_dir

def get_base_models():
    """
    Get a dictionary of base regression models.
    
    Returns:
        dict: Dictionary of model name to model object
    """
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=0.001),
        'Elastic Net': ElasticNet(alpha=0.001, l1_ratio=0.5),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
        'LightGBM': lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    }
    
    return models

def train_model(model, X_train, y_train, model_name=None):
    """
    Train a regression model.
    
    Args:
        model: Model object to train
        X_train (array-like): Training features
        y_train (array-like): Training target
        model_name (str, optional): Name of the model for logging
        
    Returns:
        object: Trained model
    """
    # Start timer
    start_time = time.time()
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Calculate training time
    training_time = time.time() - start_time
    
    # Log training information
    if model_name:
        print(f"Trained {model_name} in {training_time:.2f} seconds")
    
    return model

def evaluate_model(model, X, y, cv=5):
    """
    Evaluate a model using cross-validation.
    
    Args:
        model: Model object to evaluate
        X (array-like): Features
        y (array-like): Target
        cv (int): Number of cross-validation folds
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    # Define cross-validation strategy
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    
    # Calculate cross-validation scores
    neg_mse_scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error')
    r2_scores = cross_val_score(model, X, y, cv=kf, scoring='r2')
    neg_mae_scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_absolute_error')
    
    # Convert negative MSE to RMSE
    rmse_scores = np.sqrt(-neg_mse_scores)
    mae_scores = -neg_mae_scores
    
    # Calculate mean and std of metrics
    mean_rmse = rmse_scores.mean()
    std_rmse = rmse_scores.std()
    mean_r2 = r2_scores.mean()
    std_r2 = r2_scores.std()
    mean_mae = mae_scores.mean()
    std_mae = mae_scores.std()
    
    # Return metrics
    return {
        'rmse_mean': mean_rmse,
        'rmse_std': std_rmse,
        'r2_mean': mean_r2,
        'r2_std': std_r2,
        'mae_mean': mean_mae,
        'mae_std': std_mae
    }

def compare_models(models, X_train, y_train, cv=5):
    """
    Compare multiple regression models using cross-validation.
    
    Args:
        models (dict): Dictionary of model name to model object
        X_train (array-like): Training features
        y_train (array-like): Training target
        cv (int): Number of cross-validation folds
        
    Returns:
        pandas.DataFrame: DataFrame with model comparison results
    """
    # Initialize results list
    results = []
    
    # Evaluate each model
    for name, model in models.items():
        print(f"Evaluating {name}...")
        
        # Start timer
        start_time = time.time()
        
        # Evaluate model
        metrics = evaluate_model(model, X_train, y_train, cv=cv)
        
        # Calculate evaluation time
        eval_time = time.time() - start_time
        
        # Add results to list
        results.append({
            'Model': name,
            'RMSE': metrics['rmse_mean'],
            'RMSE Std': metrics['rmse_std'],
            'R²': metrics['r2_mean'],
            'R² Std': metrics['r2_std'],
            'MAE': metrics['mae_mean'],
            'MAE Std': metrics['mae_std'],
            'Evaluation Time (s)': eval_time
        })
    
    # Create DataFrame from results
    results_df = pd.DataFrame(results)
    
    # Sort by RMSE (ascending)
    results_df = results_df.sort_values('RMSE')
    
    return results_df

def train_and_save_model(model, X_train, y_train, model_name, feature_names=None):
    """
    Train a model and save it to disk.
    
    Args:
        model: Model object to train
        X_train (array-like): Training features
        y_train (array-like): Training target
        model_name (str): Name of the model for saving
        feature_names (list, optional): List of feature names
        
    Returns:
        object: Trained model
    """
    # Train the model
    trained_model = train_model(model, X_train, y_train, model_name)
    
    # Get models directory
    models_dir = get_models_directory()
    
    # Create model file path
    model_path = models_dir / f"{model_name.replace(' ', '_').lower()}.pkl"
    
    # Save the model
    joblib.dump(trained_model, model_path)
    print(f"Model saved to {model_path}")
    
    # Save feature names if provided
    if feature_names is not None:
        feature_names_path = models_dir / f"{model_name.replace(' ', '_').lower()}_features.pkl"
        joblib.dump(feature_names, feature_names_path)
        print(f"Feature names saved to {feature_names_path}")
    
    return trained_model

def load_model(model_name):
    """
    Load a saved model from disk.
    
    Args:
        model_name (str): Name of the model to load
        
    Returns:
        object: Loaded model
    """
    # Get models directory
    models_dir = get_models_directory()
    
    # Create model file path
    model_path = models_dir / f"{model_name.replace(' ', '_').lower()}.pkl"
    
    # Check if model file exists
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load the model
    model = joblib.load(model_path)
    print(f"Model loaded from {model_path}")
    
    return model

def get_feature_importance(model, feature_names):
    """
    Get feature importance from a trained model.
    
    Args:
        model: Trained model object
        feature_names (list): List of feature names
        
    Returns:
        pandas.DataFrame: DataFrame with feature importance
    """
    # Check if model has feature_importances_ attribute
    if hasattr(model, 'feature_importances_'):
        # Get feature importances
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # For linear models, use absolute coefficients
        importances = np.abs(model.coef_)
    else:
        raise ValueError("Model does not have feature_importances_ or coef_ attribute")
    
    # Create DataFrame with feature names and importances
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    
    # Sort by importance (descending)
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    return importance_df

def predict(model, X):
    """
    Make predictions using a trained model.
    
    Args:
        model: Trained model object
        X (array-like): Features to predict on
        
    Returns:
        array-like: Predictions
    """
    return model.predict(X)

def create_submission_file(predictions, test_ids, filename='submission.csv'):
    """
    Create a submission file for Kaggle.
    
    Args:
        predictions (array-like): Predicted values
        test_ids (array-like): Test IDs
        filename (str): Output filename
        
    Returns:
        str: Path to the submission file
    """
    # Create submission DataFrame
    submission = pd.DataFrame({
        'Id': test_ids,
        'SalePrice': predictions
    })
    
    # Save to CSV
    submission.to_csv(filename, index=False)
    print(f"Submission file saved to {filename}")
    
    return filename 