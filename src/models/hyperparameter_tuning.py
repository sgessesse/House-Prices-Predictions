"""
Hyperparameter Tuning Module

This module provides functions to tune hyperparameters of regression models.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
import time
from scipy.stats import randint, uniform
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import learning_curve

def rmse_scorer():
    """
    Create an RMSE scorer for hyperparameter tuning.
    
    Returns:
        callable: RMSE scorer
    """
    return make_scorer(lambda y, y_pred: np.sqrt(mean_squared_error(y, y_pred)), greater_is_better=False)

def get_param_grid(model_name):
    """
    Get parameter grid for a specific model.
    
    Args:
        model_name (str): Name of the model
        
    Returns:
        dict: Parameter grid for grid search
    """
    param_grids = {
        'Ridge Regression': {
            'alpha': [0.01, 0.1, 1.0, 10.0, 100.0],
            'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
        },
        'Lasso Regression': {
            'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0],
            'max_iter': [1000, 2000, 3000],
            'selection': ['cyclic', 'random']
        },
        'Elastic Net': {
            'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0],
            'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
            'max_iter': [1000, 2000, 3000]
        },
        'Random Forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['auto', 'sqrt']
        },
        'Gradient Boosting': {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 4, 5],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'subsample': [0.8, 0.9, 1.0]
        },
        'XGBoost': {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 4, 5, 6],
            'min_child_weight': [1, 3, 5],
            'gamma': [0, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        },
        'LightGBM': {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 4, 5, 6, -1],
            'num_leaves': [31, 50, 70],
            'min_child_samples': [20, 30, 50],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
    }
    
    return param_grids.get(model_name, {})

def get_random_param_distributions(model_name):
    """
    Get parameter distributions for randomized search.
    
    Args:
        model_name (str): Name of the model
        
    Returns:
        dict: Parameter distributions for randomized search
    """
    param_distributions = {
        'Ridge Regression': {
            'alpha': uniform(0.001, 100),
            'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
        },
        'Lasso Regression': {
            'alpha': uniform(0.0001, 1),
            'max_iter': randint(1000, 5000),
            'selection': ['cyclic', 'random']
        },
        'Elastic Net': {
            'alpha': uniform(0.0001, 1),
            'l1_ratio': uniform(0, 1),
            'max_iter': randint(1000, 5000)
        },
        'Random Forest': {
            'n_estimators': randint(50, 300),
            'max_depth': randint(5, 50),
            'min_samples_split': randint(2, 20),
            'min_samples_leaf': randint(1, 10),
            'max_features': ['auto', 'sqrt', None]
        },
        'Gradient Boosting': {
            'n_estimators': randint(50, 500),
            'learning_rate': uniform(0.01, 0.3),
            'max_depth': randint(3, 10),
            'min_samples_split': randint(2, 20),
            'min_samples_leaf': randint(1, 10),
            'subsample': uniform(0.7, 0.3)
        },
        'XGBoost': {
            'n_estimators': randint(50, 500),
            'learning_rate': uniform(0.01, 0.3),
            'max_depth': randint(3, 10),
            'min_child_weight': randint(1, 10),
            'gamma': uniform(0, 0.5),
            'subsample': uniform(0.7, 0.3),
            'colsample_bytree': uniform(0.7, 0.3)
        },
        'LightGBM': {
            'n_estimators': randint(50, 500),
            'learning_rate': uniform(0.01, 0.3),
            'max_depth': randint(3, 10),
            'num_leaves': randint(20, 100),
            'min_child_samples': randint(10, 100),
            'subsample': uniform(0.7, 0.3),
            'colsample_bytree': uniform(0.7, 0.3)
        }
    }
    
    return param_distributions.get(model_name, {})

def tune_hyperparameters(model, model_name, X_train, y_train, method='grid', cv=5, n_iter=50, verbose=1):
    """
    Tune hyperparameters for a model.
    
    Args:
        model: Model object to tune
        model_name (str): Name of the model
        X_train (array-like): Training features
        y_train (array-like): Training target
        method (str): Tuning method ('grid' or 'random')
        cv (int): Number of cross-validation folds
        n_iter (int): Number of iterations for randomized search
        verbose (int): Verbosity level
        
    Returns:
        tuple: (Best model, best parameters, search results)
    """
    # Create RMSE scorer
    scorer = rmse_scorer()
    
    # Start timer
    start_time = time.time()
    
    if method == 'grid':
        # Get parameter grid
        param_grid = get_param_grid(model_name)
        
        if not param_grid:
            print(f"No parameter grid defined for {model_name}. Skipping hyperparameter tuning.")
            return model, {}, None
        
        # Create grid search
        search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring=scorer,
            cv=cv,
            n_jobs=-1,
            verbose=verbose,
            return_train_score=True
        )
    
    elif method == 'random':
        # Get parameter distributions
        param_distributions = get_random_param_distributions(model_name)
        
        if not param_distributions:
            print(f"No parameter distributions defined for {model_name}. Skipping hyperparameter tuning.")
            return model, {}, None
        
        # Create randomized search
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_distributions,
            n_iter=n_iter,
            scoring=scorer,
            cv=cv,
            n_jobs=-1,
            verbose=verbose,
            random_state=42,
            return_train_score=True
        )
    
    else:
        raise ValueError(f"Invalid method: {method}. Use 'grid' or 'random'.")
    
    # Fit the search
    search.fit(X_train, y_train)
    
    # Calculate tuning time
    tuning_time = time.time() - start_time
    
    # Get best model and parameters
    best_model = search.best_estimator_
    best_params = search.best_params_
    
    # Log results
    print(f"Hyperparameter tuning completed in {tuning_time:.2f} seconds")
    print(f"Best parameters: {best_params}")
    print(f"Best RMSE: {-search.best_score_:.4f}")
    
    return best_model, best_params, search

def plot_search_results(search_results, param_name, figsize=(10, 6)):
    """
    Plot search results for a specific parameter.
    
    Args:
        search_results: Search results from GridSearchCV or RandomizedSearchCV
        param_name (str): Parameter name to plot
        figsize (tuple): Figure size
        
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    # Extract results
    results = pd.DataFrame(search_results.cv_results_)
    
    # Create parameter column name
    param_col = f'param_{param_name}'
    
    # Check if parameter exists in results
    if param_col not in results.columns:
        print(f"Parameter {param_name} not found in search results")
        return None
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Convert parameter values to string for categorical plotting
    results[param_col] = results[param_col].astype(str)
    
    # Plot mean test score vs parameter value
    sns.lineplot(x=param_col, y='mean_test_score', data=results, marker='o', label='Test Score', ax=ax)
    
    # Plot mean train score vs parameter value
    sns.lineplot(x=param_col, y='mean_train_score', data=results, marker='o', label='Train Score', ax=ax)
    
    # Set title and labels
    ax.set_title(f'Validation Curve for {param_name}', fontsize=16, pad=20)
    ax.set_xlabel(param_name, fontsize=14)
    ax.set_ylabel('Score (negative RMSE)', fontsize=14)
    ax.legend(fontsize=12)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust layout
    fig.tight_layout()
    
    return fig

def plot_learning_curves(model, X_train, y_train, cv=5, train_sizes=np.linspace(0.1, 1.0, 10), figsize=(10, 6)):
    """
    Plot learning curves for a model.
    
    Args:
        model: Model object
        X_train (array-like): Training features
        y_train (array-like): Training target
        cv (int): Number of cross-validation folds
        train_sizes (array-like): Training set sizes to plot
        figsize (tuple): Figure size
        
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    # Create RMSE scorer
    scorer = rmse_scorer()
    
    # Calculate learning curves
    train_sizes, train_scores, test_scores = learning_curve(
        model, X_train, y_train, cv=cv, train_sizes=train_sizes,
        scoring=scorer, n_jobs=-1
    )
    
    # Calculate mean and std for train scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    
    # Calculate mean and std for test scores
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot learning curves
    ax.plot(train_sizes, -train_mean, 'o-', color='blue', label='Training score')
    ax.plot(train_sizes, -test_mean, 'o-', color='green', label='Cross-validation score')
    
    # Plot standard deviation bands
    ax.fill_between(train_sizes, -train_mean - train_std, -train_mean + train_std, alpha=0.1, color='blue')
    ax.fill_between(train_sizes, -test_mean - test_std, -test_mean + test_std, alpha=0.1, color='green')
    
    # Set title and labels
    ax.set_title('Learning Curves', fontsize=16, pad=20)
    ax.set_xlabel('Training Set Size', fontsize=14)
    ax.set_ylabel('RMSE', fontsize=14)
    ax.legend(fontsize=12)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def compare_tuned_models(models, X_train, y_train, X_test, y_test):
    """
    Compare tuned models on test data.
    
    Args:
        models (dict): Dictionary of model name to tuned model object
        X_train (array-like): Training features
        y_train (array-like): Training target
        X_test (array-like): Test features
        y_test (array-like): Test target
        
    Returns:
        pandas.DataFrame: DataFrame with model comparison results
    """
    # Initialize results list
    results = []
    
    # Evaluate each model
    for name, model in models.items():
        print(f"Evaluating {name}...")
        
        # Make predictions on test data
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Add results to list
        results.append({
            'Model': name,
            'Test RMSE': rmse,
            'Test R²': r2
        })
    
    # Create DataFrame from results
    results_df = pd.DataFrame(results)
    
    # Sort by Test RMSE (ascending)
    results_df = results_df.sort_values('Test RMSE')
    
    return results_df 