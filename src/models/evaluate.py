"""
Model Evaluation Module

This module provides functions to evaluate regression models for house price prediction.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def calculate_metrics(y_true, y_pred):
    """
    Calculate regression metrics.
    
    Args:
        y_true (array-like): True target values
        y_pred (array-like): Predicted target values
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    # Calculate metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Calculate additional metrics
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    # Calculate explained variance
    explained_variance = 1 - (np.var(y_true - y_pred) / np.var(y_true))
    
    # Return metrics
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'MAPE (%)': mape,
        'Explained Variance': explained_variance
    }

def print_metrics(metrics):
    """
    Print evaluation metrics in a formatted way.
    
    Args:
        metrics (dict): Dictionary of evaluation metrics
    """
    print("=" * 40)
    print("Model Evaluation Metrics:")
    print("=" * 40)
    
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    print("=" * 40)

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
    # Make cross-validated predictions
    y_pred_cv = cross_val_predict(model, X, y, cv=cv)
    
    # Calculate metrics
    metrics = calculate_metrics(y, y_pred_cv)
    
    # Print metrics
    print_metrics(metrics)
    
    return metrics

def plot_residuals_analysis(y_true, y_pred, figsize=(16, 12)):
    """
    Plot residuals analysis for regression model.
    
    Args:
        y_true (array-like): True target values
        y_pred (array-like): Predicted target values
        figsize (tuple): Figure size
    """
    # Calculate residuals
    residuals = y_true - y_pred
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Flatten axes array for easy iteration
    axes = axes.flatten()
    
    # Plot 1: Residuals vs Predicted
    axes[0].scatter(y_pred, residuals, alpha=0.6, color='steelblue')
    axes[0].axhline(y=0, color='red', linestyle='--')
    axes[0].set_title('Residuals vs Predicted Values', fontsize=14)
    axes[0].set_xlabel('Predicted Values', fontsize=12)
    axes[0].set_ylabel('Residuals', fontsize=12)
    axes[0].grid(True, linestyle='--', alpha=0.7)
    
    # Plot 2: Histogram of Residuals
    sns.histplot(residuals, kde=True, ax=axes[1], color='steelblue')
    axes[1].axvline(x=0, color='red', linestyle='--')
    axes[1].set_title('Distribution of Residuals', fontsize=14)
    axes[1].set_xlabel('Residuals', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].grid(True, linestyle='--', alpha=0.7)
    
    # Plot 3: Q-Q Plot
    stats.probplot(residuals, plot=axes[2])
    axes[2].set_title('Q-Q Plot of Residuals', fontsize=14)
    axes[2].grid(True, linestyle='--', alpha=0.7)
    
    # Plot 4: Actual vs Predicted
    axes[3].scatter(y_true, y_pred, alpha=0.6, color='steelblue')
    
    # Add perfect prediction line
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    axes[3].plot([min_val, max_val], [min_val, max_val], 'r--')
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred)
    
    # Add metrics as text
    metrics_text = f'RMSE: {metrics["RMSE"]:.3f}\nMAE: {metrics["MAE"]:.3f}\nR²: {metrics["R²"]:.3f}'
    axes[3].annotate(metrics_text, xy=(0.05, 0.95), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                    ha='left', va='top', fontsize=10)
    
    axes[3].set_title('Actual vs Predicted Values', fontsize=14)
    axes[3].set_xlabel('Actual Values', fontsize=12)
    axes[3].set_ylabel('Predicted Values', fontsize=12)
    axes[3].grid(True, linestyle='--', alpha=0.7)
    
    # Set title
    fig.suptitle('Residuals Analysis', fontsize=18, y=0.98)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    return fig

def plot_prediction_error_by_feature(X, y_true, y_pred, feature_name, figsize=(12, 8)):
    """
    Plot prediction error by a specific feature.
    
    Args:
        X (pandas.DataFrame): Feature dataframe
        y_true (array-like): True target values
        y_pred (array-like): Predicted target values
        feature_name (str): Name of the feature to analyze
        figsize (tuple): Figure size
    """
    # Check if feature exists
    if feature_name not in X.columns:
        print(f"Feature {feature_name} not found in the dataframe")
        return None
    
    # Calculate absolute error
    abs_error = np.abs(y_true - y_pred)
    
    # Create dataframe with feature and error
    error_df = pd.DataFrame({
        'Feature': X[feature_name],
        'Absolute Error': abs_error
    })
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Check if feature is categorical or numeric
    if X[feature_name].dtype == 'object' or len(X[feature_name].unique()) < 10:
        # For categorical features, use boxplot
        sns.boxplot(x='Feature', y='Absolute Error', data=error_df, ax=ax)
        plt.xticks(rotation=45)
    else:
        # For numeric features, use scatter plot
        ax.scatter(X[feature_name], abs_error, alpha=0.6, color='steelblue')
        
        # Add trend line
        z = np.polyfit(X[feature_name], abs_error, 1)
        p = np.poly1d(z)
        ax.plot(X[feature_name], p(X[feature_name]), "r--", alpha=0.5)
        
        # Add correlation coefficient
        corr = np.corrcoef(X[feature_name], abs_error)[0, 1]
        ax.annotate(f"Correlation: {corr:.3f}", xy=(0.05, 0.95), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    # Set title and labels
    ax.set_title(f'Prediction Error by {feature_name}', fontsize=16, pad=20)
    ax.set_xlabel(feature_name, fontsize=14)
    ax.set_ylabel('Absolute Error', fontsize=14)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def plot_error_distribution(y_true, y_pred, bins=30, figsize=(12, 8)):
    """
    Plot the distribution of prediction errors.
    
    Args:
        y_true (array-like): True target values
        y_pred (array-like): Predicted target values
        bins (int): Number of bins for histogram
        figsize (tuple): Figure size
    """
    # Calculate errors
    errors = y_true - y_pred
    abs_errors = np.abs(errors)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Distribution of errors
    sns.histplot(errors, kde=True, bins=bins, ax=ax1, color='steelblue')
    ax1.axvline(x=0, color='red', linestyle='--')
    
    # Add statistics
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    
    # Add text box with statistics
    stats_text = f'Mean: {mean_error:.3f}\nStd Dev: {std_error:.3f}'
    ax1.annotate(stats_text, xy=(0.05, 0.95), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                ha='left', va='top', fontsize=12)
    
    ax1.set_title('Distribution of Errors', fontsize=14)
    ax1.set_xlabel('Error (Actual - Predicted)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Plot 2: Distribution of absolute errors
    sns.histplot(abs_errors, kde=True, bins=bins, ax=ax2, color='steelblue')
    
    # Add statistics
    mean_abs_error = np.mean(abs_errors)
    median_abs_error = np.median(abs_errors)
    
    # Add text box with statistics
    stats_text = f'Mean: {mean_abs_error:.3f}\nMedian: {median_abs_error:.3f}'
    ax2.annotate(stats_text, xy=(0.05, 0.95), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                ha='left', va='top', fontsize=12)
    
    ax2.set_title('Distribution of Absolute Errors', fontsize=14)
    ax2.set_xlabel('Absolute Error', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Set title
    fig.suptitle('Error Distribution Analysis', fontsize=18, y=0.98)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    return fig

def plot_prediction_intervals(y_true, y_pred, confidence=0.95, figsize=(12, 8)):
    """
    Plot prediction intervals based on residuals.
    
    Args:
        y_true (array-like): True target values
        y_pred (array-like): Predicted target values
        confidence (float): Confidence level for intervals
        figsize (tuple): Figure size
    """
    # Calculate residuals
    residuals = y_true - y_pred
    
    # Calculate prediction interval
    alpha = 1 - confidence
    z_score = stats.norm.ppf(1 - alpha/2)
    
    # Calculate standard deviation of residuals
    residual_std = np.std(residuals)
    
    # Calculate prediction intervals
    lower_bound = y_pred - z_score * residual_std
    upper_bound = y_pred + z_score * residual_std
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Sort data for better visualization
    sorted_indices = np.argsort(y_true)
    y_true_sorted = y_true[sorted_indices]
    y_pred_sorted = y_pred[sorted_indices]
    lower_bound_sorted = lower_bound[sorted_indices]
    upper_bound_sorted = upper_bound[sorted_indices]
    
    # Plot actual values
    ax.scatter(range(len(y_true_sorted)), y_true_sorted, alpha=0.6, color='blue', label='Actual')
    
    # Plot predicted values
    ax.plot(range(len(y_pred_sorted)), y_pred_sorted, color='red', label='Predicted')
    
    # Plot prediction intervals
    ax.fill_between(range(len(y_pred_sorted)), lower_bound_sorted, upper_bound_sorted, 
                     alpha=0.2, color='red', label=f'{confidence*100:.0f}% Prediction Interval')
    
    # Calculate coverage
    coverage = np.mean((y_true >= lower_bound) & (y_true <= upper_bound))
    
    # Set title and labels
    ax.set_title(f'Prediction Intervals (Coverage: {coverage*100:.1f}%)', fontsize=16, pad=20)
    ax.set_xlabel('Observation Index (sorted)', fontsize=14)
    ax.set_ylabel('Target Value', fontsize=14)
    ax.legend(fontsize=12)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def compare_models_performance(models_dict, X, y, cv=5):
    """
    Compare performance of multiple models.
    
    Args:
        models_dict (dict): Dictionary of model name to model object
        X (array-like): Features
        y (array-like): Target
        cv (int): Number of cross-validation folds
        
    Returns:
        pandas.DataFrame: DataFrame with model comparison results
    """
    # Initialize results list
    results = []
    
    # Evaluate each model
    for name, model in models_dict.items():
        print(f"Evaluating {name}...")
        
        # Make cross-validated predictions
        y_pred_cv = cross_val_predict(model, X, y, cv=cv)
        
        # Calculate metrics
        metrics = calculate_metrics(y, y_pred_cv)
        
        # Add results to list
        results.append({
            'Model': name,
            'RMSE': metrics['RMSE'],
            'MAE': metrics['MAE'],
            'R²': metrics['R²'],
            'MAPE (%)': metrics['MAPE (%)'],
            'Explained Variance': metrics['Explained Variance']
        })
    
    # Create DataFrame from results
    results_df = pd.DataFrame(results)
    
    # Sort by RMSE (ascending)
    results_df = results_df.sort_values('RMSE')
    
    return results_df

def plot_models_comparison(comparison_df, metric='RMSE', figsize=(12, 8)):
    """
    Plot comparison of multiple models.
    
    Args:
        comparison_df (pandas.DataFrame): DataFrame with model comparison results
        metric (str): Metric to plot
        figsize (tuple): Figure size
    """
    # Check if metric exists
    if metric not in comparison_df.columns:
        print(f"Metric {metric} not found in the dataframe")
        return None
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Sort by metric
    if metric in ['RMSE', 'MAE', 'MAPE (%)']:
        # Lower is better
        sorted_df = comparison_df.sort_values(metric)
    else:
        # Higher is better
        sorted_df = comparison_df.sort_values(metric, ascending=False)
    
    # Plot bar chart
    bars = sns.barplot(x='Model', y=metric, data=sorted_df, palette='viridis', ax=ax)
    
    # Add metric values
    for i, p in enumerate(ax.patches):
        height = p.get_height()
        ax.text(p.get_x() + p.get_width()/2., height + 0.01,
                f'{height:.4f}', ha='center', va='bottom')
    
    # Set title and labels
    ax.set_title(f'Model Comparison by {metric}', fontsize=16, pad=20)
    ax.set_xlabel('Model', fontsize=14)
    ax.set_ylabel(metric, fontsize=14)
    
    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')
    
    # Add grid
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig 