"""
Visualization Module

This module provides functions to create visualizations for the house prices dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as ticker

# Set default style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')

def configure_plot_style():
    """
    Configure the plot style for consistent, professional visualizations.
    """
    # Set font sizes
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12
    
    # Set figure size
    plt.rcParams['figure.figsize'] = (12, 8)
    
    # Set font family
    plt.rcParams['font.family'] = 'sans-serif'
    
    # Set grid style
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.alpha'] = 0.7
    
    # Set custom color palette
    custom_palette = sns.color_palette('viridis', 10)
    sns.set_palette(custom_palette)

def plot_missing_values(missing_info, figsize=(12, 8)):
    """
    Plot missing values information.
    
    Args:
        missing_info (pandas.DataFrame): DataFrame with missing value information
        figsize (tuple): Figure size
    """
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create a horizontal bar plot
    sns.barplot(x=missing_info['Missing Percentage'], 
                y=missing_info.index, 
                palette='viridis',
                ax=ax)
    
    # Add percentage labels
    for i, p in enumerate(ax.patches):
        width = p.get_width()
        plt.text(width + 0.5, p.get_y() + p.get_height()/2, 
                 f'{width:.1f}% ({missing_info["Missing Values"].iloc[i]})', 
                 ha='left', va='center')
    
    # Set labels and title
    plt.title('Percentage of Missing Values by Feature', fontsize=16, pad=20)
    plt.xlabel('Missing Percentage (%)', fontsize=14)
    plt.ylabel('Features', fontsize=14)
    
    # Add grid lines
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def plot_distribution(df, column, target_column=None, bins=30, figsize=(12, 8)):
    """
    Plot the distribution of a numeric column with optional target variable coloring.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        column (str): Column to plot
        target_column (str, optional): Target column for coloring
        bins (int): Number of bins for histogram
        figsize (tuple): Figure size
    """
    if target_column and target_column in df.columns:
        # Create a figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot histogram with KDE
        sns.histplot(df[column], kde=True, bins=bins, ax=ax1, color='steelblue')
        
        # Add mean and median lines
        mean_val = df[column].mean()
        median_val = df[column].median()
        ax1.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
        ax1.axvline(median_val, color='green', linestyle='-.', linewidth=2, label=f'Median: {median_val:.2f}')
        
        # Add legend
        ax1.legend()
        
        # Set labels and title
        ax1.set_title(f'Distribution of {column}', fontsize=16, pad=20)
        ax1.set_xlabel(column, fontsize=14)
        ax1.set_ylabel('Frequency', fontsize=14)
        
        # Plot scatter plot with target variable
        scatter = sns.scatterplot(x=column, y=target_column, data=df, ax=ax2, alpha=0.6, 
                                 hue=pd.qcut(df[target_column], q=5) if len(df[target_column].unique()) > 10 else df[target_column])
        
        # Add regression line
        sns.regplot(x=column, y=target_column, data=df, ax=ax2, scatter=False, line_kws={'color': 'red'})
        
        # Calculate correlation
        correlation = df[[column, target_column]].corr().iloc[0, 1]
        
        # Set labels and title
        ax2.set_title(f'{column} vs {target_column} (Correlation: {correlation:.2f})', fontsize=16, pad=20)
        ax2.set_xlabel(column, fontsize=14)
        ax2.set_ylabel(target_column, fontsize=14)
        
        # Adjust layout
        plt.tight_layout()
        
    else:
        # Create a figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot histogram with KDE
        sns.histplot(df[column], kde=True, bins=bins, color='steelblue', ax=ax)
        
        # Add mean and median lines
        mean_val = df[column].mean()
        median_val = df[column].median()
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
        ax.axvline(median_val, color='green', linestyle='-.', linewidth=2, label=f'Median: {median_val:.2f}')
        
        # Add skewness and kurtosis
        skewness = df[column].skew()
        kurtosis = df[column].kurt()
        
        # Add text box with statistics
        stats_text = f'Skewness: {skewness:.2f}\nKurtosis: {kurtosis:.2f}'
        ax.annotate(stats_text, xy=(0.05, 0.95), xycoords='axes fraction', 
                    bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8),
                    ha='left', va='top', fontsize=12)
        
        # Add legend
        ax.legend()
        
        # Set labels and title
        ax.set_title(f'Distribution of {column}', fontsize=16, pad=20)
        ax.set_xlabel(column, fontsize=14)
        ax.set_ylabel('Frequency', fontsize=14)
        
        # Adjust layout
        plt.tight_layout()
    
    return fig

def plot_categorical_distribution(df, column, target_column=None, figsize=(12, 8), top_n=None, sort_by='count'):
    """
    Plot the distribution of a categorical column with optional target variable statistics.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        column (str): Categorical column to plot
        target_column (str, optional): Target column for statistics
        figsize (tuple): Figure size
        top_n (int, optional): Show only top N categories
        sort_by (str): Sort categories by 'count' or 'target_mean'
    """
    if target_column and target_column in df.columns:
        # Create a figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Calculate category counts and target means
        category_stats = df.groupby(column)[target_column].agg(['count', 'mean']).reset_index()
        
        # Sort by specified criterion
        if sort_by == 'count':
            category_stats = category_stats.sort_values('count', ascending=False)
        elif sort_by == 'target_mean':
            category_stats = category_stats.sort_values('mean', ascending=False)
        
        # Limit to top N categories if specified
        if top_n is not None and len(category_stats) > top_n:
            category_stats = category_stats.iloc[:top_n]
            other_count = df[~df[column].isin(category_stats[column])][column].count()
            other_mean = df[~df[column].isin(category_stats[column])][target_column].mean()
            other_row = pd.DataFrame({column: ['Other'], 'count': [other_count], 'mean': [other_mean]})
            category_stats = pd.concat([category_stats, other_row], ignore_index=True)
        
        # Plot count distribution
        sns.barplot(x='count', y=column, data=category_stats, ax=ax1, palette='viridis')
        
        # Add count labels
        for i, p in enumerate(ax1.patches):
            width = p.get_width()
            ax1.text(width + 0.5, p.get_y() + p.get_height()/2, 
                     f'{width:.0f}', ha='left', va='center')
        
        # Set labels and title
        ax1.set_title(f'Count of {column} Categories', fontsize=16, pad=20)
        ax1.set_xlabel('Count', fontsize=14)
        ax1.set_ylabel(column, fontsize=14)
        
        # Plot target mean by category
        bars = sns.barplot(x='mean', y=column, data=category_stats, ax=ax2, palette='viridis')
        
        # Add mean labels
        for i, p in enumerate(ax2.patches):
            width = p.get_width()
            ax2.text(width + 0.01, p.get_y() + p.get_height()/2, 
                     f'{width:.2f}', ha='left', va='center')
        
        # Add global mean line
        global_mean = df[target_column].mean()
        ax2.axvline(global_mean, color='red', linestyle='--', linewidth=2, 
                   label=f'Global Mean: {global_mean:.2f}')
        ax2.legend()
        
        # Set labels and title
        ax2.set_title(f'Mean {target_column} by {column}', fontsize=16, pad=20)
        ax2.set_xlabel(f'Mean {target_column}', fontsize=14)
        ax2.set_ylabel(column, fontsize=14)
        
        # Adjust layout
        plt.tight_layout()
        
    else:
        # Create a figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Calculate category counts
        category_counts = df[column].value_counts()
        
        # Limit to top N categories if specified
        if top_n is not None and len(category_counts) > top_n:
            top_categories = category_counts.nlargest(top_n).index
            other_count = category_counts[~category_counts.index.isin(top_categories)].sum()
            category_counts = category_counts[category_counts.index.isin(top_categories)]
            category_counts['Other'] = other_count
        
        # Plot count distribution
        sns.barplot(x=category_counts.values, y=category_counts.index, palette='viridis', ax=ax)
        
        # Add count labels
        for i, p in enumerate(ax.patches):
            width = p.get_width()
            ax.text(width + 0.5, p.get_y() + p.get_height()/2, 
                     f'{width:.0f} ({width/len(df)*100:.1f}%)', ha='left', va='center')
        
        # Set labels and title
        ax.set_title(f'Distribution of {column}', fontsize=16, pad=20)
        ax.set_xlabel('Count', fontsize=14)
        ax.set_ylabel(column, fontsize=14)
        
        # Adjust layout
        plt.tight_layout()
    
    return fig

def plot_correlation_matrix(df, columns=None, figsize=(14, 12), cmap='viridis', annotate=True, threshold=0.5, target_column=None):
    """
    Plot correlation matrix for selected columns.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        columns (list, optional): List of columns to include. If None, use all numeric columns.
        figsize (tuple): Figure size
        cmap (str): Colormap for heatmap
        annotate (bool): Whether to annotate the heatmap with correlation values
        threshold (float): Correlation threshold for filtering features (absolute value)
        target_column (str, optional): Target column to filter correlations against
    """
    # Select columns
    if columns is None:
        # Use all numeric columns
        numeric_df = df.select_dtypes(include=['int64', 'float64'])
    else:
        numeric_df = df[columns]
    
    # Calculate correlation matrix
    corr_matrix = numeric_df.corr()
    
    # Filter for highly correlated features
    if target_column is not None and target_column in corr_matrix.columns:
        # Filter based on correlation with target column
        high_corr_features = corr_matrix.index[abs(corr_matrix[target_column]) > threshold]
        # Always include the target column
        if target_column not in high_corr_features:
            high_corr_features = pd.Index(list(high_corr_features) + [target_column])
        corr_matrix = corr_matrix.loc[high_corr_features, high_corr_features]
    else:
        # Filter based on any high correlations
        # Create a mask for correlations above threshold
        mask = (abs(corr_matrix) > threshold) & (abs(corr_matrix) < 1.0)
        # Get features that have at least one high correlation
        high_corr_features = corr_matrix.columns[mask.any()]
        corr_matrix = corr_matrix.loc[high_corr_features, high_corr_features]
    
    # Create a mask for the upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Create custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    
    # Create the figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot the heatmap
    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                annot=annotate, fmt='.2f', square=True, linewidths=.5, cbar_kws={"shrink": .8}, ax=ax)
    
    # Set title
    ax.set_title('Correlation Matrix (Features with |correlation| > {})'.format(threshold), fontsize=18, pad=20)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def plot_pairplot(df, columns, target_column=None, diag_kind='kde', height=2.5, aspect=1):
    """
    Create a pairplot for selected columns.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        columns (list): List of columns to include
        target_column (str, optional): Target column for coloring
        diag_kind (str): Kind of plot for diagonal ('hist' or 'kde')
        height (float): Height of each subplot
        aspect (float): Aspect ratio of each subplot
    """
    # Create a subset of the dataframe with selected columns
    plot_df = df[columns].copy()
    
    # Add target column if specified
    if target_column and target_column in df.columns:
        plot_df[target_column] = df[target_column]
        
        # If target has many unique values, create bins
        if len(df[target_column].unique()) > 10:
            plot_df['target_binned'] = pd.qcut(df[target_column], q=5, labels=[f'Q{i+1}' for i in range(5)])
            hue_column = 'target_binned'
        else:
            hue_column = target_column
    else:
        hue_column = None
    
    # Create the pairplot
    g = sns.pairplot(plot_df, hue=hue_column, diag_kind=diag_kind, 
                    height=height, aspect=aspect, plot_kws={'alpha': 0.6})
    
    # Set title
    g.fig.suptitle('Pairwise Relationships', fontsize=18, y=1.02)
    
    # Adjust layout
    plt.tight_layout()
    
    return g

def plot_boxplots(df, numeric_columns, figsize=(14, 10)):
    """
    Create boxplots for multiple numeric columns.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        numeric_columns (list): List of numeric columns to plot
        figsize (tuple): Figure size
    """
    # Calculate number of rows and columns for subplots
    n_cols = 3
    n_rows = (len(numeric_columns) + n_cols - 1) // n_cols
    
    # Create figure and subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Flatten axes array for easy iteration
    axes = axes.flatten()
    
    # Create boxplots
    for i, column in enumerate(numeric_columns):
        if i < len(axes):
            # Create boxplot
            sns.boxplot(y=df[column], ax=axes[i], color='skyblue')
            
            # Add swarmplot for data points
            sns.swarmplot(y=df[column], ax=axes[i], color='darkblue', alpha=0.5, size=3)
            
            # Set title and labels
            axes[i].set_title(f'Boxplot of {column}', fontsize=14)
            axes[i].set_ylabel(column, fontsize=12)
            
            # Add grid
            axes[i].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Hide empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    # Set title
    fig.suptitle('Boxplots of Numeric Features', fontsize=18, y=0.98)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    return fig

def plot_feature_importance(feature_names, importances, top_n=20, figsize=(12, 8)):
    """
    Plot feature importances.
    
    Args:
        feature_names (list): List of feature names
        importances (list): List of feature importances
        top_n (int): Number of top features to show
        figsize (tuple): Figure size
    """
    # Create a DataFrame with feature names and importances
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    
    # Sort by importance and get top N features
    feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False).head(top_n)
    
    # Create the figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot horizontal bar chart
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis', ax=ax)
    
    # Add importance values
    for i, p in enumerate(ax.patches):
        width = p.get_width()
        ax.text(width + 0.01, p.get_y() + p.get_height()/2, 
                 f'{width:.4f}', ha='left', va='center')
    
    # Set title and labels
    ax.set_title('Feature Importance', fontsize=16, pad=20)
    ax.set_xlabel('Importance', fontsize=14)
    ax.set_ylabel('Feature', fontsize=14)
    
    # Add grid
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def plot_residuals_analysis(y_true, y_pred, figsize=(14, 10)):
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
    
    axes[3].set_title('Actual vs Predicted Values', fontsize=14)
    axes[3].set_xlabel('Actual Values', fontsize=12)
    axes[3].set_ylabel('Predicted Values', fontsize=12)
    axes[3].grid(True, linestyle='--', alpha=0.7)
    
    # Calculate metrics
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    r2 = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
    
    # Add metrics as text
    metrics_text = f'MSE: {mse:.2f}\nRMSE: {rmse:.2f}\nMAE: {mae:.2f}\nR²: {r2:.2f}'
    fig.text(0.5, 0.01, metrics_text, ha='center', va='bottom', 
             bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8),
             fontsize=12)
    
    # Set title
    fig.suptitle('Residuals Analysis', fontsize=18, y=0.98)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, bottom=0.15)
    
    return fig

def plot_learning_curve(train_sizes, train_scores, test_scores, figsize=(10, 6)):
    """
    Plot learning curve for a model.
    
    Args:
        train_sizes (array-like): Training set sizes
        train_scores (array-like): Training scores
        test_scores (array-like): Test scores
        figsize (tuple): Figure size
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate mean and std for train scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    
    # Calculate mean and std for test scores
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    # Plot learning curve
    ax.plot(train_sizes, train_mean, 'o-', color='blue', label='Training score')
    ax.plot(train_sizes, test_mean, 'o-', color='green', label='Cross-validation score')
    
    # Plot standard deviation bands
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='green')
    
    # Set title and labels
    ax.set_title('Learning Curve', fontsize=16, pad=20)
    ax.set_xlabel('Training Set Size', fontsize=14)
    ax.set_ylabel('Score', fontsize=14)
    ax.legend(loc='best', fontsize=12)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig 