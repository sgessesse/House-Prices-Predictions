"""
House Price Prediction - Main Script

This script runs the entire house price prediction pipeline.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import logging
import time

# Import custom modules
from src.data.loader import load_train_data, load_test_data
from src.data.preprocessor import (
    identify_missing_values, handle_missing_values, 
    encode_categorical_features, handle_outliers,
    normalize_features, preprocess_data
)
from src.features.engineer import (
    create_total_area, create_age_features, 
    create_quality_features, create_bathroom_features,
    create_has_features, create_interaction_features,
    FeatureEngineer
)
from src.models.train import (
    get_base_models, train_model, evaluate_model,
    compare_models, train_and_save_model, get_feature_importance,
    predict, create_submission_file
)
from src.models.hyperparameter_tuning import (
    tune_hyperparameters, plot_learning_curves
)
from src.models.evaluate import (
    calculate_metrics, plot_residuals_analysis,
    plot_error_distribution, plot_models_comparison
)
from src.visualization.plots import (
    configure_plot_style, plot_missing_values,
    plot_distribution, plot_categorical_distribution,
    plot_correlation_matrix, plot_feature_importance
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('house_price_prediction.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def create_output_directory():
    """
    Create output directory for results.
    
    Returns:
        Path: Path object pointing to the output directory
    """
    output_dir = Path('output')
    os.makedirs(output_dir, exist_ok=True)
    
    # Create subdirectories
    os.makedirs(output_dir / 'figures', exist_ok=True)
    os.makedirs(output_dir / 'models', exist_ok=True)
    os.makedirs(output_dir / 'submissions', exist_ok=True)
    
    return output_dir

def save_figure(fig, filename, output_dir):
    """
    Save a matplotlib figure to the output directory.
    
    Args:
        fig: Matplotlib figure
        filename (str): Filename
        output_dir (Path): Output directory
    """
    fig_path = output_dir / 'figures' / filename
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Figure saved to {fig_path}")

def run_exploratory_data_analysis(train_df, output_dir):
    """
    Run exploratory data analysis on the training data.
    
    Args:
        train_df (pandas.DataFrame): Training dataframe
        output_dir (Path): Output directory
    """
    logger.info("Running exploratory data analysis...")
    
    # Configure plot style
    configure_plot_style()
    
    # Identify missing values
    missing_info = identify_missing_values(train_df)
    logger.info(f"Found {len(missing_info)} features with missing values")
    
    # Plot missing values
    if not missing_info.empty:
        fig = plot_missing_values(missing_info)
        save_figure(fig, 'missing_values.png', output_dir)
    
    # Plot distribution of target variable
    if 'SalePrice' in train_df.columns:
        fig = plot_distribution(train_df, 'SalePrice')
        save_figure(fig, 'saleprice_distribution.png', output_dir)
        
        # Log transform of target variable
        train_df['LogSalePrice'] = np.log1p(train_df['SalePrice'])
        fig = plot_distribution(train_df, 'LogSalePrice')
        save_figure(fig, 'log_saleprice_distribution.png', output_dir)
    
    # Plot distribution of important numeric features
    numeric_features = ['LotArea', 'GrLivArea', 'TotalBsmtSF', 'GarageArea']
    for feature in numeric_features:
        if feature in train_df.columns:
            fig = plot_distribution(train_df, feature, target_column='SalePrice')
            save_figure(fig, f'{feature}_distribution.png', output_dir)
    
    # Plot distribution of important categorical features
    categorical_features = ['Neighborhood', 'OverallQual', 'ExterQual', 'KitchenQual']
    for feature in categorical_features:
        if feature in train_df.columns:
            fig = plot_categorical_distribution(train_df, feature, target_column='SalePrice')
            save_figure(fig, f'{feature}_distribution.png', output_dir)
    
    # Plot correlation matrix
    numeric_df = train_df.select_dtypes(include=['int64', 'float64'])
    if len(numeric_df.columns) > 1:
        fig = plot_correlation_matrix(numeric_df, threshold=0.5, target_column='SalePrice')
        save_figure(fig, 'correlation_matrix.png', output_dir)
    
    logger.info("Exploratory data analysis completed")

def run_data_preprocessing(train_df, test_df=None):
    """
    Run data preprocessing on the training and test data.
    
    Args:
        train_df (pandas.DataFrame): Training dataframe
        test_df (pandas.DataFrame, optional): Test dataframe
        
    Returns:
        tuple: Processed X_train, y_train, X_test, preprocessor, feature_names, target_log_transformed
    """
    logger.info("Running data preprocessing...")
    
    # Handle missing values
    train_df = handle_missing_values(train_df, strategy='advanced')
    if test_df is not None:
        test_df = handle_missing_values(test_df, strategy='advanced')
    
    # Handle outliers
    train_df = handle_outliers(train_df, method='iqr')
    
    # Preprocess data
    X_train, y_train, X_test, preprocessor, feature_names = preprocess_data(
        train_df, test_df, target_column='SalePrice'
    )
    
    # Apply log transformation to the target variable
    if y_train is not None:
        # Check for non-positive values
        if np.any(y_train <= 0):
            logger.warning(f"Found {np.sum(y_train <= 0)} non-positive values in target variable. Replacing with small positive values.")
            y_train = np.maximum(y_train, 1.0)  # Replace non-positive values with 1.0
        
        # Apply log transformation
        y_train = np.log1p(y_train)
        logger.info(f"Applied log transformation to target variable. Range: [{y_train.min():.4f}, {y_train.max():.4f}], Mean: {y_train.mean():.4f}")
        target_log_transformed = True
    else:
        target_log_transformed = False
    
    logger.info(f"Data preprocessing completed. X_train shape: {X_train.shape}")
    if X_test is not None:
        logger.info(f"X_test shape: {X_test.shape}")
    
    return X_train, y_train, X_test, preprocessor, feature_names, target_log_transformed

def run_feature_engineering(train_df, test_df=None):
    """
    Run feature engineering on the training and test data.
    
    Args:
        train_df (pandas.DataFrame): Training dataframe
        test_df (pandas.DataFrame, optional): Test dataframe
        
    Returns:
        tuple: Processed train_df, test_df
    """
    logger.info("Running feature engineering...")
    
    # Create feature engineer
    feature_engineer = FeatureEngineer(
        create_area=True,
        create_age=True,
        create_quality=True,
        create_bathroom=True,
        create_has=True,
        create_interaction=True
    )
    
    # Apply feature engineering to training data
    train_df = feature_engineer.transform(train_df)
    
    # Apply feature engineering to test data if provided
    if test_df is not None:
        test_df = feature_engineer.transform(test_df)
    
    logger.info(f"Feature engineering completed. Train shape: {train_df.shape}")
    if test_df is not None:
        logger.info(f"Test shape: {test_df.shape}")
    
    return train_df, test_df

def run_model_training(X_train, y_train, feature_names, output_dir):
    """
    Run model training and evaluation.
    
    Args:
        X_train (array-like): Training features
        y_train (array-like): Training target
        feature_names (list): List of feature names
        output_dir (Path): Output directory
        
    Returns:
        dict: Dictionary of trained models
    """
    logger.info("Running model training...")
    
    # Get base models
    models = get_base_models()
    
    # Compare models
    results_df = compare_models(models, X_train, y_train, cv=5)
    logger.info("Model comparison results:")
    logger.info("\n" + results_df.to_string())
    
    # Save results to CSV
    results_path = output_dir / 'models' / 'model_comparison.csv'
    results_df.to_csv(results_path, index=False)
    logger.info(f"Model comparison results saved to {results_path}")
    
    # Plot model comparison
    fig = plot_models_comparison(results_df, metric='RMSE')
    save_figure(fig, 'model_comparison.png', output_dir)
    
    # Train and save best model
    best_model_name = results_df.iloc[0]['Model']
    best_model = models[best_model_name]
    
    logger.info(f"Training best model: {best_model_name}")
    trained_model = train_and_save_model(best_model, X_train, y_train, best_model_name, feature_names)
    
    # Get feature importance
    if hasattr(trained_model, 'feature_importances_') or hasattr(trained_model, 'coef_'):
        importance_df = get_feature_importance(trained_model, feature_names)
        logger.info("Feature importance:")
        logger.info("\n" + importance_df.head(10).to_string())
        
        # Plot feature importance
        fig = plot_feature_importance(importance_df['Feature'], importance_df['Importance'])
        save_figure(fig, 'feature_importance.png', output_dir)
    
    # Tune hyperparameters for best model
    logger.info(f"Tuning hyperparameters for {best_model_name}")
    tuned_model, best_params, search = tune_hyperparameters(
        models[best_model_name], best_model_name, X_train, y_train, method='random', n_iter=50
    )
    
    # Train and save tuned model
    logger.info(f"Training tuned {best_model_name}")
    tuned_model = train_and_save_model(tuned_model, X_train, y_train, f"{best_model_name}_tuned", feature_names)
    
    # Plot learning curves for tuned model
    fig = plot_learning_curves(tuned_model, X_train, y_train)
    save_figure(fig, 'learning_curves.png', output_dir)
    
    # Return trained models
    trained_models = {
        best_model_name: trained_model,
        f"{best_model_name}_tuned": tuned_model
    }
    
    logger.info("Model training completed")
    
    return trained_models

def run_model_evaluation(models, X_train, y_train, output_dir):
    """
    Run model evaluation.
    
    Args:
        models (dict): Dictionary of trained models
        X_train (array-like): Training features
        y_train (array-like): Training target
        output_dir (Path): Output directory
    """
    logger.info("Running model evaluation...")
    
    # Evaluate each model
    for name, model in models.items():
        logger.info(f"Evaluating {name}")
        
        # Make predictions
        y_pred = model.predict(X_train)
        
        # Calculate metrics
        metrics = calculate_metrics(y_train, y_pred)
        logger.info(f"Metrics for {name}:")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")
        
        # Plot residuals analysis
        fig = plot_residuals_analysis(y_train, y_pred)
        save_figure(fig, f'{name}_residuals.png', output_dir)
        
        # Plot error distribution
        fig = plot_error_distribution(y_train, y_pred)
        save_figure(fig, f'{name}_error_distribution.png', output_dir)
    
    logger.info("Model evaluation completed")

def run_prediction(model, X_test, test_ids, output_dir, target_log_transformed=True):
    """
    Run prediction on test data and create submission file.
    
    Args:
        model: Trained model
        X_test (array-like): Test features
        test_ids (array-like): Test IDs
        output_dir (Path): Output directory
        target_log_transformed (bool): Whether the target variable was log-transformed
        
    Returns:
        str: Path to the submission file
    """
    logger.info("Running prediction on test data...")
    
    # Make predictions
    predictions = predict(model, X_test)
    
    # Log prediction statistics before transformation
    logger.info(f"Raw predictions - min: {np.min(predictions):.4f}, max: {np.max(predictions):.4f}, mean: {np.mean(predictions):.4f}")
    
    # Check for NaN or infinite values in raw predictions
    if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
        logger.warning(f"Found {np.sum(np.isnan(predictions))} NaN and {np.sum(np.isinf(predictions))} infinite values in raw predictions")
        # Replace NaN and infinite values with the mean of valid predictions
        valid_preds = predictions[~np.isnan(predictions) & ~np.isinf(predictions)]
        if len(valid_preds) > 0:
            replacement_value = np.mean(valid_preds)
        else:
            replacement_value = 12.0  # Fallback value (approximately mean of log-transformed SalePrice)
        predictions = np.nan_to_num(predictions, nan=replacement_value, posinf=replacement_value, neginf=replacement_value)
    
    # If we used log transform on the target, we need to transform back
    if target_log_transformed:
        logger.info("Applying inverse log transformation to predictions")
        # Clip predictions to a safe range before applying expm1 to prevent overflow
        predictions = np.clip(predictions, 0, 30)  # Maximum reasonable log price ~30 (corresponds to ~$10^13)
        predictions = np.expm1(predictions)
    
    # Log prediction statistics after transformation
    logger.info(f"Transformed predictions - min: {np.min(predictions):.4f}, max: {np.max(predictions):.4f}, mean: {np.mean(predictions):.4f}")
    
    # Check for NaN or infinite values after transformation
    if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
        logger.warning(f"Found {np.sum(np.isnan(predictions))} NaN and {np.sum(np.isinf(predictions))} infinite values after transformation")
        # Replace NaN and infinite values with reasonable values
        valid_preds = predictions[~np.isnan(predictions) & ~np.isinf(predictions)]
        if len(valid_preds) > 0:
            replacement_value = np.mean(valid_preds)
        else:
            replacement_value = 180000.0  # Fallback value (approximately mean SalePrice)
        predictions = np.nan_to_num(predictions, nan=replacement_value, posinf=replacement_value, neginf=replacement_value)
    
    # Clip predictions to reasonable range for house prices
    # Assuming reasonable range is $10,000 to $1,000,000
    predictions = np.clip(predictions, 10000, 1000000)
    
    # Create submission file
    submission_path = output_dir / 'submissions' / 'submission.csv'
    create_submission_file(predictions, test_ids, filename=str(submission_path))
    
    logger.info(f"Prediction completed. Submission file saved to {submission_path}")
    
    return submission_path

def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='House Price Prediction Pipeline')
    
    parser.add_argument('--skip-eda', action='store_true', help='Skip exploratory data analysis')
    parser.add_argument('--skip-preprocessing', action='store_true', help='Skip data preprocessing')
    parser.add_argument('--skip-feature-engineering', action='store_true', help='Skip feature engineering')
    parser.add_argument('--skip-model-training', action='store_true', help='Skip model training')
    parser.add_argument('--skip-model-evaluation', action='store_true', help='Skip model evaluation')
    parser.add_argument('--skip-prediction', action='store_true', help='Skip prediction on test data')
    
    return parser.parse_args()

def main():
    """
    Main function to run the pipeline.
    """
    # Start timer
    start_time = time.time()
    
    # Parse arguments
    args = parse_arguments()
    
    # Create output directory
    output_dir = create_output_directory()
    
    # Load data
    logger.info("Loading data...")
    train_df = load_train_data()
    test_df = load_test_data()
    test_ids = test_df['Id'].copy()
    
    # Run exploratory data analysis
    if not args.skip_eda:
        run_exploratory_data_analysis(train_df, output_dir)
    
    # Run feature engineering
    if not args.skip_feature_engineering:
        train_df, test_df = run_feature_engineering(train_df, test_df)
    
    # Run data preprocessing
    target_log_transformed = True  # Default value
    if not args.skip_preprocessing:
        X_train, y_train, X_test, preprocessor, feature_names, target_log_transformed = run_data_preprocessing(train_df, test_df)
    
    # Run model training
    trained_models = None
    if not args.skip_model_training:
        trained_models = run_model_training(X_train, y_train, feature_names, output_dir)
    
    # Run model evaluation
    if not args.skip_model_evaluation and trained_models is not None:
        run_model_evaluation(trained_models, X_train, y_train, output_dir)
    
    # Run prediction on test data
    if not args.skip_prediction and trained_models is not None:
        # Use the tuned model for prediction
        best_model_name = list(trained_models.keys())[0]
        tuned_model_name = f"{best_model_name}_tuned"
        
        if tuned_model_name in trained_models:
            best_model = trained_models[tuned_model_name]
        else:
            best_model = trained_models[best_model_name]
        
        run_prediction(best_model, X_test, test_ids, output_dir, target_log_transformed)
    
    # Calculate total runtime
    total_time = time.time() - start_time
    logger.info(f"Pipeline completed in {total_time:.2f} seconds")

if __name__ == "__main__":
    main() 