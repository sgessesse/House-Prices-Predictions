# House Price Prediction Project

## Overview
This project aims to predict house prices using advanced regression techniques. The dataset contains 79 explanatory variables describing various aspects of residential homes in Ames, Iowa. The goal is to predict the final price of each home.

## Project Structure
```
├── Data/                      # Raw data files
│   ├── train.csv              # Training data
│   ├── test.csv               # Test data
│   ├── data_description.txt   # Description of all features
│   └── sample_submission.csv  # Sample submission format
├── notebooks/                 # Jupyter notebooks
│   └── exploratory_analysis.ipynb  # EDA and visualization
├── output/                    # Output files
│   ├── figures/               # Visualizations and plots
│   ├── models/                # Trained models and metrics
│   └── submissions/           # Prediction submissions
├── src/                       # Source code
│   ├── data/                  # Data processing modules
│   │   ├── __init__.py
│   │   ├── loader.py          # Data loading functions
│   │   └── preprocessor.py    # Data preprocessing functions
│   ├── features/              # Feature engineering
│   │   ├── __init__.py
│   │   └── engineer.py        # Feature creation and transformation
│   ├── models/                # Model training and evaluation
│   │   ├── __init__.py
│   │   ├── train.py           # Model training functions
│   │   ├── evaluate.py        # Model evaluation functions
│   │   └── hyperparameter_tuning.py  # Hyperparameter optimization
│   └── visualization/         # Visualization utilities
│       ├── __init__.py
│       └── plots.py           # Plotting functions
├── main.py                    # Main script to run the pipeline
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation
```

## Installation
```bash
# Clone the repository
git clone https://github.com/sgessesse/House-Prices-Predictions.git
cd House-Prices-Predictions

# Create a virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage
1. Run the main script to execute the entire pipeline:
```bash
python main.py
```

2. You can also run specific parts of the pipeline using command-line arguments:
```bash
# Skip exploratory data analysis
python main.py --skip-eda

# Skip model training
python main.py --skip-model-training

# Skip prediction
python main.py --skip-prediction
```

3. Or explore the notebooks for detailed analysis:
```bash
jupyter notebook notebooks/exploratory_analysis.ipynb
```

## Methodology
1. **Data Preprocessing**: 
   - Handled missing values using appropriate imputation strategies
   - Encoded categorical variables using one-hot encoding
   - Normalized numerical features using standard scaling
   - Applied log transformation to the target variable (SalePrice)

2. **Exploratory Data Analysis**: 
   - Visualized distributions of key features
   - Analyzed correlations between features and target variable
   - Identified and handled outliers

3. **Feature Engineering**: 
   - Created total area feature by combining living area, basement, and garage areas
   - Generated age-related features (house age, remodel age)
   - Created quality-related features by combining quality indicators
   - Added interaction features between important variables

4. **Model Selection**: 
   - Compared multiple regression models using cross-validation
   - Evaluated models based on RMSE, R², and MAE metrics

5. **Hyperparameter Tuning**: 
   - Optimized the best-performing models using grid search
   - Analyzed learning curves to prevent overfitting

6. **Evaluation**: 
   - Assessed model performance using cross-validation
   - Analyzed residuals and error distributions

## Models Implemented
- Linear Regression
- Ridge Regression
- Lasso Regression
- Elastic Net
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM

## Results
After evaluating multiple regression models, the Elastic Net model achieved the best performance:

| Model | RMSE | R² | MAE |
|-------|------|----|----|
| Elastic Net | 0.1231 | 0.9015 | 0.0815 |
| Lasso Regression | 0.1254 | 0.8980 | 0.0830 |
| Gradient Boosting | 0.1282 | 0.8933 | 0.0856 |
| Ridge Regression | 0.1288 | 0.8917 | 0.0846 |
| XGBoost | 0.1312 | 0.8877 | 0.0868 |
| LightGBM | 0.1315 | 0.8884 | 0.0874 |
| Linear Regression | 0.1339 | 0.8826 | 0.0873 |
| Random Forest | 0.1367 | 0.8794 | 0.0917 |

After hyperparameter tuning, the final Elastic Net model achieved:
- RMSE: 0.1073
- R²: 0.9278
- MAE: 0.0729
- MAPE: 0.6105%

### Key Features
The most important features for predicting house prices were:
1. MSZoning_C (all) - Commercial zoning
2. RoofMatl_ClyTile - Clay tile roof material
3. TotalArea - Combined area of the house
4. Condition2_PosN - Proximity to positive features
5. Neighborhood_Crawfor - Crawford neighborhood

### Visualizations
The project generated various visualizations to aid in understanding the data and model performance:
- Feature distributions
- Correlation matrix
- Feature importance
- Residual plots
- Error distributions
- Model comparison

## Future Improvements
1. Implement stacking or blending of multiple models
2. Explore more advanced feature engineering techniques
3. Experiment with neural network approaches
4. Incorporate external data sources (e.g., economic indicators)
5. Implement feature selection to reduce dimensionality

## License
This project is licensed under the MIT License - see the LICENSE file for details. 