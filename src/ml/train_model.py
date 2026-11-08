#!/usr/bin/env python3
"""
House Price Prediction Model Training Pipeline
Trains and fine-tunes ML models with 70/30 train/test split.
"""

import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
from typing import Dict, Any, Tuple, List

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.impute import SimpleImputer
import xgboost as xgb
import lightgbm as lgb

# Paths
DATA_DIR = Path(__file__).parent.parent.parent / "data"
MODEL_DIR = Path(__file__).parent.parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = DATA_DIR / "train_data.csv"
TEST_PATH = DATA_DIR / "test_data.csv"
MODEL_PATH = MODEL_DIR / "house_price_model.pkl"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.pkl"
METRICS_PATH = MODEL_DIR / "model_metrics.json"
FEATURE_IMPORTANCE_PATH = MODEL_DIR / "feature_importance.json"

# Feature configuration
TARGET = 'price'

# Features to drop (not predictive or leak target)
DROP_FEATURES = [
    'sample_id',
    'price',  # target
    'price_5yr_projection',  # derived from target
    'loan_amount',  # derived from price and down_payment
    'total_interest_paid',  # derived
    'max_affordable_price',  # derived from income
    'amenity_value',  # derived from amenities
    'down_payment_amount',  # derived from price
]

CATEGORICAL_FEATURES = [
    'state', 'location_type', 'property_type',
    'affordability_category', 'user_needs'
]

NUMERICAL_FEATURES = [
    'property_age', 'square_footage', 'bedrooms', 'bathrooms',
    'down_payment_pct', 'months_to_save_down_payment', 'annual_income',
    'interest_rate', 'loan_term_years', 'monthly_mortgage_payment',
    'credit_score', 'inflation_rate_assumed',
    'dti_ratio', 'recommended_down_payment_pct',
    'amenity_count'
]

# Amenities is text, will be processed separately
TEXT_FEATURES = ['amenities']

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and test data."""
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
    return train_df, test_df

def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepare features and target from dataframe."""
    # Drop non-predictive features
    X = df.drop(columns=[c for c in DROP_FEATURES if c in df.columns], errors='ignore')
    y = df[TARGET]

    # Process amenities text feature - count and binary flags for top amenities
    if 'amenities' in X.columns:
        # Create binary features for common amenities
        all_amenities = set()
        for am in X['amenities'].fillna(''):
            all_amenities.update([a.strip() for a in am.split(',') if a.strip()])

        top_amenities = ['pool', 'garage', 'garden', 'fireplace', 'central_ac',
                         'heating', 'basement', 'security_system', 'smart_home',
                         'solar_panels', 'parking']

        for amenity in top_amenities:
            X[f'has_{amenity}'] = X['amenities'].fillna('').apply(lambda x: 1 if amenity in x else 0)

        X = X.drop(columns=['amenities'])

    return X, y

def create_preprocessor() -> ColumnTransformer:
    """Create the preprocessing pipeline."""
    # Numerical pipeline
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Categorical pipeline
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Combine
    preprocessor = ColumnTransformer([
        ('num', numerical_pipeline, NUMERICAL_FEATURES),
        ('cat', categorical_pipeline, CATEGORICAL_FEATURES)
    ], remainder='passthrough')  # Pass through the amenity binary features

    return preprocessor

def get_models() -> Dict[str, Any]:
    """Get dictionary of models to try."""
    return {
        'random_forest': RandomForestRegressor(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        ),
        'gradient_boosting': GradientBoostingRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        ),
        'extra_trees': ExtraTreesRegressor(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        ),
        'xgboost': xgb.XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        ),
        'lightgbm': lgb.LGBMRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbosity=-1
        ),
        'ridge': Ridge(alpha=1.0),
        'lasso': Lasso(alpha=0.1, max_iter=2000),
        'elastic_net': ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=2000)
    }

def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate evaluation metrics."""
    return {
        'mse': float(mean_squared_error(y_true, y_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'r2': float(r2_score(y_true, y_pred)),
        'mape': float(mean_absolute_percentage_error(y_true, y_pred))
    }

def train_and_evaluate(X_train: pd.DataFrame, y_train: pd.Series,
                       X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    """Train multiple models and select the best."""
    preprocessor = create_preprocessor()

    models = get_models()
    results = {}

    print("Training models...")
    for name, model in models.items():
        print(f"  Training {name}...")

        # Create pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])

        # Train
        pipeline.fit(X_train, y_train)

        # Predict
        y_pred_train = pipeline.predict(X_train)
        y_pred_test = pipeline.predict(X_test)

        # Evaluate
        train_metrics = evaluate_model(y_train.values, y_pred_train)
        test_metrics = evaluate_model(y_test.values, y_pred_test)

        # Cross-validation
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='neg_root_mean_squared_error', n_jobs=-1)
        cv_rmse = -cv_scores.mean()
        cv_std = cv_scores.std()

        results[name] = {
            'pipeline': pipeline,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'cv_rmse': float(cv_rmse),
            'cv_std': float(cv_std)
        }

        print(f"    Test RMSE: {test_metrics['rmse']:.2f}, R²: {test_metrics['r2']:.4f}, CV RMSE: {cv_rmse:.2f}±{cv_std:.2f}")

    return results

def fine_tune_best_model(X_train: pd.DataFrame, y_train: pd.Series,
                          best_model_name: str, best_pipeline: Pipeline) -> Pipeline:
    """Fine-tune the best model using RandomizedSearchCV."""
    print(f"\nFine-tuning {best_model_name}...")

    # Parameter grids for different models
    param_grids = {
        'random_forest': {
            'model__n_estimators': [200, 300, 500],
            'model__max_depth': [15, 20, 25, None],
            'model__min_samples_split': [2, 5, 10],
            'model__min_samples_leaf': [1, 2, 4],
            'model__max_features': ['sqrt', 'log2', None]
        },
        'gradient_boosting': {
            'model__n_estimators': [200, 300, 500],
            'model__max_depth': [4, 6, 8],
            'model__learning_rate': [0.05, 0.1, 0.15],
            'model__subsample': [0.7, 0.8, 0.9],
            'model__min_samples_split': [2, 5, 10]
        },
        'extra_trees': {
            'model__n_estimators': [200, 300, 500],
            'model__max_depth': [15, 20, 25, None],
            'model__min_samples_split': [2, 5, 10],
            'model__min_samples_leaf': [1, 2, 4],
            'model__max_features': ['sqrt', 'log2', None]
        },
        'xgboost': {
            'model__n_estimators': [300, 500, 800],
            'model__max_depth': [4, 6, 8, 10],
            'model__learning_rate': [0.03, 0.05, 0.1],
            'model__subsample': [0.7, 0.8, 0.9],
            'model__colsample_bytree': [0.7, 0.8, 0.9],
            'model__reg_alpha': [0, 0.1, 1],
            'model__reg_lambda': [0, 0.1, 1]
        },
        'lightgbm': {
            'model__n_estimators': [300, 500, 800],
            'model__max_depth': [4, 6, 8, 10, -1],
            'model__learning_rate': [0.03, 0.05, 0.1],
            'model__subsample': [0.7, 0.8, 0.9],
            'model__colsample_bytree': [0.7, 0.8, 0.9],
            'model__reg_alpha': [0, 0.1, 1],
            'model__reg_lambda': [0, 0.1, 1],
            'model__num_leaves': [31, 63, 127]
        }
    }

    param_grid = param_grids.get(best_model_name, {})

    if not param_grid:
        print(f"  No parameter grid for {best_model_name}, skipping fine-tuning")
        return best_pipeline

    # Randomized search
    search = RandomizedSearchCV(
        best_pipeline,
        param_grid,
        n_iter=30,
        cv=3,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )

    search.fit(X_train, y_train)

    print(f"  Best params: {search.best_params_}")
    print(f"  Best CV RMSE: {-search.best_score_:.2f}")

    return search.best_estimator_

def get_feature_importance(pipeline: Pipeline, feature_names: List[str]) -> Dict[str, float]:
    """Extract feature importance from the trained model."""
    model = pipeline.named_steps['model']
    preprocessor = pipeline.named_steps['preprocessor']

    # Get feature names after preprocessing
    # Numerical features keep their names
    # Categorical features get expanded
    cat_encoder = preprocessor.named_transformers_['cat'].named_steps['encoder']
    cat_feature_names = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES)

    all_feature_names = NUMERICAL_FEATURES + list(cat_feature_names)

    # Add amenity binary features
    amenity_features = [f'has_{a}' for a in ['pool', 'garage', 'garden', 'fireplace', 'central_ac',
                                              'heating', 'basement', 'security_system', 'smart_home',
                                              'solar_panels', 'parking']]
    all_feature_names += amenity_features

    # Get importance
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_)
    else:
        return {}

    # Create dictionary
    importance_dict = dict(zip(all_feature_names, importances))
    # Sort by importance
    importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))

    return importance_dict

def save_model(pipeline: Pipeline, metrics: Dict, importance: Dict):
    """Save model, preprocessor, metrics, and feature importance."""
    # Save full pipeline
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    # Save preprocessor separately for API use
    joblib.dump(pipeline.named_steps['preprocessor'], PREPROCESSOR_PATH)
    print(f"Preprocessor saved to {PREPROCESSOR_PATH}")

    # Save metrics
    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {METRICS_PATH}")

    # Save feature importance
    with open(FEATURE_IMPORTANCE_PATH, 'w') as f:
        json.dump(importance, f, indent=2)
    print(f"Feature importance saved to {FEATURE_IMPORTANCE_PATH}")

def main():
    print("=" * 60)
    print("HOUSE PRICE PREDICTOR - MODEL TRAINING PIPELINE")
    print("=" * 60)

    # Load data
    train_df, test_df = load_data()

    # Prepare features
    X_train, y_train = prepare_features(train_df)
    X_test, y_test = prepare_features(test_df)

    print(f"\nFeature shapes - Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Features: {list(X_train.columns)}")

    # Train and evaluate models
    results = train_and_evaluate(X_train, y_train, X_test, y_test)

    # Find best model based on test RMSE
    best_model_name = min(results.keys(), key=lambda k: results[k]['test_metrics']['rmse'])
    best_pipeline = results[best_model_name]['pipeline']
    best_metrics = results[best_model_name]['test_metrics']

    print(f"\n{'='*60}")
    print(f"BEST MODEL: {best_model_name}")
    print(f"{'='*60}")
    print(f"Test Metrics: {best_metrics}")

    # Fine-tune best model
    tuned_pipeline = fine_tune_best_model(X_train, y_train, best_model_name, best_pipeline)

    # Evaluate tuned model
    y_pred_tuned = tuned_pipeline.predict(X_test)
    tuned_metrics = evaluate_model(y_test.values, y_pred_tuned)
    print(f"\nTuned Model Test Metrics: {tuned_metrics}")

    # Get feature importance
    feature_names = list(X_train.columns)
    importance = get_feature_importance(tuned_pipeline, feature_names)
    print(f"\nTop 20 Features:")
    for i, (feat, imp) in enumerate(list(importance.items())[:20]):
        print(f"  {i+1}. {feat}: {imp:.6f}")

    # Save everything
    save_data = {
        'best_model': best_model_name,
        'test_metrics': tuned_metrics,
        'all_results': {k: {
            'train_metrics': v['train_metrics'],
            'test_metrics': v['test_metrics'],
            'cv_rmse': v['cv_rmse'],
            'cv_std': v['cv_std']
        } for k, v in results.items()}
    }
    save_model(tuned_pipeline, save_data, importance)

    print(f"\n{'='*60}")
    print("TRAINING COMPLETE!")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()