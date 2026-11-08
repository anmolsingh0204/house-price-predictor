#!/usr/bin/env python3
"""
Unit tests for the ML pipeline - data generation, preprocessing, training, and prediction.
"""

import pytest
import numpy as np
import pandas as pd
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.train_model import (
    generate_dataset,
    prepare_features,
    create_preprocessor,
    get_models,
    evaluate_model,
    load_data
)
from data.generate_dataset import (
    generate_base_price,
    generate_amenities_score,
    calculate_down_payment,
    calculate_loan_details,
    calculate_inflation_adjustment,
    calculate_affordability,
    generate_dataset as gen_dataset
)


class TestDataGeneration:
    """Tests for synthetic data generation."""

    def test_generate_base_price(self):
        """Test base price generation with various inputs."""
        price = generate_base_price('CA', 'urban', 'single_family', 2000, 10, 3, 2)
        assert price > 0
        assert isinstance(price, float)

    def test_state_multipliers(self):
        """Test that state multipliers affect price."""
        ca_price = generate_base_price('CA', 'suburban', 'single_family', 2000, 20, 3, 2)
        ms_price = generate_base_price('MS', 'suburban', 'single_family', 2000, 20, 3, 2)
        assert ca_price > ms_price

    def test_location_type_multipliers(self):
        """Test location type multipliers."""
        urban_price = generate_base_price('TX', 'urban', 'single_family', 2000, 20, 3, 2)
        rural_price = generate_base_price('TX', 'rural', 'single_family', 2000, 20, 3, 2)
        assert urban_price > rural_price

    def test_property_type_multipliers(self):
        """Test property type multipliers."""
        sf_price = generate_base_price('TX', 'suburban', 'single_family', 2000, 20, 3, 2)
        mh_price = generate_base_price('TX', 'suburban', 'mobile_home', 2000, 20, 3, 2)
        assert sf_price > mh_price

    def test_generate_amenities_score(self):
        """Test amenities generation."""
        amenities, value = generate_amenities_score()
        assert isinstance(amenities, list)
        assert isinstance(value, (int, float))
        assert value >= 0

    def test_calculate_down_payment(self):
        """Test down payment calculation."""
        price = 400000
        dp, dp_pct, months, income = calculate_down_payment(price, 0.20)
        assert dp == 80000
        assert dp_pct == 0.20
        assert months > 0
        assert income > 0

    def test_calculate_loan_details(self):
        """Test loan calculation."""
        price = 400000
        down = 80000
        loan = calculate_loan_details(price, down, 720)

        assert loan['loan_amount'] == 320000
        assert loan['interest_rate'] > 0
        assert loan['term_years'] in [15, 20, 30]
        assert loan['monthly_payment'] > 0
        assert loan['total_interest'] > 0
        assert 580 <= loan['credit_score'] <= 850

    def test_calculate_inflation_adjustment(self):
        """Test inflation projection."""
        price = 400000
        price_5yr = calculate_inflation_adjustment(price, 5)
        assert price_5yr > price
        # ~4% annually for 5 years = ~21.6% total
        expected = price * (1.04 ** 5)
        assert abs(price_5yr - expected) < 1

    def test_calculate_affordability(self):
        """Test affordability calculation."""
        income = 100000
        payment = 2000
        result = calculate_affordability(income, payment)

        assert 'dti_ratio' in result
        assert 'affordability_category' in result
        assert 'max_affordable_price' in result
        assert result['dti_ratio'] == 0.24  # 2000 / (100000/12)
        assert result['affordability_category'] == 'excellent'

    def test_generate_dataset_shape(self):
        """Test dataset generation produces correct shape."""
        df = gen_dataset(n_samples=100)
        assert len(df) == 100
        assert 'price' in df.columns
        assert 'property_age' in df.columns
        assert 'state' in df.columns
        assert len(df.columns) > 20

    def test_dataset_has_all_states(self):
        """Test all states are represented."""
        df = gen_dataset(n_samples=1000)
        unique_states = df['state'].unique()
        assert len(unique_states) > 40  # Most states should appear

    def test_dataset_no_nulls(self):
        """Test dataset has no null values in key columns."""
        df = gen_dataset(n_samples=100)
        key_cols = ['price', 'property_age', 'square_footage', 'state', 'annual_income']
        for col in key_cols:
            assert df[col].notna().all()


class TestFeaturePreparation:
    """Tests for feature preparation."""

    def test_prepare_features_drops_columns(self):
        """Test that non-predictive columns are dropped."""
        df = gen_dataset(n_samples=50)
        X, y = prepare_features(df)

        assert 'price' not in X.columns
        assert 'sample_id' not in X.columns
        assert 'loan_amount' not in X.columns
        assert len(y) == len(df)
        assert (y == df['price']).all()

    def test_prepare_features_creates_amenity_flags(self):
        """Test amenity binary features are created."""
        df = gen_dataset(n_samples=50)
        X, y = prepare_features(df)

        amenity_cols = [c for c in X.columns if c.startswith('has_')]
        assert len(amenity_cols) > 0
        assert 'has_pool' in amenity_cols
        assert 'has_garage' in amenity_cols

    def test_prepare_features_preserves_numerical(self):
        """Test numerical features are preserved."""
        df = gen_dataset(n_samples=50)
        X, y = prepare_features(df)

        for col in ['property_age', 'square_footage', 'bedrooms', 'bathrooms', 'annual_income']:
            assert col in X.columns


class TestPreprocessor:
    """Tests for the preprocessing pipeline."""

    def test_create_preprocessor(self):
        """Test preprocessor creation."""
        preprocessor = create_preprocessor()
        assert preprocessor is not None
        assert hasattr(preprocessor, 'transform')

    def test_preprocessor_fit_transform(self):
        """Test preprocessor fit and transform."""
        df = gen_dataset(n_samples=100)
        X, y = prepare_features(df)

        preprocessor = create_preprocessor()
        X_transformed = preprocessor.fit_transform(X)

        assert X_transformed.shape[0] == X.shape[0]
        assert X_transformed.shape[1] > 0
        assert not np.isnan(X_transformed).any()


class TestModels:
    """Tests for model definitions and evaluation."""

    def test_get_models(self):
        """Test model dictionary creation."""
        models = get_models()
        assert isinstance(models, dict)
        assert len(models) >= 5
        assert 'random_forest' in models
        assert 'xgboost' in models
        assert 'lightgbm' in models

    def test_evaluate_model(self):
        """Test evaluation metrics."""
        y_true = np.array([100, 200, 300, 400, 500])
        y_pred = np.array([110, 190, 310, 390, 510])

        metrics = evaluate_model(y_true, y_pred)

        assert 'mse' in metrics
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'r2' in metrics
        assert 'mape' in metrics
        assert metrics['r2'] > 0.9  # Should be high for close predictions


class TestTrainTestSplit:
    """Tests for train/test split."""

    def test_load_data_creates_split(self):
        """Test that data loading works with generated files."""
        # This would need actual files - skip if not present
        train_path = Path(__file__).parent.parent.parent / "data" / "train_data.csv"
        test_path = Path(__file__).parent.parent.parent / "data" / "test_data.csv"

        if train_path.exists() and test_path.exists():
            train_df, test_df = load_data()
            assert len(train_df) > 0
            assert len(test_df) > 0
            # 70/30 split approximately
            total = len(train_df) + len(test_df)
            train_ratio = len(train_df) / total
            assert 0.65 < train_ratio < 0.75


class TestIntegration:
    """Integration tests for the full pipeline."""

    def test_full_pipeline_small_dataset(self):
        """Test training on a small dataset."""
        df = gen_dataset(n_samples=200)
        X, y = prepare_features(df)

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        preprocessor = create_preprocessor()
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.pipeline import Pipeline

        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', RandomForestRegressor(n_estimators=10, max_depth=5, random_state=42))
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        metrics = evaluate_model(y_test.values, y_pred)
        assert metrics['r2'] > 0.5  # Should have some predictive power

    def test_prediction_output_format(self):
        """Test prediction returns expected format."""
        df = gen_dataset(n_samples=100)
        X, y = prepare_features(df)

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        preprocessor = create_preprocessor()
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.pipeline import Pipeline

        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', RandomForestRegressor(n_estimators=10, random_state=42))
        ])

        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test.iloc[:1])

        assert isinstance(pred[0], (float, np.floating))
        assert pred[0] > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])