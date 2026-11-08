#!/usr/bin/env python3
"""
Unit tests for the REST API endpoints.
"""

import pytest
import json
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.app import (
    prepare_input_features,
    make_prediction,
    create_app,
    STATES,
    LOCATION_TYPES,
    PROPERTY_TYPES,
    AMENITIES_LIST
)


class TestAPIUtilities:
    """Tests for API utility functions."""

    def test_prepare_input_features_basic(self):
        """Test basic feature preparation."""
        data = {
            'property_age': 20,
            'square_footage': 2000,
            'bedrooms': 3,
            'bathrooms': 2.0,
            'state': 'CA',
            'location_type': 'suburban',
            'property_type': 'single_family',
            'down_payment_pct': 0.20,
            'annual_income': 100000,
            'credit_score': 720,
            'loan_term_years': 30,
            'amenities': ['pool', 'garage'],
            'user_needs': 'general'
        }

        df = prepare_input_features(data)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert df['property_age'].iloc[0] == 20
        assert df['state'].iloc[0] == 'CA'
        assert df['has_pool'].iloc[0] == 1
        assert df['has_garage'].iloc[0] == 1
        assert df['has_garden'].iloc[0] == 0

    def test_prepare_input_features_calculates_interest_rate(self):
        """Test interest rate calculation based on credit score."""
        # Excellent credit
        data = {'credit_score': 780, 'annual_income': 100000}
        df = prepare_input_features(data)
        assert df['interest_rate'].iloc[0] == 0.055  # 6.5% - 1%

        # Good credit
        data = {'credit_score': 720, 'annual_income': 100000}
        df = prepare_input_features(data)
        assert df['interest_rate'].iloc[0] == 0.060  # 6.5% - 0.5%

        # Fair credit
        data = {'credit_score': 680, 'annual_income': 100000}
        df = prepare_input_features(data)
        assert df['interest_rate'].iloc[0] == 0.065  # base rate

        # Poor credit
        data = {'credit_score': 600, 'annual_income': 100000}
        df = prepare_input_features(data)
        assert df['interest_rate'].iloc[0] == 0.080  # 6.5% + 1.5%

    def test_prepare_input_features_defaults(self):
        """Test default values are applied."""
        data = {'annual_income': 100000}
        df = prepare_input_features(data)

        assert df['property_age'].iloc[0] == 20
        assert df['square_footage'].iloc[0] == 2000
        assert df['bedrooms'].iloc[0] == 3
        assert df['down_payment_pct'].iloc[0] == 0.20
        assert df['loan_term_years'].iloc[0] == 30


class TestAPIConstants:
    """Tests for API constants."""

    def test_states_list(self):
        """Test states list has all 50 states + DC."""
        assert len(STATES) == 51
        assert 'CA' in STATES
        assert 'NY' in STATES
        assert 'TX' in STATES
        assert 'DC' in STATES

    def test_location_types(self):
        """Test location types."""
        assert LOCATION_TYPES == ['urban', 'suburban', 'rural']

    def test_property_types(self):
        """Test property types."""
        assert 'single_family' in PROPERTY_TYPES
        assert 'condo' in PROPERTY_TYPES
        assert 'townhouse' in PROPERTY_TYPES

    def test_amenities_list(self):
        """Test amenities list."""
        assert 'pool' in AMENITIES_LIST
        assert 'garage' in AMENITIES_LIST
        assert 'smart_home' in AMENITIES_LIST
        assert len(AMENITIES_LIST) >= 10


class TestFlaskApp:
    """Tests for Flask application endpoints."""

    @pytest.fixture
    def app(self):
        """Create test Flask app."""
        app = create_app()
        app.config['TESTING'] = True
        return app

    @pytest.fixture
    def client(self, app):
        """Create test client."""
        return app.test_client()

    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get('/health')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'healthy'
        assert 'timestamp' in data

    def test_get_states_endpoint(self, client):
        """Test states list endpoint."""
        response = client.get('/api/states')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] is True
        assert len(data['data']) == 51

    def test_get_location_types_endpoint(self, client):
        """Test location types endpoint."""
        response = client.get('/api/location-types')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] is True
        assert data['data'] == ['urban', 'suburban', 'rural']

    def test_get_property_types_endpoint(self, client):
        """Test property types endpoint."""
        response = client.get('/api/property-types')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] is True
        assert 'single_family' in data['data']

    def test_get_amenities_endpoint(self, client):
        """Test amenities endpoint."""
        response = client.get('/api/amenities')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] is True
        assert 'pool' in data['data']

    def test_predict_missing_fields(self, client):
        """Test prediction with missing required fields."""
        response = client.post('/api/predict', json={})
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data

    def test_predict_invalid_state(self, client):
        """Test prediction with invalid state."""
        data = {
            'property_age': 20,
            'square_footage': 2000,
            'bedrooms': 3,
            'bathrooms': 2,
            'state': 'XX',  # Invalid
            'location_type': 'suburban',
            'property_type': 'single_family',
            'annual_income': 100000
        }
        response = client.post('/api/predict', json=data)
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data

    def test_predict_invalid_location_type(self, client):
        """Test prediction with invalid location type."""
        data = {
            'property_age': 20,
            'square_footage': 2000,
            'bedrooms': 3,
            'bathrooms': 2,
            'state': 'CA',
            'location_type': 'invalid',
            'property_type': 'single_family',
            'annual_income': 100000
        }
        response = client.post('/api/predict', json=data)
        assert response.status_code == 400

    def test_predict_invalid_property_type(self, client):
        """Test prediction with invalid property type."""
        data = {
            'property_age': 20,
            'square_footage': 2000,
            'bedrooms': 3,
            'bathrooms': 2,
            'state': 'CA',
            'location_type': 'suburban',
            'property_type': 'invalid',
            'annual_income': 100000
        }
        response = client.post('/api/predict', json=data)
        assert response.status_code == 400

    def test_affordability_endpoint(self, client):
        """Test affordability analysis endpoint."""
        data = {
            'annual_income': 100000,
            'down_payment_pct': 0.20,
            'credit_score': 720,
            'monthly_debts': 0
        }
        response = client.post('/api/affordability', json=data)
        assert response.status_code == 200
        result = json.loads(response.data)
        assert result['success'] is True
        assert 'analysis' in result['data']
        assert 'recommendation' in result['data']

    def test_market_data_endpoint(self, client):
        """Test market data endpoint."""
        response = client.get('/api/market-data')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] is True
        assert 'national' in data['data']
        assert 'regional' in data['data']

    def test_inflation_calculator_endpoint(self, client):
        """Test inflation calculator endpoint."""
        data = {
            'current_price': 400000,
            'years': 5,
            'inflation_rate': 0.04
        }
        response = client.post('/api/inflation-calculator', json=data)
        assert response.status_code == 200
        result = json.loads(response.data)
        assert result['success'] is True
        assert result['data']['future_price'] > 400000
        assert len(result['data']['yearly_projection']) == 5

    def test_loan_calculator_endpoint(self, client):
        """Test loan calculator endpoint."""
        data = {
            'price': 400000,
            'down_payment_pct': 0.20,
            'interest_rate': 0.065,
            'loan_term_years': 30
        }
        response = client.post('/api/loan-calculator', json=data)
        assert response.status_code == 200
        result = json.loads(response.data)
        assert result['success'] is True
        assert result['data']['monthly_payment'] > 0
        assert result['data']['total_interest'] > 0
        assert len(result['data']['amortization_schedule']) == 12
        assert len(result['data']['yearly_summary']) == 30


class TestPredictionLogic:
    """Tests for prediction logic with mocked model."""

    @patch('api.app.model')
    @patch('api.app.preprocessor')
    @patch('api.app.feature_importance', {'square_footage': 0.3, 'location_type_suburban': 0.2})
    def test_make_prediction(self, mock_preprocessor, mock_model):
        """Test full prediction pipeline with mocked model."""
        # Setup mocks
        mock_preprocessor.feature_names_in_ = np.array([
            'property_age', 'square_footage', 'bedrooms', 'bathrooms',
            'state', 'location_type', 'property_type', 'down_payment_pct',
            'months_to_save_down_payment', 'annual_income', 'interest_rate',
            'loan_term_years', 'credit_score', 'inflation_rate_assumed',
            'recommended_down_payment_pct', 'amenity_count',
            'affordability_category', 'user_needs',
            'has_pool', 'has_garage', 'has_garden', 'has_fireplace',
            'has_central_ac', 'has_heating', 'has_basement',
            'has_security_system', 'has_smart_home', 'has_solar_panels',
            'has_gym_access', 'has_community_pool', 'has_parking', 'has_storage'
        ])
        mock_preprocessor.transform.return_value = np.array([[1.0] * 34])
        mock_model.predict.return_value = np.array([450000.0])
        mock_model.named_steps = {'model': Mock()}
        mock_model.named_steps['model'].estimators_ = [
            Mock(predict=Mock(return_value=np.array([450000.0])))
            for _ in range(10)
        ]

        data = {
            'property_age': 20,
            'square_footage': 2000,
            'bedrooms': 3,
            'bathrooms': 2.0,
            'state': 'CA',
            'location_type': 'suburban',
            'property_type': 'single_family',
            'down_payment_pct': 0.20,
            'annual_income': 120000,
            'credit_score': 740,
            'loan_term_years': 30,
            'amenities': ['pool', 'garage'],
            'user_needs': 'general'
        }

        result = make_prediction(data)

        assert 'predicted_price' in result
        assert result['predicted_price'] == 450000.0
        assert 'confidence_interval' in result
        assert 'loan_details' in result
        assert 'affordability' in result
        assert 'inflation_projection' in result
        assert 'feature_contributions' in result


if __name__ == '__main__':
    pytest.main([__file__, '-v'])