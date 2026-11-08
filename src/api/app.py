#!/usr/bin/env python3
"""
REST API for House Price Predictor
Provides HTTP endpoints for predictions, market data, and model info.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
MODEL_DIR = Path(__file__).parent.parent.parent / "models"
MODEL_PATH = MODEL_DIR / "house_price_model.pkl"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.pkl"
FEATURE_IMPORTANCE_PATH = MODEL_DIR / "feature_importance.json"
METRICS_PATH = MODEL_DIR / "model_metrics.json"

# Global model cache
model = None
preprocessor = None
feature_importance = None
model_metrics = None

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Configuration
STATES = [
    'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
    'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
    'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
    'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
    'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'DC'
]

LOCATION_TYPES = ['urban', 'suburban', 'rural']
PROPERTY_TYPES = ['single_family', 'condo', 'townhouse', 'multi_family', 'mobile_home']

AMENITIES_LIST = [
    'pool', 'garage', 'garden', 'fireplace', 'central_ac', 'heating',
    'basement', 'attic', 'balcony', 'security_system', 'smart_home',
    'solar_panels', 'gym_access', 'community_pool', 'parking', 'storage'
]

def load_model():
    """Load the trained model and preprocessor."""
    global model, preprocessor, feature_importance, model_metrics

    try:
        if MODEL_PATH.exists():
            model = joblib.load(MODEL_PATH)
            logger.info(f"Model loaded from {MODEL_PATH}")
        else:
            logger.warning(f"Model not found at {MODEL_PATH}")
            return False

        if PREPROCESSOR_PATH.exists():
            preprocessor = joblib.load(PREPROCESSOR_PATH)
            logger.info(f"Preprocessor loaded from {PREPROCESSOR_PATH}")
        else:
            logger.warning(f"Preprocessor not found at {PREPROCESSOR_PATH}")
            return False

        if FEATURE_IMPORTANCE_PATH.exists():
            with open(FEATURE_IMPORTANCE_PATH) as f:
                feature_importance = json.load(f)

        if METRICS_PATH.exists():
            with open(METRICS_PATH) as f:
                model_metrics = json.load(f)

        return True
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False

def prepare_input_features(data: dict) -> pd.DataFrame:
    """Convert request data to model input format."""
    # Base features
    features = {
        'property_age': data.get('property_age', 20),
        'square_footage': data.get('square_footage', 2000),
        'bedrooms': data.get('bedrooms', 3),
        'bathrooms': data.get('bathrooms', 2.0),
        'state': data.get('state', 'CA'),
        'location_type': data.get('location_type', 'suburban'),
        'property_type': data.get('property_type', 'single_family'),
        'down_payment_pct': data.get('down_payment_pct', 0.20),
        'months_to_save_down_payment': data.get('months_to_save_down_payment', 24),
        'annual_income': data.get('annual_income', 100000),
        'interest_rate': data.get('interest_rate', 0.065),
        'loan_term_years': data.get('loan_term_years', 30),
        'credit_score': data.get('credit_score', 720),
        'inflation_rate_assumed': 0.04,
        'recommended_down_payment_pct': 0.20,
        'amenity_count': len(data.get('amenities', [])),
    }

    # Calculate interest rate based on credit score if not provided
    if 'interest_rate' not in data:
        credit_score = data.get('credit_score', 720)
        base_rate = 0.065
        if credit_score >= 760:
            features['interest_rate'] = base_rate - 0.01
        elif credit_score >= 700:
            features['interest_rate'] = base_rate - 0.005
        elif credit_score >= 660:
            features['interest_rate'] = base_rate
        elif credit_score >= 620:
            features['interest_rate'] = base_rate + 0.005
        else:
            features['interest_rate'] = base_rate + 0.015

    # Add amenity binary features
    amenities = data.get('amenities', [])
    for amenity in AMENITIES_LIST:
        features[f'has_{amenity}'] = 1 if amenity in amenities else 0

    # Affordability category placeholder
    features['affordability_category'] = 'good'
    features['user_needs'] = data.get('user_needs', 'general')

    return pd.DataFrame([features])

def make_prediction(data: dict) -> Dict[str, Any]:
    """Make a price prediction with full analysis."""
    if model is None or preprocessor is None:
        raise ValueError("Model not loaded")

    # Prepare features
    X = prepare_input_features(data)

    # Ensure column order matches training
    if hasattr(preprocessor, 'feature_names_in_'):
        expected_cols = preprocessor.feature_names_in_
        for col in expected_cols:
            if col not in X.columns:
                X[col] = 0
        X = X[expected_cols]

    # Transform
    X_transformed = preprocessor.transform(X)

    # Predict
    predicted_price = float(model.predict(X_transformed)[0])

    # Calculate confidence interval
    if hasattr(model.named_steps['model'], 'estimators_'):
        tree_preds = np.array([tree.predict(X_transformed) for tree in model.named_steps['model'].estimators_])
        pred_std = float(tree_preds.std())
        ci_lower = predicted_price - 1.96 * pred_std
        ci_upper = predicted_price + 1.96 * pred_std
    else:
        ci_lower = predicted_price * 0.9
        ci_upper = predicted_price * 1.1

    # Down payment
    down_payment_pct = data.get('down_payment_pct', 0.20)
    down_payment = predicted_price * down_payment_pct

    # Loan details
    loan_amount = predicted_price - down_payment
    interest_rate = features['interest_rate']
    loan_term_years = data.get('loan_term_years', 30)
    monthly_rate = interest_rate / 12
    n_payments = loan_term_years * 12

    if monthly_rate > 0:
        monthly_payment = loan_amount * (monthly_rate * (1 + monthly_rate)**n_payments) / ((1 + monthly_rate)**n_payments - 1)
    else:
        monthly_payment = loan_amount / n_payments

    total_interest = monthly_payment * n_payments - loan_amount

    # Affordability
    annual_income = data.get('annual_income', 100000)
    gross_monthly_income = annual_income / 12
    dti_ratio = monthly_payment / gross_monthly_income if gross_monthly_income > 0 else 0

    if dti_ratio <= 0.28:
        affordability_category = 'excellent'
    elif dti_ratio <= 0.36:
        affordability_category = 'good'
    elif dti_ratio <= 0.43:
        affordability_category = 'fair'
    elif dti_ratio <= 0.50:
        affordability_category = 'stretched'
    else:
        affordability_category = 'unaffordable'

    # Max affordable price (28% rule)
    max_monthly = gross_monthly_income * 0.28
    if monthly_rate > 0:
        max_loan = max_monthly * ((1 + monthly_rate)**n_payments - 1) / (monthly_rate * (1 + monthly_rate)**n_payments)
    else:
        max_loan = max_monthly * n_payments
    max_affordable = max_loan / (1 - down_payment_pct) if down_payment_pct < 1 else max_loan

    # Inflation projection
    price_5yr = predicted_price * (1.04 ** 5)
    price_10yr = predicted_price * (1.04 ** 10)

    # Feature contributions
    contributions = {}
    if feature_importance:
        top_features = list(feature_importance.items())[:15]
        contributions = dict(top_features)

    return {
        'predicted_price': round(predicted_price, 2),
        'confidence_interval': {
            'lower': round(ci_lower, 2),
            'upper': round(ci_upper, 2),
            'confidence_level': 0.95
        },
        'down_payment': round(down_payment, 2),
        'down_payment_pct': down_payment_pct,
        'monthly_payment': round(monthly_payment, 2),
        'loan_details': {
            'loan_amount': round(loan_amount, 2),
            'interest_rate': round(interest_rate, 4),
            'term_years': loan_term_years,
            'total_interest': round(total_interest, 2),
            'total_paid': round(loan_amount + total_interest, 2)
        },
        'affordability': {
            'dti_ratio': round(dti_ratio, 4),
            'category': affordability_category,
            'max_affordable_price': round(max_affordable, 2),
            'monthly_income': round(gross_monthly_income, 2),
            'recommended_max_payment': round(max_monthly, 2)
        },
        'inflation_projection': {
            'current_price': round(predicted_price, 2),
            'price_5yr': round(price_5yr, 2),
            'price_10yr': round(price_10yr, 2),
            'assumed_annual_inflation': 0.04
        },
        'feature_contributions': contributions,
        'timestamp': datetime.now().isoformat()
    }

# API Routes
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Make a house price prediction."""
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No data provided'}), 400

        # Validate required fields
        required_fields = ['property_age', 'square_footage', 'bedrooms', 'bathrooms',
                          'state', 'location_type', 'property_type', 'annual_income']

        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400

        # Validate enums
        if data['state'] not in STATES:
            return jsonify({'error': f'Invalid state. Must be one of: {STATES}'}), 400

        if data['location_type'] not in LOCATION_TYPES:
            return jsonify({'error': f'Invalid location_type. Must be one of: {LOCATION_TYPES}'}), 400

        if data['property_type'] not in PROPERTY_TYPES:
            return jsonify({'error': f'Invalid property_type. Must be one of: {PROPERTY_TYPES}'}), 400

        # Make prediction
        result = make_prediction(data)

        return jsonify({
            'success': True,
            'data': result
        })

    except ValueError as e:
        return jsonify({'error': str(e)}), 500
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/affordability', methods=['POST'])
def affordability():
    """Analyze affordability for given parameters."""
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No data provided'}), 400

        annual_income = data.get('annual_income', 100000)
        down_payment_pct = data.get('down_payment_pct', 0.20)
        credit_score = data.get('credit_score', 720)
        monthly_debts = data.get('monthly_debts', 0)
        loan_term = data.get('loan_term_years', 30)

        # Interest rate based on credit score
        base_rate = 0.065
        if credit_score >= 760:
            rate = base_rate - 0.01
        elif credit_score >= 700:
            rate = base_rate - 0.005
        elif credit_score >= 660:
            rate = base_rate
        elif credit_score >= 620:
            rate = base_rate + 0.005
        else:
            rate = base_rate + 0.015

        # Price points to analyze
        price_points = [150000, 200000, 250000, 300000, 350000, 400000, 500000, 600000, 750000, 1000000]

        analysis = []
        for price in price_points:
            down = price * down_payment_pct
            loan = price - down
            monthly_rate = rate / 12
            n_payments = loan_term * 12

            if monthly_rate > 0:
                payment = loan * (monthly_rate * (1 + monthly_rate)**n_payments) / ((1 + monthly_rate)**n_payments - 1)
            else:
                payment = loan / n_payments

            total_debt = payment + monthly_debts
            dti = total_debt / (annual_income / 12) if annual_income > 0 else 0

            analysis.append({
                'price': price,
                'down_payment': round(down, 2),
                'loan_amount': round(loan, 2),
                'monthly_payment': round(payment, 2),
                'dti_ratio': round(dti, 4),
                'affordable': dti <= 0.36,
                'affordability_category': (
                    'excellent' if dti <= 0.28 else
                    'good' if dti <= 0.36 else
                    'fair' if dti <= 0.43 else
                    'stretched' if dti <= 0.50 else
                    'unaffordable'
                )
            })

        # Find recommended max price
        affordable_options = [a for a in analysis if a['affordable']]
        recommendation = max(affordable_options, key=lambda x: x['price']) if affordable_options else analysis[0]

        return jsonify({
            'success': True,
            'data': {
                'annual_income': annual_income,
                'down_payment_pct': down_payment_pct,
                'interest_rate': round(rate, 4),
                'analysis': analysis,
                'recommendation': recommendation
            }
        })

    except Exception as e:
        logger.error(f"Affordability error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/market-data', methods=['GET'])
def market_data():
    """Get current market data."""
    # Simulated market data
    data = {
        'national': {
            'median_price': 420000,
            'avg_price': 485000,
            'price_change_1m': 0.012,
            'price_change_3m': 0.035,
            'price_change_1y': 0.085,
            'inventory_months': 3.2,
            'days_on_market': 28,
            'mortgage_rate_30yr': 6.5,
            'mortgage_rate_15yr': 5.75,
            'mortgage_rate_30yr_change_1w': -0.05,
            'sales_volume': 4200000
        },
        'regional': {
            'West': {'median_price': 650000, 'change_1y': 0.12, 'inventory_months': 2.8},
            'Northeast': {'median_price': 520000, 'change_1y': 0.08, 'inventory_months': 3.5},
            'Midwest': {'median_price': 280000, 'change_1y': 0.06, 'inventory_months': 3.8},
            'South': {'median_price': 350000, 'change_1y': 0.10, 'inventory_months': 3.0},
            'Mountain': {'median_price': 450000, 'change_1y': 0.11, 'inventory_months': 2.5}
        },
        'state_hot_markets': [
            {'state': 'CA', 'median_price': 780000, 'change_1y': 0.15},
            {'state': 'NY', 'median_price': 680000, 'change_1y': 0.10},
            {'state': 'WA', 'median_price': 580000, 'change_1y': 0.12},
            {'state': 'CO', 'median_price': 520000, 'change_1y': 0.11},
            {'state': 'MA', 'median_price': 560000, 'change_1y': 0.09}
        ],
        'timestamp': datetime.now().isoformat()
    }

    return jsonify({
        'success': True,
        'data': data
    })

@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Get model information and metrics."""
    if model_metrics is None:
        return jsonify({'error': 'Model metrics not available'}), 500

    return jsonify({
        'success': True,
        'data': {
            'best_model': model_metrics.get('best_model', 'unknown'),
            'test_metrics': model_metrics.get('test_metrics', {}),
            'all_models': model_metrics.get('all_results', {}),
            'feature_count': len(feature_importance) if feature_importance else 0,
            'top_features': list(feature_importance.items())[:20] if feature_importance else []
        }
    })

@app.route('/api/states', methods=['GET'])
def get_states():
    """Get list of valid states."""
    return jsonify({
        'success': True,
        'data': STATES
    })

@app.route('/api/location-types', methods=['GET'])
def get_location_types():
    """Get list of valid location types."""
    return jsonify({
        'success': True,
        'data': LOCATION_TYPES
    })

@app.route('/api/property-types', methods=['GET'])
def get_property_types():
    """Get list of valid property types."""
    return jsonify({
        'success': True,
        'data': PROPERTY_TYPES
    })

@app.route('/api/amenities', methods=['GET'])
def get_amenities():
    """Get list of available amenities."""
    return jsonify({
        'success': True,
        'data': AMENITIES_LIST
    })

@app.route('/api/inflation-calculator', methods=['POST'])
def inflation_calculator():
    """Calculate future value with inflation."""
    try:
        data = request.get_json()
        current_price = data.get('current_price', 400000)
        years = data.get('years', 5)
        inflation_rate = data.get('inflation_rate', 0.04)

        future_price = current_price * ((1 + inflation_rate) ** years)
        total_appreciation = future_price - current_price
        annual_appreciation = total_appreciation / years

        return jsonify({
            'success': True,
            'data': {
                'current_price': current_price,
                'future_price': round(future_price, 2),
                'years': years,
                'inflation_rate': inflation_rate,
                'total_appreciation': round(total_appreciation, 2),
                'annual_appreciation': round(annual_appreciation, 2),
                'yearly_projection': [
                    {
                        'year': i + 1,
                        'price': round(current_price * ((1 + inflation_rate) ** (i + 1)), 2)
                    }
                    for i in range(years)
                ]
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/loan-calculator', methods=['POST'])
def loan_calculator():
    """Calculate loan payment details."""
    try:
        data = request.get_json()
        price = data.get('price', 400000)
        down_payment_pct = data.get('down_payment_pct', 0.20)
        interest_rate = data.get('interest_rate', 0.065)
        loan_term_years = data.get('loan_term_years', 30)

        down_payment = price * down_payment_pct
        loan_amount = price - down_payment
        monthly_rate = interest_rate / 12
        n_payments = loan_term_years * 12

        if monthly_rate > 0:
            monthly_payment = loan_amount * (monthly_rate * (1 + monthly_rate)**n_payments) / ((1 + monthly_rate)**n_payments - 1)
        else:
            monthly_payment = loan_amount / n_payments

        total_paid = monthly_payment * n_payments
        total_interest = total_paid - loan_amount

        # Amortization schedule (first 12 months + yearly summaries)
        schedule = []
        balance = loan_amount
        for month in range(1, min(13, n_payments + 1)):
            interest_paid = balance * monthly_rate
            principal_paid = monthly_payment - interest_paid
            balance -= principal_paid
            schedule.append({
                'month': month,
                'payment': round(monthly_payment, 2),
                'principal': round(principal_paid, 2),
                'interest': round(interest_paid, 2),
                'balance': round(max(0, balance), 2)
            })

        # Yearly summaries
        yearly = []
        balance = loan_amount
        for year in range(1, loan_term_years + 1):
            yearly_principal = 0
            yearly_interest = 0
            for month in range(12):
                if month + (year - 1) * 12 >= n_payments:
                    break
                interest_paid = balance * monthly_rate
                principal_paid = monthly_payment - interest_paid
                balance -= principal_paid
                yearly_principal += principal_paid
                yearly_interest += interest_paid
            yearly.append({
                'year': year,
                'principal_paid': round(yearly_principal, 2),
                'interest_paid': round(yearly_interest, 2),
                'remaining_balance': round(max(0, balance), 2)
            })

        return jsonify({
            'success': True,
            'data': {
                'price': price,
                'down_payment': round(down_payment, 2),
                'down_payment_pct': down_payment_pct,
                'loan_amount': round(loan_amount, 2),
                'interest_rate': interest_rate,
                'loan_term_years': loan_term_years,
                'monthly_payment': round(monthly_payment, 2),
                'total_paid': round(total_paid, 2),
                'total_interest': round(total_interest, 2),
                'amortization_schedule': schedule,
                'yearly_summary': yearly
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Error handlers
@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def server_error(e):
    logger.error(f"Server error: {e}")
    return jsonify({'error': 'Internal server error'}), 500

def create_app():
    """Create and configure the Flask app."""
    load_model()
    return app

if __name__ == '__main__':
    # Load model on startup
    load_model()

    # Run server
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)