#!/usr/bin/env python3
"""
WebSocket Server for Real-time House Price Predictions
Provides real-time price predictions, market data streaming, and affordability analysis.
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import joblib

import websockets
from websockets.server import WebSocketServerProtocol

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
MODEL_DIR = Path(__file__).parent.parent.parent / "models"
MODEL_PATH = MODEL_DIR / "house_price_model.pkl"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.pkl"
FEATURE_IMPORTANCE_PATH = MODEL_DIR / "feature_importance.json"

# Global model cache
model = None
preprocessor = None
feature_importance = None

# Connected clients
connected_clients = set()

# Market data simulation
MARKET_TRENDS = {
    'national_avg_price': 420000,
    'price_change_1m': 0.012,
    'price_change_3m': 0.035,
    'price_change_1y': 0.085,
    'inventory_months': 3.2,
    'days_on_market': 28,
    'mortgage_rate_30yr': 6.5,
    'mortgage_rate_15yr': 5.75
}

@dataclass
class PredictionRequest:
    property_age: int
    square_footage: int
    bedrooms: int
    bathrooms: float
    state: str
    location_type: str
    property_type: str
    down_payment_pct: float
    annual_income: float
    credit_score: int
    loan_term_years: int = 30
    amenities: list = None
    user_needs: str = "general"

@dataclass
class PredictionResponse:
    predicted_price: float
    confidence_interval: dict
    down_payment: float
    monthly_payment: float
    loan_details: dict
    affordability: dict
    inflation_projection: dict
    feature_contributions: dict
    timestamp: str
    request_id: str

def load_model():
    """Load the trained model and preprocessor."""
    global model, preprocessor, feature_importance

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
        logger.info("Feature importance loaded")

    return True

def prepare_input_features(request: PredictionRequest) -> pd.DataFrame:
    """Convert request to model input format."""
    # Base features
    data = {
        'property_age': request.property_age,
        'square_footage': request.square_footage,
        'bedrooms': request.bedrooms,
        'bathrooms': request.bathrooms,
        'state': request.state,
        'location_type': request.location_type,
        'property_type': request.property_type,
        'down_payment_pct': request.down_payment_pct,
        'months_to_save_down_payment': request.down_payment_pct * 100,  # rough estimate
        'annual_income': request.annual_income,
        'interest_rate': 0.065,  # Will be calculated based on credit score
        'loan_term_years': request.loan_term_years,
        'credit_score': request.credit_score,
        'inflation_rate_assumed': 0.04,
        'recommended_down_payment_pct': 0.20,
        'amenity_count': len(request.amenities) if request.amenities else 0,
    }

    # Calculate interest rate based on credit score
    base_rate = 0.065
    if request.credit_score >= 760:
        data['interest_rate'] = base_rate - 0.01
    elif request.credit_score >= 700:
        data['interest_rate'] = base_rate - 0.005
    elif request.credit_score >= 660:
        data['interest_rate'] = base_rate
    elif request.credit_score >= 620:
        data['interest_rate'] = base_rate + 0.005
    else:
        data['interest_rate'] = base_rate + 0.015

    # Add amenity binary features
    all_amenities = ['pool', 'garage', 'garden', 'fireplace', 'central_ac',
                     'heating', 'basement', 'security_system', 'smart_home',
                     'solar_panels', 'parking']

    if request.amenities:
        for amenity in all_amenities:
            data[f'has_{amenity}'] = 1 if amenity in request.amenities else 0
    else:
        for amenity in all_amenities:
            data[f'has_{amenity}'] = 0

    # Affordability category (will be predicted)
    data['affordability_category'] = 'good'  # placeholder
    data['user_needs'] = request.user_needs

    return pd.DataFrame([data])

def predict_price(request: PredictionRequest) -> PredictionResponse:
    """Make a price prediction with full analysis."""
    if model is None or preprocessor is None:
        raise ValueError("Model not loaded")

    # Prepare features
    X = prepare_input_features(request)

    # Ensure column order matches training
    expected_cols = preprocessor.feature_names_in_ if hasattr(preprocessor, 'feature_names_in_') else None
    if expected_cols is not None:
        # Add missing columns with default values
        for col in expected_cols:
            if col not in X.columns:
                X[col] = 0
        X = X[expected_cols]

    # Transform
    X_transformed = preprocessor.transform(X)

    # Predict
    predicted_price = model.predict(X_transformed)[0]

    # Calculate confidence interval (using prediction intervals approximation)
    # For ensemble methods, we can use std of predictions from individual trees
    if hasattr(model.named_steps['model'], 'estimators_'):
        tree_preds = np.array([tree.predict(X_transformed) for tree in model.named_steps['model'].estimators_])
        pred_std = tree_preds.std()
        ci_lower = predicted_price - 1.96 * pred_std
        ci_upper = predicted_price + 1.96 * pred_std
    else:
        # Approximate with 10% range
        ci_lower = predicted_price * 0.9
        ci_upper = predicted_price * 1.1

    # Down payment
    down_payment = predicted_price * request.down_payment_pct

    # Loan details
    loan_amount = predicted_price - down_payment
    monthly_rate = data['interest_rate'] / 12
    n_payments = request.loan_term_years * 12
    if monthly_rate > 0:
        monthly_payment = loan_amount * (monthly_rate * (1 + monthly_rate)**n_payments) / ((1 + monthly_rate)**n_payments - 1)
    else:
        monthly_payment = loan_amount / n_payments
    total_interest = monthly_payment * n_payments - loan_amount

    # Affordability
    gross_monthly_income = request.annual_income / 12
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
    max_affordable = max_loan / (1 - request.down_payment_pct) if request.down_payment_pct < 1 else max_loan

    # Inflation projection (5 years)
    price_5yr = predicted_price * (1.04 ** 5)
    price_10yr = predicted_price * (1.04 ** 10)

    # Feature contributions (simplified)
    contributions = {}
    if feature_importance:
        # Get top features
        top_features = list(feature_importance.items())[:10]
        contributions = dict(top_features)

    return PredictionResponse(
        predicted_price=round(predicted_price, 2),
        confidence_interval={
            'lower': round(ci_lower, 2),
            'upper': round(ci_upper, 2),
            'confidence_level': 0.95
        },
        down_payment=round(down_payment, 2),
        monthly_payment=round(monthly_payment, 2),
        loan_details={
            'loan_amount': round(loan_amount, 2),
            'interest_rate': round(data['interest_rate'], 4),
            'term_years': request.loan_term_years,
            'total_interest': round(total_interest, 2),
            'total_paid': round(loan_amount + total_interest, 2)
        },
        affordability={
            'dti_ratio': round(dti_ratio, 4),
            'category': affordability_category,
            'max_affordable_price': round(max_affordable, 2),
            'monthly_income': round(gross_monthly_income, 2),
            'recommended_max_payment': round(max_monthly, 2)
        },
        inflation_projection={
            'current_price': round(predicted_price, 2),
            'price_5yr': round(price_5yr, 2),
            'price_10yr': round(price_10yr, 2),
            'assumed_annual_inflation': 0.04
        },
        feature_contributions=contributions,
        timestamp=datetime.now().isoformat(),
        request_id=f"req_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    )

async def handle_prediction(websocket: WebSocketServerProtocol, message: dict):
    """Handle prediction request."""
    try:
        request = PredictionRequest(**message.get('data', {}))
        response = predict_price(request)

        await websocket.send(json.dumps({
            'type': 'prediction',
            'data': asdict(response)
        }))
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        await websocket.send(json.dumps({
            'type': 'error',
            'message': str(e)
        }))

async def handle_market_data(websocket: WebSocketServerProtocol):
    """Send current market data."""
    # Simulate real-time market data updates
    market_data = MARKET_TRENDS.copy()
    market_data['timestamp'] = datetime.now().isoformat()
    market_data['regional_data'] = generate_regional_data()

    await websocket.send(json.dumps({
        'type': 'market_data',
        'data': market_data
    }))

def generate_regional_data() -> dict:
    """Generate simulated regional market data."""
    regions = {
        'West': {'avg_price': 650000, 'change_1y': 0.12, 'inventory': 2.8},
        'Northeast': {'avg_price': 520000, 'change_1y': 0.08, 'inventory': 3.5},
        'Midwest': {'avg_price': 280000, 'change_1y': 0.06, 'inventory': 3.8},
        'South': {'avg_price': 350000, 'change_1y': 0.10, 'inventory': 3.0},
        'Mountain': {'avg_price': 450000, 'change_1y': 0.11, 'inventory': 2.5}
    }
    return regions

async def handle_affordability_analysis(websocket: WebSocketServerProtocol, message: dict):
    """Handle affordability analysis request."""
    try:
        data = message.get('data', {})
        annual_income = data.get('annual_income', 100000)
        down_payment_pct = data.get('down_payment_pct', 0.20)
        credit_score = data.get('credit_score', 720)
        monthly_debts = data.get('monthly_debts', 0)
        location = data.get('location', 'national')

        # Calculate affordability at different price points
        price_points = [200000, 300000, 400000, 500000, 600000, 750000, 1000000]

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

        analysis = []
        for price in price_points:
            down = price * down_payment_pct
            loan = price - down
            monthly_rate = rate / 12
            n_payments = 360
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
                'affordable': dti <= 0.36
            })

        await websocket.send(json.dumps({
            'type': 'affordability_analysis',
            'data': {
                'annual_income': annual_income,
                'analysis': analysis,
                'recommendation': next((a for a in reversed(analysis) if a['affordable']), analysis[0])
            }
        }))
    except Exception as e:
        logger.error(f"Affordability analysis error: {e}")
        await websocket.send(json.dumps({
            'type': 'error',
            'message': str(e)
        }))

async def broadcast_market_updates():
    """Periodically broadcast market updates to all connected clients."""
    while True:
        await asyncio.sleep(30)  # Update every 30 seconds

        if connected_clients:
            market_data = MARKET_TRENDS.copy()
            market_data['timestamp'] = datetime.now().isoformat()
            market_data['regional_data'] = generate_regional_data()

            message = json.dumps({
                'type': 'market_update',
                'data': market_data
            })

            # Send to all connected clients
            disconnected = set()
            for client in connected_clients:
                try:
                    await client.send(message)
                except websockets.exceptions.ConnectionClosed:
                    disconnected.add(client)

            # Remove disconnected clients
            connected_clients -= disconnected

async def handler(websocket: WebSocketServerProtocol):
    """Main WebSocket connection handler."""
    client_id = id(websocket)
    connected_clients.add(websocket)
    logger.info(f"Client connected: {client_id} (total: {len(connected_clients)})")

    try:
        # Send welcome message
        await websocket.send(json.dumps({
            'type': 'welcome',
            'message': 'Connected to House Price Predictor WebSocket Server',
            'client_id': client_id,
            'timestamp': datetime.now().isoformat()
        }))

        # Send initial market data
        await handle_market_data(websocket)

        async for message in websocket:
            try:
                data = json.loads(message)
                msg_type = data.get('type')

                if msg_type == 'predict':
                    await handle_prediction(websocket, data)
                elif msg_type == 'market_data':
                    await handle_market_data(websocket)
                elif msg_type == 'affordability':
                    await handle_affordability_analysis(websocket, data)
                elif msg_type == 'ping':
                    await websocket.send(json.dumps({'type': 'pong', 'timestamp': datetime.now().isoformat()}))
                else:
                    await websocket.send(json.dumps({
                        'type': 'error',
                        'message': f'Unknown message type: {msg_type}'
                    }))
            except json.JSONDecodeError:
                await websocket.send(json.dumps({
                    'type': 'error',
                    'message': 'Invalid JSON'
                }))
            except Exception as e:
                logger.error(f"Message handling error: {e}")
                await websocket.send(json.dumps({
                    'type': 'error',
                    'message': str(e)
                }))

    except websockets.exceptions.ConnectionClosed:
        logger.info(f"Client disconnected: {client_id}")
    finally:
        connected_clients.discard(websocket)
        logger.info(f"Client removed: {client_id} (total: {len(connected_clients)})")

async def main():
    """Start the WebSocket server."""
    # Load model
    if not load_model():
        logger.error("Failed to load model. Server starting without ML capabilities.")

    # Start market data broadcaster
    broadcast_task = asyncio.create_task(broadcast_market_updates())

    # Start server
    host = '0.0.0.0'
    port = 8765

    logger.info(f"Starting WebSocket server on ws://{host}:{port}")

    async with websockets.serve(handler, host, port):
        await asyncio.Future()  # Run forever

if __name__ == '__main__':
    asyncio.run(main())