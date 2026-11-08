#!/usr/bin/env python3
"""
Synthetic House Price Dataset Generator
Generates realistic housing data with all required features for the House Price Predictor.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json

# Set seed for reproducibility
np.random.seed(42)

# Configuration
N_SAMPLES = 50000
OUTPUT_PATH = Path(__file__).parent / "housing_data.csv"
METADATA_PATH = Path(__file__).parent / "dataset_metadata.json"

# US States with regional price multipliers
STATE_MULTIPLIERS = {
    'CA': 2.5, 'NY': 2.3, 'MA': 2.0, 'WA': 1.8, 'CO': 1.6,
    'OR': 1.5, 'VA': 1.4, 'MD': 1.4, 'NJ': 1.5, 'CT': 1.6,
    'IL': 1.2, 'MN': 1.2, 'TX': 1.1, 'FL': 1.2, 'GA': 1.1,
    'NC': 1.05, 'AZ': 1.1, 'TN': 0.95, 'PA': 1.1, 'OH': 0.9,
    'MI': 0.85, 'IN': 0.85, 'WI': 0.9, 'MO': 0.85, 'AL': 0.8,
    'LA': 0.85, 'KY': 0.8, 'SC': 0.9, 'OK': 0.8, 'AR': 0.75,
    'MS': 0.7, 'WV': 0.7, 'NM': 0.85, 'NV': 1.3, 'UT': 1.2,
    'ID': 1.0, 'WY': 0.85, 'MT': 0.9, 'ND': 0.9, 'SD': 0.85,
    'NE': 0.9, 'KS': 0.85, 'IA': 0.85, 'ME': 0.95, 'NH': 1.1,
    'VT': 1.0, 'RI': 1.3, 'DE': 1.2, 'DC': 2.2, 'HI': 2.4,
    'AK': 1.5
}

STATES = list(STATE_MULTIPLIERS.keys())

# Urban/Rural/Suburban classifications
LOCATION_TYPES = ['urban', 'suburban', 'rural']
LOCATION_MULTIPLIERS = {'urban': 1.3, 'suburban': 1.1, 'rural': 0.8}

# Property types
PROPERTY_TYPES = ['single_family', 'condo', 'townhouse', 'multi_family', 'mobile_home']
PROPERTY_MULTIPLIERS = {
    'single_family': 1.0,
    'condo': 0.85,
    'townhouse': 0.9,
    'multi_family': 1.15,
    'mobile_home': 0.5
}

# Amenities
AMENITIES_LIST = [
    'pool', 'garage', 'garden', 'fireplace', 'central_ac', 'heating',
    'basement', 'attic', 'balcony', 'security_system', 'smart_home',
    'solar_panels', 'gym_access', 'community_pool', 'parking', 'storage'
]

def generate_base_price(state, location_type, property_type, sqft, age, bedrooms, bathrooms):
    """Generate base price based on features."""
    base_price_per_sqft = 150  # Base price per sqft

    # Apply multipliers
    price = base_price_per_sqft * sqft
    price *= STATE_MULTIPLIERS[state]
    price *= LOCATION_MULTIPLIERS[location_type]
    price *= PROPERTY_MULTIPLIERS[property_type]

    # Age depreciation (older homes worth less, but vintage adds value after 50 years)
    if age < 5:
        price *= 1.1  # New construction premium
    elif age < 20:
        price *= 1.0
    elif age < 50:
        price *= 0.95 ** ((age - 20) / 10)
    else:
        price *= 0.85 * (1 + (age - 50) * 0.002)  # Vintage value

    # Bedroom/bathroom adjustments
    price *= (1 + (bedrooms - 3) * 0.05)
    price *= (1 + (bathrooms - 2) * 0.03)

    return price

def generate_amenities_score():
    """Generate random amenities and calculate score."""
    n_amenities = np.random.poisson(5)
    selected = np.random.choice(AMENITIES_LIST, size=min(n_amenities, len(AMENITIES_LIST)), replace=False)

    # Amenity value adds
    amenity_values = {
        'pool': 25000, 'garage': 15000, 'garden': 8000, 'fireplace': 5000,
        'central_ac': 10000, 'heating': 8000, 'basement': 20000, 'attic': 5000,
        'balcony': 3000, 'security_system': 5000, 'smart_home': 8000,
        'solar_panels': 15000, 'gym_access': 3000, 'community_pool': 5000,
        'parking': 10000, 'storage': 3000
    }

    total_value = sum(amenity_values.get(a, 0) for a in selected)
    return selected.tolist(), total_value

def calculate_down_payment(price, down_payment_pct=None):
    """Calculate down payment amount and time to save."""
    if down_payment_pct is None:
        down_payment_pct = np.random.uniform(0.03, 0.25)  # 3-25%

    down_payment = price * down_payment_pct

    # Time to save (months) based on income
    annual_income = price * np.random.uniform(0.15, 0.4)  # Income is 15-40% of house price
    monthly_savings = annual_income / 12 * np.random.uniform(0.1, 0.3)  # Save 10-30% of income
    months_to_save = down_payment / monthly_savings if monthly_savings > 0 else 0

    return down_payment, down_payment_pct, months_to_save, annual_income

def calculate_loan_details(price, down_payment, credit_score=None):
    """Calculate loan details."""
    loan_amount = price - down_payment

    if credit_score is None:
        credit_score = np.random.randint(580, 850)

    # Interest rate based on credit score
    base_rate = 0.065  # 6.5% base
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

    # Add some random variation
    rate += np.random.uniform(-0.005, 0.005)
    rate = max(0.03, min(0.12, rate))

    # Loan term
    term_years = np.random.choice([15, 20, 30], p=[0.15, 0.1, 0.75])

    # Monthly payment
    monthly_rate = rate / 12
    n_payments = term_years * 12
    if monthly_rate > 0:
        monthly_payment = loan_amount * (monthly_rate * (1 + monthly_rate)**n_payments) / ((1 + monthly_rate)**n_payments - 1)
    else:
        monthly_payment = loan_amount / n_payments

    total_interest = monthly_payment * n_payments - loan_amount

    return {
        'loan_amount': loan_amount,
        'interest_rate': rate,
        'term_years': term_years,
        'monthly_payment': monthly_payment,
        'total_interest': total_interest,
        'credit_score': credit_score
    }

def calculate_inflation_adjustment(price, years_from_now=0):
    """Calculate inflation-adjusted price."""
    # Historical average inflation ~3%, housing inflation ~4%
    housing_inflation_rate = 0.04
    adjusted_price = price * ((1 + housing_inflation_rate) ** years_from_now)
    return adjusted_price

def calculate_affordability(annual_income, monthly_payment, monthly_debts=0):
    """Calculate affordability metrics."""
    # DTI ratio
    gross_monthly_income = annual_income / 12
    total_monthly_debt = monthly_payment + monthly_debts
    dti_ratio = total_monthly_debt / gross_monthly_income if gross_monthly_income > 0 else 0

    # Affordability category
    if dti_ratio <= 0.28:
        affordability = 'excellent'
    elif dti_ratio <= 0.36:
        affordability = 'good'
    elif dti_ratio <= 0.43:
        affordability = 'fair'
    elif dti_ratio <= 0.50:
        affordability = 'stretched'
    else:
        affordability = 'unaffordable'

    # Recommended max price based on 28% rule
    max_monthly_payment = gross_monthly_income * 0.28
    # Reverse calculate max loan amount
    rate = 0.065
    term = 30
    monthly_rate = rate / 12
    n_payments = term * 12
    if monthly_rate > 0:
        max_loan = max_monthly_payment * ((1 + monthly_rate)**n_payments - 1) / (monthly_rate * (1 + monthly_rate)**n_payments)
    else:
        max_loan = max_monthly_payment * n_payments
    max_price = max_loan / 0.8  # Assuming 20% down

    return {
        'dti_ratio': dti_ratio,
        'affordability_category': affordability,
        'max_affordable_price': max_price,
        'recommended_down_payment_pct': 0.20
    }

def generate_dataset(n_samples=N_SAMPLES):
    """Generate the complete dataset."""
    print(f"Generating {n_samples} samples...")

    data = []

    for i in range(n_samples):
        # Basic features
        state = np.random.choice(STATES, p=[1/len(STATES)]*len(STATES))
        location_type = np.random.choice(LOCATION_TYPES, p=[0.3, 0.5, 0.2])
        property_type = np.random.choice(PROPERTY_TYPES, p=[0.5, 0.15, 0.15, 0.1, 0.1])

        # Physical characteristics
        sqft = int(np.random.lognormal(np.log(1800), 0.4))
        sqft = np.clip(sqft, 500, 8000)

        age = int(np.random.exponential(25))
        age = np.clip(age, 0, 100)

        bedrooms = np.random.choice([1, 2, 3, 4, 5, 6], p=[0.05, 0.15, 0.35, 0.3, 0.1, 0.05])
        bathrooms = np.random.choice([1, 1.5, 2, 2.5, 3, 3.5, 4], p=[0.1, 0.15, 0.3, 0.2, 0.15, 0.05, 0.05])

        # Generate base price
        base_price = generate_base_price(state, location_type, property_type, sqft, age, bedrooms, bathrooms)

        # Amenities
        amenities, amenity_value = generate_amenities_score()
        price = base_price + amenity_value

        # Add market noise
        price *= np.random.lognormal(0, 0.1)
        price = round(price, -3)  # Round to nearest thousand

        # Down payment
        down_payment, down_payment_pct, months_to_save, annual_income = calculate_down_payment(price)

        # Loan details
        loan_details = calculate_loan_details(price, down_payment)

        # Inflation adjustment (for 5 years out)
        price_5yr = calculate_inflation_adjustment(price, 5)

        # Affordability
        affordability = calculate_affordability(
            annual_income,
            loan_details['monthly_payment'],
            np.random.uniform(0, 500)  # Other monthly debts
        )

        # User needs (what they're looking for)
        needs = []
        if bedrooms >= 4:
            needs.append('large_family')
        if location_type == 'urban':
            needs.append('commute_friendly')
        if 'garage' in amenities:
            needs.append('parking')
        if property_type == 'condo':
            needs.append('low_maintenance')
        if age < 5:
            needs.append('move_in_ready')
        if 'pool' in amenities:
            needs.append('entertainment')

        # Create record
        record = {
            # Target
            'price': price,

            # Features
            'property_age': age,
            'square_footage': sqft,
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'state': state,
            'location_type': location_type,
            'property_type': property_type,

            # Down payment
            'down_payment_amount': round(down_payment, 2),
            'down_payment_pct': round(down_payment_pct, 4),
            'months_to_save_down_payment': round(months_to_save, 1),
            'annual_income': round(annual_income, 2),

            # Loan
            'loan_amount': round(loan_details['loan_amount'], 2),
            'interest_rate': round(loan_details['interest_rate'], 4),
            'loan_term_years': loan_details['term_years'],
            'monthly_mortgage_payment': round(loan_details['monthly_payment'], 2),
            'total_interest_paid': round(loan_details['total_interest'], 2),
            'credit_score': loan_details['credit_score'],

            # Inflation
            'price_5yr_projection': round(price_5yr, 2),
            'inflation_rate_assumed': 0.04,

            # Affordability
            'dti_ratio': round(affordability['dti_ratio'], 4),
            'affordability_category': affordability['affordability_category'],
            'max_affordable_price': round(affordability['max_affordable_price'], 2),
            'recommended_down_payment_pct': affordability['recommended_down_payment_pct'],

            # Amenities
            'amenities': ','.join(amenities),
            'amenity_count': len(amenities),
            'amenity_value': round(amenity_value, 2),

            # User needs
            'user_needs': ','.join(needs) if needs else 'general',

            # Metadata
            'sample_id': i
        }

        data.append(record)

        if (i + 1) % 10000 == 0:
            print(f"  Generated {i + 1} samples...")

    df = pd.DataFrame(data)
    return df

def save_metadata(df):
    """Save dataset metadata."""
    metadata = {
        'n_samples': len(df),
        'n_features': len(df.columns),
        'features': df.columns.tolist(),
        'target': 'price',
        'categorical_features': ['state', 'location_type', 'property_type', 'affordability_category', 'user_needs'],
        'numerical_features': [c for c in df.columns if c not in ['state', 'location_type', 'property_type', 'affordability_category', 'user_needs', 'amenities', 'sample_id']],
        'states': STATES,
        'location_types': LOCATION_TYPES,
        'property_types': PROPERTY_TYPES,
        'amenities_list': AMENITIES_LIST,
        'train_test_split': '70/30',
        'generated_date': pd.Timestamp.now().isoformat()
    }

    with open(METADATA_PATH, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Metadata saved to {METADATA_PATH}")

def main():
    df = generate_dataset()

    # Save dataset
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Dataset saved to {OUTPUT_PATH}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nPrice statistics:")
    print(df['price'].describe())
    print(f"\nAffordability distribution:")
    print(df['affordability_category'].value_counts())
    print(f"\nLocation type distribution:")
    print(df['location_type'].value_counts())
    print(f"\nProperty type distribution:")
    print(df['property_type'].value_counts())

    save_metadata(df)

    # Create train/test split
    from sklearn.model_selection import train_test_split

    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['affordability_category'])

    train_path = Path(__file__).parent / "train_data.csv"
    test_path = Path(__file__).parent / "test_data.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"\nTrain set: {train_path} ({len(train_df)} samples)")
    print(f"Test set: {test_path} ({len(test_df)} samples)")

if __name__ == '__main__':
    main()