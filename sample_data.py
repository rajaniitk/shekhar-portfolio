#!/usr/bin/env python3
"""
Sample Data Generator for Advanced EDA & ML Platform
Creates various sample datasets to demonstrate platform capabilities
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import string

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_sales_dataset():
    """Generate a comprehensive sales dataset"""
    n_records = 10000
    
    # Generate dates
    start_date = datetime(2020, 1, 1)
    dates = [start_date + timedelta(days=x) for x in range(n_records)]
    
    # Generate product categories
    categories = ['Electronics', 'Clothing', 'Home & Garden', 'Sports', 'Books', 'Beauty', 'Toys']
    
    # Generate customer segments
    segments = ['Premium', 'Standard', 'Budget']
    
    # Generate regions
    regions = ['North', 'South', 'East', 'West', 'Central']
    
    # Generate data
    data = {
        'date': np.random.choice(dates, n_records),
        'category': np.random.choice(categories, n_records),
        'product_id': [f'P{str(i).zfill(4)}' for i in np.random.randint(1, 1000, n_records)],
        'customer_segment': np.random.choice(segments, n_records, p=[0.2, 0.5, 0.3]),
        'region': np.random.choice(regions, n_records),
        'quantity': np.random.poisson(3, n_records) + 1,
        'unit_price': np.random.gamma(2, 25),
        'discount_percent': np.random.exponential(5),
        'customer_age': np.random.normal(40, 15),
        'is_weekend': None,  # Will be calculated
        'is_holiday': None,  # Will be calculated
        'weather_temp': np.random.normal(20, 10),
        'customer_satisfaction': np.random.beta(8, 2) * 10,  # Skewed towards higher ratings
        'marketing_spend': np.random.exponential(100),
        'competitor_price': None,  # Will be calculated
        'total_amount': None,  # Will be calculated
        'profit_margin': np.random.normal(0.3, 0.1),
    }
    
    df = pd.DataFrame(data)
    
    # Calculate derived fields
    df['is_weekend'] = df['date'].dt.weekday >= 5
    df['is_holiday'] = np.random.choice([True, False], len(df), p=[0.05, 0.95])
    df['total_amount'] = df['quantity'] * df['unit_price'] * (1 - df['discount_percent'] / 100)
    df['competitor_price'] = df['unit_price'] * np.random.normal(1.1, 0.2, len(df))
    
    # Add some missing values intentionally
    missing_indices = np.random.choice(df.index, size=int(0.05 * len(df)), replace=False)
    df.loc[missing_indices, 'customer_satisfaction'] = np.nan
    
    missing_indices = np.random.choice(df.index, size=int(0.03 * len(df)), replace=False)
    df.loc[missing_indices, 'weather_temp'] = np.nan
    
    # Ensure realistic constraints
    df['customer_age'] = df['customer_age'].clip(18, 85)
    df['discount_percent'] = df['discount_percent'].clip(0, 50)
    df['unit_price'] = df['unit_price'].clip(1, 1000)
    df['customer_satisfaction'] = df['customer_satisfaction'].clip(1, 10)
    
    return df

def generate_employee_dataset():
    """Generate an employee dataset with various HR metrics"""
    n_records = 5000
    
    departments = ['Engineering', 'Sales', 'Marketing', 'HR', 'Finance', 'Operations', 'Support']
    positions = ['Junior', 'Mid', 'Senior', 'Lead', 'Manager', 'Director']
    education = ['High School', 'Bachelor', 'Master', 'PhD']
    locations = ['New York', 'San Francisco', 'Austin', 'Seattle', 'Boston', 'Remote']
    
    data = {
        'employee_id': [f'EMP{str(i).zfill(4)}' for i in range(1, n_records + 1)],
        'department': np.random.choice(departments, n_records),
        'position_level': np.random.choice(positions, n_records, p=[0.25, 0.25, 0.2, 0.15, 0.1, 0.05]),
        'education': np.random.choice(education, n_records, p=[0.1, 0.5, 0.3, 0.1]),
        'location': np.random.choice(locations, n_records),
        'years_experience': np.random.exponential(5),
        'years_at_company': None,  # Will be calculated
        'age': np.random.normal(35, 10),
        'gender': np.random.choice(['Male', 'Female', 'Other'], n_records, p=[0.45, 0.5, 0.05]),
        'salary': None,  # Will be calculated based on other factors
        'performance_rating': np.random.normal(3.5, 0.8),
        'training_hours': np.random.gamma(2, 10),
        'projects_completed': np.random.poisson(8),
        'overtime_hours': np.random.exponential(5),
        'job_satisfaction': np.random.beta(6, 2) * 10,
        'work_from_home_days': np.random.choice(range(6), n_records, p=[0.3, 0.2, 0.2, 0.15, 0.1, 0.05]),
        'has_promotion': np.random.choice([True, False], n_records, p=[0.15, 0.85]),
        'resignation_risk': None,  # Will be calculated
    }
    
    df = pd.DataFrame(data)
    
    # Calculate derived fields
    df['years_at_company'] = df['years_experience'] * np.random.uniform(0.3, 1.0, len(df))
    df['age'] = df['age'].clip(22, 65)
    df['years_experience'] = df['years_experience'].clip(0, 40)
    df['years_at_company'] = df['years_at_company'].clip(0, df['years_experience'])
    df['performance_rating'] = df['performance_rating'].clip(1, 5)
    
    # Calculate salary based on multiple factors
    base_salary = 50000
    dept_multiplier = {'Engineering': 1.3, 'Sales': 1.1, 'Marketing': 1.0, 'HR': 0.9, 
                      'Finance': 1.2, 'Operations': 1.0, 'Support': 0.9}
    position_multiplier = {'Junior': 0.8, 'Mid': 1.0, 'Senior': 1.3, 'Lead': 1.6, 
                          'Manager': 2.0, 'Director': 3.0}
    education_multiplier = {'High School': 0.9, 'Bachelor': 1.0, 'Master': 1.2, 'PhD': 1.4}
    
    df['salary'] = (base_salary * 
                   df['department'].map(dept_multiplier) *
                   df['position_level'].map(position_multiplier) *
                   df['education'].map(education_multiplier) *
                   (1 + df['years_experience'] * 0.02) *
                   np.random.normal(1, 0.1, len(df)))
    
    # Calculate resignation risk
    df['resignation_risk'] = (
        (df['job_satisfaction'] < 5) * 0.3 +
        (df['overtime_hours'] > 10) * 0.2 +
        (df['salary'] < df['salary'].quantile(0.25)) * 0.2 +
        (df['performance_rating'] < 3) * 0.15 +
        np.random.uniform(0, 0.15, len(df))
    ).clip(0, 1)
    
    # Add some missing values
    missing_indices = np.random.choice(df.index, size=int(0.02 * len(df)), replace=False)
    df.loc[missing_indices, 'training_hours'] = np.nan
    
    return df

def generate_customer_dataset():
    """Generate a customer dataset for churn analysis"""
    n_records = 8000
    
    subscription_types = ['Basic', 'Premium', 'Enterprise']
    acquisition_channels = ['Online', 'Referral', 'Direct Sales', 'Partner', 'Social Media']
    
    data = {
        'customer_id': [f'CUST{str(i).zfill(5)}' for i in range(1, n_records + 1)],
        'age': np.random.normal(40, 15),
        'gender': np.random.choice(['M', 'F'], n_records),
        'subscription_type': np.random.choice(subscription_types, n_records, p=[0.5, 0.35, 0.15]),
        'monthly_charges': None,  # Will be calculated
        'total_charges': None,  # Will be calculated
        'tenure_months': np.random.exponential(12),
        'contract_length': np.random.choice([1, 12, 24], n_records, p=[0.4, 0.4, 0.2]),
        'acquisition_channel': np.random.choice(acquisition_channels, n_records),
        'support_calls': np.random.poisson(2),
        'payment_method': np.random.choice(['Credit Card', 'Bank Transfer', 'Digital Wallet'], n_records),
        'paperless_billing': np.random.choice([True, False], n_records, p=[0.7, 0.3]),
        'auto_pay': np.random.choice([True, False], n_records, p=[0.6, 0.4]),
        'family_size': np.random.poisson(2) + 1,
        'internet_usage_gb': np.random.gamma(3, 15),
        'streaming_services': np.random.poisson(2),
        'satisfaction_score': np.random.beta(7, 3) * 10,
        'last_interaction_days': np.random.exponential(30),
        'promotional_offers_used': np.random.poisson(1),
        'churned': None,  # Will be calculated
    }
    
    df = pd.DataFrame(data)
    
    # Calculate monthly charges based on subscription type
    base_charges = {'Basic': 29.99, 'Premium': 59.99, 'Enterprise': 99.99}
    df['monthly_charges'] = (df['subscription_type'].map(base_charges) * 
                           np.random.normal(1, 0.1, len(df)))
    
    # Calculate total charges
    df['total_charges'] = df['monthly_charges'] * df['tenure_months']
    
    # Calculate churn probability
    churn_prob = (
        (df['satisfaction_score'] < 5) * 0.4 +
        (df['support_calls'] > 5) * 0.3 +
        (df['last_interaction_days'] > 60) * 0.2 +
        (df['tenure_months'] < 3) * 0.2 +
        np.random.uniform(0, 0.1, len(df))
    ).clip(0, 1)
    
    df['churned'] = np.random.binomial(1, churn_prob, len(df))
    
    # Clean up data
    df['age'] = df['age'].clip(18, 80)
    df['tenure_months'] = df['tenure_months'].clip(1, 72)
    df['satisfaction_score'] = df['satisfaction_score'].clip(1, 10)
    df['family_size'] = df['family_size'].clip(1, 8)
    
    # Add missing values
    missing_indices = np.random.choice(df.index, size=int(0.03 * len(df)), replace=False)
    df.loc[missing_indices, 'total_charges'] = np.nan
    
    return df

def generate_medical_dataset():
    """Generate a medical dataset for health analysis"""
    n_records = 6000
    
    conditions = ['Diabetes', 'Hypertension', 'Heart Disease', 'Asthma', 'None']
    blood_types = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']
    insurance = ['Private', 'Medicare', 'Medicaid', 'None']
    
    data = {
        'patient_id': [f'PAT{str(i).zfill(5)}' for i in range(1, n_records + 1)],
        'age': np.random.normal(45, 20),
        'gender': np.random.choice(['Male', 'Female'], n_records),
        'blood_type': np.random.choice(blood_types, n_records),
        'weight_kg': np.random.normal(75, 15),
        'height_cm': np.random.normal(170, 10),
        'bmi': None,  # Will be calculated
        'blood_pressure_systolic': np.random.normal(125, 20),
        'blood_pressure_diastolic': np.random.normal(80, 15),
        'heart_rate': np.random.normal(75, 15),
        'cholesterol': np.random.normal(200, 40),
        'blood_sugar': np.random.normal(100, 25),
        'smoking': np.random.choice(['Never', 'Former', 'Current'], n_records, p=[0.5, 0.3, 0.2]),
        'alcohol_consumption': np.random.choice(['None', 'Light', 'Moderate', 'Heavy'], n_records, p=[0.3, 0.4, 0.25, 0.05]),
        'exercise_hours_week': np.random.exponential(3),
        'primary_condition': np.random.choice(conditions, n_records, p=[0.15, 0.2, 0.1, 0.1, 0.45]),
        'medications_count': np.random.poisson(2),
        'hospital_visits_year': np.random.poisson(1),
        'insurance_type': np.random.choice(insurance, n_records, p=[0.5, 0.2, 0.2, 0.1]),
        'annual_cost': None,  # Will be calculated
        'risk_score': None,  # Will be calculated
    }
    
    df = pd.DataFrame(data)
    
    # Calculate BMI
    df['bmi'] = df['weight_kg'] / (df['height_cm'] / 100) ** 2
    
    # Calculate annual cost based on various factors
    base_cost = 2000
    condition_multiplier = {'Diabetes': 2.5, 'Hypertension': 1.8, 'Heart Disease': 3.0, 'Asthma': 1.5, 'None': 1.0}
    age_multiplier = (df['age'] / 40) ** 1.5
    
    df['annual_cost'] = (base_cost * 
                        df['primary_condition'].map(condition_multiplier) *
                        age_multiplier *
                        (1 + df['medications_count'] * 0.2) *
                        (1 + df['hospital_visits_year'] * 0.5) *
                        np.random.normal(1, 0.3, len(df)))
    
    # Calculate risk score
    df['risk_score'] = (
        (df['age'] > 60) * 20 +
        (df['bmi'] > 30) * 15 +
        (df['smoking'] == 'Current') * 25 +
        (df['cholesterol'] > 240) * 10 +
        (df['blood_pressure_systolic'] > 140) * 15 +
        (df['primary_condition'] != 'None') * 20 +
        np.random.uniform(0, 10, len(df))
    ).clip(0, 100)
    
    # Clean up data
    df['age'] = df['age'].clip(0, 100)
    df['weight_kg'] = df['weight_kg'].clip(30, 200)
    df['height_cm'] = df['height_cm'].clip(140, 220)
    df['blood_pressure_systolic'] = df['blood_pressure_systolic'].clip(80, 200)
    df['blood_pressure_diastolic'] = df['blood_pressure_diastolic'].clip(50, 120)
    df['heart_rate'] = df['heart_rate'].clip(40, 120)
    df['cholesterol'] = df['cholesterol'].clip(100, 400)
    df['blood_sugar'] = df['blood_sugar'].clip(70, 300)
    df['exercise_hours_week'] = df['exercise_hours_week'].clip(0, 20)
    df['annual_cost'] = df['annual_cost'].clip(500, 50000)
    
    # Add missing values
    missing_indices = np.random.choice(df.index, size=int(0.04 * len(df)), replace=False)
    df.loc[missing_indices, 'cholesterol'] = np.nan
    
    missing_indices = np.random.choice(df.index, size=int(0.02 * len(df)), replace=False)
    df.loc[missing_indices, 'blood_sugar'] = np.nan
    
    return df

def main():
    """Generate all sample datasets"""
    print("Generating sample datasets...")
    
    # Generate datasets
    datasets = {
        'sales_data.csv': generate_sales_dataset(),
        'employee_data.csv': generate_employee_dataset(),
        'customer_data.csv': generate_customer_dataset(),
        'medical_data.csv': generate_medical_dataset()
    }
    
    # Save datasets
    for filename, df in datasets.items():
        df.to_csv(filename, index=False)
        print(f"Generated {filename}: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"  - Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
        print(f"  - Missing values: {df.isnull().sum().sum()}")
        print(f"  - Data types: {df.dtypes.value_counts().to_dict()}")
        print()
    
    print("Sample datasets generated successfully!")
    print("\nDataset descriptions:")
    print("1. sales_data.csv - E-commerce sales with seasonal patterns, customer segments")
    print("2. employee_data.csv - HR data with salary, performance, churn risk")
    print("3. customer_data.csv - Customer churn analysis with subscription data")
    print("4. medical_data.csv - Healthcare data with risk assessment")

if __name__ == "__main__":
    main()