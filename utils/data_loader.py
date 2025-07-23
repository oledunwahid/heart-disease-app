"""
Data Loading Utilities for Heart Disease Prediction App
=======================================================
"""
import pandas as pd
import numpy as np
import streamlit as st
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


@st.cache_data
def load_heart_disease_dataset():
    """
    Load the heart disease dataset from multiple possible sources
    """
    try:
        # Try to load from UCI ML Repository first
        try:
            from ucimlrepo import fetch_ucirepo
            st.info("🔄 Loading dataset from UCI ML Repository...")
            heart_disease = fetch_ucirepo(id=45)
            X = heart_disease.data.features
            y = heart_disease.data.targets
            
            # Combine features and target
            df = pd.concat([X, y], axis=1)
            st.success("✅ Successfully loaded UCI Heart Disease Dataset!")
            return df
            
        except Exception as e:
            st.warning(f"⚠️ Could not load from UCI repository: {str(e)}")
            
        # Try to load from local files
        possible_files = [
            'data/heart_disease.csv',
            'data/heart_disease_complete.csv',
            'heart_disease.csv',
            'heart_disease_complete.csv'
        ]
        
        for file_path in possible_files:
            if os.path.exists(file_path):
                st.info(f"📁 Loading dataset from {file_path}...")
                df = pd.read_csv(file_path)
                st.success(f"✅ Successfully loaded dataset from {file_path}!")
                return df
        
        # Generate sample data if no file found
        st.warning("⚠️ No dataset file found. Generating sample data for demonstration...")
        return generate_sample_data()
        
    except Exception as e:
        st.error(f"❌ Error loading dataset: {str(e)}")
        return generate_sample_data()


def generate_sample_data():
    """
    Generate realistic sample heart disease data for demonstration
    """
    np.random.seed(42)
    n_samples = 303
    
    # Generate realistic medical data
    ages = np.random.normal(54, 9, n_samples).astype(int)
    ages = np.clip(ages, 29, 79)
    
    data = {
        'age': ages,
        'sex': np.random.choice([0, 1], n_samples, p=[0.32, 0.68]),  # More males
        'cp': np.random.choice([0, 1, 2, 3], n_samples, p=[0.47, 0.17, 0.29, 0.07]),
        'trestbps': np.random.normal(131, 17, n_samples).astype(int),
        'chol': np.random.normal(246, 51, n_samples).astype(int),
        'fbs': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
        'restecg': np.random.choice([0, 1, 2], n_samples, p=[0.52, 0.46, 0.02]),
        'thalach': np.random.normal(149, 23, n_samples).astype(int),
        'exang': np.random.choice([0, 1], n_samples, p=[0.67, 0.33]),
        'oldpeak': np.random.exponential(1.0, n_samples).round(1),
        'slope': np.random.choice([0, 1, 2], n_samples, p=[0.21, 0.50, 0.29]),
        'ca': np.random.choice([0, 1, 2, 3], n_samples, p=[0.59, 0.22, 0.12, 0.07]),
        'thal': np.random.choice([0, 1, 2, 3], n_samples, p=[0.54, 0.36, 0.07, 0.03]),
        'num': np.random.choice([0, 1], n_samples, p=[0.54, 0.46])  # Target variable
    }
    
    # Add some correlations to make it more realistic
    for i in range(n_samples):
        # Older age increases risk
        if data['age'][i] > 60:
            data['num'][i] = np.random.choice([0, 1], p=[0.3, 0.7])
        
        # Chest pain correlation
        if data['cp'][i] == 0:  # Typical angina
            data['num'][i] = np.random.choice([0, 1], p=[0.2, 0.8])
    
    df = pd.DataFrame(data)
    
    # Ensure realistic ranges
    df['trestbps'] = df['trestbps'].clip(94, 200)
    df['chol'] = df['chol'].clip(126, 564)
    df['thalach'] = df['thalach'].clip(71, 202)
    df['oldpeak'] = df['oldpeak'].clip(0, 6.2)
    
    st.info("📊 Generated 303 realistic sample records for demonstration")
    return df


@st.cache_data
def preprocess_data(df):
    """
    Preprocess the heart disease dataset
    """
    try:
        # Make a copy to avoid modifying original data
        df_processed = df.copy()
        
        # Handle missing values
        missing_info = df_processed.isnull().sum()
        if missing_info.sum() > 0:
            st.warning("⚠️ Found missing values. Applying median imputation...")
            
            # Separate numeric and categorical columns
            numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
            categorical_cols = df_processed.select_dtypes(exclude=[np.number]).columns
            
            # Impute numeric columns with median
            if len(numeric_cols) > 0:
                imputer_numeric = SimpleImputer(strategy='median')
                df_processed[numeric_cols] = imputer_numeric.fit_transform(df_processed[numeric_cols])
            
            # Impute categorical columns with mode
            if len(categorical_cols) > 0:
                imputer_categorical = SimpleImputer(strategy='most_frequent')
                df_processed[categorical_cols] = imputer_categorical.fit_transform(df_processed[categorical_cols])
        
        # Ensure target column exists
        target_columns = ['num', 'target', 'heart_disease', 'diagnosis']
        target_col = None
        
        for col in target_columns:
            if col in df_processed.columns:
                target_col = col
                break
        
        if target_col is None:
            st.error("❌ No target column found. Expected 'num', 'target', 'heart_disease', or 'diagnosis'")
            return None, None, None, None
        
        # Prepare features and target
        X = df_processed.drop(target_col, axis=1)
        y = df_processed[target_col]
        
        # Convert target to binary if needed
        if y.nunique() > 2:
            y = (y > 0).astype(int)
        
        # Store feature names
        feature_names = X.columns.tolist()
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Convert back to DataFrames for easier handling
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_names)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_names)
        
        preprocessing_info = {
            'original_shape': df.shape,
            'processed_shape': df_processed.shape,
            'missing_values_filled': missing_info.sum(),
            'train_size': len(X_train),
            'test_size': len(X_test),
            'feature_count': len(feature_names),
            'target_distribution': y.value_counts().to_dict()
        }
        
        return (X_train_scaled, X_test_scaled, y_train, y_test, 
                scaler, feature_names, preprocessing_info)
        
    except Exception as e:
        st.error(f"❌ Error in preprocessing: {str(e)}")
        return None, None, None, None, None, None, None


def get_feature_info():
    """
    Return feature information dictionary
    """
    return {
        'age': '👴 Patient age in years (29-79)',
        'sex': '👫 Gender (0: Female, 1: Male)',
        'cp': '💔 Chest pain type (0: Typical Angina, 1: Atypical Angina, 2: Non-Anginal, 3: Asymptomatic)',
        'trestbps': '🩺 Resting blood pressure in mm Hg (94-200)',
        'chol': '🧪 Serum cholesterol in mg/dl (126-564)',
        'fbs': '🍯 Fasting blood sugar > 120 mg/dl (0: No, 1: Yes)',
        'restecg': '📈 Resting ECG results (0: Normal, 1: ST-T Abnormality, 2: LV Hypertrophy)',
        'thalach': '💓 Maximum heart rate achieved (71-202)',
        'exang': '🏃 Exercise induced angina (0: No, 1: Yes)',
        'oldpeak': '📊 ST depression induced by exercise (0-6.2)',
        'slope': '📈 Slope of peak exercise ST segment (0: Up, 1: Flat, 2: Down)',
        'ca': '🫀 Number of major vessels colored by fluoroscopy (0-3)',
        'thal': '🩸 Thalassemia (0: Normal, 1: Fixed Defect, 2: Reversible Defect)'
    }