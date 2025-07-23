"""
Model Training Utilities for Heart Disease Prediction App
=========================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, classification_report
)


@st.cache_resource
def train_all_models(X_train, y_train, X_test, y_test):
    """
    Train all machine learning models and return results
    """
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=100, 
            max_depth=10, 
            random_state=42,
            n_jobs=-1
        ),
        'K-Nearest Neighbors': KNeighborsClassifier(
            n_neighbors=5
        ),
        'Logistic Regression': LogisticRegression(
            random_state=42, 
            max_iter=1000,
            solver='liblinear'
        ),
        'Support Vector Machine': SVC(
            kernel='rbf', 
            random_state=42, 
            probability=True
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100, 
            random_state=42
        ),
        'Neural Network': MLPClassifier(
            hidden_layer_sizes=(100,), 
            max_iter=1000, 
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
    }
    
    results = {}
    trained_models = {}
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, (name, model) in enumerate(models.items()):
        try:
            status_text.text(f"🔄 Training {name}...")
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = None
            
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            if y_pred_proba is not None:
                auc = roc_auc_score(y_test, y_pred_proba)
            else:
                auc = roc_auc_score(y_test, y_pred)
            
            # Store results
            results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc_roc': auc,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            trained_models[name] = model
            
            # Update progress
            progress = (i + 1) / len(models)
            progress_bar.progress(progress)
            
        except Exception as e:
            st.error(f"❌ Error training {name}: {str(e)}")
            results[name] = {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'auc_roc': 0.0,
                'predictions': None,
                'probabilities': None
            }
    
    status_text.text("✅ All models trained successfully!")
    progress_bar.empty()
    
    return trained_models, results


def get_feature_importance(model, feature_names):
    """
    Extract feature importance from the model
    """
    try:
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_[0])
        else:
            return None
        
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        return feature_importance_df
        
    except Exception as e:
        st.error(f"Error extracting feature importance: {str(e)}")
        return None


def make_prediction(model, features_scaled, model_name):
    """
    Make a single prediction with a trained model
    """
    try:
        prediction = model.predict(features_scaled)[0]
        
        if hasattr(model, 'predict_proba'):
            probability = model.predict_proba(features_scaled)[0]
        else:
            probability = [1-prediction, prediction]
        
        return {
            'prediction': int(prediction),
            'probability': probability,
            'risk_level': 'High Risk' if prediction == 1 else 'Low Risk',
            'model_used': model_name
        }
        
    except Exception as e:
        st.error(f"Error making prediction with {model_name}: {str(e)}")
        return None


def get_clinical_interpretations():
    """
    Return clinical interpretations for features
    """
    return {
        'thalach': 'Maximum heart rate reflects cardiac functional capacity and exercise tolerance',
        'cp': 'Chest pain type is a primary symptom for cardiovascular disease assessment',
        'thal': 'Thalassemia affects blood oxygen transport capacity and cardiac stress',
        'ca': 'Number of major vessels indicates anatomical extent of coronary artery disease',
        'oldpeak': 'ST depression shows exercise-induced ischemic changes',
        'age': 'Age is a fundamental cardiovascular risk factor',
        'chol': 'Cholesterol level is an established risk factor for atherosclerosis',
        'trestbps': 'Blood pressure is a major modifiable cardiovascular risk factor'
    }


def get_model_analysis():
    """
    Return model pros and cons analysis
    """
    return {
        'Random Forest': {
            'pros': ['High accuracy', 'Feature importance', 'Handles overfitting well', 'Works with mixed data types'],
            'cons': ['Less interpretable', 'Can be slow on large datasets', 'Memory intensive']
        },
        'K-Nearest Neighbors': {
            'pros': ['Simple and intuitive', 'No assumptions about data', 'Good for local patterns'],
            'cons': ['Sensitive to irrelevant features', 'Computationally expensive', 'Sensitive to scale']
        },
        'Logistic Regression': {
            'pros': ['Highly interpretable', 'Fast training', 'Probabilistic output', 'No overfitting with regularization'],
            'cons': ['Assumes linear relationship', 'Sensitive to outliers', 'Requires feature scaling']
        },
        'Support Vector Machine': {
            'pros': ['Effective in high dimensions', 'Memory efficient', 'Versatile with kernels'],
            'cons': ['Slow on large datasets', 'No probabilistic output', 'Sensitive to feature scaling']
        },
        'Gradient Boosting': {
            'pros': ['High predictive power', 'Handles mixed data types', 'Feature importance'],
            'cons': ['Prone to overfitting', 'Requires tuning', 'Computationally intensive']
        },
        'Neural Network': {
            'pros': ['Can learn complex patterns', 'Flexible architecture', 'Good for large datasets'],
            'cons': ['Black box', 'Requires lots of data', 'Many hyperparameters']
        }
    }