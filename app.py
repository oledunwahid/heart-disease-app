#!/usr/bin/env python3
"""
Heart Disease Prediction App - Main Application
==============================================
Author: Khaled Makkawirelang
Institution: Universitas Trilogi, Program Studi Teknik Informatika
Description: AI-powered cardiovascular risk assessment using machine learning
"""

import streamlit as st
import pandas as pd
import numpy as np
import warnings
from datetime import datetime
import json
from sklearn.model_selection import cross_val_score

# Import utilities
from utils import (
    load_heart_disease_dataset, preprocess_data, get_feature_info,
    train_all_models, get_feature_importance, make_prediction, 
    get_clinical_interpretations, get_model_analysis,
    create_performance_comparison_chart, create_confusion_matrix_plot,
    create_roc_curve_plot, create_feature_importance_plot,
    create_dataset_overview_plots, create_radar_chart,
    create_prediction_trends_plot, create_risk_distribution_plot
)

# Suppress warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Heart Disease Prediction App",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/heart-disease-app',
        'Report a bug': "https://github.com/yourusername/heart-disease-app/issues",
        'About': "# Heart Disease Prediction App\nBuilt with ❤️ using Streamlit and Scikit-learn"
    }
)

# =============================================================================
# CUSTOM CSS AND STYLING - MINIMAL CONTRAST FIXES ONLY
# =============================================================================

st.markdown("""
<style>
    /* Main container styling */
    .main-header {
        font-size: 3.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: white !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    /* Success and warning cards */
    .success-card {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    
    .warning-card {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: bold;
        font-size: 1.1rem;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Progress bar styling */
    .stProgress .st-bo {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Info boxes - FIXED CONTRAST ISSUE */
    .info-box {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
        color: #1f2937;
    }
    
    .info-box h4 {
        color: #1f2937 !important;
    }
    
    .info-box p, .info-box li {
        color: #374151 !important;
    }
    
    /* Feature importance styling - FIXED CONTRAST ISSUE */
    .feature-item {
        background: white;
        padding: 0.8rem;
        margin: 0.3rem 0;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .feature-item strong {
        color: #1f2937 !important;
    }
    
    .feature-item small {
        color: #6b7280 !important;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'model_metrics' not in st.session_state:
    st.session_state.model_metrics = {}
if 'feature_names' not in st.session_state:
    st.session_state.feature_names = []
if 'predictions_log' not in st.session_state:
    st.session_state.predictions_log = []
if 'dataset' not in st.session_state:
    st.session_state.dataset = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None

# =============================================================================
# MAIN APPLICATION FUNCTIONS
# =============================================================================

def main():
    """Main application function"""
    # Header
    st.markdown('<h1 class="main-header">Heart Disease Prediction App</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p style="font-size: 1.1rem; color: white;">
            Advanced machine learning system for early detection and prediction of heart disease
        </p>
        <p style="font-style: italic; color: white;">
            Based on research by Khaled Makkawirelang - Universitas Trilogi
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown("### 🧭 Navigation")
        page = st.selectbox(
            "Choose a page:",
            ["🏠 Home", "🔮 Predict", "📊 Visuals", "📝 Logs", "⚙️ Model Info", "📖 About"],
            index=0
        )
        
        st.markdown("---")
        
        # Model training status
        if st.session_state.models_trained:
            st.success("✅ Models Ready")
            best_model = max(st.session_state.model_metrics.items(), key=lambda x: x[1]['accuracy'])
            st.info(f"🏆 Best: {best_model[0]}\n📊 Accuracy: {best_model[1]['accuracy']:.1%}")
        else:
            st.warning("⏳ Models Not Trained")
            if st.button("🚀 Train Models Now"):
                initialize_models()
        
        st.markdown("---")
        
        # Quick stats
        if st.session_state.dataset is not None:
            st.markdown("### 📊 Quick Stats")
            st.metric("Dataset Size", len(st.session_state.dataset))
            if st.session_state.predictions_log:
                st.metric("Predictions Made", len(st.session_state.predictions_log))
    
    # Page routing
    if page == "🏠 Home":
        home_page()
    elif page == "🔮 Predict":
        predict_page()
    elif page == "📊 Visuals":
        visuals_page()
    elif page == "📝 Logs":
        logs_page()
    elif page == "⚙️ Model Info":
        model_info_page()
    elif page == "📖 About":
        about_page()


def initialize_models():
    """Initialize and train all machine learning models"""
    with st.spinner("🔄 Loading and preprocessing data..."):
        # Load dataset
        df = load_heart_disease_dataset()
        if df is None:
            st.error("❌ Failed to load dataset!")
            return
        
        st.session_state.dataset = df
        
        # Preprocess data
        result = preprocess_data(df)
        if result[0] is None:
            st.error("❌ Failed to preprocess data!")
            return
        
        X_train, X_test, y_train, y_test, scaler, feature_names, preprocessing_info = result
        
        # Store in session state
        st.session_state.scaler = scaler
        st.session_state.feature_names = feature_names
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test
    
    with st.spinner("🤖 Training machine learning models..."):
        # Train models
        models, results = train_all_models(X_train, y_train, X_test, y_test)
        
        # Store results
        st.session_state.models = models
        st.session_state.model_metrics = results
        st.session_state.models_trained = True
    
    st.success("🎉 All models trained successfully!")
    st.balloons()


def home_page():
    """Home page with overview and quick stats"""
    # Load data if not already loaded
    if st.session_state.dataset is None:
        initialize_models()
    
    if st.session_state.dataset is None:
        st.error("❌ Unable to load dataset. Please check your data files.")
        return
    
    df = st.session_state.dataset
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h2>📊</h2>
            <h3>{len(df)}</h3>
            <p>Total Patients</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if st.session_state.models_trained:
            best_accuracy = max(st.session_state.model_metrics.values(), key=lambda x: x['accuracy'])['accuracy']
            st.markdown(f"""
            <div class="metric-card">
                <h2>🎯</h2>
                <h3>{best_accuracy:.1%}</h3>
                <p>Best Accuracy</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="metric-card">
                <h2>⏳</h2>
                <h3>Pending</h3>
                <p>Model Training</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        target_col = 'num' if 'num' in df.columns else df.columns[-1]
        disease_count = df[target_col].sum()
        st.markdown(f"""
        <div class="metric-card">
            <h2>❤️</h2>
            <h3>{disease_count}</h3>
            <p>Disease Cases</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        healthy_count = len(df) - disease_count
        st.markdown(f"""
        <div class="metric-card">
            <h2>💚</h2>
            <h3>{healthy_count}</h3>
            <p>Healthy Cases</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Dataset overview
    st.markdown("### 📈 Dataset Overview")
    
    if st.checkbox("Show detailed dataset analysis", value=True):
        overview_fig = create_dataset_overview_plots(df)
        st.plotly_chart(overview_fig, use_container_width=True)
    
    # Feature information
    st.markdown("### 📋 Medical Features Information")
    
    feature_info = get_feature_info()
    cols = st.columns(2)
    for i, (feature, description) in enumerate(feature_info.items()):
        with cols[i % 2]:
            st.markdown(f"""
            <div class="feature-item">
                <strong>{feature.upper()}</strong><br>
                <small>{description}</small>
            </div>
            """, unsafe_allow_html=True)
    
    # Quick actions
    st.markdown("### 🚀 Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔮 Make Prediction", key="home_predict"):
            st.info("Navigate to the Predict page using the sidebar!")
    
    with col2:
        if st.button("📊 View Analytics", key="home_analytics"):
            st.info("Navigate to the Visuals page using the sidebar!")
    
    with col3:
        if st.button("📝 Check Logs", key="home_logs"):
            st.info("Navigate to the Logs page using the sidebar!")


def predict_page():
    """Prediction interface page"""
    st.markdown("### 🔮 Heart Disease Risk Prediction")
    
    if not st.session_state.models_trained:
        st.warning("⚠️ Models not trained yet. Please train the models first.")
        if st.button("🚀 Train Models Now"):
            initialize_models()
        return
    
    st.markdown("Enter the patient's medical information below to get an AI-powered cardiovascular risk assessment.")
    
    # Create two columns for input form
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### 📋 Patient Information")
        
        # Basic Demographics
        st.markdown("**👥 Demographics**")
        col_demo1, col_demo2 = st.columns(2)
        
        with col_demo1:
            age = st.slider("Age (years)", 20, 100, 50, help="Patient's age in years")
        
        with col_demo2:
            sex = st.selectbox("Gender", 
                             options=[0, 1], 
                             format_func=lambda x: "Female" if x == 0 else "Male",
                             help="Patient's biological sex")
        
        # Clinical Measurements
        st.markdown("**🩺 Clinical Measurements**")
        col_clin1, col_clin2 = st.columns(2)
        
        with col_clin1:
            trestbps = st.slider("Resting Blood Pressure (mm Hg)", 80, 220, 120,
                               help="Systolic blood pressure at rest")
            chol = st.slider("Serum Cholesterol (mg/dl)", 100, 500, 200,
                           help="Total cholesterol level in blood")
            fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl",
                             options=[0, 1],
                             format_func=lambda x: "No" if x == 0 else "Yes",
                             help="Whether fasting blood sugar exceeds 120 mg/dl")
        
        with col_clin2:
            cp = st.selectbox("Chest Pain Type",
                            options=[0, 1, 2, 3],
                            format_func=lambda x: [
                                "Typical Angina", "Atypical Angina", 
                                "Non-Anginal Pain", "Asymptomatic"
                            ][x],
                            help="Type of chest pain experienced")
            
            restecg = st.selectbox("Resting ECG Results",
                                 options=[0, 1, 2],
                                 format_func=lambda x: [
                                     "Normal", "ST-T Wave Abnormality", 
                                     "Left Ventricular Hypertrophy"
                                 ][x],
                                 help="Resting electrocardiographic results")
        
        # Exercise Test Results
        st.markdown("**🏃 Exercise Test Results**")
        col_ex1, col_ex2 = st.columns(2)
        
        with col_ex1:
            thalach = st.slider("Maximum Heart Rate Achieved", 60, 220, 150,
                              help="Maximum heart rate during exercise test")
            exang = st.selectbox("Exercise Induced Angina",
                               options=[0, 1],
                               format_func=lambda x: "No" if x == 0 else "Yes",
                               help="Whether exercise induced chest pain")
        
        with col_ex2:
            oldpeak = st.slider("ST Depression", 0.0, 8.0, 1.0, 0.1,
                              help="ST depression induced by exercise relative to rest")
            slope = st.selectbox("ST Slope",
                               options=[0, 1, 2],
                               format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x],
                               help="Slope of peak exercise ST segment")
        
        # Advanced Diagnostics
        st.markdown("**🔬 Advanced Diagnostics**")
        col_adv1, col_adv2 = st.columns(2)
        
        with col_adv1:
            ca = st.selectbox("Number of Major Vessels",
                            options=[0, 1, 2, 3],
                            help="Number of major vessels (0-3) colored by fluoroscopy")
        
        with col_adv2:
            thal = st.selectbox("Thalassemia",
                              options=[0, 1, 2, 3],
                              format_func=lambda x: [
                                  "Normal", "Fixed Defect", 
                                  "Reversible Defect", "Unknown"
                              ][x],
                              help="Blood disorder affecting oxygen transport")
        
        # Prediction buttons
        st.markdown("---")
        col_btn1, col_btn2, col_btn3 = st.columns(3)
        
        with col_btn1:
            predict_single = st.button("🔮 Predict with Best Model", 
                                     type="primary",
                                     use_container_width=True)
        
        with col_btn2:
            predict_all = st.button("📊 Compare All Models",
                                   use_container_width=True)
        
        with col_btn3:
            reset_form = st.button("🔄 Reset Form",
                                  use_container_width=True)
        
        if reset_form:
            st.rerun()
    
    # Results column
    with col2:
        st.markdown("#### 🎯 Prediction Results")
        
        # Prepare input features
        features = [age, sex, cp, trestbps, chol, fbs, restecg, 
                   thalach, exang, oldpeak, slope, ca, thal]
        
        if predict_single or predict_all:
            # Scale features
            features_scaled = st.session_state.scaler.transform([features])
            
            if predict_single:
                # Use best model
                best_model_name = max(st.session_state.model_metrics.items(), 
                                    key=lambda x: x[1]['accuracy'])[0]
                model = st.session_state.models[best_model_name]
                
                # Make prediction using utility function
                result = make_prediction(model, features_scaled, best_model_name)
                
                if result:
                    # Display result
                    if result['prediction'] == 1:
                        st.markdown(f"""
                        <div class="warning-card">
                            <h2>⚠️ HIGH RISK</h2>
                            <h3>{result['probability'][1]:.1%} Risk Probability</h3>
                            <p>Model: {best_model_name}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.error("⚠️ **High cardiovascular risk detected.** Please consult with a healthcare professional.")
                    else:
                        st.markdown(f"""
                        <div class="success-card">
                            <h2>✅ LOW RISK</h2>
                            <h3>{result['probability'][0]:.1%} Healthy Probability</h3>
                            <p>Model: {best_model_name}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.success("✅ **Low cardiovascular risk.** Continue maintaining a healthy lifestyle.")
                    
                    # Progress bar
                    risk_prob = result['probability'][1]
                    st.progress(risk_prob)
                    
                    # Save to logs
                    log_entry = {
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'model_used': best_model_name,
                        'prediction': result['prediction'],
                        'probability': float(result['probability'][1]),
                        'risk_level': result['risk_level'],
                        'features': {
                            'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps,
                            'chol': chol, 'fbs': fbs, 'restecg': restecg,
                            'thalach': thalach, 'exang': exang, 'oldpeak': oldpeak,
                            'slope': slope, 'ca': ca, 'thal': thal
                        }
                    }
                    st.session_state.predictions_log.append(log_entry)
                    
                    st.success("✅ Prediction saved to logs!")
            
            elif predict_all:
                # Compare all models
                st.markdown("**🔍 All Models Comparison:**")
                
                results_data = []
                for model_name, model in st.session_state.models.items():
                    result = make_prediction(model, features_scaled, model_name)
                    if result:
                        results_data.append({
                            'Model': model_name,
                            'Prediction': result['risk_level'],
                            'Risk Probability': f"{result['probability'][1]:.1%}"
                        })
                
                if results_data:
                    results_df = pd.DataFrame(results_data)
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Consensus
                    high_risk_count = sum(1 for r in results_data if r['Prediction'] == 'High Risk')
                    consensus_risk = high_risk_count >= len(results_data) / 2
                    
                    if consensus_risk:
                        st.warning(f"⚠️ **Consensus: HIGH RISK** ({high_risk_count}/{len(results_data)} models)")
                    else:
                        st.success(f"✅ **Consensus: LOW RISK** ({len(results_data)-high_risk_count}/{len(results_data)} models)")


def visuals_page():
    """Data visualization and analytics page"""
    st.markdown("### 📊 Data Visualizations & Analytics")
    
    if not st.session_state.models_trained:
        st.warning("⚠️ Models not trained yet. Please train the models first.")
        if st.button("🚀 Train Models Now"):
            initialize_models()
        return
    
    # Tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Dataset Analysis", 
        "🎯 Model Performance", 
        "🔍 Feature Importance", 
        "📊 Advanced Analytics"
    ])
    
    with tab1:
        st.markdown("#### 📈 Dataset Analysis")
        
        df = st.session_state.dataset
        
        # Dataset overview plot
        overview_fig = create_dataset_overview_plots(df)
        st.plotly_chart(overview_fig, use_container_width=True)
        
        # Statistical summary
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📊 Statistical Summary**")
            st.dataframe(df.describe(), use_container_width=True)
        
        with col2:
            st.markdown("**🔍 Data Quality Report**")
            quality_report = {
                'Total Records': len(df),
                'Features': len(df.columns) - 1,
                'Missing Values': df.isnull().sum().sum(),
                'Duplicate Records': df.duplicated().sum(),
                'Disease Cases': df.iloc[:, -1].sum(),
                'Healthy Cases': len(df) - df.iloc[:, -1].sum()
            }
            
            for metric, value in quality_report.items():
                st.metric(metric, value)
    
    with tab2:
        st.markdown("#### 🎯 Model Performance Analysis")
        
        # Performance comparison chart
        perf_fig = create_performance_comparison_chart(st.session_state.model_metrics)
        st.plotly_chart(perf_fig, use_container_width=True)
        
        # ROC curves
        roc_fig = create_roc_curve_plot(st.session_state.model_metrics, st.session_state.y_test)
        st.plotly_chart(roc_fig, use_container_width=True)
        
        # Detailed metrics table
        st.markdown("**📋 Detailed Performance Metrics**")
        metrics_df = pd.DataFrame(st.session_state.model_metrics).T
        metrics_df = metrics_df.round(4)
        st.dataframe(metrics_df, use_container_width=True)
    
    with tab3:
        st.markdown("#### 🔍 Feature Importance Analysis")
        
        # Select model for feature importance
        model_names = list(st.session_state.models.keys())
        selected_model = st.selectbox("Select Model for Feature Importance:", model_names)
        
        if selected_model:
            model = st.session_state.models[selected_model]
            feature_importance_df = get_feature_importance(model, st.session_state.feature_names)
            
            if feature_importance_df is not None:
                # Feature importance plot
                importance_fig = create_feature_importance_plot(feature_importance_df, selected_model)
                if importance_fig:
                    st.plotly_chart(importance_fig, use_container_width=True)
                
                # Feature importance table
                st.markdown("**📋 Feature Importance Ranking**")
                st.dataframe(feature_importance_df, use_container_width=True)
                
                # Clinical interpretation
                st.markdown("**🏥 Clinical Interpretation**")
                top_features = feature_importance_df.head(5)
                clinical_interpretations = get_clinical_interpretations()
                
                for _, row in top_features.iterrows():
                    feature = row['Feature']
                    importance = row['Importance']
                    interpretation = clinical_interpretations.get(feature, 'Important clinical parameter')
                    
                    st.markdown(f"""
                    <div class="info-box">
                        <strong>{feature.upper()}</strong> (Importance: {importance:.4f})<br>
                        <small>{interpretation}</small>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info(f"Feature importance not available for {selected_model}")
    
    with tab4:
        st.markdown("#### 📊 Advanced Analytics")
        
        # Confusion matrices for all models
        st.markdown("**🎯 Confusion Matrices**")
        
        model_names = list(st.session_state.model_metrics.keys())
        selected_models = st.multiselect(
            "Select models to compare:",
            model_names,
            default=model_names[:3]
        )
        
        if selected_models:
            cols = st.columns(min(len(selected_models), 3))
            
            for i, model_name in enumerate(selected_models):
                if model_name in st.session_state.model_metrics:
                    result = st.session_state.model_metrics[model_name]
                    if result['predictions'] is not None:
                        cm_fig = create_confusion_matrix_plot(
                            st.session_state.y_test, 
                            result['predictions'], 
                            model_name
                        )
                        with cols[i % 3]:
                            st.plotly_chart(cm_fig, use_container_width=True)


def logs_page():
    """Prediction logs and history page"""
    st.markdown("### 📝 Prediction Logs & History")
    
    if not st.session_state.predictions_log:
        st.info("📋 No predictions made yet. Go to the Predict page to make some predictions!")
        return
    
    # Summary statistics
    logs_df = pd.DataFrame(st.session_state.predictions_log)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Predictions", len(logs_df))
    
    with col2:
        high_risk_count = len(logs_df[logs_df['risk_level'] == 'High Risk'])
        st.metric("High Risk Cases", high_risk_count)
    
    with col3:
        low_risk_count = len(logs_df[logs_df['risk_level'] == 'Low Risk'])
        st.metric("Low Risk Cases", low_risk_count)
    
    with col4:
        if len(logs_df) > 0:
            avg_risk = logs_df['probability'].mean()
            st.metric("Average Risk", f"{avg_risk:.1%}")
    
    # Logs visualization
    st.markdown("#### 📊 Prediction Trends")
    
    if len(logs_df) > 1:
        # Convert timestamp to datetime
        logs_df['datetime'] = pd.to_datetime(logs_df['timestamp'])
        
        # Time series plot
        trends_fig = create_prediction_trends_plot(logs_df)
        if trends_fig:
            st.plotly_chart(trends_fig, use_container_width=True)
        
        # Risk distribution
        risk_dist_fig = create_risk_distribution_plot(logs_df)
        if risk_dist_fig:
            st.plotly_chart(risk_dist_fig, use_container_width=True)
    
    # Logs table
    st.markdown("#### 📋 Detailed Prediction Logs")
    
    # Display options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        show_features = st.checkbox("Show Patient Features", value=False)
    
    with col2:
        risk_filter = st.selectbox(
            "Filter by Risk Level:",
            ["All", "High Risk", "Low Risk"]
        )
    
    with col3:
        sort_by = st.selectbox(
            "Sort by:",
            ["Timestamp (Newest)", "Timestamp (Oldest)", "Risk Probability (High-Low)", "Risk Probability (Low-High)"]
        )
    
    # Apply filters and display
    filtered_logs = logs_df.copy()
    
    if risk_filter != "All":
        filtered_logs = filtered_logs[filtered_logs['risk_level'] == risk_filter]
    
    # Apply sorting
    if sort_by == "Timestamp (Newest)":
        filtered_logs = filtered_logs.sort_values('datetime', ascending=False)
    elif sort_by == "Timestamp (Oldest)":
        filtered_logs = filtered_logs.sort_values('datetime', ascending=True)
    elif sort_by == "Risk Probability (High-Low)":
        filtered_logs = filtered_logs.sort_values('probability', ascending=False)
    elif sort_by == "Risk Probability (Low-High)":
        filtered_logs = filtered_logs.sort_values('probability', ascending=True)
    
    # Select columns to display
    display_columns = ['timestamp', 'model_used', 'risk_level', 'probability']
    
    if show_features and 'features' in filtered_logs.columns:
        # Expand features into separate columns
        features_df = pd.json_normalize(filtered_logs['features'])
        display_df = pd.concat([filtered_logs[display_columns], features_df], axis=1)
    else:
        display_df = filtered_logs[display_columns]
    
    st.dataframe(display_df, use_container_width=True)


def model_info_page():
    """Model information and technical details page"""
    st.markdown("### ⚙️ Model Information & Technical Details")
    
    if not st.session_state.models_trained:
        st.warning("⚠️ Models not trained yet. Please train the models first.")
        if st.button("🚀 Train Models Now"):
            initialize_models()
        return
    
    # Model overview
    st.markdown("#### 🤖 Model Overview")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="info-box">
            <h4>🧠 Machine Learning Algorithms</h4>
            <p>This application implements 6 different machine learning algorithms for comprehensive cardiovascular risk assessment:</p>
            <ul>
                <li><strong>Random Forest:</strong> Ensemble method with multiple decision trees</li>
                <li><strong>K-Nearest Neighbors:</strong> Instance-based learning algorithm</li>
                <li><strong>Logistic Regression:</strong> Linear model with sigmoid activation</li>
                <li><strong>Support Vector Machine:</strong> Kernel-based classification</li>
                <li><strong>Gradient Boosting:</strong> Sequential ensemble method</li>
                <li><strong>Neural Network:</strong> Multi-layer perceptron</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Best model info
        best_model_name = max(st.session_state.model_metrics.items(), key=lambda x: x[1]['accuracy'])[0]
        best_accuracy = st.session_state.model_metrics[best_model_name]['accuracy']
        
        st.markdown(f"""
        <div class="metric-card">
            <h3>🏆 Best Model</h3>
            <h2>{best_model_name}</h2>
            <p>Accuracy: {best_accuracy:.1%}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Detailed model metrics
    st.markdown("#### 📊 Detailed Model Performance")
    
    metrics_df = pd.DataFrame(st.session_state.model_metrics).T
    metrics_df = metrics_df.round(4)
    st.dataframe(metrics_df, use_container_width=True)
    
    # Model comparison
    st.markdown("#### 🔍 Model Comparison Analysis")
    
    tab1, tab2, tab3 = st.tabs(["📈 Performance Metrics", "🎯 Confusion Matrices", "⚖️ Pros & Cons"])
    
    with tab1:
        # Performance radar chart
        radar_fig = create_radar_chart(st.session_state.model_metrics)
        st.plotly_chart(radar_fig, use_container_width=True)
    
    with tab2:
        # Show confusion matrices for all models
        model_names = list(st.session_state.model_metrics.keys())
        
        cols = st.columns(3)
        for i, model_name in enumerate(model_names):
            result = st.session_state.model_metrics[model_name]
            if result['predictions'] is not None:
                cm_fig = create_confusion_matrix_plot(
                    st.session_state.y_test,
                    result['predictions'],
                    model_name
                )
                with cols[i % 3]:
                    st.plotly_chart(cm_fig, use_container_width=True)
    
    with tab3:
        # Model pros and cons
        model_analysis = get_model_analysis()
        
        for model_name, analysis in model_analysis.items():
            if model_name in st.session_state.models:
                with st.expander(f"📋 {model_name} Analysis"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**✅ Advantages:**")
                        for pro in analysis['pros']:
                            st.markdown(f"• {pro}")
                    
                    with col2:
                        st.markdown("**⚠️ Limitations:**")
                        for con in analysis['cons']:
                            st.markdown(f"• {con}")


def about_page():
    """About page with project information"""
    st.markdown("### 📖 About This Application")
    
    st.markdown("""
    <div style="text-align: center; margin: 2rem 0;">
        <h2 style="color: #667eea;">Heart Disease Prediction App</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Project information
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="info-box">
            <h4>🔬 Research Background</h4>
            <p>This application is based on the academic research paper:</p>
            <p><strong>"Cardiovascular Disease Prediction using Machine Learning: A Comprehensive Analysis of the UCI Heart Disease Dataset"</strong></p>
            <p>The research achieved <strong>88.5% accuracy</strong> using Random Forest algorithm, representing a 5.2% improvement over previous benchmarks.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
            <h4>🎯 Key Features</h4>
            <ul>
                <li>🤖 <strong>6 Machine Learning Algorithms:</strong> Comprehensive comparison of different ML approaches</li>
                <li>📊 <strong>Real-time Predictions:</strong> Instant cardiovascular risk assessment</li>
                <li>📈 <strong>Advanced Analytics:</strong> Detailed visualizations and performance metrics</li>
                <li>📝 <strong>Prediction Logging:</strong> Track and analyze prediction history</li>
                <li>🔍 <strong>Feature Importance:</strong> Understand which factors matter most</li>
                <li>⚕️ <strong>Clinical Relevance:</strong> Results aligned with medical knowledge</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>👨‍🎓 Author</h3>
            <h4>Khaled Makkawirelang</h4>
            <p>Student Researcher<br>
            Universitas Trilogi<br>
            Program Studi Teknik Informatika<br>
            Jakarta, Indonesia</p>
            <p><strong>📧 Contact:</strong><br>
            makkawirelang@gmail.com</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
            <h4>🛠️ Technical Stack</h4>
            <ul>
                <li><strong>Frontend:</strong> Streamlit</li>
                <li><strong>ML Library:</strong> Scikit-learn</li>
                <li><strong>Data Analysis:</strong> Pandas, NumPy</li>
                <li><strong>Visualization:</strong> Plotly, Matplotlib</li>
                <li><strong>Language:</strong> Python 3.8+</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Model performance summary
    st.markdown("#### 🏆 Research Results Summary")
    
    if st.session_state.models_trained:
        results_summary = pd.DataFrame(st.session_state.model_metrics).T
        results_summary = results_summary.round(4)
        st.dataframe(results_summary, use_container_width=True)
        
        best_model = max(st.session_state.model_metrics.items(), key=lambda x: x[1]['accuracy'])
        
        st.success(f"""
        🎉 **Best Performance Achieved:**
        - **Algorithm:** {best_model[0]}
        - **Accuracy:** {best_model[1]['accuracy']:.1%}
        - **Precision:** {best_model[1]['precision']:.1%}
        - **Recall:** {best_model[1]['recall']:.1%}
        - **F1-Score:** {best_model[1]['f1_score']:.1%}
        - **AUC-ROC:** {best_model[1]['auc_roc']:.3f}
        """)
    else:
        st.info("🔄 Train the models to see performance results!")
    
    # Disclaimer
    st.markdown("#### ⚠️ Important Disclaimer")
    
    st.warning("""
    **MEDICAL DISCLAIMER:**
    
    This application is designed for educational and research purposes only. It should NOT be used as a substitute for professional medical advice, diagnosis, or treatment. 
    
    - Always consult with qualified healthcare professionals for medical decisions
    - This tool provides risk assessment, not definitive diagnosis
    - Results should be interpreted in conjunction with clinical examination and other diagnostic tests
    """)


# =============================================================================
# APPLICATION ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()