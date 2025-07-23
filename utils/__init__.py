"""
Heart Disease Prediction App - Utilities Package
===============================================
"""

from .data_loader import (
    load_heart_disease_dataset,
    generate_sample_data,
    preprocess_data,
    get_feature_info
)

from .model_trainer import (
    train_all_models,
    get_feature_importance,
    make_prediction,
    get_clinical_interpretations,
    get_model_analysis
)

from .visualizations import (
    create_performance_comparison_chart,
    create_confusion_matrix_plot,
    create_roc_curve_plot,
    create_feature_importance_plot,
    create_dataset_overview_plots,
    create_radar_chart,
    create_prediction_trends_plot,
    create_risk_distribution_plot
)

__all__ = [
    # Data loading
    'load_heart_disease_dataset',
    'generate_sample_data',
    'preprocess_data',
    'get_feature_info',
    
    # Model training
    'train_all_models',
    'get_feature_importance',
    'make_prediction',
    'get_clinical_interpretations',
    'get_model_analysis',
    
    # Visualizations
    'create_performance_comparison_chart',
    'create_confusion_matrix_plot',
    'create_roc_curve_plot',
    'create_feature_importance_plot',
    'create_dataset_overview_plots',
    'create_radar_chart',
    'create_prediction_trends_plot',
    'create_risk_distribution_plot'
]