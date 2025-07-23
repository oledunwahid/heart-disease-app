"""
Visualization Utilities for Heart Disease Prediction App
=======================================================
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import roc_curve, confusion_matrix


def create_performance_comparison_chart(results):
    """
    Create a comprehensive performance comparison chart
    """
    # Prepare data for plotting
    models = list(results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
    
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=('Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC', 'Overall Comparison'),
        specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]]
    )
    
    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3']
    
    # Individual metric charts
    positions = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2)]
    
    for i, metric in enumerate(metrics):
        if i < 5:  # Only plot first 5 metrics individually
            row, col = positions[i]
            values = [results[model][metric] for model in models]
            
            fig.add_trace(
                go.Bar(
                    x=models,
                    y=values,
                    name=metric.replace('_', ' ').title(),
                    marker_color=colors[i],
                    showlegend=False,
                    text=[f'{v:.3f}' for v in values],
                    textposition='auto'
                ),
                row=row, col=col
            )
    
    # Overall comparison chart
    for i, metric in enumerate(metrics):
        values = [results[model][metric] for model in models]
        fig.add_trace(
            go.Bar(
                x=models,
                y=values,
                name=metric.replace('_', ' ').title(),
                marker_color=colors[i],
                legendgroup=metric,
                showlegend=True
            ),
            row=2, col=3
        )
    
    fig.update_layout(
        title_text="Model Performance Comparison",
        title_font_size=20,
        height=600,
        barmode='group'
    )
    
    return fig


def create_confusion_matrix_plot(y_test, y_pred, model_name):
    """
    Create an interactive confusion matrix plot
    """
    cm = confusion_matrix(y_test, y_pred)
    
    fig = px.imshow(
        cm,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Blues",
        title=f"Confusion Matrix - {model_name}"
    )
    
    fig.update_layout(
        xaxis_title="Predicted",
        yaxis_title="Actual",
        xaxis=dict(tickmode='array', tickvals=[0, 1], ticktext=['No Disease', 'Disease']),
        yaxis=dict(tickmode='array', tickvals=[0, 1], ticktext=['No Disease', 'Disease'])
    )
    
    return fig


def create_roc_curve_plot(results, y_test):
    """
    Create ROC curves for all models
    """
    fig = go.Figure()
    
    for model_name, result in results.items():
        if result['probabilities'] is not None:
            fpr, tpr, _ = roc_curve(y_test, result['probabilities'])
            auc_score = result['auc_roc']
            
            fig.add_trace(go.Scatter(
                x=fpr,
                y=tpr,
                mode='lines',
                name=f'{model_name} (AUC = {auc_score:.3f})',
                line=dict(width=2)
            ))
    
    # Add diagonal line
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(dash='dash', color='red')
    ))
    
    fig.update_layout(
        title='ROC Curves - Model Comparison',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        width=600,
        height=500
    )
    
    return fig


def create_feature_importance_plot(feature_importance_df, model_name):
    """
    Create feature importance plot
    """
    if feature_importance_df is None or len(feature_importance_df) == 0:
        return None
    
    # Take top 10 features
    top_features = feature_importance_df.head(10)
    
    fig = px.bar(
        top_features,
        x='Importance',
        y='Feature',
        orientation='h',
        title=f'Top 10 Feature Importance - {model_name}',
        color='Importance',
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        height=400
    )
    
    return fig


def create_dataset_overview_plots(df):
    """
    Create comprehensive dataset overview plots
    """
    # Create subplots
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=(
            'Age Distribution', 'Gender Distribution', 'Chest Pain Types',
            'Heart Disease Distribution', 'Cholesterol vs Age', 'Max Heart Rate vs Age'
        ),
        specs=[[{"type": "histogram"}, {"type": "pie"}, {"type": "bar"}],
               [{"type": "pie"}, {"type": "scatter"}, {"type": "scatter"}]]
    )
    
    # Age distribution
    fig.add_trace(
        go.Histogram(x=df['age'], nbinsx=20, name='Age', showlegend=False),
        row=1, col=1
    )
    
    # Gender distribution
    gender_counts = df['sex'].value_counts()
    fig.add_trace(
        go.Pie(
            values=gender_counts.values,
            labels=['Female', 'Male'],
            name='Gender',
            showlegend=False
        ),
        row=1, col=2
    )
    
    # Chest pain types
    cp_counts = df['cp'].value_counts().sort_index()
    cp_labels = ['Typical Angina', 'Atypical Angina', 'Non-Anginal', 'Asymptomatic']
    fig.add_trace(
        go.Bar(
            x=cp_labels[:len(cp_counts)],
            y=cp_counts.values,
            name='Chest Pain',
            showlegend=False
        ),
        row=1, col=3
    )
    
    # Heart disease distribution
    target_col = 'num' if 'num' in df.columns else df.columns[-1]
    disease_counts = df[target_col].value_counts()
    fig.add_trace(
        go.Pie(
            values=disease_counts.values,
            labels=['No Disease', 'Disease'],
            name='Heart Disease',
            showlegend=False
        ),
        row=2, col=1
    )
    
    # Cholesterol vs Age
    fig.add_trace(
        go.Scatter(
            x=df['age'],
            y=df['chol'],
            mode='markers',
            marker=dict(
                color=df[target_col],
                colorscale='RdYlBu',
                size=5,
                opacity=0.6
            ),
            name='Cholesterol vs Age',
            showlegend=False
        ),
        row=2, col=2
    )
    
    # Max Heart Rate vs Age
    fig.add_trace(
        go.Scatter(
            x=df['age'],
            y=df['thalach'],
            mode='markers',
            marker=dict(
                color=df[target_col],
                colorscale='RdYlBu',
                size=5,
                opacity=0.6
            ),
            name='Max HR vs Age',
            showlegend=False
        ),
        row=2, col=3
    )
    
    fig.update_layout(
        title_text="Dataset Overview and Analysis",
        height=700,
        showlegend=False
    )
    
    return fig


def create_radar_chart(model_metrics):
    """
    Create performance radar chart for models
    """
    metrics_for_radar = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
    
    fig = go.Figure()
    
    for model_name in model_metrics.keys():
        values = [model_metrics[model_name][metric] for metric in metrics_for_radar]
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=metrics_for_radar,
            fill='toself',
            name=model_name
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title="Model Performance Radar Chart"
    )
    
    return fig


def create_prediction_trends_plot(logs_df):
    """
    Create prediction trends visualization
    """
    if len(logs_df) <= 1:
        return None
        
    # Time series plot
    fig = px.scatter(
        logs_df,
        x='datetime',
        y='probability',
        color='risk_level',
        title='Prediction Risk Levels Over Time',
        labels={'probability': 'Risk Probability', 'datetime': 'Timestamp'}
    )
    
    return fig


def create_risk_distribution_plot(logs_df):
    """
    Create risk probability distribution plot
    """
    if len(logs_df) <= 1:
        return None
        
    fig = px.histogram(
        logs_df,
        x='probability',
        color='risk_level',
        title='Risk Probability Distribution',
        nbins=20
    )
    
    return fig