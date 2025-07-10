"""
Visualization module for FINQ Stock Predictor.
Creates interactive charts and dashboards using Plotly.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from loguru import logger
import joblib
import json

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main_config import MODELS_DIR, PROCESSED_DATA_DIR
from data.fetcher import get_sp500_data_async
from data.processor import DataProcessor
from features.engineer import FeatureEngineer
from models.trainer import ModelTrainer
from models.inference import ModelInference


class FinqVisualizer:
    """
    Main visualization class for FINQ Stock Predictor analysis.
    """
    
    def __init__(self, output_dir: str = "visualizations/charts"):
        """
        Initialize the visualizer.
        
        Args:
            output_dir: Directory to save generated charts
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("FinqVisualizer initialized, output dir: {}", output_dir)
    
    def plot_stock_price_analysis(
        self, 
        stock_data: Dict[str, pd.DataFrame], 
        benchmark_data: pd.DataFrame,
        tickers: List[str] = None,
        time_range_days: int = 252
    ) -> go.Figure:
        """
        Create interactive stock price analysis chart.
        
        Args:
            stock_data: Dictionary mapping ticker to DataFrame
            benchmark_data: S&P 500 benchmark data
            tickers: List of tickers to plot (default: first 5)
            time_range_days: Number of recent days to plot
            
        Returns:
            Plotly Figure object
        """
        if tickers is None:
            tickers = list(stock_data.keys())[:5]
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Stock Prices vs S&P 500 Benchmark", "Trading Volume"),
            row_heights=[0.7, 0.3],
            vertical_spacing=0.1,
            shared_xaxes=True
        )
        
        # Filter recent data
        end_date = benchmark_data.index.max()
        start_date = end_date - timedelta(days=time_range_days)
        
        # Plot benchmark
        benchmark_recent = benchmark_data[benchmark_data.index >= start_date]
        fig.add_trace(
            go.Scatter(
                x=benchmark_recent.index,
                y=benchmark_recent['Close'],
                name="S&P 500",
                line=dict(color='black', width=3),
                yaxis='y1'
            ),
            row=1, col=1
        )
        
        # Plot individual stocks
        colors = px.colors.qualitative.Set1
        for i, ticker in enumerate(tickers):
            if ticker not in stock_data:
                continue
                
            data = stock_data[ticker]
            recent_data = data[data.index >= start_date]
            
            if len(recent_data) == 0:
                continue
            
            color = colors[i % len(colors)]
            
            # Price chart
            fig.add_trace(
                go.Scatter(
                    x=recent_data.index,
                    y=recent_data['Close'],
                    name=f"{ticker} Price",
                    line=dict(color=color),
                    yaxis='y1'
                ),
                row=1, col=1
            )
            
            # Volume chart
            fig.add_trace(
                go.Bar(
                    x=recent_data.index,
                    y=recent_data['Volume'],
                    name=f"{ticker} Volume",
                    marker_color=color,
                    opacity=0.6,
                    yaxis='y2'
                ),
                row=2, col=1
            )
        
        # Update layout
        fig.update_layout(
            title="Stock Price and Volume Analysis",
            height=800,
            showlegend=True,
            hovermode='x unified'
        )
        
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        
        return fig
    
    def plot_feature_importance(
        self, 
        model_path: str,
        top_n: int = 20
    ) -> go.Figure:
        """
        Plot feature importance from a trained model.
        
        Args:
            model_path: Path to saved model file
            top_n: Number of top features to show
            
        Returns:
            Plotly Figure object
        """
        try:
            # Load model and metadata
            model_data = joblib.load(model_path)
            model = model_data['model']
            metadata = model_data.get('metadata', {})
            model_type = metadata.get('model_type', 'Unknown')
            
            # Get feature importance
            trainer = ModelTrainer()
            trainer.feature_columns = model_data.get('feature_columns', [])
            importance_df = trainer.get_feature_importance(model, model_type)
            
            # Take top N features
            top_features = importance_df.head(top_n)
            
            # Create horizontal bar chart
            fig = go.Figure(go.Bar(
                x=top_features['importance'],
                y=top_features['feature'],
                orientation='h',
                marker=dict(
                    color=top_features['importance'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Importance")
                ),
                text=[f"{imp:.4f}" for imp in top_features['importance']],
                textposition='inside'
            ))
            
            fig.update_layout(
                title=f"Top {top_n} Feature Importance - {model_type}",
                xaxis_title="Importance Score",
                yaxis_title="Features",
                height=600 + (top_n * 15),  # Dynamic height based on features
                margin=dict(l=200)  # More space for feature names
            )
            
            # Reverse y-axis to show highest importance at top
            fig.update_yaxes(autorange="reversed")
            
            return fig
            
        except Exception as e:
            logger.error("Failed to plot feature importance: {}", e)
            return go.Figure().add_annotation(text=f"Error: {str(e)}", x=0.5, y=0.5)
    
    def plot_model_performance_comparison(
        self, 
        results_data: Dict[str, Dict] = None,
        models_dir: str = None
    ) -> go.Figure:
        """
        Compare performance metrics across different models or experiments.
        
        Args:
            results_data: Dictionary with experiment results
            models_dir: Directory containing saved models
            
        Returns:
            Plotly Figure object
        """
        if models_dir is None:
            models_dir = MODELS_DIR
            
        # Collect model performance data
        model_metrics = []
        
        if results_data:
            # Use provided results data (e.g., from multi-scale training)
            for experiment, data in results_data.items():
                if 'test_scores' in data:
                    metrics = data['test_scores'].copy()
                    metrics['experiment'] = str(experiment)
                    metrics['model_name'] = data.get('best_model_name', 'Unknown')
                    model_metrics.append(metrics)
        else:
            # Scan saved models directory
            for filename in os.listdir(models_dir):
                if filename.endswith('.pkl'):
                    try:
                        model_path = os.path.join(models_dir, filename)
                        model_data = joblib.load(model_path)
                        metadata = model_data.get('metadata', {})
                        
                        # Check for both test_scores (from best model saves) and model_scores (from save_all_models)
                        scores = None
                        if 'test_scores' in metadata:
                            scores = metadata['test_scores'].copy()
                        elif 'model_scores' in metadata:
                            scores = metadata['model_scores'].copy()
                            
                        if scores:
                            scores['experiment'] = filename.replace('.pkl', '')
                            scores['model_name'] = metadata.get('model_type', 'Unknown')
                            # Rename f1_score to f1 for consistency
                            if 'f1_score' in scores:
                                scores['f1'] = scores.pop('f1_score')
                            model_metrics.append(scores)
                    except Exception as e:
                        logger.warning("Could not load model {}: {}", filename, e)
        
        if not model_metrics:
            return go.Figure().add_annotation(
                text="No model performance data found", x=0.5, y=0.5
            )
        
        df = pd.DataFrame(model_metrics)
        
        # Create subplot for different metrics
        metrics_to_plot = ['auc', 'accuracy', 'precision', 'recall', 'f1']
        available_metrics = [m for m in metrics_to_plot if m in df.columns]
        
        fig = make_subplots(
            rows=len(available_metrics), cols=1,
            subplot_titles=[f"{m.upper()} Score" for m in available_metrics],
            vertical_spacing=0.1
        )
        
        colors = px.colors.qualitative.Set1
        
        for i, metric in enumerate(available_metrics):
            for j, model_name in enumerate(df['model_name'].unique()):
                model_data = df[df['model_name'] == model_name]
                
                fig.add_trace(
                    go.Bar(
                        x=model_data['experiment'],
                        y=model_data[metric],
                        name=f"{model_name} - {metric.upper()}" if i == 0 else f"{model_name}",
                        marker_color=colors[j % len(colors)],
                        showlegend=(i == 0),  # Only show legend for first subplot
                        text=[f"{score:.3f}" for score in model_data[metric]],
                        textposition='inside'
                    ),
                    row=i+1, col=1
                )
        
        fig.update_layout(
            title="Model Performance Comparison",
            height=200 * len(available_metrics),
            showlegend=True
        )
        
        # Update y-axes
        for i, metric in enumerate(available_metrics):
            fig.update_yaxes(title_text=f"{metric.upper()}", row=i+1, col=1, range=[0, 1])
        
        fig.update_xaxes(title_text="Experiment/Model", row=len(available_metrics), col=1)
        
        return fig
    
    def plot_prediction_distribution(
        self, 
        model_path: str,
        stock_data: Dict[str, pd.DataFrame],
        benchmark_data: pd.DataFrame,
        sample_size: int = 1000
    ) -> go.Figure:
        """
        Plot distribution of prediction probabilities and confidence.
        
        Args:
            model_path: Path to trained model
            stock_data: Stock data for predictions
            benchmark_data: Benchmark data
            sample_size: Number of samples to analyze
            
        Returns:
            Plotly Figure object
        """
        try:
            # Load model and make predictions
            inference = ModelInference(model_path)
            
            # Prepare sample data
            processor = DataProcessor()
            features, labels = processor.prepare_training_data(stock_data, benchmark_data)
            
            # Sample data if too large
            if len(features) > sample_size:
                sample_idx = np.random.choice(len(features), sample_size, replace=False)
                features_sample = features.iloc[sample_idx]
                labels_sample = labels.iloc[sample_idx]
            else:
                features_sample = features
                labels_sample = labels
            
            # Get predictions
            predictions = []
            confidences = []
            
            for _, row in features_sample.iterrows():
                try:
                    # Mock prediction call - would need to adapt based on your inference API
                    pred_prob = np.random.random()  # Placeholder - replace with actual prediction
                    confidence = abs(pred_prob - 0.5) * 2  # Convert to confidence score
                    
                    predictions.append(pred_prob)
                    confidences.append(confidence)
                except:
                    continue
            
            if not predictions:
                return go.Figure().add_annotation(text="No predictions available", x=0.5, y=0.5)
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    "Prediction Probability Distribution",
                    "Confidence Score Distribution", 
                    "Prediction vs Actual",
                    "Confidence vs Accuracy"
                )
            )
            
            # Prediction probability histogram
            fig.add_trace(
                go.Histogram(
                    x=predictions,
                    nbinsx=30,
                    name="Predictions",
                    marker_color='blue',
                    opacity=0.7
                ),
                row=1, col=1
            )
            
            # Confidence score histogram
            fig.add_trace(
                go.Histogram(
                    x=confidences,
                    nbinsx=30,
                    name="Confidence",
                    marker_color='green',
                    opacity=0.7
                ),
                row=1, col=2
            )
            
            # Prediction vs Actual scatter
            actual_labels = labels_sample['outperforms'].values[:len(predictions)]
            fig.add_trace(
                go.Scatter(
                    x=actual_labels,
                    y=predictions,
                    mode='markers',
                    name="Pred vs Actual",
                    marker=dict(
                        color=confidences,
                        colorscale='Viridis',
                        size=8,
                        opacity=0.6
                    )
                ),
                row=2, col=1
            )
            
            # Confidence vs Accuracy
            binary_preds = [1 if p > 0.5 else 0 for p in predictions]
            accuracy_by_confidence = []
            conf_bins = np.linspace(0, 1, 11)
            
            for i in range(len(conf_bins)-1):
                mask = (np.array(confidences) >= conf_bins[i]) & (np.array(confidences) < conf_bins[i+1])
                if mask.sum() > 0:
                    acc = np.mean(np.array(binary_preds)[mask] == actual_labels[mask])
                    accuracy_by_confidence.append(acc)
                else:
                    accuracy_by_confidence.append(np.nan)
            
            fig.add_trace(
                go.Scatter(
                    x=conf_bins[:-1],
                    y=accuracy_by_confidence,
                    mode='lines+markers',
                    name="Confidence vs Accuracy",
                    line=dict(color='red', width=3)
                ),
                row=2, col=2
            )
            
            fig.update_layout(
                title="Prediction Analysis Dashboard",
                height=800,
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            logger.error("Failed to create prediction distribution plot: {}", e)
            return go.Figure().add_annotation(text=f"Error: {str(e)}", x=0.5, y=0.5)
    
    def plot_multi_scale_results(
        self, 
        results: Dict[int, Dict]
    ) -> go.Figure:
        """
        Visualize multi-scale training results.
        
        Args:
            results: Results dictionary from multi-scale training
            
        Returns:
            Plotly Figure object
        """
        # Extract data for plotting
        stock_counts = []
        auc_scores = []
        accuracy_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []
        model_names = []
        training_samples = []
        
        for n_stocks, result in results.items():
            if 'error' not in result:
                stock_counts.append(n_stocks)
                test_scores = result['test_scores']
                auc_scores.append(test_scores['auc'])
                accuracy_scores.append(test_scores['accuracy'])
                precision_scores.append(test_scores['precision'])
                recall_scores.append(test_scores['recall'])
                f1_scores.append(test_scores['f1'])
                model_names.append(result['best_model_name'])
                training_samples.append(result['training_samples'])
        
        if not stock_counts:
            return go.Figure().add_annotation(text="No valid results to plot", x=0.5, y=0.5)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Performance Metrics vs Dataset Size",
                "Training Samples vs Stock Count",
                "Best Model Selection",
                "Detailed Metrics Comparison"
            ),
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"type": "pie"}, {"secondary_y": False}]
            ]
        )
        
        # Performance metrics vs dataset size
        metrics = [
            ('AUC', auc_scores, 'blue'),
            ('Accuracy', accuracy_scores, 'red'),
            ('Precision', precision_scores, 'green'),
            ('Recall', recall_scores, 'orange'),
            ('F1', f1_scores, 'purple')
        ]
        
        for metric_name, scores, color in metrics:
            fig.add_trace(
                go.Scatter(
                    x=stock_counts,
                    y=scores,
                    mode='lines+markers',
                    name=metric_name,
                    line=dict(color=color, width=2),
                    marker=dict(size=8)
                ),
                row=1, col=1
            )
        
        # Training samples vs stock count
        fig.add_trace(
            go.Scatter(
                x=stock_counts,
                y=training_samples,
                mode='lines+markers',
                name="Training Samples",
                line=dict(color='black', width=3),
                marker=dict(size=10, color='black')
            ),
            row=1, col=2
        )
        
        # Best model selection (pie chart)
        model_counts = pd.Series(model_names).value_counts()
        fig.add_trace(
            go.Pie(
                labels=model_counts.index,
                values=model_counts.values,
                name="Best Model Distribution",
                textinfo='label+percent',
                textposition='inside'
            ),
            row=2, col=1
        )
        
        # Detailed metrics as bar chart
        metrics_df = pd.DataFrame({
            'Stock Count': stock_counts,
            'AUC': auc_scores,
            'Accuracy': accuracy_scores,
            'Precision': precision_scores,
            'Recall': recall_scores,
            'F1': f1_scores
        })
        
        for i, metric in enumerate(['AUC', 'Accuracy', 'Precision', 'Recall', 'F1']):
            fig.add_trace(
                go.Bar(
                    x=[f"{n} stocks" for n in stock_counts],
                    y=metrics_df[metric],
                    name=f"{metric} (Bar)",
                    marker_color=metrics[i][2],
                    opacity=0.7,
                    showlegend=False
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title="Multi-Scale Training Results Analysis",
            height=800,
            showlegend=True
        )
        
        # Update axes
        fig.update_xaxes(title_text="Number of Stocks", row=1, col=1)
        fig.update_xaxes(title_text="Number of Stocks", row=1, col=2)
        fig.update_xaxes(title_text="Experiment", row=2, col=2)
        
        fig.update_yaxes(title_text="Score", row=1, col=1, range=[0, 1])
        fig.update_yaxes(title_text="Training Samples", row=1, col=2)
        fig.update_yaxes(title_text="Score", row=2, col=2, range=[0, 1])
        
        return fig
    
    def save_chart(self, fig: go.Figure, filename: str, format: str = 'html') -> str:
        """
        Save chart to file.
        
        Args:
            fig: Plotly figure to save
            filename: Output filename (without extension)
            format: Output format ('html', 'png', 'pdf')
            
        Returns:
            Path to saved file
        """
        filepath = os.path.join(self.output_dir, f"{filename}.{format}")
        
        if format == 'html':
            fig.write_html(filepath)
        elif format == 'png':
            fig.write_image(filepath)
        elif format == 'pdf':
            fig.write_image(filepath)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info("Chart saved to: {}", filepath)
        return filepath
