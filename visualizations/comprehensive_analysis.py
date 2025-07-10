#!/usr/bin/env python3
"""
Comprehensive visualization analysis for FINQ Stock Predictor.
Analyzes data, models, and experiment results using Plotly.
"""

import asyncio
import os
import sys
import glob
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
from loguru import logger

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("❌ Plotly not available. Install with: pip install plotly kaleido")
    sys.exit(1)

from data.fetcher import get_sp500_data_async
from main_config import MODELS_DIR


async def fetch_sample_data(max_stocks=20):
    """Fetch sample stock data for analysis."""
    print(f"📊 Fetching data for {max_stocks} stocks...")
    stock_data, benchmark_data = await get_sp500_data_async(max_stocks=max_stocks)
    print(f"✅ Fetched data for {len(stock_data)} stocks")
    return stock_data, benchmark_data


def analyze_saved_models():
    """Analyze all saved models and extract performance metrics."""
    print("\n🔍 Analyzing saved models...")
    
    model_files = glob.glob(os.path.join("models/saved", "*.pkl"))
    if not model_files:
        print("⚠️  No saved models found")
        return None
    
    model_data = []
    
    for model_file in model_files:
        try:
            # Load model metadata
            model_info = joblib.load(model_file)
            metadata = model_info.get('metadata', {})
            
            # Extract info from filename
            filename = os.path.basename(model_file)
            parts = filename.replace('.pkl', '').split('_')
            
            # Handle different filename patterns (e.g., svm_10stocks_timestamp or model_type_stocks_timestamp)
            if len(parts) >= 3:
                # Find the part with 'stocks' 
                stocks_part = None
                model_parts = []
                for i, part in enumerate(parts):
                    if 'stocks' in part:
                        stocks_part = part
                        model_parts = parts[:i]
                        break
                
                model_type = '_'.join(model_parts) if model_parts else parts[0]
                stocks_count = int(stocks_part.replace('stocks', '')) if stocks_part and 'stocks' in stocks_part else 0
            else:
                model_type = parts[0] if parts else 'unknown'
                stocks_count = 0
            
            # Get test scores if available
            test_scores = metadata.get('model_scores', {})
            
            model_data.append({
                'file': filename,
                'model_type': model_type,
                'stocks_count': stocks_count,
                'auc': test_scores.get('auc', 0),
                'accuracy': test_scores.get('accuracy', 0),
                'precision': test_scores.get('precision', 0),
                'recall': test_scores.get('recall', 0),
                'training_samples': metadata.get('training_samples', 0),
                'test_samples': metadata.get('test_samples', 0),
                'features_count': metadata.get('features_count', 0),
                'training_date': metadata.get('training_date', '')
            })
            
        except Exception as e:
            print(f"⚠️  Could not analyze {model_file}: {e}")
            continue
    
    if model_data:
        df = pd.DataFrame(model_data)
        print(f"✅ Analyzed {len(df)} models")
        return df
    else:
        print("❌ No model data could be extracted")
        return None


def create_stock_price_visualization(stock_data, benchmark_data, save_path="visualizations/charts"):
    """Create interactive stock price analysis charts."""
    print("\n📈 Creating stock price analysis...")
    
    os.makedirs(save_path, exist_ok=True)
    
    # Select top 8 stocks for visualization
    tickers = list(stock_data.keys())[:8]
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Stock Prices vs Benchmark (S&P 500)', 'Volume Analysis'),
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3]
    )
    
    # Add benchmark
    fig.add_trace(
        go.Scatter(
            x=benchmark_data.index,
            y=benchmark_data['Close'],
            name='S&P 500 (Benchmark)',
            line=dict(color='black', width=2),
            hovertemplate='<b>S&P 500</b><br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Color palette
    colors = px.colors.qualitative.Set2
    
    # Add stock prices
    for i, ticker in enumerate(tickers):
        if ticker in stock_data:
            data = stock_data[ticker]
            color = colors[i % len(colors)]
            
            # Price chart
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['Close'],
                    name=ticker,
                    line=dict(color=color),
                    hovertemplate=f'<b>{ticker}</b><br>Date: %{{x}}<br>Price: $%{{y:.2f}}<extra></extra>'
                ),
                row=1, col=1
            )
            
            # Volume chart
            fig.add_trace(
                go.Bar(
                    x=data.index,
                    y=data['Volume'],
                    name=f'{ticker} Volume',
                    marker_color=color,
                    opacity=0.6,
                    showlegend=False,
                    hovertemplate=f'<b>{ticker} Volume</b><br>Date: %{{x}}<br>Volume: %{{y:,.0f}}<extra></extra>'
                ),
                row=2, col=1
            )
    
    # Update layout
    fig.update_layout(
        title={
            'text': 'FINQ Stock Predictor - Stock Price & Volume Analysis',
            'x': 0.5,
            'font': {'size': 20}
        },
        height=800,
        showlegend=True,
        hovermode='x unified'
    )
    
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    # Save chart
    file_path = os.path.join(save_path, "stock_price_analysis.html")
    fig.write_html(file_path)
    print(f"✅ Saved: {file_path}")
    
    return fig


def create_model_performance_visualization(model_df, save_path="visualizations/charts"):
    """Create model performance comparison charts."""
    print("\n📊 Creating model performance analysis...")
    
    if model_df is None or model_df.empty:
        print("⚠️  No model data available for visualization")
        return None
    
    os.makedirs(save_path, exist_ok=True)
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('AUC Performance by Stock Count', 'Accuracy by Model Type',
                       'Model Performance Comparison', 'Training vs Test Samples'),
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    # Color mapping for model types
    model_types = model_df['model_type'].unique()
    colors = px.colors.qualitative.Set1[:len(model_types)]
    color_map = dict(zip(model_types, colors))
    
    # 1. AUC vs Stock Count (line chart)
    for model_type in model_types:
        subset = model_df[model_df['model_type'] == model_type].sort_values('stocks_count')
        fig.add_trace(
            go.Scatter(
                x=subset['stocks_count'],
                y=subset['auc'],
                mode='lines+markers',
                name=f'{model_type} AUC',
                line=dict(color=color_map[model_type]),
                hovertemplate=f'<b>{model_type}</b><br>Stocks: %{{x}}<br>AUC: %{{y:.4f}}<extra></extra>'
            ),
            row=1, col=1
        )
    
    # 2. Accuracy by Model Type (box plot)
    for model_type in model_types:
        subset = model_df[model_df['model_type'] == model_type]
        fig.add_trace(
            go.Box(
                y=subset['accuracy'],
                name=f'{model_type}',
                marker_color=color_map[model_type],
                showlegend=False
            ),
            row=1, col=2
        )
    
    # 3. Performance heatmap-style chart
    metrics = ['auc', 'accuracy', 'precision', 'recall']
    avg_performance = model_df.groupby('model_type')[metrics].mean()
    
    fig.add_trace(
        go.Heatmap(
            z=avg_performance.values,
            x=metrics,
            y=avg_performance.index,
            colorscale='RdYlBu_r',
            hovertemplate='Model: %{y}<br>Metric: %{x}<br>Score: %{z:.4f}<extra></extra>',
            showscale=True
        ),
        row=2, col=1
    )
    
    # 4. Training vs Test samples
    fig.add_trace(
        go.Scatter(
            x=model_df['training_samples'],
            y=model_df['test_samples'],
            mode='markers',
            marker=dict(
                size=10,
                color=model_df['auc'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="AUC Score")
            ),
            text=model_df['model_type'],
            hovertemplate='Training: %{x}<br>Test: %{y}<br>Model: %{text}<br>AUC: %{marker.color:.4f}<extra></extra>',
            showlegend=False
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        title={
            'text': 'FINQ Stock Predictor - Model Performance Analysis',
            'x': 0.5,
            'font': {'size': 20}
        },
        height=800,
        showlegend=True
    )
    
    # Update axes
    fig.update_xaxes(title_text="Number of Stocks", row=1, col=1)
    fig.update_yaxes(title_text="AUC Score", row=1, col=1)
    fig.update_yaxes(title_text="Accuracy", row=1, col=2)
    fig.update_xaxes(title_text="Training Samples", row=2, col=2)
    fig.update_yaxes(title_text="Test Samples", row=2, col=2)
    
    # Save chart
    file_path = os.path.join(save_path, "model_performance_analysis.html")
    fig.write_html(file_path)
    print(f"✅ Saved: {file_path}")
    
    return fig


def create_multi_scale_analysis(model_df, save_path="visualizations/charts"):
    """Create multi-scale training analysis visualization."""
    print("\n📏 Creating multi-scale training analysis...")
    
    if model_df is None or model_df.empty:
        print("⚠️  No model data available for multi-scale analysis")
        return None
    
    os.makedirs(save_path, exist_ok=True)
    
    # Create comprehensive multi-scale analysis
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Scalability: Performance vs Dataset Size', 'Model Efficiency: Samples vs Features',
                       'Performance Distribution by Model Type', 'Best Performing Configurations'),
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    # Color mapping
    model_types = model_df['model_type'].unique()
    colors = px.colors.qualitative.Set2[:len(model_types)]
    color_map = dict(zip(model_types, colors))
    
    # 1. Scalability analysis
    for model_type in model_types:
        subset = model_df[model_df['model_type'] == model_type].sort_values('stocks_count')
        if len(subset) > 0:
            fig.add_trace(
                go.Scatter(
                    x=subset['stocks_count'],
                    y=subset['auc'],
                    mode='lines+markers',
                    name=f'{model_type}',
                    line=dict(color=color_map[model_type], width=3),
                    marker=dict(size=8),
                    hovertemplate=f'<b>{model_type}</b><br>Stocks: %{{x}}<br>AUC: %{{y:.4f}}<extra></extra>'
                ),
                row=1, col=1
            )
    
    # 2. Efficiency analysis (samples vs features)
    for model_type in model_types:
        subset = model_df[model_df['model_type'] == model_type]
        if len(subset) > 0:
            fig.add_trace(
                go.Scatter(
                    x=subset['training_samples'],
                    y=subset['features_count'],
                    mode='markers',
                    marker=dict(
                        size=subset['auc'] * 20,  # Size based on AUC
                        color=color_map[model_type],
                        opacity=0.7,
                        line=dict(width=1, color='white')
                    ),
                    name=f'{model_type} (size=AUC)',
                    showlegend=False,
                    hovertemplate=f'<b>{model_type}</b><br>Training Samples: %{{x}}<br>Features: %{{y}}<br>AUC: %{{marker.size:.4f}}<extra></extra>'
                ),
                row=1, col=2
            )
    
    # 3. Performance distribution by model type
    for model_type in model_types:
        subset = model_df[model_df['model_type'] == model_type]
        if len(subset) > 0:
            fig.add_trace(
                go.Violin(
                    y=subset['auc'],
                    name=model_type,
                    box_visible=True,
                    meanline_visible=True,
                    marker_color=color_map[model_type],
                    showlegend=False
                ),
                row=2, col=1
            )
    
    # 4. Best configurations
    best_configs = model_df.loc[model_df.groupby('stocks_count')['auc'].idxmax()]
    
    fig.add_trace(
        go.Bar(
            x=best_configs['stocks_count'],
            y=best_configs['auc'],
            text=best_configs['model_type'],
            textposition='inside',
            marker=dict(
                color=best_configs['auc'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="AUC Score", x=1.02)
            ),
            hovertemplate='Stocks: %{x}<br>Best AUC: %{y:.4f}<br>Model: %{text}<extra></extra>',
            showlegend=False
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        title={
            'text': 'FINQ Stock Predictor - Multi-Scale Training Analysis',
            'x': 0.5,
            'font': {'size': 20}
        },
        height=800,
        showlegend=True
    )
    
    # Update axes
    fig.update_xaxes(title_text="Number of Stocks", row=1, col=1)
    fig.update_yaxes(title_text="AUC Score", row=1, col=1)
    fig.update_xaxes(title_text="Training Samples", row=1, col=2)
    fig.update_yaxes(title_text="Number of Features", row=1, col=2)
    fig.update_yaxes(title_text="AUC Distribution", row=2, col=1)
    fig.update_xaxes(title_text="Number of Stocks", row=2, col=2)
    fig.update_yaxes(title_text="Best AUC Score", row=2, col=2)
    
    # Save chart
    file_path = os.path.join(save_path, "multi_scale_analysis.html")
    fig.write_html(file_path)
    print(f"✅ Saved: {file_path}")
    
    return fig


def create_dashboard_index(charts_dir="visualizations/charts"):
    """Create an HTML dashboard index."""
    print("\n🌐 Creating interactive dashboard...")
    
    html_files = glob.glob(os.path.join(charts_dir, "*.html"))
    
    if not html_files:
        print("⚠️  No charts found to include in dashboard")
        return
    
    chart_info = {
        "stock_price_analysis.html": {
            "title": "📈 Stock Price Analysis",
            "description": "Interactive price charts and volume analysis for selected stocks vs S&P 500 benchmark"
        },
        "model_performance_analysis.html": {
            "title": "🎯 Model Performance Analysis",
            "description": "Comprehensive analysis of model performance across different metrics and configurations"
        },
        "multi_scale_analysis.html": {
            "title": "📏 Multi-Scale Training Analysis",
            "description": "Scalability analysis showing how models perform with different dataset sizes"
        }
    }
    
    # Generate chart cards
    chart_cards = ""
    for html_file in sorted(html_files):
        filename = os.path.basename(html_file)
        info = chart_info.get(filename, {
            "title": f"📊 {filename.replace('.html', '').replace('_', ' ').title()}",
            "description": "Interactive visualization and analysis"
        })
        
        chart_cards += f'''
        <div class="chart-card">
            <h3>{info["title"]}</h3>
            <p>{info["description"]}</p>
            <a href="{filename}" target="_blank">Open Chart →</a>
        </div>'''
    
    # Create complete HTML
    dashboard_html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FINQ Stock Predictor - Analytics Dashboard</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f8f9fa; line-height: 1.6; }}
        .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 40px 20px; border-radius: 12px; margin-bottom: 30px; text-align: center; }}
        .header h1 {{ font-size: 2.5em; margin-bottom: 10px; }}
        .header p {{ font-size: 1.1em; opacity: 0.9; }}
        .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 30px 0; }}
        .stat-card {{ background: white; padding: 25px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center; }}
        .stat-number {{ font-size: 2.5em; font-weight: bold; color: #667eea; margin-bottom: 5px; }}
        .stat-label {{ color: #6c757d; font-weight: 500; }}
        .charts-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); gap: 25px; }}
        .chart-card {{ background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); transition: transform 0.3s ease, box-shadow 0.3s ease; }}
        .chart-card:hover {{ transform: translateY(-5px); box-shadow: 0 8px 25px rgba(0,0,0,0.15); }}
        .chart-card h3 {{ color: #2c3e50; margin-bottom: 15px; font-size: 1.4em; }}
        .chart-card p {{ color: #6c757d; margin-bottom: 20px; }}
        .chart-card a {{ display: inline-block; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 12px 24px; text-decoration: none; border-radius: 6px; font-weight: 500; transition: transform 0.2s ease; }}
        .chart-card a:hover {{ transform: scale(1.05); }}
        .footer {{ margin-top: 50px; text-align: center; color: #6c757d; padding: 30px; background: white; border-radius: 12px; }}
        .feature-highlight {{ background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; padding: 20px; border-radius: 10px; margin: 20px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 FINQ Stock Predictor</h1>
            <h2>Interactive Analytics Dashboard</h2>
            <p>Comprehensive visualization and analysis of ML experiments, data patterns, and model performance</p>
        </div>
        
        <div class="stats">
            <div class="stat-card">
                <div class="stat-number">{len(html_files)}</div>
                <div class="stat-label">Interactive Charts</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">Multi-Scale</div>
                <div class="stat-label">Training Analysis</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">Real-Time</div>
                <div class="stat-label">Data Insights</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">ML</div>
                <div class="stat-label">Model Performance</div>
            </div>
        </div>
        
        <div class="feature-highlight">
            <h3>🎯 Key Features</h3>
            <p>• Interactive Plotly charts with zoom, pan, and hover capabilities<br>
               • Multi-scale training analysis across different dataset sizes<br>
               • Comprehensive model performance comparisons<br>
               • Real-time stock price and volume analysis</p>
        </div>
        
        <div class="charts-grid">{chart_cards}
        </div>
        
        <div class="footer">
            <h3>📚 How to Use This Dashboard</h3>
            <p>Click on any chart above to open it in a new tab. All charts are fully interactive:</p>
            <ul style="text-align: left; max-width: 600px; margin: 15px auto;">
                <li><strong>Zoom:</strong> Click and drag to zoom into specific time periods or data ranges</li>
                <li><strong>Pan:</strong> Hold shift and drag to pan around the chart</li>
                <li><strong>Hover:</strong> Hover over data points for detailed information</li>
                <li><strong>Toggle:</strong> Click legend items to show/hide data series</li>
                <li><strong>Download:</strong> Use the toolbar to download charts as images</li>
            </ul>
            <p style="margin-top: 20px;"><em>Generated by FINQ Stock Predictor Analytics Engine</em></p>
        </div>
    </div>
</body>
</html>'''
    
    # Save dashboard
    dashboard_path = os.path.join(charts_dir, "index.html")
    with open(dashboard_path, 'w', encoding='utf-8') as f:
        f.write(dashboard_html)
    
    print(f"✅ Dashboard created: {dashboard_path}")
    return dashboard_path


async def main():
    """Main analysis function."""
    print("🎨 FINQ Stock Predictor - Comprehensive Visualization Analysis")
    print("=" * 70)
    
    try:
        # 1. Fetch sample data
        stock_data, benchmark_data = await fetch_sample_data(max_stocks=15)
        
        # 2. Analyze saved models
        model_df = analyze_saved_models()
        
        # 3. Create visualizations
        charts_dir = "visualizations/charts"
        os.makedirs(charts_dir, exist_ok=True)
        
        # Stock price analysis
        create_stock_price_visualization(stock_data, benchmark_data, charts_dir)
        
        # Model performance analysis
        if model_df is not None:
            create_model_performance_visualization(model_df, charts_dir)
            create_multi_scale_analysis(model_df, charts_dir)
        
        # 4. Create dashboard
        dashboard_path = create_dashboard_index(charts_dir)
        
        # 5. Summary
        print("\n🎉 Analysis Complete!")
        print("=" * 50)
        print(f"📁 Charts directory: {os.path.abspath(charts_dir)}")
        if dashboard_path:
            abs_dashboard = os.path.abspath(dashboard_path)
            print(f"🌐 Dashboard: file://{abs_dashboard}")
            print(f"📱 Open the dashboard in your browser to explore interactive charts!")
        
        if model_df is not None:
            print(f"\n📊 Model Analysis Summary:")
            print(f"   • Analyzed {len(model_df)} trained models")
            print(f"   • Model types: {', '.join(model_df['model_type'].unique())}")
            print(f"   • Stock counts: {sorted(model_df['stocks_count'].unique())}")
            best_model = model_df.loc[model_df['auc'].idxmax()]
            print(f"   • Best performing: {best_model['model_type']} with {best_model['stocks_count']} stocks (AUC: {best_model['auc']:.4f})")
        
    except Exception as e:
        logger.error("Visualization analysis failed: {}", str(e))
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    if not PLOTLY_AVAILABLE:
        print("❌ Please install required dependencies: pip install plotly kaleido")
        sys.exit(1)
    
    asyncio.run(main())
