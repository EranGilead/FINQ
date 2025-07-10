#!/usr/bin/env python3
"""
Comprehensive visualization generator for FINQ Stock Predictor.
Run all visualizations and create complete analysis dashboard.
"""

import asyncio
import os
import sys
import argparse
from datetime import datetime
from loguru import logger

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from visualizations.visualizer import FinqVisualizer
    from visualizations.financial_analyzer import FinancialAnalyzer
    from data.fetcher import get_sp500_data_async
    from data.processor import DataProcessor
    from features.engineer import FeatureEngineer
    PLOTLY_AVAILABLE = True
except ImportError as e:
    PLOTLY_AVAILABLE = False
    print(f"⚠️  Plotly not available: {e}")
    print("Install with: pip install plotly kaleido")


async def run_comprehensive_analysis(
    max_stocks: int = 20,
    output_dir: str = "visualizations/charts",
    include_financial: bool = True,
    include_models: bool = True
):
    """
    Run comprehensive visualization analysis.
    
    Args:
        max_stocks: Maximum number of stocks to analyze
        output_dir: Output directory for charts
        include_financial: Include financial analysis charts
        include_models: Include model analysis charts
    """
    if not PLOTLY_AVAILABLE:
        print("❌ Cannot run visualizations without Plotly. Please install requirements:")
        print("   pip install plotly kaleido")
        return
    
    print("🎨 FINQ Stock Predictor - Comprehensive Analysis")
    print("=" * 60)
    
    # Initialize components
    visualizer = FinqVisualizer(output_dir)
    if include_financial:
        financial_analyzer = FinancialAnalyzer()
    
    # Fetch data
    print(f"\n📊 Fetching data for {max_stocks} stocks...")
    stock_data, benchmark_data = await get_sp500_data_async(max_stocks=max_stocks)
    print(f"✅ Fetched data for {len(stock_data)} stocks")
    
    generated_charts = []
    
    # 1. Basic stock price analysis
    print("\n📈 Generating stock price analysis...")
    try:
        price_fig = visualizer.plot_stock_price_analysis(
            stock_data, benchmark_data, 
            tickers=list(stock_data.keys())[:8],
            time_range_days=252
        )
        path = visualizer.save_chart(price_fig, "stock_price_analysis")
        generated_charts.append(("Stock Price Analysis", path))
        print(f"✅ Saved: {os.path.basename(path)}")
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    # 2. Financial analysis (if enabled)
    if include_financial:
        print("\n💰 Generating financial analysis...")
        
        # Returns analysis
        try:
            returns_fig = financial_analyzer.plot_returns_analysis(
                stock_data, benchmark_data, 
                tickers=list(stock_data.keys())[:8]
            )
            path = visualizer.save_chart(returns_fig, "returns_analysis")
            generated_charts.append(("Returns Analysis", path))
            print(f"✅ Returns analysis: {os.path.basename(path)}")
        except Exception as e:
            print(f"❌ Returns analysis failed: {e}")
        
        # Volatility analysis
        try:
            vol_fig = financial_analyzer.plot_volatility_analysis(
                stock_data, benchmark_data
            )
            path = visualizer.save_chart(vol_fig, "volatility_analysis")
            generated_charts.append(("Volatility Analysis", path))
            print(f"✅ Volatility analysis: {os.path.basename(path)}")
        except Exception as e:
            print(f"❌ Volatility analysis failed: {e}")
        
        # Sector performance
        try:
            sector_fig = financial_analyzer.plot_sector_performance(
                stock_data, benchmark_data
            )
            path = visualizer.save_chart(sector_fig, "sector_performance")
            generated_charts.append(("Sector Performance", path))
            print(f"✅ Sector performance: {os.path.basename(path)}")
        except Exception as e:
            print(f"❌ Sector performance failed: {e}")
    
    # 3. Feature engineering analysis
    print("\n🔧 Generating feature analysis...")
    try:
        # Process data and engineer features
        processor = DataProcessor()
        engineer = FeatureEngineer()
        
        features, labels = processor.prepare_training_data(stock_data, benchmark_data)
        enhanced_features = engineer.engineer_features_multiple_stocks(stock_data)
        
        # Combine features
        all_features = []
        for ticker, ticker_features in enhanced_features.items():
            ticker_features['ticker'] = ticker
            ticker_features['date'] = ticker_features.index
            all_features.append(ticker_features)
        
        combined_features = pd.concat(all_features, ignore_index=True)
        merged_data = pd.merge(combined_features, labels, on=['ticker', 'date'], how='inner')
        
        print(f"✅ Generated {len(merged_data.columns)} features for {len(merged_data)} samples")
        
        # Feature correlation analysis
        if include_financial:
            try:
                corr_fig = financial_analyzer.plot_feature_correlation_matrix(
                    merged_data, target_col='outperforms', top_n=25
                )
                path = visualizer.save_chart(corr_fig, "feature_correlation")
                generated_charts.append(("Feature Correlation", path))
                print(f"✅ Feature correlation: {os.path.basename(path)}")
            except Exception as e:
                print(f"❌ Feature correlation failed: {e}")
        
    except Exception as e:
        print(f"❌ Feature engineering failed: {e}")
    
    # 4. Model analysis (if enabled and models exist)
    if include_models:
        print("\n🤖 Generating model analysis...")
        
        import glob
        model_files = glob.glob("models/saved/*.pkl")
        
        if model_files:
            # Feature importance
            try:
                latest_model = max(model_files, key=os.path.getctime)
                importance_fig = visualizer.plot_feature_importance(latest_model, top_n=30)
                path = visualizer.save_chart(importance_fig, "feature_importance")
                generated_charts.append(("Feature Importance", path))
                print(f"✅ Feature importance: {os.path.basename(path)}")
            except Exception as e:
                print(f"❌ Feature importance failed: {e}")
            
            # Model performance comparison
            try:
                performance_fig = visualizer.plot_model_performance_comparison()
                path = visualizer.save_chart(performance_fig, "model_performance")
                generated_charts.append(("Model Performance", path))
                print(f"✅ Model performance: {os.path.basename(path)}")
            except Exception as e:
                print(f"❌ Model performance failed: {e}")
        else:
            print("⚠️  No saved models found. Run training first:")
            print("   python train.py --save-model --max-stocks 20")
    
    # 5. Create comprehensive dashboard
    print("\n🌐 Creating comprehensive dashboard...")
    dashboard_path = create_comprehensive_dashboard(generated_charts, output_dir)
    
    # Summary
    print("\n" + "=" * 60)
    print("🎉 Comprehensive Analysis Complete!")
    print("=" * 60)
    print(f"📊 Generated {len(generated_charts)} visualizations")
    print(f"📁 Charts saved in: {output_dir}")
    print(f"🌐 Dashboard: file://{os.path.abspath(dashboard_path)}")
    
    print("\n📈 Generated Charts:")
    for name, path in generated_charts:
        print(f"   • {name}: {os.path.basename(path)}")
    
    return generated_charts, dashboard_path


def create_comprehensive_dashboard(charts: list, output_dir: str) -> str:
    """Create comprehensive HTML dashboard."""
    
    dashboard_html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>FINQ Stock Predictor - Comprehensive Analysis Dashboard</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
        .header {{ 
            background: rgba(255,255,255,0.95); 
            padding: 30px; 
            border-radius: 15px; 
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .header h1 {{ color: #2c3e50; margin-bottom: 10px; font-size: 2.5em; }}
        .header h2 {{ color: #3498db; margin-bottom: 15px; }}
        .header p {{ color: #7f8c8d; font-size: 1.1em; }}
        .stats {{ 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
            gap: 20px; 
            margin: 30px 0; 
        }}
        .stat-card {{ 
            background: rgba(255,255,255,0.9); 
            padding: 25px; 
            border-radius: 12px; 
            text-align: center;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            transition: transform 0.3s ease;
        }}
        .stat-card:hover {{ transform: translateY(-5px); }}
        .stat-number {{ font-size: 2.5em; font-weight: bold; color: #e74c3c; margin-bottom: 5px; }}
        .stat-label {{ color: #7f8c8d; font-size: 1.1em; }}
        .charts-grid {{ 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); 
            gap: 25px; 
            margin: 30px 0; 
        }}
        .chart-card {{ 
            background: rgba(255,255,255,0.95); 
            padding: 25px; 
            border-radius: 15px;
            box-shadow: 0 8px 25px rgba(0,0,0,0.1);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }}
        .chart-card:hover {{ 
            transform: translateY(-8px); 
            box-shadow: 0 15px 40px rgba(0,0,0,0.2);
        }}
        .chart-card h3 {{ color: #2c3e50; margin-bottom: 15px; font-size: 1.4em; }}
        .chart-card p {{ color: #7f8c8d; margin-bottom: 20px; line-height: 1.6; }}
        .chart-link {{ 
            display: inline-block; 
            background: linear-gradient(45deg, #3498db, #2980b9); 
            color: white; 
            padding: 12px 25px; 
            text-decoration: none; 
            border-radius: 8px; 
            transition: all 0.3s ease;
            font-weight: 500;
        }}
        .chart-link:hover {{ 
            background: linear-gradient(45deg, #2980b9, #1f5f8b);
            transform: scale(1.05);
        }}
        .footer {{ 
            background: rgba(255,255,255,0.95); 
            padding: 30px; 
            border-radius: 15px; 
            text-align: center; 
            margin-top: 30px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }}
        .timestamp {{ color: #95a5a6; font-size: 0.9em; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 FINQ Stock Predictor</h1>
            <h2>Comprehensive Analysis Dashboard</h2>
            <p>Interactive financial data analysis and machine learning insights</p>
        </div>
        
        <div class="stats">
            <div class="stat-card">
                <div class="stat-number">{len(charts)}</div>
                <div class="stat-label">Interactive Charts</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">S&P 500</div>
                <div class="stat-label">Market Analysis</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">ML</div>
                <div class="stat-label">Model Insights</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">Real-time</div>
                <div class="stat-label">Data Updates</div>
            </div>
        </div>
        
        <div class="charts-grid">
"""
    
    # Chart descriptions with icons
    chart_info = {
        "Stock Price Analysis": {
            "icon": "📈",
            "description": "Interactive price charts and volume analysis showing stock performance vs S&P 500 benchmark over time."
        },
        "Returns Analysis": {
            "icon": "💰", 
            "description": "Distribution of returns, outperformance rates, and risk-return analysis across different stocks."
        },
        "Volatility Analysis": {
            "icon": "📊",
            "description": "Rolling volatility patterns and risk comparison between individual stocks and market benchmark."
        },
        "Sector Performance": {
            "icon": "🏭",
            "description": "Performance breakdown by industry sectors showing returns, risk, and Sharpe ratios."
        },
        "Feature Correlation": {
            "icon": "🔗",
            "description": "Correlation matrix heatmap showing relationships between technical indicators and target variable."
        },
        "Feature Importance": {
            "icon": "🎯",
            "description": "Ranking of most important features driving model predictions with importance scores."
        },
        "Model Performance": {
            "icon": "🤖",
            "description": "Comparative analysis of different ML models across accuracy, precision, recall, and AUC metrics."
        }
    }
    
    # Add chart cards
    for name, path in charts:
        info = chart_info.get(name, {"icon": "📊", "description": "Interactive data visualization and analysis."})
        filename = os.path.basename(path)
        
        dashboard_html += f"""
            <div class="chart-card">
                <h3>{info['icon']} {name}</h3>
                <p>{info['description']}</p>
                <a href="{filename}" target="_blank" class="chart-link">
                    View Interactive Chart →
                </a>
            </div>
"""
    
    dashboard_html += f"""
        </div>
        
        <div class="footer">
            <h3>🔧 How to Use This Dashboard</h3>
            <p style="margin: 15px 0; line-height: 1.8;">
                Click on any chart above to open it in a new tab. All charts are fully interactive - 
                you can zoom, pan, hover for details, toggle data series, and export as images.
                For best experience, use a modern web browser with JavaScript enabled.
            </p>
            <div class="timestamp">
                Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} by FINQ Stock Predictor
            </div>
        </div>
    </div>
</body>
</html>
"""
    
    # Save dashboard
    dashboard_path = os.path.join(output_dir, "comprehensive_dashboard.html")
    with open(dashboard_path, 'w', encoding='utf-8') as f:
        f.write(dashboard_html)
    
    return dashboard_path


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Generate comprehensive FINQ visualizations")
    parser.add_argument("--max-stocks", type=int, default=20, help="Maximum number of stocks to analyze")
    parser.add_argument("--output-dir", type=str, default="visualizations/charts", help="Output directory")
    parser.add_argument("--no-financial", action="store_true", help="Skip financial analysis charts")
    parser.add_argument("--no-models", action="store_true", help="Skip model analysis charts")
    
    args = parser.parse_args()
    
    try:
        charts, dashboard = await run_comprehensive_analysis(
            max_stocks=args.max_stocks,
            output_dir=args.output_dir,
            include_financial=not args.no_financial,
            include_models=not args.no_models
        )
        
        print(f"\n🎉 Success! Open dashboard: file://{os.path.abspath(dashboard)}")
        
    except KeyboardInterrupt:
        print("\n👋 Analysis interrupted by user")
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        logger.error("Comprehensive analysis error: {}", e)
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import pandas as pd  # Import here to avoid import errors in main
    asyncio.run(main())
