#!/usr/bin/env python3
"""
Demo script for FINQ Stock Predictor visualizations.
Generates interactive charts and dashboards for data analysis.
"""

import asyncio
import os
import sys
from loguru import logger

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visualizations.visualizer import FinqVisualizer
from data.fetcher import get_sp500_data_async
from data.processor import DataProcessor
from models.trainer import ModelTrainer
from train import train_multi_scale_models
import glob


async def generate_data_visualizations():
    """Generate visualizations for stock data analysis."""
    print("📊 Generating Data Visualizations")
    print("=" * 50)
    
    # Initialize visualizer
    visualizer = FinqVisualizer()
    
    # Fetch sample data
    print("Fetching stock data...")
    stock_data, benchmark_data = await get_sp500_data_async(max_stocks=10)
    print(f"✅ Fetched data for {len(stock_data)} stocks")
    
    # 1. Stock Price Analysis
    print("\n📈 Creating stock price analysis chart...")
    price_fig = visualizer.plot_stock_price_analysis(
        stock_data, benchmark_data, 
        tickers=list(stock_data.keys())[:5],
        time_range_days=180
    )
    price_path = visualizer.save_chart(price_fig, "stock_price_analysis")
    print(f"✅ Saved: {price_path}")
    
    return visualizer, stock_data, benchmark_data


def generate_model_visualizations(visualizer: FinqVisualizer):
    """Generate visualizations for model analysis."""
    print("\n🤖 Generating Model Visualizations")
    print("=" * 50)
    
    # Find saved models
    model_files = glob.glob(os.path.join("models/saved", "*.pkl"))
    
    if not model_files:
        print("⚠️  No saved models found. Run training first:")
        print("   python train.py --save-model --max-stocks 10")
        return
    
    print(f"Found {len(model_files)} saved models")
    
    # 1. Feature Importance (use most recent model)
    latest_model = max(model_files, key=os.path.getctime)
    print(f"\n📊 Creating feature importance chart from: {os.path.basename(latest_model)}")
    
    try:
        importance_fig = visualizer.plot_feature_importance(latest_model, top_n=25)
        importance_path = visualizer.save_chart(importance_fig, "feature_importance")
        print(f"✅ Saved: {importance_path}")
    except Exception as e:
        print(f"❌ Failed to create feature importance chart: {e}")
    
    # 2. Model Performance Comparison
    print("\n📈 Creating model performance comparison...")
    try:
        performance_fig = visualizer.plot_model_performance_comparison()
        performance_path = visualizer.save_chart(performance_fig, "model_performance_comparison")
        print(f"✅ Saved: {performance_path}")
    except Exception as e:
        print(f"❌ Failed to create performance comparison: {e}")


async def generate_multi_scale_visualizations(visualizer: FinqVisualizer):
    """Generate visualizations from multi-scale training."""
    print("\n📏 Generating Multi-Scale Training Visualizations")
    print("=" * 50)
    
    print("Running quick multi-scale training experiment...")
    
    try:
        # Fetch data for multi-scale training
        stock_data, benchmark_data = await get_sp500_data_async(max_stocks=20)
        
        # Mock arguments for multi-scale training
        class MockArgs:
            def __init__(self):
                self.async_training = False
                self.max_workers = None
                self.save_model = False
                self.save_all_models = False
                self.tune_hyperparameters = False
        
        args = MockArgs()
        
        # Run multi-scale training with small increments for demo
        results = await train_multi_scale_models(
            stock_data, benchmark_data, 
            max_stocks=15, scale_step=5, args=args
        )
        
        if results:
            print(f"✅ Multi-scale training completed with {len(results)} experiments")
            
            # Create multi-scale visualization
            multiscale_fig = visualizer.plot_multi_scale_results(results)
            multiscale_path = visualizer.save_chart(multiscale_fig, "multi_scale_results")
            print(f"✅ Saved: {multiscale_path}")
        else:
            print("⚠️  No multi-scale results to visualize")
            
    except Exception as e:
        print(f"❌ Multi-scale visualization failed: {e}")
        logger.error("Multi-scale visualization error: {}", e)


async def create_dashboard_index():
    """Create an index HTML file linking all visualizations."""
    print("\n🌐 Creating Dashboard Index")
    print("=" * 50)
    
    charts_dir = "visualizations/charts"
    html_files = glob.glob(os.path.join(charts_dir, "*.html"))
    
    if not html_files:
        print("⚠️  No HTML charts found to index")
        return
    
    # Create index HTML
    index_html = """
<!DOCTYPE html>
<html>
<head>
    <title>FINQ Stock Predictor - Visualization Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .header { background: #2c3e50; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
        .chart-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        .chart-card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .chart-card h3 { margin-top: 0; color: #2c3e50; }
        .chart-card a { display: inline-block; background: #3498db; color: white; padding: 10px 20px; text-decoration: none; border-radius: 4px; margin-top: 10px; }
        .chart-card a:hover { background: #2980b9; }
        .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }
        .stat-box { background: white; padding: 15px; border-radius: 8px; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .stat-number { font-size: 2em; font-weight: bold; color: #e74c3c; }
        .stat-label { color: #7f8c8d; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 FINQ Stock Predictor</h1>
        <h2>Interactive Visualization Dashboard</h2>
        <p>Explore data patterns, model performance, and experiment results</p>
    </div>
    
    <div class="stats">
        <div class="stat-box">
            <div class="stat-number">{chart_count}</div>
            <div class="stat-label">Interactive Charts</div>
        </div>
        <div class="stat-box">
            <div class="stat-number">Real-time</div>
            <div class="stat-label">Data Analysis</div>
        </div>
        <div class="stat-box">
            <div class="stat-number">ML</div>
            <div class="stat-label">Model Insights</div>
        </div>
    </div>
    
    <div class="chart-grid">
""".format(chart_count=len(html_files))
    
    # Chart descriptions
    chart_descriptions = {
        "stock_price_analysis": {
            "title": "📈 Stock Price Analysis",
            "description": "Interactive price charts and volume analysis for selected S&P 500 stocks compared to benchmark."
        },
        "feature_importance": {
            "title": "🎯 Feature Importance",
            "description": "Top features driving model predictions with importance scores and rankings."
        },
        "model_performance_comparison": {
            "title": "📊 Model Performance",
            "description": "Comparative analysis of different ML models across various metrics (AUC, accuracy, precision, recall)."
        },
        "multi_scale_results": {
            "title": "📏 Multi-Scale Training",
            "description": "Performance analysis across different dataset sizes showing scalability patterns."
        },
        "prediction_distribution": {
            "title": "🎲 Prediction Analysis",
            "description": "Distribution of prediction probabilities and confidence scores with accuracy analysis."
        }
    }
    
    # Add chart cards
    for html_file in sorted(html_files):
        filename = os.path.basename(html_file)
        chart_name = filename.replace('.html', '')
        
        chart_info = chart_descriptions.get(chart_name, {
            "title": f"📊 {chart_name.replace('_', ' ').title()}",
            "description": "Interactive visualization and analysis chart."
        })
        
        chart_card = f'''
        <div class="chart-card">
            <h3>{chart_info["title"]}</h3>
            <p>{chart_info["description"]}</p>
            <a href="{filename}" target="_blank">View Chart →</a>
        </div>
'''
        index_html += chart_card
    
    # Close HTML
    closing_html = '''
    </div>
    
    <div style="margin-top: 40px; padding: 20px; background: white; border-radius: 8px; text-align: center;">
        <h3>🔧 How to Use</h3>
        <p>Click on any chart above to open it in a new tab. Charts are interactive - you can zoom, pan, hover for details, and toggle data series.</p>
        <p><strong>Tip:</strong> For best experience, use a modern web browser with JavaScript enabled.</p>
    </div>
    
    <footer style="margin-top: 20px; text-align: center; color: #7f8c8d;">
        <p>Generated by FINQ Stock Predictor Visualization System</p>
    </footer>
</body>
</html>
'''
    index_html += closing_html
    
    # Save index file
    index_path = os.path.join(charts_dir, "index.html")
    with open(index_path, 'w') as f:
        f.write(index_html)
    
    print(f"✅ Dashboard index created: {index_path}")
    print(f"🌐 Open in browser: file://{os.path.abspath(index_path)}")
    
    return index_path


async def main():
    """Main demo function."""
    print("🎨 FINQ Stock Predictor - Visualization Demo")
    print("=" * 60)
    
    try:
        # Generate data visualizations
        visualizer, stock_data, benchmark_data = await generate_data_visualizations()
        
        # Generate model visualizations
        generate_model_visualizations(visualizer)
        
        # Generate multi-scale visualizations (optional - takes time)
        user_input = input("\n❓ Run multi-scale training demo? (y/N): ").lower().strip()
        if user_input == 'y':
            await generate_multi_scale_visualizations(visualizer)
        else:
            print("⏭️  Skipping multi-scale demo")
        
        # Create dashboard index
        index_path = await create_dashboard_index()
        
        print("\n" + "=" * 60)
        print("🎉 Visualization Demo Complete!")
        print("=" * 60)
        print(f"📁 Charts saved in: visualizations/charts/")
        print(f"🌐 Dashboard: file://{os.path.abspath(index_path)}")
        print("\n💡 Next Steps:")
        print("   1. Open the dashboard in your web browser")
        print("   2. Explore the interactive charts")
        print("   3. Run more training experiments to see updated visualizations")
        
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        logger.error("Visualization demo error: {}", e)
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
