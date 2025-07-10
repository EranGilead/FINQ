#!/usr/bin/env python3
"""
Summary script to display all available visualizations for FINQ Stock Predictor.
Shows what charts are available and provides easy access to the dashboard.
"""

import os
import glob
import webbrowser
from datetime import datetime


def list_available_charts():
    """List all available visualization charts."""
    charts_dir = "visualizations/charts"
    
    if not os.path.exists(charts_dir):
        print("❌ Charts directory not found. Run visualization scripts first.")
        return []
    
    html_files = glob.glob(os.path.join(charts_dir, "*.html"))
    
    if not html_files:
        print("❌ No charts found. Run visualization scripts first.")
        return []
    
    chart_descriptions = {
        "index.html": "📊 Main Dashboard - Central hub for all visualizations",
        "stock_price_analysis.html": "📈 Stock Price Analysis - Interactive price charts and volume analysis",
        "model_performance_analysis.html": "🎯 Model Performance Analysis - Comprehensive model metrics comparison",
        "multi_scale_analysis.html": "📏 Multi-Scale Training Analysis - Scalability performance analysis",
        "financial_returns_analysis.html": "💰 Financial Returns Analysis - Risk-return, Sharpe ratios, beta/alpha",
        "correlation_analysis.html": "🔗 Correlation Analysis - Stock correlation heatmap and network visualization",
        "feature_importance.html": "🎯 Feature Importance - Top predictive features analysis",
        "model_performance_comparison.html": "📊 Model Comparison - Side-by-side model performance"
    }
    
    print("🎨 FINQ Stock Predictor - Available Visualizations")
    print("=" * 60)
    
    for html_file in sorted(html_files):
        filename = os.path.basename(html_file)
        description = chart_descriptions.get(filename, "📊 Interactive Chart")
        file_size = os.path.getsize(html_file) / 1024  # KB
        modified_time = datetime.fromtimestamp(os.path.getmtime(html_file))
        
        print(f"\n{description}")
        print(f"   📁 File: {filename}")
        print(f"   📐 Size: {file_size:.1f} KB")
        print(f"   🕒 Modified: {modified_time.strftime('%Y-%m-%d %H:%M')}")
        print(f"   🌐 Path: file://{os.path.abspath(html_file)}")
    
    return html_files


def open_dashboard():
    """Open the main dashboard in the default browser."""
    dashboard_path = "visualizations/charts/index.html"
    
    if not os.path.exists(dashboard_path):
        print("❌ Dashboard not found. Run visualization scripts first:")
        print("   python visualizations/comprehensive_analysis.py")
        return False
    
    abs_path = os.path.abspath(dashboard_path)
    dashboard_url = f"file://{abs_path}"
    
    try:
        print(f"🌐 Opening dashboard in your default browser...")
        print(f"   URL: {dashboard_url}")
        webbrowser.open(dashboard_url)
        return True
    except Exception as e:
        print(f"❌ Could not open browser automatically: {e}")
        print(f"🌐 Please open this URL manually: {dashboard_url}")
        return False


def show_usage_examples():
    """Show examples of how to generate visualizations."""
    print("\n🚀 How to Generate Visualizations")
    print("=" * 40)
    
    examples = [
        {
            "title": "📊 Complete Analysis Suite",
            "command": "python visualizations/comprehensive_analysis.py",
            "description": "Generates all main charts: stock analysis, model performance, multi-scale training"
        },
        {
            "title": "💰 Financial Analysis Extension",
            "command": "python visualizations/financial_analysis.py", 
            "description": "Adds financial metrics: returns, risk, correlations, Sharpe ratios"
        },
        {
            "title": "🎯 Train Models First (Optional)",
            "command": "python train.py --save-all-models --scale-step 10 --max-stocks 50",
            "description": "Generate multi-scale training data for richer visualizations"
        },
        {
            "title": "🌐 Open Dashboard",
            "command": "python visualizations/show_dashboard.py --open",
            "description": "Open the interactive dashboard in your browser"
        }
    ]
    
    for example in examples:
        print(f"\n{example['title']}")
        print(f"   Command: {example['command']}")
        print(f"   Description: {example['description']}")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="FINQ Stock Predictor Visualization Dashboard")
    parser.add_argument("--open", action="store_true", help="Open dashboard in browser")
    parser.add_argument("--list", action="store_true", help="List available charts")
    parser.add_argument("--examples", action="store_true", help="Show usage examples")
    
    args = parser.parse_args()
    
    if args.examples:
        show_usage_examples()
    elif args.list or not (args.open):
        charts = list_available_charts()
        if charts:
            print(f"\n✅ Found {len(charts)} visualization charts")
    
    if args.open:
        print("\n" + "="*60)
        success = open_dashboard()
        if success:
            print("✅ Dashboard opened successfully!")
        else:
            print("⚠️  Dashboard path provided above - please open manually")
    
    if not any([args.open, args.list, args.examples]):
        # Default behavior: show charts and offer to open dashboard
        charts = list_available_charts()
        if charts:
            print(f"\n{'='*60}")
            print("🌐 To open the interactive dashboard, run:")
            print("   python visualizations/show_dashboard.py --open")
            print("\n📚 For usage examples, run:")
            print("   python visualizations/show_dashboard.py --examples")


if __name__ == "__main__":
    main()
