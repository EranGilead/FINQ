# FINQ Stock Predictor - Visualization System

This directory contains a comprehensive suite of interactive visualization tools built with Plotly to analyze data, experiments, and model performance for the FINQ Stock Predictor.

## 🎨 Available Visualizations

### 📊 Main Dashboard (`index.html`)
Central hub providing access to all visualizations with an elegant, responsive interface.

### 📈 Stock Analysis
- **Stock Price Analysis**: Interactive price charts with volume analysis, comparing selected stocks vs S&P 500 benchmark
- **Financial Returns Analysis**: Risk-return scatter plots, Sharpe ratios, beta/alpha analysis, and daily returns distributions
- **Correlation Analysis**: Correlation heatmaps and network visualizations showing relationships between stock returns

### 🎯 Model Performance
- **Model Performance Analysis**: Comprehensive comparison of model metrics (AUC, accuracy, precision, recall) across different configurations
- **Multi-Scale Training Analysis**: Scalability analysis showing how models perform with different dataset sizes
- **Feature Importance**: Top predictive features with importance scores and rankings

## 🚀 Quick Start

### Generate All Visualizations
```bash
# Complete analysis suite (stock analysis, model performance, multi-scale training)
python visualizations/comprehensive_analysis.py

# Financial analysis extension (returns, risk, correlations)
python visualizations/financial_analysis.py

# Open the interactive dashboard
python visualizations/show_dashboard.py --open
```

### View Available Charts
```bash
# List all available visualizations
python visualizations/show_dashboard.py --list

# Show usage examples
python visualizations/show_dashboard.py --examples
```

## 📁 File Structure

```
visualizations/
├── charts/                          # Generated HTML charts
│   ├── index.html                   # Main dashboard
│   ├── stock_price_analysis.html    # Price and volume charts
│   ├── financial_returns_analysis.html # Risk-return analysis
│   ├── correlation_analysis.html    # Stock correlations
│   ├── model_performance_analysis.html # Model metrics
│   ├── multi_scale_analysis.html    # Scalability analysis
│   └── feature_importance.html      # Feature rankings
├── comprehensive_analysis.py        # Main visualization generator
├── financial_analysis.py           # Financial metrics analyzer
├── show_dashboard.py               # Dashboard viewer/launcher
├── demo_clean.py                   # Simple demo runner
├── visualizer.py                   # Core visualization class
├── financial_analyzer.py           # Financial analysis utilities
├── demo.py                         # Original demo script
├── analyze.py                      # Analysis runner
└── README.md                       # This file
```

## 🎯 Visualization Features

### Interactive Charts
- **Zoom & Pan**: Click and drag to zoom, shift+drag to pan
- **Hover Details**: Hover over data points for detailed information
- **Toggle Series**: Click legend items to show/hide data series
- **Download**: Use toolbar to save charts as images
- **Responsive**: Charts adapt to screen size

### Analysis Types

#### 📈 Financial Analysis
- **Risk-Return Plots**: Volatility vs annual returns with Sharpe ratio coloring
- **Beta-Alpha Analysis**: Market sensitivity and excess returns
- **Correlation Networks**: Visual representation of stock relationships
- **Returns Distributions**: Histogram analysis of daily returns

#### 🎯 Model Performance
- **Multi-Model Comparison**: Performance across different algorithms
- **Scalability Analysis**: How performance changes with dataset size
- **Feature Importance**: Which features drive predictions
- **Cross-Validation Metrics**: Robust performance assessment

#### 📊 Data Quality
- **Price Trends**: Long-term and short-term price movements
- **Volume Patterns**: Trading volume analysis
- **Benchmark Comparison**: Performance relative to S&P 500
- **Time Series Analysis**: Temporal patterns and seasonality

## 🛠️ Technical Details

### Dependencies
- **Plotly**: Interactive plotting library
- **Kaleido**: Static image export (for headless environments)
- **Pandas**: Data manipulation
- **NumPy**: Numerical computations
- **Asyncio**: Asynchronous data fetching

### Performance Optimization
- **Async Data Fetching**: Parallel data retrieval for faster loading
- **Efficient Processing**: Vectorized operations with pandas/numpy
- **Smart Caching**: Reuse processed data when possible
- **Responsive Design**: Charts adapt to different screen sizes

### Chart Types Used
- **Line Charts**: Time series data (prices, performance trends)
- **Scatter Plots**: Correlation analysis, risk-return relationships
- **Bar Charts**: Categorical comparisons, feature importance
- **Heatmaps**: Correlation matrices, performance grids
- **Violin Plots**: Distribution analysis
- **Histograms**: Returns and probability distributions
- **Network Plots**: Correlation relationships

## 📊 Data Sources

### Stock Data
- **Price Data**: OHLC prices from Yahoo Finance
- **Volume Data**: Trading volumes and patterns
- **Benchmark Data**: S&P 500 index for comparison
- **Technical Indicators**: Moving averages, volatility measures

### Model Data
- **Training Results**: Performance metrics from saved models
- **Feature Data**: Engineered features and importance scores
- **Cross-Validation**: Robust performance estimates
- **Multi-Scale Results**: Performance across different dataset sizes

## 🎨 Customization

### Adding New Charts
1. Create a new function in the appropriate analysis script
2. Use Plotly's `make_subplots` for complex layouts
3. Add proper hover templates and styling
4. Save to the charts directory
5. Update the dashboard index

### Styling Guidelines
- **Color Schemes**: Use Plotly's built-in color palettes (Set2, Viridis, etc.)
- **Hover Templates**: Provide detailed information on hover
- **Responsive Layout**: Use percentage-based sizing
- **Professional Styling**: Clean, business-appropriate aesthetics

## 🔧 Troubleshooting

### Common Issues

#### Charts Not Displaying
```bash
# Check if Plotly is installed
pip install plotly kaleido

# Verify chart files exist
ls -la visualizations/charts/
```

#### Performance Issues
```bash
# Reduce dataset size for testing
python visualizations/comprehensive_analysis.py --max-stocks 10

# Use cached data when available
# (Data is automatically cached after first fetch)
```

#### Browser Compatibility
- **Recommended**: Chrome, Firefox, Safari (latest versions)
- **JavaScript**: Must be enabled for interactivity
- **Local Files**: Some browsers may block local file access

### Debug Mode
```bash
# Run with detailed logging
python -u visualizations/comprehensive_analysis.py

# Check for errors in specific charts
python -c "import plotly; print(plotly.__version__)"
```

## 📈 Future Enhancements

### Planned Features
- **Real-time Updates**: Live data feeds and automatic refresh
- **Custom Dashboards**: User-configurable chart combinations
- **Export Options**: PDF reports, PowerPoint integration
- **Advanced Analytics**: Statistical significance testing, confidence intervals

### Integration Ideas
- **MLflow Integration**: Track experiments and model versions
- **Jupyter Notebooks**: Embedded charts in analysis notebooks
- **Web Deployment**: Host dashboards on cloud platforms
- **API Endpoints**: Programmatic access to chart data

## 📞 Support

For issues or questions about the visualization system:

1. **Check Logs**: Look for error messages in terminal output
2. **Verify Dependencies**: Ensure all required packages are installed
3. **Test with Sample Data**: Use smaller datasets to isolate issues
4. **Browser Console**: Check for JavaScript errors in browser developer tools

---

**Generated by FINQ Stock Predictor Visualization System**
*Last Updated: July 2025*
