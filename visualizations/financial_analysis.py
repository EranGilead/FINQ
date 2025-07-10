#!/usr/bin/env python3
"""
Financial analysis visualizations for FINQ Stock Predictor.
Creates additional charts focused on financial metrics and returns analysis.
"""

import asyncio
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

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


def calculate_returns(stock_data, benchmark_data, period_days=252):
    """Calculate returns and metrics for stocks vs benchmark."""
    returns_data = {}
    
    for ticker, data in stock_data.items():
        if len(data) < period_days:
            continue
            
        # Calculate daily returns
        stock_returns = data['Close'].pct_change().dropna()
        
        # Align with benchmark
        aligned_data = pd.DataFrame({
            'stock': stock_returns,
            'benchmark': benchmark_data['Close'].pct_change()
        }).dropna()
        
        if len(aligned_data) < 100:  # Need sufficient data
            continue
            
        # Calculate metrics
        stock_annual_return = aligned_data['stock'].mean() * 252
        stock_volatility = aligned_data['stock'].std() * np.sqrt(252)
        benchmark_annual_return = aligned_data['benchmark'].mean() * 252
        benchmark_volatility = aligned_data['benchmark'].std() * np.sqrt(252)
        
        # Sharpe ratio (assuming risk-free rate of 2%)
        risk_free_rate = 0.02
        stock_sharpe = (stock_annual_return - risk_free_rate) / stock_volatility if stock_volatility > 0 else 0
        
        # Beta calculation
        covariance = np.cov(aligned_data['stock'], aligned_data['benchmark'])[0, 1]
        benchmark_variance = np.var(aligned_data['benchmark'])
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
        
        # Alpha calculation
        alpha = stock_annual_return - (risk_free_rate + beta * (benchmark_annual_return - risk_free_rate))
        
        returns_data[ticker] = {
            'annual_return': stock_annual_return,
            'volatility': stock_volatility,
            'sharpe_ratio': stock_sharpe,
            'beta': beta,
            'alpha': alpha,
            'daily_returns': aligned_data['stock']
        }
    
    return returns_data


def create_returns_analysis(stock_data, benchmark_data, save_path="visualizations/charts"):
    """Create returns and risk analysis visualization."""
    print("\n💰 Creating returns and risk analysis...")
    
    os.makedirs(save_path, exist_ok=True)
    
    # Calculate returns
    returns_data = calculate_returns(stock_data, benchmark_data)
    
    if not returns_data:
        print("⚠️  No sufficient data for returns analysis")
        return None
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Risk-Return Scatter Plot', 'Sharpe Ratio Comparison',
                       'Beta vs Alpha Analysis', 'Daily Returns Distribution'),
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    # Prepare data for plots
    tickers = list(returns_data.keys())
    returns = [returns_data[t]['annual_return'] for t in tickers]
    volatilities = [returns_data[t]['volatility'] for t in tickers]
    sharpe_ratios = [returns_data[t]['sharpe_ratio'] for t in tickers]
    betas = [returns_data[t]['beta'] for t in tickers]
    alphas = [returns_data[t]['alpha'] for t in tickers]
    
    # 1. Risk-Return scatter plot
    fig.add_trace(
        go.Scatter(
            x=volatilities,
            y=returns,
            mode='markers+text',
            text=tickers,
            textposition='top center',
            marker=dict(
                size=12,
                color=sharpe_ratios,
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="Sharpe Ratio", x=0.48)
            ),
            name='Stocks',
            hovertemplate='<b>%{text}</b><br>Return: %{y:.2%}<br>Volatility: %{x:.2%}<br>Sharpe: %{marker.color:.3f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # 2. Sharpe ratio bar chart
    colors = ['green' if sr > 0 else 'red' for sr in sharpe_ratios]
    fig.add_trace(
        go.Bar(
            x=tickers,
            y=sharpe_ratios,
            marker_color=colors,
            name='Sharpe Ratio',
            showlegend=False,
            hovertemplate='<b>%{x}</b><br>Sharpe Ratio: %{y:.3f}<extra></extra>'
        ),
        row=1, col=2
    )
    
    # 3. Beta vs Alpha
    fig.add_trace(
        go.Scatter(
            x=betas,
            y=alphas,
            mode='markers+text',
            text=tickers,
            textposition='top center',
            marker=dict(
                size=12,
                color=returns,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Annual Return", x=1.02)
            ),
            name='Beta vs Alpha',
            showlegend=False,
            hovertemplate='<b>%{text}</b><br>Beta: %{x:.3f}<br>Alpha: %{y:.2%}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Add beta = 1 line
    fig.add_vline(x=1, line_dash="dash", line_color="gray", row=2, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
    
    # 4. Daily returns distribution (show top 5 stocks)
    top_sharpe_tickers = sorted(tickers, key=lambda t: returns_data[t]['sharpe_ratio'], reverse=True)[:5]
    colors_dist = px.colors.qualitative.Set2
    
    for i, ticker in enumerate(top_sharpe_tickers):
        daily_returns = returns_data[ticker]['daily_returns']
        fig.add_trace(
            go.Histogram(
                x=daily_returns,
                name=f'{ticker}',
                opacity=0.7,
                marker_color=colors_dist[i % len(colors_dist)],
                nbinsx=50,
                showlegend=False,
                hovertemplate=f'<b>{ticker}</b><br>Return: %{{x:.2%}}<br>Count: %{{y}}<extra></extra>'
            ),
            row=2, col=2
        )
    
    # Update layout
    fig.update_layout(
        title={
            'text': 'FINQ Stock Predictor - Financial Returns & Risk Analysis',
            'x': 0.5,
            'font': {'size': 20}
        },
        height=800,
        showlegend=False
    )
    
    # Update axes
    fig.update_xaxes(title_text="Volatility (Annual)", row=1, col=1)
    fig.update_yaxes(title_text="Annual Return", row=1, col=1)
    fig.update_yaxes(title_text="Sharpe Ratio", row=1, col=2)
    fig.update_xaxes(title_text="Beta", row=2, col=1)
    fig.update_yaxes(title_text="Alpha", row=2, col=1)
    fig.update_xaxes(title_text="Daily Return", row=2, col=2)
    fig.update_yaxes(title_text="Frequency", row=2, col=2)
    
    # Save chart
    file_path = os.path.join(save_path, "financial_returns_analysis.html")
    fig.write_html(file_path)
    print(f"✅ Saved: {file_path}")
    
    return fig


def create_correlation_analysis(stock_data, benchmark_data, save_path="visualizations/charts"):
    """Create correlation analysis visualization."""
    print("\n🔗 Creating correlation analysis...")
    
    os.makedirs(save_path, exist_ok=True)
    
    # Prepare price data
    price_data = {}
    for ticker, data in stock_data.items():
        price_data[ticker] = data['Close']
    
    price_data['SPY'] = benchmark_data['Close']
    
    # Create DataFrame and calculate returns
    prices_df = pd.DataFrame(price_data).dropna()
    returns_df = prices_df.pct_change().dropna()
    
    if len(returns_df.columns) < 3:
        print("⚠️  Not enough data for correlation analysis")
        return None
    
    # Calculate correlation matrix
    correlation_matrix = returns_df.corr()
    
    # Create visualization
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Returns Correlation Heatmap', 'Price Correlation Network'),
        horizontal_spacing=0.15,
        specs=[[{"type": "heatmap"}, {"type": "scatter"}]]
    )
    
    # 1. Correlation heatmap
    fig.add_trace(
        go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=correlation_matrix.round(3).values,
            texttemplate="%{text}",
            textfont={"size": 10},
            hovertemplate='%{y} vs %{x}<br>Correlation: %{z:.3f}<extra></extra>',
            showscale=True,
            colorbar=dict(title="Correlation", x=0.48)
        ),
        row=1, col=1
    )
    
    # 2. Network-style visualization of correlations
    # Show only correlations above threshold
    threshold = 0.6
    tickers = list(correlation_matrix.columns)
    n_tickers = len(tickers)
    
    # Position tickers in a circle
    angles = np.linspace(0, 2*np.pi, n_tickers, endpoint=False)
    x_pos = np.cos(angles)
    y_pos = np.sin(angles)
    
    # Add ticker nodes
    fig.add_trace(
        go.Scatter(
            x=x_pos,
            y=y_pos,
            mode='markers+text',
            text=tickers,
            textposition='middle center',
            marker=dict(
                size=20,
                color='lightblue',
                line=dict(width=2, color='darkblue')
            ),
            name='Tickers',
            showlegend=False,
            hovertemplate='<b>%{text}</b><extra></extra>'
        ),
        row=1, col=2
    )
    
    # Add correlation lines
    for i in range(n_tickers):
        for j in range(i+1, n_tickers):
            corr_val = correlation_matrix.iloc[i, j]
            if abs(corr_val) >= threshold:
                color = 'red' if corr_val > 0 else 'blue'
                width = abs(corr_val) * 5
                
                fig.add_trace(
                    go.Scatter(
                        x=[x_pos[i], x_pos[j]],
                        y=[y_pos[i], y_pos[j]],
                        mode='lines',
                        line=dict(color=color, width=width),
                        showlegend=False,
                        hoverinfo='skip'
                    ),
                    row=1, col=2
                )
    
    # Update layout
    fig.update_layout(
        title={
            'text': 'FINQ Stock Predictor - Correlation Analysis',
            'x': 0.5,
            'font': {'size': 20}
        },
        height=600
    )
    
    # Update subplot 2 to remove axes
    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False, row=1, col=2)
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False, row=1, col=2)
    
    # Save chart
    file_path = os.path.join(save_path, "correlation_analysis.html")
    fig.write_html(file_path)
    print(f"✅ Saved: {file_path}")
    
    return fig


async def main():
    """Main financial analysis function."""
    print("💼 FINQ Stock Predictor - Financial Analysis Extension")
    print("=" * 60)
    
    try:
        # Fetch data
        print("📊 Fetching data for financial analysis...")
        stock_data, benchmark_data = await get_sp500_data_async(max_stocks=12)
        print(f"✅ Fetched data for {len(stock_data)} stocks")
        
        # Create financial analysis charts
        charts_dir = "visualizations/charts"
        os.makedirs(charts_dir, exist_ok=True)
        
        # Returns and risk analysis
        create_returns_analysis(stock_data, benchmark_data, charts_dir)
        
        # Correlation analysis
        create_correlation_analysis(stock_data, benchmark_data, charts_dir)
        
        print("\n🎉 Financial Analysis Complete!")
        print("=" * 50)
        print(f"📁 Charts directory: {os.path.abspath(charts_dir)}")
        print("🌐 Refresh your dashboard to see the new financial analysis charts!")
        
    except Exception as e:
        print(f"❌ Financial analysis failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    if not PLOTLY_AVAILABLE:
        print("❌ Please install required dependencies: pip install plotly kaleido")
        sys.exit(1)
    
    asyncio.run(main())
