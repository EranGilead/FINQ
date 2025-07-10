"""
Financial analysis visualizations for FINQ Stock Predictor.
Specialized charts for financial data analysis and model insights.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from loguru import logger
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class FinancialAnalyzer:
    """
    Specialized financial analysis and visualization tools.
    """
    
    def __init__(self):
        """Initialize the financial analyzer."""
        logger.info("FinancialAnalyzer initialized")
    
    def plot_returns_analysis(
        self, 
        stock_data: Dict[str, pd.DataFrame],
        benchmark_data: pd.DataFrame,
        tickers: List[str] = None,
        prediction_horizon: int = 5
    ) -> go.Figure:
        """
        Analyze returns distribution and outperformance patterns.
        
        Args:
            stock_data: Dictionary mapping ticker to DataFrame
            benchmark_data: Benchmark data
            tickers: List of tickers to analyze
            prediction_horizon: Days ahead for return calculation
            
        Returns:
            Plotly Figure object
        """
        if tickers is None:
            tickers = list(stock_data.keys())[:8]
        
        # Calculate returns for analysis
        returns_data = []
        
        for ticker in tickers:
            if ticker not in stock_data:
                continue
                
            data = stock_data[ticker]
            
            # Calculate forward returns
            stock_returns = data['Close'].pct_change(periods=prediction_horizon).dropna()
            
            # Align with benchmark
            common_dates = stock_returns.index.intersection(benchmark_data.index)
            if len(common_dates) < 50:  # Need enough data
                continue
                
            stock_ret_aligned = stock_returns.loc[common_dates]
            bench_ret_aligned = benchmark_data['Close'].pct_change(periods=prediction_horizon).loc[common_dates]
            
            # Calculate excess returns
            excess_returns = stock_ret_aligned - bench_ret_aligned
            outperformance = (excess_returns > 0).astype(int)
            
            for i, date in enumerate(common_dates):
                returns_data.append({
                    'ticker': ticker,
                    'date': date,
                    'stock_return': stock_ret_aligned.iloc[i],
                    'benchmark_return': bench_ret_aligned.iloc[i],
                    'excess_return': excess_returns.iloc[i],
                    'outperforms': outperformance.iloc[i]
                })
        
        if not returns_data:
            return go.Figure().add_annotation(text="No sufficient data for analysis", x=0.5, y=0.5)
        
        df = pd.DataFrame(returns_data)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Returns Distribution by Stock",
                "Outperformance Rate by Stock",
                "Excess Returns Over Time",
                "Risk-Return Scatter"
            )
        )
        
        # 1. Returns distribution (box plot)
        for ticker in df['ticker'].unique():
            ticker_data = df[df['ticker'] == ticker]
            fig.add_trace(
                go.Box(
                    y=ticker_data['stock_return'] * 100,  # Convert to percentage
                    name=ticker,
                    showlegend=False
                ),
                row=1, col=1
            )
        
        # 2. Outperformance rate (bar chart)
        outperf_rates = df.groupby('ticker')['outperforms'].mean()
        fig.add_trace(
            go.Bar(
                x=outperf_rates.index,
                y=outperf_rates.values * 100,
                name="Outperformance Rate",
                marker_color='green',
                showlegend=False
            ),
            row=1, col=2
        )
        
        # 3. Excess returns over time (time series)
        colors = px.colors.qualitative.Set1
        for i, ticker in enumerate(df['ticker'].unique()):
            ticker_data = df[df['ticker'] == ticker].sort_values('date')
            fig.add_trace(
                go.Scatter(
                    x=ticker_data['date'],
                    y=ticker_data['excess_return'].cumsum() * 100,
                    name=f"{ticker} Cumulative Excess",
                    line=dict(color=colors[i % len(colors)]),
                    showlegend=(i < 5)  # Limit legend entries
                ),
                row=2, col=1
            )
        
        # 4. Risk-Return scatter
        risk_return_data = df.groupby('ticker').agg({
            'stock_return': ['mean', 'std'],
            'outperforms': 'mean'
        }).round(4)
        
        risk_return_data.columns = ['mean_return', 'volatility', 'outperf_rate']
        
        fig.add_trace(
            go.Scatter(
                x=risk_return_data['volatility'] * 100,
                y=risk_return_data['mean_return'] * 100,
                mode='markers+text',
                text=risk_return_data.index,
                textposition="top center",
                marker=dict(
                    size=risk_return_data['outperf_rate'] * 100,
                    color=risk_return_data['outperf_rate'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Outperf %")
                ),
                name="Risk-Return",
                showlegend=False
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=f"Financial Returns Analysis ({prediction_horizon}-Day Horizon)",
            height=800,
            showlegend=True
        )
        
        # Update axes
        fig.update_yaxes(title_text="Return (%)", row=1, col=1)
        fig.update_yaxes(title_text="Outperformance (%)", row=1, col=2)
        fig.update_yaxes(title_text="Cumulative Excess Return (%)", row=2, col=1)
        fig.update_yaxes(title_text="Mean Return (%)", row=2, col=2)
        fig.update_xaxes(title_text="Stock", row=1, col=1)
        fig.update_xaxes(title_text="Stock", row=1, col=2)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_xaxes(title_text="Volatility (%)", row=2, col=2)
        
        return fig
    
    def plot_feature_correlation_matrix(
        self,
        features_df: pd.DataFrame,
        target_col: str = 'outperforms',
        top_n: int = 20
    ) -> go.Figure:
        """
        Create correlation matrix heatmap for features.
        
        Args:
            features_df: DataFrame with features and target
            target_col: Target column name
            top_n: Number of top correlated features to show
            
        Returns:
            Plotly Figure object
        """
        # Select numeric columns only
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        
        if target_col not in numeric_cols:
            return go.Figure().add_annotation(text=f"Target column '{target_col}' not found", x=0.5, y=0.5)
        
        # Calculate correlation with target
        correlations = features_df[numeric_cols].corr()[target_col].abs().sort_values(ascending=False)
        
        # Select top N features (excluding target itself)
        top_features = correlations.drop(target_col, errors='ignore').head(top_n).index.tolist()
        top_features.append(target_col)  # Add target back
        
        # Create correlation matrix for selected features
        corr_matrix = features_df[top_features].corr()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.round(3).values,
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title=f"Feature Correlation Matrix (Top {top_n} vs {target_col})",
            width=800,
            height=800,
            xaxis_showgrid=False,
            yaxis_showgrid=False,
            xaxis_zeroline=False,
            yaxis_zeroline=False
        )
        
        return fig
    
    def plot_volatility_analysis(
        self,
        stock_data: Dict[str, pd.DataFrame],
        benchmark_data: pd.DataFrame,
        window: int = 20
    ) -> go.Figure:
        """
        Analyze volatility patterns across stocks.
        
        Args:
            stock_data: Dictionary mapping ticker to DataFrame
            benchmark_data: Benchmark data
            window: Rolling window for volatility calculation
            
        Returns:
            Plotly Figure object
        """
        # Calculate rolling volatility
        volatility_data = {}
        
        # Benchmark volatility
        bench_returns = benchmark_data['Close'].pct_change()
        bench_vol = bench_returns.rolling(window=window).std() * np.sqrt(252) * 100  # Annualized %
        volatility_data['S&P 500'] = bench_vol.dropna()
        
        # Stock volatilities
        for ticker, data in list(stock_data.items())[:10]:  # Limit to 10 stocks
            returns = data['Close'].pct_change()
            vol = returns.rolling(window=window).std() * np.sqrt(252) * 100  # Annualized %
            volatility_data[ticker] = vol.dropna()
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=(
                f"Rolling Volatility ({window}-Day Window)",
                "Average Volatility by Stock"
            ),
            row_heights=[0.7, 0.3]
        )
        
        # Time series of volatility
        colors = px.colors.qualitative.Set1
        for i, (ticker, vol_series) in enumerate(volatility_data.items()):
            color = 'black' if ticker == 'S&P 500' else colors[i % len(colors)]
            width = 3 if ticker == 'S&P 500' else 1
            
            fig.add_trace(
                go.Scatter(
                    x=vol_series.index,
                    y=vol_series,
                    name=ticker,
                    line=dict(color=color, width=width)
                ),
                row=1, col=1
            )
        
        # Average volatility bar chart
        avg_vols = {ticker: vol.mean() for ticker, vol in volatility_data.items()}
        sorted_vols = sorted(avg_vols.items(), key=lambda x: x[1], reverse=True)
        
        fig.add_trace(
            go.Bar(
                x=[ticker for ticker, _ in sorted_vols],
                y=[vol for _, vol in sorted_vols],
                marker_color=['red' if ticker == 'S&P 500' else 'blue' for ticker, _ in sorted_vols],
                showlegend=False
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title="Volatility Analysis",
            height=800,
            showlegend=True
        )
        
        fig.update_yaxes(title_text="Volatility (%)", row=1, col=1)
        fig.update_yaxes(title_text="Avg Volatility (%)", row=2, col=1)
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_xaxes(title_text="Stock", row=2, col=1)
        
        return fig
    
    def plot_sector_performance(
        self,
        stock_data: Dict[str, pd.DataFrame],
        benchmark_data: pd.DataFrame
    ) -> go.Figure:
        """
        Analyze performance by sector (simplified sector mapping).
        
        Args:
            stock_data: Dictionary mapping ticker to DataFrame
            benchmark_data: Benchmark data
            
        Returns:
            Plotly Figure object
        """
        # Simplified sector mapping (subset of stocks)
        sector_mapping = {
            'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology', 'GOOG': 'Technology',
            'META': 'Technology', 'NVDA': 'Technology', 'AMZN': 'Technology', 'TSLA': 'Technology',
            'JPM': 'Financial', 'BAC': 'Financial', 'WFC': 'Financial', 'GS': 'Financial',
            'UNH': 'Healthcare', 'JNJ': 'Healthcare', 'PFE': 'Healthcare', 'ABT': 'Healthcare',
            'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy', 'EOG': 'Energy',
            'PG': 'Consumer Goods', 'KO': 'Consumer Goods', 'WMT': 'Consumer Goods', 'HD': 'Consumer Goods'
        }
        
        # Calculate sector performance
        sector_performance = {}
        
        for ticker, data in stock_data.items():
            sector = sector_mapping.get(ticker, 'Other')
            
            if len(data) < 100:  # Need sufficient data
                continue
            
            # Calculate total return over available period
            total_return = (data['Close'].iloc[-1] / data['Close'].iloc[0] - 1) * 100
            
            # Calculate volatility
            returns = data['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252) * 100
            
            # Calculate max drawdown
            cumulative = (1 + returns).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdown = (cumulative / rolling_max - 1) * 100
            max_drawdown = drawdown.min()
            
            if sector not in sector_performance:
                sector_performance[sector] = []
            
            sector_performance[sector].append({
                'ticker': ticker,
                'total_return': total_return,
                'volatility': volatility,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': total_return / volatility if volatility > 0 else 0
            })
        
        if not sector_performance:
            return go.Figure().add_annotation(text="No sector data available", x=0.5, y=0.5)
        
        # Aggregate sector statistics
        sector_stats = {}
        for sector, stocks in sector_performance.items():
            df = pd.DataFrame(stocks)
            sector_stats[sector] = {
                'avg_return': df['total_return'].mean(),
                'avg_volatility': df['volatility'].mean(),
                'avg_sharpe': df['sharpe_ratio'].mean(),
                'count': len(stocks)
            }
        
        # Create visualization
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Average Return by Sector",
                "Risk-Return by Sector",
                "Sharpe Ratio by Sector",
                "Stock Count by Sector"
            )
        )
        
        sectors = list(sector_stats.keys())
        returns = [sector_stats[s]['avg_return'] for s in sectors]
        volatilities = [sector_stats[s]['avg_volatility'] for s in sectors]
        sharpe_ratios = [sector_stats[s]['avg_sharpe'] for s in sectors]
        counts = [sector_stats[s]['count'] for s in sectors]
        
        # Average return bar chart
        fig.add_trace(
            go.Bar(x=sectors, y=returns, name="Avg Return", marker_color='green'),
            row=1, col=1
        )
        
        # Risk-return scatter
        fig.add_trace(
            go.Scatter(
                x=volatilities, y=returns,
                mode='markers+text',
                text=sectors,
                textposition="top center",
                marker=dict(size=15, color='blue'),
                name="Risk-Return",
                showlegend=False
            ),
            row=1, col=2
        )
        
        # Sharpe ratio
        fig.add_trace(
            go.Bar(x=sectors, y=sharpe_ratios, name="Sharpe Ratio", marker_color='orange'),
            row=2, col=1
        )
        
        # Stock count
        fig.add_trace(
            go.Bar(x=sectors, y=counts, name="Stock Count", marker_color='purple'),
            row=2, col=2
        )
        
        fig.update_layout(
            title="Sector Performance Analysis",
            height=800,
            showlegend=False
        )
        
        # Update axes
        fig.update_yaxes(title_text="Return (%)", row=1, col=1)
        fig.update_yaxes(title_text="Return (%)", row=1, col=2)
        fig.update_yaxes(title_text="Sharpe Ratio", row=2, col=1)
        fig.update_yaxes(title_text="Count", row=2, col=2)
        fig.update_xaxes(title_text="Sector", row=1, col=1)
        fig.update_xaxes(title_text="Volatility (%)", row=1, col=2)
        fig.update_xaxes(title_text="Sector", row=2, col=1)
        fig.update_xaxes(title_text="Sector", row=2, col=2)
        
        return fig
