"""
Feature configuration for FINQ Stock Predictor.
Contains all feature selection and parameter settings.
"""

# Feature Selection Configuration
FEATURE_SELECTION = {
    # Technical Indicators Features 
    "sma_features": True,  # Simple Moving Averages and ratios - ESSENTIAL
    "ema_features": False,  # Exponential Moving Averages and ratios
    "rsi_features": True,  # Relative Strength Index - ESSENTIAL
    "bb_features": False,   # Bollinger Bands (upper, lower, middle, width, position)
    "macd_features": False, # MACD (line, signal, histogram)
    "atr_features": False,  # Average True Range and ratio
    "volume_features": True, # Volume SMA and ratio - ESSENTIAL
    "obv_features": False,  # On-Balance Volume
    "stoch_features": False, # Stochastic Oscillator (K and D)
    "cci_features": False,  # Commodity Channel Index
    "williams_r_features": False, # Williams %R
    
    # Price-Based Features - KEEP ALL BASIC ONES
    "basic_price_features": True,  # OHLCV data - ESSENTIAL
    "price_ratio_features": True,  # high/low, close/open, high/close, low/close ratios - ESSENTIAL
    "return_features": True,       # daily, overnight, intraday returns - ESSENTIAL
    "volatility_features": True,   # high-low spread, body size, shadows - ESSENTIAL
    "multi_day_returns": True,     # 2d, 3d, 5d, 10d, 20d returns and volatility - ESSENTIAL
    
    # Advanced Features
    "momentum_features": True,     # Price momentum indicators (ROC, acceleration, relative strength)
    "trend_features": True,        # Trend-following indicators (MA trends, ADX-like)
    "mean_reversion_features": True, # Mean reversion indicators (distance from MA, z-scores)
    "market_regime_features": True, # Market regime indicators (volatility regimes, correlations)
    
    # Benchmark/Relative Features - KEEP ALL - VERY IMPORTANT FOR STOCK PREDICTION
    "benchmark_features": True,    # Relative performance vs benchmark - ESSENTIAL
    "benchmark_correlation": True, # Correlation with benchmark - ESSENTIAL
    "benchmark_beta": True,        # Beta coefficient with benchmark - ESSENTIAL
    
    # Feature Categories not yet implemented in engineer.py
    "seasonality_features": False,  # Day of week, month effects (not implemented)
    "interaction_features": False,  # Feature interactions (partially implemented)
}

# Feature Parameters Configuration
FEATURE_PARAMS = {
    # Technical Indicators (existing)
    "sma_periods": [5, 10, 20, 50],
    "ema_periods": [12, 26],
    "rsi_period": 14,
    "bb_period": 20,
    "bb_std": 2,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "atr_period": 14,
    "volume_sma_period": 20,
    
    # Price-based feature parameters
    "return_periods": [2, 3, 5, 10, 20],  # Multi-day return periods
    "volatility_periods": [5, 10, 20],    # Volatility calculation periods
    
    # Momentum parameters
    "momentum_periods": [5, 10, 20],      # Price momentum periods
    "roc_periods": [10, 20],              # Rate of change periods
    
    # Trend parameters
    "trend_periods": [10, 20, 50],        # Trend detection periods
    "adx_period": 14,                     # ADX period
    
    # Mean reversion parameters
    "mean_reversion_periods": [10, 20],   # Mean reversion periods
    "zscore_periods": [20, 50],           # Z-score periods
    
    # Benchmark parameters
    "benchmark_periods": [5, 10, 20],     # Benchmark comparison periods
    "correlation_periods": [20, 50],      # Correlation calculation periods
    "beta_periods": [50, 100],            # Beta calculation periods
    
    # Market regime parameters
    "regime_periods": [20, 50],           # Market regime detection periods
    "volatility_regime_periods": [20],    # Volatility regime periods
}
