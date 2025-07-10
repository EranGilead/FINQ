"""
Feature engineering module for FINQ Stock Predictor.
Generates technical indicators and advanced features from OHLCV data.
"""

import pandas as pd
import numpy as np
import ta
from typing import Dict, List, Optional
from loguru import logger
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import specialized feature configurations
from config.features import FEATURE_PARAMS, FEATURE_SELECTION


class FeatureEngineer:
    """
    Handles feature engineering for stock prediction.
    """
    
    def __init__(self):
        """Initialize FeatureEngineer."""
        logger.info("FeatureEngineer initialized")
    
    def create_technical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create technical indicators from OHLCV data.
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            DataFrame with technical features
        """
        logger.debug("Creating technical features")
        
        features = pd.DataFrame(index=data.index)
        
        # Price and volume data
        high = data['High']
        low = data['Low']
        close = data['Close']
        volume = data['Volume']
        
        # Moving averages
        if FEATURE_SELECTION.get('sma_features', True):
            for period in FEATURE_PARAMS['sma_periods']:
                features[f'sma_{period}'] = ta.trend.SMAIndicator(close, window=period).sma_indicator()
                features[f'close_sma_{period}_ratio'] = close / features[f'sma_{period}']
        
        # Exponential moving averages
        if FEATURE_SELECTION.get('ema_features', True):
            for period in FEATURE_PARAMS['ema_periods']:
                features[f'ema_{period}'] = ta.trend.EMAIndicator(close, window=period).ema_indicator()
                features[f'close_ema_{period}_ratio'] = close / features[f'ema_{period}']
        
        # RSI
        if FEATURE_SELECTION.get('rsi_features', True):
            rsi_period = FEATURE_PARAMS['rsi_period']
            features['rsi'] = ta.momentum.RSIIndicator(close, window=rsi_period).rsi()
        
        # Bollinger Bands
        if FEATURE_SELECTION.get('bb_features', True):
            bb_period = FEATURE_PARAMS['bb_period']
            bb_std = FEATURE_PARAMS['bb_std']
            bb_indicator = ta.volatility.BollingerBands(close, window=bb_period, window_dev=bb_std)
            features['bb_upper'] = bb_indicator.bollinger_hband()
            features['bb_lower'] = bb_indicator.bollinger_lband()
            features['bb_middle'] = bb_indicator.bollinger_mavg()
            features['bb_width'] = (features['bb_upper'] - features['bb_lower']) / features['bb_middle']
            features['bb_position'] = (close - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
        
        # MACD
        if FEATURE_SELECTION.get('macd_features', True):
            macd_fast = FEATURE_PARAMS['macd_fast']
            macd_slow = FEATURE_PARAMS['macd_slow']
            macd_signal = FEATURE_PARAMS['macd_signal']
            macd_indicator = ta.trend.MACD(close, window_fast=macd_fast, window_slow=macd_slow, window_sign=macd_signal)
            features['macd'] = macd_indicator.macd()
            features['macd_signal'] = macd_indicator.macd_signal()
            features['macd_histogram'] = macd_indicator.macd_diff()
        
        # Average True Range (ATR)
        if FEATURE_SELECTION.get('atr_features', True):
            atr_period = FEATURE_PARAMS['atr_period']
            features['atr'] = ta.volatility.AverageTrueRange(high, low, close, window=atr_period).average_true_range()
            features['atr_ratio'] = features['atr'] / close
        
        # Volume indicators
        if FEATURE_SELECTION.get('volume_features', True):
            volume_sma_period = FEATURE_PARAMS['volume_sma_period']
            features['volume_sma'] = volume.rolling(window=volume_sma_period).mean()
            features['volume_ratio'] = volume / features['volume_sma']
        
        # On-Balance Volume
        if FEATURE_SELECTION.get('obv_features', True):
            features['obv'] = ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()
        
        # Stochastic oscillator
        if FEATURE_SELECTION.get('stoch_features', True):
            features['stoch_k'] = ta.momentum.StochasticOscillator(high, low, close).stoch()
            features['stoch_d'] = ta.momentum.StochasticOscillator(high, low, close).stoch_signal()
        
        # Commodity Channel Index
        if FEATURE_SELECTION.get('cci_features', True):
            features['cci'] = ta.trend.CCIIndicator(high, low, close).cci()
        
        # Williams %R
        if FEATURE_SELECTION.get('williams_r_features', True):
            features['williams_r'] = ta.momentum.WilliamsRIndicator(high, low, close).williams_r()
        
        logger.debug("Created {} technical features", len(features.columns))
        return features
    
    def create_price_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create price-based features.
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            DataFrame with price features
        """
        logger.debug("Creating price features")
        
        features = pd.DataFrame(index=data.index)
        
        # Basic price features
        if FEATURE_SELECTION.get('basic_price_features', True):
            features['close'] = data['Close']
            features['open'] = data['Open']
            features['high'] = data['High']
            features['low'] = data['Low']
            features['volume'] = data['Volume']
        
        # Price relationships
        if FEATURE_SELECTION.get('price_ratio_features', True):
            features['high_low_ratio'] = data['High'] / data['Low']
            features['close_open_ratio'] = data['Close'] / data['Open']
            features['high_close_ratio'] = data['High'] / data['Close']
            features['low_close_ratio'] = data['Low'] / data['Close']
        
        # Intraday features
        if FEATURE_SELECTION.get('return_features', True):
            features['daily_return'] = data['Close'].pct_change()
            features['overnight_return'] = data['Open'] / data['Close'].shift(1) - 1
            features['intraday_return'] = data['Close'] / data['Open'] - 1
        
        # Price volatility
        if FEATURE_SELECTION.get('volatility_features', True):
            features['high_low_spread'] = (data['High'] - data['Low']) / data['Close']
            features['body_size'] = abs(data['Close'] - data['Open']) / data['Close']
            features['upper_shadow'] = (data['High'] - np.maximum(data['Open'], data['Close'])) / data['Close']
            features['lower_shadow'] = (np.minimum(data['Open'], data['Close']) - data['Low']) / data['Close']
        
        # Multi-day returns
        if FEATURE_SELECTION.get('multi_day_returns', True):
            for period in FEATURE_PARAMS['return_periods']:
                features[f'return_{period}d'] = data['Close'].pct_change(periods=period)
                features[f'volatility_{period}d'] = features['daily_return'].rolling(window=period).std()
        
        logger.debug("Created {} price features", len(features.columns))
        return features
    
    def create_volume_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create volume-based features.
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            DataFrame with volume features
        """
        logger.debug("Creating volume features")
        
        features = pd.DataFrame(index=data.index)
        
        # Only create volume features if enabled
        if not FEATURE_SELECTION.get('volume_features', True):
            return features
        
        # Basic volume features
        features['volume'] = data['Volume']
        features['volume_change'] = data['Volume'].pct_change()
        
        # Volume moving averages
        for period in [5, 10, 20, 50]:
            features[f'volume_ma_{period}'] = data['Volume'].rolling(window=period).mean()
            features[f'volume_ratio_{period}'] = data['Volume'] / features[f'volume_ma_{period}']
        
        # Volume-price features
        features['volume_price_trend'] = ta.volume.VolumePriceTrendIndicator(data['Close'], data['Volume']).volume_price_trend()
        features['negative_volume_index'] = ta.volume.NegativeVolumeIndexIndicator(data['Close'], data['Volume']).negative_volume_index()
        
        # Price-volume correlation
        for period in [5, 10, 20]:
            price_changes = data['Close'].pct_change()
            volume_changes = data['Volume'].pct_change()
            features[f'price_volume_corr_{period}'] = price_changes.rolling(window=period).corr(volume_changes)
        
        logger.debug("Created {} volume features", len(features.columns))
        return features
    
    def create_momentum_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create momentum-based features.
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            DataFrame with momentum features
        """
        logger.debug("Creating momentum features")
        
        features = pd.DataFrame(index=data.index)
        
        # Only create momentum features if enabled
        if not FEATURE_SELECTION.get('momentum_features', True):
            return features
        
        close = data['Close']
        
        # Rate of change
        for period in FEATURE_PARAMS['roc_periods']:
            features[f'roc_{period}'] = ta.momentum.ROCIndicator(close, window=period).roc()
        
        # Price momentum
        for period in FEATURE_PARAMS['momentum_periods']:
            features[f'momentum_{period}'] = close / close.shift(period) - 1
        
        # Acceleration
        for period in [5, 10]:
            returns = close.pct_change(periods=period)
            features[f'acceleration_{period}'] = returns - returns.shift(period)
        
        # Relative strength compared to moving averages
        for period in FEATURE_PARAMS['trend_periods']:
            sma = close.rolling(window=period).mean()
            features[f'rs_sma_{period}'] = (close - sma) / sma
        
        logger.debug("Created {} momentum features", len(features.columns))
        return features
    
    def create_volatility_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create volatility-based features.
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            DataFrame with volatility features
        """
        logger.debug("Creating volatility features")
        
        features = pd.DataFrame(index=data.index)
        
        # Only create volatility features if enabled
        if not FEATURE_SELECTION.get('volatility_features', True):
            return features
        
        # Basic volatility
        returns = data['Close'].pct_change()
        
        for period in FEATURE_PARAMS['volatility_periods']:
            features[f'volatility_{period}'] = returns.rolling(window=period).std()
            features[f'volatility_rank_{period}'] = features[f'volatility_{period}'].rolling(window=period*2).rank() / (period*2)
        
        # Volatility ratios
        if len(FEATURE_PARAMS['volatility_periods']) >= 2:
            short_vol = min(FEATURE_PARAMS['volatility_periods'])
            long_vol = max(FEATURE_PARAMS['volatility_periods'])
            features[f'vol_ratio_{short_vol}_{long_vol}'] = features[f'volatility_{short_vol}'] / features[f'volatility_{long_vol}']
        
        # Realized volatility
        features['realized_vol'] = np.sqrt(252) * returns.rolling(window=20).std()
        
        # Volatility clustering
        features['vol_clustering'] = features['volatility_20'].rolling(window=5).std()
        
        logger.debug("Created {} volatility features", len(features.columns))
        return features
    
    def create_trend_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create trend-based features.
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            DataFrame with trend features
        """
        logger.debug("Creating trend features")
        
        features = pd.DataFrame(index=data.index)
        
        # Only create trend features if enabled
        if not FEATURE_SELECTION.get('trend_features', True):
            return features
        
        close = data['Close']
        
        # Moving average trends
        for period in FEATURE_PARAMS['trend_periods']:
            ma = close.rolling(window=period).mean()
            features[f'ma_trend_{period}'] = (ma - ma.shift(5)) / ma.shift(5)
        
        # Price position relative to moving averages
        for period in FEATURE_PARAMS['trend_periods']:
            ma = close.rolling(window=period).mean()
            features[f'price_ma_position_{period}'] = (close - ma) / ma
        
        # Trend strength
        for period in [10, 20]:
            slope = (close - close.shift(period)) / period
            features[f'trend_strength_{period}'] = slope / close.rolling(window=period).std()
        
        # Moving average convergence/divergence
        trend_periods = FEATURE_PARAMS['trend_periods']
        if len(trend_periods) >= 2:
            short_period = min(trend_periods)
            long_period = max(trend_periods)
            features[f'ma_convergence_{short_period}_{long_period}'] = (close.rolling(short_period).mean() - close.rolling(long_period).mean()) / close.rolling(long_period).mean()
            
            # Add additional convergence if we have a mid-range period
            if len(trend_periods) >= 3:
                mid_period = sorted(trend_periods)[1]
                features[f'ma_convergence_{mid_period}_{long_period}'] = (close.rolling(mid_period).mean() - close.rolling(long_period).mean()) / close.rolling(long_period).mean()
        
        logger.debug("Created {} trend features", len(features.columns))
        return features
    
    def create_advanced_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create advanced features with higher predictive power.
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            DataFrame with advanced features
        """
        logger.debug("Creating advanced features")
        
        features = pd.DataFrame(index=data.index)
        
        close = data['Close']
        high = data['High']
        low = data['Low']
        volume = data['Volume']
        
        # 1. CROSS-TIMEFRAME FEATURES (Multi-horizon signals)
        if FEATURE_SELECTION.get('trend_features', True):
            # Momentum across different timeframes
            for fast, slow in [(5, 10), (10, 20), (20, 50)]:
                sma_fast = close.rolling(fast).mean()
                sma_slow = close.rolling(slow).mean()
                features[f'momentum_cross_{fast}_{slow}'] = (sma_fast > sma_slow).astype(int)
                features[f'momentum_strength_{fast}_{slow}'] = (sma_fast - sma_slow) / sma_slow
        
        # 2. VOLATILITY REGIME FEATURES (Market conditions)
        if FEATURE_SELECTION.get('market_regime_features', True):
            # Historical volatility across timeframes
            for window in FEATURE_PARAMS['regime_periods']:
                returns = close.pct_change()
                vol = returns.rolling(window).std() * np.sqrt(252) # Annualized
                features[f'volatility_{window}d'] = vol
                features[f'vol_regime_{window}d'] = (vol > vol.rolling(252).quantile(0.7)).astype(int)
        
        # 3. VOLUME-PRICE RELATIONSHIP FEATURES (Smart money)
        if FEATURE_SELECTION.get('volume_features', True):
            # Price-Volume Trend
            features['pvt'] = ((close - close.shift(1)) / close.shift(1) * volume).cumsum()
            
            # Volume-Weighted Average Price (VWAP) deviation
            vwap = (close * volume).rolling(20).sum() / volume.rolling(20).sum()
            features['vwap_deviation'] = (close - vwap) / vwap
            
            # Volume momentum
            features['volume_momentum'] = volume.pct_change(5)
        
        # 4. MEAN REVERSION FEATURES (Contrarian signals)
        if FEATURE_SELECTION.get('mean_reversion_features', True):
            # Distance from moving averages (normalized)
            for period in FEATURE_PARAMS['mean_reversion_periods']:
                sma = close.rolling(period).mean()
                features[f'mean_reversion_{period}'] = (close - sma) / sma
                # Z-score of price relative to moving average
                features[f'price_zscore_{period}'] = (close - sma) / close.rolling(period).std()
        
        # 5. TREND STRENGTH FEATURES (Trend quality)
        if FEATURE_SELECTION.get('trend_features', True):
            # ADX-like trend strength
            for period in [14, 28]:
                tr = np.maximum(high - low, 
                               np.maximum(abs(high - close.shift(1)), 
                                        abs(low - close.shift(1))))
                atr = tr.rolling(period).mean()
                dm_plus = np.where((high - high.shift(1)) > (low.shift(1) - low), 
                                  np.maximum(high - high.shift(1), 0), 0)
                dm_minus = np.where((low.shift(1) - low) > (high - high.shift(1)), 
                                   np.maximum(low.shift(1) - low, 0), 0)
                
                di_plus = 100 * (pd.Series(dm_plus).rolling(period).mean() / atr)
                di_minus = 100 * (pd.Series(dm_minus).rolling(period).mean() / atr)
                dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
                adx = dx.rolling(period).mean()
                
                features[f'trend_strength_{period}'] = adx
        
        # 6. REGIME CHANGE FEATURES (Market shifts)
        if FEATURE_SELECTION.get('market_regime_features', True):
            # Volatility regime changes
            vol_short = close.pct_change().rolling(10).std()
            vol_long = close.pct_change().rolling(50).std()
            features['vol_regime_change'] = vol_short / vol_long
            
            # Correlation with market (using SPY-like behavior)
            market_proxy = close.pct_change().rolling(60).corr(close.pct_change())
            features['market_correlation'] = market_proxy
        
        logger.debug("Created {} advanced features", len(features.columns))
        return features

    def create_interaction_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between existing features.
        
        Args:
            features: Existing features DataFrame
            
        Returns:
            DataFrame with interaction features
        """
        logger.debug("Creating interaction features")
        
        interaction_features = pd.DataFrame(index=features.index)
        
        # Select key features for interactions
        key_features = ['rsi', 'volume_ratio', 'close_sma_20_ratio', 'macd', 'bb_position']
        
        # Create pairwise interactions for most important features
        for i, feat1 in enumerate(key_features):
            if feat1 in features.columns:
                for feat2 in key_features[i+1:]:
                    if feat2 in features.columns:
                        # Multiplicative interaction
                        interaction_features[f'{feat1}_x_{feat2}'] = features[feat1] * features[feat2]
                        
                        # Ratio interaction (avoid division by zero)
                        safe_feat2 = features[feat2].replace(0, np.nan)
                        interaction_features[f'{feat1}_div_{feat2}'] = features[feat1] / safe_feat2
        
        # RSI-Volume interaction (strong signal)
        if 'rsi' in features.columns and 'volume_ratio' in features.columns:
            interaction_features['rsi_volume_strength'] = (
                (features['rsi'] > 70) | (features['rsi'] < 30)
            ).astype(int) * features['volume_ratio']
        
        logger.debug("Created {} interaction features", len(interaction_features.columns))
        return interaction_features

    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all features for a given stock.
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            DataFrame with all engineered features
        """
        logger.debug("Engineering features for stock data with {} rows", len(data))
        
        # Generate all feature types
        price_features = self.create_price_features(data)
        technical_features = self.create_technical_features(data)
        volume_features = self.create_volume_features(data)
        momentum_features = self.create_momentum_features(data)
        volatility_features = self.create_volatility_features(data)
        trend_features = self.create_trend_features(data)
        advanced_features = self.create_advanced_features(data)
        seasonality_features = self.create_seasonality_features(data)
        
        # Combine all features
        all_features = pd.concat([
            price_features,
            technical_features,
            volume_features,
            momentum_features,
            volatility_features,
            trend_features,
            advanced_features,
            seasonality_features
        ], axis=1)
        
        # Remove duplicate columns
        all_features = all_features.loc[:, ~all_features.columns.duplicated()]
        
        # Create interaction features if enabled
        if FEATURE_SELECTION.get('interaction_features', False):
            interaction_features = self.create_interaction_features(all_features)
            all_features = pd.concat([all_features, interaction_features], axis=1)
            all_features = all_features.loc[:, ~all_features.columns.duplicated()]
        
        # Only backward fill missing values to prevent lookahead bias
        all_features = all_features.bfill()
        
        # drop features that are mostly NaN (>50% missing) or use median imputation
        nan_ratio = all_features.isnull().sum() / len(all_features)
        high_nan_features = nan_ratio[nan_ratio > 0.5].index
        
        if len(high_nan_features) > 0:
            logger.warning("Dropping {} features with >50% NaN values: {}", 
                          len(high_nan_features), list(high_nan_features))
            all_features = all_features.drop(columns=high_nan_features)
        
        # For any remaining NaNs, use median imputation as a last resort
        remaining_nans = all_features.isnull().sum().sum()
        if remaining_nans > 0:
            logger.warning("Using median imputation for {} remaining NaN values", remaining_nans)
            numeric_cols = all_features.select_dtypes(include=[np.number]).columns
            all_features[numeric_cols] = all_features[numeric_cols].fillna(all_features[numeric_cols].median())
        
        logger.debug("Generated {} features", len(all_features.columns))
        return all_features
    
    def engineer_features_multiple_stocks(
        self, 
        stock_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """
        Engineer features for multiple stocks.
        
        Args:
            stock_data: Dictionary mapping ticker to DataFrame
            
        Returns:
            Dictionary mapping ticker to engineered features
        """
        logger.info("Engineering features for {} stocks", len(stock_data))
        
        engineered_features = {}
        
        for ticker, data in stock_data.items():
            try:
                features = self.engineer_features(data)
                features['ticker'] = ticker
                engineered_features[ticker] = features
                logger.debug("Engineered {} features for {}", len(features.columns), ticker)
            except Exception as e:
                logger.warning("Failed to engineer features for {}: {}", ticker, str(e))
                continue
        
        logger.info("Successfully engineered features for {} stocks", len(engineered_features))
        return engineered_features
    
    def select_features(
        self, 
        features: pd.DataFrame,
        importance_threshold: float = 0.01,
        correlation_threshold: float = 0.95
    ) -> pd.DataFrame:
        """
        Select features based on importance and correlation.
        
        Args:
            features: Feature DataFrame
            importance_threshold: Minimum feature importance threshold
            correlation_threshold: Maximum correlation threshold
            
        Returns:
            DataFrame with selected features
        """
        logger.debug("Selecting features with importance_threshold={}, correlation_threshold={}", 
                    importance_threshold, correlation_threshold)
        
        # Remove features with too many missing values
        missing_ratio = features.isnull().sum() / len(features)
        valid_features = features.loc[:, missing_ratio < 0.5]
        
        # Remove highly correlated features
        corr_matrix = valid_features.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > correlation_threshold)]
        selected_features = valid_features.drop(columns=to_drop)
        
        logger.debug("Feature selection: {} -> {} features", len(features.columns), len(selected_features.columns))
        return selected_features
    
    def select_best_features(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        method: str = 'mutual_info',
        n_features: int = 50
    ) -> List[str]:
        """
        Select the best features using various selection methods.
        
        Args:
            features: Feature DataFrame
            labels: Target labels
            method: Selection method ('mutual_info', 'f_score', 'recursive')
            n_features: Number of features to select
            
        Returns:
            List of selected feature names
        """
        from sklearn.feature_selection import (
            mutual_info_classif, f_classif, SelectKBest, RFE
        )
        from sklearn.ensemble import RandomForestClassifier
        
        logger.info("Selecting {} best features using {} method", n_features, method)
        
        # Remove non-numeric columns
        numeric_features = features.select_dtypes(include=[np.number])
        
        # Handle missing values
        numeric_features = numeric_features.fillna(numeric_features.median())
        
        if method == 'mutual_info':
            # Mutual information for non-linear relationships
            scores = mutual_info_classif(numeric_features, labels, random_state=42)
            feature_scores = pd.Series(scores, index=numeric_features.columns)
            
        elif method == 'f_score':
            # F-score for linear relationships
            scores, _ = f_classif(numeric_features, labels)
            feature_scores = pd.Series(scores, index=numeric_features.columns)
            
        elif method == 'recursive':
            # Recursive feature elimination with Random Forest
            rf = RandomForestClassifier(n_estimators=50, random_state=42)
            rfe = RFE(estimator=rf, n_features_to_select=n_features)
            rfe.fit(numeric_features, labels)
            return numeric_features.columns[rfe.support_].tolist()
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Get top features
        selected_features = feature_scores.nlargest(n_features).index.tolist()
        
        logger.info("Selected features (top 10): {}", selected_features[:10])
        return selected_features

    def analyze_feature_importance(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        model_type: str = 'random_forest'
    ) -> pd.DataFrame:
        """
        Analyze feature importance using tree-based models.
        
        Args:
            features: Feature DataFrame
            labels: Target labels
            model_type: Model type for importance analysis
            
        Returns:
            DataFrame with feature importance scores
        """
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.model_selection import cross_val_score
        
        logger.info("Analyzing feature importance using {}", model_type)
        
        # Prepare data
        numeric_features = features.select_dtypes(include=[np.number])
        numeric_features = numeric_features.fillna(numeric_features.median())
        
        # Train model
        if model_type == 'random_forest':
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == 'gradient_boosting':
            model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model.fit(numeric_features, labels)
        
        # Get feature importance
        importance_df = pd.DataFrame({
            'feature': numeric_features.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Add cross-validation score for each feature (optional, computationally expensive)
        logger.info("Top 10 most important features:")
        for _, row in importance_df.head(10).iterrows():
            logger.info("  {}: {:.4f}", row['feature'], row['importance'])
        
        return importance_df
    
    def create_relative_features(
        self, 
        stock_data: pd.DataFrame, 
        benchmark_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Create relative features comparing stock to market index.
        
        Args:
            stock_data: Stock OHLCV DataFrame
            benchmark_data: Benchmark/index OHLCV DataFrame
            
        Returns:
            DataFrame with relative features
        """
        logger.debug("Creating relative features vs benchmark")
        
        features = pd.DataFrame(index=stock_data.index)
        
        # Align benchmark data with stock data dates
        aligned_benchmark = benchmark_data.reindex(stock_data.index, method='ffill')
        
        stock_close = stock_data['Close']
        benchmark_close = aligned_benchmark['Close']
        stock_volume = stock_data['Volume']
        benchmark_volume = aligned_benchmark['Volume']
        
        # Price ratios
        features['stock_index_ratio'] = stock_close / benchmark_close
        features['stock_index_ratio_change'] = features['stock_index_ratio'].pct_change()
        
        # Relative returns
        stock_returns = stock_close.pct_change()
        benchmark_returns = benchmark_close.pct_change()
        features['excess_return_1d'] = stock_returns - benchmark_returns
        
        # Multi-period relative returns
        for period in [5, 10, 20, 60]:
            stock_ret_period = stock_close.pct_change(period)
            benchmark_ret_period = benchmark_close.pct_change(period)
            features[f'excess_return_{period}d'] = stock_ret_period - benchmark_ret_period
            features[f'relative_return_{period}d'] = stock_ret_period / benchmark_ret_period - 1
        
        # Relative volatility
        for period in [20, 60]:
            stock_vol = stock_returns.rolling(period).std()
            benchmark_vol = benchmark_returns.rolling(period).std()
            features[f'relative_volatility_{period}d'] = stock_vol / benchmark_vol
        
        # Beta calculation (rolling)
        for period in [60, 120, 252]:
            covariance = stock_returns.rolling(period).cov(benchmark_returns)
            benchmark_var = benchmark_returns.rolling(period).var()
            features[f'beta_{period}d'] = covariance / benchmark_var
        
        # Correlation with market
        for period in [30, 60, 120]:
            features[f'correlation_{period}d'] = stock_returns.rolling(period).corr(benchmark_returns)
        
        # Relative strength index vs market
        for period in [20, 50]:
            stock_rs = stock_close / stock_close.rolling(period).mean()
            benchmark_rs = benchmark_close / benchmark_close.rolling(period).mean()
            features[f'relative_strength_{period}d'] = stock_rs / benchmark_rs
        
        # Enhanced volume analysis relative to index composition
        
        # 1. BASIC VOLUME RATIOS
        features['volume_ratio_vs_index'] = stock_volume / benchmark_volume
        
        # 2. MARKET CAP & INDEX WEIGHT PROXIES
        # Use price ratio as a proxy for relative market cap influence
        # (This is a simplification - in reality we'd use actual market cap data)
        price_ratio = stock_close / benchmark_close
        
        # Estimate relative weight in index (rough approximation)
        rolling_price_ratio = price_ratio.rolling(window=252).mean()  # Annual average
        features['estimated_index_weight'] = rolling_price_ratio
        features['weight_adjusted_volume'] = stock_volume * rolling_price_ratio
        
        # 3. VOLUME LEADERSHIP & PARTICIPATION
        # Volume participation rate (stock volume relative to its own average vs index)
        stock_vol_avg = stock_volume.rolling(20).mean()
        index_vol_avg = benchmark_volume.rolling(20).mean()
        
        stock_vol_participation = stock_volume / stock_vol_avg
        index_vol_participation = benchmark_volume / index_vol_avg
        
        features['volume_participation_ratio'] = stock_vol_participation / index_vol_participation
        features['volume_leadership'] = (stock_vol_participation > index_vol_participation).astype(int)
        
        # 4. RELATIVE VOLUME INTENSITY
        # When this stock has unusual volume, how does it correlate with index volume?
        for period in [5, 20]:
            # Rolling correlation between volume spikes
            stock_vol_zscore = (stock_volume - stock_volume.rolling(period).mean()) / stock_volume.rolling(period).std()
            index_vol_zscore = (benchmark_volume - benchmark_volume.rolling(period).mean()) / benchmark_volume.rolling(period).std()
            
            features[f'volume_spike_correlation_{period}d'] = stock_vol_zscore.rolling(period).corr(index_vol_zscore)
            
            # Is this stock driving index volume or following it?
            features[f'volume_leadership_strength_{period}d'] = stock_vol_zscore - index_vol_zscore
        
        # 5. VOLUME-WEIGHTED RELATIVE PERFORMANCE
        # Does this stock outperform when it has high relative volume?
        volume_weighted_returns = stock_returns * (stock_volume / stock_volume.rolling(20).mean())
        index_volume_weighted_returns = benchmark_returns * (benchmark_volume / benchmark_volume.rolling(20).mean())
        
        features['volume_weighted_excess_return'] = volume_weighted_returns - index_volume_weighted_returns
        
        # 6. BIG STOCK INFLUENCE METRICS
        # For large stocks that can move the index
        # Measure how much this stock's price movement explains index movement
        for period in [20, 60]:
            # When this stock moves with high volume, does it predict index movement?
            high_volume_mask = stock_volume > stock_volume.rolling(period).quantile(0.8)
            
            # Price change correlation during high volume periods
            stock_price_change = stock_returns
            index_price_change = benchmark_returns
            
            # Calculate enhanced correlation during high volume
            volume_weight = (stock_volume / stock_volume.rolling(period).mean()).clip(0, 3)  # Cap at 3x normal
            weighted_stock_returns = stock_price_change * volume_weight
            
            features[f'volume_weighted_correlation_{period}d'] = weighted_stock_returns.rolling(period).corr(index_price_change)
        
        # Relative momentum
        for period in [10, 20, 50]:
            stock_momentum = stock_close / stock_close.shift(period)
            benchmark_momentum = benchmark_close / benchmark_close.shift(period)
            features[f'relative_momentum_{period}d'] = stock_momentum / benchmark_momentum - 1
        
        # Tracking error (rolling standard deviation of excess returns)
        for period in [30, 60]:
            features[f'tracking_error_{period}d'] = features['excess_return_1d'].rolling(period).std()
        
        # Information ratio (excess return / tracking error)
        for period in [60, 120]:
            avg_excess_return = features['excess_return_1d'].rolling(period).mean()
            tracking_error = features['excess_return_1d'].rolling(period).std()
            features[f'information_ratio_{period}d'] = avg_excess_return / tracking_error
        
        # Relative moving averages
        for period in [20, 50]:
            stock_ma = stock_close.rolling(period).mean()
            benchmark_ma = benchmark_close.rolling(period).mean()
            features[f'relative_ma_{period}d'] = (stock_close / stock_ma) / (benchmark_close / benchmark_ma)
        
        # Enhanced volume analysis relative to index composition
        
        # 1. MARKET CAP & INDEX WEIGHT PROXIES
        # Use price ratio as a proxy for relative market cap influence
        # (This is a simplification - in reality we'd use actual market cap data)
        price_ratio = stock_close / benchmark_close
        
        # Estimate relative weight in index (this is a rough approximation)
        # In reality, you'd want actual index weights from S&P data
        rolling_price_ratio = price_ratio.rolling(window=252).mean()  # Annual average
        estimated_index_weight = rolling_price_ratio / rolling_price_ratio.rolling(window=252).sum()
        
        features['estimated_index_weight'] = estimated_index_weight
        features['weight_adjusted_volume'] = stock_volume * estimated_index_weight
        
        # 2. VOLUME LEADERSHIP & PARTICIPATION
        # Volume participation rate (stock volume relative to its own average vs index)
        stock_vol_avg = stock_volume.rolling(20).mean()
        index_vol_avg = benchmark_volume.rolling(20).mean()
        
        stock_vol_participation = stock_volume / stock_vol_avg
        index_vol_participation = benchmark_volume / index_vol_avg
        
        features['volume_participation_ratio'] = stock_vol_participation / index_vol_participation
        features['volume_leadership'] = (stock_vol_participation > index_vol_participation).astype(int)
        
        # 3. RELATIVE VOLUME INTENSITY
        # When this stock has unusual volume, how does it correlate with index volume?
        for period in [5, 20, 60]:
            # Rolling correlation between volume spikes
            stock_vol_zscore = (stock_volume - stock_volume.rolling(period).mean()) / stock_volume.rolling(period).std()
            index_vol_zscore = (benchmark_volume - benchmark_volume.rolling(period).mean()) / benchmark_volume.rolling(period).std()
            
            features[f'volume_spike_correlation_{period}d'] = stock_vol_zscore.rolling(period).corr(index_vol_zscore)
            
            # Is this stock driving index volume or following it?
            features[f'volume_leadership_strength_{period}d'] = stock_vol_zscore - index_vol_zscore
        
        # 4. VOLUME-WEIGHTED RELATIVE PERFORMANCE
        # Does this stock outperform when it has high relative volume?
        volume_weighted_returns = stock_returns * (stock_volume / stock_volume.rolling(20).mean())
        index_volume_weighted_returns = benchmark_returns * (benchmark_volume / benchmark_volume.rolling(20).mean())
        
        features['volume_weighted_excess_return'] = volume_weighted_returns - index_volume_weighted_returns
        
        # 5. BIG STOCK INFLUENCE METRICS
        # For large stocks that can move the index
        # Measure how much this stock's price movement explains index movement
        for period in [20, 60]:
            # Price change correlation weighted by volume
            stock_price_change = stock_close.pct_change()
            index_price_change = benchmark_close.pct_change()
            
            # When this stock moves with high volume, does it predict index movement?
            high_volume_mask = stock_volume > stock_volume.rolling(period).quantile(0.8)
            
            # Calculate conditional correlation during high volume periods
            conditional_corr = stock_price_change[high_volume_mask].rolling(period).corr(
                index_price_change[high_volume_mask]
            )
            features[f'high_volume_price_correlation_{period}d'] = conditional_corr.reindex(stock_close.index)
        
        # 6. INDEX IMPACT ESTIMATION
        # Estimate how much this stock's movement contributes to index movement
        # This is especially important for large-cap stocks
        stock_contribution_to_index = stock_returns * estimated_index_weight
        features['estimated_index_contribution'] = stock_contribution_to_index
        
        # Rolling measure of how much this stock explains index variance
        for period in [30, 90]:
            stock_index_covariance = stock_returns.rolling(period).cov(benchmark_returns)
            index_variance = benchmark_returns.rolling(period).var()
            
            # What % of index movement is explained by this stock?
            features[f'index_variance_explained_{period}d'] = (stock_index_covariance * estimated_index_weight) / index_variance
        
        logger.debug("Created {} relative features", len(features.columns))
        return features

    def engineer_features_with_benchmark(
        self, 
        stock_data: pd.DataFrame, 
        benchmark_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generate all features including relative features against benchmark.
        
        Args:
            stock_data: Stock OHLCV DataFrame
            benchmark_data: Benchmark OHLCV DataFrame
            
        Returns:
            DataFrame with all engineered features including relative features
        """
        logger.debug("Engineering features with benchmark for stock data with {} rows", len(stock_data))
        
        # Generate standard features
        standard_features = self.engineer_features(stock_data)
        
        # Generate relative features
        relative_features = self.create_relative_features(stock_data, benchmark_data)
        
        # Combine all features
        all_features = pd.concat([standard_features, relative_features], axis=1)
        
        # Remove duplicate columns
        all_features = all_features.loc[:, ~all_features.columns.duplicated()]
        
        # Handle NaNs as before
        all_features = all_features.bfill()
        
        # Drop features that are mostly NaN (>50% missing)
        nan_ratio = all_features.isnull().sum() / len(all_features)
        high_nan_features = nan_ratio[nan_ratio > 0.5].index
        
        if len(high_nan_features) > 0:
            logger.warning("Dropping {} features with >50% NaN values: {}", 
                          len(high_nan_features), list(high_nan_features))
            all_features = all_features.drop(columns=high_nan_features)
        
        # For any remaining NaNs, use median imputation
        remaining_nans = all_features.isnull().sum().sum()
        if remaining_nans > 0:
            logger.warning("Using median imputation for {} remaining NaN values", remaining_nans)
            numeric_cols = all_features.select_dtypes(include=[np.number]).columns
            all_features[numeric_cols] = all_features[numeric_cols].fillna(all_features[numeric_cols].median())
        
        logger.debug("Generated {} total features (including {} relative features)", 
                    len(all_features.columns), len(relative_features.columns))
        return all_features

    def engineer_features_multiple_stocks_with_benchmark(
        self,
        stock_data: Dict[str, pd.DataFrame],
        benchmark_data: pd.DataFrame
    ) -> Dict[str, pd.DataFrame]:
        """
        Engineer features for multiple stocks with benchmark comparisons.
        
        Args:
            stock_data: Dictionary mapping ticker to DataFrame
            benchmark_data: Benchmark OHLCV DataFrame
            
        Returns:
            Dictionary mapping ticker to engineered features
        """
        logger.info("Engineering features with benchmark for {} stocks", len(stock_data))
        
        engineered_features = {}
        
        for ticker, data in stock_data.items():
            try:
                features = self.engineer_features_with_benchmark(data, benchmark_data)
                features['ticker'] = ticker
                engineered_features[ticker] = features
                logger.debug("Engineered {} features for {} (including relative features)", 
                           len(features.columns), ticker)
            except Exception as e:
                logger.warning("Failed to engineer features for {}: {}", ticker, str(e))
                continue
        
        logger.info("Successfully engineered features for {} stocks", len(engineered_features))
        return engineered_features
    
    def create_seasonality_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create seasonality-based features.
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            DataFrame with seasonality features
        """
        logger.debug("Creating seasonality features")
        
        features = pd.DataFrame(index=data.index)
        
        # Only create seasonality features if enabled
        if not FEATURE_SELECTION.get('seasonality_features', False):
            return features
        
        # Day of week effects
        features['day_of_week'] = data.index.dayofweek
        features['is_monday'] = (data.index.dayofweek == 0).astype(int)
        features['is_friday'] = (data.index.dayofweek == 4).astype(int)
        
        # Month effects
        features['month'] = data.index.month
        features['is_january'] = (data.index.month == 1).astype(int)
        features['is_december'] = (data.index.month == 12).astype(int)
        
        # Quarter effects
        features['quarter'] = data.index.quarter
        features['is_q1'] = (data.index.quarter == 1).astype(int)
        features['is_q4'] = (data.index.quarter == 4).astype(int)
        
        # Beginning/end of month effects
        features['is_month_start'] = data.index.is_month_start.astype(int)
        features['is_month_end'] = data.index.is_month_end.astype(int)
        
        # Day of month
        features['day_of_month'] = data.index.day
        features['is_first_week'] = (data.index.day <= 7).astype(int)
        features['is_last_week'] = (data.index.day >= 24).astype(int)
        
        # Holiday effects (approximate - you might want to use a holiday calendar)
        # Check if it's near major holidays (simplified)
        features['is_near_holidays'] = (
            (data.index.month == 12) & (data.index.day >= 20) |  # Christmas
            (data.index.month == 1) & (data.index.day <= 5) |   # New Year
            (data.index.month == 11) & (data.index.day >= 20)    # Thanksgiving
        ).astype(int)
        
        logger.debug("Created {} seasonality features", len(features.columns))
        return features
