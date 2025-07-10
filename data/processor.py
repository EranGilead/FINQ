"""
Data processing module for FINQ Stock Predictor.
Handles data preprocessing and label generation.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from loguru import logger
import os
import sys
import glob

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main_config import PREDICTION_HORIZON_DAYS, PROCESSED_DATA_DIR


class DataProcessor:
    """
    Handles data preprocessing and label generation for stock prediction.
    """
    
    def __init__(self):
        """Initialize DataProcessor."""
        logger.info("DataProcessor initialized")
    
    def create_labels(
        self, 
        stock_data: pd.DataFrame, 
        benchmark_data: pd.DataFrame,
        prediction_horizon: int = PREDICTION_HORIZON_DAYS
    ) -> pd.DataFrame:
        """
        Create labels indicating whether stock will outperform benchmark.
        
        Args:
            stock_data: Stock OHLCV data
            benchmark_data: Benchmark OHLCV data
            prediction_horizon: Number of days to predict ahead
            
        Returns:
            DataFrame with labels and returns
        """
        logger.debug("Creating labels with prediction horizon: {} days", prediction_horizon)
        
        # Calculate returns for both stock and benchmark
        stock_returns = self._calculate_forward_returns(stock_data['Close'], prediction_horizon)
        benchmark_returns = self._calculate_forward_returns(benchmark_data['Close'], prediction_horizon)
        
        # Align dates between stock and benchmark
        aligned_stock_returns, aligned_benchmark_returns = self._align_returns(
            stock_returns, benchmark_returns
        )
        
        # Create outperformance labels
        outperformance = aligned_stock_returns > aligned_benchmark_returns
        
        # Create result DataFrame
        result = pd.DataFrame({
            'stock_return': aligned_stock_returns,
            'benchmark_return': aligned_benchmark_returns,
            'excess_return': aligned_stock_returns - aligned_benchmark_returns,
            'outperforms': outperformance.astype(int)
        })
        
        # Remove rows with NaN values (due to forward-looking calculation)
        result = result.dropna()
        
        logger.debug("Created {} labels, positive rate: {:.2%}", 
                    len(result), result['outperforms'].mean())
        
        return result
    
    def prepare_training_data_with_features(
        self,
        stock_data: Dict[str, pd.DataFrame],
        benchmark_data: pd.DataFrame,
        feature_engineer=None,
        prediction_horizon: int = PREDICTION_HORIZON_DAYS
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare training data with full feature engineering including relative features.
        
        Args:
            stock_data: Dictionary mapping ticker to DataFrame
            benchmark_data: Benchmark OHLCV data
            feature_engineer: FeatureEngineer instance for creating features
            prediction_horizon: Number of days to predict ahead
            
        Returns:
            Tuple of (features_df, labels_df)
        """
        logger.info("Preparing training data with full features for {} stocks", len(stock_data))
        
        all_features = []
        all_labels = []
        
        for ticker, data in stock_data.items():
            try:
                # Create labels for this stock
                labels = self.create_labels(data, benchmark_data, prediction_horizon)
                
                # Add ticker information
                labels['ticker'] = ticker
                labels['date'] = labels.index
                
                # Generate full features including relative features if feature_engineer provided
                if feature_engineer:
                    features = feature_engineer.engineer_features_with_benchmark(data, benchmark_data)
                else:
                    features = self._create_basic_features(data)
                
                # Align features with labels
                aligned_features = features.loc[labels.index]
                aligned_features['ticker'] = ticker
                aligned_features['date'] = aligned_features.index
                
                all_features.append(aligned_features)
                all_labels.append(labels)
                
                logger.debug("Processed {} samples for {} with {} features", 
                           len(labels), ticker, len(features.columns))
                
            except Exception as e:
                logger.warning("Failed to process {}: {}", ticker, str(e))
                continue
        
        if not all_features:
            raise ValueError("No valid stock data could be processed")
        
        # Combine all stocks
        combined_features = pd.concat(all_features, ignore_index=True)
        combined_labels = pd.concat(all_labels, ignore_index=True)
        
        logger.info("Prepared training data: {} samples from {} stocks with {} features", 
                   len(combined_features), len(stock_data), len(combined_features.columns))
        
        return combined_features, combined_labels
    
    def create_basic_processed_data(
        self,
        stock_data: Dict[str, pd.DataFrame],
        benchmark_data: pd.DataFrame,
        prediction_horizon: int = PREDICTION_HORIZON_DAYS
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create basic processed data with only labels and basic features.
        This is the first step in the pipeline, separate from feature engineering.
        
        Args:
            stock_data: Dictionary of stock data by ticker
            benchmark_data: Benchmark data
            prediction_horizon: Number of days to predict ahead
            
        Returns:
            Tuple of (basic_features, labels) DataFrames
        """
        logger.info("Creating basic processed data for {} stocks", len(stock_data))
        
        all_features = []
        all_labels = []
        
        for ticker, data in stock_data.items():
            try:
                # Create labels for this stock
                labels = self.create_labels(data, benchmark_data, prediction_horizon)
                
                # Add ticker information
                labels['ticker'] = ticker
                labels['date'] = labels.index
                
                # Create only basic features (no advanced feature engineering)
                features = self._create_basic_features(data)
                
                # Align features with labels
                aligned_features = features.loc[labels.index]
                aligned_features['ticker'] = ticker
                aligned_features['date'] = aligned_features.index
                
                all_features.append(aligned_features)
                all_labels.append(labels)
                
                logger.debug("Processed {} samples for {} with {} basic features", 
                           len(labels), ticker, len(features.columns))
                
            except Exception as e:
                logger.warning("Failed to process {}: {}", ticker, str(e))
                continue
        
        if not all_features:
            raise ValueError("No valid stock data could be processed")
        
        # Combine all stocks
        combined_features = pd.concat(all_features, ignore_index=True)
        combined_labels = pd.concat(all_labels, ignore_index=True)
        
        logger.info("Created basic processed data: {} samples from {} stocks with {} basic features", 
                   len(combined_features), len(stock_data), len(combined_features.columns))
        
        return combined_features, combined_labels
    
    def prepare_training_data(
        self,
        stock_data: Dict[str, pd.DataFrame],
        benchmark_data: pd.DataFrame,
        prediction_horizon: int = PREDICTION_HORIZON_DAYS,
        feature_engineer: Optional[object] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare training data with optional feature engineering.
        
        Args:
            stock_data: Dictionary mapping ticker to DataFrame
            benchmark_data: Benchmark OHLCV data
            prediction_horizon: Number of days to predict ahead
            feature_engineer: FeatureEngineer instance for creating features
            
        Returns:
            Tuple of (features_df, labels_df)
        """
        logger.info("Preparing training data for {} stocks", len(stock_data))
        
        all_features = []
        all_labels = []
        
        for ticker, data in stock_data.items():
            try:
                # Create labels for this stock
                labels = self.create_labels(data, benchmark_data, prediction_horizon)
                
                # Add ticker information
                labels['ticker'] = ticker
                labels['date'] = labels.index
                
                # Generate features, either basic or engineered
                if feature_engineer:
                    features = feature_engineer.engineer_features_with_benchmark(data, benchmark_data)
                else:
                    features = self._create_basic_features(data)
                
                # Align features with labels
                aligned_features = features.loc[labels.index]
                aligned_features['ticker'] = ticker
                aligned_features['date'] = aligned_features.index
                
                all_features.append(aligned_features)
                all_labels.append(labels)
                
                logger.debug("Processed {} samples for {} with {} features", 
                           len(labels), ticker, len(features.columns))
                
            except Exception as e:
                logger.warning("Failed to process {}: {}", ticker, str(e))
                continue
        
        if not all_features:
            raise ValueError("No valid stock data could be processed")
        
        # Combine all stocks
        combined_features = pd.concat(all_features, ignore_index=True)
        combined_labels = pd.concat(all_labels, ignore_index=True)
        
        logger.info("Prepared training data: {} samples from {} stocks with {} features", 
                   len(combined_features), len(stock_data), len(combined_features.columns))
        
        return combined_features, combined_labels
    
    def create_time_series_splits(
        self,
        features: pd.DataFrame,
        labels: pd.DataFrame,
        test_size: float = 0.2,
        validation_size: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create time-series aware train/validation/test splits.
        
        Args:
            features: Feature DataFrame
            labels: Label DataFrame
            test_size: Proportion of data for testing
            validation_size: Proportion of remaining data for validation
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        logger.info("Creating time-series splits with test_size={}, validation_size={}", 
                   test_size, validation_size)
        
        # Ensure data is sorted by date
        features = features.sort_values('date').reset_index(drop=True)
        labels = labels.sort_values('date').reset_index(drop=True)
        
        # Validate alignment
        assert len(features) == len(labels), "Features and labels must have same length"
        assert (features['date'] == labels['date']).all(), "Features and labels must have same dates"
        
        n_samples = len(features)
        
        # Calculate split points (time-series aware)
        test_start = int(n_samples * (1 - test_size))
        val_start = int(test_start * (1 - validation_size))
        
        # Split data
        X_train = features.iloc[:val_start].copy()
        X_val = features.iloc[val_start:test_start].copy()
        X_test = features.iloc[test_start:].copy()
        
        y_train = labels.iloc[:val_start].copy()
        y_val = labels.iloc[val_start:test_start].copy()
        y_test = labels.iloc[test_start:].copy()
        
        logger.info("Data splits - Train: {}, Val: {}, Test: {}", 
                   len(X_train), len(X_val), len(X_test))
        
        # Log label distribution
        for split_name, y_split in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
            positive_rate = y_split['outperforms'].mean()
            logger.info("{} positive rate: {:.2%}", split_name, positive_rate)
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def _calculate_forward_returns(
        self, 
        prices: pd.Series, 
        horizon: int
    ) -> pd.Series:
        """
        Calculate forward returns for given horizon.
        
        Args:
            prices: Price series
            horizon: Number of periods to look ahead
            
        Returns:
            Forward returns series
        """
        future_prices = prices.shift(-horizon)
        returns = (future_prices - prices) / prices
        return returns
    
    def _align_returns(
        self, 
        stock_returns: pd.Series, 
        benchmark_returns: pd.Series
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Align stock and benchmark returns by date.
        
        Args:
            stock_returns: Stock returns series
            benchmark_returns: Benchmark returns series
            
        Returns:
            Tuple of aligned returns
        """
        # Find common dates
        common_dates = stock_returns.index.intersection(benchmark_returns.index)
        
        if len(common_dates) == 0:
            raise ValueError("No overlapping dates between stock and benchmark data")
        
        aligned_stock = stock_returns.loc[common_dates]
        aligned_benchmark = benchmark_returns.loc[common_dates]
        
        logger.debug("Aligned returns: {} common dates", len(common_dates))
        
        return aligned_stock, aligned_benchmark
    
    def _create_basic_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create basic features from OHLCV data.
        
        Args:
            data: OHLCV data
            
        Returns:
            DataFrame with basic features
        """
        features = pd.DataFrame(index=data.index)
        
        # Price-based features
        features['close'] = data['Close']
        features['open'] = data['Open']
        features['high'] = data['High']
        features['low'] = data['Low']
        features['volume'] = data['Volume']
        
        # Basic derived features
        features['price_range'] = (data['High'] - data['Low']) / data['Close']
        features['open_close_ratio'] = data['Open'] / data['Close']
        features['high_close_ratio'] = data['High'] / data['Close']
        features['low_close_ratio'] = data['Low'] / data['Close']
        
        # Simple returns
        features['return_1d'] = data['Close'].pct_change()
        features['return_5d'] = data['Close'].pct_change(periods=5)
        features['return_20d'] = data['Close'].pct_change(periods=20)
        
        # Volume features
        features['volume_ratio'] = data['Volume'] / data['Volume'].rolling(20).mean()
        
        return features
    
    def save_processed_data(
        self,
        features: pd.DataFrame,
        labels: pd.DataFrame,
        filename_prefix: str = "processed_data"
    ) -> None:
        """
        Save processed data to disk.
        
        Args:
            features: Feature DataFrame
            labels: Label DataFrame
            filename_prefix: Prefix for saved files
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        features_path = os.path.join(PROCESSED_DATA_DIR, f"{filename_prefix}_features_{timestamp}.pkl")
        labels_path = os.path.join(PROCESSED_DATA_DIR, f"{filename_prefix}_labels_{timestamp}.pkl")
        
        features.to_pickle(features_path)
        labels.to_pickle(labels_path)
        
        logger.info("Saved processed data to {} and {}", features_path, labels_path)
    
    def load_processed_data(
        self,
        features_path: str,
        labels_path: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load processed data from disk.
        
        Args:
            features_path: Path to features file
            labels_path: Path to labels file
            
        Returns:
            Tuple of (features, labels)
        """
        features = pd.read_pickle(features_path)
        labels = pd.read_pickle(labels_path)
        
        logger.info("Loaded processed data: {} samples", len(features))
        
        return features, labels
    
    def list_saved_processed_data(self) -> List[Tuple[str, str, str]]:
        """
        List all saved processed data files.
        
        Returns:
            List of tuples (prefix, features_path, labels_path, timestamp)
        """
        saved_files = []
        features_files = glob.glob(os.path.join(PROCESSED_DATA_DIR, "*_features_*.pkl"))
        
        for features_file in features_files:
            # Extract prefix and timestamp from filename
            filename = os.path.basename(features_file)
            if filename.startswith("training_") or filename.startswith("processed_data_") or filename.startswith("test_"):
                parts = filename.split("_")
                if len(parts) >= 3:
                    # Find the "features" part and extract everything before it
                    try:
                        features_index = parts.index("features")
                        prefix = "_".join(parts[:features_index])  # Everything before 'features'
                        timestamp = parts[-1].replace(".pkl", "")
                        
                        # Find corresponding labels file
                        labels_file = features_file.replace("_features_", "_labels_")
                        if os.path.exists(labels_file):
                            saved_files.append((prefix, features_file, labels_file, timestamp))
                    except ValueError:
                        # "features" not found in filename, skip this file
                        continue
        
        # Sort by timestamp (newest first)
        saved_files.sort(key=lambda x: x[3], reverse=True)
        
        return saved_files
    
    def get_latest_processed_data(self, prefix: str = None) -> Optional[Tuple[str, str]]:
        """
        Get the latest processed data files for a given prefix.
        
        Args:
            prefix: Prefix to search for (e.g., "training_random_forest")
            
        Returns:
            Tuple of (features_path, labels_path) or None if not found
        """
        saved_files = self.list_saved_processed_data()
        
        if prefix:
            # Filter by prefix
            matching_files = [f for f in saved_files if f[0] == prefix]
            if matching_files:
                return matching_files[0][1], matching_files[0][2]  # features_path, labels_path
        else:
            # Return the latest file regardless of prefix
            if saved_files:
                return saved_files[0][1], saved_files[0][2]
        
        return None

    def get_or_create_processed_data(
        self, 
        stock_data: Dict[str, pd.DataFrame], 
        benchmark_data: pd.DataFrame,
        max_cache_hours: int = 24
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get basic processed data from cache if it matches the provided stock data,
        otherwise create new basic processed data (labels + basic features only).
        
        Args:
            stock_data: Dictionary of stock data by ticker
            benchmark_data: Benchmark data
            max_cache_hours: Maximum age of cache in hours
            
        Returns:
            Tuple of (basic_features, labels) DataFrames
        """
        logger.info("Checking for cached basic processed data")
        
        # Check for recent processed data
        latest_data = self.get_latest_processed_data()
        
        if latest_data:
            features_path, labels_path = latest_data
            
            # Check if the processed data is recent
            cache_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(features_path))
            
            if cache_age < timedelta(hours=max_cache_hours):
                logger.info("Found recent processed data (age: {:.1f} hours), validating compatibility", 
                           cache_age.total_seconds() / 3600)
                
                # Load and validate the cached data
                features, labels = self.load_processed_data(features_path, labels_path)
                
                if self._validate_cached_data(features, labels, stock_data):
                    logger.info("✅ Cached data validation passed, using cached data")
                    logger.info("Loaded processed data: {} samples", len(features))
                    logger.info("Label distribution: {}", labels['outperforms'].value_counts().to_dict())
                    return features, labels
                else:
                    logger.info("❌ Cached data validation failed, creating new processed data")
            else:
                os.remove(features_path) # Remove outdated cache
                os.remove(labels_path)  # Remove outdated cache
                logger.info("Found processed data but it's outdated (age: {:.1f} hours), creating new data", 
                           cache_age.total_seconds() / 3600)
        else:
            logger.info("No processed data found, creating new processed data")
        
        # Create new basic processed data from scratch (no feature engineering)
        logger.info("Processing basic data from scratch")
        
        features, labels = self.create_basic_processed_data(stock_data, benchmark_data)
        
        logger.info("Basic processed data: {} samples", len(features))
        logger.info("Label distribution: {}", labels['outperforms'].value_counts().to_dict())
        
        # Save the newly processed data for future use
        logger.info("Saving basic processed data for future use")
        self.save_processed_data(features, labels, filename_prefix="basic_processed_data")
        
        return features, labels
    
    def _validate_cached_data(
        self,
        features: pd.DataFrame,
        labels: pd.DataFrame,
        current_stock_data: Dict[str, pd.DataFrame]
    ) -> bool:
        """
        Validate that cached data matches the current stock data.
        
        Args:
            features: Cached features DataFrame
            labels: Cached labels DataFrame
            current_stock_data: Current stock data dictionary
            
        Returns:
            True if cached data is compatible, False otherwise
        """
        # Validate tickers match
        current_tickers = set(current_stock_data.keys())
        cached_tickers = set(features['ticker'].unique()) if 'ticker' in features.columns else set()
        
        tickers_match = current_tickers == cached_tickers
        stock_count_match = len(current_stock_data) == len(cached_tickers)
        
        if not tickers_match or not stock_count_match:
            logger.warning("Ticker validation failed:")
            logger.warning("  - Current stocks: {} vs Cached stocks: {}", len(current_tickers), len(cached_tickers))
            logger.warning("  - Stock count match: {}", stock_count_match)
            logger.warning("  - Tickers match: {}", tickers_match)
            
            if not tickers_match:
                missing_in_cache = current_tickers - cached_tickers
                extra_in_cache = cached_tickers - current_tickers
                if missing_in_cache:
                    logger.warning("  - Missing in cache: {}", sorted(list(missing_in_cache))[:10])
                if extra_in_cache:
                    logger.warning("  - Extra in cache: {}", sorted(list(extra_in_cache))[:10])
            
            return False
        
        # Validate date ranges are compatible (at least 80% overlap)
        date_range_compatible = True
        if 'date' in features.columns:
            cached_date_range = (features['date'].min(), features['date'].max())
            
            # Get sample date range from stock data
            sample_ticker = list(current_tickers)[0]
            stock_date_range = (current_stock_data[sample_ticker].index.min(), 
                              current_stock_data[sample_ticker].index.max())
            
            # Convert timezone-aware dates to timezone-naive for comparison
            if hasattr(stock_date_range[0], 'tz_localize'):
                stock_date_range = (stock_date_range[0].tz_localize(None), stock_date_range[1].tz_localize(None))
            elif hasattr(stock_date_range[0], 'tz_convert'):
                stock_date_range = (stock_date_range[0].tz_convert(None), stock_date_range[1].tz_convert(None))
            
            # Ensure cached dates are also timezone-naive
            if hasattr(cached_date_range[0], 'tz_localize'):
                cached_date_range = (cached_date_range[0].tz_localize(None), cached_date_range[1].tz_localize(None))
            elif hasattr(cached_date_range[0], 'tz_convert'):
                cached_date_range = (cached_date_range[0].tz_convert(None), cached_date_range[1].tz_convert(None))
            
            # Check if date ranges overlap significantly (at least 80% overlap)
            overlap_start = max(cached_date_range[0], stock_date_range[0])
            overlap_end = min(cached_date_range[1], stock_date_range[1])
            
            if overlap_start <= overlap_end:
                overlap_days = (overlap_end - overlap_start).days
                cached_days = (cached_date_range[1] - cached_date_range[0]).days
                overlap_ratio = overlap_days / cached_days if cached_days > 0 else 0
                date_range_compatible = overlap_ratio >= 0.8
            else:
                date_range_compatible = False
            
            if not date_range_compatible:
                logger.warning("Date range validation failed:")
                logger.warning("  - Cached date range: {} to {}", cached_date_range[0], cached_date_range[1])
                logger.warning("  - Current date range: {} to {}", stock_date_range[0], stock_date_range[1])
                logger.warning("  - Overlap ratio: {:.2%}", overlap_ratio if 'overlap_ratio' in locals() else 0)
                return False
        
        logger.info("Data validation passed:")
        logger.info("  - Stock count: {} stocks", len(current_tickers))
        logger.info("  - Tickers match: ✅")
        logger.info("  - Date range compatible: ✅")
        
        return True
    
    def validate_data_quality(
        self,
        features: pd.DataFrame,
        labels: pd.DataFrame,
        max_nan_ratio: float = 0.5
    ) -> bool:
        """
        Validate data quality, checking for excessive NaN values and other issues.
        
        Args:
            features: Feature DataFrame
            labels: Label DataFrame
            max_nan_ratio: Maximum allowed ratio of NaN values per feature
            
        Returns:
            True if data quality is acceptable, False otherwise
        """
        logger.info("Validating data quality...")
        
        # Check for NaN patterns in features
        nan_ratios = features.isnull().sum() / len(features)
        high_nan_features = nan_ratios[nan_ratios > max_nan_ratio]
        
        if len(high_nan_features) > 0:
            logger.warning("Found {} features with >{:.0%} NaN values:", 
                          len(high_nan_features), max_nan_ratio)
            for feature, ratio in high_nan_features.items():
                logger.warning("  - {}: {:.2%} NaN", feature, ratio)
        
        # Check for infinite values
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        inf_counts = {}
        for col in numeric_cols:
            inf_count = np.isinf(features[col]).sum()
            if inf_count > 0:
                inf_counts[col] = inf_count
        
        if inf_counts:
            logger.warning("Found infinite values in {} features:", len(inf_counts))
            for feature, count in inf_counts.items():
                logger.warning("  - {}: {} infinite values", feature, count)
        
        # Check labels
        label_nan_count = labels.isnull().sum().sum()
        if label_nan_count > 0:
            logger.error("Found {} NaN values in labels - this should not happen!", label_nan_count)
            return False
        
        # Summary
        total_features = len(features.columns)
        problematic_features = len(high_nan_features) + len(inf_counts)
        
        logger.info("Data quality summary:")
        logger.info("  - Total features: {}", total_features)
        logger.info("  - Features with high NaN ratio: {}", len(high_nan_features))
        logger.info("  - Features with infinite values: {}", len(inf_counts))
        logger.info("  - Data quality score: {:.1%}", 
                   (total_features - problematic_features) / total_features)
        
        return True  # Return True as we can handle these issues with preprocessing

if __name__ == "__main__":
    # Test the data processor
    from data.fetcher import get_sp500_data
    
    logger.info("Testing DataProcessor...")
    
    # Get sample data
    stock_data, benchmark_data = get_sp500_data(max_stocks=3)
    
    # Process data
    processor = DataProcessor()
    features, labels = processor.prepare_training_data(stock_data, benchmark_data)
    
    logger.info("Processed data shape: features={}, labels={}", features.shape, labels.shape)
    logger.info("Label distribution: {}", labels['outperforms'].value_counts())
    
    # Test time series splits
    X_train, X_val, X_test, y_train, y_val, y_test = processor.create_time_series_splits(
        features, labels
    )
    
    logger.info("Split completed successfully")
