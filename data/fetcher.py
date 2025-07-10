"""
Data fetching module for FINQ Stock Predictor.
Handles downloading and preprocessing stock data from Yahoo Finance.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from loguru import logger
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main_config import (
    BENCHMARK_TICKER, STOCKS_TICKERS, DATA_DIR, PROCESSED_DATA_DIR,
    DEFAULT_START_DATE, DEFAULT_END_DATE, MIN_TRADING_DAYS,
    MAX_MISSING_DATA_RATIO, MIN_VOLUME_THRESHOLD, MARKET_TIMEZONE
)



class DataFetcher:
    """
    Handles fetching and preprocessing stock data from Yahoo Finance.
    """
    
    def __init__(self, cache_enabled: bool = True):
        """
        Initialize DataFetcher.
        
        Args:
            cache_enabled: Whether to cache downloaded data locally
        """
        self.cache_enabled = cache_enabled
        logger.info("DataFetcher initialized with cache_enabled={}", cache_enabled)
    
    def fetch_stock_data(
        self, 
        ticker: str, 
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Fetch stock data for a given ticker.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date for data fetch
            end_date: End date for data fetch
            
        Returns:
            DataFrame with OHLCV data
            
        Raises:
            ValueError: If ticker is invalid or data fetch fails
        """
        start_date = start_date or DEFAULT_START_DATE
        end_date = end_date or DEFAULT_END_DATE
        
        # Standardize input dates to market timezone
        start_date = start_date.astimezone(MARKET_TIMEZONE)
        end_date = end_date.astimezone(MARKET_TIMEZONE)
        
        logger.info("Fetching data for ticker: {} from {} to {}", ticker, start_date, end_date)
        
        # Check cache first
        if self.cache_enabled:
            cached_data = self._load_cached_data(ticker, start_date, end_date)
            if cached_data is not None:
                logger.info("Loaded cached data for {}", ticker)
                return cached_data
        
        try:
            # Download data from Yahoo Finance
            stock = yf.Ticker(ticker)
            data = stock.history(start=start_date, end=end_date)
            
            if data.empty:
                raise ValueError(f"No data found for ticker {ticker}")
            
            # Convert/standardize datetime index to market timezone (US Eastern)
            # This ensures all datetime handling is timezone-aware and consistent
            if data.index.tz is None:
                # Timezone-naive data - localize to market timezone
                data.index = data.index.tz_localize(MARKET_TIMEZONE)
                logger.debug("Localized timezone-naive index to market timezone for {}", ticker)
            else:
                # Already timezone-aware - convert to market timezone
                data.index = data.index.tz_convert(MARKET_TIMEZONE)
                logger.debug("Converted timezone-aware index to market timezone for {}", ticker)
            
            # Validate data quality
            validated_data = self._validate_data(data, ticker)
            
            # Cache data if enabled
            if self.cache_enabled:
                self._cache_data(validated_data, ticker, start_date, end_date)
            
            logger.info("Successfully fetched {} rows for {}", len(validated_data), ticker)
            return validated_data
            
        except Exception as e:
            logger.error("Failed to fetch data for {}: {}", ticker, str(e))
            raise ValueError(f"Failed to fetch data for {ticker}: {str(e)}")
    
    def fetch_benchmark_data(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Fetch S&P 500 benchmark data.
        
        Args:
            start_date: Start date for data fetch
            end_date: End date for data fetch
            
        Returns:
            DataFrame with S&P 500 OHLCV data
        """
        logger.info("Fetching benchmark data (S&P 500)")
        return self.fetch_stock_data(BENCHMARK_TICKER, start_date, end_date)
    
    def fetch_multiple_stocks(
        self,
        tickers: List[str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple stocks.
        
        Args:
            tickers: List of stock ticker symbols
            start_date: Start date for data fetch
            end_date: End date for data fetch
            
        Returns:
            Dictionary mapping ticker to DataFrame
        """
        logger.info("Fetching data for {} tickers", len(tickers))
        
        results = {}
        failed_tickers = []
        
        for ticker in tickers:
            try:
                data = self.fetch_stock_data(ticker, start_date, end_date)
                results[ticker] = data
            except Exception as e:
                logger.warning("Failed to fetch data for {}: {}", ticker, str(e))
                failed_tickers.append(ticker)
        
        if failed_tickers:
            logger.warning("Failed to fetch data for {} tickers: {}", 
                         len(failed_tickers), failed_tickers)
        
        logger.info("Successfully fetched data for {} out of {} tickers", 
                   len(results), len(tickers))
        
        return results
    
    async def fetch_multiple_stocks_async(
        self,
        tickers: List[str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        max_workers: Optional[int] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple stocks asynchronously using ThreadPoolExecutor.
        
        Args:
            tickers: List of stock ticker symbols
            start_date: Start date for data fetch
            end_date: End date for data fetch
            max_workers: Maximum number of concurrent workers (default: min(32, len(tickers)))
            
        Returns:
            Dictionary mapping ticker to DataFrame
        """
        if max_workers is None:
            max_workers = min(32, len(tickers))  # Reasonable default
            
        logger.info("Fetching data for {} tickers asynchronously with {} workers", 
                   len(tickers), max_workers)
        
        results = {}
        failed_tickers = []
        
        def fetch_single_stock(ticker: str) -> Tuple[str, Optional[pd.DataFrame]]:
            """Helper function to fetch single stock data."""
            try:
                data = self.fetch_stock_data(ticker, start_date, end_date)
                return ticker, data
            except Exception as e:
                logger.warning("Failed to fetch data for {}: {}", ticker, str(e))
                return ticker, None
        
        # Use ThreadPoolExecutor for concurrent execution
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            futures = {
                loop.run_in_executor(executor, fetch_single_stock, ticker): ticker 
                for ticker in tickers
            }
            
            # Collect results as they complete
            for future in asyncio.as_completed(futures):
                ticker, data = await future
                if data is not None:
                    results[ticker] = data
                    logger.debug("Completed fetching data for {}", ticker)
                else:
                    failed_tickers.append(ticker)
        
        if failed_tickers:
            logger.warning("Failed to fetch data for {} tickers: {}", 
                         len(failed_tickers), failed_tickers)
        
        logger.info("Successfully fetched data for {} out of {} tickers", 
                   len(results), len(tickers))
        
        return results

    def _validate_data(self, data: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Validate and clean stock data.
        
        Args:
            data: Raw stock data
            ticker: Stock ticker symbol
            
        Returns:
            Validated and cleaned data
            
        Raises:
            ValueError: If data quality is insufficient
        """
        logger.debug("Validating data for {}", ticker)
        
        # Check minimum trading days
        if len(data) < MIN_TRADING_DAYS:
            raise ValueError(f"Insufficient data for {ticker}: {len(data)} days < {MIN_TRADING_DAYS}")
        
        # Check for required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns for {ticker}: {missing_columns}")
        
        # Check for excessive missing data
        missing_ratio = data[required_columns].isnull().sum().sum() / (len(data) * len(required_columns))
        if missing_ratio > MAX_MISSING_DATA_RATIO:
            raise ValueError(f"Excessive missing data for {ticker}: {missing_ratio:.2%}")
        
        # Remove rows with invalid prices (negative or zero)
        price_columns = ['Open', 'High', 'Low', 'Close']
        valid_prices = (data[price_columns] > 0).all(axis=1)
        data = data[valid_prices]
        
        # Remove rows with invalid volume
        data = data[data['Volume'] >= MIN_VOLUME_THRESHOLD]
        
        # Forward fill missing values
        data[required_columns] = data[required_columns].ffill()
        
        # Drop any remaining rows with NaN
        data = data.dropna()
        
        logger.debug("Validation completed for {}: {} rows remaining", ticker, len(data))
        return data
    
    def _cache_data(
        self, 
        data: pd.DataFrame, 
        ticker: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> None:
        """
        Cache stock data locally.
        
        Args:
            data: Stock data to cache
            ticker: Stock ticker symbol
            start_date: Start date of data
            end_date: End date of data
        """
        try:
            cache_filename = f"{ticker}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.pkl"
            cache_path = os.path.join(DATA_DIR, cache_filename)
            
            # Save with metadata
            cache_data = {
                'data': data,
                'ticker': ticker,
                'start_date': start_date,
                'end_date': end_date,
                'cached_at': datetime.now()
            }
            
            data.to_pickle(cache_path)
            logger.debug("Cached data for {} at {}", ticker, cache_path)
            
        except Exception as e:
            logger.warning("Failed to cache data for {}: {}", ticker, str(e))
    
    def _load_cached_data(
        self, 
        ticker: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> Optional[pd.DataFrame]:
        """
        Load cached stock data if available.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date of data
            end_date: End date of data
            
        Returns:
            Cached data or None if not available
        """
        try:
            cache_filename = f"{ticker}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.pkl"
            cache_path = os.path.join(DATA_DIR, cache_filename)
            
            if os.path.exists(cache_path):
                # Check if cache is recent (within 1 day)
                cache_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_path))
                if cache_age < timedelta(days=1):
                    data = pd.read_pickle(cache_path)
                    
                    # Standardize cached data to market timezone
                    if data.index.tz is None:
                        # Timezone-naive cached data - localize to market timezone
                        data.index = data.index.tz_localize(MARKET_TIMEZONE)
                        logger.debug("Localized cached timezone-naive data to market timezone for {}", ticker)
                    else:
                        # Already timezone-aware - convert to market timezone
                        data.index = data.index.tz_convert(MARKET_TIMEZONE)
                        logger.debug("Converted cached timezone-aware data to market timezone for {}", ticker)
                    
                    logger.debug("Loaded cached data for {}", ticker)
                    return data
                else:
                    logger.debug("Cache expired for {}, removing old cache file", ticker)
                    os.remove(cache_path)
        except Exception as e:
            logger.warning("Failed to load cached data for {}: {}", ticker, str(e))
        
        return None
    

async def get_sp500_data_async(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    max_stocks: Optional[int] = None,
    max_workers: Optional[int] = None
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Async version to fetch S&P 500 stocks and benchmark data concurrently.
    
    Args:
        start_date: Start date for data fetch
        end_date: End date for data fetch
        max_stocks: Maximum number of stocks to fetch (for testing)
        max_workers: Maximum number of concurrent workers
        
    Returns:
        Tuple of (stock_data_dict, benchmark_data)
    """
    fetcher = DataFetcher()
    
    # Limit stocks for testing if specified
    tickers = STOCKS_TICKERS[:max_stocks] if max_stocks else STOCKS_TICKERS
    
    # Fetch stock data and benchmark data concurrently
    stock_task = fetcher.fetch_multiple_stocks_async(tickers, start_date, end_date, max_workers)
    
    # Create a separate task for benchmark data
    async def fetch_benchmark():
        return fetcher.fetch_benchmark_data(start_date, end_date)
    
    # Run both tasks concurrently
    stock_data, benchmark_data = await asyncio.gather(
        stock_task,
        fetch_benchmark()
    )
    
    return stock_data, benchmark_data


def get_sp500_data(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    max_stocks: Optional[int] = None
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Convenience function to fetch S&P 500 stocks and benchmark data.
    
    Args:
        start_date: Start date for data fetch
        end_date: End date for data fetch
        max_stocks: Maximum number of stocks to fetch (for testing)
        
    Returns:
        Tuple of (stock_data_dict, benchmark_data)
    """
    fetcher = DataFetcher()
    
    # Limit stocks for testing if specified
    tickers = STOCKS_TICKERS[:max_stocks] if max_stocks else STOCKS_TICKERS
    
    # Fetch stock data
    stock_data = fetcher.fetch_multiple_stocks(tickers, start_date, end_date)
    
    # Fetch benchmark data
    benchmark_data = fetcher.fetch_benchmark_data(start_date, end_date)
    
    return stock_data, benchmark_data


if __name__ == "__main__":
    # Test the data fetcher
    logger.info("Testing DataFetcher...")
    
    # Test with a small subset
    stock_data, benchmark_data = get_sp500_data(max_stocks=5)
    
    logger.info("Fetched data for {} stocks", len(stock_data))
    logger.info("Benchmark data shape: {}", benchmark_data.shape)
    
    # Show sample data
    if stock_data:
        sample_ticker = list(stock_data.keys())[0]
        sample_data = stock_data[sample_ticker]
        logger.info("Sample data for {}:", sample_ticker)
        logger.info(sample_data.head())
