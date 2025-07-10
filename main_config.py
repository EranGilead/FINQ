"""
Main configuration file for FINQ Stock Predictor.
Contains core settings and imports specialized configurations.
"""

import os
import pytz
from datetime import datetime, timedelta

# Timezone Configuration
MARKET_TIMEZONE = pytz.timezone('US/Eastern')  # Standard timezone for all financial data (NYSE timezone)

# Data Configuration
DATA_SOURCE = "yahoo"
BENCHMARK_TICKER = "^GSPC"  # S&P 500 index
PREDICTION_HORIZON_DAYS = 5
LOOKBACK_PERIOD_DAYS = 252  # 1 year of trading days

# Model Configuration
MODEL_PARAMS = {
    "random_state": 42, # seed
    "test_size": 0.2,
    "validation_size": 0.2,
    "min_samples_for_training": 100
}

# API Configuration
API_CONFIG = {
    "host": "127.0.0.1",
    "port": 8000,
    "reload": True,
    "log_level": "info"
}

# Logging Configuration
LOG_CONFIG = {
    "level": "INFO",
    "format": "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    "serialize": False
}

# File Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models", "saved")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# Create directories if they don't exist
for dir_path in [DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, LOGS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# S&P 500 stocks (subset for testing)
STOCKS_TICKERS = [
    "AAPL","MSFT","AMZN","NVDA","GOOGL","GOOG","META","TSLA","UNH",
    "LLY","JPM","XOM","JNJ","V","PG","AVGO","MA","HD","CVX",
    "MRK","ABBV","PEP","COST","ADBE","KO","CSCO","WMT","TMO","MCD",
    "PFE","CRM","BAC","ACN","CMCSA","LIN","NFLX","ABT","ORCL","DHR",
    "AMD","WFC","DIS","TXN","PM","VZ","INTU","COP","CAT","AMGN",
    "NEE","INTC","UNP","LOW","IBM","BMY","SPGI","RTX","HON","BA",
    "UPS","GE","QCOM","AMAT","NKE","PLD","NOW","BKNG","SBUX","MS",
    "ELV","MDT","GS","DE","ADP","LMT","TJX","T","BLK","ISRG",
    "MDLZ","GILD","MMC","AXP","SYK","REGN","VRTX","ETN","LRCX","ADI",
    "SCHW","CVS","ZTS","CI","CB","AMT","SLB","C","BDX","MO",
    "PGR","TMUS","FI","SO","EOG","BSX","CME","EQIX","MU","DUK",
    "PANW","PYPL","AON","SNPS","ITW","KLAC","LULU","ICE","APD","SHW",
    "CDNS","CSX","NOC","CL","MPC","HUM","FDX","WM","MCK","TGT",
    "ORLY","HCA","FCX","EMR","MMM","MCO","ROP","CMG","PSX",
    "MAR","PH","APH","GD","USB","NXPI","AJG","NSC","PNC","VLO",
    "F","MSI","GM","TT","EW","CARR","AZO","ADSK","TDG","ANET",
    "SRE","ECL","OXY","PCAR","ADM","MNST","KMB","PSA","CCI","CHTR",
    "MCHP","MSCI","CTAS","WMB","AIG","STZ","HES","NUE","ROST","AFL",
    "AEP","IDXX","D","WELL","DXCM","HLT","ON","COF","PAYX",
]

# Date ranges for development
DEFAULT_START_DATE = datetime.now() - timedelta(days=365 * 3)  # 3 years back
DEFAULT_END_DATE = datetime.now()

# Validation thresholds
MIN_TRADING_DAYS = 30
MAX_MISSING_DATA_RATIO = 0.1
MIN_VOLUME_THRESHOLD = 1000

# Debug mode
DEBUG = os.getenv("DEBUG", "False").lower() == "true"

# Async Configuration
ASYNC_CONFIG = {
    "max_fetch_workers": 16,  # Default number of workers for async data fetching
    "max_training_workers": 4,  # Default number of workers for async model training
    "enable_async_fetch": True,  # Default: use async data fetching
    "enable_async_training": False  # Default: use sync model training
}
