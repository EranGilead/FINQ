"""
FINQ Stock Predictor - A machine learning system for predicting stock outperformance.
"""

__version__ = "1.0.0"
__author__ = "FINQ AI Team"
__email__ = "ai@finq.com"

from . import main_config
from . import data
from . import features
from . import models
from . import api

__all__ = ['config', 'data', 'features', 'models', 'api']
