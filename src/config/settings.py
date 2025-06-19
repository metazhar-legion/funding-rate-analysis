"""
Configuration settings for the funding rate analysis application.
Loads environment variables from .env file.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Load environment variables
load_dotenv(BASE_DIR / '.env')

# Database settings
DB_TYPE = os.getenv('DB_TYPE', 'sqlite')
DB_PATH = os.getenv('DB_PATH', 'data/database.sqlite')
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = os.getenv('DB_PORT', '5432')
DB_NAME = os.getenv('DB_NAME', 'funding_rate_analysis')
DB_USER = os.getenv('DB_USER', '')
DB_PASSWORD = os.getenv('DB_PASSWORD', '')

# Database connection string
if DB_TYPE == 'sqlite':
    DATABASE_URL = f"sqlite:///{BASE_DIR / DB_PATH}"
else:
    DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Exchange API settings
# dYdX
DYDX_API_KEY = os.getenv('DYDX_API_KEY', '')
DYDX_API_SECRET = os.getenv('DYDX_API_SECRET', '')
DYDX_PASSPHRASE = os.getenv('DYDX_PASSPHRASE', '')

# GMX
GMX_API_KEY = os.getenv('GMX_API_KEY', '')
GMX_API_SECRET = os.getenv('GMX_API_SECRET', '')

# Synthetix
SYNTHETIX_API_KEY = os.getenv('SYNTHETIX_API_KEY', '')
SYNTHETIX_API_SECRET = os.getenv('SYNTHETIX_API_SECRET', '')

# Perpetual Protocol
PERP_API_KEY = os.getenv('PERP_API_KEY', '')
PERP_API_SECRET = os.getenv('PERP_API_SECRET', '')

# Application settings
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
DATA_RETENTION_DAYS = int(os.getenv('DATA_RETENTION_DAYS', '365'))
COLLECTION_INTERVAL = int(os.getenv('COLLECTION_INTERVAL', '3600'))
TIMEZONE = os.getenv('TIMEZONE', 'UTC')

# Feature flags
ENABLE_DASHBOARD = os.getenv('ENABLE_DASHBOARD', 'false').lower() == 'true'
ENABLE_NOTIFICATIONS = os.getenv('ENABLE_NOTIFICATIONS', 'false').lower() == 'true'

# Asset categories for analysis
ASSET_CATEGORIES = {
    'equity_indices': ['S&P500', 'NASDAQ', 'DOW', 'FTSE', 'DAX', 'NIKKEI'],
    'stocks': ['TSLA', 'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META'],
    'treasuries': ['US10Y', 'US2Y', 'US30Y'],
    'commodities': ['GOLD', 'SILVER', 'OIL', 'NATURAL_GAS'],
    'forex': ['EUR/USD', 'GBP/USD', 'USD/JPY', 'USD/CHF'],
}

# Exchange settings
EXCHANGES = {
    'dydx': {
        'name': 'dYdX',
        'base_url': 'https://api.dydx.exchange',
        'requires_auth': True,
        'supports_websocket': True,
    },
    'gmx': {
        'name': 'GMX',
        'base_url': 'https://api.gmx.io',
        'requires_auth': True,
        'supports_websocket': False,
    },
    'synthetix': {
        'name': 'Synthetix',
        'base_url': 'https://api.synthetix.io',
        'requires_auth': True,
        'supports_websocket': True,
    },
    'perpetual': {
        'name': 'Perpetual Protocol',
        'base_url': 'https://api.perp.exchange',
        'requires_auth': True,
        'supports_websocket': True,
    },
}

# Data storage paths
RAW_DATA_DIR = BASE_DIR / 'data' / 'raw'
PROCESSED_DATA_DIR = BASE_DIR / 'data' / 'processed'

# Ensure directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
