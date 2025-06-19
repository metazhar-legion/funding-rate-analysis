"""
Base exchange client for interacting with DEX APIs.
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple

import aiohttp
import ccxt
import pandas as pd

from src.config import settings
from src.database.models import Exchange as ExchangeModel
from src.database.models import Asset as AssetModel
from src.database.models import FundingRate as FundingRateModel


class BaseExchangeClient(ABC):
    """
    Base class for all exchange clients.
    
    This abstract class defines the interface that all exchange clients must implement.
    """
    
    def __init__(self, exchange_code: str):
        """
        Initialize the exchange client.
        
        Args:
            exchange_code: The code of the exchange (e.g., 'dydx', 'gmx')
        """
        self.exchange_code = exchange_code
        self.exchange_config = settings.EXCHANGES.get(exchange_code, {})
        self.exchange_name = self.exchange_config.get('name', exchange_code.upper())
        self.base_url = self.exchange_config.get('base_url', '')
        self.requires_auth = self.exchange_config.get('requires_auth', False)
        self.supports_websocket = self.exchange_config.get('supports_websocket', False)
        self.logger = logging.getLogger(f"exchange.{exchange_code}")
        
        # Initialize session
        self.session = None
        
        # Initialize CCXT client if available
        self.ccxt_client = None
        try:
            if hasattr(ccxt, exchange_code):
                self.ccxt_client = getattr(ccxt, exchange_code)({
                    'enableRateLimit': True,
                })
                self.logger.info(f"Initialized CCXT client for {exchange_code}")
        except Exception as e:
            self.logger.warning(f"Failed to initialize CCXT client: {e}")
    
    async def __aenter__(self):
        """
        Enter async context.
        """
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        Exit async context.
        """
        if self.session:
            await self.session.close()
            self.session = None
    
    @abstractmethod
    async def get_supported_assets(self) -> List[Dict[str, Any]]:
        """
        Get list of supported assets on the exchange.
        
        Returns:
            List of asset dictionaries with keys:
                - symbol: Asset symbol
                - name: Asset name
                - asset_type: Type of asset (equity_index, stock, etc.)
                - category: Asset category
        """
        pass
    
    @abstractmethod
    async def get_funding_rates(
        self, 
        symbol: str, 
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Get funding rates for a specific asset.
        
        Args:
            symbol: Asset symbol
            start_time: Start time for data retrieval
            end_time: End time for data retrieval
            
        Returns:
            List of funding rate dictionaries with keys:
                - timestamp: Timestamp of the funding rate
                - rate: Funding rate as a decimal
                - rate_annualized: Annualized funding rate
                - payment_interval_hours: Payment interval in hours
                - open_interest_long: Open interest for long positions (optional)
                - open_interest_short: Open interest for short positions (optional)
        """
        pass
    
    @abstractmethod
    async def get_current_funding_rate(self, symbol: str) -> Dict[str, Any]:
        """
        Get current funding rate for a specific asset.
        
        Args:
            symbol: Asset symbol
            
        Returns:
            Dictionary with current funding rate information
        """
        pass
    
    async def _make_request(
        self, 
        method: str, 
        endpoint: str, 
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Make HTTP request to the exchange API.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            params: Query parameters
            headers: HTTP headers
            data: Request body data
            
        Returns:
            API response as dictionary
        """
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        url = f"{self.base_url}{endpoint}"
        headers = headers or {}
        
        try:
            async with self.session.request(
                method, 
                url, 
                params=params, 
                headers=headers, 
                json=data
            ) as response:
                response_data = await response.json()
                if response.status >= 400:
                    self.logger.error(f"API error: {response.status} - {response_data}")
                    raise Exception(f"API error: {response.status} - {response_data}")
                return response_data
        except Exception as e:
            self.logger.error(f"Request error: {e}")
            raise
    
    def categorize_asset(self, symbol: str, name: str) -> Tuple[str, str]:
        """
        Categorize an asset based on its symbol and name.
        
        Args:
            symbol: Asset symbol
            name: Asset name
            
        Returns:
            Tuple of (asset_type, category)
        """
        symbol_upper = symbol.upper()
        name_upper = name.upper()
        
        # Check for equity indices
        for index in settings.ASSET_CATEGORIES['equity_indices']:
            if index in symbol_upper or index in name_upper:
                return 'equity_index', index
        
        # Check for stocks
        for stock in settings.ASSET_CATEGORIES['stocks']:
            if stock in symbol_upper:
                return 'stock', stock
        
        # Check for treasuries
        for treasury in settings.ASSET_CATEGORIES['treasuries']:
            if treasury in symbol_upper or treasury in name_upper:
                return 'treasury', treasury
        
        # Check for commodities
        for commodity in settings.ASSET_CATEGORIES['commodities']:
            if commodity in symbol_upper or commodity in name_upper:
                return 'commodity', commodity
        
        # Check for forex
        for forex in settings.ASSET_CATEGORIES['forex']:
            forex_clean = forex.replace('/', '')
            if forex_clean in symbol_upper or forex in symbol_upper:
                return 'forex', forex
        
        # Default to 'other' if no match
        return 'other', 'uncategorized'
    
    def annualize_funding_rate(self, rate: float, payment_interval_hours: float) -> float:
        """
        Convert a funding rate to an annualized rate.
        
        Args:
            rate: Funding rate as a decimal
            payment_interval_hours: Payment interval in hours
            
        Returns:
            Annualized funding rate
        """
        payments_per_year = (365 * 24) / payment_interval_hours
        return rate * payments_per_year
