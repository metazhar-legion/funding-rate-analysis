"""
dYdX exchange client for interacting with the dYdX API.
"""

import time
import hmac
import hashlib
import base64
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

import pandas as pd

from src.data_collection.exchanges.base import BaseExchangeClient
from src.config import settings


class DydxClient(BaseExchangeClient):
    """
    Client for interacting with the dYdX API.
    """
    
    def __init__(self):
        """
        Initialize the dYdX client.
        """
        super().__init__('dydx')
        self.api_key = settings.DYDX_API_KEY
        self.api_secret = settings.DYDX_API_SECRET
        self.passphrase = settings.DYDX_PASSPHRASE
        
        # dYdX specific settings
        self.funding_interval_hours = 8  # dYdX has 8-hour funding intervals
    
    async def get_supported_assets(self) -> List[Dict[str, Any]]:
        """
        Get list of supported assets on dYdX.
        
        Returns:
            List of asset dictionaries
        """
        response = await self._make_request('GET', '/v3/markets')
        markets = response.get('markets', {})
        
        assets = []
        for symbol, market_data in markets.items():
            # Filter for perpetual contracts
            if market_data.get('type') == 'PERPETUAL':
                # Extract asset info
                name = market_data.get('name', symbol)
                
                # Categorize asset
                asset_type, category = self.categorize_asset(symbol, name)
                
                assets.append({
                    'symbol': symbol,
                    'name': name,
                    'asset_type': asset_type,
                    'category': category,
                })
        
        return assets
    
    async def get_funding_rates(
        self, 
        symbol: str, 
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Get historical funding rates for a specific asset on dYdX.
        
        Args:
            symbol: Asset symbol
            start_time: Start time for data retrieval (default: 7 days ago)
            end_time: End time for data retrieval (default: now)
            
        Returns:
            List of funding rate dictionaries
        """
        if not start_time:
            start_time = datetime.utcnow() - timedelta(days=7)
        if not end_time:
            end_time = datetime.utcnow()
        
        # Convert to ISO format
        start_iso = start_time.isoformat() + 'Z'
        end_iso = end_time.isoformat() + 'Z'
        
        params = {
            'market': symbol,
            'effectiveBeforeOrAt': end_iso,
            'effectiveAfterOrAt': start_iso,
            'limit': 100,  # Maximum allowed by API
        }
        
        all_rates = []
        has_more = True
        
        while has_more:
            response = await self._make_request('GET', '/v3/historical-funding', params=params)
            rates = response.get('historicalFunding', [])
            
            if not rates:
                break
            
            for rate in rates:
                timestamp = datetime.fromisoformat(rate.get('effectiveAt').replace('Z', '+00:00'))
                funding_rate = float(rate.get('rate', 0))
                
                all_rates.append({
                    'timestamp': timestamp,
                    'rate': funding_rate,
                    'rate_annualized': self.annualize_funding_rate(funding_rate, self.funding_interval_hours),
                    'payment_interval_hours': self.funding_interval_hours,
                    'open_interest_long': float(rate.get('openInterest', 0)) if 'openInterest' in rate else None,
                    'open_interest_short': float(rate.get('openInterest', 0)) if 'openInterest' in rate else None,
                })
            
            # Check if we need to paginate
            if len(rates) < params['limit']:
                has_more = False
            else:
                # Update the effectiveBeforeOrAt parameter for the next page
                params['effectiveBeforeOrAt'] = rates[-1].get('effectiveAt')
        
        return all_rates
    
    async def get_current_funding_rate(self, symbol: str) -> Dict[str, Any]:
        """
        Get current funding rate for a specific asset on dYdX.
        
        Args:
            symbol: Asset symbol
            
        Returns:
            Dictionary with current funding rate information
        """
        response = await self._make_request('GET', f'/v3/markets/{symbol}')
        market_data = response.get('market', {})
        
        # Extract current funding rate
        current_rate = float(market_data.get('nextFundingRate', 0))
        next_funding_time = datetime.fromisoformat(
            market_data.get('nextFundingAt', '').replace('Z', '+00:00')
        )
        
        return {
            'timestamp': datetime.utcnow(),
            'rate': current_rate,
            'rate_annualized': self.annualize_funding_rate(current_rate, self.funding_interval_hours),
            'payment_interval_hours': self.funding_interval_hours,
            'next_funding_time': next_funding_time,
        }
    
    async def _make_request(
        self, 
        method: str, 
        endpoint: str, 
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Make authenticated request to dYdX API.
        
        Args:
            method: HTTP method
            endpoint: API endpoint
            params: Query parameters
            data: Request body data
            
        Returns:
            API response
        """
        # Only authenticate if API keys are provided
        if self.api_key and self.api_secret and self.passphrase and self.requires_auth:
            headers = self._generate_auth_headers(method, endpoint, params, data)
        else:
            headers = {}
        
        return await super()._make_request(method, endpoint, params, headers, data)
    
    def _generate_auth_headers(
        self, 
        method: str, 
        endpoint: str, 
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        """
        Generate authentication headers for dYdX API.
        
        Args:
            method: HTTP method
            endpoint: API endpoint
            params: Query parameters
            data: Request body data
            
        Returns:
            Dictionary of authentication headers
        """
        timestamp = str(int(time.time() * 1000))
        signature_path = endpoint
        
        if params:
            query_string = '&'.join([f"{k}={v}" for k, v in sorted(params.items())])
            signature_path = f"{endpoint}?{query_string}"
        
        message = timestamp + method + signature_path
        if data:
            message += str(data)
        
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).digest()
        
        signature_base64 = base64.b64encode(signature).decode('utf-8')
        
        return {
            'DYDX-API-KEY': self.api_key,
            'DYDX-SIGNATURE': signature_base64,
            'DYDX-TIMESTAMP': timestamp,
            'DYDX-PASSPHRASE': self.passphrase,
        }
