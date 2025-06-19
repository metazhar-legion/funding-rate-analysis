"""
GMX exchange client for interacting with the GMX API.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json

from src.data_collection.exchanges.base import BaseExchangeClient
from src.config import settings


class GmxClient(BaseExchangeClient):
    """
    Client for interacting with the GMX API.
    
    Note: GMX doesn't have a traditional REST API like centralized exchanges.
    This client uses subgraph queries and on-chain data to retrieve funding rates.
    """
    
    def __init__(self):
        """
        Initialize the GMX client.
        """
        super().__init__('gmx')
        self.api_key = settings.GMX_API_KEY
        self.api_secret = settings.GMX_API_SECRET
        
        # GMX specific settings
        self.funding_interval_hours = 1  # GMX has hourly funding intervals
        self.subgraph_url = "https://api.thegraph.com/subgraphs/name/gmx-io/gmx-stats"
    
    async def get_supported_assets(self) -> List[Dict[str, Any]]:
        """
        Get list of supported assets on GMX.
        
        Returns:
            List of asset dictionaries
        """
        # GraphQL query to fetch all markets
        query = """
        {
          markets(first: 100) {
            id
            name
            symbol
            indexToken {
              symbol
              name
            }
          }
        }
        """
        
        response = await self._make_subgraph_request(query)
        markets = response.get('data', {}).get('markets', [])
        
        assets = []
        for market in markets:
            symbol = market.get('symbol', '')
            name = market.get('name', symbol)
            
            # Get the underlying index token
            index_token = market.get('indexToken', {})
            index_symbol = index_token.get('symbol', symbol)
            index_name = index_token.get('name', name)
            
            # Categorize asset
            asset_type, category = self.categorize_asset(index_symbol, index_name)
            
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
        Get historical funding rates for a specific asset on GMX.
        
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
        
        # Convert to Unix timestamps
        start_timestamp = int(start_time.timestamp())
        end_timestamp = int(end_time.timestamp())
        
        # GraphQL query to fetch historical funding rates
        query = """
        {
          fundingRates(
            where: {
              marketAddress: "%s",
              timestamp_gte: %d,
              timestamp_lte: %d
            },
            orderBy: timestamp,
            orderDirection: desc,
            first: 1000
          ) {
            timestamp
            rate
            marketAddress
            longOpenInterest
            shortOpenInterest
          }
        }
        """ % (symbol, start_timestamp, end_timestamp)
        
        response = await self._make_subgraph_request(query)
        funding_rates = response.get('data', {}).get('fundingRates', [])
        
        all_rates = []
        for rate_data in funding_rates:
            timestamp = datetime.fromtimestamp(int(rate_data.get('timestamp', 0)))
            funding_rate = float(rate_data.get('rate', 0)) / 1e6  # Convert from basis points
            
            all_rates.append({
                'timestamp': timestamp,
                'rate': funding_rate,
                'rate_annualized': self.annualize_funding_rate(funding_rate, self.funding_interval_hours),
                'payment_interval_hours': self.funding_interval_hours,
                'open_interest_long': float(rate_data.get('longOpenInterest', 0)),
                'open_interest_short': float(rate_data.get('shortOpenInterest', 0)),
            })
        
        return all_rates
    
    async def get_current_funding_rate(self, symbol: str) -> Dict[str, Any]:
        """
        Get current funding rate for a specific asset on GMX.
        
        Args:
            symbol: Asset symbol
            
        Returns:
            Dictionary with current funding rate information
        """
        # GraphQL query to fetch the most recent funding rate
        query = """
        {
          fundingRates(
            where: {
              marketAddress: "%s"
            },
            orderBy: timestamp,
            orderDirection: desc,
            first: 1
          ) {
            timestamp
            rate
            marketAddress
            longOpenInterest
            shortOpenInterest
          }
        }
        """ % symbol
        
        response = await self._make_subgraph_request(query)
        funding_rates = response.get('data', {}).get('fundingRates', [])
        
        if not funding_rates:
            return {
                'timestamp': datetime.utcnow(),
                'rate': 0.0,
                'rate_annualized': 0.0,
                'payment_interval_hours': self.funding_interval_hours,
                'next_funding_time': datetime.utcnow() + timedelta(hours=self.funding_interval_hours),
            }
        
        rate_data = funding_rates[0]
        timestamp = datetime.fromtimestamp(int(rate_data.get('timestamp', 0)))
        funding_rate = float(rate_data.get('rate', 0)) / 1e6  # Convert from basis points
        
        # Calculate next funding time (GMX updates hourly)
        next_hour = timestamp.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        
        return {
            'timestamp': timestamp,
            'rate': funding_rate,
            'rate_annualized': self.annualize_funding_rate(funding_rate, self.funding_interval_hours),
            'payment_interval_hours': self.funding_interval_hours,
            'next_funding_time': next_hour,
            'open_interest_long': float(rate_data.get('longOpenInterest', 0)),
            'open_interest_short': float(rate_data.get('shortOpenInterest', 0)),
        }
    
    async def _make_subgraph_request(self, query: str) -> Dict[str, Any]:
        """
        Make request to GMX subgraph.
        
        Args:
            query: GraphQL query
            
        Returns:
            API response
        """
        headers = {
            'Content-Type': 'application/json',
        }
        
        data = {
            'query': query
        }
        
        response = await self._make_request('POST', '', data=data, headers=headers)
        return response
