"""
Main data collection orchestrator for funding rate analysis.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Type

import pandas as pd
from sqlalchemy.orm import Session

from src.config import settings
from src.database.db import get_db, init_db
from src.database.models import Exchange as ExchangeModel
from src.database.models import Asset as AssetModel
from src.database.models import FundingRate as FundingRateModel
from src.data_collection.exchanges.base import BaseExchangeClient
from src.data_collection.exchanges.dydx import DydxClient
from src.data_collection.exchanges.gmx import GmxClient
# Import other exchange clients as they are implemented
# from src.data_collection.exchanges.synthetix import SynthetixClient
# from src.data_collection.exchanges.perpetual import PerpetualClient

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger('data_collector')


class FundingRateCollector:
    """
    Main class for collecting funding rate data from all exchanges.
    """
    
    def __init__(self):
        """
        Initialize the funding rate collector.
        """
        self.db = next(get_db())
        self.exchange_clients: Dict[str, Type[BaseExchangeClient]] = {
            'dydx': DydxClient,
            'gmx': GmxClient,
            # Add other exchange clients as they are implemented
            # 'synthetix': SynthetixClient,
            # 'perpetual': PerpetualClient,
        }
    
    async def initialize_exchanges(self):
        """
        Initialize exchange records in the database.
        """
        logger.info("Initializing exchanges in database")
        
        for exchange_code, exchange_config in settings.EXCHANGES.items():
            # Check if exchange already exists
            exchange = self.db.query(ExchangeModel).filter_by(code=exchange_code).first()
            
            if not exchange:
                # Create new exchange record
                exchange = ExchangeModel(
                    name=exchange_config['name'],
                    code=exchange_code,
                    url=exchange_config['base_url'],
                    description=f"DEX offering perpetual contracts for various assets",
                )
                self.db.add(exchange)
                self.db.commit()
                logger.info(f"Added exchange {exchange_code} to database")
    
    async def update_assets(self):
        """
        Update asset records in the database.
        """
        logger.info("Updating assets in database")
        
        for exchange_code, client_class in self.exchange_clients.items():
            # Get exchange from database
            exchange = self.db.query(ExchangeModel).filter_by(code=exchange_code).first()
            
            if not exchange:
                logger.warning(f"Exchange {exchange_code} not found in database")
                continue
            
            # Initialize client
            async with client_class() as client:
                try:
                    # Get supported assets
                    assets = await client.get_supported_assets()
                    
                    for asset_data in assets:
                        # Check if asset already exists
                        asset = self.db.query(AssetModel).filter_by(
                            exchange_id=exchange.id,
                            symbol=asset_data['symbol']
                        ).first()
                        
                        if not asset:
                            # Create new asset record
                            asset = AssetModel(
                                exchange_id=exchange.id,
                                symbol=asset_data['symbol'],
                                name=asset_data['name'],
                                asset_type=asset_data['asset_type'],
                                category=asset_data['category'],
                            )
                            self.db.add(asset)
                            logger.info(f"Added asset {asset_data['symbol']} to database")
                        else:
                            # Update existing asset
                            asset.name = asset_data['name']
                            asset.asset_type = asset_data['asset_type']
                            asset.category = asset_data['category']
                            asset.updated_at = datetime.utcnow()
                            logger.debug(f"Updated asset {asset_data['symbol']}")
                    
                    self.db.commit()
                except Exception as e:
                    self.db.rollback()
                    logger.error(f"Error updating assets for {exchange_code}: {e}")
    
    async def collect_funding_rates(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        exchange_codes: Optional[List[str]] = None,
        asset_symbols: Optional[List[str]] = None,
    ):
        """
        Collect funding rates for all exchanges and assets.
        
        Args:
            start_time: Start time for data retrieval (default: 7 days ago)
            end_time: End time for data retrieval (default: now)
            exchange_codes: List of exchange codes to collect data for (default: all)
            asset_symbols: List of asset symbols to collect data for (default: all)
        """
        if not start_time:
            start_time = datetime.utcnow() - timedelta(days=7)
        if not end_time:
            end_time = datetime.utcnow()
        
        logger.info(f"Collecting funding rates from {start_time} to {end_time}")
        
        # Filter exchanges if specified
        exchange_clients = self.exchange_clients
        if exchange_codes:
            exchange_clients = {
                code: client for code, client in self.exchange_clients.items()
                if code in exchange_codes
            }
        
        for exchange_code, client_class in exchange_clients.items():
            # Get exchange from database
            exchange = self.db.query(ExchangeModel).filter_by(code=exchange_code).first()
            
            if not exchange:
                logger.warning(f"Exchange {exchange_code} not found in database")
                continue
            
            # Get assets for this exchange
            query = self.db.query(AssetModel).filter_by(exchange_id=exchange.id)
            if asset_symbols:
                query = query.filter(AssetModel.symbol.in_(asset_symbols))
            
            assets = query.all()
            
            if not assets:
                logger.warning(f"No assets found for exchange {exchange_code}")
                continue
            
            # Initialize client
            async with client_class() as client:
                for asset in assets:
                    try:
                        logger.info(f"Collecting funding rates for {asset.symbol} on {exchange.name}")
                        
                        # Get funding rates
                        rates = await client.get_funding_rates(asset.symbol, start_time, end_time)
                        
                        # Store funding rates
                        for rate_data in rates:
                            # Check if funding rate already exists
                            existing_rate = self.db.query(FundingRateModel).filter_by(
                                exchange_id=exchange.id,
                                asset_id=asset.id,
                                timestamp=rate_data['timestamp']
                            ).first()
                            
                            if not existing_rate:
                                # Create new funding rate record
                                funding_rate = FundingRateModel(
                                    exchange_id=exchange.id,
                                    asset_id=asset.id,
                                    timestamp=rate_data['timestamp'],
                                    rate=rate_data['rate'],
                                    rate_annualized=rate_data['rate_annualized'],
                                    payment_interval_hours=rate_data['payment_interval_hours'],
                                    open_interest_long=rate_data.get('open_interest_long'),
                                    open_interest_short=rate_data.get('open_interest_short'),
                                )
                                self.db.add(funding_rate)
                        
                        self.db.commit()
                        logger.info(f"Stored {len(rates)} funding rates for {asset.symbol}")
                    
                    except Exception as e:
                        self.db.rollback()
                        logger.error(f"Error collecting funding rates for {asset.symbol} on {exchange.name}: {e}")
    
    async def collect_current_funding_rates(
        self,
        exchange_codes: Optional[List[str]] = None,
        asset_symbols: Optional[List[str]] = None,
    ):
        """
        Collect current funding rates for all exchanges and assets.
        
        Args:
            exchange_codes: List of exchange codes to collect data for (default: all)
            asset_symbols: List of asset symbols to collect data for (default: all)
        """
        logger.info("Collecting current funding rates")
        
        # Filter exchanges if specified
        exchange_clients = self.exchange_clients
        if exchange_codes:
            exchange_clients = {
                code: client for code, client in self.exchange_clients.items()
                if code in exchange_codes
            }
        
        results = []
        
        for exchange_code, client_class in exchange_clients.items():
            # Get exchange from database
            exchange = self.db.query(ExchangeModel).filter_by(code=exchange_code).first()
            
            if not exchange:
                logger.warning(f"Exchange {exchange_code} not found in database")
                continue
            
            # Get assets for this exchange
            query = self.db.query(AssetModel).filter_by(exchange_id=exchange.id)
            if asset_symbols:
                query = query.filter(AssetModel.symbol.in_(asset_symbols))
            
            assets = query.all()
            
            if not assets:
                logger.warning(f"No assets found for exchange {exchange_code}")
                continue
            
            # Initialize client
            async with client_class() as client:
                for asset in assets:
                    try:
                        # Get current funding rate
                        rate_data = await client.get_current_funding_rate(asset.symbol)
                        
                        # Add exchange and asset info
                        rate_data['exchange'] = exchange.name
                        rate_data['exchange_code'] = exchange.code
                        rate_data['asset'] = asset.symbol
                        rate_data['asset_name'] = asset.name
                        rate_data['asset_type'] = asset.asset_type
                        rate_data['asset_category'] = asset.category
                        
                        results.append(rate_data)
                        
                        # Store in database
                        existing_rate = self.db.query(FundingRateModel).filter_by(
                            exchange_id=exchange.id,
                            asset_id=asset.id,
                            timestamp=rate_data['timestamp']
                        ).first()
                        
                        if not existing_rate:
                            # Create new funding rate record
                            funding_rate = FundingRateModel(
                                exchange_id=exchange.id,
                                asset_id=asset.id,
                                timestamp=rate_data['timestamp'],
                                rate=rate_data['rate'],
                                rate_annualized=rate_data['rate_annualized'],
                                payment_interval_hours=rate_data['payment_interval_hours'],
                                open_interest_long=rate_data.get('open_interest_long'),
                                open_interest_short=rate_data.get('open_interest_short'),
                            )
                            self.db.add(funding_rate)
                        
                        self.db.commit()
                    
                    except Exception as e:
                        self.db.rollback()
                        logger.error(f"Error collecting current funding rate for {asset.symbol} on {exchange.name}: {e}")
        
        return results
    
    def export_funding_rates_to_csv(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        exchange_codes: Optional[List[str]] = None,
        asset_symbols: Optional[List[str]] = None,
        output_path: Optional[str] = None,
    ):
        """
        Export funding rates to CSV file.
        
        Args:
            start_time: Start time for data retrieval (default: all time)
            end_time: End time for data retrieval (default: now)
            exchange_codes: List of exchange codes to export data for (default: all)
            asset_symbols: List of asset symbols to export data for (default: all)
            output_path: Path to output CSV file (default: data/processed/funding_rates_{timestamp}.csv)
        """
        if not end_time:
            end_time = datetime.utcnow()
        
        # Build query
        query = self.db.query(
            FundingRateModel,
            ExchangeModel.name.label('exchange_name'),
            ExchangeModel.code.label('exchange_code'),
            AssetModel.symbol.label('asset_symbol'),
            AssetModel.name.label('asset_name'),
            AssetModel.asset_type,
            AssetModel.category,
        ).join(
            ExchangeModel, FundingRateModel.exchange_id == ExchangeModel.id
        ).join(
            AssetModel, FundingRateModel.asset_id == AssetModel.id
        )
        
        # Apply filters
        if start_time:
            query = query.filter(FundingRateModel.timestamp >= start_time)
        
        query = query.filter(FundingRateModel.timestamp <= end_time)
        
        if exchange_codes:
            query = query.filter(ExchangeModel.code.in_(exchange_codes))
        
        if asset_symbols:
            query = query.filter(AssetModel.symbol.in_(asset_symbols))
        
        # Execute query
        results = query.all()
        
        # Convert to DataFrame
        data = []
        for row in results:
            funding_rate = row[0]  # FundingRateModel instance
            exchange_name = row[1]
            exchange_code = row[2]
            asset_symbol = row[3]
            asset_name = row[4]
            asset_type = row[5]
            category = row[6]
            
            data.append({
                'timestamp': funding_rate.timestamp,
                'exchange': exchange_name,
                'exchange_code': exchange_code,
                'asset': asset_symbol,
                'asset_name': asset_name,
                'asset_type': asset_type,
                'category': category,
                'rate': funding_rate.rate,
                'rate_annualized': funding_rate.rate_annualized,
                'payment_interval_hours': funding_rate.payment_interval_hours,
                'open_interest_long': funding_rate.open_interest_long,
                'open_interest_short': funding_rate.open_interest_short,
            })
        
        df = pd.DataFrame(data)
        
        # Generate output path if not provided
        if not output_path:
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            output_path = f"{settings.PROCESSED_DATA_DIR}/funding_rates_{timestamp}.csv"
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        logger.info(f"Exported {len(df)} funding rates to {output_path}")
        
        return output_path


async def main():
    """
    Main entry point for data collection.
    """
    # Initialize database
    init_db()
    
    # Initialize collector
    collector = FundingRateCollector()
    
    # Initialize exchanges
    await collector.initialize_exchanges()
    
    # Update assets
    await collector.update_assets()
    
    # Collect historical funding rates (last 30 days)
    start_time = datetime.utcnow() - timedelta(days=30)
    await collector.collect_funding_rates(start_time=start_time)
    
    # Export to CSV
    collector.export_funding_rates_to_csv(start_time=start_time)
    
    logger.info("Data collection completed")


if __name__ == "__main__":
    asyncio.run(main())
