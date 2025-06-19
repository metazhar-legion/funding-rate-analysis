"""
Main analysis module for funding rate data.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple

import pandas as pd
import numpy as np
from sqlalchemy.orm import Session

from src.config import settings
from src.database.db import get_db
from src.database.models import Exchange, Asset, FundingRate, OptimizationResult

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger('analyzer')


class FundingRateAnalyzer:
    """
    Main class for analyzing funding rate data.
    """
    
    def __init__(self):
        """
        Initialize the funding rate analyzer.
        """
        self.db = next(get_db())
    
    def load_funding_rates(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        exchange_codes: Optional[List[str]] = None,
        asset_symbols: Optional[List[str]] = None,
        asset_types: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Load funding rates from database into a pandas DataFrame.
        
        Args:
            start_time: Start time for data retrieval
            end_time: End time for data retrieval
            exchange_codes: List of exchange codes to filter by
            asset_symbols: List of asset symbols to filter by
            asset_types: List of asset types to filter by
            categories: List of asset categories to filter by
            
        Returns:
            DataFrame with funding rate data
        """
        # Build query
        query = self.db.query(
            FundingRate,
            Exchange.name.label('exchange_name'),
            Exchange.code.label('exchange_code'),
            Asset.symbol.label('asset_symbol'),
            Asset.name.label('asset_name'),
            Asset.asset_type,
            Asset.category,
        ).join(
            Exchange, FundingRate.exchange_id == Exchange.id
        ).join(
            Asset, FundingRate.asset_id == Asset.id
        )
        
        # Apply filters
        if start_time:
            query = query.filter(FundingRate.timestamp >= start_time)
        
        if end_time:
            query = query.filter(FundingRate.timestamp <= end_time)
        
        if exchange_codes:
            query = query.filter(Exchange.code.in_(exchange_codes))
        
        if asset_symbols:
            query = query.filter(Asset.symbol.in_(asset_symbols))
        
        if asset_types:
            query = query.filter(Asset.asset_type.in_(asset_types))
        
        if categories:
            query = query.filter(Asset.category.in_(categories))
        
        # Execute query
        results = query.all()
        
        # Convert to DataFrame
        data = []
        for row in results:
            funding_rate = row[0]  # FundingRate instance
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
        
        # Sort by timestamp
        if not df.empty:
            df = df.sort_values('timestamp')
        
        return df
    
    def calculate_funding_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate statistics for funding rates.
        
        Args:
            df: DataFrame with funding rate data
            
        Returns:
            DataFrame with funding rate statistics
        """
        if df.empty:
            return pd.DataFrame()
        
        # Group by exchange and asset
        grouped = df.groupby(['exchange', 'asset'])
        
        # Calculate statistics
        stats = grouped.agg({
            'rate': ['mean', 'median', 'std', 'min', 'max'],
            'rate_annualized': ['mean', 'median', 'std', 'min', 'max'],
            'timestamp': ['min', 'max', 'count'],
        }).reset_index()
        
        # Flatten multi-level columns
        stats.columns = ['_'.join(col).strip('_') for col in stats.columns.values]
        
        # Add additional information
        stats['days_covered'] = (stats['timestamp_max'] - stats['timestamp_min']).dt.total_seconds() / (24 * 3600)
        stats['data_points'] = stats['timestamp_count']
        
        # Calculate volatility (annualized standard deviation)
        stats['volatility'] = stats['rate_std'] * np.sqrt(365 * 24 / df['payment_interval_hours'].mean())
        
        # Drop unnecessary columns
        stats = stats.drop(['timestamp_count'], axis=1)
        
        return stats
    
    def find_optimal_exchange_path(
        self,
        df: pd.DataFrame,
        asset_category: str,
        lookback_days: int = 30,
        rebalance_frequency_days: int = 7,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Find the optimal exchange path to minimize funding costs.
        
        Args:
            df: DataFrame with funding rate data
            asset_category: Asset category to analyze
            lookback_days: Number of days to look back for analysis
            rebalance_frequency_days: How often to rebalance (in days)
            
        Returns:
            Tuple of (optimal_path_df, savings_df)
        """
        if df.empty:
            return pd.DataFrame(), pd.DataFrame()
        
        # Filter by asset category
        category_df = df[df['category'] == asset_category].copy()
        
        if category_df.empty:
            return pd.DataFrame(), pd.DataFrame()
        
        # Get unique assets in this category
        assets = category_df['asset'].unique()
        
        results = []
        savings = []
        
        for asset in assets:
            # Filter by asset
            asset_df = category_df[category_df['asset'] == asset].copy()
            
            # Get unique exchanges for this asset
            exchanges = asset_df['exchange'].unique()
            
            if len(exchanges) <= 1:
                # Skip if only one exchange available
                continue
            
            # Calculate daily average rates for each exchange
            asset_df['date'] = asset_df['timestamp'].dt.date
            daily_rates = asset_df.groupby(['exchange', 'date'])['rate_annualized'].mean().reset_index()
            
            # Pivot to get exchanges as columns
            pivot_df = daily_rates.pivot(index='date', columns='exchange', values='rate_annualized')
            
            # Forward fill missing values (use previous day's rate if missing)
            pivot_df = pivot_df.fillna(method='ffill')
            
            # Calculate optimal path
            optimal_path = []
            current_date = pivot_df.index.min()
            end_date = pivot_df.index.max()
            
            while current_date <= end_date:
                # Get rates for current date
                if current_date in pivot_df.index:
                    rates = pivot_df.loc[current_date]
                    
                    # Find exchange with lowest funding rate
                    best_exchange = rates.idxmin()
                    best_rate = rates.min()
                    
                    # Find exchange with highest funding rate (worst case)
                    worst_exchange = rates.idxmax()
                    worst_rate = rates.max()
                    
                    # Calculate savings compared to worst case
                    savings_rate = worst_rate - best_rate
                    
                    optimal_path.append({
                        'asset': asset,
                        'category': asset_category,
                        'date': current_date,
                        'best_exchange': best_exchange,
                        'best_rate': best_rate,
                        'worst_exchange': worst_exchange,
                        'worst_rate': worst_rate,
                        'savings_rate': savings_rate,
                    })
                    
                    # Add to savings
                    savings.append({
                        'asset': asset,
                        'category': asset_category,
                        'date': current_date,
                        'savings_rate': savings_rate,
                    })
                
                # Move to next rebalance date
                current_date += timedelta(days=rebalance_frequency_days)
            
            # Add to results
            results.extend(optimal_path)
        
        # Convert to DataFrames
        optimal_path_df = pd.DataFrame(results)
        savings_df = pd.DataFrame(savings)
        
        # Calculate cumulative savings
        if not savings_df.empty:
            # Convert annual rates to daily rates
            savings_df['daily_savings_rate'] = savings_df['savings_rate'] / 365
            
            # Calculate cumulative savings (compounded)
            savings_df['cumulative_savings'] = (1 + savings_df['daily_savings_rate']).cumprod() - 1
        
        return optimal_path_df, savings_df
    
    def analyze_correlation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze correlation between funding rates across exchanges.
        
        Args:
            df: DataFrame with funding rate data
            
        Returns:
            DataFrame with correlation matrix
        """
        if df.empty:
            return pd.DataFrame()
        
        # Resample to daily frequency
        df['date'] = df['timestamp'].dt.date
        daily_rates = df.groupby(['exchange', 'asset', 'date'])['rate_annualized'].mean().reset_index()
        
        # Create a unique identifier for each exchange-asset pair
        daily_rates['exchange_asset'] = daily_rates['exchange'] + ' - ' + daily_rates['asset']
        
        # Pivot to get exchange-asset pairs as columns
        pivot_df = daily_rates.pivot(index='date', columns='exchange_asset', values='rate_annualized')
        
        # Calculate correlation matrix
        corr_matrix = pivot_df.corr()
        
        return corr_matrix
    
    def save_optimization_result(
        self,
        asset_category: str,
        asset_symbol: str,
        start_date: datetime,
        end_date: datetime,
        strategy_name: str,
        strategy_description: str,
        annualized_funding_cost: float,
        annualized_return: float,
        net_return: float,
        risk_metric: Optional[float] = None,
    ) -> OptimizationResult:
        """
        Save optimization result to database.
        
        Args:
            asset_category: Asset category
            asset_symbol: Asset symbol
            start_date: Start date of analysis
            end_date: End date of analysis
            strategy_name: Name of strategy
            strategy_description: Description of strategy
            annualized_funding_cost: Annualized funding cost
            annualized_return: Annualized return
            net_return: Net return (return - funding cost)
            risk_metric: Risk metric (optional)
            
        Returns:
            OptimizationResult instance
        """
        result = OptimizationResult(
            asset_category=asset_category,
            asset_symbol=asset_symbol,
            start_date=start_date,
            end_date=end_date,
            strategy_name=strategy_name,
            strategy_description=strategy_description,
            annualized_funding_cost=annualized_funding_cost,
            annualized_return=annualized_return,
            net_return=net_return,
            risk_metric=risk_metric,
        )
        
        self.db.add(result)
        self.db.commit()
        
        return result


def main():
    """
    Main entry point for analysis.
    """
    # Initialize analyzer
    analyzer = FundingRateAnalyzer()
    
    # Set default time range (last 30 days)
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=30)
    
    # Load funding rates
    df = analyzer.load_funding_rates(start_time=start_time)
    
    if df.empty:
        logger.warning("No funding rate data found")
        return
    
    # Calculate statistics
    stats = analyzer.calculate_funding_statistics(df)
    logger.info(f"Calculated statistics for {len(stats)} exchange-asset pairs")
    
    # Save statistics to CSV
    stats_path = f"{settings.PROCESSED_DATA_DIR}/funding_stats_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
    stats.to_csv(stats_path, index=False)
    logger.info(f"Saved statistics to {stats_path}")
    
    # Analyze each asset category
    for category in df['category'].unique():
        if category == 'uncategorized':
            continue
        
        logger.info(f"Analyzing category: {category}")
        
        # Find optimal exchange path
        optimal_path, savings = analyzer.find_optimal_exchange_path(df, category)
        
        if not optimal_path.empty:
            # Save results to CSV
            path_path = f"{settings.PROCESSED_DATA_DIR}/optimal_path_{category}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
            optimal_path.to_csv(path_path, index=False)
            logger.info(f"Saved optimal path for {category} to {path_path}")
            
            # Save savings to CSV
            savings_path = f"{settings.PROCESSED_DATA_DIR}/savings_{category}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
            savings.to_csv(savings_path, index=False)
            logger.info(f"Saved savings for {category} to {savings_path}")
    
    # Analyze correlation
    corr_matrix = analyzer.analyze_correlation(df)
    
    if not corr_matrix.empty:
        # Save correlation matrix to CSV
        corr_path = f"{settings.PROCESSED_DATA_DIR}/correlation_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
        corr_matrix.to_csv(corr_path)
        logger.info(f"Saved correlation matrix to {corr_path}")
    
    logger.info("Analysis completed")


if __name__ == "__main__":
    main()
