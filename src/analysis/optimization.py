"""
Optimization strategies for funding rate analysis.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple

import pandas as pd
import numpy as np
from scipy.optimize import minimize

from src.config import settings

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger('optimization')


class FundingRateOptimizer:
    """
    Class for optimizing strategies based on funding rates.
    """
    
    def __init__(self, funding_data: pd.DataFrame):
        """
        Initialize the optimizer with funding rate data.
        
        Args:
            funding_data: DataFrame with funding rate data
        """
        self.funding_data = funding_data
        
        # Ensure data is sorted by timestamp
        if not funding_data.empty:
            self.funding_data = funding_data.sort_values('timestamp')
    
    def optimize_exchange_allocation(
        self,
        asset: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        rebalance_frequency_days: int = 7,
        transaction_cost: float = 0.0005,  # 5 basis points per transaction
    ) -> Tuple[pd.DataFrame, float]:
        """
        Optimize exchange allocation to minimize funding costs.
        
        Args:
            asset: Asset symbol
            start_date: Start date for optimization
            end_date: End date for optimization
            rebalance_frequency_days: How often to rebalance (in days)
            transaction_cost: Cost per transaction as a decimal
            
        Returns:
            Tuple of (allocation_df, total_cost)
        """
        # Filter data
        df = self.funding_data[self.funding_data['asset'] == asset].copy()
        
        if df.empty:
            logger.warning(f"No data found for asset {asset}")
            return pd.DataFrame(), 0.0
        
        # Apply date filters
        if start_date:
            df = df[df['timestamp'] >= start_date]
        
        if end_date:
            df = df[df['timestamp'] <= end_date]
        
        if df.empty:
            logger.warning(f"No data found for asset {asset} in the specified date range")
            return pd.DataFrame(), 0.0
        
        # Get unique exchanges
        exchanges = df['exchange'].unique()
        
        if len(exchanges) <= 1:
            logger.warning(f"Only one exchange found for asset {asset}, optimization not needed")
            return pd.DataFrame(), 0.0
        
        # Calculate daily average rates for each exchange
        df['date'] = df['timestamp'].dt.date
        daily_rates = df.groupby(['exchange', 'date'])['rate_annualized'].mean().reset_index()
        
        # Pivot to get exchanges as columns
        pivot_df = daily_rates.pivot(index='date', columns='exchange', values='rate_annualized')
        
        # Forward fill missing values (use previous day's rate if missing)
        pivot_df = pivot_df.fillna(method='ffill')
        
        # Calculate optimal allocation
        allocations = []
        total_cost = 0.0
        current_exchange = None
        
        # Convert dates to datetime objects
        dates = [datetime.combine(date, datetime.min.time()) for date in pivot_df.index]
        
        # Define rebalance dates
        if not dates:
            return pd.DataFrame(), 0.0
        
        start_date = dates[0]
        end_date = dates[-1]
        
        rebalance_dates = []
        current_date = start_date
        while current_date <= end_date:
            if current_date.date() in pivot_df.index:
                rebalance_dates.append(current_date)
            current_date += timedelta(days=rebalance_frequency_days)
        
        # Calculate optimal allocation for each rebalance date
        for date in rebalance_dates:
            if date.date() not in pivot_df.index:
                continue
            
            # Get rates for current date
            rates = pivot_df.loc[date.date()]
            
            # Find exchange with lowest funding rate
            best_exchange = rates.idxmin()
            best_rate = rates.min()
            
            # Calculate transaction cost if switching exchanges
            switch_cost = 0.0
            if current_exchange is not None and current_exchange != best_exchange:
                switch_cost = transaction_cost
            
            # Calculate cost for this period
            days_until_next_rebalance = rebalance_frequency_days
            if rebalance_dates.index(date) < len(rebalance_dates) - 1:
                next_date = rebalance_dates[rebalance_dates.index(date) + 1]
                days_until_next_rebalance = (next_date - date).days
            
            period_cost = (best_rate / 365) * days_until_next_rebalance + switch_cost
            total_cost += period_cost
            
            # Update current exchange
            current_exchange = best_exchange
            
            # Add to allocations
            allocations.append({
                'date': date,
                'exchange': best_exchange,
                'funding_rate': best_rate,
                'switch_cost': switch_cost,
                'period_cost': period_cost,
                'cumulative_cost': total_cost,
            })
        
        # Convert to DataFrame
        allocation_df = pd.DataFrame(allocations)
        
        return allocation_df, total_cost
    
    def optimize_multi_exchange_portfolio(
        self,
        asset_category: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        rebalance_frequency_days: int = 7,
        transaction_cost: float = 0.0005,  # 5 basis points per transaction
    ) -> pd.DataFrame:
        """
        Optimize a portfolio across multiple exchanges and assets within a category.
        
        Args:
            asset_category: Asset category to optimize
            start_date: Start date for optimization
            end_date: End date for optimization
            rebalance_frequency_days: How often to rebalance (in days)
            transaction_cost: Cost per transaction as a decimal
            
        Returns:
            DataFrame with portfolio allocations
        """
        # Filter data by category
        df = self.funding_data[self.funding_data['category'] == asset_category].copy()
        
        if df.empty:
            logger.warning(f"No data found for category {asset_category}")
            return pd.DataFrame()
        
        # Apply date filters
        if start_date:
            df = df[df['timestamp'] >= start_date]
        
        if end_date:
            df = df[df['timestamp'] <= end_date]
        
        if df.empty:
            logger.warning(f"No data found for category {asset_category} in the specified date range")
            return pd.DataFrame()
        
        # Get unique assets in this category
        assets = df['asset'].unique()
        
        # Optimize each asset separately
        all_allocations = []
        
        for asset in assets:
            logger.info(f"Optimizing allocations for {asset}")
            
            allocation_df, total_cost = self.optimize_exchange_allocation(
                asset=asset,
                start_date=start_date,
                end_date=end_date,
                rebalance_frequency_days=rebalance_frequency_days,
                transaction_cost=transaction_cost,
            )
            
            if not allocation_df.empty:
                # Add asset information
                allocation_df['asset'] = asset
                allocation_df['category'] = asset_category
                
                all_allocations.append(allocation_df)
        
        if not all_allocations:
            return pd.DataFrame()
        
        # Combine all allocations
        portfolio_df = pd.concat(all_allocations, ignore_index=True)
        
        return portfolio_df
    
    def calculate_optimal_leverage(
        self,
        asset: str,
        target_exposure: float = 1.0,  # Target exposure (1.0 = 100%)
        max_leverage: float = 3.0,     # Maximum leverage allowed
        risk_tolerance: float = 0.1,   # Risk tolerance (higher = more risk)
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Calculate optimal leverage based on funding rates and volatility.
        
        Args:
            asset: Asset symbol
            target_exposure: Target exposure as a decimal
            max_leverage: Maximum leverage allowed
            risk_tolerance: Risk tolerance parameter
            start_date: Start date for calculation
            end_date: End date for calculation
            
        Returns:
            Dictionary with optimal leverage parameters
        """
        # Filter data
        df = self.funding_data[self.funding_data['asset'] == asset].copy()
        
        if df.empty:
            logger.warning(f"No data found for asset {asset}")
            return {}
        
        # Apply date filters
        if start_date:
            df = df[df['timestamp'] >= start_date]
        
        if end_date:
            df = df[df['timestamp'] <= end_date]
        
        if df.empty:
            logger.warning(f"No data found for asset {asset} in the specified date range")
            return {}
        
        # Calculate average funding rate across exchanges
        avg_funding_rate = df['rate_annualized'].mean()
        
        # Calculate funding rate volatility
        funding_volatility = df['rate_annualized'].std()
        
        # Define objective function to minimize
        # We want to maximize exposure while minimizing funding costs and risk
        def objective(leverage):
            # Calculate effective exposure
            effective_exposure = leverage[0] * target_exposure
            
            # Calculate funding cost
            funding_cost = leverage[0] * avg_funding_rate
            
            # Calculate risk penalty (higher leverage = higher risk)
            risk_penalty = risk_tolerance * (leverage[0] ** 2) * funding_volatility
            
            # Objective is to minimize negative exposure + funding cost + risk
            # (equivalent to maximizing exposure - funding cost - risk)
            return -effective_exposure + funding_cost + risk_penalty
        
        # Initial guess
        initial_leverage = 1.0
        
        # Bounds for leverage
        bounds = [(0.1, max_leverage)]
        
        # Optimize
        result = minimize(objective, [initial_leverage], bounds=bounds, method='L-BFGS-B')
        
        # Extract optimal leverage
        optimal_leverage = result.x[0]
        
        # Calculate metrics
        effective_exposure = optimal_leverage * target_exposure
        funding_cost = optimal_leverage * avg_funding_rate
        net_return = effective_exposure - funding_cost
        
        return {
            'asset': asset,
            'optimal_leverage': optimal_leverage,
            'effective_exposure': effective_exposure,
            'funding_cost': funding_cost,
            'net_return': net_return,
            'avg_funding_rate': avg_funding_rate,
            'funding_volatility': funding_volatility,
        }
    
    def simulate_strategy_returns(
        self,
        allocation_df: pd.DataFrame,
        price_returns: Optional[pd.DataFrame] = None,
        initial_capital: float = 10000.0,
    ) -> pd.DataFrame:
        """
        Simulate returns of an allocation strategy.
        
        Args:
            allocation_df: DataFrame with allocation strategy
            price_returns: DataFrame with price returns (optional)
            initial_capital: Initial capital
            
        Returns:
            DataFrame with strategy returns
        """
        if allocation_df.empty:
            return pd.DataFrame()
        
        # Create a copy of allocation DataFrame
        result_df = allocation_df.copy()
        
        # Initialize portfolio value
        result_df['portfolio_value'] = initial_capital
        
        # If price returns are provided, incorporate them
        if price_returns is not None and not price_returns.empty:
            # Merge price returns with allocations
            merged_df = pd.merge_asof(
                result_df.sort_values('date'),
                price_returns.sort_values('date'),
                on='date',
                by='asset',
                direction='forward'
            )
            
            if not merged_df.empty:
                # Calculate returns including price changes and funding costs
                merged_df['daily_return'] = merged_df['price_return'] - (merged_df['funding_rate'] / 365)
                
                # Calculate cumulative returns
                merged_df['cumulative_return'] = (1 + merged_df['daily_return']).cumprod()
                
                # Calculate portfolio value
                merged_df['portfolio_value'] = initial_capital * merged_df['cumulative_return']
                
                return merged_df
        
        # If no price returns, just use funding costs
        else:
            # Calculate daily funding cost
            result_df['daily_cost'] = result_df['funding_rate'] / 365
            
            # Calculate cumulative costs
            result_df['cumulative_cost'] = result_df['cumulative_cost'] / initial_capital
            
            # Calculate portfolio value
            result_df['portfolio_value'] = initial_capital * (1 - result_df['cumulative_cost'])
            
            return result_df
        
        return pd.DataFrame()


def main():
    """
    Main entry point for optimization.
    """
    # This is just a placeholder for demonstration
    # In a real scenario, you would load data and run optimizations
    logger.info("Optimization module loaded")


if __name__ == "__main__":
    main()
