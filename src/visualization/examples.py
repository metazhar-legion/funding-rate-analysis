#!/usr/bin/env python
"""
Example script to demonstrate visualization capabilities with sample data.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
import random

import pandas as pd
import numpy as np

from src.config import settings
from src.database.db import init_db, get_db
from src.database.models import Exchange, Asset, FundingRate
from src.visualization.visualizer import FundingRateVisualizer

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger('visualization_examples')


def generate_sample_data():
    """
    Generate sample funding rate data for visualization examples.
    This is useful for testing visualizations before real data is collected.
    """
    logger.info("Generating sample funding rate data...")
    
    # Initialize database
    init_db()
    db = next(get_db())
    
    # Sample exchanges
    exchanges = [
        {"name": "dYdX", "code": "DYDX", "url": "https://dydx.exchange"},
        {"name": "GMX", "code": "GMX", "url": "https://gmx.io"},
        {"name": "Synthetix", "code": "SNX", "url": "https://synthetix.io"},
        {"name": "Perpetual Protocol", "code": "PERP", "url": "https://perp.com"},
    ]
    
    # Sample assets by category
    assets_by_category = {
        "Equity Index": [
            {"symbol": "SPX", "name": "S&P 500", "asset_type": "index"},
            {"symbol": "NDX", "name": "Nasdaq 100", "asset_type": "index"},
            {"symbol": "DJI", "name": "Dow Jones Industrial Average", "asset_type": "index"},
        ],
        "Stock": [
            {"symbol": "TSLA", "name": "Tesla Inc", "asset_type": "stock"},
            {"symbol": "AAPL", "name": "Apple Inc", "asset_type": "stock"},
            {"symbol": "MSFT", "name": "Microsoft Corporation", "asset_type": "stock"},
            {"symbol": "AMZN", "name": "Amazon.com Inc", "asset_type": "stock"},
        ],
        "Treasury": [
            {"symbol": "US10Y", "name": "US 10-Year Treasury", "asset_type": "bond"},
            {"symbol": "US02Y", "name": "US 2-Year Treasury", "asset_type": "bond"},
        ],
        "Commodity": [
            {"symbol": "GOLD", "name": "Gold", "asset_type": "commodity"},
            {"symbol": "SLVR", "name": "Silver", "asset_type": "commodity"},
            {"symbol": "OIL", "name": "Crude Oil", "asset_type": "commodity"},
        ],
    }
    
    # Insert exchanges
    exchange_ids = {}
    for exchange_data in exchanges:
        exchange = Exchange(**exchange_data)
        db.add(exchange)
        db.flush()
        exchange_ids[exchange.code] = exchange.id
    
    # Insert assets
    asset_ids = {}
    for category, assets in assets_by_category.items():
        for asset_data in assets:
            # Add to each exchange with slight variations
            for exchange_code in exchange_ids.keys():
                # Not all assets available on all exchanges
                if random.random() > 0.7:
                    continue
                    
                asset = Asset(
                    exchange_id=exchange_ids[exchange_code],
                    symbol=asset_data["symbol"],
                    name=asset_data["name"],
                    asset_type=asset_data["asset_type"],
                    category=category,
                    is_active=True,
                )
                db.add(asset)
                db.flush()
                asset_ids[(exchange_code, asset.symbol)] = asset.id
    
    # Generate funding rate data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)  # 3 months of data
    
    funding_rates = []
    current_date = start_date
    
    # Different funding rate patterns for different assets
    patterns = {
        "SPX": lambda t: 0.01 * np.sin(t/10) + 0.005,  # Oscillating around positive
        "NDX": lambda t: 0.015 * np.sin(t/8) + 0.008,  # Oscillating around positive, higher amplitude
        "DJI": lambda t: 0.008 * np.sin(t/12) + 0.003,  # Oscillating around positive, lower amplitude
        "TSLA": lambda t: 0.02 * np.sin(t/5) - 0.005,  # Oscillating around negative
        "AAPL": lambda t: 0.012 * np.sin(t/7) + 0.002,  # Oscillating around positive
        "MSFT": lambda t: 0.01 * np.sin(t/9) + 0.001,   # Oscillating around positive
        "AMZN": lambda t: 0.015 * np.sin(t/6) - 0.002,  # Oscillating around negative
        "US10Y": lambda t: 0.005 * np.sin(t/15) + 0.001,  # Low volatility, slightly positive
        "US02Y": lambda t: 0.003 * np.sin(t/20) + 0.0005,  # Very low volatility, slightly positive
        "GOLD": lambda t: -0.01 * np.sin(t/10) - 0.002,  # Oscillating around negative
        "SLVR": lambda t: -0.012 * np.sin(t/9) - 0.003,  # Oscillating around negative
        "OIL": lambda t: 0.018 * np.sin(t/7) - 0.001,  # High volatility, oscillating
    }
    
    # Default pattern for any asset not explicitly defined
    default_pattern = lambda t: 0.008 * np.sin(t/10)
    
    # Generate time series data
    t = 0
    while current_date < end_date:
        # 8-hour intervals
        current_date += timedelta(hours=8)
        t += 1
        
        for (exchange_code, asset_symbol), asset_id in asset_ids.items():
            # Add some randomness to make exchanges differ
            exchange_factor = {
                "DYDX": 1.0,
                "GMX": 1.1,
                "SNX": 0.9,
                "PERP": 1.05
            }.get(exchange_code, 1.0)
            
            # Get pattern for this asset
            pattern_func = patterns.get(asset_symbol, default_pattern)
            
            # Calculate base rate using the pattern
            base_rate = pattern_func(t)
            
            # Apply exchange factor and add some noise
            rate = base_rate * exchange_factor + random.uniform(-0.002, 0.002)
            
            # Annualize the rate (assuming 3 funding payments per day)
            rate_annualized = rate * 3 * 365
            
            # Random open interest
            open_interest = random.uniform(1000000, 10000000)
            open_interest_ratio = random.uniform(0.4, 0.6)
            
            funding_rate = FundingRate(
                exchange_id=exchange_ids[exchange_code],
                asset_id=asset_id,
                timestamp=current_date,
                rate=rate,
                rate_annualized=rate_annualized,
                payment_interval_hours=8,
                open_interest_long=open_interest * open_interest_ratio,
                open_interest_short=open_interest * (1 - open_interest_ratio),
            )
            
            funding_rates.append(funding_rate)
    
    # Insert funding rates in batches to avoid memory issues
    batch_size = 1000
    for i in range(0, len(funding_rates), batch_size):
        batch = funding_rates[i:i+batch_size]
        db.add_all(batch)
        db.commit()
        logger.info(f"Inserted batch of {len(batch)} funding rates")
    
    # Generate some optimization results
    logger.info("Generating sample optimization results...")
    
    # For each asset, find the best and worst exchange on each date
    for asset_symbol in set(symbol for _, symbol in asset_ids.keys()):
        # Query for this asset across all exchanges
        query = """
        SELECT 
            a.symbol as asset,
            e.name as exchange,
            DATE(fr.timestamp) as date,
            AVG(fr.rate_annualized) as avg_rate
        FROM 
            funding_rates fr
        JOIN 
            assets a ON fr.asset_id = a.id
        JOIN 
            exchanges e ON fr.exchange_id = e.id
        WHERE 
            a.symbol = :asset_symbol
        GROUP BY 
            a.symbol, e.name, DATE(fr.timestamp)
        ORDER BY 
            date, avg_rate
        """
        
        results = db.execute(query, {"asset_symbol": asset_symbol}).fetchall()
        
        # Group by date
        by_date = {}
        for row in results:
            date_str = row[2]
            if date_str not in by_date:
                by_date[date_str] = []
            by_date[date_str].append(row)
        
        # Find best and worst for each date
        for date_str, rows in by_date.items():
            if len(rows) < 2:
                continue  # Need at least 2 exchanges to compare
                
            # Sort by rate (ascending)
            sorted_rows = sorted(rows, key=lambda x: x[3])
            
            best_row = sorted_rows[0]  # Lowest funding rate
            worst_row = sorted_rows[-1]  # Highest funding rate
            
            # Insert optimization result
            db.execute("""
            INSERT INTO optimization_results 
                (asset, date, best_exchange, best_rate, worst_exchange, worst_rate, created_at)
            VALUES 
                (:asset, :date, :best_exchange, :best_rate, :worst_exchange, :worst_rate, :created_at)
            """, {
                "asset": asset_symbol,
                "date": date_str,
                "best_exchange": best_row[1],
                "best_rate": best_row[3],
                "worst_exchange": worst_row[1],
                "worst_rate": worst_row[3],
                "created_at": datetime.now()
            })
    
    db.commit()
    logger.info("Sample data generation complete!")


def generate_example_visualizations(output_dir: str):
    """
    Generate example visualizations using the sample data.
    
    Args:
        output_dir: Directory to save visualizations
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize visualizer
    visualizer = FundingRateVisualizer(output_dir=output_dir)
    
    # Load all funding rate data
    df = visualizer.load_funding_rates()
    
    if df.empty:
        logger.warning("No funding rate data found. Run generate_sample_data() first.")
        return
    
    logger.info("Generating example visualizations...")
    
    # 1. Funding rates over time for equity indices
    equity_df = df[df['category'] == 'Equity Index']
    fig = visualizer.plot_funding_rates_over_time(
        equity_df,
        category='Equity Index',
        use_plotly=True,
    )
    if fig:
        save_path = output_path / "equity_indices_funding_rates.html"
        fig.write_html(str(save_path))
        logger.info(f"Saved equity indices funding rates chart to {save_path}")
    
    # 2. Asset comparison for stocks
    stock_df = df[df['category'] == 'Stock']
    stock_symbols = stock_df['asset'].unique()
    if len(stock_symbols) >= 2:
        fig = visualizer.plot_funding_rate_comparison(
            stock_df,
            asset_symbols=list(stock_symbols)[:4],  # Limit to 4 stocks for clarity
            use_plotly=True,
        )
        if fig:
            save_path = output_path / "stock_comparison.html"
            fig.write_html(str(save_path))
            logger.info(f"Saved stock comparison chart to {save_path}")
    
    # 3. Funding rate heatmap
    fig = visualizer.plot_funding_rate_heatmap(df)
    if fig:
        save_path = output_path / "funding_rate_heatmap.png"
        fig.savefig(str(save_path), dpi=300, bbox_inches='tight')
        logger.info(f"Saved funding rate heatmap to {save_path}")
    
    # 4. Optimal path visualization for S&P 500
    db = next(get_db())
    optimization_results = db.execute("""
        SELECT asset, date, best_exchange, best_rate, worst_exchange, worst_rate 
        FROM optimization_results 
        WHERE asset = 'SPX'
    """).fetchall()
    
    if optimization_results:
        optimal_path_df = pd.DataFrame(
            optimization_results,
            columns=['asset', 'date', 'best_exchange', 'best_rate', 'worst_exchange', 'worst_rate']
        )
        optimal_path_df['date'] = pd.to_datetime(optimal_path_df['date'])
        
        fig = visualizer.plot_optimal_path(
            optimal_path_df,
            asset='SPX',
            use_plotly=True,
        )
        if fig:
            save_path = output_path / "spx_optimal_path.html"
            fig.write_html(str(save_path))
            logger.info(f"Saved S&P 500 optimal path chart to {save_path}")
    
    logger.info(f"All example visualizations have been saved to {output_dir}")


if __name__ == '__main__':
    # Generate sample data
    generate_sample_data()
    
    # Generate example visualizations
    example_dir = settings.PROCESSED_DATA_DIR / 'examples'
    generate_example_visualizations(str(example_dir))
    
    print(f"Example visualizations saved to {example_dir}")
    print("You can now run the dashboard to explore the data interactively:")
    print("python src/run_dashboard.py")
