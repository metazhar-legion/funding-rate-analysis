#!/usr/bin/env python
"""
Script to generate funding rate visualizations and save them to files.
"""

import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from src.config import settings
from src.database.db import init_db, get_db
from src.database.models import Exchange, Asset, FundingRate
from src.visualization.visualizer import FundingRateVisualizer

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger('generate_visualizations')


def generate_visualizations(output_dir: str, days: int = 30, use_plotly: bool = True):
    """
    Generate funding rate visualizations and save them to files.
    
    Args:
        output_dir: Directory to save visualizations
        days: Number of days of data to include
        use_plotly: Whether to use Plotly (interactive) or Matplotlib (static)
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize visualizer
    visualizer = FundingRateVisualizer(output_dir=output_dir)
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    logger.info(f"Generating visualizations for period: {start_date} to {end_date}")
    
    # Get all funding rate data
    df = visualizer.load_funding_rates(start_time=start_date, end_time=end_date)
    
    if df.empty:
        logger.warning("No funding rate data found in the specified date range")
        return
    
    # Get unique categories, assets, and exchanges
    categories = df['category'].unique()
    assets = df['asset'].unique()
    exchanges = df['exchange'].unique()
    
    logger.info(f"Found {len(categories)} categories, {len(assets)} assets, and {len(exchanges)} exchanges")
    
    # 1. Generate funding rate over time charts for each category
    for category in categories:
        if pd.isna(category):
            continue
            
        logger.info(f"Generating funding rate chart for category: {category}")
        
        # Filter data for this category
        category_df = df[df['category'] == category]
        
        # Generate plot
        fig = visualizer.plot_funding_rates_over_time(
            category_df,
            category=category,
            start_time=start_date,
            end_time=end_date,
            use_plotly=use_plotly,
        )
        
        if fig:
            # Save plot
            file_ext = 'html' if use_plotly else 'png'
            save_path = output_path / f"funding_rates_{category.lower().replace(' ', '_')}.{file_ext}"
            
            if use_plotly:
                fig.write_html(str(save_path))
            else:
                fig.savefig(str(save_path), dpi=300, bbox_inches='tight')
                
            logger.info(f"Saved funding rate chart to {save_path}")
    
    # 2. Generate asset comparison charts for each category
    for category in categories:
        if pd.isna(category):
            continue
            
        # Filter data for this category
        category_df = df[df['category'] == category]
        category_assets = category_df['asset'].unique()
        
        if len(category_assets) < 2:
            continue
        
        logger.info(f"Generating asset comparison chart for category: {category}")
        
        # Generate plot
        fig = visualizer.plot_funding_rate_comparison(
            category_df,
            asset_symbols=list(category_assets),
            start_time=start_date,
            end_time=end_date,
            use_plotly=use_plotly,
        )
        
        if fig:
            # Save plot
            file_ext = 'html' if use_plotly else 'png'
            save_path = output_path / f"asset_comparison_{category.lower().replace(' ', '_')}.{file_ext}"
            
            if use_plotly:
                fig.write_html(str(save_path))
            else:
                fig.savefig(str(save_path), dpi=300, bbox_inches='tight')
                
            logger.info(f"Saved asset comparison chart to {save_path}")
    
    # 3. Generate funding rate heatmap
    logger.info("Generating funding rate heatmap")
    
    fig = visualizer.plot_funding_rate_heatmap(
        df,
        start_time=start_date,
        end_time=end_date,
    )
    
    if fig:
        # Save plot
        save_path = output_path / "funding_rate_heatmap.png"
        fig.savefig(str(save_path), dpi=300, bbox_inches='tight')
        logger.info(f"Saved funding rate heatmap to {save_path}")
    
    # 4. Generate optimal path visualizations (if data available)
    try:
        # Query for optimization results
        db = next(get_db())
        optimization_results = db.query(
            "SELECT asset, date, best_exchange, best_rate, worst_exchange, worst_rate FROM optimization_results"
        ).fetchall()
        
        if optimization_results:
            # Convert to DataFrame
            optimal_path_df = pd.DataFrame(
                optimization_results,
                columns=['asset', 'date', 'best_exchange', 'best_rate', 'worst_exchange', 'worst_rate']
            )
            
            # Generate optimal path charts for each asset
            for asset in optimal_path_df['asset'].unique():
                logger.info(f"Generating optimal path chart for asset: {asset}")
                
                fig = visualizer.plot_optimal_path(
                    optimal_path_df,
                    asset=asset,
                    use_plotly=use_plotly,
                )
                
                if fig:
                    # Save plot
                    file_ext = 'html' if use_plotly else 'png'
                    save_path = output_path / f"optimal_path_{asset.lower().replace(' ', '_')}.{file_ext}"
                    
                    if use_plotly:
                        fig.write_html(str(save_path))
                    else:
                        fig.savefig(str(save_path), dpi=300, bbox_inches='tight')
                        
                    logger.info(f"Saved optimal path chart to {save_path}")
    except Exception as e:
        logger.warning(f"Could not generate optimal path visualizations: {e}")
    
    logger.info(f"All visualizations have been saved to {output_dir}")


def main():
    """Run the visualization generator."""
    parser = argparse.ArgumentParser(description='Generate funding rate visualizations')
    parser.add_argument('--output-dir', type=str, default=str(settings.PROCESSED_DATA_DIR / 'visualizations'),
                        help='Directory to save visualizations')
    parser.add_argument('--days', type=int, default=30, help='Number of days of data to include')
    parser.add_argument('--static', action='store_true', help='Generate static (Matplotlib) instead of interactive (Plotly) visualizations')
    args = parser.parse_args()
    
    # Initialize database
    logger.info("Initializing database...")
    init_db()
    
    # Generate visualizations
    generate_visualizations(
        output_dir=args.output_dir,
        days=args.days,
        use_plotly=not args.static,
    )


if __name__ == '__main__':
    main()
