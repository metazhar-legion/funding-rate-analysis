"""
Visualization module for funding rate analysis.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.config import settings
from src.database.db import get_db
from src.database.models import Exchange, Asset, FundingRate

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger('visualizer')

# Set default style for matplotlib
plt.style.use('ggplot')
sns.set_theme(style="whitegrid")


class FundingRateVisualizer:
    """
    Class for visualizing funding rate data.
    """
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize the funding rate visualizer.
        
        Args:
            output_dir: Directory to save visualizations (default: settings.PROCESSED_DATA_DIR)
        """
        self.db = next(get_db())
        self.output_dir = Path(output_dir) if output_dir else settings.PROCESSED_DATA_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
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
    
    def plot_funding_rates_over_time(
        self,
        df: pd.DataFrame,
        asset_symbol: Optional[str] = None,
        category: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        use_plotly: bool = True,
        save_path: Optional[str] = None,
    ) -> Any:
        """
        Plot funding rates over time for a specific asset or category.
        
        Args:
            df: DataFrame with funding rate data
            asset_symbol: Asset symbol to plot (optional)
            category: Asset category to plot (optional)
            start_time: Start time for plotting
            end_time: End time for plotting
            use_plotly: Whether to use Plotly (interactive) or Matplotlib (static)
            save_path: Path to save the plot (optional)
            
        Returns:
            Plot figure
        """
        if df.empty:
            logger.warning("No data to plot")
            return None
        
        # Filter data
        plot_df = df.copy()
        
        if asset_symbol:
            plot_df = plot_df[plot_df['asset'] == asset_symbol]
        
        if category:
            plot_df = plot_df[plot_df['category'] == category]
        
        if start_time:
            plot_df = plot_df[plot_df['timestamp'] >= start_time]
        
        if end_time:
            plot_df = plot_df[plot_df['timestamp'] <= end_time]
        
        if plot_df.empty:
            logger.warning("No data to plot after filtering")
            return None
        
        # Determine title
        if asset_symbol:
            title = f"Funding Rates for {asset_symbol}"
        elif category:
            title = f"Funding Rates for {category} Assets"
        else:
            title = "Funding Rates Over Time"
        
        if use_plotly:
            # Create Plotly figure
            fig = px.line(
                plot_df,
                x='timestamp',
                y='rate_annualized',
                color='exchange',
                hover_data=['asset', 'rate', 'payment_interval_hours'],
                title=title,
                labels={
                    'timestamp': 'Date',
                    'rate_annualized': 'Annualized Funding Rate',
                    'exchange': 'Exchange',
                },
            )
            
            # Add horizontal line at y=0
            fig.add_shape(
                type='line',
                x0=plot_df['timestamp'].min(),
                y0=0,
                x1=plot_df['timestamp'].max(),
                y1=0,
                line=dict(color='gray', width=1, dash='dash'),
            )
            
            # Update layout
            fig.update_layout(
                xaxis_title='Date',
                yaxis_title='Annualized Funding Rate',
                legend_title='Exchange',
                hovermode='closest',
            )
            
            # Save if path provided
            if save_path:
                fig.write_html(save_path)
            
            return fig
        
        else:
            # Create Matplotlib figure
            plt.figure(figsize=(12, 6))
            
            # Plot each exchange
            for exchange in plot_df['exchange'].unique():
                exchange_df = plot_df[plot_df['exchange'] == exchange]
                plt.plot(
                    exchange_df['timestamp'],
                    exchange_df['rate_annualized'],
                    label=exchange,
                )
            
            # Add horizontal line at y=0
            plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
            
            # Add labels and title
            plt.xlabel('Date')
            plt.ylabel('Annualized Funding Rate')
            plt.title(title)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Format x-axis dates
            plt.gcf().autofmt_xdate()
            
            # Save if path provided
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            return plt.gcf()
    
    def plot_funding_rate_comparison(self,
        df: pd.DataFrame,
        asset_symbols: List[str],
        exchange: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        use_plotly: bool = True,
        save_path: Optional[str] = None,
    ) -> Any:
        """
        Plot funding rate comparison between different assets on the same exchange.
        
        Args:
            df: DataFrame with funding rate data
            asset_symbols: List of asset symbols to compare
            exchange: Exchange to filter by (optional)
            start_time: Start time for plotting
            end_time: End time for plotting
            use_plotly: Whether to use Plotly (interactive) or Matplotlib (static)
            save_path: Path to save the plot (optional)
            
        Returns:
            Plot figure
        """
        if df.empty:
            logger.warning("No data to plot")
            return None
        
        # Filter data
        plot_df = df[df['asset'].isin(asset_symbols)].copy()
        
        if exchange:
            plot_df = plot_df[plot_df['exchange'] == exchange]
        
        if start_time:
            plot_df = plot_df[plot_df['timestamp'] >= start_time]
        
        if end_time:
            plot_df = plot_df[plot_df['timestamp'] <= end_time]
        
        if plot_df.empty:
            logger.warning("No data to plot after filtering")
            return None
        
        # Determine title
        if exchange:
            title = f"Funding Rate Comparison on {exchange}"
        else:
            title = "Funding Rate Comparison Across Exchanges"
        
        if use_plotly:
            # Create Plotly figure
            fig = px.line(
                plot_df,
                x='timestamp',
                y='rate_annualized',
                color='asset',
                facet_col='exchange' if not exchange else None,
                hover_data=['exchange', 'rate', 'payment_interval_hours'],
                title=title,
                labels={
                    'timestamp': 'Date',
                    'rate_annualized': 'Annualized Funding Rate',
                    'asset': 'Asset',
                    'exchange': 'Exchange',
                },
            )
            
            # Add horizontal line at y=0
            fig.add_shape(
                type='line',
                x0=plot_df['timestamp'].min(),
                y0=0,
                x1=plot_df['timestamp'].max(),
                y1=0,
                line=dict(color='gray', width=1, dash='dash'),
            )
            
            # Update layout
            fig.update_layout(
                xaxis_title='Date',
                yaxis_title='Annualized Funding Rate',
                legend_title='Asset',
                hovermode='closest',
            )
            
            # Save if path provided
            if save_path:
                fig.write_html(save_path)
            
            return fig
        
        else:
            # Create Matplotlib figure
            plt.figure(figsize=(12, 6))
            
            # Plot each asset
            for asset in plot_df['asset'].unique():
                asset_df = plot_df[plot_df['asset'] == asset]
                plt.plot(
                    asset_df['timestamp'],
                    asset_df['rate_annualized'],
                    label=asset,
                )
            
            # Add horizontal line at y=0
            plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
            
            # Add labels and title
            plt.xlabel('Date')
            plt.ylabel('Annualized Funding Rate')
            plt.title(title)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Format x-axis dates
            plt.gcf().autofmt_xdate()
            
            # Save if path provided
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            return plt.gcf()
    
    def plot_funding_rate_heatmap(
        self,
        df: pd.DataFrame,
        category: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        save_path: Optional[str] = None,
    ) -> Any:
        """
        Plot a heatmap of funding rates across exchanges and assets.
        
        Args:
            df: DataFrame with funding rate data
            category: Asset category to filter by (optional)
            start_time: Start time for plotting
            end_time: End time for plotting
            save_path: Path to save the plot (optional)
            
        Returns:
            Plot figure
        """
        if df.empty:
            logger.warning("No data to plot")
            return None
        
        # Filter data
        plot_df = df.copy()
        
        if category:
            plot_df = plot_df[plot_df['category'] == category]
        
        if start_time:
            plot_df = plot_df[plot_df['timestamp'] >= start_time]
        
        if end_time:
            plot_df = plot_df[plot_df['timestamp'] <= end_time]
        
        if plot_df.empty:
            logger.warning("No data to plot after filtering")
            return None
        
        # Calculate average funding rate for each exchange-asset pair
        heatmap_df = plot_df.groupby(['exchange', 'asset'])['rate_annualized'].mean().reset_index()
        
        # Pivot data for heatmap
        pivot_df = heatmap_df.pivot(index='asset', columns='exchange', values='rate_annualized')
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Create heatmap
        ax = sns.heatmap(
            pivot_df,
            cmap='RdBu_r',
            center=0,
            annot=True,
            fmt='.4f',
            linewidths=0.5,
            cbar_kws={'label': 'Avg. Annualized Funding Rate'},
        )
        
        # Set title
        if category:
            plt.title(f"Average Funding Rates for {category} Assets")
        else:
            plt.title("Average Funding Rates Across Exchanges and Assets")
        
        # Rotate x-axis labels
        plt.xticks(rotation=45, ha='right')
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def plot_optimal_path(
        self,
        optimal_path_df: pd.DataFrame,
        asset: str,
        use_plotly: bool = True,
        save_path: Optional[str] = None,
    ) -> Any:
        """
        Plot the optimal exchange path for minimizing funding costs.
        
        Args:
            optimal_path_df: DataFrame with optimal path data
            asset: Asset to plot
            use_plotly: Whether to use Plotly (interactive) or Matplotlib (static)
            save_path: Path to save the plot (optional)
            
        Returns:
            Plot figure
        """
        if optimal_path_df.empty:
            logger.warning("No data to plot")
            return None
        
        # Filter data
        plot_df = optimal_path_df[optimal_path_df['asset'] == asset].copy()
        
        if plot_df.empty:
            logger.warning(f"No optimal path data for asset {asset}")
            return None
        
        title = f"Optimal Exchange Path for {asset}"
        
        if use_plotly:
            # Create Plotly figure
            fig = go.Figure()
            
            # Add line for best rate
            fig.add_trace(go.Scatter(
                x=plot_df['date'],
                y=plot_df['best_rate'],
                mode='lines+markers',
                name='Best Rate',
                line=dict(color='green', width=2),
                marker=dict(size=8),
            ))
            
            # Add line for worst rate
            fig.add_trace(go.Scatter(
                x=plot_df['date'],
                y=plot_df['worst_rate'],
                mode='lines+markers',
                name='Worst Rate',
                line=dict(color='red', width=2),
                marker=dict(size=8),
            ))
            
            # Add exchange annotations
            for i, row in plot_df.iterrows():
                fig.add_annotation(
                    x=row['date'],
                    y=row['best_rate'],
                    text=row['best_exchange'],
                    showarrow=True,
                    arrowhead=1,
                    ax=0,
                    ay=-40,
                )
            
            # Add horizontal line at y=0
            fig.add_shape(
                type='line',
                x0=plot_df['date'].min(),
                y0=0,
                x1=plot_df['date'].max(),
                y1=0,
                line=dict(color='gray', width=1, dash='dash'),
            )
            
            # Update layout
            fig.update_layout(
                title=title,
                xaxis_title='Date',
                yaxis_title='Annualized Funding Rate',
                legend_title='Rate Type',
                hovermode='closest',
            )
            
            # Save if path provided
            if save_path:
                fig.write_html(save_path)
            
            return fig
        
        else:
            # Create Matplotlib figure
            plt.figure(figsize=(12, 6))
            
            # Plot best and worst rates
            plt.plot(
                plot_df['date'],
                plot_df['best_rate'],
                'go-',
                label='Best Rate',
                linewidth=2,
                markersize=8,
            )
            
            plt.plot(
                plot_df['date'],
                plot_df['worst_rate'],
                'ro-',
                label='Worst Rate',
                linewidth=2,
                markersize=8,
            )
            
            # Add exchange annotations
            for i, row in plot_df.iterrows():
                plt.annotate(
                    row['best_exchange'],
                    (row['date'], row['best_rate']),
                    textcoords="offset points",
                    xytext=(0, -15),
                    ha='center',
                )
            
            # Add horizontal line at y=0
            plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
            
            # Add labels and title
            plt.xlabel('Date')
            plt.ylabel('Annualized Funding Rate')
            plt.title(title)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Format x-axis dates
            plt.gcf().autofmt_xdate()
            
            # Save if path provided
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            return plt.gcf()
