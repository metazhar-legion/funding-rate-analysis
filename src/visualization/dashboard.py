"""
Interactive dashboard for funding rate analysis.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd
import dash
from dash import dcc, html, callback
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.config import settings
from src.visualization.visualizer import FundingRateVisualizer

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger('dashboard')


class FundingRateDashboard:
    """
    Interactive dashboard for funding rate analysis using Dash.
    """
    
    def __init__(self, host: str = '0.0.0.0', port: int = 8050):
        """
        Initialize the funding rate dashboard.
        
        Args:
            host: Host to run the dashboard on
            port: Port to run the dashboard on
        """
        self.host = host
        self.port = port
        self.visualizer = FundingRateVisualizer()
        self.app = dash.Dash(__name__, title="Funding Rate Analysis Dashboard")
        self._setup_layout()
        self._setup_callbacks()
    
    def _setup_layout(self):
        """Set up the dashboard layout."""
        self.app.layout = html.Div([
            # Header
            html.Div([
                html.H1("Funding Rate Analysis Dashboard"),
                html.P("Analyze and visualize funding rates across decentralized exchanges"),
            ], className="header"),
            
            # Filters
            html.Div([
                html.Div([
                    html.Label("Date Range:"),
                    dcc.DatePickerRange(
                        id='date-range',
                        start_date_placeholder_text="Start Date",
                        end_date_placeholder_text="End Date",
                        calendar_orientation='horizontal',
                    ),
                ], className="filter-item"),
                
                html.Div([
                    html.Label("Exchange:"),
                    dcc.Dropdown(
                        id='exchange-dropdown',
                        options=[],  # Will be populated in callback
                        multi=True,
                        placeholder="Select Exchange(s)",
                    ),
                ], className="filter-item"),
                
                html.Div([
                    html.Label("Asset Category:"),
                    dcc.Dropdown(
                        id='category-dropdown',
                        options=[],  # Will be populated in callback
                        placeholder="Select Asset Category",
                    ),
                ], className="filter-item"),
                
                html.Div([
                    html.Label("Assets:"),
                    dcc.Dropdown(
                        id='asset-dropdown',
                        options=[],  # Will be populated in callback
                        multi=True,
                        placeholder="Select Asset(s)",
                    ),
                ], className="filter-item"),
                
                html.Button('Apply Filters', id='apply-filters', n_clicks=0),
            ], className="filters-container"),
            
            # Main content
            html.Div([
                # Tab navigation
                dcc.Tabs([
                    # Funding Rate Trends Tab
                    dcc.Tab(label="Funding Rate Trends", children=[
                        html.Div([
                            dcc.Graph(id='funding-rate-trend-chart'),
                        ], className="chart-container"),
                    ]),
                    
                    # Asset Comparison Tab
                    dcc.Tab(label="Asset Comparison", children=[
                        html.Div([
                            dcc.Graph(id='asset-comparison-chart'),
                        ], className="chart-container"),
                    ]),
                    
                    # Exchange Comparison Tab
                    dcc.Tab(label="Exchange Comparison", children=[
                        html.Div([
                            dcc.Graph(id='exchange-comparison-chart'),
                        ], className="chart-container"),
                    ]),
                    
                    # Heatmap Tab
                    dcc.Tab(label="Funding Rate Heatmap", children=[
                        html.Div([
                            dcc.Graph(id='funding-rate-heatmap'),
                        ], className="chart-container"),
                    ]),
                    
                    # Optimal Path Tab
                    dcc.Tab(label="Optimal Exchange Path", children=[
                        html.Div([
                            dcc.Graph(id='optimal-path-chart'),
                        ], className="chart-container"),
                    ]),
                    
                    # Statistics Tab
                    dcc.Tab(label="Statistics", children=[
                        html.Div([
                            html.Div(id='statistics-table'),
                        ], className="chart-container"),
                    ]),
                ], id='tabs'),
            ], className="main-content"),
            
            # Store for data
            dcc.Store(id='funding-data-store'),
            dcc.Store(id='optimal-path-store'),
            
            # Footer
            html.Div([
                html.P("Funding Rate Analysis Tool - © 2025"),
            ], className="footer"),
        ], className="dashboard-container")
    
    def _setup_callbacks(self):
        """Set up the dashboard callbacks."""
        
        @self.app.callback(
            [
                Output('exchange-dropdown', 'options'),
                Output('category-dropdown', 'options'),
            ],
            [Input('apply-filters', 'n_clicks')],
            prevent_initial_call=False
        )
        def load_initial_data(_):
            """Load initial dropdown data."""
            try:
                # Load exchanges
                exchanges = self.visualizer.db.query(
                    "SELECT DISTINCT exchange_name, exchange_code FROM exchanges"
                ).fetchall()
                exchange_options = [
                    {'label': f"{row[0]} ({row[1]})", 'value': row[0]}
                    for row in exchanges
                ]
                
                # Load categories
                categories = self.visualizer.db.query(
                    "SELECT DISTINCT category FROM assets WHERE category IS NOT NULL"
                ).fetchall()
                category_options = [
                    {'label': cat[0], 'value': cat[0]}
                    for cat in categories
                ]
                
                return exchange_options, category_options
                
            except Exception as e:
                logger.error(f"Error loading initial data: {e}")
                return [], []
        
        @self.app.callback(
            Output('asset-dropdown', 'options'),
            [
                Input('category-dropdown', 'value'),
                Input('exchange-dropdown', 'value'),
            ],
            prevent_initial_call=True
        )
        def update_asset_options(category, exchanges):
            """Update asset options based on selected category and exchanges."""
            try:
                query = "SELECT DISTINCT a.symbol, a.name FROM assets a JOIN exchanges e ON a.exchange_id = e.id WHERE 1=1"
                params = {}
                
                if category:
                    query += " AND a.category = :category"
                    params['category'] = category
                
                if exchanges and len(exchanges) > 0:
                    query += " AND e.name IN :exchanges"
                    params['exchanges'] = tuple(exchanges) if len(exchanges) > 1 else f"('{exchanges[0]}')"
                
                assets = self.visualizer.db.query(query, params).fetchall()
                
                asset_options = [
                    {'label': f"{row[0]} ({row[1]})", 'value': row[0]}
                    for row in assets
                ]
                
                return asset_options
                
            except Exception as e:
                logger.error(f"Error updating asset options: {e}")
                return []
        
        @self.app.callback(
            Output('funding-data-store', 'data'),
            [Input('apply-filters', 'n_clicks')],
            [
                State('date-range', 'start_date'),
                State('date-range', 'end_date'),
                State('exchange-dropdown', 'value'),
                State('category-dropdown', 'value'),
                State('asset-dropdown', 'value'),
            ],
            prevent_initial_call=True
        )
        def load_funding_data(n_clicks, start_date, end_date, exchanges, category, assets):
            """Load funding rate data based on filters."""
            if n_clicks == 0:
                return {}
            
            try:
                # Convert string dates to datetime
                start_time = pd.to_datetime(start_date) if start_date else None
                end_time = pd.to_datetime(end_date) if end_date else None
                
                # Load data
                df = self.visualizer.load_funding_rates(
                    start_time=start_time,
                    end_time=end_time,
                    exchange_codes=exchanges,
                    asset_symbols=assets,
                    categories=[category] if category else None,
                )
                
                # Convert to dict for storage
                return df.to_dict('records') if not df.empty else {}
                
            except Exception as e:
                logger.error(f"Error loading funding data: {e}")
                return {}
        
        @self.app.callback(
            Output('funding-rate-trend-chart', 'figure'),
            [Input('funding-data-store', 'data')],
            prevent_initial_call=True
        )
        def update_trend_chart(data):
            """Update funding rate trend chart."""
            if not data:
                return go.Figure().update_layout(title="No data available")
            
            try:
                df = pd.DataFrame(data)
                
                # Create figure
                fig = px.line(
                    df,
                    x='timestamp',
                    y='rate_annualized',
                    color='exchange',
                    facet_row='asset',
                    hover_data=['asset', 'rate', 'payment_interval_hours'],
                    title="Funding Rate Trends",
                    labels={
                        'timestamp': 'Date',
                        'rate_annualized': 'Annualized Funding Rate',
                        'exchange': 'Exchange',
                    },
                )
                
                # Add horizontal line at y=0
                fig.add_shape(
                    type='line',
                    x0=df['timestamp'].min(),
                    y0=0,
                    x1=df['timestamp'].max(),
                    y1=0,
                    line=dict(color='gray', width=1, dash='dash'),
                )
                
                # Update layout
                fig.update_layout(
                    height=600,
                    xaxis_title='Date',
                    yaxis_title='Annualized Funding Rate',
                    legend_title='Exchange',
                    hovermode='closest',
                )
                
                return fig
                
            except Exception as e:
                logger.error(f"Error updating trend chart: {e}")
                return go.Figure().update_layout(title=f"Error: {str(e)}")
        
        # Add more callbacks for other tabs...
    
    def run(self):
        """Run the dashboard."""
        self.app.run_server(host=self.host, port=self.port, debug=True)


if __name__ == '__main__':
    dashboard = FundingRateDashboard()
    dashboard.run()
