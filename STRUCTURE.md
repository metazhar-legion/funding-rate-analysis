# Project Structure

```
funding-rate-analysis/
├── .env                    # Environment variables (API keys, etc.)
├── .env.example            # Example environment file template
├── .gitignore              # Git ignore file
├── README.md               # Project documentation
├── STRUCTURE.md            # This file - project structure documentation
├── requirements.txt        # Python dependencies
├── src/                    # Source code
│   ├── __init__.py         # Makes src a Python package
│   ├── config/             # Configuration files and settings
│   │   ├── __init__.py
│   │   └── settings.py     # Application settings and constants
│   ├── data_collection/    # Data collection modules
│   │   ├── __init__.py
│   │   ├── collector.py    # Main data collection orchestrator
│   │   ├── exchanges/      # Exchange-specific API clients
│   │   │   ├── __init__.py
│   │   │   ├── base.py     # Base exchange client class
│   │   │   ├── dydx.py     # dYdX exchange client
│   │   │   ├── gmx.py      # GMX exchange client
│   │   │   ├── synthetix.py # Synthetix exchange client
│   │   │   └── ...         # Other exchange clients
│   │   └── models.py       # Data models for collected data
│   ├── database/           # Database interaction
│   │   ├── __init__.py
│   │   ├── db.py           # Database connection and setup
│   │   └── models.py       # Database models
│   ├── analysis/           # Analysis modules
│   │   ├── __init__.py
│   │   ├── analyzer.py     # Main analysis orchestrator
│   │   ├── funding_rate.py # Funding rate analysis
│   │   ├── correlation.py  # Correlation analysis between exchanges
│   │   ├── optimization.py # Optimization strategies
│   │   └── metrics.py      # Analysis metrics and calculations
│   ├── visualization/      # Visualization modules
│   │   ├── __init__.py
│   │   ├── visualizer.py   # Main visualization orchestrator
│   │   ├── charts.py       # Chart generation
│   │   └── dashboard.py    # Interactive dashboard (optional)
│   └── utils/              # Utility functions
│       ├── __init__.py
│       ├── logger.py       # Logging utilities
│       ├── helpers.py      # Helper functions
│       └── constants.py    # Constant values
├── tests/                  # Test suite
│   ├── __init__.py
│   ├── test_data_collection.py
│   ├── test_analysis.py
│   └── test_visualization.py
└── data/                   # Data storage (can be gitignored)
    ├── raw/                # Raw data from exchanges
    └── processed/          # Processed analysis results
```

## Key Components

### Data Collection
- `collector.py`: Orchestrates the data collection process
- Exchange-specific clients in `exchanges/` directory
- Handles API rate limiting, pagination, and error handling

### Database
- Stores historical funding rate data
- Supports both SQLite (development) and PostgreSQL (production)
- Implements data models for exchanges, assets, and funding rates

### Analysis
- `analyzer.py`: Main analysis entry point
- Specialized analysis modules for different metrics
- Optimization strategies for minimizing funding costs

### Visualization
- `visualizer.py`: Creates visualizations of funding rate data
- Supports various chart types for different analysis needs
- Optional interactive dashboard for real-time monitoring

## Data Flow

1. **Collection**: Exchange clients fetch funding rate data
2. **Storage**: Data is normalized and stored in the database
3. **Analysis**: Analytical tools process the historical data
4. **Visualization**: Results are presented in visual format
5. **Optimization**: Strategies are generated based on analysis
