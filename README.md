# Funding Rate Analysis

A Python application for collecting, analyzing, and visualizing funding rate data from various decentralized exchanges (DEXes) that offer perpetual contracts for equity indices (like S&P500) and other real-world assets (RWAs) such as stocks, treasuries, and commodities.

## Purpose

This application aims to:

1. Collect historical funding rate data from multiple DEXes
2. Analyze funding rate patterns and trends
3. Identify optimal strategies for holding assets while minimizing funding rate expenses
4. Visualize funding rate comparisons across different platforms
5. Help users simulate 1:1 asset holdings using leveraged exposure through perpetuals

## Features

- Data collection from multiple DEX APIs
- Historical funding rate storage and management
- Analysis tools for funding rate patterns
- Visualization of funding rate trends
- Strategy optimization for minimizing costs

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/funding-rate-analysis.git
cd funding-rate-analysis

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env file with your API keys
```

## Usage

```bash
# Collect funding rate data
python -m src.data_collection.collector

# Run analysis
python -m src.analysis.analyzer

# Generate visualizations
python -m src.visualization.visualizer
```

## Project Structure

See [STRUCTURE.md](STRUCTURE.md) for details on the project organization.

## License

MIT
