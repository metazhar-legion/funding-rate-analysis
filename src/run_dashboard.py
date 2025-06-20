#!/usr/bin/env python
"""
Script to run the funding rate analysis dashboard.
"""

import argparse
import logging
from datetime import datetime, timedelta

from src.config import settings
from src.database.db import init_db
from src.visualization.dashboard import FundingRateDashboard

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger('run_dashboard')


def main():
    """Run the funding rate analysis dashboard."""
    parser = argparse.ArgumentParser(description='Run the funding rate analysis dashboard')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to run the dashboard on')
    parser.add_argument('--port', type=int, default=8050, help='Port to run the dashboard on')
    args = parser.parse_args()
    
    # Initialize database
    logger.info("Initializing database...")
    init_db()
    
    # Run dashboard
    logger.info(f"Starting dashboard on {args.host}:{args.port}...")
    dashboard = FundingRateDashboard(host=args.host, port=args.port)
    dashboard.run()


if __name__ == '__main__':
    main()
