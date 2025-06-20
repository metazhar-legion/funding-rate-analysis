#!/usr/bin/env python
"""
Main entry point for the visualization module.
"""

import argparse
import logging
import sys
from pathlib import Path

from src.config import settings
from src.database.db import init_db
from src.visualization.examples import generate_sample_data, generate_example_visualizations
from src.visualization.dashboard import FundingRateDashboard

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger('visualization')


def main():
    """Main entry point for the visualization module."""
    parser = argparse.ArgumentParser(description='Funding Rate Analysis Visualization Tools')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Dashboard command
    dashboard_parser = subparsers.add_parser('dashboard', help='Run the interactive dashboard')
    dashboard_parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to run the dashboard on')
    dashboard_parser.add_argument('--port', type=int, default=8050, help='Port to run the dashboard on')
    
    # Generate sample data command
    sample_parser = subparsers.add_parser('sample', help='Generate sample data for testing')
    
    # Generate examples command
    examples_parser = subparsers.add_parser('examples', help='Generate example visualizations')
    examples_parser.add_argument('--output-dir', type=str, 
                               default=str(settings.PROCESSED_DATA_DIR / 'examples'),
                               help='Directory to save example visualizations')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create necessary directories
    settings.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Initialize database
    init_db()
    
    # Handle commands
    if args.command == 'dashboard':
        logger.info(f"Starting dashboard on {args.host}:{args.port}...")
        dashboard = FundingRateDashboard(host=args.host, port=args.port)
        dashboard.run()
        
    elif args.command == 'sample':
        logger.info("Generating sample data...")
        generate_sample_data()
        logger.info("Sample data generation complete!")
        
    elif args.command == 'examples':
        logger.info(f"Generating example visualizations in {args.output_dir}...")
        generate_example_visualizations(args.output_dir)
        logger.info("Example visualization generation complete!")
        
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
