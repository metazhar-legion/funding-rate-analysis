"""
Database models for the funding rate analysis application.
"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, UniqueConstraint
from sqlalchemy.orm import relationship

from src.database.db import Base


class Exchange(Base):
    """Exchange model representing a DEX that offers perpetual contracts."""
    __tablename__ = "exchanges"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    code = Column(String, unique=True, index=True)
    url = Column(String)
    description = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    assets = relationship("Asset", back_populates="exchange")
    funding_rates = relationship("FundingRate", back_populates="exchange")

    def __repr__(self):
        return f"<Exchange {self.name}>"


class Asset(Base):
    """Asset model representing a tradable asset on an exchange."""
    __tablename__ = "assets"

    id = Column(Integer, primary_key=True, index=True)
    exchange_id = Column(Integer, ForeignKey("exchanges.id"), index=True)
    symbol = Column(String, index=True)
    name = Column(String)
    asset_type = Column(String, index=True)  # equity_index, stock, treasury, commodity, etc.
    category = Column(String, index=True)  # For grouping related assets
    is_active = Column(Integer, default=1)  # 1 = active, 0 = inactive
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    exchange = relationship("Exchange", back_populates="assets")
    funding_rates = relationship("FundingRate", back_populates="asset")

    # Unique constraint for exchange_id + symbol
    __table_args__ = (
        UniqueConstraint('exchange_id', 'symbol', name='uix_exchange_symbol'),
    )

    def __repr__(self):
        return f"<Asset {self.symbol} on {self.exchange.name}>"


class FundingRate(Base):
    """Funding rate model representing a funding rate at a specific time."""
    __tablename__ = "funding_rates"

    id = Column(Integer, primary_key=True, index=True)
    exchange_id = Column(Integer, ForeignKey("exchanges.id"), index=True)
    asset_id = Column(Integer, ForeignKey("assets.id"), index=True)
    timestamp = Column(DateTime, index=True)
    rate = Column(Float)  # Funding rate as a decimal (e.g., 0.0001 for 0.01%)
    rate_annualized = Column(Float, nullable=True)  # Annualized funding rate for comparison
    payment_interval_hours = Column(Float)  # How often funding is paid (in hours)
    open_interest_long = Column(Float, nullable=True)  # Open interest for long positions
    open_interest_short = Column(Float, nullable=True)  # Open interest for short positions
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    exchange = relationship("Exchange", back_populates="funding_rates")
    asset = relationship("Asset", back_populates="funding_rates")

    # Unique constraint for exchange_id + asset_id + timestamp
    __table_args__ = (
        UniqueConstraint('exchange_id', 'asset_id', 'timestamp', name='uix_exchange_asset_timestamp'),
    )

    def __repr__(self):
        return f"<FundingRate {self.asset.symbol} on {self.exchange.name} at {self.timestamp}>"


class OptimizationResult(Base):
    """Optimization result model for storing optimal strategies."""
    __tablename__ = "optimization_results"

    id = Column(Integer, primary_key=True, index=True)
    asset_category = Column(String, index=True)
    asset_symbol = Column(String, index=True)
    start_date = Column(DateTime)
    end_date = Column(DateTime)
    strategy_name = Column(String)
    strategy_description = Column(String)
    annualized_funding_cost = Column(Float)
    annualized_return = Column(Float)
    net_return = Column(Float)
    risk_metric = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<OptimizationResult {self.asset_symbol} {self.strategy_name}>"
