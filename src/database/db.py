"""
Database connection and setup module.
"""

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session

from src.config import settings

# Create SQLAlchemy engine
engine = create_engine(
    settings.DATABASE_URL,
    echo=(settings.LOG_LEVEL == 'DEBUG'),
    pool_pre_ping=True,
    connect_args={"check_same_thread": False} if settings.DB_TYPE == 'sqlite' else {}
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Session = scoped_session(SessionLocal)

# Base class for all models
Base = declarative_base()


def get_db():
    """
    Get database session.
    
    Yields:
        Session: Database session
    """
    db = Session()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """
    Initialize database by creating all tables.
    """
    # Import models here to avoid circular imports
    from src.database.models import Exchange, Asset, FundingRate  # noqa
    
    # Create tables
    Base.metadata.create_all(bind=engine)


def close_db():
    """
    Close database connection.
    """
    Session.remove()
