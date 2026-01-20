"""Database connection and session management."""

from typing import Generator, Optional
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.engine import Engine

from .models import Base
from config.settings import get_settings


_engine: Optional[Engine] = None
_SessionLocal: Optional[sessionmaker] = None


def get_engine() -> Engine:
    """Get or create the database engine."""
    global _engine
    if _engine is None:
        settings = get_settings()
        settings.ensure_directories()
        _engine = create_engine(
            settings.database_url,
            echo=False,  # Set to True for SQL debugging
            future=True,
        )
    return _engine


def get_session_factory() -> sessionmaker:
    """Get or create the session factory."""
    global _SessionLocal
    if _SessionLocal is None:
        engine = get_engine()
        _SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=engine,
        )
    return _SessionLocal


def get_session() -> Generator[Session, None, None]:
    """
    Get a database session.

    Usage:
        with next(get_session()) as session:
            # use session

    Or as a context manager:
        for session in get_session():
            # use session
    """
    SessionLocal = get_session_factory()
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def init_db() -> None:
    """Initialize the database, creating all tables."""
    engine = get_engine()
    Base.metadata.create_all(bind=engine)
    print(f"Database initialized at: {get_settings().database_url}")


def drop_db() -> None:
    """Drop all tables in the database. USE WITH CAUTION."""
    engine = get_engine()
    Base.metadata.drop_all(bind=engine)
    print("All tables dropped.")


def reset_db() -> None:
    """Reset the database by dropping and recreating all tables."""
    drop_db()
    init_db()
    print("Database reset complete.")
