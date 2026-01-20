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
    Get a database session as a context manager.

    Usage:
        with get_session() as session:
            # use session
    """
    SessionLocal = get_session_factory()
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


class SessionContext:
    """Context manager for database sessions."""

    def __init__(self):
        self._session = None
        self._session_factory = get_session_factory()

    def __enter__(self) -> Session:
        self._session = self._session_factory()
        return self._session

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._session:
            self._session.close()
        return False


def get_session_context() -> SessionContext:
    """Get a database session context manager.

    Usage:
        with get_session_context() as session:
            # use session
    """
    return SessionContext()


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
