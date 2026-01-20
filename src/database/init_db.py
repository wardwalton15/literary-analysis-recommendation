"""Database initialization script."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.database.connection import init_db, get_engine
from src.database.models import Base
from config.settings import get_settings


def main():
    """Initialize the database."""
    settings = get_settings()
    settings.ensure_directories()

    print(f"Initializing database at: {settings.database_url}")
    init_db()

    # Verify tables were created
    engine = get_engine()
    from sqlalchemy import inspect

    inspector = inspect(engine)
    tables = inspector.get_table_names()

    print(f"\nCreated {len(tables)} tables:")
    for table in tables:
        print(f"  - {table}")

    print("\nDatabase initialization complete!")


if __name__ == "__main__":
    main()
