"""Application settings and configuration."""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Settings:
    """Application settings."""

    # Paths
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    data_dir: Path = field(init=False)
    raw_data_dir: Path = field(init=False)
    processed_data_dir: Path = field(init=False)
    external_data_dir: Path = field(init=False)

    # Database
    database_url: str = field(init=False)

    # Scraping
    scrape_delay: float = 1.0
    user_agent: str = "LiteraryAnalysisProject/1.0 (educational research)"
    max_retries: int = 3
    retry_backoff: float = 2.0

    # Gutenberg
    gutenberg_base_url: str = "https://www.gutenberg.org"
    gutenberg_text_url: str = "https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
    gutenberg_ebook_url: str = "https://www.gutenberg.org/ebooks/{book_id}"

    def __post_init__(self):
        """Initialize derived paths."""
        self.data_dir = self.project_root / "data"
        self.raw_data_dir = self.data_dir / "raw"
        self.processed_data_dir = self.data_dir / "processed"
        self.external_data_dir = self.data_dir / "external"

        # Database path
        db_path = self.data_dir / "literary_analysis.db"
        self.database_url = f"sqlite:///{db_path}"

    def ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        for dir_path in [
            self.raw_data_dir,
            self.processed_data_dir,
            self.external_data_dir,
            self.raw_data_dir / "kafka",
            self.raw_data_dir / "corpus",
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get or create the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
