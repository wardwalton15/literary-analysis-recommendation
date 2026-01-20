"""Tests for the Gutenberg scraper module."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile

from src.scrapers.gutenberg_scraper import GutenbergScraper, GutenbergBatchDownloader, BookInfo


class TestGutenbergScraper:
    """Tests for GutenbergScraper class."""

    def test_init_default_delay(self):
        """Test scraper initialization with default delay."""
        scraper = GutenbergScraper()
        assert scraper.delay == 1.0

    def test_init_custom_delay(self):
        """Test scraper initialization with custom delay."""
        scraper = GutenbergScraper(delay=2.5)
        assert scraper.delay == 2.5

    def test_clean_gutenberg_text_removes_header(self):
        """Test that Gutenberg headers are removed."""
        scraper = GutenbergScraper()
        raw_text = """Some header stuff

*** START OF THE PROJECT GUTENBERG EBOOK THE METAMORPHOSIS ***

The actual book content starts here.
This is the story text.

*** END OF THE PROJECT GUTENBERG EBOOK THE METAMORPHOSIS ***

Some footer stuff"""

        cleaned = scraper.clean_gutenberg_text(raw_text)

        assert "START OF THE PROJECT GUTENBERG" not in cleaned
        assert "END OF THE PROJECT GUTENBERG" not in cleaned
        assert "actual book content" in cleaned
        assert "story text" in cleaned

    def test_clean_gutenberg_text_handles_no_markers(self):
        """Test cleaning text that has no Gutenberg markers."""
        scraper = GutenbergScraper()
        text = "This is just plain text without markers."
        cleaned = scraper.clean_gutenberg_text(text)
        assert cleaned == text

    def test_clean_gutenberg_text_strips_whitespace(self):
        """Test that extra whitespace is stripped."""
        scraper = GutenbergScraper()
        raw_text = """

*** START OF THE PROJECT GUTENBERG EBOOK ***

   Content with spaces

*** END OF THE PROJECT GUTENBERG EBOOK ***

"""
        cleaned = scraper.clean_gutenberg_text(raw_text)
        assert not cleaned.startswith("\n")
        assert not cleaned.endswith("\n")


class TestBookInfo:
    """Tests for BookInfo dataclass."""

    def test_bookinfo_creation(self):
        """Test creating a BookInfo object."""
        info = BookInfo(
            gutenberg_id=5200,
            title="The Metamorphosis",
            author="Franz Kafka",
            language="en",
        )
        assert info.gutenberg_id == 5200
        assert info.title == "The Metamorphosis"
        assert info.author == "Franz Kafka"

    def test_bookinfo_to_dict(self):
        """Test converting BookInfo to dictionary."""
        info = BookInfo(
            gutenberg_id=5200,
            title="The Metamorphosis",
            author="Franz Kafka",
        )
        d = info.to_dict()
        assert d["gutenberg_id"] == 5200
        assert d["title"] == "The Metamorphosis"
        assert d["author"] == "Franz Kafka"

    def test_bookinfo_defaults(self):
        """Test BookInfo default values."""
        info = BookInfo(gutenberg_id=1, title="Test")
        assert info.author is None
        assert info.language is None
        assert info.subjects is None


class TestGutenbergBatchDownloader:
    """Tests for GutenbergBatchDownloader class."""

    def test_init(self):
        """Test batch downloader initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            downloader = GutenbergBatchDownloader(
                book_ids=[1, 2, 3],
                output_dir=Path(tmpdir),
                delay=0.5,
            )
            assert len(downloader.book_ids) == 3
            assert downloader.output_dir == Path(tmpdir)

    def test_already_downloaded_false(self):
        """Test checking for non-existent file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            downloader = GutenbergBatchDownloader(
                book_ids=[1],
                output_dir=Path(tmpdir),
            )
            assert not downloader._already_downloaded(1)

    def test_already_downloaded_true(self):
        """Test checking for existing file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            (tmppath / "pg1.txt").write_text("content")

            downloader = GutenbergBatchDownloader(
                book_ids=[1],
                output_dir=tmppath,
            )
            assert downloader._already_downloaded(1)

    def test_text_path(self):
        """Test text file path generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            downloader = GutenbergBatchDownloader(
                book_ids=[5200],
                output_dir=Path(tmpdir),
            )
            path = downloader._text_path(5200)
            assert path.name == "pg5200.txt"

    def test_metadata_path(self):
        """Test metadata file path generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            downloader = GutenbergBatchDownloader(
                book_ids=[5200],
                output_dir=Path(tmpdir),
            )
            path = downloader._metadata_path(5200)
            assert path.name == "pg5200_meta.json"
