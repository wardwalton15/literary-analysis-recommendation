"""Pytest configuration and fixtures."""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture(scope="session")
def project_root_path():
    """Return the project root path."""
    return project_root


@pytest.fixture
def sample_gutenberg_text():
    """Sample text with Gutenberg headers/footers for testing."""
    return """The Project Gutenberg eBook of Test Book

This eBook is for the use of anyone anywhere in the United States.

*** START OF THE PROJECT GUTENBERG EBOOK TEST BOOK ***

Title: Test Book
Author: Test Author

CHAPTER I

This is the actual content of the book. It contains multiple paragraphs
and represents the kind of text we would find in a classic literary work.

The narrative continues here with more interesting content that would
normally span many pages in a real book.

CHAPTER II

The second chapter begins with new adventures and developments in the story.

*** END OF THE PROJECT GUTENBERG EBOOK TEST BOOK ***

End of the Project Gutenberg EBook

This file should be named testbook.txt
"""


@pytest.fixture
def sample_book_metadata():
    """Sample book metadata for testing."""
    return {
        "gutenberg_id": 5200,
        "title": "The Metamorphosis",
        "author": "Franz Kafka",
        "language": "en",
        "subjects": ["Fiction", "Psychological fiction", "Metamorphosis -- Fiction"],
        "publication_year": 1915,
    }
