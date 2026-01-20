"""Tests for database models."""

import pytest
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database.models import Base, Author, Book, BookMetadata, AnalysisResult, EmbeddingMetadata


@pytest.fixture
def engine():
    """Create an in-memory SQLite database for testing."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return engine


@pytest.fixture
def session(engine):
    """Create a database session for testing."""
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


class TestAuthorModel:
    """Tests for the Author model."""

    def test_create_author(self, session):
        """Test creating an author."""
        author = Author(
            name="Franz Kafka",
            birth_year=1883,
            death_year=1924,
            nationality="Austrian",
        )
        session.add(author)
        session.commit()

        assert author.id is not None
        assert author.name == "Franz Kafka"
        assert author.birth_year == 1883

    def test_author_repr(self, session):
        """Test author string representation."""
        author = Author(name="Franz Kafka")
        session.add(author)
        session.commit()

        repr_str = repr(author)
        assert "Franz Kafka" in repr_str
        assert str(author.id) in repr_str

    def test_author_books_relationship(self, session):
        """Test author-books relationship."""
        author = Author(name="Franz Kafka")
        book = Book(title="The Metamorphosis", author=author)

        session.add_all([author, book])
        session.commit()

        assert len(author.books) == 1
        assert author.books[0].title == "The Metamorphosis"


class TestBookModel:
    """Tests for the Book model."""

    def test_create_book(self, session):
        """Test creating a book."""
        book = Book(
            gutenberg_id=5200,
            title="The Metamorphosis",
            language="en",
        )
        session.add(book)
        session.commit()

        assert book.id is not None
        assert book.gutenberg_id == 5200
        assert book.title == "The Metamorphosis"

    def test_book_with_author(self, session):
        """Test creating a book with an author."""
        author = Author(name="Franz Kafka")
        book = Book(
            title="The Trial",
            author=author,
            gutenberg_id=7849,
        )

        session.add(book)
        session.commit()

        assert book.author_id == author.id
        assert book.author.name == "Franz Kafka"

    def test_book_word_count(self, session):
        """Test book with word count."""
        book = Book(
            title="Test Book",
            full_text="One two three four five",
            word_count=5,
        )
        session.add(book)
        session.commit()

        assert book.word_count == 5

    def test_book_unique_gutenberg_id(self, session):
        """Test that gutenberg_id must be unique."""
        book1 = Book(title="Book 1", gutenberg_id=5200)
        book2 = Book(title="Book 2", gutenberg_id=5200)

        session.add(book1)
        session.commit()

        session.add(book2)
        with pytest.raises(Exception):  # IntegrityError
            session.commit()


class TestBookMetadataModel:
    """Tests for the BookMetadata model."""

    def test_create_metadata(self, session):
        """Test creating book metadata."""
        book = Book(title="Test Book")
        session.add(book)
        session.commit()

        metadata = BookMetadata(
            book_id=book.id,
            source="goodreads",
            rating=4.5,
            rating_count=1000,
        )
        session.add(metadata)
        session.commit()

        assert metadata.id is not None
        assert metadata.rating == 4.5

    def test_metadata_book_relationship(self, session):
        """Test metadata-book relationship."""
        book = Book(title="Test Book")
        metadata = BookMetadata(
            book=book,
            source="goodreads",
            rating=4.0,
        )

        session.add(metadata)
        session.commit()

        assert metadata.book.title == "Test Book"
        assert len(book.metadata_entries) == 1

    def test_unique_book_source(self, session):
        """Test that book_id + source must be unique."""
        book = Book(title="Test Book")
        session.add(book)
        session.commit()

        meta1 = BookMetadata(book_id=book.id, source="goodreads", rating=4.0)
        meta2 = BookMetadata(book_id=book.id, source="goodreads", rating=4.5)

        session.add(meta1)
        session.commit()

        session.add(meta2)
        with pytest.raises(Exception):  # IntegrityError
            session.commit()


class TestAnalysisResultModel:
    """Tests for the AnalysisResult model."""

    def test_create_analysis_result(self, session):
        """Test creating an analysis result."""
        book = Book(title="Test Book")
        session.add(book)
        session.commit()

        result = AnalysisResult(
            book_id=book.id,
            analysis_type="sentiment",
            results_json='{"positive": 0.7, "negative": 0.3}',
            model_version="1.0",
        )
        session.add(result)
        session.commit()

        assert result.id is not None
        assert result.analysis_type == "sentiment"


class TestEmbeddingMetadataModel:
    """Tests for the EmbeddingMetadata model."""

    def test_create_embedding_metadata(self, session):
        """Test creating embedding metadata."""
        book = Book(title="Test Book")
        session.add(book)
        session.commit()

        embedding = EmbeddingMetadata(
            book_id=book.id,
            chunk_index=0,
            chunk_text="First chunk of text",
            embedding_model="all-MiniLM-L6-v2",
            embedding_dimension=384,
        )
        session.add(embedding)
        session.commit()

        assert embedding.id is not None
        assert embedding.chunk_index == 0
        assert embedding.embedding_dimension == 384
