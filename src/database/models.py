"""SQLAlchemy ORM models for the Literary Analysis database."""

from datetime import datetime
from typing import Optional, List
from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    Float,
    DateTime,
    ForeignKey,
    Index,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship

# SQLAlchemy 1.4+ compatibility: declarative_base moved in 2.0
try:
    from sqlalchemy.orm import declarative_base
except ImportError:
    from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class Author(Base):
    """Author entity representing literary authors."""

    __tablename__ = "authors"

    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    birth_year = Column(Integer, nullable=True)
    death_year = Column(Integer, nullable=True)
    nationality = Column(String(100), nullable=True)
    biography = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationships
    books = relationship("Book", back_populates="author")

    __table_args__ = (Index("idx_author_name", "name"),)

    def __repr__(self) -> str:
        return f"<Author(id={self.id}, name='{self.name}')>"


class Book(Base):
    """Book entity representing literary works."""

    __tablename__ = "books"

    id = Column(Integer, primary_key=True)
    gutenberg_id = Column(Integer, unique=True, nullable=True)
    title = Column(String(500), nullable=False)
    author_id = Column(Integer, ForeignKey("authors.id"), nullable=True)
    publication_year = Column(Integer, nullable=True)
    language = Column(String(10), default="en", nullable=False)
    genre = Column(String(100), nullable=True)
    word_count = Column(Integer, nullable=True)
    full_text = Column(Text, nullable=True)
    source = Column(String(50), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationships
    author = relationship("Author", back_populates="books")
    metadata_entries = relationship(
        "BookMetadata", back_populates="book", cascade="all, delete-orphan"
    )
    analysis_results = relationship(
        "AnalysisResult", back_populates="book", cascade="all, delete-orphan"
    )
    embedding_metadata = relationship(
        "EmbeddingMetadata", back_populates="book", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("idx_book_title", "title"),
        Index("idx_book_author", "author_id"),
        Index("idx_book_gutenberg", "gutenberg_id"),
    )

    def __repr__(self) -> str:
        return f"<Book(id={self.id}, title='{self.title[:50]}...')>"


class BookMetadata(Base):
    """Enrichment metadata from external sources like Goodreads."""

    __tablename__ = "book_metadata"

    id = Column(Integer, primary_key=True)
    book_id = Column(Integer, ForeignKey("books.id"), nullable=False)
    source = Column(String(50), nullable=False)  # 'goodreads', 'gutenberg', etc.
    rating = Column(Float, nullable=True)
    rating_count = Column(Integer, nullable=True)
    review_count = Column(Integer, nullable=True)
    description = Column(Text, nullable=True)
    subjects = Column(Text, nullable=True)  # JSON array
    isbn = Column(String(20), nullable=True)
    isbn13 = Column(String(20), nullable=True)
    raw_json = Column(Text, nullable=True)
    match_confidence = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    book = relationship("Book", back_populates="metadata_entries")

    __table_args__ = (
        UniqueConstraint("book_id", "source", name="uq_book_source"),
        Index("idx_metadata_book", "book_id"),
        Index("idx_metadata_source", "source"),
    )

    def __repr__(self) -> str:
        return f"<BookMetadata(book_id={self.book_id}, source='{self.source}')>"


class AnalysisResult(Base):
    """NLP analysis results for books (Phase 2)."""

    __tablename__ = "analysis_results"

    id = Column(Integer, primary_key=True)
    book_id = Column(Integer, ForeignKey("books.id"), nullable=False)
    analysis_type = Column(String(50), nullable=False)  # 'sentiment', 'complexity', 'topics', etc.
    results_json = Column(Text, nullable=False)
    model_version = Column(String(100), nullable=True)
    processing_time_seconds = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    book = relationship("Book", back_populates="analysis_results")

    __table_args__ = (
        UniqueConstraint("book_id", "analysis_type", "model_version", name="uq_analysis"),
        Index("idx_analysis_book", "book_id"),
        Index("idx_analysis_type", "analysis_type"),
    )

    def __repr__(self) -> str:
        return f"<AnalysisResult(book_id={self.book_id}, type='{self.analysis_type}')>"


class EmbeddingMetadata(Base):
    """Metadata for text embeddings used in recommendations (Phase 4)."""

    __tablename__ = "embeddings_metadata"

    id = Column(Integer, primary_key=True)
    book_id = Column(Integer, ForeignKey("books.id"), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    chunk_start = Column(Integer, nullable=True)
    chunk_end = Column(Integer, nullable=True)
    chunk_text = Column(Text, nullable=True)
    embedding_model = Column(String(100), nullable=False)
    vector_db_id = Column(String(100), nullable=True)  # Reference to FAISS index position
    embedding_dimension = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    book = relationship("Book", back_populates="embedding_metadata")

    __table_args__ = (
        UniqueConstraint(
            "book_id", "chunk_index", "embedding_model", name="uq_embedding_chunk"
        ),
        Index("idx_embedding_book", "book_id"),
        Index("idx_embedding_model", "embedding_model"),
    )

    def __repr__(self) -> str:
        return f"<EmbeddingMetadata(book_id={self.book_id}, chunk={self.chunk_index})>"
