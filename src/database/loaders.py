"""Data loaders for populating the database with scraped content."""

import json
import logging
from pathlib import Path
from typing import Optional, List, Dict

from sqlalchemy.orm import Session
from sqlalchemy import select

from .models import Author, Book, BookMetadata
from .connection import get_session_factory
from src.scrapers.kafka_manifest import KAFKA_WORKS, KAFKA_AUTHOR, KafkaWork
from src.scrapers.corpus_manifest import CLASSIC_WORKS, ClassicWork
from config.settings import get_settings


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Loader for populating the database with literary works."""

    def __init__(self, session: Optional[Session] = None):
        """
        Initialize the data loader.

        Args:
            session: SQLAlchemy session (creates one if not provided)
        """
        if session is None:
            SessionLocal = get_session_factory()
            self._session = SessionLocal()
            self._owns_session = True
        else:
            self._session = session
            self._owns_session = False

        self.settings = get_settings()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._owns_session:
            self._session.close()

    @property
    def session(self) -> Session:
        return self._session

    def get_or_create_author(
        self,
        name: str,
        birth_year: Optional[int] = None,
        death_year: Optional[int] = None,
        nationality: Optional[str] = None,
        biography: Optional[str] = None,
    ) -> Author:
        """
        Get an existing author or create a new one.

        Args:
            name: Author's name
            birth_year: Year of birth
            death_year: Year of death
            nationality: Author's nationality
            biography: Author's biography

        Returns:
            Author object
        """
        stmt = select(Author).where(Author.name == name)
        author = self.session.execute(stmt).scalar_one_or_none()

        if author is None:
            author = Author(
                name=name,
                birth_year=birth_year,
                death_year=death_year,
                nationality=nationality,
                biography=biography,
            )
            self.session.add(author)
            self.session.flush()
            logger.info(f"Created new author: {name}")

        return author

    def get_book_by_gutenberg_id(self, gutenberg_id: int) -> Optional[Book]:
        """Get a book by its Gutenberg ID."""
        stmt = select(Book).where(Book.gutenberg_id == gutenberg_id)
        return self.session.execute(stmt).scalar_one_or_none()

    def load_book_from_file(
        self,
        gutenberg_id: int,
        title: str,
        author: Author,
        text_path: Path,
        publication_year: Optional[int] = None,
        language: str = "en",
        genre: Optional[str] = None,
        source: str = "gutenberg",
    ) -> Book:
        """
        Load a book from a text file into the database.

        Args:
            gutenberg_id: Project Gutenberg ID
            title: Book title
            author: Author object
            text_path: Path to the text file
            publication_year: Year of publication
            language: Language code
            genre: Genre classification
            source: Data source identifier

        Returns:
            Book object
        """
        # Check if book already exists
        existing = self.get_book_by_gutenberg_id(gutenberg_id)
        if existing is not None:
            logger.info(f"Book already exists: {title} (ID: {gutenberg_id})")
            return existing

        # Read text content
        if text_path.exists():
            full_text = text_path.read_text(encoding="utf-8", errors="replace")
            word_count = len(full_text.split())
        else:
            full_text = None
            word_count = None
            logger.warning(f"Text file not found: {text_path}")

        book = Book(
            gutenberg_id=gutenberg_id,
            title=title,
            author_id=author.id,
            publication_year=publication_year,
            language=language,
            genre=genre,
            word_count=word_count,
            full_text=full_text,
            source=source,
        )
        self.session.add(book)
        self.session.flush()
        logger.info(f"Loaded book: {title} ({word_count or 0} words)")

        return book

    def load_kafka_works(self, data_dir: Optional[Path] = None) -> List[Book]:
        """
        Load all Kafka works from downloaded files.

        Args:
            data_dir: Directory containing Kafka text files

        Returns:
            List of Book objects
        """
        if data_dir is None:
            data_dir = self.settings.raw_data_dir / "kafka"

        # Create Kafka author
        author = self.get_or_create_author(
            name=KAFKA_AUTHOR["name"],
            birth_year=KAFKA_AUTHOR["birth_year"],
            death_year=KAFKA_AUTHOR["death_year"],
            nationality=KAFKA_AUTHOR["nationality"],
            biography=KAFKA_AUTHOR["biography"],
        )

        books = []
        for work in KAFKA_WORKS:
            text_path = data_dir / f"pg{work.gutenberg_id}.txt"

            book = self.load_book_from_file(
                gutenberg_id=work.gutenberg_id,
                title=work.title,
                author=author,
                text_path=text_path,
                publication_year=work.year_published,
                genre=work.work_type,
            )
            books.append(book)

        self.session.commit()
        logger.info(f"Loaded {len(books)} Kafka works")
        return books

    def load_corpus_from_manifest(
        self, data_dir: Optional[Path] = None
    ) -> List[Book]:
        """
        Load broader corpus from manifest and downloaded files.

        Args:
            data_dir: Directory containing corpus text files

        Returns:
            List of Book objects
        """
        if data_dir is None:
            data_dir = self.settings.raw_data_dir / "corpus"

        # Cache authors to avoid repeated lookups
        author_cache: Dict[str, Author] = {}
        books = []

        for work in CLASSIC_WORKS:
            # Get or create author
            if work.author not in author_cache:
                author_cache[work.author] = self.get_or_create_author(name=work.author)
            author = author_cache[work.author]

            text_path = data_dir / f"pg{work.gutenberg_id}.txt"

            # Only load if text file exists
            if text_path.exists():
                book = self.load_book_from_file(
                    gutenberg_id=work.gutenberg_id,
                    title=work.title,
                    author=author,
                    text_path=text_path,
                    publication_year=work.year,
                    genre=work.genre,
                )
                books.append(book)
            else:
                logger.debug(f"Skipping (no text file): {work.title}")

        self.session.commit()
        logger.info(f"Loaded {len(books)} corpus books")
        return books

    def load_from_metadata_files(self, data_dir: Path) -> List[Book]:
        """
        Load books from metadata JSON files in a directory.

        Args:
            data_dir: Directory containing metadata JSON files

        Returns:
            List of Book objects
        """
        author_cache: Dict[str, Author] = {}
        books = []

        for meta_path in data_dir.glob("*_meta.json"):
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                gutenberg_id = meta["gutenberg_id"]

                # Get author
                author_name = meta.get("author", "Unknown")
                if author_name not in author_cache:
                    author_cache[author_name] = self.get_or_create_author(name=author_name)
                author = author_cache[author_name]

                # Find text file
                text_path = data_dir / f"pg{gutenberg_id}.txt"

                book = self.load_book_from_file(
                    gutenberg_id=gutenberg_id,
                    title=meta.get("title", f"Unknown (ID: {gutenberg_id})"),
                    author=author,
                    text_path=text_path,
                    language=meta.get("language", "en"),
                )
                books.append(book)

            except Exception as e:
                logger.error(f"Error loading {meta_path}: {e}")

        self.session.commit()
        logger.info(f"Loaded {len(books)} books from metadata files")
        return books

    def add_goodreads_metadata(
        self,
        book: Book,
        rating: Optional[float] = None,
        rating_count: Optional[int] = None,
        description: Optional[str] = None,
        subjects: Optional[List[str]] = None,
        match_confidence: Optional[float] = None,
        raw_data: Optional[dict] = None,
    ) -> BookMetadata:
        """
        Add Goodreads metadata to a book.

        Args:
            book: Book object
            rating: Average rating
            rating_count: Number of ratings
            description: Book description
            subjects: List of subjects/genres
            match_confidence: Confidence score for the match
            raw_data: Raw Goodreads data

        Returns:
            BookMetadata object
        """
        metadata = BookMetadata(
            book_id=book.id,
            source="goodreads",
            rating=rating,
            rating_count=rating_count,
            description=description,
            subjects=json.dumps(subjects) if subjects else None,
            match_confidence=match_confidence,
            raw_json=json.dumps(raw_data) if raw_data else None,
        )
        self.session.add(metadata)
        self.session.flush()
        return metadata

    def get_statistics(self) -> dict:
        """
        Get database statistics.

        Returns:
            Dictionary with counts and stats
        """
        from sqlalchemy import func

        stats = {
            "total_authors": self.session.query(func.count(Author.id)).scalar(),
            "total_books": self.session.query(func.count(Book.id)).scalar(),
            "books_with_text": self.session.query(func.count(Book.id)).filter(
                Book.full_text.isnot(None)
            ).scalar(),
            "total_word_count": self.session.query(func.sum(Book.word_count)).scalar() or 0,
            "metadata_entries": self.session.query(func.count(BookMetadata.id)).scalar(),
        }

        # Get Kafka-specific stats
        kafka_author = self.session.query(Author).filter(
            Author.name == "Franz Kafka"
        ).first()
        if kafka_author:
            stats["kafka_books"] = self.session.query(func.count(Book.id)).filter(
                Book.author_id == kafka_author.id
            ).scalar()
        else:
            stats["kafka_books"] = 0

        return stats
