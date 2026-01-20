"""Data quality validation utilities."""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Tuple

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.database.models import Author, Book, BookMetadata
from src.database.connection import get_session_factory
from src.scrapers.kafka_manifest import KAFKA_WORKS
from config.settings import get_settings


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class QualityCheck:
    """Result of a single quality check."""

    name: str
    passed: bool
    expected: Optional[str] = None
    actual: Optional[str] = None
    message: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DataQualityReport:
    """Complete data quality report."""

    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    checks: List[QualityCheck] = field(default_factory=list)
    statistics: Dict = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """Check if all quality checks passed."""
        return all(check.passed for check in self.checks)

    @property
    def pass_rate(self) -> float:
        """Calculate the pass rate of checks."""
        if not self.checks:
            return 0.0
        return sum(1 for c in self.checks if c.passed) / len(self.checks)

    def add_check(self, check: QualityCheck) -> None:
        """Add a quality check result."""
        self.checks.append(check)
        if not check.passed:
            self.errors.append(f"FAILED: {check.name} - {check.message}")

    def add_warning(self, message: str) -> None:
        """Add a warning message."""
        self.warnings.append(message)

    def to_dict(self) -> dict:
        """Convert report to dictionary."""
        return {
            "timestamp": self.timestamp,
            "passed": self.passed,
            "pass_rate": self.pass_rate,
            "checks": [c.to_dict() for c in self.checks],
            "statistics": self.statistics,
            "warnings": self.warnings,
            "errors": self.errors,
        }

    def to_json(self) -> str:
        """Convert report to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    def save(self, path: Path) -> None:
        """Save report to a JSON file."""
        path.write_text(self.to_json(), encoding="utf-8")
        logger.info(f"Quality report saved to: {path}")

    def print_summary(self) -> None:
        """Print a summary of the report."""
        print("\n" + "=" * 60)
        print("DATA QUALITY REPORT")
        print("=" * 60)
        print(f"Timestamp: {self.timestamp}")
        print(f"Overall Status: {'PASSED' if self.passed else 'FAILED'}")
        print(f"Pass Rate: {self.pass_rate:.1%}")
        print("-" * 60)

        print("\nCHECKS:")
        for check in self.checks:
            status = "PASS" if check.passed else "FAIL"
            print(f"  [{status}] {check.name}")
            if not check.passed and check.message:
                print(f"      -> {check.message}")

        if self.warnings:
            print("\nWARNINGS:")
            for warning in self.warnings:
                print(f"  [!] {warning}")

        print("\nSTATISTICS:")
        for key, value in self.statistics.items():
            print(f"  {key}: {value}")

        print("=" * 60 + "\n")


class DataQualityValidator:
    """Validator for checking data quality."""

    def __init__(self, session: Optional[Session] = None):
        """
        Initialize the validator.

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

    def check_kafka_works_complete(self, min_count: int = 5) -> QualityCheck:
        """Check if Kafka works are loaded."""
        kafka_author = self.session.query(Author).filter(
            Author.name == "Franz Kafka"
        ).first()

        if kafka_author is None:
            return QualityCheck(
                name="Kafka works complete",
                passed=False,
                expected=f">= {min_count}",
                actual="0 (no Kafka author found)",
                message="Franz Kafka author not found in database",
            )

        count = self.session.query(func.count(Book.id)).filter(
            Book.author_id == kafka_author.id,
            Book.full_text.isnot(None),
        ).scalar()

        passed = count >= min_count
        return QualityCheck(
            name="Kafka works complete",
            passed=passed,
            expected=f">= {min_count}",
            actual=str(count),
            message=None if passed else f"Only {count} Kafka works with text found",
        )

    def check_corpus_size(self, min_count: int = 50) -> QualityCheck:
        """Check if corpus has minimum number of books."""
        count = self.session.query(func.count(Book.id)).filter(
            Book.full_text.isnot(None)
        ).scalar()

        passed = count >= min_count
        return QualityCheck(
            name="Corpus size adequate",
            passed=passed,
            expected=f">= {min_count}",
            actual=str(count),
            message=None if passed else f"Only {count} books with text found",
        )

    def check_no_empty_texts(self) -> QualityCheck:
        """Check that books marked with text actually have content."""
        # Find books with text field set but empty or very short
        empty_count = self.session.query(func.count(Book.id)).filter(
            Book.full_text.isnot(None),
            func.length(Book.full_text) < 100,
        ).scalar()

        passed = empty_count == 0
        return QualityCheck(
            name="No empty texts",
            passed=passed,
            expected="0",
            actual=str(empty_count),
            message=None if passed else f"{empty_count} books have empty/very short text",
        )

    def check_no_duplicate_gutenberg_ids(self) -> QualityCheck:
        """Check for duplicate Gutenberg IDs."""
        from sqlalchemy import func as sqlfunc

        # Find duplicates
        subq = self.session.query(
            Book.gutenberg_id,
            sqlfunc.count(Book.id).label("cnt")
        ).filter(
            Book.gutenberg_id.isnot(None)
        ).group_by(Book.gutenberg_id).having(
            sqlfunc.count(Book.id) > 1
        ).subquery()

        dup_count = self.session.query(sqlfunc.count()).select_from(subq).scalar()

        passed = dup_count == 0
        return QualityCheck(
            name="No duplicate Gutenberg IDs",
            passed=passed,
            expected="0",
            actual=str(dup_count),
            message=None if passed else f"{dup_count} duplicate Gutenberg IDs found",
        )

    def check_all_books_have_authors(self) -> QualityCheck:
        """Check that all books have an associated author."""
        orphan_count = self.session.query(func.count(Book.id)).filter(
            Book.author_id.is_(None)
        ).scalar()

        passed = orphan_count == 0
        return QualityCheck(
            name="All books have authors",
            passed=passed,
            expected="0 orphans",
            actual=str(orphan_count),
            message=None if passed else f"{orphan_count} books without authors",
        )

    def check_metadata_coverage(self, min_rate: float = 0.3) -> QualityCheck:
        """Check Goodreads metadata coverage."""
        total_books = self.session.query(func.count(Book.id)).scalar()
        if total_books == 0:
            return QualityCheck(
                name="Metadata coverage",
                passed=False,
                expected=f">= {min_rate:.0%}",
                actual="N/A (no books)",
                message="No books in database",
            )

        books_with_metadata = self.session.query(
            func.count(func.distinct(BookMetadata.book_id))
        ).scalar()

        coverage = books_with_metadata / total_books
        passed = coverage >= min_rate

        return QualityCheck(
            name="Metadata coverage",
            passed=passed,
            expected=f">= {min_rate:.0%}",
            actual=f"{coverage:.1%} ({books_with_metadata}/{total_books})",
            message=None if passed else f"Only {coverage:.1%} coverage",
        )

    def check_text_files_exist(self, data_dir: Optional[Path] = None) -> QualityCheck:
        """Check that expected text files exist on disk."""
        if data_dir is None:
            data_dir = self.settings.raw_data_dir

        kafka_dir = data_dir / "kafka"
        corpus_dir = data_dir / "corpus"

        missing = []

        # Check Kafka files
        for work in KAFKA_WORKS:
            path = kafka_dir / f"pg{work.gutenberg_id}.txt"
            if not path.exists():
                missing.append(f"kafka/pg{work.gutenberg_id}.txt")

        passed = len(missing) == 0
        return QualityCheck(
            name="Text files exist",
            passed=passed,
            expected="All expected files present",
            actual=f"{len(missing)} missing" if missing else "All present",
            message=None if passed else f"Missing: {', '.join(missing[:5])}{'...' if len(missing) > 5 else ''}",
        )

    def get_statistics(self) -> dict:
        """Get database statistics for the report."""
        total_books = self.session.query(func.count(Book.id)).scalar()
        books_with_text = self.session.query(func.count(Book.id)).filter(
            Book.full_text.isnot(None)
        ).scalar()
        total_words = self.session.query(func.sum(Book.word_count)).scalar() or 0
        total_authors = self.session.query(func.count(Author.id)).scalar()
        metadata_count = self.session.query(func.count(BookMetadata.id)).scalar()

        # Get Kafka stats
        kafka_author = self.session.query(Author).filter(
            Author.name == "Franz Kafka"
        ).first()
        kafka_books = 0
        if kafka_author:
            kafka_books = self.session.query(func.count(Book.id)).filter(
                Book.author_id == kafka_author.id
            ).scalar()

        return {
            "total_books": total_books,
            "books_with_text": books_with_text,
            "total_word_count": total_words,
            "average_word_count": total_words // books_with_text if books_with_text > 0 else 0,
            "total_authors": total_authors,
            "kafka_books": kafka_books,
            "metadata_entries": metadata_count,
        }

    def validate(self, strict: bool = False) -> DataQualityReport:
        """
        Run all validation checks.

        Args:
            strict: If True, requires all checks to pass for a passing report

        Returns:
            DataQualityReport with all check results
        """
        report = DataQualityReport()

        # Run all checks
        checks = [
            self.check_kafka_works_complete(min_count=3 if not strict else 5),
            self.check_corpus_size(min_count=20 if not strict else 50),
            self.check_no_empty_texts(),
            self.check_no_duplicate_gutenberg_ids(),
            self.check_all_books_have_authors(),
            self.check_text_files_exist(),
        ]

        # Add optional checks (warnings only)
        metadata_check = self.check_metadata_coverage(min_rate=0.0)
        if not metadata_check.passed:
            report.add_warning("Low metadata coverage - consider enriching with Goodreads data")

        for check in checks:
            report.add_check(check)

        # Add statistics
        report.statistics = self.get_statistics()

        return report


def validate_corpus(strict: bool = False) -> Tuple[bool, DataQualityReport]:
    """
    Convenience function to validate the corpus.

    Args:
        strict: Whether to use strict validation criteria

    Returns:
        Tuple of (passed, report)
    """
    with DataQualityValidator() as validator:
        report = validator.validate(strict=strict)
        return report.passed, report
