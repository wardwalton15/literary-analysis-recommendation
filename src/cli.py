"""Command-line interface for the Literary Analysis Platform."""

import argparse
import sys
import logging
from pathlib import Path

from config.settings import get_settings
from src.database.connection import init_db, reset_db
from src.database.loaders import DataLoader
from src.scrapers.gutenberg_scraper import GutenbergScraper, GutenbergBatchDownloader
from src.scrapers.kafka_manifest import get_kafka_book_ids, KAFKA_WORKS
from src.scrapers.corpus_manifest import get_corpus_book_ids
from src.utils.data_quality import validate_corpus


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def cmd_init(args):
    """Initialize the database."""
    settings = get_settings()
    settings.ensure_directories()
    init_db()
    print("Database initialized successfully.")


def cmd_reset(args):
    """Reset the database (drops all tables and recreates them)."""
    confirm = input("This will delete all data. Are you sure? (yes/no): ")
    if confirm.lower() == "yes":
        reset_db()
        print("Database reset successfully.")
    else:
        print("Aborted.")


def cmd_scrape_kafka(args):
    """Scrape Kafka works from Project Gutenberg."""
    settings = get_settings()
    settings.ensure_directories()

    output_dir = settings.raw_data_dir / "kafka"
    output_dir.mkdir(parents=True, exist_ok=True)

    book_ids = get_kafka_book_ids()
    print(f"Scraping {len(book_ids)} Kafka works...")

    downloader = GutenbergBatchDownloader(
        book_ids=book_ids,
        output_dir=output_dir,
        delay=args.delay,
    )

    summary = downloader.download_all(resume=not args.force)
    print(f"\nDownload complete:")
    print(f"  Downloaded: {summary['downloaded']}")
    print(f"  Failed: {summary['failed']}")
    print(f"  Skipped (existing): {summary['skipped_existing']}")


def cmd_scrape_corpus(args):
    """Scrape the broader classic literature corpus."""
    settings = get_settings()
    settings.ensure_directories()

    output_dir = settings.raw_data_dir / "corpus"
    output_dir.mkdir(parents=True, exist_ok=True)

    book_ids = get_corpus_book_ids()

    if args.limit:
        book_ids = book_ids[: args.limit]

    print(f"Scraping {len(book_ids)} classic works...")

    downloader = GutenbergBatchDownloader(
        book_ids=book_ids,
        output_dir=output_dir,
        delay=args.delay,
    )

    summary = downloader.download_all(resume=not args.force)
    print(f"\nDownload complete:")
    print(f"  Downloaded: {summary['downloaded']}")
    print(f"  Failed: {summary['failed']}")
    print(f"  Skipped (existing): {summary['skipped_existing']}")


def cmd_load_data(args):
    """Load downloaded texts into the database."""
    settings = get_settings()

    with DataLoader() as loader:
        # Load Kafka works
        kafka_dir = settings.raw_data_dir / "kafka"
        if kafka_dir.exists() and any(kafka_dir.glob("*.txt")):
            print("Loading Kafka works...")
            kafka_books = loader.load_kafka_works(kafka_dir)
            print(f"  Loaded {len(kafka_books)} Kafka works")

        # Load broader corpus
        corpus_dir = settings.raw_data_dir / "corpus"
        if corpus_dir.exists() and any(corpus_dir.glob("*.txt")):
            print("Loading corpus books...")
            corpus_books = loader.load_corpus_from_manifest(corpus_dir)
            print(f"  Loaded {len(corpus_books)} corpus books")

        # Print statistics
        stats = loader.get_statistics()
        print("\nDatabase Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")


def cmd_validate(args):
    """Run data quality validation."""
    passed, report = validate_corpus(strict=args.strict)

    report.print_summary()

    if args.output:
        report.save(Path(args.output))

    sys.exit(0 if passed else 1)


def cmd_stats(args):
    """Show database statistics."""
    with DataLoader() as loader:
        stats = loader.get_statistics()

    print("\nDatabase Statistics:")
    print("-" * 40)
    for key, value in stats.items():
        label = key.replace("_", " ").title()
        print(f"  {label}: {value:,}")


def cmd_test_scraper(args):
    """Test the scraper with a single book."""
    book_id = args.book_id or 5200  # Default to The Metamorphosis

    print(f"Testing scraper with book ID {book_id}...")

    scraper = GutenbergScraper(delay=0.5)

    try:
        metadata, text = scraper.scrape_book(book_id, clean=True)
        print(f"\nTitle: {metadata.title}")
        print(f"Author: {metadata.author}")
        print(f"Language: {metadata.language}")
        print(f"Subjects: {', '.join(metadata.subjects or [])}")
        print(f"\nText preview ({len(text)} characters total):")
        print("-" * 40)
        print(text[:500])
        print("...")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Literary Analysis & Recommendation Platform CLI"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # init command
    init_parser = subparsers.add_parser("init", help="Initialize the database")
    init_parser.set_defaults(func=cmd_init)

    # reset command
    reset_parser = subparsers.add_parser("reset", help="Reset the database")
    reset_parser.set_defaults(func=cmd_reset)

    # scrape-kafka command
    kafka_parser = subparsers.add_parser(
        "scrape-kafka", help="Scrape Kafka works from Gutenberg"
    )
    kafka_parser.add_argument(
        "--delay", type=float, default=1.0, help="Delay between requests (seconds)"
    )
    kafka_parser.add_argument(
        "--force", action="store_true", help="Re-download existing files"
    )
    kafka_parser.set_defaults(func=cmd_scrape_kafka)

    # scrape-corpus command
    corpus_parser = subparsers.add_parser(
        "scrape-corpus", help="Scrape classic literature corpus"
    )
    corpus_parser.add_argument(
        "--delay", type=float, default=1.0, help="Delay between requests (seconds)"
    )
    corpus_parser.add_argument(
        "--force", action="store_true", help="Re-download existing files"
    )
    corpus_parser.add_argument(
        "--limit", type=int, help="Limit number of books to download"
    )
    corpus_parser.set_defaults(func=cmd_scrape_corpus)

    # load-data command
    load_parser = subparsers.add_parser(
        "load-data", help="Load downloaded texts into database"
    )
    load_parser.set_defaults(func=cmd_load_data)

    # validate command
    validate_parser = subparsers.add_parser(
        "validate", help="Run data quality validation"
    )
    validate_parser.add_argument(
        "--strict", action="store_true", help="Use strict validation criteria"
    )
    validate_parser.add_argument(
        "--output", "-o", help="Save report to JSON file"
    )
    validate_parser.set_defaults(func=cmd_validate)

    # stats command
    stats_parser = subparsers.add_parser("stats", help="Show database statistics")
    stats_parser.set_defaults(func=cmd_stats)

    # test-scraper command
    test_parser = subparsers.add_parser("test-scraper", help="Test the scraper")
    test_parser.add_argument(
        "--book-id", type=int, help="Gutenberg book ID to test with"
    )
    test_parser.set_defaults(func=cmd_test_scraper)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
