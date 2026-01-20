"""Command-line interface for the Literary Analysis Platform."""

import argparse
import sys
import logging
import json
from pathlib import Path

from config.settings import get_settings
from src.database.connection import init_db, reset_db, get_session_context
from src.database.loaders import DataLoader
from src.database.models import Book, AnalysisResult
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


def cmd_analyze(args):
    """Run NLP analysis on books."""
    from src.analysis import AnalysisService

    service = AnalysisService(use_spacy=not args.no_spacy)

    # Determine which analyses to run
    analyses = None
    if args.type:
        analyses = [args.type]

    if args.book_id:
        # Analyze specific book
        print(f"Analyzing book ID {args.book_id}...")
        try:
            result = service.analyze_and_save(args.book_id, analyses)
            print(f"\nAnalysis complete for: {result.book_title}")
            print(f"Processing time: {result.processing_time:.2f}s")

            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(result.to_dict(), f, indent=2)
                print(f"Results saved to: {args.output}")
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)

    elif args.kafka:
        # Analyze all Kafka works
        print("Analyzing Kafka works...")
        results = service.analyze_kafka_works(language=args.language, analyses=analyses)
        print(f"\nAnalyzed {len(results)} Kafka works")

        for result in results:
            print(f"  - {result.book_title}: {result.processing_time:.2f}s")

    elif args.corpus:
        # Analyze corpus
        limit = args.limit
        print(f"Analyzing corpus (limit: {limit or 'all'})...")
        results = service.analyze_corpus(limit=limit, language=args.language, analyses=analyses)
        print(f"\nAnalyzed {len(results)} books")

    else:
        print("Please specify --book-id, --kafka, or --corpus")
        sys.exit(1)


def cmd_analyze_summary(args):
    """Show analysis summary."""
    from src.analysis import AnalysisService

    service = AnalysisService(use_spacy=False)
    summary = service.get_analysis_summary()

    print("\nAnalysis Summary:")
    print("-" * 40)
    print(f"  Total books with text: {summary['total_books_with_text']}")
    print(f"  Books analyzed: {summary['books_analyzed']}")
    print(f"  Coverage: {summary['coverage_percent']}%")
    print("\n  Analysis counts:")
    for analysis_type, count in summary['analysis_counts'].items():
        print(f"    {analysis_type}: {count}")


def cmd_show_analysis(args):
    """Show analysis results for a book."""
    from src.analysis import AnalysisService

    with get_session_context() as session:
        book = session.query(Book).get(args.book_id)
        if not book:
            print(f"Book {args.book_id} not found")
            sys.exit(1)

        print(f"\nAnalysis results for: {book.title}")
        print("-" * 50)

        # Get analysis results
        results = session.query(AnalysisResult).filter(
            AnalysisResult.book_id == args.book_id
        ).all()

        if not results:
            print("No analysis results found. Run 'analyze' first.")
            sys.exit(1)

        for result in results:
            print(f"\n[{result.analysis_type}] (v{result.model_version})")

            if args.verbose:
                data = json.loads(result.results_json)
                print(json.dumps(data, indent=2)[:2000])
            else:
                # Show summary
                data = json.loads(result.results_json)
                if result.analysis_type == 'complexity':
                    rd = data.get('readability', {})
                    print(f"  Flesch Reading Ease: {rd.get('flesch_reading_ease', 'N/A')}")
                    print(f"  Flesch-Kincaid Grade: {rd.get('flesch_kincaid_grade', 'N/A')}")
                elif result.analysis_type == 'sentiment':
                    ov = data.get('overall', {})
                    print(f"  Polarity: {ov.get('polarity', 'N/A')}")
                    print(f"  VADER Compound: {ov.get('vader_compound', 'N/A')}")
                elif result.analysis_type == 'topics':
                    print(f"  Topics found: {data.get('num_topics', 'N/A')}")
                    print(f"  Theme labels: {', '.join(data.get('theme_labels', []))}")
                elif result.analysis_type == 'entities':
                    sm = data.get('summary', {})
                    print(f"  Characters: {sm.get('character_count', 'N/A')}")
                    print(f"  Locations: {sm.get('location_count', 'N/A')}")


def cmd_list_books(args):
    """List books in the database."""
    with get_session_context() as session:
        query = session.query(Book)

        if args.language:
            query = query.filter(Book.language == args.language)

        if args.with_text:
            query = query.filter(Book.full_text.isnot(None))

        if args.analyzed:
            analyzed_ids = session.query(AnalysisResult.book_id).distinct()
            query = query.filter(Book.id.in_(analyzed_ids))

        books = query.limit(args.limit or 100).all()

        print(f"\nBooks ({len(books)} shown):")
        print("-" * 70)
        for book in books:
            text_status = "+" if book.full_text else "-"
            analysis_count = session.query(AnalysisResult).filter(
                AnalysisResult.book_id == book.id
            ).count()
            analyzed_status = f"[{analysis_count}]" if analysis_count else "[ ]"

            print(f"  {book.id:4d} {text_status} {analyzed_status} {book.title[:50]}")


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

    # analyze command (Phase 2)
    analyze_parser = subparsers.add_parser(
        "analyze", help="Run NLP analysis on books"
    )
    analyze_parser.add_argument(
        "--book-id", type=int, help="Analyze specific book by ID"
    )
    analyze_parser.add_argument(
        "--kafka", action="store_true", help="Analyze all Kafka works"
    )
    analyze_parser.add_argument(
        "--corpus", action="store_true", help="Analyze entire corpus"
    )
    analyze_parser.add_argument(
        "--limit", type=int, help="Limit number of books (for corpus)"
    )
    analyze_parser.add_argument(
        "--language", default="en", help="Filter by language (default: en)"
    )
    analyze_parser.add_argument(
        "--type", choices=["preprocessing", "complexity", "sentiment", "topics", "entities"],
        help="Run specific analysis type only"
    )
    analyze_parser.add_argument(
        "--no-spacy", action="store_true", help="Skip spaCy-based entity analysis"
    )
    analyze_parser.add_argument(
        "--output", "-o", help="Save results to JSON file"
    )
    analyze_parser.set_defaults(func=cmd_analyze)

    # analyze-summary command
    summary_parser = subparsers.add_parser(
        "analyze-summary", help="Show analysis summary"
    )
    summary_parser.set_defaults(func=cmd_analyze_summary)

    # show-analysis command
    show_parser = subparsers.add_parser(
        "show-analysis", help="Show analysis results for a book"
    )
    show_parser.add_argument(
        "book_id", type=int, help="Book ID to show analysis for"
    )
    show_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show detailed results"
    )
    show_parser.set_defaults(func=cmd_show_analysis)

    # list-books command
    list_parser = subparsers.add_parser(
        "list-books", help="List books in database"
    )
    list_parser.add_argument(
        "--language", help="Filter by language"
    )
    list_parser.add_argument(
        "--with-text", action="store_true", help="Only show books with text"
    )
    list_parser.add_argument(
        "--analyzed", action="store_true", help="Only show analyzed books"
    )
    list_parser.add_argument(
        "--limit", type=int, help="Limit number of books shown"
    )
    list_parser.set_defaults(func=cmd_list_books)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
