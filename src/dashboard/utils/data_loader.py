"""Data loader utilities for dashboard database access."""

import json
from typing import Optional, List, Dict, Any
from functools import lru_cache

from src.database.connection import get_session_context
from src.database.models import Book, Author, AnalysisResult


def get_all_books() -> List[Dict[str, Any]]:
    """Get all books with their authors."""
    with get_session_context() as session:
        books = session.query(Book).all()
        return [
            {
                "id": book.id,
                "title": book.title,
                "author": book.author.name if book.author else "Unknown",
                "author_id": book.author_id,
                "word_count": book.word_count,
                "publication_year": book.publication_year,
                "language": book.language,
            }
            for book in books
        ]


def get_kafka_books() -> List[Dict[str, Any]]:
    """Get only Kafka's English works."""
    with get_session_context() as session:
        kafka_author = session.query(Author).filter(
            Author.name.ilike("%kafka%")
        ).first()

        if not kafka_author:
            return []

        books = session.query(Book).filter(
            Book.author_id == kafka_author.id,
            Book.language == "en"
        ).all()

        return [
            {
                "id": book.id,
                "title": book.title,
                "author": kafka_author.name,
                "word_count": book.word_count,
                "publication_year": book.publication_year,
            }
            for book in books
        ]


def get_book_by_id(book_id: int) -> Optional[Dict[str, Any]]:
    """Get a single book by ID."""
    with get_session_context() as session:
        book = session.query(Book).filter(Book.id == book_id).first()
        if not book:
            return None
        return {
            "id": book.id,
            "title": book.title,
            "author": book.author.name if book.author else "Unknown",
            "word_count": book.word_count,
            "publication_year": book.publication_year,
            "language": book.language,
            "full_text": book.full_text,
        }


def get_analysis_result(book_id: int, analysis_type: str) -> Optional[Dict[str, Any]]:
    """Get analysis result for a book and analysis type."""
    with get_session_context() as session:
        result = session.query(AnalysisResult).filter(
            AnalysisResult.book_id == book_id,
            AnalysisResult.analysis_type == analysis_type
        ).first()

        if not result:
            return None

        return json.loads(result.results_json)


def get_all_analysis_types(book_id: int) -> List[str]:
    """Get all available analysis types for a book."""
    with get_session_context() as session:
        results = session.query(AnalysisResult.analysis_type).filter(
            AnalysisResult.book_id == book_id
        ).distinct().all()
        return [r[0] for r in results]


def get_sentiment_data(book_id: int) -> Optional[Dict[str, Any]]:
    """Get sentiment analysis data for a book."""
    return get_analysis_result(book_id, "sentiment")


def get_complexity_data(book_id: int) -> Optional[Dict[str, Any]]:
    """Get complexity analysis data for a book."""
    return get_analysis_result(book_id, "complexity")


def get_topics_data(book_id: int) -> Optional[Dict[str, Any]]:
    """Get topic modeling data for a book."""
    return get_analysis_result(book_id, "topics")


def get_entities_data(book_id: int) -> Optional[Dict[str, Any]]:
    """Get entity/character network data for a book."""
    return get_analysis_result(book_id, "entities")


def get_preprocessing_data(book_id: int) -> Optional[Dict[str, Any]]:
    """Get preprocessing data for a book."""
    return get_analysis_result(book_id, "preprocessing")


def get_corpus_statistics() -> Dict[str, Any]:
    """Get overall corpus statistics."""
    with get_session_context() as session:
        books = session.query(Book).filter(Book.language == "en").all()

        total_words = sum(b.word_count or 0 for b in books)

        # Count books with analysis
        analyzed_books = session.query(Book).join(AnalysisResult).distinct().all()

        # Get analysis type counts
        analysis_counts = {}
        for book in analyzed_books:
            types = get_all_analysis_types(book.id)
            for t in types:
                analysis_counts[t] = analysis_counts.get(t, 0) + 1

        return {
            "total_books": len(books),
            "total_words": total_words,
            "analyzed_books": len(analyzed_books),
            "analysis_types": analysis_counts,
        }


def get_comparative_data(book_ids: List[int]) -> Dict[str, Dict[str, Any]]:
    """Get analysis data for multiple books for comparison."""
    comparative = {}

    for book_id in book_ids:
        book = get_book_by_id(book_id)
        if book:
            comparative[book["title"]] = {
                "book": book,
                "sentiment": get_sentiment_data(book_id),
                "complexity": get_complexity_data(book_id),
                "topics": get_topics_data(book_id),
                "entities": get_entities_data(book_id),
            }

    return comparative
