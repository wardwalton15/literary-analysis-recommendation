"""Analysis service for orchestrating NLP analysis and database storage.

Coordinates all analysis modules and persists results to the database.
"""

import json
import time
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy.orm import Session

from src.database.models import Book, AnalysisResult
from src.database.connection import get_session_context
from src.preprocessing.text_processor import TextProcessor
from src.analysis.complexity_analyzer import ComplexityAnalyzer, ComplexityMetrics
from src.analysis.sentiment_analyzer import SentimentAnalyzer, SentimentResult
from src.analysis.topic_modeler import TopicModeler, TopicResult
from src.analysis.entity_analyzer import EntityAnalyzer, EntityAnalysisResult

logger = logging.getLogger(__name__)


@dataclass
class FullAnalysisResult:
    """Container for complete analysis results."""
    book_id: int
    book_title: str
    preprocessing: Dict[str, Any]
    complexity: ComplexityMetrics
    sentiment: SentimentResult
    topics: TopicResult
    entities: EntityAnalysisResult
    processing_time: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'book_id': self.book_id,
            'book_title': self.book_title,
            'preprocessing': self.preprocessing,
            'complexity': self.complexity.to_dict(),
            'sentiment': self.sentiment.to_dict(),
            'topics': self.topics.to_dict(),
            'entities': self.entities.to_dict(),
            'processing_time_seconds': round(self.processing_time, 2),
            'analyzed_at': datetime.utcnow().isoformat(),
        }


class AnalysisService:
    """Service for running and storing literary analysis."""

    # Analysis type constants
    ANALYSIS_PREPROCESSING = 'preprocessing'
    ANALYSIS_COMPLEXITY = 'complexity'
    ANALYSIS_SENTIMENT = 'sentiment'
    ANALYSIS_TOPICS = 'topics'
    ANALYSIS_ENTITIES = 'entities'
    ANALYSIS_FULL = 'full'

    # Model version for tracking
    MODEL_VERSION = '1.0.0'

    def __init__(
        self,
        use_spacy: bool = True,
        spacy_model: str = 'en_core_web_sm',
    ):
        """Initialize the analysis service.

        Args:
            use_spacy: Whether to use spaCy for NER
            spacy_model: spaCy model to load
        """
        self.text_processor = TextProcessor()
        self.complexity_analyzer = ComplexityAnalyzer()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.topic_modeler = TopicModeler()

        # Try to load spaCy, fall back to regex if unavailable
        try:
            self.entity_analyzer = EntityAnalyzer(
                use_spacy=use_spacy,
                spacy_model=spacy_model,
            )
        except Exception as e:
            logger.warning(f"spaCy not available ({e}), using regex-based NER fallback")
            self.entity_analyzer = EntityAnalyzer(use_spacy=False)

    def _clean_text(self, text: str) -> str:
        """Clean text for analysis.

        Args:
            text: Raw text

        Returns:
            Cleaned text
        """
        # Remove Gutenberg headers/footers
        text = self.text_processor.clean_gutenberg_text(text)
        # Basic cleaning
        text = self.text_processor.clean_text(text)
        return text

    def analyze_preprocessing(self, text: str) -> Dict[str, Any]:
        """Run preprocessing analysis.

        Args:
            text: Input text

        Returns:
            Dictionary with preprocessing results
        """
        processed = self.text_processor.process(text, clean_gutenberg=True)
        return processed.to_dict()

    def analyze_complexity(self, text: str) -> ComplexityMetrics:
        """Run complexity analysis.

        Args:
            text: Input text

        Returns:
            ComplexityMetrics object
        """
        text = self._clean_text(text)
        return self.complexity_analyzer.analyze(text)

    def analyze_sentiment(self, text: str) -> SentimentResult:
        """Run sentiment analysis.

        Args:
            text: Input text

        Returns:
            SentimentResult object
        """
        text = self._clean_text(text)
        return self.sentiment_analyzer.analyze(text)

    def analyze_topics(
        self,
        text: str,
        num_topics: int = 5,
        method: str = 'lda_gensim',
    ) -> TopicResult:
        """Run topic modeling.

        Args:
            text: Input text
            num_topics: Number of topics to extract
            method: 'lda_gensim', 'lda_sklearn', or 'nmf'

        Returns:
            TopicResult object
        """
        text = self._clean_text(text)
        return self.topic_modeler.extract_themes_from_text(
            text,
            num_topics=num_topics,
        )

    def analyze_entities(self, text: str) -> EntityAnalysisResult:
        """Run entity analysis.

        Args:
            text: Input text

        Returns:
            EntityAnalysisResult object
        """
        text = self._clean_text(text)
        return self.entity_analyzer.analyze(text)

    def analyze_book(
        self,
        book: Book,
        analyses: List[str] = None,
    ) -> FullAnalysisResult:
        """Run full analysis on a book.

        Args:
            book: Book model instance
            analyses: List of analyses to run (None = all)

        Returns:
            FullAnalysisResult object
        """
        start_time = time.time()

        if analyses is None:
            analyses = [
                self.ANALYSIS_PREPROCESSING,
                self.ANALYSIS_COMPLEXITY,
                self.ANALYSIS_SENTIMENT,
                self.ANALYSIS_TOPICS,
                self.ANALYSIS_ENTITIES,
            ]

        text = book.full_text
        if not text:
            raise ValueError(f"Book {book.id} has no text content")

        logger.info(f"Analyzing book: {book.title} ({book.id})")

        # Run requested analyses
        preprocessing = {}
        complexity = None
        sentiment = None
        topics = None
        entities = None

        if self.ANALYSIS_PREPROCESSING in analyses:
            logger.info("  Running preprocessing analysis...")
            preprocessing = self.analyze_preprocessing(text)

        if self.ANALYSIS_COMPLEXITY in analyses:
            logger.info("  Running complexity analysis...")
            complexity = self.analyze_complexity(text)

        if self.ANALYSIS_SENTIMENT in analyses:
            logger.info("  Running sentiment analysis...")
            sentiment = self.analyze_sentiment(text)

        if self.ANALYSIS_TOPICS in analyses:
            logger.info("  Running topic modeling...")
            topics = self.analyze_topics(text)

        if self.ANALYSIS_ENTITIES in analyses:
            logger.info("  Running entity analysis...")
            entities = self.analyze_entities(text)

        processing_time = time.time() - start_time
        logger.info(f"  Analysis complete in {processing_time:.2f}s")

        return FullAnalysisResult(
            book_id=book.id,
            book_title=book.title,
            preprocessing=preprocessing,
            complexity=complexity,
            sentiment=sentiment,
            topics=topics,
            entities=entities,
            processing_time=processing_time,
        )

    def save_analysis_result(
        self,
        session: Session,
        book_id: int,
        analysis_type: str,
        results: Dict[str, Any],
        processing_time: float = None,
    ) -> AnalysisResult:
        """Save analysis result to database.

        Args:
            session: Database session
            book_id: Book ID
            analysis_type: Type of analysis
            results: Results dictionary
            processing_time: Time taken for analysis

        Returns:
            AnalysisResult model instance
        """
        # Check for existing result
        existing = session.query(AnalysisResult).filter(
            AnalysisResult.book_id == book_id,
            AnalysisResult.analysis_type == analysis_type,
            AnalysisResult.model_version == self.MODEL_VERSION,
        ).first()

        if existing:
            # Update existing
            existing.results_json = json.dumps(results)
            existing.processing_time_seconds = processing_time
            session.commit()
            return existing

        # Create new
        result = AnalysisResult(
            book_id=book_id,
            analysis_type=analysis_type,
            results_json=json.dumps(results),
            model_version=self.MODEL_VERSION,
            processing_time_seconds=processing_time,
        )
        session.add(result)
        session.commit()
        return result

    def save_full_analysis(
        self,
        session: Session,
        result: FullAnalysisResult,
    ) -> List[AnalysisResult]:
        """Save all analysis results for a book.

        Args:
            session: Database session
            result: FullAnalysisResult object

        Returns:
            List of saved AnalysisResult objects
        """
        saved = []

        if result.preprocessing:
            ar = self.save_analysis_result(
                session,
                result.book_id,
                self.ANALYSIS_PREPROCESSING,
                result.preprocessing,
            )
            saved.append(ar)

        if result.complexity:
            ar = self.save_analysis_result(
                session,
                result.book_id,
                self.ANALYSIS_COMPLEXITY,
                result.complexity.to_dict(),
            )
            saved.append(ar)

        if result.sentiment:
            ar = self.save_analysis_result(
                session,
                result.book_id,
                self.ANALYSIS_SENTIMENT,
                result.sentiment.to_dict(),
            )
            saved.append(ar)

        if result.topics:
            ar = self.save_analysis_result(
                session,
                result.book_id,
                self.ANALYSIS_TOPICS,
                result.topics.to_dict(),
            )
            saved.append(ar)

        if result.entities:
            ar = self.save_analysis_result(
                session,
                result.book_id,
                self.ANALYSIS_ENTITIES,
                result.entities.to_dict(),
            )
            saved.append(ar)

        # Save full combined result
        ar = self.save_analysis_result(
            session,
            result.book_id,
            self.ANALYSIS_FULL,
            result.to_dict(),
            result.processing_time,
        )
        saved.append(ar)

        return saved

    def get_analysis_result(
        self,
        session: Session,
        book_id: int,
        analysis_type: str,
    ) -> Optional[Dict[str, Any]]:
        """Retrieve analysis result from database.

        Args:
            session: Database session
            book_id: Book ID
            analysis_type: Type of analysis

        Returns:
            Results dictionary or None
        """
        result = session.query(AnalysisResult).filter(
            AnalysisResult.book_id == book_id,
            AnalysisResult.analysis_type == analysis_type,
        ).order_by(AnalysisResult.created_at.desc()).first()

        if result:
            return json.loads(result.results_json)
        return None

    def analyze_and_save(
        self,
        book_id: int,
        analyses: List[str] = None,
    ) -> FullAnalysisResult:
        """Analyze a book and save results to database.

        Args:
            book_id: Book ID to analyze
            analyses: List of analyses to run

        Returns:
            FullAnalysisResult object
        """
        with get_session_context() as session:
            book = session.query(Book).get(book_id)
            if not book:
                raise ValueError(f"Book {book_id} not found")

            result = self.analyze_book(book, analyses)
            self.save_full_analysis(session, result)

            return result

    def analyze_kafka_works(
        self,
        language: str = 'en',
        analyses: List[str] = None,
    ) -> List[FullAnalysisResult]:
        """Analyze all Kafka works in the database.

        Args:
            language: Filter by language
            analyses: List of analyses to run

        Returns:
            List of FullAnalysisResult objects
        """
        results = []

        with get_session_context() as session:
            # Find Kafka books
            kafka_books = session.query(Book).join(Book.author).filter(
                Book.author.has(name='Franz Kafka'),
                Book.language == language,
                Book.full_text.isnot(None),
            ).all()

            if not kafka_books:
                # Try alternative query
                kafka_books = session.query(Book).filter(
                    Book.title.ilike('%kafka%') | Book.title.ilike('%metamorphosis%') | Book.title.ilike('%trial%'),
                    Book.language == language,
                    Book.full_text.isnot(None),
                ).all()

            logger.info(f"Found {len(kafka_books)} Kafka works to analyze")

            for book in kafka_books:
                try:
                    result = self.analyze_book(book, analyses)
                    self.save_full_analysis(session, result)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error analyzing {book.title}: {e}")

        return results

    def analyze_corpus(
        self,
        limit: int = None,
        language: str = 'en',
        analyses: List[str] = None,
    ) -> List[FullAnalysisResult]:
        """Analyze all books in corpus.

        Args:
            limit: Maximum number of books to analyze
            language: Filter by language
            analyses: List of analyses to run

        Returns:
            List of FullAnalysisResult objects
        """
        results = []

        with get_session_context() as session:
            query = session.query(Book).filter(
                Book.language == language,
                Book.full_text.isnot(None),
            )

            if limit:
                query = query.limit(limit)

            books = query.all()
            logger.info(f"Found {len(books)} books to analyze")

            for i, book in enumerate(books):
                logger.info(f"Progress: {i + 1}/{len(books)}")
                try:
                    result = self.analyze_book(book, analyses)
                    self.save_full_analysis(session, result)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error analyzing {book.title}: {e}")

        return results

    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get summary of all analyses in database.

        Returns:
            Summary statistics dictionary
        """
        with get_session_context() as session:
            total_books = session.query(Book).filter(
                Book.full_text.isnot(None)
            ).count()

            analyzed_books = session.query(AnalysisResult.book_id).distinct().count()

            analysis_counts = {}
            for analysis_type in [
                self.ANALYSIS_COMPLEXITY,
                self.ANALYSIS_SENTIMENT,
                self.ANALYSIS_TOPICS,
                self.ANALYSIS_ENTITIES,
            ]:
                count = session.query(AnalysisResult).filter(
                    AnalysisResult.analysis_type == analysis_type
                ).count()
                analysis_counts[analysis_type] = count

            return {
                'total_books_with_text': total_books,
                'books_analyzed': analyzed_books,
                'coverage_percent': round(100 * analyzed_books / max(total_books, 1), 1),
                'analysis_counts': analysis_counts,
            }
