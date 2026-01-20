"""Tests for the NLP analysis pipeline (Phase 2)."""

import pytest
from unittest.mock import MagicMock, patch

from src.preprocessing.text_processor import TextProcessor, ProcessedText
from src.analysis.complexity_analyzer import ComplexityAnalyzer, ComplexityMetrics
from src.analysis.sentiment_analyzer import SentimentAnalyzer, SentimentResult
from src.analysis.topic_modeler import TopicModeler
from src.analysis.entity_analyzer import EntityAnalyzer


# Sample text for testing
SAMPLE_TEXT = """
One morning, when Gregor Samsa woke from troubled dreams, he found himself
transformed in his bed into a horrible vermin. He lay on his armour-like back,
and if he lifted his head a little he could see his brown belly, slightly domed
and divided by arches into stiff sections. The bedding was hardly able to cover
it and seemed ready to slide off any moment. His many legs, pitifully thin
compared with the size of the rest of him, waved about helplessly as he looked.

"What's happened to me?" he thought. It wasn't a dream. His room, a proper human
room although a little too small, lay peacefully between its four familiar walls.
A collection of textile samples lay spread out on the table - Samsa was a
travelling salesman - and above it there hung a picture that he had recently cut
out of an illustrated magazine and housed in a nice, gilded frame.
"""

GUTENBERG_TEXT = """
*** START OF THE PROJECT GUTENBERG EBOOK THE METAMORPHOSIS ***

One morning, when Gregor Samsa woke from troubled dreams, he found himself
transformed in his bed into a horrible vermin.

*** END OF THE PROJECT GUTENBERG EBOOK THE METAMORPHOSIS ***
"""


class TestTextProcessor:
    """Tests for text preprocessing."""

    def test_clean_gutenberg_text(self):
        """Test removal of Gutenberg headers/footers."""
        processor = TextProcessor()
        cleaned = processor.clean_gutenberg_text(GUTENBERG_TEXT)

        assert "PROJECT GUTENBERG" not in cleaned
        assert "Gregor Samsa" in cleaned

    def test_clean_text(self):
        """Test basic text cleaning."""
        processor = TextProcessor()
        text = "  Multiple   spaces   and\n\nnewlines  "
        cleaned = processor.clean_text(text)

        assert "  " not in cleaned
        assert cleaned.strip() == cleaned

    def test_tokenize_sentences(self):
        """Test sentence tokenization."""
        processor = TextProcessor()
        sentences = processor.tokenize_sentences(SAMPLE_TEXT)

        assert len(sentences) > 1
        assert any("Gregor Samsa" in s for s in sentences)

    def test_tokenize_words(self):
        """Test word tokenization."""
        processor = TextProcessor()
        tokens = processor.tokenize_words("Hello world, this is a test.")

        assert "hello" in tokens or "Hello" in tokens
        assert len(tokens) > 0

    def test_filter_tokens(self):
        """Test token filtering."""
        processor = TextProcessor()
        tokens = ["the", "quick", "brown", "fox", "a", "an"]

        # Filter stopwords
        filtered = processor.filter_tokens(tokens, remove_stopwords=True)
        assert "the" not in filtered
        assert "quick" in filtered

    def test_process_full_pipeline(self):
        """Test full preprocessing pipeline."""
        processor = TextProcessor()
        result = processor.process(SAMPLE_TEXT)

        assert isinstance(result, ProcessedText)
        assert result.word_count > 0
        assert result.sentence_count > 0
        assert len(result.tokens) > 0
        assert len(result.lemmas) > 0

    def test_processed_text_to_dict(self):
        """Test ProcessedText serialization."""
        processor = TextProcessor()
        result = processor.process(SAMPLE_TEXT)

        data = result.to_dict()
        assert "word_count" in data
        assert "sentence_count" in data


class TestComplexityAnalyzer:
    """Tests for complexity analysis."""

    def test_analyze_basic(self):
        """Test basic complexity analysis."""
        analyzer = ComplexityAnalyzer()
        result = analyzer.analyze(SAMPLE_TEXT)

        assert isinstance(result, ComplexityMetrics)
        assert result.word_count > 0
        assert result.sentence_count > 0

    def test_readability_scores(self):
        """Test that readability scores are calculated."""
        analyzer = ComplexityAnalyzer()
        result = analyzer.analyze(SAMPLE_TEXT)

        # Flesch Reading Ease should be between 0-100 typically
        assert -100 < result.flesch_reading_ease < 200
        # Grade level should be positive
        assert result.flesch_kincaid_grade > 0

    def test_vocabulary_metrics(self):
        """Test vocabulary diversity metrics."""
        analyzer = ComplexityAnalyzer()
        result = analyzer.analyze(SAMPLE_TEXT)

        assert result.unique_words > 0
        assert 0 <= result.type_token_ratio <= 1
        assert result.hapax_legomena >= 0

    def test_to_dict(self):
        """Test ComplexityMetrics serialization."""
        analyzer = ComplexityAnalyzer()
        result = analyzer.analyze(SAMPLE_TEXT)

        data = result.to_dict()
        assert "readability" in data
        assert "vocabulary" in data
        assert "text_statistics" in data

    def test_reading_level_interpretation(self):
        """Test reading level interpretation."""
        analyzer = ComplexityAnalyzer()
        result = analyzer.analyze(SAMPLE_TEXT)

        level = result.get_reading_level()
        assert isinstance(level, str)
        assert len(level) > 0


class TestSentimentAnalyzer:
    """Tests for sentiment analysis."""

    def test_analyze_basic(self):
        """Test basic sentiment analysis."""
        analyzer = SentimentAnalyzer()
        result = analyzer.analyze(SAMPLE_TEXT)

        assert isinstance(result, SentimentResult)
        assert -1 <= result.polarity <= 1
        assert 0 <= result.subjectivity <= 1

    def test_vader_scores(self):
        """Test VADER sentiment scores."""
        analyzer = SentimentAnalyzer()
        result = analyzer.analyze(SAMPLE_TEXT)

        assert -1 <= result.vader_compound <= 1
        assert 0 <= result.vader_positive <= 1
        assert 0 <= result.vader_negative <= 1
        assert 0 <= result.vader_neutral <= 1

    def test_sentence_counts(self):
        """Test sentence sentiment classification."""
        analyzer = SentimentAnalyzer()
        result = analyzer.analyze(SAMPLE_TEXT)

        total = result.positive_sentences + result.negative_sentences + result.neutral_sentences
        assert total == result.sentence_count

    def test_sentiment_progression(self):
        """Test emotional arc tracking."""
        analyzer = SentimentAnalyzer()
        result = analyzer.analyze(SAMPLE_TEXT)

        assert len(result.sentiment_progression) > 0
        assert all(-1 <= v <= 1 for v in result.sentiment_progression)

    def test_to_dict(self):
        """Test SentimentResult serialization."""
        analyzer = SentimentAnalyzer()
        result = analyzer.analyze(SAMPLE_TEXT)

        data = result.to_dict()
        assert "overall" in data
        assert "sentence_statistics" in data
        assert "emotional_arc" in data

    def test_positive_text(self):
        """Test sentiment on positive text."""
        analyzer = SentimentAnalyzer()
        positive_text = "I love this wonderful, amazing, beautiful day! Everything is fantastic!"
        result = analyzer.analyze(positive_text)

        assert result.vader_compound > 0
        assert result.get_overall_sentiment() in ["Positive", "Strongly Positive"]

    def test_negative_text(self):
        """Test sentiment on negative text."""
        analyzer = SentimentAnalyzer()
        negative_text = "This is terrible, awful, horrible. I hate everything about it."
        result = analyzer.analyze(negative_text)

        assert result.vader_compound < 0
        assert result.get_overall_sentiment() in ["Negative", "Strongly Negative"]


class TestTopicModeler:
    """Tests for topic modeling."""

    def test_preprocess_text(self):
        """Test text preprocessing for topic modeling."""
        modeler = TopicModeler()
        tokens = modeler.preprocess_text(SAMPLE_TEXT)

        assert len(tokens) > 0
        assert all(isinstance(t, str) for t in tokens)
        # Stopwords should be removed
        assert "the" not in tokens

    def test_preprocess_documents(self):
        """Test document preprocessing."""
        modeler = TopicModeler()
        docs = [SAMPLE_TEXT, "Another document about something else."]
        processed, phraser = modeler.preprocess_documents(docs, use_phrases=False)

        assert len(processed) == 2
        assert all(isinstance(doc, list) for doc in processed)

    def test_literary_themes(self):
        """Test literary theme definitions exist."""
        modeler = TopicModeler()

        assert "alienation" in modeler.LITERARY_THEMES
        assert "bureaucracy" in modeler.LITERARY_THEMES
        assert "transformation" in modeler.LITERARY_THEMES


class TestEntityAnalyzer:
    """Tests for entity analysis."""

    def test_should_filter_entity(self):
        """Test entity filtering logic."""
        analyzer = EntityAnalyzer(use_spacy=False)

        assert analyzer._should_filter_entity("Mr.")
        assert analyzer._should_filter_entity("Chapter 1")
        assert not analyzer._should_filter_entity("Gregor Samsa")

    def test_normalize_entity(self):
        """Test entity normalization."""
        analyzer = EntityAnalyzer(use_spacy=False)

        assert analyzer._normalize_entity("Gregor's") == "Gregor"
        assert analyzer._normalize_entity("  Samsa  ") == "Samsa"

    def test_extract_entities_regex(self):
        """Test regex-based entity extraction."""
        analyzer = EntityAnalyzer(use_spacy=False)
        entities = analyzer.extract_entities_regex(SAMPLE_TEXT)

        assert "PERSON" in entities
        # Should find Gregor Samsa
        names = [e.text for e in entities.get("PERSON", [])]
        assert any("Gregor" in n or "Samsa" in n for n in names)

    def test_analyze_without_spacy(self):
        """Test analysis without spaCy (regex fallback)."""
        analyzer = EntityAnalyzer(use_spacy=False)
        result = analyzer.analyze(SAMPLE_TEXT, build_network=False)

        assert result.total_entities > 0
        assert len(result.characters) > 0

    def test_result_to_dict(self):
        """Test EntityAnalysisResult serialization."""
        analyzer = EntityAnalyzer(use_spacy=False)
        result = analyzer.analyze(SAMPLE_TEXT, build_network=False)

        data = result.to_dict()
        assert "summary" in data
        assert "characters" in data


class TestIntegration:
    """Integration tests for the full pipeline."""

    def test_full_text_analysis(self):
        """Test running all analyses on sample text."""
        # Preprocessing
        processor = TextProcessor()
        processed = processor.process(SAMPLE_TEXT)
        assert processed.word_count > 0

        # Complexity
        complexity = ComplexityAnalyzer()
        complexity_result = complexity.analyze(SAMPLE_TEXT)
        assert complexity_result.flesch_reading_ease != 0

        # Sentiment
        sentiment = SentimentAnalyzer()
        sentiment_result = sentiment.analyze(SAMPLE_TEXT)
        assert sentiment_result.sentence_count > 0

        # Entities (without spaCy for faster testing)
        entities = EntityAnalyzer(use_spacy=False)
        entity_result = entities.analyze(SAMPLE_TEXT, build_network=False)
        assert entity_result.total_entities > 0

    def test_kafka_specific_themes(self):
        """Test that Kafka-specific themes can be detected."""
        kafka_text = """
        Gregor felt isolated and alienated from his family. The transformation
        had made him a stranger in his own home. His father's authority seemed
        absolute, bureaucratic in its coldness. The guilt weighed heavily on him
        as he contemplated his meaningless existence.
        """

        modeler = TopicModeler()
        tokens = modeler.preprocess_text(kafka_text)

        # Check that thematic words are preserved
        assert any(word in tokens for word in ["isolated", "alienated", "guilt", "transformation"])
