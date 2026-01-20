"""Tests for the manifest modules."""

import pytest

from src.scrapers.kafka_manifest import (
    KAFKA_WORKS,
    KAFKA_AUTHOR,
    get_kafka_book_ids,
    get_kafka_works,
    get_kafka_work_by_id,
)
from src.scrapers.corpus_manifest import (
    CLASSIC_WORKS,
    get_corpus_book_ids,
    get_corpus_works,
    get_works_by_movement,
    get_works_by_tag,
    get_works_by_author,
)


class TestKafkaManifest:
    """Tests for the Kafka manifest."""

    def test_kafka_works_not_empty(self):
        """Test that Kafka works list is not empty."""
        assert len(KAFKA_WORKS) > 0

    def test_kafka_works_have_required_fields(self):
        """Test that all Kafka works have required fields."""
        for work in KAFKA_WORKS:
            assert work.gutenberg_id is not None
            assert work.title is not None
            assert work.gutenberg_id > 0

    def test_kafka_author_info(self):
        """Test Kafka author information."""
        assert KAFKA_AUTHOR["name"] == "Franz Kafka"
        assert KAFKA_AUTHOR["birth_year"] == 1883
        assert KAFKA_AUTHOR["death_year"] == 1924

    def test_get_kafka_book_ids(self):
        """Test getting Kafka book IDs."""
        ids = get_kafka_book_ids()
        assert isinstance(ids, list)
        assert len(ids) == len(KAFKA_WORKS)
        assert 5200 in ids  # The Metamorphosis

    def test_get_kafka_works(self):
        """Test getting Kafka works list."""
        works = get_kafka_works()
        assert len(works) == len(KAFKA_WORKS)
        # Verify it's a copy, not the original
        works.pop()
        assert len(get_kafka_works()) == len(KAFKA_WORKS)

    def test_get_kafka_work_by_id_found(self):
        """Test finding a Kafka work by ID."""
        work = get_kafka_work_by_id(5200)
        assert work is not None
        assert work.title == "Metamorphosis"  # Gutenberg title

    def test_get_kafka_work_by_id_not_found(self):
        """Test searching for non-existent Kafka work."""
        work = get_kafka_work_by_id(99999)
        assert work is None

    def test_metamorphosis_in_works(self):
        """Test that Metamorphosis is in the works list."""
        titles = [w.title for w in KAFKA_WORKS]
        assert "Metamorphosis" in titles  # Gutenberg title


class TestCorpusManifest:
    """Tests for the classic literature corpus manifest."""

    def test_corpus_not_empty(self):
        """Test that corpus is not empty."""
        assert len(CLASSIC_WORKS) > 0

    def test_corpus_has_minimum_size(self):
        """Test that corpus has at least 50 works."""
        assert len(CLASSIC_WORKS) >= 50

    def test_corpus_works_have_required_fields(self):
        """Test that all corpus works have required fields."""
        for work in CLASSIC_WORKS:
            assert work.gutenberg_id is not None
            assert work.title is not None
            assert work.author is not None

    def test_get_corpus_book_ids_unique(self):
        """Test that corpus book IDs are unique."""
        ids = get_corpus_book_ids()
        assert len(ids) == len(set(ids))

    def test_get_works_by_movement(self):
        """Test filtering works by literary movement."""
        modernist = get_works_by_movement("Modernism")
        assert len(modernist) > 0
        for work in modernist:
            assert "modernism" in work.movement.lower()

    def test_get_works_by_tag(self):
        """Test filtering works by thematic tag."""
        alienation_works = get_works_by_tag("alienation")
        assert len(alienation_works) > 0
        for work in alienation_works:
            assert "alienation" in [t.lower() for t in work.thematic_tags]

    def test_get_works_by_author(self):
        """Test filtering works by author."""
        dostoevsky = get_works_by_author("Dostoevsky")
        assert len(dostoevsky) > 0
        for work in dostoevsky:
            assert "dostoevsky" in work.author.lower()

    def test_corpus_includes_major_authors(self):
        """Test that corpus includes major literary figures."""
        authors = {w.author for w in CLASSIC_WORKS}
        expected_authors = [
            "Fyodor Dostoevsky",
            "Leo Tolstoy",
            "Charles Dickens",
            "Jane Austen",
            "Herman Melville",
        ]
        for author in expected_authors:
            assert any(author in a for a in authors), f"{author} not found in corpus"

    def test_corpus_includes_classic_works(self):
        """Test that corpus includes well-known classics."""
        titles = {w.title for w in CLASSIC_WORKS}
        expected_titles = [
            "Crime and Punishment",
            "Pride and Prejudice",
            "Moby Dick",
            "Frankenstein",
        ]
        for title in expected_titles:
            assert title in titles, f"{title} not found in corpus"
