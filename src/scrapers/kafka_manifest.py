"""Franz Kafka works manifest for Project Gutenberg."""

from dataclasses import dataclass
from typing import Optional, List


@dataclass
class KafkaWork:
    """A work by Franz Kafka available on Project Gutenberg."""

    gutenberg_id: int
    title: str
    original_title: Optional[str] = None
    year_written: Optional[int] = None
    year_published: Optional[int] = None
    work_type: str = "novel"  # novel, novella, short_story, collection
    language: str = "en"  # en or de
    notes: Optional[str] = None


# Kafka works available on Project Gutenberg
# Verified IDs from https://www.gutenberg.org/ebooks/author/1735
KAFKA_WORKS = [
    # === English Translations ===
    KafkaWork(
        gutenberg_id=5200,
        title="Metamorphosis",
        original_title="Die Verwandlung",
        year_written=1912,
        year_published=1915,
        work_type="novella",
        language="en",
        notes="Kafka's most famous work about Gregor Samsa's transformation",
    ),
    KafkaWork(
        gutenberg_id=7849,
        title="The Trial",
        original_title="Der Process",
        year_written=1914,
        year_published=1925,
        work_type="novel",
        language="en",
        notes="Unfinished novel about Josef K.'s arrest and prosecution",
    ),
    # === German Originals ===
    KafkaWork(
        gutenberg_id=22367,
        title="Die Verwandlung",
        original_title="Die Verwandlung",
        year_written=1912,
        year_published=1915,
        work_type="novella",
        language="de",
        notes="Original German text of The Metamorphosis",
    ),
    KafkaWork(
        gutenberg_id=69327,
        title="Der Prozess: Roman",
        original_title="Der Process",
        year_written=1914,
        year_published=1925,
        work_type="novel",
        language="de",
        notes="Original German text of The Trial",
    ),
    KafkaWork(
        gutenberg_id=25791,
        title="In der Strafkolonie",
        original_title="In der Strafkolonie",
        year_written=1914,
        year_published=1919,
        work_type="short_story",
        language="de",
        notes="In the Penal Colony - German original, story about execution machine",
    ),
    KafkaWork(
        gutenberg_id=30655,
        title="Ein Hungerkünstler",
        original_title="Ein Hungerkünstler",
        year_written=1922,
        year_published=1922,
        work_type="short_story",
        language="de",
        notes="A Hunger Artist - German original, story about a professional faster",
    ),
    KafkaWork(
        gutenberg_id=21593,
        title="Das Urteil: Eine Geschichte",
        original_title="Das Urteil",
        year_written=1912,
        year_published=1913,
        work_type="short_story",
        language="de",
        notes="The Judgment - German original, story about father-son conflict",
    ),
    KafkaWork(
        gutenberg_id=21989,
        title="Ein Landarzt: Kleine Erzählungen",
        original_title="Ein Landarzt",
        year_written=1917,
        year_published=1919,
        work_type="collection",
        language="de",
        notes="A Country Doctor - German original, short story collection",
    ),
    KafkaWork(
        gutenberg_id=23532,
        title="Betrachtung",
        original_title="Betrachtung",
        year_written=1912,
        year_published=1912,
        work_type="collection",
        language="de",
        notes="Contemplation/Meditation - Kafka's first published book, prose pieces",
    ),
]

# Author metadata
KAFKA_AUTHOR = {
    "name": "Franz Kafka",
    "birth_year": 1883,
    "death_year": 1924,
    "nationality": "Austrian (Bohemian)",
    "biography": (
        "Franz Kafka (1883-1924) was a German-speaking Bohemian novelist and short "
        "story writer, widely regarded as one of the major figures of 20th-century "
        "literature. His work, which fuses elements of realism and the fantastic, "
        "typically features isolated protagonists facing bizarre or surrealistic "
        "predicaments and incomprehensible socio-bureaucratic powers. His themes of "
        "alienation, existential anxiety, guilt, and absurdity have become integral "
        "to modern literature. Major works include The Metamorphosis, The Trial, "
        "and The Castle."
    ),
}


def get_kafka_book_ids() -> List[int]:
    """Get list of Gutenberg IDs for Kafka works."""
    return [work.gutenberg_id for work in KAFKA_WORKS]


def get_kafka_works() -> List[KafkaWork]:
    """Get all Kafka works."""
    return KAFKA_WORKS.copy()


def get_kafka_work_by_id(gutenberg_id: int) -> Optional[KafkaWork]:
    """Get a specific Kafka work by Gutenberg ID."""
    for work in KAFKA_WORKS:
        if work.gutenberg_id == gutenberg_id:
            return work
    return None


def get_english_kafka_works() -> List[KafkaWork]:
    """Get only English Kafka works."""
    return [work for work in KAFKA_WORKS if work.language == "en"]


def get_german_kafka_works() -> List[KafkaWork]:
    """Get only German Kafka works."""
    return [work for work in KAFKA_WORKS if work.language == "de"]
