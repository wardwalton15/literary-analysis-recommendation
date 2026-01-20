"""Text complexity and readability analysis.

Provides comprehensive metrics for analyzing literary text complexity,
including readability scores, vocabulary diversity, and sentence structure.
"""

import re
import math
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from collections import Counter

import textstat
from nltk.tokenize import sent_tokenize, word_tokenize


@dataclass
class ComplexityMetrics:
    """Container for text complexity analysis results."""

    # Readability scores
    flesch_reading_ease: float
    flesch_kincaid_grade: float
    gunning_fog: float
    smog_index: float
    coleman_liau_index: float
    automated_readability_index: float
    dale_chall_score: float
    linsear_write: float

    # Text statistics
    word_count: int
    sentence_count: int
    syllable_count: int
    character_count: int
    avg_word_length: float
    avg_sentence_length: float
    avg_syllables_per_word: float

    # Vocabulary metrics
    unique_words: int
    type_token_ratio: float
    hapax_legomena: int  # Words appearing only once
    hapax_ratio: float
    vocabulary_richness: float  # Yule's K

    # Sentence structure
    complex_sentence_ratio: float
    simple_sentence_ratio: float
    avg_clause_count: float

    # Word length distribution
    short_words_ratio: float  # <= 4 chars
    medium_words_ratio: float  # 5-8 chars
    long_words_ratio: float  # >= 9 chars

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'readability': {
                'flesch_reading_ease': self.flesch_reading_ease,
                'flesch_kincaid_grade': self.flesch_kincaid_grade,
                'gunning_fog': self.gunning_fog,
                'smog_index': self.smog_index,
                'coleman_liau_index': self.coleman_liau_index,
                'automated_readability_index': self.automated_readability_index,
                'dale_chall_score': self.dale_chall_score,
                'linsear_write': self.linsear_write,
            },
            'text_statistics': {
                'word_count': self.word_count,
                'sentence_count': self.sentence_count,
                'syllable_count': self.syllable_count,
                'character_count': self.character_count,
                'avg_word_length': round(self.avg_word_length, 2),
                'avg_sentence_length': round(self.avg_sentence_length, 2),
                'avg_syllables_per_word': round(self.avg_syllables_per_word, 2),
            },
            'vocabulary': {
                'unique_words': self.unique_words,
                'type_token_ratio': round(self.type_token_ratio, 4),
                'hapax_legomena': self.hapax_legomena,
                'hapax_ratio': round(self.hapax_ratio, 4),
                'vocabulary_richness': round(self.vocabulary_richness, 4),
            },
            'sentence_structure': {
                'complex_sentence_ratio': round(self.complex_sentence_ratio, 4),
                'simple_sentence_ratio': round(self.simple_sentence_ratio, 4),
                'avg_clause_count': round(self.avg_clause_count, 2),
            },
            'word_length_distribution': {
                'short_words_ratio': round(self.short_words_ratio, 4),
                'medium_words_ratio': round(self.medium_words_ratio, 4),
                'long_words_ratio': round(self.long_words_ratio, 4),
            },
        }

    def get_reading_level(self) -> str:
        """Interpret the Flesch-Kincaid grade level."""
        grade = self.flesch_kincaid_grade
        if grade < 1:
            return "Kindergarten"
        elif grade < 6:
            return f"Elementary (Grade {int(grade)})"
        elif grade < 9:
            return f"Middle School (Grade {int(grade)})"
        elif grade < 13:
            return f"High School (Grade {int(grade)})"
        elif grade < 17:
            return "College"
        else:
            return "Graduate/Professional"

    def get_readability_interpretation(self) -> str:
        """Interpret the Flesch Reading Ease score."""
        score = self.flesch_reading_ease
        if score >= 90:
            return "Very Easy (5th grade)"
        elif score >= 80:
            return "Easy (6th grade)"
        elif score >= 70:
            return "Fairly Easy (7th grade)"
        elif score >= 60:
            return "Standard (8th-9th grade)"
        elif score >= 50:
            return "Fairly Difficult (10th-12th grade)"
        elif score >= 30:
            return "Difficult (College)"
        else:
            return "Very Difficult (Graduate)"


class ComplexityAnalyzer:
    """Analyzer for text complexity and readability metrics."""

    def __init__(self):
        """Initialize the complexity analyzer."""
        # Subordinating conjunctions for clause detection
        self.subordinating_conjunctions = {
            'after', 'although', 'as', 'because', 'before', 'if', 'once',
            'since', 'than', 'that', 'though', 'till', 'unless', 'until',
            'when', 'where', 'whether', 'while', 'which', 'who', 'whom',
            'whose', 'whereas', 'whereby', 'whenever', 'wherever',
        }

        # Coordinating conjunctions
        self.coordinating_conjunctions = {'and', 'but', 'or', 'nor', 'for', 'yet', 'so'}

    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word using simple heuristics.

        Args:
            word: Input word

        Returns:
            Syllable count
        """
        word = word.lower()
        count = 0
        vowels = 'aeiouy'
        if word and word[0] in vowels:
            count += 1
        for i in range(1, len(word)):
            if word[i] in vowels and word[i - 1] not in vowels:
                count += 1
        # Handle silent e
        if word.endswith('e'):
            count -= 1
        # Handle le endings
        if word.endswith('le') and len(word) > 2 and word[-3] not in vowels:
            count += 1
        if count == 0:
            count = 1
        return count

    def _calculate_yules_k(self, tokens: List[str]) -> float:
        """Calculate Yule's K measure of vocabulary richness.

        Lower values indicate more diverse vocabulary.

        Args:
            tokens: List of word tokens

        Returns:
            Yule's K value
        """
        word_freq = Counter(t.lower() for t in tokens if t.isalpha())

        if not word_freq:
            return 0.0

        # Get frequency of frequencies
        freq_of_freq = Counter(word_freq.values())

        n = sum(word_freq.values())
        if n <= 1:
            return 0.0

        # Calculate M1 and M2
        m1 = n
        m2 = sum(f * (count ** 2) for f, count in freq_of_freq.items())

        # Yule's K
        try:
            k = 10000 * (m2 - m1) / (m1 ** 2)
        except ZeroDivisionError:
            k = 0.0

        return k

    def _analyze_sentence_complexity(
        self,
        sentences: List[str],
        tokens_per_sentence: List[List[str]],
    ) -> Dict[str, float]:
        """Analyze sentence structure complexity.

        Args:
            sentences: List of sentences
            tokens_per_sentence: List of token lists for each sentence

        Returns:
            Dictionary with complexity metrics
        """
        if not sentences:
            return {
                'complex_ratio': 0.0,
                'simple_ratio': 0.0,
                'avg_clauses': 0.0,
            }

        complex_count = 0
        simple_count = 0
        total_clauses = 0

        for sentence, tokens in zip(sentences, tokens_per_sentence):
            lower_tokens = [t.lower() for t in tokens]

            # Count potential clauses based on conjunctions and punctuation
            clause_markers = sum(
                1 for t in lower_tokens
                if t in self.subordinating_conjunctions or t in self.coordinating_conjunctions
            )

            # Add clause markers from punctuation
            clause_markers += sentence.count(';') + sentence.count(':')

            clauses = 1 + clause_markers
            total_clauses += clauses

            # Simple sentence: single clause
            # Complex sentence: multiple clauses
            if clauses == 1:
                simple_count += 1
            else:
                complex_count += 1

        return {
            'complex_ratio': complex_count / len(sentences),
            'simple_ratio': simple_count / len(sentences),
            'avg_clauses': total_clauses / len(sentences),
        }

    def _analyze_word_lengths(self, tokens: List[str]) -> Dict[str, float]:
        """Analyze word length distribution.

        Args:
            tokens: List of word tokens

        Returns:
            Dictionary with word length ratios
        """
        alpha_tokens = [t for t in tokens if t.isalpha()]

        if not alpha_tokens:
            return {
                'short_ratio': 0.0,
                'medium_ratio': 0.0,
                'long_ratio': 0.0,
            }

        total = len(alpha_tokens)

        short_count = sum(1 for t in alpha_tokens if len(t) <= 4)
        medium_count = sum(1 for t in alpha_tokens if 5 <= len(t) <= 8)
        long_count = sum(1 for t in alpha_tokens if len(t) >= 9)

        return {
            'short_ratio': short_count / total,
            'medium_ratio': medium_count / total,
            'long_ratio': long_count / total,
        }

    def analyze(self, text: str) -> ComplexityMetrics:
        """Perform comprehensive complexity analysis on text.

        Args:
            text: Input text to analyze

        Returns:
            ComplexityMetrics object with all metrics
        """
        # Basic tokenization
        sentences = sent_tokenize(text)
        tokens = word_tokenize(text)
        alpha_tokens = [t for t in tokens if t.isalpha()]

        # Text statistics
        word_count = len(alpha_tokens)
        sentence_count = len(sentences) if sentences else 1
        character_count = sum(len(t) for t in alpha_tokens)

        # Syllable count
        syllable_count = sum(self._count_syllables(t) for t in alpha_tokens)

        # Averages
        avg_word_length = character_count / word_count if word_count > 0 else 0
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
        avg_syllables_per_word = syllable_count / word_count if word_count > 0 else 0

        # Vocabulary metrics
        word_freq = Counter(t.lower() for t in alpha_tokens)
        unique_words = len(word_freq)
        type_token_ratio = unique_words / word_count if word_count > 0 else 0

        # Hapax legomena (words appearing once)
        hapax_legomena = sum(1 for count in word_freq.values() if count == 1)
        hapax_ratio = hapax_legomena / unique_words if unique_words > 0 else 0

        # Vocabulary richness (Yule's K)
        vocabulary_richness = self._calculate_yules_k(alpha_tokens)

        # Sentence complexity
        tokens_per_sentence = [word_tokenize(s) for s in sentences]
        sentence_complexity = self._analyze_sentence_complexity(
            sentences, tokens_per_sentence
        )

        # Word length distribution
        word_lengths = self._analyze_word_lengths(alpha_tokens)

        # Readability scores using textstat
        flesch_reading_ease = textstat.flesch_reading_ease(text)
        flesch_kincaid_grade = textstat.flesch_kincaid_grade(text)
        gunning_fog = textstat.gunning_fog(text)
        smog_index = textstat.smog_index(text)
        coleman_liau_index = textstat.coleman_liau_index(text)
        automated_readability_index = textstat.automated_readability_index(text)
        dale_chall_score = textstat.dale_chall_readability_score(text)
        linsear_write = textstat.linsear_write_formula(text)

        return ComplexityMetrics(
            flesch_reading_ease=flesch_reading_ease,
            flesch_kincaid_grade=flesch_kincaid_grade,
            gunning_fog=gunning_fog,
            smog_index=smog_index,
            coleman_liau_index=coleman_liau_index,
            automated_readability_index=automated_readability_index,
            dale_chall_score=dale_chall_score,
            linsear_write=linsear_write,
            word_count=word_count,
            sentence_count=sentence_count,
            syllable_count=syllable_count,
            character_count=character_count,
            avg_word_length=avg_word_length,
            avg_sentence_length=avg_sentence_length,
            avg_syllables_per_word=avg_syllables_per_word,
            unique_words=unique_words,
            type_token_ratio=type_token_ratio,
            hapax_legomena=hapax_legomena,
            hapax_ratio=hapax_ratio,
            vocabulary_richness=vocabulary_richness,
            complex_sentence_ratio=sentence_complexity['complex_ratio'],
            simple_sentence_ratio=sentence_complexity['simple_ratio'],
            avg_clause_count=sentence_complexity['avg_clauses'],
            short_words_ratio=word_lengths['short_ratio'],
            medium_words_ratio=word_lengths['medium_ratio'],
            long_words_ratio=word_lengths['long_ratio'],
        )

    def analyze_by_section(
        self,
        text: str,
        section_size: int = 5000,
    ) -> List[Dict[str, Any]]:
        """Analyze complexity across sections of text.

        Useful for tracking complexity changes throughout a work.

        Args:
            text: Input text
            section_size: Size of each section in words

        Returns:
            List of metrics for each section
        """
        words = text.split()
        sections = []

        for i in range(0, len(words), section_size):
            section_words = words[i:i + section_size]
            section_text = ' '.join(section_words)

            metrics = self.analyze(section_text)
            sections.append({
                'section_index': i // section_size,
                'start_word': i,
                'end_word': min(i + section_size, len(words)),
                'metrics': metrics.to_dict(),
            })

        return sections

    def compare_texts(
        self,
        texts: List[str],
        labels: List[str] = None,
    ) -> Dict[str, List[Any]]:
        """Compare complexity metrics across multiple texts.

        Args:
            texts: List of texts to compare
            labels: Optional labels for each text

        Returns:
            Dictionary with comparison data
        """
        if labels is None:
            labels = [f"Text {i + 1}" for i in range(len(texts))]

        results = {
            'labels': labels,
            'flesch_reading_ease': [],
            'flesch_kincaid_grade': [],
            'type_token_ratio': [],
            'avg_sentence_length': [],
            'vocabulary_richness': [],
        }

        for text in texts:
            metrics = self.analyze(text)
            results['flesch_reading_ease'].append(metrics.flesch_reading_ease)
            results['flesch_kincaid_grade'].append(metrics.flesch_kincaid_grade)
            results['type_token_ratio'].append(metrics.type_token_ratio)
            results['avg_sentence_length'].append(metrics.avg_sentence_length)
            results['vocabulary_richness'].append(metrics.vocabulary_richness)

        return results
