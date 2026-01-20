"""Sentiment and emotional analysis for literary texts.

Provides multi-level sentiment analysis including overall sentiment,
chapter/section sentiment progression, and emotional arc tracking.
"""

import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

from textblob import TextBlob
from nltk.tokenize import sent_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Ensure VADER lexicon is downloaded
import nltk
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)


@dataclass
class SentimentResult:
    """Container for sentiment analysis results."""

    # Overall scores
    polarity: float  # -1 (negative) to 1 (positive)
    subjectivity: float  # 0 (objective) to 1 (subjective)

    # VADER scores
    vader_compound: float  # -1 to 1 overall sentiment
    vader_positive: float  # Proportion of positive sentiment
    vader_negative: float  # Proportion of negative sentiment
    vader_neutral: float  # Proportion of neutral sentiment

    # Statistics
    sentence_count: int
    positive_sentences: int
    negative_sentences: int
    neutral_sentences: int

    # Emotional arc data
    sentiment_progression: List[float] = field(default_factory=list)
    emotional_peaks: List[Dict[str, Any]] = field(default_factory=list)
    emotional_valleys: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'overall': {
                'polarity': round(self.polarity, 4),
                'subjectivity': round(self.subjectivity, 4),
                'vader_compound': round(self.vader_compound, 4),
                'vader_positive': round(self.vader_positive, 4),
                'vader_negative': round(self.vader_negative, 4),
                'vader_neutral': round(self.vader_neutral, 4),
            },
            'sentence_statistics': {
                'total': self.sentence_count,
                'positive': self.positive_sentences,
                'negative': self.negative_sentences,
                'neutral': self.neutral_sentences,
                'positive_ratio': round(self.positive_sentences / max(self.sentence_count, 1), 4),
                'negative_ratio': round(self.negative_sentences / max(self.sentence_count, 1), 4),
            },
            'emotional_arc': {
                'progression': [round(s, 4) for s in self.sentiment_progression],
                'peaks': self.emotional_peaks,
                'valleys': self.emotional_valleys,
            },
        }

    def get_overall_sentiment(self) -> str:
        """Get a text description of the overall sentiment."""
        if self.vader_compound >= 0.05:
            if self.vader_compound >= 0.5:
                return "Strongly Positive"
            return "Positive"
        elif self.vader_compound <= -0.05:
            if self.vader_compound <= -0.5:
                return "Strongly Negative"
            return "Negative"
        return "Neutral"

    def get_emotional_tone(self) -> str:
        """Describe the emotional tone based on subjectivity and variance."""
        if self.subjectivity > 0.6:
            return "Highly Emotional/Subjective"
        elif self.subjectivity > 0.4:
            return "Moderately Emotional"
        else:
            return "Objective/Detached"


class SentimentAnalyzer:
    """Comprehensive sentiment analyzer for literary texts."""

    def __init__(self):
        """Initialize sentiment analysis tools."""
        self.vader = SentimentIntensityAnalyzer()

        # Literary-specific sentiment modifiers
        self.literary_positive_words = {
            'hope', 'love', 'joy', 'peace', 'beauty', 'light', 'grace',
            'freedom', 'triumph', 'redemption', 'salvation', 'sublime',
        }
        self.literary_negative_words = {
            'despair', 'death', 'darkness', 'doom', 'guilt', 'shame',
            'alienation', 'absurd', 'futile', 'anguish', 'torment',
            'isolation', 'suffering', 'dread', 'anxiety', 'horror',
        }

    def _analyze_sentence_vader(self, sentence: str) -> Dict[str, float]:
        """Analyze a single sentence with VADER.

        Args:
            sentence: Input sentence

        Returns:
            VADER scores dictionary
        """
        return self.vader.polarity_scores(sentence)

    def _analyze_sentence_textblob(self, sentence: str) -> Tuple[float, float]:
        """Analyze a single sentence with TextBlob.

        Args:
            sentence: Input sentence

        Returns:
            Tuple of (polarity, subjectivity)
        """
        blob = TextBlob(sentence)
        return blob.sentiment.polarity, blob.sentiment.subjectivity

    def _classify_sentiment(self, compound_score: float) -> str:
        """Classify sentiment as positive, negative, or neutral.

        Args:
            compound_score: VADER compound score

        Returns:
            Sentiment classification
        """
        if compound_score >= 0.05:
            return 'positive'
        elif compound_score <= -0.05:
            return 'negative'
        return 'neutral'

    def _find_emotional_extremes(
        self,
        progression: List[float],
        window_size: int = 5,
        threshold: float = 0.3,
    ) -> Tuple[List[Dict], List[Dict]]:
        """Find emotional peaks and valleys in sentiment progression.

        Args:
            progression: List of sentiment scores
            window_size: Window size for smoothing
            threshold: Minimum absolute value for extreme

        Returns:
            Tuple of (peaks, valleys) lists
        """
        peaks = []
        valleys = []

        if len(progression) < window_size * 2:
            return peaks, valleys

        # Smooth the progression with moving average
        smoothed = []
        for i in range(len(progression)):
            start = max(0, i - window_size // 2)
            end = min(len(progression), i + window_size // 2 + 1)
            smoothed.append(sum(progression[start:end]) / (end - start))

        # Find local maxima (peaks) and minima (valleys)
        for i in range(1, len(smoothed) - 1):
            if smoothed[i] > smoothed[i - 1] and smoothed[i] > smoothed[i + 1]:
                if smoothed[i] > threshold:
                    peaks.append({
                        'index': i,
                        'value': round(smoothed[i], 4),
                        'position_ratio': round(i / len(smoothed), 4),
                    })
            elif smoothed[i] < smoothed[i - 1] and smoothed[i] < smoothed[i + 1]:
                if smoothed[i] < -threshold:
                    valleys.append({
                        'index': i,
                        'value': round(smoothed[i], 4),
                        'position_ratio': round(i / len(smoothed), 4),
                    })

        return peaks, valleys

    def analyze(self, text: str, granularity: int = 100) -> SentimentResult:
        """Perform comprehensive sentiment analysis on text.

        Args:
            text: Input text to analyze
            granularity: Number of points for sentiment progression

        Returns:
            SentimentResult object with all metrics
        """
        sentences = sent_tokenize(text)

        if not sentences:
            return SentimentResult(
                polarity=0.0, subjectivity=0.0,
                vader_compound=0.0, vader_positive=0.0,
                vader_negative=0.0, vader_neutral=0.0,
                sentence_count=0, positive_sentences=0,
                negative_sentences=0, neutral_sentences=0,
            )

        # Analyze each sentence
        polarities = []
        subjectivities = []
        vader_compounds = []
        vader_positives = []
        vader_negatives = []
        vader_neutrals = []

        positive_count = 0
        negative_count = 0
        neutral_count = 0

        for sentence in sentences:
            # TextBlob analysis
            polarity, subjectivity = self._analyze_sentence_textblob(sentence)
            polarities.append(polarity)
            subjectivities.append(subjectivity)

            # VADER analysis
            vader_scores = self._analyze_sentence_vader(sentence)
            vader_compounds.append(vader_scores['compound'])
            vader_positives.append(vader_scores['pos'])
            vader_negatives.append(vader_scores['neg'])
            vader_neutrals.append(vader_scores['neu'])

            # Classify sentence
            classification = self._classify_sentiment(vader_scores['compound'])
            if classification == 'positive':
                positive_count += 1
            elif classification == 'negative':
                negative_count += 1
            else:
                neutral_count += 1

        # Calculate averages
        avg_polarity = sum(polarities) / len(polarities)
        avg_subjectivity = sum(subjectivities) / len(subjectivities)
        avg_vader_compound = sum(vader_compounds) / len(vader_compounds)
        avg_vader_positive = sum(vader_positives) / len(vader_positives)
        avg_vader_negative = sum(vader_negatives) / len(vader_negatives)
        avg_vader_neutral = sum(vader_neutrals) / len(vader_neutrals)

        # Create sentiment progression (sample to granularity points)
        if len(vader_compounds) > granularity:
            step = len(vader_compounds) / granularity
            progression = [
                vader_compounds[int(i * step)]
                for i in range(granularity)
            ]
        else:
            progression = vader_compounds

        # Find emotional extremes
        peaks, valleys = self._find_emotional_extremes(progression)

        return SentimentResult(
            polarity=avg_polarity,
            subjectivity=avg_subjectivity,
            vader_compound=avg_vader_compound,
            vader_positive=avg_vader_positive,
            vader_negative=avg_vader_negative,
            vader_neutral=avg_vader_neutral,
            sentence_count=len(sentences),
            positive_sentences=positive_count,
            negative_sentences=negative_count,
            neutral_sentences=neutral_count,
            sentiment_progression=progression,
            emotional_peaks=peaks,
            emotional_valleys=valleys,
        )

    def analyze_by_section(
        self,
        text: str,
        num_sections: int = 10,
    ) -> List[Dict[str, Any]]:
        """Analyze sentiment across sections of text.

        Useful for tracking emotional arcs throughout a narrative.

        Args:
            text: Input text
            num_sections: Number of sections to divide text into

        Returns:
            List of sentiment results for each section
        """
        words = text.split()
        section_size = len(words) // num_sections

        sections = []
        for i in range(num_sections):
            start = i * section_size
            end = start + section_size if i < num_sections - 1 else len(words)
            section_text = ' '.join(words[start:end])

            result = self.analyze(section_text, granularity=20)
            sections.append({
                'section': i + 1,
                'position': (i + 0.5) / num_sections,
                'polarity': result.polarity,
                'subjectivity': result.subjectivity,
                'vader_compound': result.vader_compound,
                'sentiment_label': result.get_overall_sentiment(),
            })

        return sections

    def analyze_emotional_arc(self, text: str, num_points: int = 50) -> Dict[str, Any]:
        """Analyze the emotional arc of a narrative.

        Identifies classic narrative patterns like:
        - Rising action
        - Falling action
        - Rags to riches
        - Tragedy
        - etc.

        Args:
            text: Input text
            num_points: Number of data points for the arc

        Returns:
            Dictionary with emotional arc data and pattern analysis
        """
        sections = self.analyze_by_section(text, num_sections=num_points)

        # Extract sentiment values
        arc = [s['vader_compound'] for s in sections]

        # Analyze arc shape
        start_sentiment = sum(arc[:5]) / 5 if len(arc) >= 5 else arc[0]
        end_sentiment = sum(arc[-5:]) / 5 if len(arc) >= 5 else arc[-1]
        middle_sentiment = sum(arc[len(arc)//3:2*len(arc)//3]) / (len(arc)//3) if len(arc) >= 3 else arc[len(arc)//2]

        # Classify arc pattern
        if end_sentiment > start_sentiment + 0.2:
            if middle_sentiment < start_sentiment:
                pattern = "Rags to Riches (fall then rise)"
            else:
                pattern = "Rising"
        elif end_sentiment < start_sentiment - 0.2:
            if middle_sentiment > start_sentiment:
                pattern = "Tragedy (rise then fall)"
            else:
                pattern = "Falling"
        elif abs(middle_sentiment - start_sentiment) > 0.2:
            if middle_sentiment > start_sentiment:
                pattern = "Man in a Hole (rise then return)"
            else:
                pattern = "Icarus (fall then return)"
        else:
            pattern = "Flat/Stable"

        # Calculate volatility (emotional intensity)
        if len(arc) > 1:
            differences = [abs(arc[i] - arc[i-1]) for i in range(1, len(arc))]
            volatility = sum(differences) / len(differences)
        else:
            volatility = 0.0

        return {
            'arc': [round(v, 4) for v in arc],
            'pattern': pattern,
            'start_sentiment': round(start_sentiment, 4),
            'middle_sentiment': round(middle_sentiment, 4),
            'end_sentiment': round(end_sentiment, 4),
            'volatility': round(volatility, 4),
            'overall_range': round(max(arc) - min(arc), 4),
        }

    def extract_emotional_keywords(
        self,
        text: str,
        top_n: int = 20,
    ) -> Dict[str, List[Tuple[str, int]]]:
        """Extract emotionally significant words from text.

        Args:
            text: Input text
            top_n: Number of top keywords per category

        Returns:
            Dictionary with positive and negative keywords
        """
        words = re.findall(r'\b[a-z]+\b', text.lower())
        word_counts = defaultdict(int)

        for word in words:
            word_counts[word] += 1

        # Get VADER lexicon
        positive_words = []
        negative_words = []

        for word, count in word_counts.items():
            if len(word) < 3:
                continue

            scores = self.vader.polarity_scores(word)

            # Check if word has strong sentiment
            if scores['compound'] >= 0.5:
                positive_words.append((word, count, scores['compound']))
            elif scores['compound'] <= -0.5:
                negative_words.append((word, count, scores['compound']))

        # Sort by frequency and sentiment strength
        positive_words.sort(key=lambda x: (-x[2], -x[1]))
        negative_words.sort(key=lambda x: (x[2], -x[1]))

        return {
            'positive': [(w, c) for w, c, _ in positive_words[:top_n]],
            'negative': [(w, c) for w, c, _ in negative_words[:top_n]],
        }
