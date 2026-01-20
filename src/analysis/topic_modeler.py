"""Topic modeling and theme extraction for literary analysis.

Provides LDA and BERTopic-based topic modeling to identify
themes and motifs in literary works.
"""

import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF

# Optional gensim import (not available on Python 3.14+)
try:
    from gensim import corpora
    from gensim.models import LdaMulticore, CoherenceModel
    from gensim.models.phrases import Phrases, Phraser
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False
    corpora = None
    LdaMulticore = None
    CoherenceModel = None
    Phrases = None
    Phraser = None

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Ensure NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


@dataclass
class TopicResult:
    """Container for topic modeling results."""

    num_topics: int
    topics: List[Dict[str, Any]]  # Each topic with words and weights
    document_topics: List[List[Tuple[int, float]]]  # Topic distribution per doc
    coherence_score: float
    model_type: str  # 'lda', 'nmf', or 'bertopic'

    # Theme interpretations
    theme_labels: List[str] = field(default_factory=list)
    dominant_topic_distribution: Dict[int, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'model_type': self.model_type,
            'num_topics': self.num_topics,
            'coherence_score': round(self.coherence_score, 4),
            'topics': self.topics,
            'theme_labels': self.theme_labels,
            'dominant_topic_distribution': self.dominant_topic_distribution,
        }

    def get_top_topic_words(self, topic_idx: int, n: int = 10) -> List[str]:
        """Get top words for a specific topic."""
        if topic_idx < len(self.topics):
            return [w for w, _ in self.topics[topic_idx]['words'][:n]]
        return []


class TopicModeler:
    """Topic modeling for literary theme extraction."""

    # Kafka-specific literary themes for interpretation
    LITERARY_THEMES = {
        'alienation': ['alone', 'isolated', 'stranger', 'apart', 'lonely', 'separate', 'detached'],
        'bureaucracy': ['office', 'official', 'law', 'court', 'procedure', 'authority', 'document'],
        'transformation': ['change', 'become', 'transform', 'metamorphosis', 'turn', 'altered'],
        'guilt': ['guilty', 'shame', 'blame', 'fault', 'sin', 'wrong', 'accused', 'innocent'],
        'family': ['father', 'mother', 'sister', 'brother', 'family', 'parent', 'child', 'son'],
        'anxiety': ['fear', 'afraid', 'worry', 'anxious', 'dread', 'terror', 'panic'],
        'absurdity': ['strange', 'absurd', 'meaningless', 'impossible', 'incomprehensible'],
        'power': ['power', 'control', 'authority', 'command', 'order', 'rule', 'dominate'],
        'identity': ['self', 'identity', 'person', 'name', 'who', 'being', 'exist'],
        'death': ['death', 'dead', 'die', 'dying', 'kill', 'end', 'final'],
        'isolation': ['room', 'door', 'closed', 'locked', 'inside', 'outside', 'confined'],
        'labor': ['work', 'job', 'duty', 'task', 'employee', 'occupation', 'business'],
    }

    def __init__(
        self,
        language: str = 'english',
        min_word_length: int = 3,
        max_df: float = 0.95,
        min_df: int = 2,
    ):
        """Initialize the topic modeler.

        Args:
            language: Language for stopwords
            min_word_length: Minimum word length to include
            max_df: Maximum document frequency (filter common words)
            min_df: Minimum document frequency (filter rare words)
        """
        self.language = language
        self.min_word_length = min_word_length
        self.max_df = max_df
        self.min_df = min_df

        # Extended stopwords for literary analysis
        base_stopwords = set(stopwords.words(language))
        literary_stopwords = {
            'said', 'would', 'could', 'one', 'upon', 'unto', 'shall',
            'thou', 'thee', 'thy', 'ye', 'hath', 'doth', 'tis', 'twas',
            'like', 'even', 'still', 'much', 'also', 'though', 'yet',
            'well', 'just', 'might', 'must', 'may', 'way', 'come',
            'go', 'went', 'came', 'get', 'got', 'see', 'saw', 'know',
            'knew', 'think', 'thought', 'take', 'took', 'make', 'made',
            'say', 'tell', 'told', 'look', 'looked', 'seem', 'seemed',
        }
        self.stopwords = base_stopwords | literary_stopwords

    def preprocess_text(self, text: str) -> List[str]:
        """Preprocess text for topic modeling.

        Args:
            text: Input text

        Returns:
            List of preprocessed tokens
        """
        # Lowercase and tokenize
        text = text.lower()
        tokens = word_tokenize(text)

        # Filter tokens
        tokens = [
            t for t in tokens
            if t.isalpha()
            and len(t) >= self.min_word_length
            and t not in self.stopwords
        ]

        return tokens

    def preprocess_documents(
        self,
        documents: List[str],
        use_phrases: bool = True,
    ) -> Tuple[List[List[str]], Optional[Any]]:
        """Preprocess multiple documents.

        Args:
            documents: List of text documents
            use_phrases: Whether to detect and include bigrams (requires gensim)

        Returns:
            Tuple of (processed documents, phraser model or None)
        """
        processed = [self.preprocess_text(doc) for doc in documents]

        phraser = None
        if use_phrases and GENSIM_AVAILABLE and Phrases is not None:
            # Detect bigrams
            phrases = Phrases(processed, min_count=5, threshold=10)
            phraser = Phraser(phrases)
            processed = [phraser[doc] for doc in processed]

        return processed, phraser

    def train_lda_gensim(
        self,
        documents: List[str],
        num_topics: int = 10,
        passes: int = 15,
        iterations: int = 400,
        random_state: int = 42,
    ) -> TopicResult:
        """Train LDA model using Gensim.

        Args:
            documents: List of text documents
            num_topics: Number of topics to extract
            passes: Number of passes through corpus
            iterations: Max iterations for training
            random_state: Random seed

        Returns:
            TopicResult with topics and metrics

        Raises:
            ImportError: If gensim is not available
        """
        if not GENSIM_AVAILABLE:
            raise ImportError(
                "Gensim is not available. Use train_lda_sklearn() or train_nmf() instead. "
                "Gensim may not be compatible with Python 3.14+."
            )

        # Preprocess documents
        processed_docs, _ = self.preprocess_documents(documents)

        # Create dictionary and corpus
        dictionary = corpora.Dictionary(processed_docs)

        # Filter extremes
        dictionary.filter_extremes(
            no_below=self.min_df,
            no_above=self.max_df,
        )

        corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

        # Train LDA model
        lda_model = LdaMulticore(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            random_state=random_state,
            passes=passes,
            iterations=iterations,
            workers=2,
            per_word_topics=True,
        )

        # Calculate coherence
        coherence_model = CoherenceModel(
            model=lda_model,
            texts=processed_docs,
            dictionary=dictionary,
            coherence='c_v',
        )
        coherence_score = coherence_model.get_coherence()

        # Extract topics
        topics = []
        for topic_idx in range(num_topics):
            topic_words = lda_model.show_topic(topic_idx, topn=20)
            topics.append({
                'id': topic_idx,
                'words': [(word, round(weight, 4)) for word, weight in topic_words],
            })

        # Get document topics
        document_topics = []
        for bow in corpus:
            doc_topics = lda_model.get_document_topics(bow)
            document_topics.append([(t, round(w, 4)) for t, w in doc_topics])

        # Calculate dominant topic distribution
        dominant_topics = [
            max(dt, key=lambda x: x[1])[0] if dt else -1
            for dt in document_topics
        ]
        dominant_distribution = dict(Counter(dominant_topics))

        # Generate theme labels
        theme_labels = self._interpret_topics(topics)

        return TopicResult(
            num_topics=num_topics,
            topics=topics,
            document_topics=document_topics,
            coherence_score=coherence_score,
            model_type='lda_gensim',
            theme_labels=theme_labels,
            dominant_topic_distribution=dominant_distribution,
        )

    def train_lda_sklearn(
        self,
        documents: List[str],
        num_topics: int = 10,
        max_iter: int = 20,
        random_state: int = 42,
    ) -> TopicResult:
        """Train LDA model using scikit-learn.

        Args:
            documents: List of text documents
            num_topics: Number of topics
            max_iter: Maximum iterations
            random_state: Random seed

        Returns:
            TopicResult with topics and metrics
        """
        # Adjust min_df for small document sets
        effective_min_df = min(self.min_df, max(1, len(documents) // 5))

        # Create vectorizer - try with stopwords, fall back without if needed
        try:
            vectorizer = CountVectorizer(
                max_df=self.max_df,
                min_df=effective_min_df,
                stop_words=list(self.stopwords),
                max_features=5000,
            )
            doc_term_matrix = vectorizer.fit_transform(documents)
        except ValueError:
            # Retry with less aggressive filtering
            vectorizer = CountVectorizer(
                max_df=0.99,
                min_df=1,
                stop_words='english',  # Use built-in stopwords
                max_features=5000,
            )
            doc_term_matrix = vectorizer.fit_transform(documents)

        feature_names = vectorizer.get_feature_names_out()

        # Train LDA
        lda = LatentDirichletAllocation(
            n_components=num_topics,
            max_iter=max_iter,
            random_state=random_state,
            learning_method='online',
            n_jobs=-1,
        )
        doc_topics = lda.fit_transform(doc_term_matrix)

        # Extract topics
        topics = []
        for topic_idx, topic in enumerate(lda.components_):
            top_indices = topic.argsort()[:-21:-1]
            top_words = [(feature_names[i], round(topic[i] / topic.sum(), 4))
                         for i in top_indices]
            topics.append({
                'id': topic_idx,
                'words': top_words,
            })

        # Document topics
        document_topics = []
        for doc_topic_dist in doc_topics:
            sorted_topics = sorted(enumerate(doc_topic_dist), key=lambda x: -x[1])
            document_topics.append([(t, round(w, 4)) for t, w in sorted_topics if w > 0.01])

        # Dominant topics
        dominant_topics = [max(enumerate(dt), key=lambda x: x[1])[0] for dt in doc_topics]
        dominant_distribution = dict(Counter(dominant_topics))

        # Coherence (approximate using perplexity)
        perplexity = lda.perplexity(doc_term_matrix)
        coherence_approx = -perplexity / 1000  # Normalized inverse perplexity

        # Theme labels
        theme_labels = self._interpret_topics(topics)

        return TopicResult(
            num_topics=num_topics,
            topics=topics,
            document_topics=document_topics,
            coherence_score=coherence_approx,
            model_type='lda_sklearn',
            theme_labels=theme_labels,
            dominant_topic_distribution=dominant_distribution,
        )

    def train_nmf(
        self,
        documents: List[str],
        num_topics: int = 10,
        max_iter: int = 200,
        random_state: int = 42,
    ) -> TopicResult:
        """Train NMF (Non-negative Matrix Factorization) model.

        NMF often produces more interpretable topics than LDA.

        Args:
            documents: List of text documents
            num_topics: Number of topics
            max_iter: Maximum iterations
            random_state: Random seed

        Returns:
            TopicResult with topics and metrics
        """
        # Adjust min_df for small document sets
        effective_min_df = min(self.min_df, max(1, len(documents) // 5))

        # Create TF-IDF vectorizer - try with stopwords, fall back without if needed
        try:
            vectorizer = TfidfVectorizer(
                max_df=self.max_df,
                min_df=effective_min_df,
                stop_words=list(self.stopwords),
                max_features=5000,
            )
            tfidf_matrix = vectorizer.fit_transform(documents)
        except ValueError:
            # Retry with less aggressive filtering
            vectorizer = TfidfVectorizer(
                max_df=0.99,
                min_df=1,
                stop_words='english',
                max_features=5000,
            )
            tfidf_matrix = vectorizer.fit_transform(documents)

        feature_names = vectorizer.get_feature_names_out()

        # Train NMF
        nmf = NMF(
            n_components=num_topics,
            max_iter=max_iter,
            random_state=random_state,
            init='nndsvd',
        )
        doc_topics = nmf.fit_transform(tfidf_matrix)

        # Extract topics
        topics = []
        for topic_idx, topic in enumerate(nmf.components_):
            top_indices = topic.argsort()[:-21:-1]
            top_words = [(feature_names[i], round(topic[i], 4))
                         for i in top_indices]
            topics.append({
                'id': topic_idx,
                'words': top_words,
            })

        # Normalize document topics
        doc_topics_normalized = doc_topics / (doc_topics.sum(axis=1, keepdims=True) + 1e-10)

        document_topics = []
        for doc_topic_dist in doc_topics_normalized:
            sorted_topics = sorted(enumerate(doc_topic_dist), key=lambda x: -x[1])
            document_topics.append([(t, round(w, 4)) for t, w in sorted_topics if w > 0.01])

        # Dominant topics
        dominant_topics = [max(enumerate(dt), key=lambda x: x[1])[0]
                          for dt in doc_topics_normalized]
        dominant_distribution = dict(Counter(dominant_topics))

        # Reconstruction error as quality metric
        reconstruction_err = nmf.reconstruction_err_
        coherence_approx = 1.0 / (1 + reconstruction_err / 1000)

        # Theme labels
        theme_labels = self._interpret_topics(topics)

        return TopicResult(
            num_topics=num_topics,
            topics=topics,
            document_topics=document_topics,
            coherence_score=coherence_approx,
            model_type='nmf',
            theme_labels=theme_labels,
            dominant_topic_distribution=dominant_distribution,
        )

    def _interpret_topics(self, topics: List[Dict]) -> List[str]:
        """Attempt to label topics based on literary themes.

        Args:
            topics: List of topic dictionaries

        Returns:
            List of theme labels
        """
        labels = []

        for topic in topics:
            topic_words = set(w for w, _ in topic['words'][:10])

            best_theme = 'Unknown'
            best_overlap = 0

            for theme, theme_words in self.LITERARY_THEMES.items():
                overlap = len(topic_words & set(theme_words))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_theme = theme.title()

            # If no match, use top words
            if best_overlap < 2:
                top_words = [w for w, _ in topic['words'][:3]]
                best_theme = '/'.join(top_words)

            labels.append(best_theme)

        return labels

    def find_optimal_topics(
        self,
        documents: List[str],
        min_topics: int = 3,
        max_topics: int = 15,
        step: int = 2,
    ) -> Dict[str, Any]:
        """Find optimal number of topics using coherence (sklearn) or perplexity.

        Args:
            documents: List of text documents
            min_topics: Minimum topics to try
            max_topics: Maximum topics to try
            step: Step size for topic count

        Returns:
            Dictionary with optimal topic count and coherence/perplexity scores
        """
        if GENSIM_AVAILABLE:
            # Use gensim coherence
            processed_docs, _ = self.preprocess_documents(documents)
            dictionary = corpora.Dictionary(processed_docs)
            dictionary.filter_extremes(no_below=self.min_df, no_above=self.max_df)
            corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

            coherence_scores = []
            topic_counts = list(range(min_topics, max_topics + 1, step))

            for num_topics in topic_counts:
                lda_model = LdaMulticore(
                    corpus=corpus,
                    id2word=dictionary,
                    num_topics=num_topics,
                    random_state=42,
                    passes=10,
                    workers=2,
                )

                coherence_model = CoherenceModel(
                    model=lda_model,
                    texts=processed_docs,
                    dictionary=dictionary,
                    coherence='c_v',
                )
                coherence_scores.append(coherence_model.get_coherence())

            # Find optimal
            optimal_idx = coherence_scores.index(max(coherence_scores))
            optimal_topics = topic_counts[optimal_idx]

            return {
                'optimal_topics': optimal_topics,
                'topic_counts': topic_counts,
                'coherence_scores': [round(c, 4) for c in coherence_scores],
                'max_coherence': round(max(coherence_scores), 4),
            }
        else:
            # Use sklearn perplexity (lower is better)
            vectorizer = CountVectorizer(
                max_df=self.max_df,
                min_df=self.min_df,
                stop_words=list(self.stopwords),
                max_features=5000,
            )
            doc_term_matrix = vectorizer.fit_transform(documents)

            perplexity_scores = []
            topic_counts = list(range(min_topics, max_topics + 1, step))

            for num_topics in topic_counts:
                lda = LatentDirichletAllocation(
                    n_components=num_topics,
                    max_iter=20,
                    random_state=42,
                    learning_method='online',
                )
                lda.fit(doc_term_matrix)
                perplexity_scores.append(lda.perplexity(doc_term_matrix))

            # Find optimal (lowest perplexity)
            optimal_idx = perplexity_scores.index(min(perplexity_scores))
            optimal_topics = topic_counts[optimal_idx]

            return {
                'optimal_topics': optimal_topics,
                'topic_counts': topic_counts,
                'perplexity_scores': [round(p, 4) for p in perplexity_scores],
                'min_perplexity': round(min(perplexity_scores), 4),
            }

    def extract_themes_from_text(
        self,
        text: str,
        chunk_size: int = 2000,
        num_topics: int = 5,
        use_nmf: bool = False,
    ) -> TopicResult:
        """Extract themes from a single text by chunking.

        Args:
            text: Input text
            chunk_size: Size of text chunks (in words)
            num_topics: Number of topics to extract
            use_nmf: Use NMF instead of LDA (often more interpretable)

        Returns:
            TopicResult with identified themes
        """
        # Split text into chunks
        words = text.split()
        chunks = []

        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i + chunk_size])
            if len(chunk.split()) >= 100:  # Minimum chunk size
                chunks.append(chunk)

        if len(chunks) < 5:
            # If text is too short, use sentence-based chunks
            from nltk.tokenize import sent_tokenize
            sentences = sent_tokenize(text)
            chunk_size = max(10, len(sentences) // 20)
            chunks = [
                ' '.join(sentences[i:i + chunk_size])
                for i in range(0, len(sentences), chunk_size)
            ]

        # Use appropriate method based on availability
        if use_nmf:
            return self.train_nmf(chunks, num_topics=num_topics)
        elif GENSIM_AVAILABLE:
            return self.train_lda_gensim(chunks, num_topics=num_topics)
        else:
            return self.train_lda_sklearn(chunks, num_topics=num_topics)
