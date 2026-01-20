"""Text preprocessing pipeline for literary analysis.

Provides comprehensive text cleaning, tokenization, and normalization
for NLP analysis of literary works.
"""

import re
import unicodedata
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from functools import lru_cache

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tag import pos_tag

# Download required NLTK data
def ensure_nltk_data():
    """Download required NLTK data if not present."""
    resources = [
        'punkt',
        'punkt_tab',
        'stopwords',
        'wordnet',
        'averaged_perceptron_tagger',
        'averaged_perceptron_tagger_eng',
    ]
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}' if 'punkt' in resource else f'corpora/{resource}' if resource in ['stopwords', 'wordnet'] else f'taggers/{resource}')
        except LookupError:
            nltk.download(resource, quiet=True)


@dataclass
class ProcessedText:
    """Container for processed text data."""
    original_text: str
    cleaned_text: str
    sentences: List[str]
    tokens: List[str]
    lemmas: List[str]
    pos_tags: List[Tuple[str, str]]
    word_count: int
    sentence_count: int
    unique_words: int

    # Optional filtered versions
    tokens_no_stopwords: List[str] = field(default_factory=list)
    tokens_alpha_only: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'cleaned_text': self.cleaned_text,
            'sentence_count': self.sentence_count,
            'word_count': self.word_count,
            'unique_words': self.unique_words,
            'token_count': len(self.tokens),
            'lemma_count': len(self.lemmas),
            'tokens_no_stopwords_count': len(self.tokens_no_stopwords),
        }


class TextProcessor:
    """Comprehensive text preprocessing pipeline for literary analysis."""

    def __init__(
        self,
        language: str = 'english',
        remove_stopwords: bool = True,
        lemmatize: bool = True,
        lowercase: bool = True,
        min_word_length: int = 2,
    ):
        """Initialize the text processor.

        Args:
            language: Language for stopwords and processing
            remove_stopwords: Whether to filter stopwords
            lemmatize: Whether to lemmatize tokens
            lowercase: Whether to convert to lowercase
            min_word_length: Minimum word length to keep
        """
        ensure_nltk_data()

        self.language = language
        self.remove_stopwords = remove_stopwords
        self.lemmatize_tokens = lemmatize
        self.lowercase = lowercase
        self.min_word_length = min_word_length

        # Initialize tools
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()

        # Load stopwords
        try:
            self._stopwords = set(stopwords.words(language))
        except OSError:
            self._stopwords = set(stopwords.words('english'))

        # Add custom literary stopwords
        self._stopwords.update({
            'said', 'would', 'could', 'one', 'upon', 'unto',
            'shall', 'thou', 'thee', 'thy', 'ye', 'hath', 'doth',
            'tis', 'twas', 'ere', 'hence', 'thence', 'hither',
            'thither', 'whither', 'wherefore', 'thereof',
        })

    def clean_gutenberg_text(self, text: str) -> str:
        """Remove Project Gutenberg headers and footers.

        Args:
            text: Raw text from Gutenberg

        Returns:
            Cleaned text with headers/footers removed
        """
        # Common Gutenberg header patterns
        header_patterns = [
            r'\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG.*?\*\*\*',
            r'Project Gutenberg[\'s]* .*?(?=\n\n)',
            r'Produced by .*?(?=\n\n)',
        ]

        # Common Gutenberg footer patterns
        footer_patterns = [
            r'\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG.*',
            r'End of (?:the )?Project Gutenberg.*',
            r'End of this Project Gutenberg.*',
        ]

        # Remove headers
        for pattern in header_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                text = text[match.end():]
                break

        # Remove footers
        for pattern in footer_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                text = text[:match.start()]
                break

        return text.strip()

    def clean_text(self, text: str, preserve_paragraphs: bool = False) -> str:
        """Clean and normalize text.

        Args:
            text: Input text
            preserve_paragraphs: Whether to keep paragraph structure

        Returns:
            Cleaned text
        """
        # Normalize unicode characters
        text = unicodedata.normalize('NFKD', text)

        # Remove control characters except newlines and tabs
        text = ''.join(
            char for char in text
            if not unicodedata.category(char).startswith('C') or char in '\n\t'
        )

        # Normalize whitespace within lines
        text = re.sub(r'[ \t]+', ' ', text)

        # Handle paragraph preservation
        if preserve_paragraphs:
            # Normalize paragraph breaks (2+ newlines -> double newline)
            text = re.sub(r'\n{2,}', '\n\n', text)
        else:
            # Replace all newlines with spaces
            text = re.sub(r'\n+', ' ', text)

        # Remove extra spaces
        text = re.sub(r' +', ' ', text)

        # Strip leading/trailing whitespace
        text = text.strip()

        return text

    def tokenize_sentences(self, text: str) -> List[str]:
        """Split text into sentences.

        Args:
            text: Input text

        Returns:
            List of sentences
        """
        sentences = sent_tokenize(text)
        # Clean up each sentence
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences

    def tokenize_words(self, text: str) -> List[str]:
        """Split text into word tokens.

        Args:
            text: Input text

        Returns:
            List of word tokens
        """
        tokens = word_tokenize(text)

        if self.lowercase:
            tokens = [t.lower() for t in tokens]

        return tokens

    def filter_tokens(
        self,
        tokens: List[str],
        alpha_only: bool = False,
        remove_stopwords: bool = None,
    ) -> List[str]:
        """Filter tokens based on criteria.

        Args:
            tokens: Input tokens
            alpha_only: Keep only alphabetic tokens
            remove_stopwords: Remove stopwords (uses instance default if None)

        Returns:
            Filtered tokens
        """
        if remove_stopwords is None:
            remove_stopwords = self.remove_stopwords

        filtered = tokens

        # Filter by alpha only
        if alpha_only:
            filtered = [t for t in filtered if t.isalpha()]

        # Filter by minimum length
        filtered = [t for t in filtered if len(t) >= self.min_word_length]

        # Remove stopwords
        if remove_stopwords:
            filtered = [t for t in filtered if t.lower() not in self._stopwords]

        return filtered

    @lru_cache(maxsize=10000)
    def _get_wordnet_pos(self, treebank_tag: str) -> str:
        """Convert treebank POS tag to WordNet POS tag."""
        from nltk.corpus import wordnet

        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN  # Default to noun

    def lemmatize(self, tokens: List[str], pos_tags: List[Tuple[str, str]] = None) -> List[str]:
        """Lemmatize tokens using POS-aware lemmatization.

        Args:
            tokens: Input tokens
            pos_tags: Optional POS tags for better lemmatization

        Returns:
            Lemmatized tokens
        """
        if pos_tags is None:
            pos_tags = pos_tag(tokens)

        lemmas = []
        for token, (_, tag) in zip(tokens, pos_tags):
            wordnet_pos = self._get_wordnet_pos(tag)
            lemma = self.lemmatizer.lemmatize(token.lower(), pos=wordnet_pos)
            lemmas.append(lemma)

        return lemmas

    def stem(self, tokens: List[str]) -> List[str]:
        """Apply Porter stemming to tokens.

        Args:
            tokens: Input tokens

        Returns:
            Stemmed tokens
        """
        return [self.stemmer.stem(t) for t in tokens]

    def get_pos_tags(self, tokens: List[str]) -> List[Tuple[str, str]]:
        """Get part-of-speech tags for tokens.

        Args:
            tokens: Input tokens

        Returns:
            List of (token, POS) tuples
        """
        return pos_tag(tokens)

    def process(self, text: str, clean_gutenberg: bool = False) -> ProcessedText:
        """Run full preprocessing pipeline on text.

        Args:
            text: Input text
            clean_gutenberg: Whether to remove Gutenberg headers/footers

        Returns:
            ProcessedText object with all processed data
        """
        # Clean Gutenberg text if requested
        if clean_gutenberg:
            text = self.clean_gutenberg_text(text)

        # Clean and normalize
        cleaned_text = self.clean_text(text)

        # Sentence tokenization
        sentences = self.tokenize_sentences(cleaned_text)

        # Word tokenization
        tokens = self.tokenize_words(cleaned_text)

        # POS tagging
        pos_tags = self.get_pos_tags(tokens)

        # Lemmatization
        if self.lemmatize_tokens:
            lemmas = self.lemmatize(tokens, pos_tags)
        else:
            lemmas = tokens.copy()

        # Filtered versions
        tokens_no_stopwords = self.filter_tokens(tokens, remove_stopwords=True)
        tokens_alpha_only = self.filter_tokens(tokens, alpha_only=True, remove_stopwords=True)

        # Calculate stats
        word_count = len([t for t in tokens if t.isalpha()])
        unique_words = len(set(t.lower() for t in tokens if t.isalpha()))

        return ProcessedText(
            original_text=text,
            cleaned_text=cleaned_text,
            sentences=sentences,
            tokens=tokens,
            lemmas=lemmas,
            pos_tags=pos_tags,
            word_count=word_count,
            sentence_count=len(sentences),
            unique_words=unique_words,
            tokens_no_stopwords=tokens_no_stopwords,
            tokens_alpha_only=tokens_alpha_only,
        )

    def process_in_chunks(
        self,
        text: str,
        chunk_size: int = 10000,
        overlap: int = 100,
    ) -> List[ProcessedText]:
        """Process large text in overlapping chunks.

        Args:
            text: Input text
            chunk_size: Size of each chunk in characters
            overlap: Overlap between chunks

        Returns:
            List of ProcessedText objects for each chunk
        """
        chunks = []
        start = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))

            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence end
                for punct in ['. ', '! ', '? ', '\n']:
                    last_punct = text.rfind(punct, start + chunk_size // 2, end)
                    if last_punct != -1:
                        end = last_punct + 1
                        break

            chunk = text[start:end]
            processed = self.process(chunk)
            chunks.append(processed)

            start = end - overlap if end < len(text) else end

        return chunks


class SpacyTextProcessor:
    """Text processor using spaCy for advanced NLP features."""

    def __init__(self, model: str = 'en_core_web_sm'):
        """Initialize spaCy processor.

        Args:
            model: spaCy model to load
        """
        import spacy

        try:
            self.nlp = spacy.load(model)
        except OSError:
            # Download model if not present
            import subprocess
            subprocess.run(['python', '-m', 'spacy', 'download', model], check=True)
            self.nlp = spacy.load(model)

        # Increase max length for literary texts
        self.nlp.max_length = 2_000_000

    def process(self, text: str, disable: List[str] = None) -> 'spacy.tokens.Doc':
        """Process text with spaCy.

        Args:
            text: Input text
            disable: Pipeline components to disable for speed

        Returns:
            spaCy Doc object
        """
        if disable:
            return self.nlp(text, disable=disable)
        return self.nlp(text)

    def process_large_text(self, text: str, batch_size: int = 100000) -> List['spacy.tokens.Doc']:
        """Process large text in batches.

        Args:
            text: Input text
            batch_size: Characters per batch

        Returns:
            List of spaCy Doc objects
        """
        docs = []
        for i in range(0, len(text), batch_size):
            chunk = text[i:i + batch_size]
            doc = self.nlp(chunk)
            docs.append(doc)
        return docs

    def get_entities(self, doc: 'spacy.tokens.Doc') -> List[Dict[str, Any]]:
        """Extract named entities from document.

        Args:
            doc: spaCy Doc object

        Returns:
            List of entity dictionaries
        """
        entities = []
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char,
            })
        return entities

    def get_noun_chunks(self, doc: 'spacy.tokens.Doc') -> List[str]:
        """Extract noun phrases from document.

        Args:
            doc: spaCy Doc object

        Returns:
            List of noun phrase strings
        """
        return [chunk.text for chunk in doc.noun_chunks]
