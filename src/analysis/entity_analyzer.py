"""Named Entity Recognition and Character Network Analysis.

Provides NER for identifying characters, locations, and other entities,
and builds relationship networks between characters in literary texts.
"""

import re
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from itertools import combinations

import networkx as nx
from nltk.tokenize import sent_tokenize


@dataclass
class Entity:
    """Represents a named entity."""
    text: str
    label: str
    count: int
    mentions: List[int] = field(default_factory=list)  # Sentence indices

    def to_dict(self) -> Dict[str, Any]:
        return {
            'text': self.text,
            'label': self.label,
            'count': self.count,
            'mentions': self.mentions[:10],  # Limit for storage
        }


@dataclass
class CharacterRelationship:
    """Represents a relationship between two characters."""
    character1: str
    character2: str
    weight: int  # Co-occurrence count
    contexts: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'character1': self.character1,
            'character2': self.character2,
            'weight': self.weight,
            'sample_contexts': self.contexts[:3],  # Limit samples
        }


@dataclass
class EntityAnalysisResult:
    """Container for entity analysis results."""

    entities: Dict[str, List[Entity]]  # Grouped by label
    total_entities: int
    unique_entities: int

    # Character-specific
    characters: List[Entity]
    locations: List[Entity]
    organizations: List[Entity]

    # Character network
    character_network: Optional[Dict[str, Any]] = None
    relationships: List[CharacterRelationship] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'summary': {
                'total_entities': self.total_entities,
                'unique_entities': self.unique_entities,
                'character_count': len(self.characters),
                'location_count': len(self.locations),
                'organization_count': len(self.organizations),
            },
            'characters': [c.to_dict() for c in self.characters[:50]],
            'locations': [l.to_dict() for l in self.locations[:30]],
            'organizations': [o.to_dict() for o in self.organizations[:20]],
            'character_network': self.character_network,
            'relationships': [r.to_dict() for r in self.relationships[:50]],
        }


class EntityAnalyzer:
    """Named Entity Recognition and Character Network Analysis."""

    # Common non-character names to filter
    FILTER_PATTERNS = [
        r'^(Mr|Mrs|Ms|Miss|Dr|Sir|Lord|Lady)\.?$',
        r'^(Chapter|Part|Book|Section|Volume)\s*\d*$',
        r'^[A-Z]$',  # Single letters
        r'^\d+$',  # Numbers
    ]

    # Entity label mappings for different NER models
    PERSON_LABELS = {'PERSON', 'PER', 'PERS'}
    LOCATION_LABELS = {'LOC', 'GPE', 'LOCATION', 'FAC'}
    ORG_LABELS = {'ORG', 'ORGANIZATION'}

    def __init__(self, use_spacy: bool = True, spacy_model: str = 'en_core_web_sm'):
        """Initialize the entity analyzer.

        Args:
            use_spacy: Whether to use spaCy for NER
            spacy_model: spaCy model to use
        """
        self.use_spacy = use_spacy
        self.nlp = None

        if use_spacy:
            self._load_spacy(spacy_model)

    def _load_spacy(self, model: str):
        """Load spaCy model."""
        import spacy

        try:
            self.nlp = spacy.load(model)
        except OSError:
            import subprocess
            subprocess.run(['python', '-m', 'spacy', 'download', model], check=True)
            self.nlp = spacy.load(model)

        # Increase max length for literary texts
        self.nlp.max_length = 2_000_000

    def _should_filter_entity(self, text: str) -> bool:
        """Check if entity should be filtered out.

        Args:
            text: Entity text

        Returns:
            True if entity should be filtered
        """
        for pattern in self.FILTER_PATTERNS:
            if re.match(pattern, text, re.IGNORECASE):
                return True
        return False

    def _normalize_entity(self, text: str) -> str:
        """Normalize entity text.

        Args:
            text: Entity text

        Returns:
            Normalized text
        """
        # Remove extra whitespace
        text = ' '.join(text.split())

        # Remove possessives
        text = re.sub(r"'s$", '', text)

        # Remove leading/trailing punctuation
        text = text.strip('.,;:!?"\'')

        return text

    def _merge_similar_entities(
        self,
        entities: Dict[str, Entity],
        threshold: float = 0.8,
    ) -> Dict[str, Entity]:
        """Merge entities that likely refer to the same person.

        Args:
            entities: Dictionary of entities
            threshold: Similarity threshold

        Returns:
            Merged entities dictionary
        """
        from rapidfuzz import fuzz

        # Sort by count (merge smaller into larger)
        sorted_entities = sorted(
            entities.items(),
            key=lambda x: x[1].count,
            reverse=True,
        )

        merged = {}
        used = set()

        for name, entity in sorted_entities:
            if name in used:
                continue

            # Find similar entities
            for other_name, other_entity in sorted_entities:
                if other_name == name or other_name in used:
                    continue

                # Check if other is substring or very similar
                is_substring = (
                    other_name.lower() in name.lower() or
                    name.lower() in other_name.lower()
                )
                similarity = fuzz.ratio(name.lower(), other_name.lower()) / 100

                if is_substring or similarity >= threshold:
                    # Merge into the longer/more common name
                    entity.count += other_entity.count
                    entity.mentions.extend(other_entity.mentions)
                    used.add(other_name)

            merged[name] = entity
            used.add(name)

        return merged

    def extract_entities_spacy(self, text: str) -> Dict[str, List[Entity]]:
        """Extract entities using spaCy.

        Args:
            text: Input text

        Returns:
            Dictionary of entities grouped by label
        """
        # Process text in chunks if too long
        max_chunk = 100000
        all_entities = defaultdict(lambda: defaultdict(lambda: {'count': 0, 'mentions': []}))

        sentences = sent_tokenize(text)
        sentence_map = {}  # Map character position to sentence index
        pos = 0
        for idx, sent in enumerate(sentences):
            for _ in sent:
                sentence_map[pos] = idx
                pos += 1
            pos += 1  # For space between sentences

        # Process in chunks
        for i in range(0, len(text), max_chunk):
            chunk = text[i:min(i + max_chunk, len(text))]
            doc = self.nlp(chunk)

            for ent in doc.ents:
                normalized = self._normalize_entity(ent.text)
                if not normalized or self._should_filter_entity(normalized):
                    continue

                label = ent.label_
                sent_idx = sentence_map.get(i + ent.start_char, 0)

                all_entities[label][normalized]['count'] += 1
                if len(all_entities[label][normalized]['mentions']) < 100:
                    all_entities[label][normalized]['mentions'].append(sent_idx)

        # Convert to Entity objects
        result = {}
        for label, entities in all_entities.items():
            result[label] = [
                Entity(
                    text=name,
                    label=label,
                    count=data['count'],
                    mentions=data['mentions'],
                )
                for name, data in entities.items()
            ]
            # Sort by count
            result[label].sort(key=lambda x: x.count, reverse=True)

        return result

    def extract_entities_regex(self, text: str) -> Dict[str, List[Entity]]:
        """Extract entities using regex patterns (fallback method).

        Args:
            text: Input text

        Returns:
            Dictionary of entities grouped by label
        """
        sentences = sent_tokenize(text)

        # Pattern for capitalized names
        name_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'

        entities = defaultdict(lambda: {'count': 0, 'mentions': []})

        for sent_idx, sentence in enumerate(sentences):
            matches = re.findall(name_pattern, sentence)
            for match in matches:
                normalized = self._normalize_entity(match)
                if not normalized or self._should_filter_entity(normalized):
                    continue
                if len(normalized) < 2:
                    continue

                entities[normalized]['count'] += 1
                if len(entities[normalized]['mentions']) < 100:
                    entities[normalized]['mentions'].append(sent_idx)

        # Convert to Entity objects (assume all are persons)
        result = {
            'PERSON': [
                Entity(
                    text=name,
                    label='PERSON',
                    count=data['count'],
                    mentions=data['mentions'],
                )
                for name, data in entities.items()
                if data['count'] >= 2  # Minimum occurrence
            ]
        }
        result['PERSON'].sort(key=lambda x: x.count, reverse=True)

        return result

    def build_character_network(
        self,
        text: str,
        characters: List[Entity],
        window_size: int = 3,
        min_weight: int = 2,
    ) -> Tuple[nx.Graph, List[CharacterRelationship]]:
        """Build a character co-occurrence network.

        Args:
            text: Input text
            characters: List of character entities
            window_size: Number of sentences for co-occurrence window
            min_weight: Minimum co-occurrences to include edge

        Returns:
            Tuple of (NetworkX graph, list of relationships)
        """
        sentences = sent_tokenize(text)

        # Create character mention lookup
        char_names = {c.text.lower(): c.text for c in characters[:50]}

        # Find character mentions in each sentence
        sentence_chars = []
        for sent in sentences:
            sent_lower = sent.lower()
            chars_in_sent = set()
            for char_lower, char_original in char_names.items():
                if char_lower in sent_lower:
                    chars_in_sent.add(char_original)
            sentence_chars.append(chars_in_sent)

        # Calculate co-occurrences within window
        co_occurrences = defaultdict(lambda: {'count': 0, 'contexts': []})

        for i, chars in enumerate(sentence_chars):
            # Get characters in window
            window_chars = set(chars)
            for j in range(max(0, i - window_size), min(len(sentences), i + window_size + 1)):
                if j != i:
                    window_chars.update(sentence_chars[j])

            # Count co-occurrences for this sentence's characters
            for char1 in chars:
                for char2 in window_chars:
                    if char1 < char2:  # Avoid duplicates
                        key = (char1, char2)
                        co_occurrences[key]['count'] += 1
                        if len(co_occurrences[key]['contexts']) < 5:
                            co_occurrences[key]['contexts'].append(sentences[i][:200])

        # Build graph
        G = nx.Graph()

        # Add character nodes
        for char in characters[:50]:
            G.add_node(char.text, count=char.count)

        # Add edges
        relationships = []
        for (char1, char2), data in co_occurrences.items():
            if data['count'] >= min_weight:
                G.add_edge(char1, char2, weight=data['count'])
                relationships.append(CharacterRelationship(
                    character1=char1,
                    character2=char2,
                    weight=data['count'],
                    contexts=data['contexts'],
                ))

        relationships.sort(key=lambda x: x.weight, reverse=True)

        return G, relationships

    def analyze_network(self, G: nx.Graph) -> Dict[str, Any]:
        """Analyze character network properties.

        Args:
            G: NetworkX graph

        Returns:
            Dictionary with network metrics
        """
        if G.number_of_nodes() == 0:
            return {
                'nodes': 0,
                'edges': 0,
                'density': 0,
                'central_characters': [],
                'communities': [],
            }

        # Basic metrics
        metrics = {
            'nodes': G.number_of_nodes(),
            'edges': G.number_of_edges(),
            'density': round(nx.density(G), 4),
        }

        # Centrality measures
        if G.number_of_nodes() > 1:
            degree_centrality = nx.degree_centrality(G)
            betweenness = nx.betweenness_centrality(G)

            # Top central characters
            central_chars = sorted(
                degree_centrality.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:10]

            metrics['central_characters'] = [
                {
                    'name': name,
                    'degree_centrality': round(dc, 4),
                    'betweenness': round(betweenness.get(name, 0), 4),
                }
                for name, dc in central_chars
            ]

            # Community detection
            try:
                communities = list(nx.community.greedy_modularity_communities(G))
                metrics['communities'] = [
                    {
                        'id': i,
                        'members': list(comm)[:10],
                        'size': len(comm),
                    }
                    for i, comm in enumerate(communities[:5])
                ]
                metrics['modularity'] = round(
                    nx.community.modularity(G, communities), 4
                )
            except Exception:
                metrics['communities'] = []
                metrics['modularity'] = 0
        else:
            metrics['central_characters'] = []
            metrics['communities'] = []

        return metrics

    def analyze(self, text: str, build_network: bool = True) -> EntityAnalysisResult:
        """Perform comprehensive entity analysis.

        Args:
            text: Input text
            build_network: Whether to build character network

        Returns:
            EntityAnalysisResult with all analysis data
        """
        # Extract entities
        if self.use_spacy and self.nlp:
            entities = self.extract_entities_spacy(text)
        else:
            entities = self.extract_entities_regex(text)

        # Separate by type
        characters = []
        locations = []
        organizations = []

        for label, entity_list in entities.items():
            if label in self.PERSON_LABELS:
                characters.extend(entity_list)
            elif label in self.LOCATION_LABELS:
                locations.extend(entity_list)
            elif label in self.ORG_LABELS:
                organizations.extend(entity_list)

        # Merge similar character names
        if characters:
            char_dict = {c.text: c for c in characters}
            char_dict = self._merge_similar_entities(char_dict)
            characters = sorted(char_dict.values(), key=lambda x: x.count, reverse=True)

        # Calculate totals
        total_entities = sum(e.count for elist in entities.values() for e in elist)
        unique_entities = sum(len(elist) for elist in entities.values())

        # Build character network
        network_data = None
        relationships = []

        if build_network and characters:
            G, relationships = self.build_character_network(text, characters)
            network_data = self.analyze_network(G)

        return EntityAnalysisResult(
            entities=entities,
            total_entities=total_entities,
            unique_entities=unique_entities,
            characters=characters,
            locations=locations,
            organizations=organizations,
            character_network=network_data,
            relationships=relationships,
        )

    def export_network_gexf(self, text: str, output_path: str) -> str:
        """Export character network to GEXF format for visualization.

        Args:
            text: Input text
            output_path: Path to save GEXF file

        Returns:
            Path to saved file
        """
        result = self.analyze(text)

        if result.characters:
            G, _ = self.build_character_network(text, result.characters)
            nx.write_gexf(G, output_path)
            return output_path

        return ""
