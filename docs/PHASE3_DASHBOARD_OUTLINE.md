# Phase 3: Interactive Dashboard - Implementation Outline

## Overview

Build a Streamlit dashboard to visualize the NLP analysis results from Phase 2, focusing on Franz Kafka's English works (Metamorphosis and The Trial).

## Technology Stack

- **Framework**: Streamlit (rapid prototyping, Python-native)
- **Visualization**: Plotly (interactive charts), NetworkX + PyVis (character networks)
- **Data**: SQLite database with analysis results from Phase 2

## Dashboard Structure

### 1. Home / Overview Page

**Purpose**: High-level summary of the Kafka corpus

**Components**:
- Welcome header with project description
- Key statistics cards:
  - Total works analyzed: 2
  - Combined word count: ~105,000
  - Analysis coverage: 5 types per work
- Quick navigation to detailed views
- Sample insights preview

### 2. Work Selector Sidebar

**Purpose**: Navigate between different works

**Components**:
- Dropdown: Select Kafka work (Metamorphosis, The Trial)
- Filter options:
  - Analysis type focus
  - Comparison mode toggle
- About section with data source info

### 3. Sentiment Analysis View

**Purpose**: Visualize emotional patterns throughout each work

**Components**:

#### 3.1 Overall Sentiment Summary
- Polarity gauge (-1 to +1)
- Subjectivity gauge (0 to 1)
- VADER compound score
- Sentiment classification badge (Positive/Negative/Neutral)

#### 3.2 Emotional Arc Chart
- **X-axis**: Narrative progression (0-100%)
- **Y-axis**: Sentiment score
- Interactive line chart with hover details
- Annotation of key emotional peaks/valleys
- Overlay of narrative pattern classification (Rising, Falling, Tragedy, etc.)

#### 3.3 Sentence-Level Distribution
- Pie chart: Positive/Negative/Neutral sentence counts
- Histogram of sentiment score distribution

#### 3.4 Emotional Keywords
- Word cloud of positive/negative terms
- Top 10 emotional words table with frequencies

### 4. Complexity Metrics View

**Purpose**: Display readability and linguistic complexity

**Components**:

#### 4.1 Readability Scores Dashboard
- Gauge charts for each metric:
  - Flesch Reading Ease (0-100, higher = easier)
  - Flesch-Kincaid Grade Level
  - Gunning Fog Index
  - SMOG Index
  - Coleman-Liau Index
  - Automated Readability Index
- Interpretation badges (e.g., "College Level", "Difficult")

#### 4.2 Text Statistics
- Metrics cards:
  - Word count
  - Sentence count
  - Average sentence length
  - Average word length (syllables)
  - Unique words

#### 4.3 Vocabulary Analysis
- Type-Token Ratio visualization
- Hapax legomena count (words appearing once)
- Vocabulary richness (Yule's K)
- Word length distribution bar chart

#### 4.4 Complexity Through Narrative
- Line chart: Readability score progression through text sections
- Identify most/least complex passages

### 5. Theme & Topic Modeling View

**Purpose**: Explore extracted themes and topics

**Components**:

#### 5.1 Topic Overview
- Number of topics extracted
- Model type used (LDA/NMF)
- Coherence/perplexity score

#### 5.2 Topic Cards
For each topic (5 total):
- Theme label (auto-interpreted: Alienation, Family, Labor, etc.)
- Top 10 words with weights
- Word cloud visualization
- Percentage of text dominated by this topic

#### 5.3 Topic Distribution Chart
- Stacked bar chart showing topic proportions across text chunks
- Interactive: click to see topic-specific passages

#### 5.4 Literary Theme Mapping
- Radar chart showing alignment with Kafka's canonical themes:
  - Alienation
  - Bureaucracy
  - Transformation
  - Guilt
  - Family
  - Anxiety/Absurdity

### 6. Character Network View

**Purpose**: Visualize character relationships

**Components**:

#### 6.1 Network Statistics
- Total characters identified
- Total relationships
- Network density
- Number of communities

#### 6.2 Interactive Network Graph
- PyVis interactive network visualization
- Node size = character importance (mention count)
- Edge thickness = relationship strength (co-occurrence)
- Node colors = community membership
- Hover for character details
- Click to highlight connections

#### 6.3 Central Characters Table
- Ranked by degree centrality
- Columns: Name, Mentions, Degree Centrality, Betweenness Centrality
- Filter/sort capabilities

#### 6.4 Community Analysis
- List of detected character communities
- Community members
- Interpretation (e.g., "Samsa Family", "Bank Officials")

#### 6.5 Top Relationships
- Table of strongest character connections
- Sample context sentences

### 7. Comparative View

**Purpose**: Compare metrics across both works

**Components**:

#### 7.1 Side-by-Side Summary
Two columns comparing Metamorphosis vs The Trial:
- Word count
- Overall sentiment
- Primary readability score
- Main themes

#### 7.2 Comparative Charts

**Sentiment Comparison**:
- Dual-axis emotional arc overlay
- Bar chart of sentiment components

**Complexity Comparison**:
- Grouped bar chart of all readability scores
- Spider/radar chart of metrics

**Theme Comparison**:
- Heatmap: Topic presence in each work
- Venn diagram of shared vs unique themes

**Character Comparison**:
- Side-by-side network metrics
- Shared character types (protagonists, authority figures)

### 8. Data Explorer (Optional Advanced View)

**Purpose**: Raw data access for power users

**Components**:
- JSON viewer for raw analysis results
- Download buttons for:
  - CSV exports
  - GEXF network file
  - Full analysis JSON
- SQL query interface (read-only)

## Implementation Plan

### Step 1: Project Setup
```bash
pip install streamlit plotly pyvis pandas
```

Create dashboard structure:
```
src/
  dashboard/
    __init__.py
    app.py           # Main Streamlit app
    pages/
      home.py
      sentiment.py
      complexity.py
      topics.py
      characters.py
      comparison.py
    components/
      charts.py       # Reusable chart components
      network.py      # Network visualization
      sidebar.py      # Navigation sidebar
    utils/
      data_loader.py  # Load from database
      formatters.py   # Data formatting helpers
```

### Step 2: Core Components (Days 1-2)
- [ ] Data loader connecting to SQLite
- [ ] Basic Streamlit app with navigation
- [ ] Sidebar with work selection
- [ ] Home page with overview stats

### Step 3: Sentiment View (Day 3)
- [ ] Sentiment gauge charts
- [ ] Emotional arc line chart
- [ ] Sentence distribution pie chart
- [ ] Keyword word cloud

### Step 4: Complexity View (Day 4)
- [ ] Readability score gauges
- [ ] Text statistics cards
- [ ] Vocabulary analysis charts
- [ ] Section-by-section complexity

### Step 5: Topics View (Day 5)
- [ ] Topic cards with word lists
- [ ] Topic distribution chart
- [ ] Theme radar chart
- [ ] Word clouds per topic

### Step 6: Character Network (Day 6)
- [ ] PyVis network visualization
- [ ] Character ranking table
- [ ] Community analysis
- [ ] Relationship display

### Step 7: Comparative View (Day 7)
- [ ] Side-by-side metrics
- [ ] Overlay charts
- [ ] Theme heatmap
- [ ] Network comparison

### Step 8: Polish & Deploy (Day 8)
- [ ] Responsive layout
- [ ] Loading states
- [ ] Error handling
- [ ] Documentation
- [ ] Optional: Streamlit Cloud deployment

## Data Requirements

The dashboard will read from these database tables:
- `books` - Title, word count, full text
- `analysis_results` - JSON results by type

Required analysis types in database:
- `preprocessing` - Token counts, unique words
- `complexity` - All readability metrics
- `sentiment` - Polarity, arc, keywords
- `topics` - Topic words, distributions
- `entities` - Characters, network data

## Sample Code Snippets

### Loading Analysis Data
```python
import json
from src.database.connection import get_session_context
from src.database.models import Book, AnalysisResult

def load_analysis(book_id: int, analysis_type: str) -> dict:
    with get_session_context() as session:
        result = session.query(AnalysisResult).filter(
            AnalysisResult.book_id == book_id,
            AnalysisResult.analysis_type == analysis_type
        ).first()
        return json.loads(result.results_json) if result else {}
```

### Sentiment Arc Chart
```python
import plotly.express as px

def plot_emotional_arc(arc_data: list) -> go.Figure:
    df = pd.DataFrame(arc_data)
    fig = px.line(
        df, x='position', y='sentiment',
        title='Emotional Arc',
        labels={'position': 'Narrative Progress (%)', 'sentiment': 'Sentiment Score'}
    )
    return fig
```

### Character Network
```python
from pyvis.network import Network

def create_network_graph(network_data: dict) -> str:
    net = Network(height='600px', width='100%', bgcolor='#222222', font_color='white')

    for char in network_data['central_characters']:
        net.add_node(char['name'], size=char['degree_centrality'] * 50)

    for rel in network_data['relationships']:
        net.add_edge(rel['character1'], rel['character2'], value=rel['weight'])

    return net.generate_html()
```

## Success Criteria

1. **Functional**: All 6 main views working with real data
2. **Interactive**: Charts respond to user input
3. **Informative**: Insights are clear and actionable
4. **Performant**: Page loads under 3 seconds
5. **Accessible**: Works on desktop and tablet

## Future Enhancements (Post-Phase 3)

- Add more Kafka works (German originals with translation)
- Expand corpus comparison (Kafka vs. other authors)
- Export reports as PDF
- User annotations and bookmarks
- Integration with Phase 4 recommendations
