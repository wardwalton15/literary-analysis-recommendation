# Literary Analysis & Recommendation Platform

An NLP-powered platform for analyzing classic literature, with a deep dive into Franz Kafka's works and ML-based book recommendations.

## Project Overview

This platform combines web scraping, natural language processing, and machine learning to:

1. **Collect** classic literature texts from Project Gutenberg
2. **Analyze** literary works using NLP techniques (sentiment, complexity, themes)
3. **Generate** embeddings for semantic similarity search
4. **Recommend** books based on textual and thematic similarities

### Key Features

- Automated scraping of Project Gutenberg texts
- SQLite database for structured storage of books and metadata
- Comprehensive Kafka corpus (The Metamorphosis, The Trial, etc.)
- Curated collection of 100+ classic literary works
- Data quality validation framework
- Extensible architecture for NLP analysis and ML recommendations

## Project Structure

```
literary-analysis/
├── data/
│   ├── raw/              # Original scraped/downloaded data
│   │   ├── kafka/        # Kafka works
│   │   └── corpus/       # Broader classic literature
│   ├── processed/        # Cleaned, transformed data
│   └── external/         # Goodreads/Kaggle datasets
├── src/
│   ├── scrapers/         # Web scraping modules
│   │   ├── gutenberg_scraper.py
│   │   ├── kafka_manifest.py
│   │   └── corpus_manifest.py
│   ├── preprocessing/    # Text cleaning pipelines
│   ├── analysis/         # NLP analysis code
│   ├── database/         # DB models and connections
│   │   ├── models.py
│   │   ├── connection.py
│   │   └── loaders.py
│   ├── utils/            # Helper functions
│   │   └── data_quality.py
│   └── cli.py            # Command-line interface
├── notebooks/            # Exploratory analysis
├── tests/                # Unit tests
├── config/               # Configuration files
│   └── settings.py
├── requirements.txt
├── setup.py
├── schema.sql
└── README.md
```

## Installation

### Prerequisites

- Python 3.10 or higher
- pip (Python package manager)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/literary-analysis.git
   cd literary-analysis
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv

   # On Windows
   venv\Scripts\activate

   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Initialize the database**
   ```bash
   python -m src.cli init
   ```

## Usage

### Command-Line Interface

The platform provides a CLI for common operations:

```bash
# Initialize the database
python -m src.cli init

# Scrape Kafka works from Project Gutenberg
python -m src.cli scrape-kafka

# Scrape the broader classic literature corpus
python -m src.cli scrape-corpus --limit 50

# Load downloaded texts into the database
python -m src.cli load-data

# Run data quality validation
python -m src.cli validate

# Show database statistics
python -m src.cli stats

# Test the scraper with a single book
python -m src.cli test-scraper --book-id 5200
```

### Programmatic Usage

```python
from src.scrapers.gutenberg_scraper import GutenbergScraper
from src.database.loaders import DataLoader
from src.database.connection import init_db

# Initialize database
init_db()

# Scrape a single book
scraper = GutenbergScraper(delay=1.0)
metadata, text = scraper.scrape_book(5200)  # The Metamorphosis
print(f"Downloaded: {metadata.title} ({len(text)} characters)")

# Load books into database
with DataLoader() as loader:
    books = loader.load_kafka_works()
    stats = loader.get_statistics()
    print(f"Loaded {stats['kafka_books']} Kafka works")
```

## Data Collection

### Kafka Works

The platform includes a curated manifest of Franz Kafka's works available on Project Gutenberg:

| Title | Gutenberg ID | Year | Type |
|-------|-------------|------|------|
| The Metamorphosis | 5200 | 1915 | Novella |
| The Trial | 7849 | 1925 | Novel |
| In the Penal Colony | 22367 | 1919 | Short Story |
| A Hunger Artist | 61097 | 1922 | Short Story |
| The Castle | 65147 | 1926 | Novel |
| The Judgment | 62846 | 1913 | Short Story |

### Classic Literature Corpus

The broader corpus includes 100+ carefully selected works representing:

- **Russian Literature**: Dostoevsky, Tolstoy, Turgenev, Gogol
- **British Literature**: Dickens, Austen, Brontë sisters, Woolf, Joyce
- **French Literature**: Hugo, Dumas, Flaubert, Zola
- **American Literature**: Melville, Twain, Poe, Hawthorne, James
- **German Literature**: Goethe, Nietzsche, Hesse
- **Classical Works**: Homer, Dante, Plato, Shakespeare

## Database Schema

The SQLite database uses the following structure:

- **authors**: Author information (name, birth/death years, nationality)
- **books**: Book data (title, full text, word count, publication year)
- **book_metadata**: Enrichment data from external sources (Goodreads ratings)
- **analysis_results**: NLP analysis results (sentiment, complexity, themes)
- **embeddings_metadata**: Vector embedding information for recommendations

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_scrapers.py
```

### Code Style

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Lint
flake8 src/ tests/
```

## Project Phases

### Phase 1: Data Collection & Setup ✅
- Project structure and environment
- Database schema design
- Gutenberg scraper implementation
- Kafka and corpus data collection
- Data quality validation

### Phase 2: NLP Analysis Pipeline (Upcoming)
- Text preprocessing
- Sentiment analysis
- Readability metrics
- Topic modeling
- Theme extraction

### Phase 3: Feature Engineering (Upcoming)
- Literary feature extraction
- Style analysis
- Narrative structure analysis

### Phase 4: Recommendation System (Upcoming)
- Text embeddings with sentence-transformers
- Vector database with FAISS
- Similarity-based recommendations
- Hybrid recommendation approach

### Phase 5: Web Interface (Upcoming)
- FastAPI backend
- React frontend
- Interactive visualizations

## Configuration

Configuration is managed through `config/settings.py`:

```python
from config.settings import get_settings

settings = get_settings()
print(settings.database_url)
print(settings.scrape_delay)
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Project Gutenberg](https://www.gutenberg.org/) for providing free access to classic literature
- The NLP and ML communities for open-source tools and libraries

## Project Status

**Phase 1 Complete** - Data Collection & Setup

- [x] GitHub repo created with proper structure
- [x] Virtual environment configured
- [x] Database schema implemented
- [x] Gutenberg scraper built
- [x] Kafka works manifest created
- [x] Broader corpus manifest created (100+ books)
- [x] Data loaders implemented
- [x] Quality validation framework
- [x] CLI interface
- [x] Unit tests
- [x] Documentation

Ready to proceed with Phase 2: NLP Analysis Pipeline.
