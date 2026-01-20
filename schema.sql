-- Literary Analysis & Recommendation Platform
-- Database Schema
-- SQLite-compatible

-- Authors table
CREATE TABLE IF NOT EXISTS authors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    birth_year INTEGER,
    death_year INTEGER,
    nationality TEXT,
    biography TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_author_name ON authors(name);

-- Books table
CREATE TABLE IF NOT EXISTS books (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    gutenberg_id INTEGER UNIQUE,
    title TEXT NOT NULL,
    author_id INTEGER REFERENCES authors(id),
    publication_year INTEGER,
    language TEXT DEFAULT 'en',
    genre TEXT,
    word_count INTEGER,
    full_text TEXT,
    source TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_book_title ON books(title);
CREATE INDEX IF NOT EXISTS idx_book_author ON books(author_id);
CREATE INDEX IF NOT EXISTS idx_book_gutenberg ON books(gutenberg_id);

-- Book metadata from external sources (Goodreads, etc.)
CREATE TABLE IF NOT EXISTS book_metadata (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    book_id INTEGER NOT NULL REFERENCES books(id) ON DELETE CASCADE,
    source TEXT NOT NULL,  -- 'goodreads', 'gutenberg', etc.
    rating REAL,
    rating_count INTEGER,
    review_count INTEGER,
    description TEXT,
    subjects TEXT,  -- JSON array of subjects/genres
    isbn TEXT,
    isbn13 TEXT,
    raw_json TEXT,
    match_confidence REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(book_id, source)
);

CREATE INDEX IF NOT EXISTS idx_metadata_book ON book_metadata(book_id);
CREATE INDEX IF NOT EXISTS idx_metadata_source ON book_metadata(source);

-- NLP analysis results (Phase 2)
CREATE TABLE IF NOT EXISTS analysis_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    book_id INTEGER NOT NULL REFERENCES books(id) ON DELETE CASCADE,
    analysis_type TEXT NOT NULL,  -- 'sentiment', 'complexity', 'topics', 'themes', etc.
    results_json TEXT NOT NULL,
    model_version TEXT,
    processing_time_seconds REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(book_id, analysis_type, model_version)
);

CREATE INDEX IF NOT EXISTS idx_analysis_book ON analysis_results(book_id);
CREATE INDEX IF NOT EXISTS idx_analysis_type ON analysis_results(analysis_type);

-- Embedding metadata for vector search (Phase 4)
CREATE TABLE IF NOT EXISTS embeddings_metadata (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    book_id INTEGER NOT NULL REFERENCES books(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    chunk_start INTEGER,
    chunk_end INTEGER,
    chunk_text TEXT,
    embedding_model TEXT NOT NULL,
    vector_db_id TEXT,  -- Reference to FAISS index position
    embedding_dimension INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(book_id, chunk_index, embedding_model)
);

CREATE INDEX IF NOT EXISTS idx_embedding_book ON embeddings_metadata(book_id);
CREATE INDEX IF NOT EXISTS idx_embedding_model ON embeddings_metadata(embedding_model);

-- Trigger to update updated_at timestamp (SQLite doesn't support this natively)
-- This would need to be handled in application code or via triggers
