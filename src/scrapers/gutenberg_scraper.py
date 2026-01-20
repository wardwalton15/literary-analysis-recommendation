"""Project Gutenberg web scraper for collecting classic literature texts."""

import re
import time
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, asdict

import requests
from bs4 import BeautifulSoup
import chardet
from tqdm import tqdm

from config.settings import get_settings


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BookInfo:
    """Information about a book from Project Gutenberg."""

    gutenberg_id: int
    title: str
    author: Optional[str] = None
    language: Optional[str] = None
    subjects: Optional[List[str]] = None
    publication_year: Optional[int] = None
    download_url: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


class GutenbergScraper:
    """Scraper for Project Gutenberg texts."""

    def __init__(self, delay: float = 1.0):
        """
        Initialize the scraper.

        Args:
            delay: Seconds to wait between requests (be respectful to the server)
        """
        settings = get_settings()
        self.base_url = settings.gutenberg_base_url
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": settings.user_agent})
        self.max_retries = settings.max_retries
        self.retry_backoff = settings.retry_backoff

    def _make_request(self, url: str) -> requests.Response:
        """Make a request with retries and rate limiting."""
        last_exception = None

        for attempt in range(self.max_retries):
            try:
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                time.sleep(self.delay)
                return response
            except requests.RequestException as e:
                last_exception = e
                wait_time = self.retry_backoff ** attempt
                logger.warning(
                    f"Request failed (attempt {attempt + 1}/{self.max_retries}): {e}. "
                    f"Retrying in {wait_time}s..."
                )
                time.sleep(wait_time)

        raise last_exception

    def get_book_text(self, book_id: int) -> str:
        """
        Fetch the plain text version of a book.

        Args:
            book_id: Project Gutenberg book ID

        Returns:
            Raw text content of the book
        """
        # Try different URL patterns
        url_patterns = [
            f"{self.base_url}/cache/epub/{book_id}/pg{book_id}.txt",
            f"{self.base_url}/files/{book_id}/{book_id}-0.txt",
            f"{self.base_url}/files/{book_id}/{book_id}.txt",
        ]

        for url in url_patterns:
            try:
                response = self._make_request(url)
                # Detect encoding
                detected = chardet.detect(response.content)
                encoding = detected.get("encoding", "utf-8")
                return response.content.decode(encoding, errors="replace")
            except requests.HTTPError as e:
                if e.response.status_code == 404:
                    continue
                raise

        raise ValueError(f"Could not find text for book ID {book_id}")

    def get_book_metadata(self, book_id: int) -> BookInfo:
        """
        Fetch metadata for a book from its Gutenberg page.

        Args:
            book_id: Project Gutenberg book ID

        Returns:
            BookInfo object with metadata
        """
        url = f"{self.base_url}/ebooks/{book_id}"
        response = self._make_request(url)
        soup = BeautifulSoup(response.content, "lxml")

        # Extract title
        title_elem = soup.select_one("h1[itemprop='name']")
        title = title_elem.get_text(strip=True) if title_elem else f"Unknown (ID: {book_id})"

        # Extract author
        author_elem = soup.select_one("a[itemprop='creator']")
        author = author_elem.get_text(strip=True) if author_elem else None

        # Extract language
        language_elem = soup.select_one("tr[itemprop='inLanguage'] td")
        language = language_elem.get_text(strip=True) if language_elem else "en"

        # Extract subjects
        subject_elems = soup.select("td[datatype='dcterms:LCSH'] a")
        subjects = [s.get_text(strip=True) for s in subject_elems] if subject_elems else []

        return BookInfo(
            gutenberg_id=book_id,
            title=title,
            author=author,
            language=language,
            subjects=subjects,
            download_url=f"{self.base_url}/cache/epub/{book_id}/pg{book_id}.txt",
        )

    def clean_gutenberg_text(self, raw_text: str) -> str:
        """
        Remove Project Gutenberg headers and footers.

        Args:
            raw_text: Raw text with Gutenberg boilerplate

        Returns:
            Cleaned text with just the book content
        """
        # Common start markers
        start_patterns = [
            r"\*\*\* START OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*",
            r"\*\*\* START OF THIS PROJECT GUTENBERG EBOOK.*?\*\*\*",
            r"\*\*\*START OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*",
            r"START OF THE PROJECT GUTENBERG EBOOK",
        ]

        # Common end markers
        end_patterns = [
            r"\*\*\* END OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*",
            r"\*\*\* END OF THIS PROJECT GUTENBERG EBOOK.*?\*\*\*",
            r"\*\*\*END OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*",
            r"END OF THE PROJECT GUTENBERG EBOOK",
        ]

        text = raw_text

        # Find and remove header
        for pattern in start_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                text = text[match.end():]
                break

        # Find and remove footer
        for pattern in end_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                text = text[: match.start()]
                break

        # Clean up extra whitespace
        text = text.strip()

        # Remove any remaining Gutenberg-specific lines at the start
        lines = text.split("\n")
        clean_lines = []
        started = False

        for line in lines:
            # Skip initial production notes
            if not started and (
                "produced by" in line.lower()
                or "transcribed by" in line.lower()
                or "updated edition" in line.lower()
                or line.strip() == ""
            ):
                continue
            started = True
            clean_lines.append(line)

        return "\n".join(clean_lines).strip()

    def scrape_book(
        self, book_id: int, clean: bool = True
    ) -> Tuple[BookInfo, str]:
        """
        Scrape a complete book with metadata and text.

        Args:
            book_id: Project Gutenberg book ID
            clean: Whether to clean the Gutenberg boilerplate

        Returns:
            Tuple of (BookInfo, text content)
        """
        logger.info(f"Scraping book ID {book_id}...")

        metadata = self.get_book_metadata(book_id)
        text = self.get_book_text(book_id)

        if clean:
            text = self.clean_gutenberg_text(text)

        logger.info(f"Successfully scraped: {metadata.title}")
        return metadata, text


class GutenbergBatchDownloader:
    """Batch downloader for multiple Gutenberg books."""

    def __init__(
        self,
        book_ids: List[int],
        output_dir: Optional[Path] = None,
        delay: float = 1.0,
    ):
        """
        Initialize the batch downloader.

        Args:
            book_ids: List of Gutenberg book IDs to download
            output_dir: Directory to save downloaded texts
            delay: Seconds between requests
        """
        self.scraper = GutenbergScraper(delay=delay)
        self.book_ids = book_ids
        self.output_dir = output_dir or get_settings().raw_data_dir / "corpus"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.failures: List[Dict] = []
        self.successes: List[BookInfo] = []

    def _text_path(self, book_id: int) -> Path:
        """Get the path for a book's text file."""
        return self.output_dir / f"pg{book_id}.txt"

    def _metadata_path(self, book_id: int) -> Path:
        """Get the path for a book's metadata file."""
        return self.output_dir / f"pg{book_id}_meta.json"

    def _already_downloaded(self, book_id: int) -> bool:
        """Check if a book has already been downloaded."""
        return self._text_path(book_id).exists()

    def _save_text(self, book_id: int, text: str) -> None:
        """Save book text to file."""
        self._text_path(book_id).write_text(text, encoding="utf-8")

    def _save_metadata(self, book_info: BookInfo) -> None:
        """Save book metadata to JSON file."""
        self._metadata_path(book_info.gutenberg_id).write_text(
            json.dumps(book_info.to_dict(), indent=2), encoding="utf-8"
        )

    def _log_failure(self, book_id: int, error: Exception) -> None:
        """Log a download failure."""
        failure = {
            "book_id": book_id,
            "error": str(error),
            "error_type": type(error).__name__,
        }
        self.failures.append(failure)
        logger.error(f"Failed to download book {book_id}: {error}")

    def download_all(self, resume: bool = True, progress: bool = True) -> dict:
        """
        Download all books in the list.

        Args:
            resume: Skip already downloaded books
            progress: Show progress bar

        Returns:
            Summary dict with success/failure counts
        """
        book_ids = self.book_ids

        if resume:
            book_ids = [
                bid for bid in book_ids if not self._already_downloaded(bid)
            ]
            logger.info(
                f"Resuming: {len(self.book_ids) - len(book_ids)} already downloaded"
            )

        iterator = tqdm(book_ids, desc="Downloading books") if progress else book_ids

        for book_id in iterator:
            try:
                metadata, text = self.scraper.scrape_book(book_id, clean=True)
                self._save_text(book_id, text)
                self._save_metadata(metadata)
                self.successes.append(metadata)
            except Exception as e:
                self._log_failure(book_id, e)

        # Save failure log
        if self.failures:
            failures_path = self.output_dir / "download_failures.json"
            failures_path.write_text(
                json.dumps(self.failures, indent=2), encoding="utf-8"
            )

        summary = {
            "total_requested": len(self.book_ids),
            "downloaded": len(self.successes),
            "failed": len(self.failures),
            "skipped_existing": len(self.book_ids) - len(book_ids) if resume else 0,
        }

        logger.info(f"Download complete: {summary}")
        return summary

    def get_manifest(self) -> List[Dict]:
        """Get manifest of all successfully downloaded books."""
        return [book.to_dict() for book in self.successes]
