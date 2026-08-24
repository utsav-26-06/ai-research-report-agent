"""
Web content extractor using trafilatura with BeautifulSoup fallback (TASK-007).
"""

import asyncio
import logging
from urllib.parse import urlparse

import trafilatura
from bs4 import BeautifulSoup
from pydantic import ValidationError

from app.models import SourceDocument
from app.tools.extraction.base import ContentExtractor, ContentExtractionError

logger = logging.getLogger(__name__)


class TrafilaturaExtractor(ContentExtractor):
    """
    Extracts main article text from a given URL using trafilatura.
    Falls back to basic BeautifulSoup extraction if trafilatura fails.
    """

    def __init__(self, min_length: int = 100):
        """
        Initialize the extractor.

        Args:
            min_length: Minimum character count to accept the extracted text.
                        Pages with less text are rejected (returns None).
        """
        self.min_length = min_length

    async def extract(
        self,
        url: str,
        sub_question_id: str,
        query: str = "",
    ) -> SourceDocument | None:
        """
        Fetch and extract text content from the URL asynchronously.

        Args:
            url: The page URL to fetch.
            sub_question_id: Provenance ID.
            query: The search query that led to this URL.

        Returns:
            SourceDocument if successful and meets length requirement.
            None if fetch/extract fails or content is too short.
        """
        if not url.startswith(("http://", "https://")):
            raise ContentExtractionError(f"Invalid URL scheme: {url}")

        domain = urlparse(url).netloc

        try:
            # We run trafilatura in a threadpool to avoid blocking the asyncio loop
            # since trafilatura.fetch_url and extract are synchronous.
            downloaded = await asyncio.to_thread(trafilatura.fetch_url, url)
            
            if not downloaded:
                logger.warning(f"Failed to download URL: {url}")
                return None

            # 1. Try trafilatura extraction
            result = await asyncio.to_thread(
                trafilatura.extract,
                downloaded,
                output_format="json",
                include_links=False,
                include_images=False,
                include_comments=False,
            )
            
            text = ""
            title = ""
            author = None
            date = None

            if result:
                import json
                try:
                    data = json.loads(result)
                    text = data.get("text", "")
                    title = data.get("title", "")
                    author = data.get("author")
                    date = data.get("date")
                except json.JSONDecodeError:
                    text = str(result)
            
            # 2. BeautifulSoup fallback if trafilatura returned empty/None
            if not text or not text.strip():
                logger.debug(f"Trafilatura returned empty for {url}, trying BeautifulSoup fallback.")
                soup = BeautifulSoup(downloaded, "html.parser")
                
                # Remove script and style elements
                for script in soup(["script", "style", "nav", "footer", "header", "aside"]):
                    script.extract()
                    
                text = soup.get_text(separator="\n", strip=True)
                if not title and soup.title:
                    title = soup.title.string or ""
                    
            text = text.strip()
            if len(text) < self.min_length:
                logger.warning(
                    f"Extracted content too short ({len(text)} chars) for {url}. "
                    f"Minimum required is {self.min_length}."
                )
                return None

            return SourceDocument(
                url=url,
                title=title.strip() if title else "",
                domain=domain,
                author=author,
                published_date=date,
                text=text,
                sub_question_id=sub_question_id,
                query=query,
            )

        except ValidationError as ve:
            # Re-raise Pydantic validation errors so we can catch them explicitly
            logger.error(f"Validation error creating SourceDocument for {url}: {ve}")
            return None
        except Exception as e:
            logger.warning(f"Unexpected error extracting {url}: {e}")
            return None
