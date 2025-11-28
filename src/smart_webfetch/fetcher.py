"""Core fetching logic with size detection and smart truncation."""

from __future__ import annotations

import re
from dataclasses import dataclass

import httpx
import tiktoken


@dataclass
class PrefetchResult:
    """Metadata about a URL without fetching full content."""

    url: str
    status_code: int
    content_type: str
    content_length: int | None
    estimated_tokens: int
    is_html: bool
    title: str | None

    @property
    def is_large(self) -> bool:
        """Check if content is likely to exceed typical context limits."""
        return self.estimated_tokens > 8000

    @property
    def size_category(self) -> str:
        """Categorize content size for user feedback."""
        tokens = self.estimated_tokens
        if tokens < 2000:
            return "small"
        elif tokens < 8000:
            return "medium"
        elif tokens < 32000:
            return "large"
        else:
            return "very_large"


@dataclass
class FetchResult:
    """Result of fetching and processing content."""

    url: str
    content: str
    token_count: int
    was_truncated: bool
    strategy_used: str
    original_tokens: int | None = None

    @property
    def truncation_ratio(self) -> float:
        """How much of the original content was kept (1.0 = all)."""
        if self.original_tokens is None or self.original_tokens == 0:
            return 1.0
        return self.token_count / self.original_tokens


class FetchError(Exception):
    """Error during fetching operations."""

    def __init__(self, message: str, url: str, status_code: int | None = None):
        super().__init__(message)
        self.url = url
        self.status_code = status_code


class SmartFetcher:
    """Intelligent HTTP fetcher with token-aware content handling."""

    # Approximate characters per token (conservative estimate for mixed content)
    CHARS_PER_TOKEN = 4

    # Partial fetch size for HTML title extraction (16KB)
    PARTIAL_FETCH_SIZE = 16384

    # Default headers
    DEFAULT_HEADERS = {
        "User-Agent": "SmartWebFetch-MCP/0.1 (+https://github.com/mathisto/smart-webfetch-mcp)",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }

    def __init__(
        self,
        default_max_tokens: int = 8000,
        timeout: float = 30.0,
        follow_redirects: bool = True,
    ):
        """
        Initialize the smart fetcher.

        Args:
            default_max_tokens: Default token limit for fetched content.
            timeout: Request timeout in seconds.
            follow_redirects: Whether to follow HTTP redirects.
        """
        self.default_max_tokens = default_max_tokens
        self._client: httpx.AsyncClient | None = None
        self._timeout = timeout
        self._follow_redirects = follow_redirects
        self._encoder: tiktoken.Encoding | None = None

    @property
    def client(self) -> httpx.AsyncClient:
        """Lazy-initialize the HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                follow_redirects=self._follow_redirects,
                timeout=self._timeout,
                headers=self.DEFAULT_HEADERS,
            )
        return self._client

    @property
    def encoder(self) -> tiktoken.Encoding:
        """Lazy-initialize the token encoder."""
        if self._encoder is None:
            self._encoder = tiktoken.get_encoding("cl100k_base")
        return self._encoder

    def count_tokens(self, text: str) -> int:
        """
        Count tokens using tiktoken's cl100k_base encoding.

        Args:
            text: Text to count tokens for.

        Returns:
            Number of tokens in the text.
        """
        return len(self.encoder.encode(text))

    def estimate_tokens_from_bytes(self, byte_count: int) -> int:
        """
        Estimate token count from byte count.

        Uses a conservative estimate assuming mixed content.

        Args:
            byte_count: Content length in bytes.

        Returns:
            Estimated token count.
        """
        # Assume mostly ASCII, with some overhead for markup
        return byte_count // self.CHARS_PER_TOKEN

    def _is_html_content_type(self, content_type: str) -> bool:
        """Check if content type indicates HTML."""
        html_types = ("text/html", "application/xhtml+xml")
        return any(ct in content_type.lower() for ct in html_types)

    def _extract_title(self, html_chunk: str) -> str | None:
        """
        Extract title from HTML content.

        Uses simple regex to avoid full HTML parsing for partial content.

        Args:
            html_chunk: Partial or full HTML content.

        Returns:
            Extracted title or None.
        """
        # Try to find <title> tag
        title_match = re.search(
            r"<title[^>]*>([^<]+)</title>",
            html_chunk,
            re.IGNORECASE | re.DOTALL,
        )
        if title_match:
            title = title_match.group(1).strip()
            # Clean up whitespace and HTML entities
            title = re.sub(r"\s+", " ", title)
            title = title.replace("&amp;", "&")
            title = title.replace("&lt;", "<")
            title = title.replace("&gt;", ">")
            title = title.replace("&quot;", '"')
            title = title.replace("&#39;", "'")
            return title[:200]  # Limit title length
        return None

    async def prefetch(self, url: str) -> PrefetchResult:
        """
        Get metadata about a URL without fetching full content.

        Performs a HEAD request first, then a partial GET if needed
        to extract title and estimate size.

        Args:
            url: URL to prefetch.

        Returns:
            PrefetchResult with URL metadata.

        Raises:
            FetchError: If the request fails.
        """
        try:
            # Try HEAD request first
            head_response = await self.client.head(url)
            head_response.raise_for_status()

            content_type = head_response.headers.get("content-type", "")
            content_length_str = head_response.headers.get("content-length")
            content_length = int(content_length_str) if content_length_str else None
            is_html = self._is_html_content_type(content_type)

            title: str | None = None
            estimated_tokens: int

            # If we have content-length, estimate tokens directly
            if content_length is not None:
                estimated_tokens = self.estimate_tokens_from_bytes(content_length)

                # For HTML, do partial GET to extract title
                if is_html:
                    title = await self._fetch_title_partial(url)
            else:
                # No content-length: do partial GET to estimate
                partial_content, actual_content_type = await self._fetch_partial(url)

                # Update content type from actual response
                if actual_content_type:
                    content_type = actual_content_type
                    is_html = self._is_html_content_type(content_type)

                if is_html:
                    title = self._extract_title(partial_content)

                # Estimate based on partial content
                partial_tokens = self.count_tokens(partial_content)
                # Assume content is at least 4x the partial fetch (heuristic)
                estimated_tokens = max(partial_tokens, partial_tokens * 4)

            return PrefetchResult(
                url=str(head_response.url),  # May differ due to redirects
                status_code=head_response.status_code,
                content_type=content_type,
                content_length=content_length,
                estimated_tokens=estimated_tokens,
                is_html=is_html,
                title=title,
            )

        except httpx.HTTPStatusError as e:
            raise FetchError(
                f"HTTP error: {e.response.status_code}",
                url=url,
                status_code=e.response.status_code,
            ) from e
        except httpx.RequestError as e:
            raise FetchError(f"Request failed: {e}", url=url) from e

    async def _fetch_title_partial(self, url: str) -> str | None:
        """Fetch just enough content to extract title."""
        try:
            async with self.client.stream("GET", url) as response:
                chunks = []
                async for chunk in response.aiter_bytes():
                    chunks.append(chunk)
                    if sum(len(c) for c in chunks) >= self.PARTIAL_FETCH_SIZE:
                        break

                content = b"".join(chunks).decode("utf-8", errors="replace")
                return self._extract_title(content)
        except Exception:
            return None

    async def _fetch_partial(self, url: str) -> tuple[str, str | None]:
        """
        Fetch partial content for size estimation.

        Returns:
            Tuple of (partial_content, content_type).
        """
        async with self.client.stream("GET", url) as response:
            response.raise_for_status()
            content_type = response.headers.get("content-type")

            chunks = []
            async for chunk in response.aiter_bytes():
                chunks.append(chunk)
                if sum(len(c) for c in chunks) >= self.PARTIAL_FETCH_SIZE:
                    break

            content = b"".join(chunks).decode("utf-8", errors="replace")
            return content, content_type

    async def fetch_raw(self, url: str) -> tuple[str, str]:
        """
        Fetch raw content from URL.

        Args:
            url: URL to fetch.

        Returns:
            Tuple of (content, content_type).

        Raises:
            FetchError: If the request fails.
        """
        try:
            response = await self.client.get(url)
            response.raise_for_status()

            content_type = response.headers.get("content-type", "text/plain")
            content = response.text

            return content, content_type

        except httpx.HTTPStatusError as e:
            raise FetchError(
                f"HTTP error: {e.response.status_code}",
                url=url,
                status_code=e.response.status_code,
            ) from e
        except httpx.RequestError as e:
            raise FetchError(f"Request failed: {e}", url=url) from e

    def _find_natural_break(self, text: str, max_chars: int) -> int:
        """
        Find a natural break point in text near the character limit.

        Looks for paragraph breaks, sentence ends, or word boundaries.

        Args:
            text: Text to find break in.
            max_chars: Maximum character position.

        Returns:
            Character position for break.
        """
        if len(text) <= max_chars:
            return len(text)

        # Search window: look back up to 500 chars from limit
        search_start = max(0, max_chars - 500)
        search_text = text[search_start:max_chars]

        # Priority 1: Double newline (paragraph break)
        para_break = search_text.rfind("\n\n")
        if para_break != -1:
            return search_start + para_break

        # Priority 2: Single newline
        line_break = search_text.rfind("\n")
        if line_break != -1:
            return search_start + line_break

        # Priority 3: Sentence end (. ! ?)
        for punct in (". ", "! ", "? "):
            sent_break = search_text.rfind(punct)
            if sent_break != -1:
                return search_start + sent_break + len(punct)

        # Priority 4: Word boundary (space)
        word_break = search_text.rfind(" ")
        if word_break != -1:
            return search_start + word_break

        # Fallback: hard cut at limit
        return max_chars

    def _truncate_to_tokens(
        self,
        text: str,
        max_tokens: int,
        strategy: str = "auto",
    ) -> tuple[str, int]:
        """
        Truncate text to fit within token limit.

        Args:
            text: Text to truncate.
            max_tokens: Maximum tokens allowed.
            strategy: Truncation strategy ('truncate' or 'auto').

        Returns:
            Tuple of (truncated_text, actual_token_count).
        """
        current_tokens = self.count_tokens(text)

        if current_tokens <= max_tokens:
            return text, current_tokens

        # Estimate character limit (with some buffer)
        estimated_chars = int(max_tokens * self.CHARS_PER_TOKEN * 0.95)

        if strategy == "truncate":
            # Hard truncate at estimated position
            truncated = text[:estimated_chars]
        else:
            # Auto: find natural break point
            break_pos = self._find_natural_break(text, estimated_chars)
            truncated = text[:break_pos]

        # Verify and adjust if needed
        actual_tokens = self.count_tokens(truncated)

        # If still over, do binary search to find exact cut
        if actual_tokens > max_tokens:
            truncated = self._binary_search_truncate(truncated, max_tokens)
            actual_tokens = self.count_tokens(truncated)

        # Add truncation indicator
        truncated = truncated.rstrip() + "\n\n[Content truncated...]"
        actual_tokens = self.count_tokens(truncated)

        return truncated, actual_tokens

    def _binary_search_truncate(self, text: str, max_tokens: int) -> str:
        """Binary search for exact token limit truncation."""
        low, high = 0, len(text)

        while low < high:
            mid = (low + high + 1) // 2
            if self.count_tokens(text[:mid]) <= max_tokens:
                low = mid
            else:
                high = mid - 1

        return text[:low]

    async def smart_fetch(
        self,
        url: str,
        max_tokens: int | None = None,
        strategy: str = "auto",
    ) -> FetchResult:
        """
        Fetch URL with automatic handling of large content.

        Performs a prefetch to check size, then fetches with appropriate
        truncation strategy if content exceeds token limit.

        Args:
            url: URL to fetch.
            max_tokens: Maximum tokens for result (uses default if None).
            strategy: Handling strategy for large content:
                - 'auto': Find natural break points for truncation.
                - 'truncate': Hard cut at token limit.
                - 'summarize': Reserved for future LLM summarization.

        Returns:
            FetchResult with content and metadata.

        Raises:
            FetchError: If the request fails.
            ValueError: If strategy is invalid.
        """
        if strategy not in ("auto", "truncate", "summarize"):
            raise ValueError(f"Invalid strategy: {strategy}")

        if strategy == "summarize":
            # For now, fall back to auto; summarization requires LLM integration
            strategy = "auto"

        effective_max = max_tokens or self.default_max_tokens

        # Step 1: Prefetch to check size
        prefetch_result = await self.prefetch(url)

        # Step 2: Fetch full content
        content, content_type = await self.fetch_raw(prefetch_result.url)
        original_tokens = self.count_tokens(content)

        # Step 3: Check if truncation needed
        if original_tokens <= effective_max:
            return FetchResult(
                url=prefetch_result.url,
                content=content,
                token_count=original_tokens,
                was_truncated=False,
                strategy_used="none",
                original_tokens=original_tokens,
            )

        # Step 4: Apply truncation strategy
        truncated_content, final_tokens = self._truncate_to_tokens(
            content,
            effective_max,
            strategy,
        )

        return FetchResult(
            url=prefetch_result.url,
            content=truncated_content,
            token_count=final_tokens,
            was_truncated=True,
            strategy_used=strategy,
            original_tokens=original_tokens,
        )

    async def close(self) -> None:
        """Close the HTTP client and release resources."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> SmartFetcher:
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
