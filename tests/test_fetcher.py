"""Tests for the fetcher module."""

import pytest
from pytest_httpx import HTTPXMock

from smart_webfetch.fetcher import FetchError, FetchResult, PrefetchResult, SmartFetcher


class TestSmartFetcher:
    """Tests for SmartFetcher class."""

    def test_count_tokens(self):
        """Test token counting."""
        fetcher = SmartFetcher()

        # Simple text
        count = fetcher.count_tokens("Hello, world!")
        assert count > 0
        assert count < 10  # Should be ~4 tokens

    def test_estimate_tokens_from_bytes(self):
        """Test byte-to-token estimation."""
        fetcher = SmartFetcher()

        # 1000 bytes should be ~250 tokens (4 bytes per token estimate)
        estimate = fetcher.estimate_tokens_from_bytes(1000)
        assert estimate == 250

    def test_prefetch_result_properties(self):
        """Test PrefetchResult helper properties."""
        # Small page
        small = PrefetchResult(
            url="https://example.com",
            status_code=200,
            content_type="text/html",
            content_length=1000,
            estimated_tokens=500,
            is_html=True,
            title="Example",
        )
        assert not small.is_large
        assert small.size_category == "small"

        # Large page
        large = PrefetchResult(
            url="https://example.com",
            status_code=200,
            content_type="text/html",
            content_length=100000,
            estimated_tokens=25000,
            is_html=True,
            title="Example",
        )
        assert large.is_large
        assert large.size_category == "large"

    def test_fetch_result_truncation_ratio(self):
        """Test FetchResult truncation ratio calculation."""
        # No truncation
        full = FetchResult(
            url="https://example.com",
            content="Hello",
            token_count=1,
            was_truncated=False,
            strategy_used="none",
            original_tokens=1,
        )
        assert full.truncation_ratio == 1.0

        # 50% truncation
        half = FetchResult(
            url="https://example.com",
            content="Hello",
            token_count=50,
            was_truncated=True,
            strategy_used="auto",
            original_tokens=100,
        )
        assert half.truncation_ratio == 0.5


class TestPrefetch:
    """Integration tests for SmartFetcher.prefetch() with mocked HTTP responses."""

    @pytest.mark.asyncio
    async def test_prefetch_small_html_page(self, httpx_mock: HTTPXMock):
        """Test prefetch on a small HTML page with content-length header."""
        html_content = """
        <!DOCTYPE html>
        <html>
        <head><title>Test Page</title></head>
        <body><p>Hello, world!</p></body>
        </html>
        """
        httpx_mock.add_response(
            method="HEAD",
            url="https://example.com/small",
            headers={
                "content-type": "text/html; charset=utf-8",
                "content-length": str(len(html_content)),
            },
        )
        httpx_mock.add_response(
            method="GET",
            url="https://example.com/small",
            content=html_content.encode(),
            headers={"content-type": "text/html; charset=utf-8"},
        )

        async with SmartFetcher() as fetcher:
            result = await fetcher.prefetch("https://example.com/small")

        assert result.status_code == 200
        assert result.is_html is True
        assert result.title == "Test Page"
        assert result.content_length == len(html_content)
        assert result.size_category == "small"
        assert result.is_large is False

    @pytest.mark.asyncio
    async def test_prefetch_large_page(self, httpx_mock: HTTPXMock):
        """Test prefetch on a large page that exceeds token limits."""
        # Create content that would estimate to ~50k tokens (200KB)
        large_content = "x" * 200_000
        httpx_mock.add_response(
            method="HEAD",
            url="https://example.com/large",
            headers={
                "content-type": "text/html",
                "content-length": str(len(large_content)),
            },
        )
        # prefetch does a partial GET for HTML pages to extract title
        httpx_mock.add_response(
            method="GET",
            url="https://example.com/large",
            content=large_content.encode(),
            headers={"content-type": "text/html"},
        )

        async with SmartFetcher() as fetcher:
            result = await fetcher.prefetch("https://example.com/large")

        assert result.is_large is True
        assert result.size_category == "very_large"
        assert result.estimated_tokens > 8000

    @pytest.mark.asyncio
    async def test_prefetch_non_html_content(self, httpx_mock: HTTPXMock):
        """Test prefetch on non-HTML content (JSON)."""
        json_content = '{"key": "value", "items": [1, 2, 3]}'
        httpx_mock.add_response(
            method="HEAD",
            url="https://api.example.com/data.json",
            headers={
                "content-type": "application/json",
                "content-length": str(len(json_content)),
            },
        )

        async with SmartFetcher() as fetcher:
            result = await fetcher.prefetch("https://api.example.com/data.json")

        assert result.is_html is False
        assert result.title is None
        assert "application/json" in result.content_type

    @pytest.mark.asyncio
    async def test_prefetch_no_content_length(self, httpx_mock: HTTPXMock):
        """Test prefetch when server doesn't provide content-length header."""
        html_content = "<html><head><title>No Length</title></head><body>Content</body></html>"
        httpx_mock.add_response(
            method="HEAD",
            url="https://example.com/streaming",
            headers={"content-type": "text/html"},
        )
        httpx_mock.add_response(
            method="GET",
            url="https://example.com/streaming",
            content=html_content.encode(),
            headers={"content-type": "text/html"},
        )

        async with SmartFetcher() as fetcher:
            result = await fetcher.prefetch("https://example.com/streaming")

        assert result.content_length is None
        assert result.title == "No Length"
        assert result.estimated_tokens > 0

    @pytest.mark.asyncio
    async def test_prefetch_http_error(self, httpx_mock: HTTPXMock):
        """Test prefetch handles HTTP errors correctly."""
        httpx_mock.add_response(
            method="HEAD",
            url="https://example.com/notfound",
            status_code=404,
        )

        async with SmartFetcher() as fetcher:
            with pytest.raises(FetchError) as exc_info:
                await fetcher.prefetch("https://example.com/notfound")

        assert exc_info.value.status_code == 404
        assert exc_info.value.url == "https://example.com/notfound"


class TestSmartFetchIntegration:
    """Integration tests for SmartFetcher.smart_fetch() with mocked HTTP responses."""

    @pytest.mark.asyncio
    async def test_smart_fetch_small_content(self, httpx_mock: HTTPXMock):
        """Test smart_fetch returns content without truncation for small pages."""
        html_content = "<html><head><title>Small</title></head><body>Hello</body></html>"
        httpx_mock.add_response(
            method="HEAD",
            url="https://example.com/small",
            headers={
                "content-type": "text/html",
                "content-length": str(len(html_content)),
            },
        )
        # smart_fetch calls prefetch (which does GET for title) then fetch_raw (another GET)
        httpx_mock.add_response(
            method="GET",
            url="https://example.com/small",
            content=html_content.encode(),
            headers={"content-type": "text/html"},
        )
        httpx_mock.add_response(
            method="GET",
            url="https://example.com/small",
            content=html_content.encode(),
            headers={"content-type": "text/html"},
        )

        async with SmartFetcher() as fetcher:
            result = await fetcher.smart_fetch("https://example.com/small")

        assert result.was_truncated is False
        assert result.strategy_used == "none"
        assert result.content == html_content
        assert result.truncation_ratio == 1.0

    @pytest.mark.asyncio
    async def test_smart_fetch_truncation_auto(self, httpx_mock: HTTPXMock):
        """Test smart_fetch truncates large content with auto strategy."""
        # Create content that exceeds token limit
        large_content = "This is a sentence. " * 5000  # ~5000 sentences
        httpx_mock.add_response(
            method="HEAD",
            url="https://example.com/large",
            headers={
                "content-type": "text/html",
                "content-length": str(len(large_content)),
            },
        )
        # smart_fetch calls prefetch (which does GET for title) then fetch_raw (another GET)
        httpx_mock.add_response(
            method="GET",
            url="https://example.com/large",
            content=large_content.encode(),
            headers={"content-type": "text/html"},
        )
        httpx_mock.add_response(
            method="GET",
            url="https://example.com/large",
            content=large_content.encode(),
            headers={"content-type": "text/html"},
        )

        async with SmartFetcher() as fetcher:
            result = await fetcher.smart_fetch(
                "https://example.com/large",
                max_tokens=1000,
                strategy="auto",
            )

        assert result.was_truncated is True
        assert result.strategy_used == "auto"
        assert result.token_count <= 1000 + 10  # Small buffer for truncation marker
        assert "[Content truncated...]" in result.content
        assert result.truncation_ratio < 1.0

    @pytest.mark.asyncio
    async def test_smart_fetch_truncation_hard(self, httpx_mock: HTTPXMock):
        """Test smart_fetch with hard truncate strategy."""
        large_content = "word " * 10000
        httpx_mock.add_response(
            method="HEAD",
            url="https://example.com/large",
            headers={
                "content-type": "text/plain",
                "content-length": str(len(large_content)),
            },
        )
        httpx_mock.add_response(
            method="GET",
            url="https://example.com/large",
            content=large_content.encode(),
            headers={"content-type": "text/plain"},
        )

        async with SmartFetcher() as fetcher:
            result = await fetcher.smart_fetch(
                "https://example.com/large",
                max_tokens=500,
                strategy="truncate",
            )

        assert result.was_truncated is True
        assert result.strategy_used == "truncate"
        assert "[Content truncated...]" in result.content

    @pytest.mark.asyncio
    async def test_smart_fetch_invalid_strategy(self, httpx_mock: HTTPXMock):
        """Test smart_fetch raises ValueError for invalid strategy."""
        async with SmartFetcher() as fetcher:
            with pytest.raises(ValueError, match="Invalid strategy"):
                await fetcher.smart_fetch(
                    "https://example.com",
                    strategy="invalid",
                )

    @pytest.mark.asyncio
    async def test_smart_fetch_uses_default_max_tokens(self, httpx_mock: HTTPXMock):
        """Test smart_fetch uses default max_tokens when not specified."""
        content = "Hello, world!"
        httpx_mock.add_response(
            method="HEAD",
            url="https://example.com",
            headers={
                "content-type": "text/plain",
                "content-length": str(len(content)),
            },
        )
        httpx_mock.add_response(
            method="GET",
            url="https://example.com",
            content=content.encode(),
            headers={"content-type": "text/plain"},
        )

        async with SmartFetcher(default_max_tokens=5000) as fetcher:
            result = await fetcher.smart_fetch("https://example.com")

        # Small content should pass through without truncation
        assert result.was_truncated is False


class TestFetchRaw:
    """Integration tests for SmartFetcher.fetch_raw() with mocked HTTP responses."""

    @pytest.mark.asyncio
    async def test_fetch_raw_success(self, httpx_mock: HTTPXMock):
        """Test fetch_raw successfully retrieves content."""
        content = "Hello, world!"
        httpx_mock.add_response(
            url="https://example.com/test",
            content=content.encode(),
            headers={"content-type": "text/plain; charset=utf-8"},
        )

        async with SmartFetcher() as fetcher:
            result_content, content_type = await fetcher.fetch_raw("https://example.com/test")

        assert result_content == content
        assert "text/plain" in content_type

    @pytest.mark.asyncio
    async def test_fetch_raw_html_content(self, httpx_mock: HTTPXMock):
        """Test fetch_raw retrieves HTML content correctly."""
        html = "<html><body><h1>Hello</h1></body></html>"
        httpx_mock.add_response(
            url="https://example.com/page.html",
            content=html.encode(),
            headers={"content-type": "text/html"},
        )

        async with SmartFetcher() as fetcher:
            content, content_type = await fetcher.fetch_raw("https://example.com/page.html")

        assert content == html
        assert "text/html" in content_type

    @pytest.mark.asyncio
    async def test_fetch_raw_http_error_404(self, httpx_mock: HTTPXMock):
        """Test fetch_raw raises FetchError on 404."""
        httpx_mock.add_response(
            url="https://example.com/missing",
            status_code=404,
        )

        async with SmartFetcher() as fetcher:
            with pytest.raises(FetchError) as exc_info:
                await fetcher.fetch_raw("https://example.com/missing")

        assert exc_info.value.status_code == 404
        assert exc_info.value.url == "https://example.com/missing"
        assert "404" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_fetch_raw_http_error_500(self, httpx_mock: HTTPXMock):
        """Test fetch_raw raises FetchError on server error."""
        httpx_mock.add_response(
            url="https://example.com/error",
            status_code=500,
        )

        async with SmartFetcher() as fetcher:
            with pytest.raises(FetchError) as exc_info:
                await fetcher.fetch_raw("https://example.com/error")

        assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    async def test_fetch_raw_default_content_type(self, httpx_mock: HTTPXMock):
        """Test fetch_raw uses default content-type when header is missing."""
        httpx_mock.add_response(
            url="https://example.com/no-type",
            content=b"raw data",
        )

        async with SmartFetcher() as fetcher:
            content, content_type = await fetcher.fetch_raw("https://example.com/no-type")

        assert content == "raw data"
        assert content_type == "text/plain"


class TestExtractor:
    """Tests for ContentExtractor class."""

    def test_extract_code_blocks(self):
        """Test code block extraction from HTML."""
        from smart_webfetch.extractors import ContentExtractor

        def count_tokens(text):
            return len(text.split())

        extractor = ContentExtractor(count_tokens)

        html = """
        <html>
        <body>
            <h1>Example</h1>
            <pre><code class="language-python">
def hello():
    print("Hello, world!")
            </code></pre>
        </body>
        </html>
        """

        blocks = extractor.extract_code_blocks(html)
        assert len(blocks) >= 1
        assert blocks[0].language == "python"
        assert "def hello" in blocks[0].content

    def test_chunk_content(self):
        """Test content chunking."""
        from smart_webfetch.extractors import ContentExtractor

        def count_tokens(text):
            # Simple word-based token count for testing
            return len(text.split())

        extractor = ContentExtractor(count_tokens)

        # Create content that will need chunking
        content = " ".join(["word"] * 100)  # 100 "tokens"

        chunks = extractor.chunk_content(content, chunk_size=30)
        assert len(chunks) >= 3
        assert chunks[0].index == 0
        assert chunks[-1].index == len(chunks) - 1
        assert all(c.total_chunks == len(chunks) for c in chunks)
