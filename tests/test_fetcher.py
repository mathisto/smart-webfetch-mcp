"""Tests for the fetcher module."""

from smart_webfetch.fetcher import FetchResult, PrefetchResult, SmartFetcher


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
