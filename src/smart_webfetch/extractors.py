"""Content extraction utilities for code blocks, sections, and chunking."""

import re
from collections.abc import Callable
from dataclasses import dataclass

from bs4 import BeautifulSoup, NavigableString, Tag
from markdownify import markdownify as md


@dataclass
class CodeBlock:
    """Represents an extracted code block with optional context."""

    language: str | None
    content: str
    context: str | None  # Surrounding heading/description


@dataclass
class Section:
    """Represents a document section with heading and content."""

    heading: str
    level: int  # h1=1, h2=2, etc
    content: str


@dataclass
class Chunk:
    """Represents a content chunk for token-limited processing."""

    index: int
    total_chunks: int
    content: str
    token_count: int


class ContentExtractor:
    """Extract and process content from HTML documents."""

    # Regex for markdown fenced code blocks
    FENCED_CODE_PATTERN = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL | re.MULTILINE)

    # Heading tags in order of level
    HEADING_TAGS = ["h1", "h2", "h3", "h4", "h5", "h6"]

    def __init__(self, token_counter: Callable[[str], int]):
        """Initialize with a token counting function.

        Args:
            token_counter: A callable that counts tokens in text.
                           Typically tiktoken.encoding_for_model(...).encode
        """
        self.count_tokens = token_counter

    def extract_code_blocks(self, html: str) -> list[CodeBlock]:
        """Extract all code blocks from HTML.

        Looks for:
        - <pre><code> blocks
        - <code> blocks with class containing language
        - Markdown fenced code blocks if content is markdown

        Args:
            html: HTML content to parse

        Returns:
            List of CodeBlock instances with language, content, and context
        """
        blocks: list[CodeBlock] = []
        soup = BeautifulSoup(html, "lxml")

        # Find all <pre> elements (most code blocks use pre)
        for pre in soup.find_all("pre"):
            code_elem = pre.find("code")
            if code_elem:
                language = self._extract_language_from_element(code_elem)
                content = self._get_text_content(code_elem)
            else:
                # <pre> without nested <code>
                language = self._extract_language_from_element(pre)
                content = self._get_text_content(pre)

            if content.strip():
                context = self._find_code_context(pre)
                blocks.append(
                    CodeBlock(language=language, content=content.strip(), context=context)
                )

        # Find standalone <code> blocks not inside <pre>
        for code in soup.find_all("code"):
            # Skip if parent is <pre> (already handled)
            if code.find_parent("pre"):
                continue

            # Only consider multi-line code blocks or those with language class
            content = self._get_text_content(code)
            language = self._extract_language_from_element(code)

            # Skip inline code (single line, no language)
            if "\n" not in content and not language:
                continue

            if content.strip():
                context = self._find_code_context(code)
                blocks.append(
                    CodeBlock(language=language, content=content.strip(), context=context)
                )

        # Also check for markdown fenced code blocks in text content
        text_content = soup.get_text()
        for match in self.FENCED_CODE_PATTERN.finditer(text_content):
            lang = match.group(1) or None
            code_content = match.group(2).strip()

            # Avoid duplicates from already-parsed HTML
            if code_content and not self._is_duplicate_block(code_content, blocks):
                blocks.append(
                    CodeBlock(
                        language=lang,
                        content=code_content,
                        context=None,  # Hard to determine context from regex match
                    )
                )

        return blocks

    def _extract_language_from_element(self, elem: Tag) -> str | None:
        """Extract programming language from element classes.

        Common patterns:
        - class="language-python"
        - class="lang-python"
        - class="python"
        - class="highlight-python"
        - data-language="python"
        """
        # Check data attributes first
        if elem.get("data-language"):
            return elem["data-language"]

        classes = elem.get("class", [])
        if isinstance(classes, str):
            classes = classes.split()

        for cls in classes:
            # language-xxx or lang-xxx
            if cls.startswith("language-") or cls.startswith("lang-"):
                return cls.split("-", 1)[1]

            # highlight-xxx (Sphinx, etc.)
            if cls.startswith("highlight-"):
                return cls.split("-", 1)[1]

            # Common language names as direct classes
            common_langs = {
                "python",
                "javascript",
                "typescript",
                "java",
                "c",
                "cpp",
                "csharp",
                "go",
                "rust",
                "ruby",
                "php",
                "swift",
                "kotlin",
                "scala",
                "sql",
                "bash",
                "shell",
                "sh",
                "zsh",
                "fish",
                "html",
                "css",
                "scss",
                "sass",
                "less",
                "json",
                "yaml",
                "xml",
                "markdown",
                "md",
                "plaintext",
                "text",
                "mojo",
            }
            if cls.lower() in common_langs:
                return cls.lower()

        return None

    def _get_text_content(self, elem: Tag) -> str:
        """Get text content preserving structure within code blocks."""
        # Use get_text with separator to handle br tags
        return elem.get_text(separator="\n")

    def _find_code_context(self, code_elem: Tag) -> str | None:
        """Find contextual information for a code block.

        Looks for:
        - Previous heading
        - Previous paragraph
        - Figcaption if in figure
        """
        # Check if inside a figure with figcaption
        figure = code_elem.find_parent("figure")
        if figure:
            figcaption = figure.find("figcaption")
            if figcaption:
                return figcaption.get_text(strip=True)

        # Look for previous sibling heading or paragraph
        for sibling in code_elem.find_all_previous(limit=5):
            if isinstance(sibling, Tag):
                if sibling.name in self.HEADING_TAGS:
                    return sibling.get_text(strip=True)
                if sibling.name == "p":
                    text = sibling.get_text(strip=True)
                    # Only use if it seems like a description
                    if len(text) < 200:
                        return text

        return None

    def _is_duplicate_block(self, content: str, blocks: list[CodeBlock]) -> bool:
        """Check if content already exists in extracted blocks."""
        normalized = content.strip()
        return any(b.content.strip() == normalized for b in blocks)

    def extract_section(self, html: str, heading: str) -> Section | None:
        """Extract content under a specific heading.

        Finds heading by text match (case-insensitive).
        Returns content until next heading of same or higher level.

        Args:
            html: HTML content to parse
            heading: Heading text to find (case-insensitive)

        Returns:
            Section with heading, level, and content, or None if not found
        """
        soup = BeautifulSoup(html, "lxml")
        heading_lower = heading.lower().strip()

        # Find all headings
        for tag in self.HEADING_TAGS:
            for h in soup.find_all(tag):
                h_text = h.get_text(strip=True)
                if h_text.lower() == heading_lower:
                    level = int(tag[1])  # h1 -> 1, h2 -> 2, etc.
                    content = self._extract_content_until_heading(h, level)
                    return Section(heading=h_text, level=level, content=content)

        return None

    def _extract_content_until_heading(self, heading_elem: Tag, level: int) -> str:
        """Extract content from heading until next same-level or higher heading."""
        content_parts: list[str] = []
        stop_tags = {f"h{i}" for i in range(1, level + 1)}

        for sibling in heading_elem.find_next_siblings():
            if isinstance(sibling, Tag):
                if sibling.name in stop_tags:
                    break
                # Convert element to markdown for cleaner output
                content_parts.append(md(str(sibling), strip=["a"]))
            elif isinstance(sibling, NavigableString):
                text = str(sibling).strip()
                if text:
                    content_parts.append(text)

        return self.clean_text("\n".join(content_parts))

    def extract_all_sections(self, html: str) -> list[Section]:
        """Extract all sections with their headings.

        Args:
            html: HTML content to parse

        Returns:
            List of Section instances ordered by document position
        """
        soup = BeautifulSoup(html, "lxml")
        sections: list[Section] = []

        # Find all headings in document order
        all_headings = soup.find_all(self.HEADING_TAGS)

        for i, h in enumerate(all_headings):
            level = int(h.name[1])
            heading_text = h.get_text(strip=True)

            # Find content until next heading (any level)
            content_parts: list[str] = []
            for sibling in h.find_next_siblings():
                if isinstance(sibling, Tag):
                    if sibling.name in self.HEADING_TAGS:
                        break
                    content_parts.append(md(str(sibling), strip=["a"]))
                elif isinstance(sibling, NavigableString):
                    text = str(sibling).strip()
                    if text:
                        content_parts.append(text)

            content = self.clean_text("\n".join(content_parts))

            sections.append(Section(heading=heading_text, level=level, content=content))

        return sections

    def chunk_content(
        self, content: str, chunk_size: int = 4000, overlap: int = 200
    ) -> list[Chunk]:
        """Split content into token-sized chunks with overlap.

        Tries to break at paragraph boundaries when possible.

        Args:
            content: Text content to chunk
            chunk_size: Target tokens per chunk
            overlap: Number of overlapping tokens between chunks

        Returns:
            List of Chunk instances with index, content, and token count
        """
        if not content.strip():
            return []

        # Count total tokens
        total_tokens = self.count_tokens(content)

        # If content fits in one chunk, return as-is
        if total_tokens <= chunk_size:
            return [Chunk(index=0, total_chunks=1, content=content, token_count=total_tokens)]

        # Split into paragraphs for smarter chunking
        paragraphs = self._split_into_paragraphs(content)
        chunks: list[Chunk] = []
        current_chunk: list[str] = []
        current_tokens = 0

        for para in paragraphs:
            para_tokens = self.count_tokens(para)

            # If single paragraph exceeds chunk size, split it
            if para_tokens > chunk_size:
                # Flush current chunk first
                if current_chunk:
                    chunk_content = "\n\n".join(current_chunk)
                    chunks.append(self._create_chunk_placeholder(chunk_content, len(chunks)))
                    current_chunk = []
                    current_tokens = 0

                # Split large paragraph by sentences
                sub_chunks = self._split_large_paragraph(para, chunk_size, overlap)
                for sub in sub_chunks:
                    chunks.append(self._create_chunk_placeholder(sub, len(chunks)))
                continue

            # Check if adding this paragraph would exceed limit
            if current_tokens + para_tokens > chunk_size:
                # Save current chunk
                if current_chunk:
                    chunk_content = "\n\n".join(current_chunk)
                    chunks.append(self._create_chunk_placeholder(chunk_content, len(chunks)))

                    # Start new chunk with overlap from previous
                    if overlap > 0 and current_chunk:
                        overlap_text = self._get_overlap_text("\n\n".join(current_chunk), overlap)
                        current_chunk = [overlap_text, para] if overlap_text else [para]
                        current_tokens = self.count_tokens("\n\n".join(current_chunk))
                    else:
                        current_chunk = [para]
                        current_tokens = para_tokens
                else:
                    current_chunk = [para]
                    current_tokens = para_tokens
            else:
                current_chunk.append(para)
                current_tokens += para_tokens

        # Don't forget the last chunk
        if current_chunk:
            chunk_content = "\n\n".join(current_chunk)
            chunks.append(self._create_chunk_placeholder(chunk_content, len(chunks)))

        # Update total_chunks in all chunks
        total = len(chunks)
        return [
            Chunk(index=c.index, total_chunks=total, content=c.content, token_count=c.token_count)
            for c in chunks
        ]

    def _split_into_paragraphs(self, content: str) -> list[str]:
        """Split content into paragraphs."""
        # Split on double newlines (paragraph breaks)
        paragraphs = re.split(r"\n\s*\n", content)
        return [p.strip() for p in paragraphs if p.strip()]

    def _split_large_paragraph(self, para: str, chunk_size: int, overlap: int) -> list[str]:
        """Split a large paragraph that exceeds chunk size."""
        # Try splitting by sentences first
        sentences = re.split(r"(?<=[.!?])\s+", para)

        if len(sentences) <= 1:
            # Fall back to word-level splitting
            return self._split_by_words(para, chunk_size, overlap)

        chunks: list[str] = []
        current: list[str] = []
        current_tokens = 0

        for sentence in sentences:
            sent_tokens = self.count_tokens(sentence)

            if sent_tokens > chunk_size:
                # Even single sentence is too big, split by words
                if current:
                    chunks.append(" ".join(current))
                    current = []
                    current_tokens = 0
                chunks.extend(self._split_by_words(sentence, chunk_size, overlap))
                continue

            if current_tokens + sent_tokens > chunk_size:
                if current:
                    chunks.append(" ".join(current))
                    # Add overlap
                    overlap_text = self._get_overlap_text(" ".join(current), overlap)
                    current = [overlap_text, sentence] if overlap_text else [sentence]
                    current_tokens = self.count_tokens(" ".join(current))
                else:
                    current = [sentence]
                    current_tokens = sent_tokens
            else:
                current.append(sentence)
                current_tokens += sent_tokens

        if current:
            chunks.append(" ".join(current))

        return chunks

    def _split_by_words(self, text: str, chunk_size: int, overlap: int) -> list[str]:
        """Split text by words when sentence splitting isn't enough."""
        words = text.split()
        chunks: list[str] = []
        current: list[str] = []
        current_tokens = 0

        for word in words:
            word_tokens = self.count_tokens(word + " ")

            if current_tokens + word_tokens > chunk_size:
                if current:
                    chunk_text = " ".join(current)
                    chunks.append(chunk_text)

                    # Calculate overlap words
                    overlap_words = self._get_overlap_words(current, overlap)
                    current = overlap_words + [word]
                    current_tokens = self.count_tokens(" ".join(current))
                else:
                    current = [word]
                    current_tokens = word_tokens
            else:
                current.append(word)
                current_tokens += word_tokens

        if current:
            chunks.append(" ".join(current))

        return chunks

    def _get_overlap_text(self, text: str, overlap_tokens: int) -> str:
        """Get the last N tokens worth of text for overlap."""
        words = text.split()
        overlap_words: list[str] = []
        tokens = 0

        for word in reversed(words):
            word_tokens = self.count_tokens(word + " ")
            if tokens + word_tokens > overlap_tokens:
                break
            overlap_words.insert(0, word)
            tokens += word_tokens

        return " ".join(overlap_words)

    def _get_overlap_words(self, words: list[str], overlap_tokens: int) -> list[str]:
        """Get words from end of list that fit in overlap token count."""
        overlap_words: list[str] = []
        tokens = 0

        for word in reversed(words):
            word_tokens = self.count_tokens(word + " ")
            if tokens + word_tokens > overlap_tokens:
                break
            overlap_words.insert(0, word)
            tokens += word_tokens

        return overlap_words

    def _create_chunk_placeholder(self, content: str, index: int) -> Chunk:
        """Create a chunk with placeholder total_chunks (updated later)."""
        return Chunk(
            index=index,
            total_chunks=0,  # Updated after all chunks created
            content=content,
            token_count=self.count_tokens(content),
        )

    def html_to_markdown(self, html: str) -> str:
        """Convert HTML to clean markdown.

        Args:
            html: HTML content to convert

        Returns:
            Cleaned markdown string
        """
        # Use markdownify for conversion
        markdown = md(
            html,
            heading_style="atx",  # Use # style headings
            bullets="-",  # Use - for lists
            strip=["script", "style", "nav", "footer", "header", "aside"],
            code_language_callback=self._detect_code_language,
        )

        return self.clean_text(markdown)

    def _detect_code_language(self, elem: Tag) -> str | None:
        """Callback for markdownify to detect code language."""
        return self._extract_language_from_element(elem)

    def clean_text(self, text: str) -> str:
        """Remove excessive whitespace, normalize newlines.

        Args:
            text: Text to clean

        Returns:
            Cleaned text with normalized whitespace
        """
        if not text:
            return ""

        # Normalize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Remove trailing whitespace from lines
        lines = [line.rstrip() for line in text.split("\n")]
        text = "\n".join(lines)

        # Collapse multiple blank lines into at most two
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Remove leading/trailing whitespace
        text = text.strip()

        return text


def create_tiktoken_counter(model: str = "gpt-4") -> Callable[[str], int]:
    """Create a token counter function using tiktoken.

    Args:
        model: Model name for tokenizer selection

    Returns:
        Function that counts tokens in a string
    """
    import tiktoken

    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # Fall back to cl100k_base for unknown models
        encoding = tiktoken.get_encoding("cl100k_base")

    def count_tokens(text: str) -> int:
        return len(encoding.encode(text))

    return count_tokens
