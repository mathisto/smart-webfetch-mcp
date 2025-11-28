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


@dataclass
class Table:
    """Represents an extracted HTML table."""

    headers: list[str]
    rows: list[list[str]]
    caption: str | None


@dataclass
class Link:
    """Represents an extracted link from a page."""

    url: str
    text: str
    is_external: bool


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

    def extract_links(
        self,
        html: str,
        base_url: str,
        filter_pattern: str | None = None,
        external_only: bool = False,
    ) -> list[Link]:
        """Extract links from HTML content.

        Args:
            html: HTML content to parse
            base_url: Base URL for resolving relative links and determining externality
            filter_pattern: Optional regex pattern to filter link URLs
            external_only: If True, only return external links

        Returns:
            List of Link instances, deduplicated by URL
        """
        from urllib.parse import urljoin, urlparse

        soup = BeautifulSoup(html, "lxml")
        base_domain = urlparse(base_url).netloc.lower()

        # Patterns to filter out
        skip_prefixes = ("javascript:", "mailto:", "tel:", "#", "data:")

        seen_urls: set[str] = set()
        links: list[Link] = []

        # Compile filter pattern if provided
        filter_re = re.compile(filter_pattern) if filter_pattern else None

        for anchor in soup.find_all("a", href=True):
            href = anchor.get("href", "")
            if not href or not isinstance(href, str):
                continue

            # Skip filtered schemes
            href_lower = href.lower().strip()
            if any(href_lower.startswith(prefix) for prefix in skip_prefixes):
                continue

            # Resolve relative URLs
            absolute_url = urljoin(base_url, href)

            # Skip duplicates
            if absolute_url in seen_urls:
                continue
            seen_urls.add(absolute_url)

            # Determine if external
            link_domain = urlparse(absolute_url).netloc.lower()
            is_external = link_domain != base_domain

            # Apply external_only filter
            if external_only and not is_external:
                continue

            # Apply pattern filter
            if filter_re and not filter_re.search(absolute_url):
                continue

            # Extract link text
            text = anchor.get_text(strip=True)
            if not text:
                # Use title attribute or URL as fallback
                text = anchor.get("title", "") or absolute_url
                if isinstance(text, list):
                    text = text[0] if text else absolute_url

            links.append(Link(url=absolute_url, text=str(text), is_external=is_external))

        return links

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

    def extract_tables(self, html: str) -> list[Table]:
        """Extract all tables from HTML and convert to structured data.

        Handles:
        - thead/tbody structure
        - th/td cells
        - colspan (simplified: repeats cell content)
        - Optional caption element

        Args:
            html: HTML content to parse

        Returns:
            List of Table instances with headers, rows, and optional caption
        """
        soup = BeautifulSoup(html, "lxml")
        tables: list[Table] = []

        for table_elem in soup.find_all("table"):
            caption = None
            caption_elem = table_elem.find("caption")
            if caption_elem:
                caption = caption_elem.get_text(strip=True)

            headers: list[str] = []
            rows: list[list[str]] = []

            # Try to find headers in thead first
            thead = table_elem.find("thead")
            if thead:
                header_row = thead.find("tr")
                if header_row:
                    headers = self._extract_table_row_cells(header_row, is_header=True)

            # Process all rows (tbody or direct tr children)
            tbody = table_elem.find("tbody")
            row_container = tbody if tbody else table_elem

            for tr in row_container.find_all("tr", recursive=False):
                # If we don't have headers yet, check if first row has th cells
                if not headers:
                    th_cells = tr.find_all("th")
                    if th_cells:
                        headers = self._extract_table_row_cells(tr, is_header=True)
                        continue

                row_cells = self._extract_table_row_cells(tr, is_header=False)
                if row_cells:
                    rows.append(row_cells)

            # Only include tables that have some content
            if headers or rows:
                # If no headers found, use empty strings matching first row length
                if not headers and rows:
                    headers = [""] * len(rows[0])
                tables.append(Table(headers=headers, rows=rows, caption=caption))

        return tables

    def _extract_table_row_cells(self, tr: Tag, is_header: bool) -> list[str]:
        """Extract cell contents from a table row.

        Args:
            tr: Table row element
            is_header: If True, look for th cells; otherwise td cells

        Returns:
            List of cell text contents, with colspan cells repeated
        """
        cells: list[str] = []
        cell_tags = tr.find_all("th" if is_header else "td")

        # If looking for td but found none, also try th (some tables use th in body)
        if not cell_tags and not is_header:
            cell_tags = tr.find_all("th")

        for cell in cell_tags:
            text = cell.get_text(strip=True)
            # Handle colspan (simplified: repeat cell content)
            colspan = 1
            colspan_attr = cell.get("colspan")
            if colspan_attr:
                try:
                    colspan = int(str(colspan_attr))
                except (ValueError, TypeError):
                    colspan = 1
            cells.extend([text] * colspan)

        return cells

    def table_to_markdown(self, table: Table) -> str:
        """Convert a Table to markdown format.

        Args:
            table: Table instance to convert

        Returns:
            Markdown table string
        """
        lines: list[str] = []

        if table.caption:
            lines.append(f"**{table.caption}**\n")

        if not table.headers and not table.rows:
            return ""

        # Determine column count
        col_count = len(table.headers) if table.headers else len(table.rows[0])

        # Normalize headers
        headers = table.headers if table.headers else [""] * col_count

        # Escape pipe characters in cells
        def escape_cell(text: str) -> str:
            return text.replace("|", "\\|")

        # Header row
        header_line = "| " + " | ".join(escape_cell(h) for h in headers) + " |"
        lines.append(header_line)

        # Separator row
        separator = "| " + " | ".join("---" for _ in headers) + " |"
        lines.append(separator)

        # Data rows
        for row in table.rows:
            # Pad row to match header length if needed
            padded_row = row + [""] * (len(headers) - len(row))
            row_line = "| " + " | ".join(escape_cell(c) for c in padded_row[: len(headers)]) + " |"
            lines.append(row_line)

        return "\n".join(lines)

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
