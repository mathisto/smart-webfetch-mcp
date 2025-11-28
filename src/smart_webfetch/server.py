"""Smart WebFetch MCP Server - Context-aware web fetching."""

from __future__ import annotations

import asyncio
import json
import traceback
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from .extractors import ContentExtractor
from .fetcher import FetchError, PrefetchResult, SmartFetcher

server = Server("smart-webfetch")

# Global instances (initialized in main)
_fetcher: SmartFetcher | None = None
_extractor: ContentExtractor | None = None


def _get_fetcher() -> SmartFetcher:
    """Get the fetcher instance, raising if not initialized."""
    if _fetcher is None:
        raise RuntimeError("Server not initialized")
    return _fetcher


def _get_extractor() -> ContentExtractor:
    """Get the extractor instance, raising if not initialized."""
    if _extractor is None:
        raise RuntimeError("Server not initialized")
    return _extractor


def _validate_timeout(timeout: float | None) -> float | None:
    """Validate and clamp timeout value to allowed range (1-120 seconds)."""
    if timeout is None:
        return None
    return max(1.0, min(120.0, float(timeout)))


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List all available tools for smart web fetching."""
    return [
        Tool(
            name="web_preflight",
            description=(
                "Check page size and metadata before fetching. Returns estimated tokens, "
                "content type, and whether fetch is safe for context window. "
                "Always call this first to avoid context flooding."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to check"},
                    "timeout": {
                        "type": "number",
                        "description": "Request timeout in seconds (default 30, max 120)",
                        "default": 30,
                        "maximum": 120,
                    },
                },
                "required": ["url"],
            },
        ),
        Tool(
            name="web_smart_fetch",
            description=(
                "Fetch URL with automatic truncation if content exceeds token limit. "
                "Use web_preflight first to check size. Returns markdown content."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to fetch"},
                    "max_tokens": {
                        "type": "integer",
                        "description": "Maximum tokens to return (default 8000)",
                        "default": 8000,
                    },
                    "strategy": {
                        "type": "string",
                        "enum": ["auto", "truncate"],
                        "description": "How to handle large content",
                        "default": "auto",
                    },
                    "timeout": {
                        "type": "number",
                        "description": "Request timeout in seconds (default 30, max 120)",
                        "default": 30,
                        "maximum": 120,
                    },
                },
                "required": ["url"],
            },
        ),
        Tool(
            name="web_fetch_code",
            description=(
                "Extract only code blocks from a page. Ideal for documentation pages "
                "with code examples. Returns code blocks with language annotations."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to extract code from"},
                    "timeout": {
                        "type": "number",
                        "description": "Request timeout in seconds (default 30, max 120)",
                        "default": 30,
                        "maximum": 120,
                    },
                },
                "required": ["url"],
            },
        ),
        Tool(
            name="web_fetch_section",
            description=(
                "Fetch only content under a specific heading. Case-insensitive heading match. "
                "Use when you only need a specific part of a large document."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to fetch from"},
                    "heading": {
                        "type": "string",
                        "description": "Heading text to find (e.g., 'Installation')",
                    },
                    "timeout": {
                        "type": "number",
                        "description": "Request timeout in seconds (default 30, max 120)",
                        "default": 30,
                        "maximum": 120,
                    },
                },
                "required": ["url", "heading"],
            },
        ),
        Tool(
            name="web_fetch_chunked",
            description=(
                "Fetch large documents in chunks. Use for paginated reading of large docs. "
                "Returns chunk content plus metadata about total chunks available."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to fetch"},
                    "chunk": {
                        "type": "integer",
                        "description": "Chunk index (0-based)",
                        "default": 0,
                    },
                    "chunk_size": {
                        "type": "integer",
                        "description": "Tokens per chunk",
                        "default": 4000,
                    },
                    "timeout": {
                        "type": "number",
                        "description": "Request timeout in seconds (default 30, max 120)",
                        "default": 30,
                        "maximum": 120,
                    },
                },
                "required": ["url"],
            },
        ),
        Tool(
            name="web_fetch_links",
            description=(
                "Extract all links from a page. Returns a markdown list of links with text "
                "and URL. Optionally filter by pattern or external links only."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to extract links from"},
                    "filter_pattern": {
                        "type": "string",
                        "description": "Regex pattern to filter link URLs (e.g., '/docs/')",
                    },
                    "external_only": {
                        "type": "boolean",
                        "description": "Only return external links",
                        "default": False,
                    },
                    "timeout": {
                        "type": "number",
                        "description": "Request timeout in seconds (default 30, max 120)",
                        "default": 30,
                        "maximum": 120,
                    },
                },
                "required": ["url"],
            },
        ),
        Tool(
            name="web_fetch_tables",
            description=(
                "Extract tables from a page and return as markdown tables. "
                "Handles thead/tbody, th/td cells, colspan, and captions."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to extract tables from"},
                    "table_index": {
                        "type": "integer",
                        "description": "Specific table index to return (0-based). Returns all tables if not specified.",
                    },
                    "timeout": {
                        "type": "number",
                        "description": "Request timeout in seconds (default 30, max 120)",
                        "default": 30,
                        "maximum": 120,
                    },
                },
                "required": ["url"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Handle tool calls for all web fetching tools."""
    if _fetcher is None or _extractor is None:
        return [
            TextContent(
                type="text",
                text=json.dumps(
                    {"error": "Server not initialized. Please restart the MCP server."},
                    indent=2,
                ),
            )
        ]

    try:
        if name == "web_preflight":
            return await handle_preflight(arguments)
        elif name == "web_smart_fetch":
            return await handle_smart_fetch(arguments)
        elif name == "web_fetch_code":
            return await handle_fetch_code(arguments)
        elif name == "web_fetch_section":
            return await handle_fetch_section(arguments)
        elif name == "web_fetch_chunked":
            return await handle_fetch_chunked(arguments)
        elif name == "web_fetch_links":
            return await handle_fetch_links(arguments)
        elif name == "web_fetch_tables":
            return await handle_fetch_tables(arguments)
        else:
            return [
                TextContent(
                    type="text",
                    text=json.dumps({"error": f"Unknown tool: {name}"}, indent=2),
                )
            ]
    except Exception as e:
        error_detail = {
            "error": str(e),
            "tool": name,
            "traceback": traceback.format_exc(),
        }
        return [TextContent(type="text", text=json.dumps(error_detail, indent=2))]


async def handle_preflight(arguments: dict[str, Any]) -> list[TextContent]:
    """Handle web_preflight tool - check page size before fetching."""
    fetcher = _get_fetcher()

    url = arguments.get("url")
    if not url:
        return [
            TextContent(
                type="text",
                text=json.dumps({"error": "Missing required parameter: url"}, indent=2),
            )
        ]

    timeout = _validate_timeout(arguments.get("timeout"))

    try:
        result = await fetcher.prefetch(url, timeout=timeout)
    except FetchError as e:
        return [
            TextContent(
                type="text",
                text=json.dumps(
                    {"error": str(e), "url": url, "status_code": e.status_code}, indent=2
                ),
            )
        ]

    # Build preflight response
    response = {
        "url": result.url,
        "status_code": result.status_code,
        "content_type": result.content_type,
        "content_length_bytes": result.content_length,
        "estimated_tokens": result.estimated_tokens,
        "size_category": result.size_category,
        "is_html": result.is_html,
        "title": result.title,
        "safe_for_context": result.estimated_tokens <= 8000,
        "recommendation": _get_preflight_recommendation(result),
    }

    return [TextContent(type="text", text=json.dumps(response, indent=2))]


def _get_preflight_recommendation(result: PrefetchResult) -> str:
    """Generate a recommendation based on preflight results."""
    if result.status_code >= 400:
        return f"Page returned error status {result.status_code}. Cannot fetch."

    if not result.is_html:
        return f"Non-HTML content ({result.content_type}). Consider using a specialized tool."

    tokens = result.estimated_tokens

    if tokens <= 4000:
        return "Small page. Safe to fetch with web_smart_fetch."
    elif tokens <= 8000:
        return "Medium page. Safe to fetch with web_smart_fetch (default limit)."
    elif tokens <= 16000:
        return (
            "Large page. Use web_smart_fetch with truncation, "
            "web_fetch_section for specific content, or web_fetch_chunked for pagination."
        )
    else:
        return (
            f"Very large page (~{tokens:,} tokens). "
            "Strongly recommend web_fetch_section or web_fetch_chunked. "
            "Full fetch will flood context window."
        )


# Cache for fetched HTML content (avoid re-fetching for section/code extraction)
_html_cache: dict[str, tuple[str, str, str]] = {}  # url -> (html, content_type, title)


async def handle_smart_fetch(arguments: dict[str, Any]) -> list[TextContent]:
    """Handle web_smart_fetch tool - fetch with automatic truncation."""
    fetcher = _get_fetcher()
    extractor = _get_extractor()

    url = arguments.get("url")
    if not url:
        return [
            TextContent(
                type="text",
                text=json.dumps({"error": "Missing required parameter: url"}, indent=2),
            )
        ]

    max_tokens = arguments.get("max_tokens", 8000)
    strategy = arguments.get("strategy", "auto")
    timeout = _validate_timeout(arguments.get("timeout"))

    try:
        # Use smart_fetch which handles truncation internally
        result = await fetcher.smart_fetch(
            url, max_tokens=max_tokens, strategy=strategy, timeout=timeout
        )
    except FetchError as e:
        return [
            TextContent(
                type="text",
                text=json.dumps(
                    {"error": str(e), "url": url, "status_code": e.status_code}, indent=2
                ),
            )
        ]

    # Convert HTML to markdown if it's HTML content
    content = result.content
    if "<" in content[:100]:  # Likely HTML
        content = extractor.html_to_markdown(content)

    # Build header with metadata
    header = "# Fetched Content\n\n"
    header += f"**Source:** {result.url}\n"
    header += f"**Tokens:** {result.token_count:,}"

    if result.was_truncated:
        header += f" (truncated from ~{result.original_tokens:,})\n"
        header += f"**Strategy:** {result.strategy_used}\n"
        header += "**Note:** Content truncated. Use web_fetch_section or "
        header += "web_fetch_chunked for complete content.\n"
    else:
        header += "\n"

    header += "\n---\n\n"

    return [TextContent(type="text", text=header + content)]


async def handle_fetch_code(arguments: dict[str, Any]) -> list[TextContent]:
    """Handle web_fetch_code tool - extract only code blocks."""
    fetcher = _get_fetcher()
    extractor = _get_extractor()

    url = arguments.get("url")
    if not url:
        return [
            TextContent(
                type="text",
                text=json.dumps({"error": "Missing required parameter: url"}, indent=2),
            )
        ]

    timeout = _validate_timeout(arguments.get("timeout"))

    try:
        # Fetch the page (no truncation needed, we'll extract just code)
        html_content, content_type = await fetcher.fetch_raw(url, timeout=timeout)
    except FetchError as e:
        return [
            TextContent(
                type="text",
                text=json.dumps(
                    {"error": str(e), "url": url, "status_code": e.status_code}, indent=2
                ),
            )
        ]

    # Extract code blocks
    code_blocks = extractor.extract_code_blocks(html_content)

    if not code_blocks:
        return [
            TextContent(
                type="text",
                text=f"# No Code Blocks Found\n\n**Source:** {url}\n\n"
                "No code blocks were detected on this page.",
            )
        ]

    # Format output
    output = "# Code Blocks\n\n"
    output += f"**Source:** {url}\n"
    output += f"**Found:** {len(code_blocks)} code block(s)\n\n---\n\n"

    for i, block in enumerate(code_blocks, 1):
        lang_label = block.language or "text"
        output += f"## Code Block {i}"
        if block.context:
            output += f" - {block.context}"
        output += "\n\n"
        output += f"```{lang_label}\n{block.content}\n```\n\n"

    total_tokens = fetcher.count_tokens(output)
    output = f"<!-- Total tokens: {total_tokens:,} -->\n\n" + output

    return [TextContent(type="text", text=output)]


async def handle_fetch_section(arguments: dict[str, Any]) -> list[TextContent]:
    """Handle web_fetch_section tool - fetch content under specific heading."""
    fetcher = _get_fetcher()
    extractor = _get_extractor()

    url = arguments.get("url")
    heading = arguments.get("heading")

    if not url:
        return [
            TextContent(
                type="text",
                text=json.dumps({"error": "Missing required parameter: url"}, indent=2),
            )
        ]
    if not heading:
        return [
            TextContent(
                type="text",
                text=json.dumps({"error": "Missing required parameter: heading"}, indent=2),
            )
        ]

    timeout = _validate_timeout(arguments.get("timeout"))

    try:
        html_content, content_type = await fetcher.fetch_raw(url, timeout=timeout)
    except FetchError as e:
        return [
            TextContent(
                type="text",
                text=json.dumps(
                    {"error": str(e), "url": url, "status_code": e.status_code}, indent=2
                ),
            )
        ]

    # Try to find exact section
    section = extractor.extract_section(html_content, heading)

    if not section:
        # List available sections to help user
        all_sections = extractor.extract_all_sections(html_content)
        available = [s.heading for s in all_sections[:20]]  # Limit to first 20
        return [
            TextContent(
                type="text",
                text=json.dumps(
                    {
                        "error": f"Section '{heading}' not found",
                        "url": url,
                        "available_sections": available,
                        "hint": "Try one of the available sections listed above",
                    },
                    indent=2,
                ),
            )
        ]

    # Format output
    output = f"# {section.heading}\n\n"
    output += f"**Source:** {url}\n"
    output += f"**Section Level:** H{section.level}\n\n---\n\n"
    output += section.content

    total_tokens = fetcher.count_tokens(output)
    output = f"<!-- Total tokens: {total_tokens:,} -->\n\n" + output

    return [TextContent(type="text", text=output)]


async def handle_fetch_chunked(arguments: dict[str, Any]) -> list[TextContent]:
    """Handle web_fetch_chunked tool - paginated fetch for large documents."""
    fetcher = _get_fetcher()
    extractor = _get_extractor()

    url = arguments.get("url")
    if not url:
        return [
            TextContent(
                type="text",
                text=json.dumps({"error": "Missing required parameter: url"}, indent=2),
            )
        ]

    chunk_index = arguments.get("chunk", 0)
    chunk_size = arguments.get("chunk_size", 4000)

    # Validate inputs
    if chunk_index < 0:
        return [
            TextContent(
                type="text",
                text=json.dumps({"error": "chunk must be >= 0"}, indent=2),
            )
        ]
    if chunk_size < 500:
        return [
            TextContent(
                type="text",
                text=json.dumps({"error": "chunk_size must be >= 500 tokens"}, indent=2),
            )
        ]

    timeout = _validate_timeout(arguments.get("timeout"))

    try:
        html_content, content_type = await fetcher.fetch_raw(url, timeout=timeout)
    except FetchError as e:
        return [
            TextContent(
                type="text",
                text=json.dumps(
                    {"error": str(e), "url": url, "status_code": e.status_code}, indent=2
                ),
            )
        ]

    # Convert to markdown and chunk
    markdown = extractor.html_to_markdown(html_content)
    chunks = extractor.chunk_content(markdown, chunk_size)

    total_chunks = len(chunks)

    if total_chunks == 0:
        return [
            TextContent(
                type="text",
                text=json.dumps({"error": "No content to chunk", "url": url}, indent=2),
            )
        ]

    if chunk_index >= total_chunks:
        return [
            TextContent(
                type="text",
                text=json.dumps(
                    {
                        "error": f"Chunk {chunk_index} does not exist",
                        "total_chunks": total_chunks,
                        "valid_range": f"0-{total_chunks - 1}",
                    },
                    indent=2,
                ),
            )
        ]

    current_chunk = chunks[chunk_index]

    # Format output with navigation metadata
    output = f"# Document - Chunk {chunk_index + 1}/{total_chunks}\n\n"
    output += f"**Source:** {url}\n"
    output += f"**Chunk:** {chunk_index + 1} of {total_chunks}\n"
    output += f"**Tokens in chunk:** {current_chunk.token_count:,}\n"

    if chunk_index > 0:
        output += f"**Previous:** chunk {chunk_index - 1}\n"
    if chunk_index < total_chunks - 1:
        output += f"**Next:** chunk {chunk_index + 1}\n"

    output += "\n---\n\n"
    output += current_chunk.content

    # Add navigation footer
    output += "\n\n---\n\n"
    output += f"*Chunk {chunk_index + 1} of {total_chunks}. "
    if chunk_index < total_chunks - 1:
        output += f"Call with chunk={chunk_index + 1} for next chunk.*"
    else:
        output += "This is the last chunk.*"

    return [TextContent(type="text", text=output)]


async def handle_fetch_links(arguments: dict[str, Any]) -> list[TextContent]:
    """Handle web_fetch_links tool - extract links from a page."""
    fetcher = _get_fetcher()
    extractor = _get_extractor()

    url = arguments.get("url")
    if not url:
        return [
            TextContent(
                type="text",
                text=json.dumps({"error": "Missing required parameter: url"}, indent=2),
            )
        ]

    filter_pattern = arguments.get("filter_pattern")
    external_only = arguments.get("external_only", False)
    timeout = _validate_timeout(arguments.get("timeout"))

    try:
        html_content, content_type = await fetcher.fetch_raw(url, timeout=timeout)
    except FetchError as e:
        return [
            TextContent(
                type="text",
                text=json.dumps(
                    {"error": str(e), "url": url, "status_code": e.status_code}, indent=2
                ),
            )
        ]

    # Extract links
    links = extractor.extract_links(
        html_content,
        base_url=url,
        filter_pattern=filter_pattern,
        external_only=external_only,
    )

    if not links:
        msg = "# No Links Found\n\n"
        msg += f"**Source:** {url}\n\n"
        if filter_pattern:
            msg += f"**Filter:** `{filter_pattern}`\n"
        if external_only:
            msg += "**External only:** Yes\n"
        msg += "\nNo matching links were found on this page."
        return [TextContent(type="text", text=msg)]

    # Format output as markdown list
    output = "# Links\n\n"
    output += f"**Source:** {url}\n"
    output += f"**Found:** {len(links)} link(s)\n"
    if filter_pattern:
        output += f"**Filter:** `{filter_pattern}`\n"
    if external_only:
        output += "**External only:** Yes\n"
    output += "\n---\n\n"

    for link in links:
        external_marker = " (external)" if link.is_external else ""
        output += f"- [{link.text}]({link.url}){external_marker}\n"

    total_tokens = fetcher.count_tokens(output)
    output = f"<!-- Total tokens: {total_tokens:,} -->\n\n" + output

    return [TextContent(type="text", text=output)]


async def handle_fetch_tables(arguments: dict[str, Any]) -> list[TextContent]:
    """Handle web_fetch_tables tool - extract tables from a page."""
    fetcher = _get_fetcher()
    extractor = _get_extractor()

    url = arguments.get("url")
    if not url:
        return [
            TextContent(
                type="text",
                text=json.dumps({"error": "Missing required parameter: url"}, indent=2),
            )
        ]

    table_index = arguments.get("table_index")
    timeout = _validate_timeout(arguments.get("timeout"))

    try:
        html_content, content_type = await fetcher.fetch_raw(url, timeout=timeout)
    except FetchError as e:
        return [
            TextContent(
                type="text",
                text=json.dumps(
                    {"error": str(e), "url": url, "status_code": e.status_code}, indent=2
                ),
            )
        ]

    # Extract tables
    tables = extractor.extract_tables(html_content)

    if not tables:
        return [
            TextContent(
                type="text",
                text=f"# No Tables Found\n\n**Source:** {url}\n\n"
                "No tables were detected on this page.",
            )
        ]

    # If specific table index requested
    if table_index is not None:
        if table_index < 0 or table_index >= len(tables):
            return [
                TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "error": f"Table index {table_index} out of range",
                            "total_tables": len(tables),
                            "valid_range": f"0-{len(tables) - 1}",
                        },
                        indent=2,
                    ),
                )
            ]
        tables = [tables[table_index]]

    # Format output
    output = "# Tables\n\n"
    output += f"**Source:** {url}\n"
    output += f"**Found:** {len(tables)} table(s)"
    if table_index is not None:
        output += f" (showing table {table_index})"
    output += "\n\n---\n\n"

    for i, table in enumerate(tables):
        if len(tables) > 1:
            output += f"## Table {i + 1}\n\n"
        output += extractor.table_to_markdown(table)
        output += "\n\n"

    total_tokens = fetcher.count_tokens(output)
    output = f"<!-- Total tokens: {total_tokens:,} -->\n\n" + output

    return [TextContent(type="text", text=output)]


def main() -> None:
    """Main entry point for the MCP server."""
    global _fetcher, _extractor

    # Import here to avoid circular imports and allow TYPE_CHECKING guard above
    from .extractors import ContentExtractor
    from .fetcher import SmartFetcher

    # Initialize global instances
    _fetcher = SmartFetcher()
    _extractor = ContentExtractor(_fetcher.count_tokens)

    async def run() -> None:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options(),
            )

    asyncio.run(run())


if __name__ == "__main__":
    main()
