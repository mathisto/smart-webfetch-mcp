# Smart WebFetch MCP Server

Context-aware web fetching for LLMs. Prevents context window flooding by checking page size before fetching and providing surgical extraction tools.

## The Problem

Standard web fetch tools dump entire pages into the context window, often:
- Exceeding token limits
- Wasting context on navigation, footers, ads
- Flooding the model with irrelevant content

## The Solution

Smart WebFetch provides 5 tools for intelligent web fetching:

| Tool | Purpose |
|------|---------|
| `web_preflight` | Check page size before fetching |
| `web_smart_fetch` | Fetch with automatic truncation |
| `web_fetch_code` | Extract only code blocks |
| `web_fetch_section` | Fetch specific heading/section |
| `web_fetch_chunked` | Paginated fetching for large docs |

## Installation

```bash
# Install from PyPI
pip install smart-webfetch-mcp

# Or with uvx (recommended for MCP)
uvx smart-webfetch-mcp
```

## Configuration

### OpenCode

Add to your `opencode.json`:

```json
{
  "mcp": {
    "smart-webfetch": {
      "type": "local",
      "command": ["uvx", "smart-webfetch-mcp"],
      "enabled": true
    }
  }
}
```

### Claude Desktop

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "smart-webfetch": {
      "command": "uvx",
      "args": ["smart-webfetch-mcp"]
    }
  }
}
```

## Usage Examples

### Check before fetching

```
Use web_preflight to check https://docs.python.org/3/library/asyncio.html
```

Response:
```json
{
  "url": "https://docs.python.org/3/library/asyncio.html",
  "estimated_tokens": 45000,
  "safe_for_context": false,
  "recommendation": "Very large page (~45,000 tokens). Use web_fetch_section or web_fetch_chunked."
}
```

### Fetch with automatic truncation

```
Use web_smart_fetch on https://example.com/docs with max_tokens=4000
```

### Extract only code examples

```
Use web_fetch_code on https://docs.python.org/3/library/asyncio-task.html
```

### Get specific section

```
Use web_fetch_section on https://docs.python.org/3/library/asyncio.html 
with heading="Running an asyncio Program"
```

### Paginated reading

```
Use web_fetch_chunked on https://large-docs.com/api with chunk=0, chunk_size=4000
```

Then continue with `chunk=1`, `chunk=2`, etc.

## Tool Reference

### web_preflight

Check page metadata before fetching.

**Parameters:**
- `url` (required): URL to check

**Returns:**
- `estimated_tokens`: Approximate token count
- `content_type`: MIME type
- `is_html`: Whether content is HTML
- `title`: Page title (if HTML)
- `safe_for_context`: Boolean (true if < 8000 tokens)
- `recommendation`: Human-readable advice

### web_smart_fetch

Fetch with automatic truncation for large pages.

**Parameters:**
- `url` (required): URL to fetch
- `max_tokens` (optional, default 8000): Maximum tokens to return
- `strategy` (optional, default "auto"): "auto" finds natural break points, "truncate" hard cuts

**Returns:** Markdown content with metadata header

### web_fetch_code

Extract only code blocks from a page.

**Parameters:**
- `url` (required): URL to extract code from

**Returns:** Code blocks with language annotations and context

### web_fetch_section

Fetch content under a specific heading.

**Parameters:**
- `url` (required): URL to fetch from
- `heading` (required): Heading text to find (case-insensitive)

**Returns:** Section content or list of available sections if not found

### web_fetch_chunked

Fetch large documents in chunks.

**Parameters:**
- `url` (required): URL to fetch
- `chunk` (optional, default 0): Chunk index (0-based)
- `chunk_size` (optional, default 4000): Tokens per chunk

**Returns:** Chunk content with navigation metadata

## Development

```bash
# Clone and install dev dependencies
git clone https://github.com/mathisto/smart-webfetch-mcp
cd smart-webfetch-mcp
pip install -e ".[dev]"

# Run tests
pytest

# Format code
ruff format .
ruff check --fix .
```

## License

MIT
