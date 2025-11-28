# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2024-11-28

### Added

- **New tools:**
  - `web_fetch_links` - Extract all links from a page with optional filtering
  - `web_fetch_tables` - Extract tables and convert to markdown format
- **URL caching** - Avoid redundant fetches with 5-minute TTL cache
- **Rate limiting** - Polite 0.5s delay between requests to same domain
- **Configurable timeout** - All tools now accept `timeout` parameter (1-120s)
- Integration tests with pytest-httpx mocking

### Changed

- User-Agent version updated to 0.2
- Now 7 tools total (up from 5)

## [0.2.0] - 2024-11-28

### Added

- GitHub Actions workflow for automated PyPI publishing
- PyPI badges in README (version, downloads, Python version, license)
- Trusted Publishing support (no API tokens needed)

## [0.1.0] - 2024-11-28

### Added

- Initial release
- 5 tools: web_preflight, web_smart_fetch, web_fetch_code, web_fetch_section, web_fetch_chunked
- Token-aware content truncation with natural break points
- Code block extraction with language detection
- Section extraction by heading
- Chunked fetching for large documents
- Support for OpenCode and Claude Desktop configuration
