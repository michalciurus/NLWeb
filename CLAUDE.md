# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## NLWeb Project Overview

NLWeb is a natural language search system that provides intelligent query processing, multi-source retrieval, and AI-powered response generation. It serves as a conversational interface for websites leveraging Schema.org structured data and supports the Model Context Protocol (MCP).

## Commands

### Development Server
```bash
# Start aiohttp server (recommended)
./startup_aiohttp.sh

# Or from code/python directory
cd code/python && python -m webserver.aiohttp_server

# Use legacy server if needed
USE_AIOHTTP=false ./startup_aiohttp.sh
```

### Testing
```bash
# Run all tests
cd code/python && ./testing/run_all_tests.sh

# Comprehensive test suite with options
./testing/run_tests_comprehensive.sh --quick  # Quick smoke tests
./testing/run_tests_comprehensive.sh --mode end_to_end

# Run specific test type
python -m testing.run_tests --type end_to_end --single --query "test query"

# Using pytest directly
pytest -v
```

### Dependencies
```bash
cd code/python
pip install -r requirements.txt       # Core dependencies
pip install -r requirements-dev.txt   # Development dependencies
```

### Docker
```bash
docker build -t nlweb .
docker-compose up
```

### Data Loading & Scraping

**Local/Development:**
```bash
# Incremental crawl with schema extraction
cd code/python
python -m scraping.incrementalCrawlAndLoad example.com
python -m scraping.incrementalCrawlAndLoad example.com --max-pages 100
python -m scraping.incrementalCrawlAndLoad example.com --resume

# Reprocess existing HTML files (re-embed and upload)
python -m scraping.incrementalCrawlAndLoad example.com --reprocess

# Specify database endpoint
python -m scraping.incrementalCrawlAndLoad example.com --database azure_ai_search

# RSS feed loading
python -m data_loading.db_load <rss_url> <site_name>

# Using CLI tool
nlweb.sh data-load  # Interactive prompts for RSS URL and site name
```

**Production Deployment:**
NLWeb does not include built-in scheduling for data loading. In production:
- Run `python -m scraping.incrementalCrawlAndLoad` via external schedulers (cron, Azure Functions, GitHub Actions)
- The incremental crawler maintains state in `crawl_status.json` for resumable operations
- Set `NLWEB_OUTPUT_DIR` environment variable to control where data is stored
- No API endpoints exist for triggering crawls - must be initiated externally

## Architecture

### Backend (Python - `code/python/`)
- **Entry Point**: `webserver/aiohttp_server.py` (async aiohttp server)
- **Core System** (`core/`): Base handlers, configuration, retrieval interfaces
- **Query Analysis** (`core/query_analysis/`): Decontextualization, memory, relevance detection
- **Specialized Handlers** (`methods/`): RAG generation, item comparison, ensemble recommendations
- **Provider System**:
  - LLM Providers: OpenAI, Anthropic, Azure, Gemini, HuggingFace, Ollama
  - Retrieval: Qdrant, Elasticsearch, Azure Search, Milvus, Postgres, Snowflake
  - Storage: Multiple backend options

### Frontend (JavaScript - `static/`)
- **Main Interface**: `fp-chat-interface.js` (ModernChatInterface class)
- **Key Components**:
  - `managed-event-source.js`: SSE handling for streaming responses
  - `conversation-manager.js`: Chat state management
  - `json-renderer.js` & `type-renderers.js`: Content rendering
  - `oauth-login.js`: OAuth authentication (Google, Facebook, Microsoft, GitHub)

### API Endpoints
- `/ask`: Main query endpoint with streaming support (modes: list, summarize, generate)
- `/sites`: Available sites listing
- `/who`: Person/entity information
- OAuth endpoints for authentication
- Conversation CRUD operations

### Configuration
YAML-based configuration in `config/`:
- `config_nlweb.yaml`: Core settings
- `config_llm.yaml`: LLM providers
- `config_retrieval.yaml`: Vector databases
- `config_oauth.yaml`: OAuth setup
- `config_webserver.yaml`: Server configuration

## Query Processing Flow
1. **Request Reception** → Route matching and parameter parsing
2. **Query Analysis** → Decontextualization, item detection, tool selection
3. **Parallel Processing**:
   - Fast track: Quick vector search
   - Full analysis: Comprehensive response generation
4. **Retrieval** → Multi-source search and aggregation
5. **Response Generation** → Based on mode (list/summarize/generate)
6. **Streaming** → Real-time SSE to frontend

## Development Notes

### Testing Strategy
- End-to-end tests: `testing/end_to_end_tests.json`
- Site retrieval tests: `testing/site_retrieval_tests.json`
- Query retrieval tests: `testing/query_retrieval_tests.json`
- Test runner supports single test execution with specific parameters

### Provider Auto-Installation
Dependencies for specific providers (e.g., qdrant-client, elasticsearch) are installed automatically at runtime when needed.

### Recent Changes
- Migration to aiohttp server (see `AIOHTTP_MIGRATION.md`)
- Fixed generate mode message handling in frontend
- Enhanced OAuth integration across providers

### Key Files for Understanding
- **System Architecture**: `systemmap.md`
- **Coding Standards**: `codingrules.md`
- **User Workflows**: `userworkflow.md`
- **Migration Guide**: `AIOHTTP_MIGRATION.md`

### Important Patterns
- Frontend uses vanilla ES6 modules, no build process
- Backend uses async Python with extensive error handling
- Streaming responses via Server-Sent Events
- Parallel processing for optimal performance
- Configuration-driven provider selection