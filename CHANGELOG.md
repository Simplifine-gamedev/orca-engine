# Changelog

All notable changes to this project will be documented in this file.

## 2025-08-20
### Major: Enhanced Indexing System - Function-Level Intelligence
- **Function-Level Chunking**: GDScript files now parsed semantically - each `func`, `signal`, and `@export` becomes a separate searchable unit with rich metadata instead of basic line-based chunks
- **Signal Flow Tracking**: Complete understanding of signal emission → connection → handler chains across entire projects
- **Multi-Hop Dependency Tracing**: Trace function call chains across files (e.g., Input → Controller → Physics → Animation) with configurable depth
- **Enhanced Search Modes**: 
  - Semantic search (AI understanding of code intent)
  - Keyword search (exact text matching with proper BM25 implementation)  
  - Hybrid search (combines both approaches)
- **Smart Search Intelligence**: Auto-detection of search intent, cross-section deduplication, exact match verification
- **Graph Centrality Analysis**: Identify architecturally important files and their structural roles in projects
- **Performance Improvements**: ~2.7x more intelligent chunks from same files (51 → 138 chunks in demo project)

### Backend: Migration to Weaviate Vector Database
- **Migrated from BigQuery to Weaviate**: Enhanced performance and better vector search capabilities
- **Three Collection Architecture**:
  - `ProjectEmbedding`: Function-level chunks with dependency metadata
  - `ProjectGraph`: File-to-file relationships  
  - `ProjectDependencies`: Function-to-function dependency mapping
- **Real-Time Updates**: All collections updated automatically when files change, including dependency re-extraction
- **Advanced Filtering**: Eliminates indexing of `.import` files and other binary assets
- **Weaviate v4 Compatibility**: Updated client implementation with proper error handling and fallback strategies

### Search & Dependencies
- **Dependency Types Tracked**: Function calls, signal emissions, node access, physics API usage, input system access
- **Context-Aware Results**: Search results include function metadata (signals emitted, functions called, nodes accessed)
- **Nuclear Filtering System**: Multi-layer filtering prevents irrelevant results (`.import` files, etc.)
- **Exact Match Verification**: Keyword search validates actual text presence in content
- **Enhanced Error Recovery**: Graceful fallbacks when Weaviate BM25 API encounters issues

### Documentation & Configuration  
- **New Technical Documentation**: Added comprehensive `backend/indexing.md` with architecture details, examples, and troubleshooting
- **Updated Backend README**: Modernized `backend/README.md` to reflect Weaviate-based system with enhanced API documentation
- **Enhanced Main README**: Added Advanced Indexing System section and backend documentation links throughout
- **Environment Configuration**: Updated setup instructions for Weaviate requirements (`WEAVIATE_URL`, `WEAVIATE_API_KEY`)

### API Enhancements
- **Enhanced `/search_across_project` endpoint**: Added `trace_dependencies`, `search_mode` parameters
- **Improved Response Structure**: Results now include `chunk_type`, `function_name`, `dependencies`, `signals_emitted` metadata
- **Debug Logging**: Comprehensive logging for function extraction, dependency tracking, and search execution
- **Performance Metrics**: Detailed timing and chunk count reporting

## 2025-08-18
- Tightened file management tools: apply-edit now outputs structured diff JSON instead of code (1e8ede8e55).
- updates sprite and sprite sheets

## 2025-08-16
- Apply-edit: create structured diffs; agent can view live and debugger errors (6626e21ec5).
- Updated `.gcloudignore` (31cc62bd4d).
- Indexing and search: improved graph creation and querying; integrated latest Godot docs indexing as a tool (dd05c8b4e5).

## 2025-08-15
- Misc: updated contact email (17ee1973c0).
- Apply-edit: ability to apply patches at specific file locations (390e3b530f).
- Apply-edit: text updates (7830f90f2e).
- README: added Discord link (d2d77e92ca); typo fix; general updates (b91198bcd6, d88c5f0296).
- Editor: script editor whitespace cleanup (de0a870f6e).

## 2025-08-14
- Initial import (clean history; removed sensitive data) (abab192012).


