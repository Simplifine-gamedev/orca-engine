# Performance Optimizations Guide

This guide explains the performance optimizations implemented for the Godot AI backend, focusing on the `apply_edit` function and vector database operations.

## 1. Apply Edit Optimizations

### What was slow?
- Complex JSON schema generation and parsing
- Multiple retry attempts with complex prompts
- No timeout handling for long-running AI requests

### Improvements:
- **Simplified prompts**: Removed JSON schema complexity, just ask for edited code directly
- **Retry with fallback**: 5 retries with 1-second delays, then automatic fallback to GPT-5
- **Timeout handling**: Added timeouts at multiple levels:
  - Connection timeout: 10 seconds
  - Response timeout: 60 seconds  
  - Body read timeout: 30 seconds
- **Lower temperature**: Set to 0.3 for more consistent edits
- **Default to faster models**: Uses claude-4 by default for apply_edit

## 2. Vector Database Optimizations

### Weaviate Integration (Recommended)
For the best performance, use Weaviate cloud vector database:

```bash
# Add to your .env file:
WEAVIATE_URL=your-weaviate-cluster-url
WEAVIATE_API_KEY=your-weaviate-api-key
```

Benefits:
- **10x faster search**: HNSW indexing with optimized parameters
- **Parallel embedding**: Up to 10 concurrent embedding operations
- **Connection pooling**: 20 persistent connections, max 100
- **Smart caching**: In-memory cache for frequent queries
- **Batch operations**: Efficient bulk inserts and updates

### BigQuery Optimizations
If using Google Cloud BigQuery:
- **Parallel embeddings**: Process multiple batches concurrently
- **Adjustable concurrency**: Set via environment variables:
  ```bash
  EMBED_BATCH_SIZE=100      # Texts per batch (default: 100)
  EMBED_MAX_PARALLEL=5      # Concurrent batches (default: 4)
  INDEX_MAX_WORKERS=32      # Indexing threads (default: CPU*2)
  ```

## 3. Configuration for Best Performance

### Environment Variables
```bash
# Weaviate (fastest option)
WEAVIATE_URL=https://your-cluster.weaviate.network
WEAVIATE_API_KEY=your-api-key

# Or Google Cloud (good option)
GCP_PROJECT_ID=your-project-id

# Embedding optimization
EMBED_BATCH_SIZE=100      # Increase for better throughput
EMBED_MAX_PARALLEL=5      # More concurrent requests

# Model selection
DEFAULT_MODEL=claude-4    # Fast default model
```

### Performance Expectations

With Weaviate:
- **Indexing**: ~50-100 chunks/second
- **Search latency**: ~50-200ms average
- **Concurrent operations**: Handles multiple users efficiently

With BigQuery:
- **Indexing**: ~20-50 chunks/second  
- **Search latency**: ~200-500ms average
- **Scales well**: Good for large datasets

## 4. Testing Performance

Run the performance test script:
```bash
cd backend
python test_weaviate_performance.py
```

This will:
- Test connection speed
- Measure indexing throughput
- Benchmark search latency
- Verify proper configuration

## 5. Troubleshooting

### Slow embedding generation
- Check `EMBED_MAX_PARALLEL` - increase for more concurrency
- Verify OpenAI API rate limits aren't being hit
- Consider using larger `EMBED_BATCH_SIZE`

### Slow searches
- Ensure Weaviate is configured (check logs for "Using Weaviate")
- Verify vector indices are properly created
- Check cache hit rates in logs

### Apply edit timeouts
- Increase timeout values in `editor_tools.cpp` if needed
- Consider using smaller file ranges for edits
- Check if the AI model is overloaded (automatic retry/fallback should help)

## 6. Known Issues & Recommendations

### Weaviate Client Version
The current implementation uses workarounds for Weaviate client v4.7.1. For better performance:
```bash
# Upgrade to latest version
pip install weaviate-client>=4.16.0
```

This will enable:
- Native filtered queries (no Python-side filtering)
- Better aggregate queries with filters
- More efficient batch operations

### Current Performance (with v4.7.1 workarounds)
- **Indexing**: ~30-35 chunks/second
- **Search latency**: ~500-700ms (includes fallback overhead)
- **With latest client**: Expected 50-200ms search latency

## 7. Monitoring

Watch the backend logs for performance metrics:
```
VECTOR_INDEX: Using Weaviate at https://... (optimized for speed)
WeaviateVector: Indexed 45 chunks from 3 files in 1.23s
APPLY_EDIT: Retry 1/5 after Overloaded error, waiting 1s...
APPLY_EDIT: Switched to gpt-5 after claude-4 failures
Query method failed: 'QueryReturn' object has no attribute 'where', trying alternative approach
```

## Future Optimizations

Potential improvements:
- Streaming responses for apply_edit
- Incremental indexing (only changed chunks)
- Distributed embedding generation
- Edge caching for common queries
- Background re-indexing
