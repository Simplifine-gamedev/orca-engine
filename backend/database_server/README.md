# Godot AI Database Server

A REST API server for retrieving 3D models from Supabase storage. This server connects to the same Supabase instance used by the GPU shape generation server and provides endpoints for Godot clients to fetch generated 3D models.

## Architecture

```
[Godot Client] → [Database Server] → [Supabase Storage] ← [GPU Server]
                      ↓
                 [File Cache]
```

- **GPU Server** (`shapegen.orcaengine.ai`) generates 3D models and stores them in Supabase
- **Database Server** (this service) provides cached access to those models
- **Godot Client** requests models through the database server
- **File Cache** improves performance by caching frequently accessed files

## Features

- ✅ **List user models** - Get all models for a specific user
- ✅ **Model search** - Search models by prompt text
- ✅ **File downloads** - Download OBJ files and reference images
- ✅ **Intelligent caching** - In-memory file cache with LRU eviction
- ✅ **Health monitoring** - Health check and statistics endpoints
- ✅ **Error handling** - Comprehensive error handling and logging
- ✅ **Cloud deployment** - Ready for Google Cloud Run deployment

## API Endpoints

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | API information and endpoint list |
| `GET` | `/health` | Health check and system status |
| `GET` | `/stats` | Server statistics and metrics |

### Model Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/models/{user_id}` | List all models for a user |
| `GET` | `/models/{user_id}/{model_id}` | Get specific model details |
| `GET` | `/models/{user_id}/search?q={query}` | Search user models by prompt |
| `GET` | `/models/recent` | Get recent public models (anonymized) |

### Download Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/download/{user_id}/{model_id}/obj` | Download OBJ file |
| `GET` | `/download/{user_id}/{model_id}/image` | Download reference image |

### Query Parameters

**List Models** (`/models/{user_id}`)
- `status` - Filter by status (e.g., `completed`, `failed`)
- `limit` - Number of results (max 100, default 50)
- `offset` - Pagination offset (default 0)

**Search Models** (`/models/{user_id}/search`)
- `q` - Search query (required)
- `type` - Filter by model type
- `status` - Filter by status (default: `completed`)
- `limit` - Number of results (max 100, default 20)

**Recent Models** (`/models/recent`)
- `limit` - Number of results (max 50, default 10)
- `status` - Filter by status (default: `completed`)

## Example Usage

### List User Models

```bash
curl "https://your-server.com/models/godot_user123?limit=10&status=completed"
```

```json
{
  "models": [
    {
      "id": "e88c0a41-f202-4f5f-a21c-6c191b27a837",
      "user_id": "godot_user123",
      "prompt": "a great white shark",
      "model_type": "text-to-3d",
      "status": "completed",
      "quality": "turbo",
      "reference_image_url": "https://...",
      "output_file_url": "https://...",
      "created_at": "2025-10-20T10:24:21.987119+00:00",
      "download_endpoints": {
        "obj_file": "/download/godot_user123/e88c0a41-f202-4f5f-a21c-6c191b27a837/obj",
        "reference_image": "/download/godot_user123/e88c0a41-f202-4f5f-a21c-6c191b27a837/image"
      }
    }
  ],
  "count": 1,
  "has_more": false
}
```

### Search Models

```bash
curl "https://your-server.com/models/godot_user123/search?q=shark&limit=5"
```

### Download Files

```bash
# Download OBJ file
curl -O "https://your-server.com/download/godot_user123/e88c0a41-f202-4f5f-a21c-6c191b27a837/obj"

# Download reference image
curl -O "https://your-server.com/download/godot_user123/e88c0a41-f202-4f5f-a21c-6c191b27a837/image"
```

### Health Check

```bash
curl "https://your-server.com/health"
```

```json
{
  "status": "healthy",
  "timestamp": "2025-11-09T...",
  "supabase_status": "connected",
  "cache_size": 15,
  "dev_mode": false,
  "version": "1.0.0"
}
```

## Local Development

### Prerequisites

- Python 3.9+
- Access to Supabase (service key required)

### Setup

1. **Clone and navigate to the database server directory:**
   ```bash
   cd backend/database_server
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment:**
   ```bash
   cp env_template.txt .env
   # Edit .env with your Supabase credentials
   ```

4. **Run the server:**
   ```bash
   python app.py
   ```

5. **Test the server:**
   ```bash
   curl http://localhost:8080/health
   curl http://localhost:8080/models/recent
   ```

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `SUPABASE_URL` | ✅ | Supabase project URL |
| `SUPABASE_KEY` | ✅ | Supabase service role key |
| `SUPABASE_PROJECT_ID` | ✅ | Supabase project ID |
| `DEV_MODE` | ❌ | Enable development features (default: `true`) |
| `PORT` | ❌ | Server port (default: `8080`) |

## Cloud Deployment

### Deploy to Google Cloud Run

1. **Authenticate with GCP:**
   ```bash
   gcloud auth login
   gcloud config set project your-project-id
   ```

2. **Create .env file with production credentials:**
   ```bash
   cp env_template.txt .env
   # Edit .env with production Supabase credentials
   # Set DEV_MODE=false
   ```

3. **Deploy:**
   ```bash
   ./deploy.sh your-gcp-project-id
   ```

4. **Test deployment:**
   ```bash
   # The deploy script will output the service URL
   curl https://your-service-url/health
   ```

### Deployment Features

- ✅ **Auto-scaling** - Scales from 0 to 10 instances based on demand
- ✅ **Secret management** - Environment variables stored securely in GCP Secret Manager
- ✅ **Health monitoring** - Built-in health checks
- ✅ **Performance tuned** - Optimized memory and CPU allocation
- ✅ **Security** - Runs as non-root user in container

## Configuration

### File Caching

The server includes intelligent file caching to improve performance:

- **Cache size**: 100 files (configurable)
- **Eviction**: LRU (Least Recently Used)
- **Storage**: In-memory (could be extended to Redis)
- **Hit rate**: Visible in `/stats` endpoint

### Performance Tuning

**Local Development:**
- `--threaded=True` for concurrent request handling
- Debug mode for detailed error messages

**Production (Cloud Run):**
- 2GB memory allocation
- 1 CPU core
- 80 concurrent requests per instance
- 300s request timeout
- Auto-scaling 0-10 instances

## Error Handling

The server provides comprehensive error handling:

| Status Code | Description |
|-------------|-------------|
| `200` | Success |
| `400` | Bad Request (invalid parameters) |
| `404` | Not Found (model/file not found) |
| `500` | Internal Server Error |
| `503` | Service Unavailable (Supabase offline) |

Error responses include detailed messages:

```json
{
  "error": "Model not found",
  "message": "No model with ID abc123 found for user user456"
}
```

## Integration with Godot

This server is designed to work with Godot's HTTP client. Example GDScript:

```gdscript
# List user models
func get_user_models(user_id: String):
    var http_request = HTTPRequest.new()
    add_child(http_request)
    http_request.request_completed.connect(_on_models_received)
    
    var url = "https://your-server.com/models/" + user_id + "?status=completed"
    http_request.request(url)

func _on_models_received(result: int, response_code: int, headers: PackedStringArray, body: PackedByteArray):
    if response_code == 200:
        var json = JSON.parse_string(body.get_string_from_utf8())
        print("Found ", json.count, " models")
        for model in json.models:
            print("Model: ", model.prompt)

# Download model file
func download_model(user_id: String, model_id: String):
    var http_request = HTTPRequest.new()
    add_child(http_request)
    http_request.request_completed.connect(_on_model_downloaded)
    
    var url = "https://your-server.com/download/" + user_id + "/" + model_id + "/obj"
    http_request.request(url)

func _on_model_downloaded(result: int, response_code: int, headers: PackedStringArray, body: PackedByteArray):
    if response_code == 200:
        # Save OBJ file or import directly
        var file = FileAccess.open("user://downloaded_model.obj", FileAccess.WRITE)
        file.store_buffer(body)
        file.close()
        print("Model downloaded successfully")
```

## Security

- ✅ **Input validation** - All user inputs are validated
- ✅ **Non-root execution** - Container runs as non-root user
- ✅ **Secret management** - Credentials stored in GCP Secret Manager
- ✅ **CORS enabled** - Configured for cross-origin requests
- ✅ **Error sanitization** - No sensitive data in error messages

## Monitoring

### Health Endpoint Response

```json
{
  "status": "healthy",
  "timestamp": "2025-11-09T12:34:56Z",
  "supabase_status": "connected",
  "cache_size": 42,
  "dev_mode": false,
  "version": "1.0.0"
}
```

### Stats Endpoint Response

```json
{
  "cache_size": 42,
  "cache_max_size": 100,
  "total_models": 1250,
  "model_stats": {
    "completed": 1100,
    "processing": 50,
    "failed": 100
  },
  "timestamp": "2025-11-09T12:34:56Z"
}
```

## Related Services

- **GPU Server**: `https://shapegen.orcaengine.ai` - Generates 3D models
- **Main Backend**: `../app.py` - Main Godot AI backend
- **Supabase**: Configured via `SUPABASE_URL` environment variable - Data storage

## License

This is part of the Orca Engine (Godot fork) project.
