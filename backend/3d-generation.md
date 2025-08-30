# 3D Model Generation

Optional feature for generating 3D models from text prompts or images.

## Configuration

Add to `backend/.env`:

```env
MODEL_3D_ENABLED=true
MODEL_3D_SERVICE_URL=http://your-3d-service.com
MODEL_3D_SECRET_KEY=your_shared_secret_key
```

## API Endpoints

### Health Check
```
GET /api/3d/health
```

### Generate from Text
```
POST /api/3d/generate/text
Content-Type: application/json

{
  "prompt": "a red sports car"
}
```

### Generate from Image
```
POST /api/3d/generate/image
Content-Type: multipart/form-data

image: [file upload]
```

### Download Model
```
GET /api/3d/download/{filename}
```

### List User Models
```
GET /api/3d/models/{user_id}
```

## Testing

Test the health endpoint:
```bash
curl http://localhost:8000/api/3d/health
```

Generate a 3D model:
```bash
curl -X POST http://localhost:8000/api/3d/generate/text \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a simple cube"}'
```

