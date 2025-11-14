# Environment Configuration for Pricing System

## Required Environment Variables

### Autumn Pricing Configuration

```bash
# Autumn API Key (Required for pricing system)
# Test: Get from https://app.useautumn.com/sandbox/dev
# Production: Get from https://app.useautumn.com/dashboard
export AUTUMN_SECRET_KEY=am_sk_your_api_key_here
```

### Development vs Production

**For Development:**
```bash
export DEV_MODE=true
export AUTUMN_SECRET_KEY=am_sk_test_your_test_key_here
```

**For Production:**
```bash
export DEV_MODE=false
export AUTUMN_SECRET_KEY=am_sk_live_your_live_key_here
```

## Deployment Configurations

### Local Development

Create a `.env` file in the backend directory:

```bash
# Pricing system
AUTUMN_SECRET_KEY=am_sk_test_your_test_key_here

# Development mode
DEV_MODE=true

# Backend URL
BACKEND_URL=http://127.0.0.1:8080
```

### Docker Deployment

Add to your Dockerfile or docker-compose.yml:

```yaml
environment:
  - AUTUMN_SECRET_KEY=am_sk_live_your_live_key_here
  - DEV_MODE=false
  - BACKEND_URL=https://your-domain.com
```

### GCP Cloud Run Deployment

Set environment variables in Cloud Run:

```bash
gcloud run services update your-service-name \
  --set-env-vars AUTUMN_SECRET_KEY=am_sk_live_your_live_key_here \
  --set-env-vars DEV_MODE=false
```

### Railway/Vercel/Heroku Deployment

Add environment variables in your platform's dashboard:

```
AUTUMN_SECRET_KEY=am_sk_live_your_live_key_here
DEV_MODE=false
BACKEND_URL=https://your-app-name.railway.app
```

## Environment Variable Validation

The pricing system includes automatic validation:

1. **No API Key**: System runs in fallback mode (unlimited requests)
2. **Invalid API Key**: System logs errors and runs in fallback mode  
3. **Valid API Key**: Full pricing enforcement enabled

Check logs for validation status:

```
AUTUMN_SECRET_KEY not set - pricing features will be disabled
```

## Security Best Practices

1. **Never commit API keys** to version control
2. **Use test keys** for development/staging
3. **Use live keys** only in production
4. **Rotate keys regularly** (every 90 days recommended)
5. **Monitor API usage** in Autumn dashboard

## Testing Configuration

To test the pricing system locally:

```bash
# Set test key
export AUTUMN_SECRET_KEY=am_sk_test_your_test_key_here

# Run the backend
cd backend
python app.py

# In another terminal, run tests
python test_pricing_integration.py
```

## Troubleshooting

### Common Issues

1. **"pricing features will be disabled"**
   - Solution: Set `AUTUMN_SECRET_KEY` environment variable

2. **"Autumn check failed: 401"**
   - Solution: Check API key is correct and active

3. **"Autumn check failed: 403"**
   - Solution: Verify products are configured in Autumn dashboard

4. **Pricing dialog not showing**
   - Solution: Check frontend can reach backend pricing endpoints

### Debug Mode

Enable debug logging:

```bash
export LOGGING_LEVEL=DEBUG
export AUTUMN_DEBUG=true
```

This will show detailed API calls to Autumn in the logs.
