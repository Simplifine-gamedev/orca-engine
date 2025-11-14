# Manual Testing Guide for Pricing System

## Current Status: ✅ Products Created, ⚠️ Feature Needed

Your test results show:
- ✅ API key is working
- ✅ Products exist (Free, Pro, Proplus)
- ⚠️ Need to add `ai-requests` feature to products

## Quick Fix Steps:

### 1. Add the Feature in Autumn Dashboard

Go to [https://app.useautumn.com](https://app.useautumn.com) and:

1. **Create Feature:**
   - Go to Features → Create Feature
   - ID: `ai-requests`
   - Type: `Metered`
   
2. **Add to Free Product:**
   - Edit Free product
   - Add Feature: `ai-requests`
   - Limit: `200` per `month`

3. **Add to Pro Product:**
   - Edit Pro product
   - Add Feature: `ai-requests`
   - Limit: `500` per `month`

4. **Add to Proplus Product:**
   - Edit Proplus product
   - Add Feature: `ai-requests`  
   - Limit: `1500` per `month`

### 2. Test Backend API (Without Running Full Backend)

```bash
cd /Users/egekaanduman/orca/orca-engine/backend
python3 test_live_pricing.py
```

Expected results:
- ✅ All 5 tests should pass
- ✅ Autumn API connection working
- ✅ Usage tracking working

### 3. Test Full Backend Server

**Terminal 1 - Start Backend:**
```bash
cd /Users/egekaanduman/orca/orca-engine/backend
python3 app.py
```

**Terminal 2 - Test Endpoints:**
```bash
# Test pricing tiers
curl http://localhost:8080/pricing/tiers | json_pp

# Expected output:
# {
#   "success": true,
#   "tiers": {
#     "free": { "requests_per_month": 200, "price": 0 },
#     "pro": { "requests_per_month": 500, "price": 20 },
#     "proplus": { "requests_per_month": 1500, "price": 60 }
#   }
# }
```

### 4. Test Rate Limiting with Real Chat Request

**With backend running, test a chat request:**

```bash
# This should work (first request)
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -H "X-Machine-ID: test-machine-123" \
  -H "X-Allow-Guest: true" \
  -d '{
    "messages": [{"role": "user", "content": "Hello"}],
    "model": "claude-sonnet-4"
  }'
```

If rate limit is hit (after 200 requests for free tier):
```json
{
  "error": "Monthly request limit exceeded",
  "pricing_info": {
    "error": "Request limit exceeded",
    "upgrade_available": true
  },
  "success": false
}
```

### 5. Test in Orca Engine Editor

1. **Compile Orca Engine:**
   ```bash
   cd /Users/egekaanduman/orca/orca-engine
   scons platform=macos target=editor dev_build=yes
   ```

2. **Start Backend:**
   ```bash
   cd backend
   python3 app.py
   ```

3. **Launch Editor:**
   ```bash
   ./bin/orca.macos.editor.arm64
   ```

4. **Test Flow:**
   - Open AI Chat dock
   - Send multiple messages
   - After limit is hit, pricing dialog should appear
   - Click "Upgrade" to see Stripe checkout

## Troubleshooting

### Issue: "feature with id ai-requests not found"
**Solution:** Add the `ai-requests` feature to your products in Autumn dashboard (see step 1 above)

### Issue: Backend won't start
**Solution:** 
```bash
# Check Python dependencies
pip3 install -r requirements.txt

# Check API key is set
echo $AUTUMN_SECRET_KEY
```

### Issue: Rate limiting not working
**Solution:** 
1. Verify feature is added to products
2. Check backend logs for Autumn API errors
3. Verify API key has correct permissions

### Issue: Pricing dialog not showing
**Solution:**
1. Verify backend is running
2. Check frontend can connect to backend
3. Look for 429 status code in network inspector

## Success Criteria

✅ **All these should work:**
1. Backend starts without errors
2. `/pricing/tiers` endpoint returns all 3 tiers
3. Autumn API connection successful (no 404 errors)
4. Chat requests are counted against limit
5. Rate limit error shows after limit exceeded
6. Pricing dialog appears in editor
7. Upgrade button redirects to Stripe

## Next Steps After Setup

Once everything works:
1. Test checkout flow with Stripe test cards
2. Verify monthly reset works
3. Test upgrade/downgrade flows
4. Monitor usage in Autumn dashboard
5. Set up production keys for launch

