# How Autumn Tracks Every Chat Message

## 🎯 YES - Autumn Tracks Every Single Chat Message

Every time a user sends a message to your AI agent, here's what happens:

## 📊 The Flow (Step-by-Step)

### User Sends Message in Orca Engine:
```
User types: "Help me create a game"
User clicks: Send button
```

### Backend Receives Request:
```python
# app.py line 3352
@app.route('/chat', methods=['POST'])
def chat():
    # 1. Get user ID from auth
    user = verify_authentication()
    # user = {"id": "google_12345678", ...}
    
    # 2. CHECK AUTUMN BEFORE PROCESSING
    allowed, pricing_info = pricing_service.check_and_track_usage(user['id'])
    #     ↑ THIS IS WHERE AUTUMN IS CALLED
    
    # 3. If limit exceeded, BLOCK the message
    if not allowed:
        return jsonify({
            "error": "Monthly request limit exceeded",
            "pricing_info": pricing_info
        }), 429
    
    # 4. Only if allowed, process the AI request
    # ... rest of chat logic ...
```

## 🔄 What `check_and_track_usage()` Does

```python
# autumn_integration.py line 129
def check_and_track_usage(self, user_id: str):
    # Step 1: CHECK with Autumn
    allowed, check_info = self.check_usage(user_id)
    # → Autumn API call: "Does user_google_12345678 have requests left?"
    # → Autumn responds: "Yes, 197 remaining"
    
    # Step 2: TRACK with Autumn
    track_success = self.track_usage(user_id)
    # → Autumn API call: "Track 1 request for user_google_12345678"
    # → Autumn updates: 197 → 196
```

## 📡 Actual API Calls to Autumn

### Check Call:
```bash
POST https://api.useautumn.com/v1/check
{
  "customer_id": "google_12345678",
  "feature_id": "ai-requests"
}
```

**Autumn Response:**
```json
{
  "allowed": true,
  "balance": 197,
  "usage": 3,
  "included_usage": 200,
  "next_reset_at": 1766289173000
}
```

### Track Call:
```bash
POST https://api.useautumn.com/v1/track
{
  "customer_id": "google_12345678",
  "feature_id": "ai-requests",
  "value": 1
}
```

**Autumn Updates:**
```json
{
  "balance": 196,  // Was 197
  "usage": 4       // Was 3
}
```

## 🎮 Real User Example

**User Journey:**

1. **New user signs up**
   - Autumn assigns: Free tier (200 requests)
   - Balance: 200/200

2. **User sends message #1: "Hello"**
   - ✅ Check: 200 requests remaining → Allowed
   - ✅ Track: -1 request
   - Balance: 199/200

3. **User sends message #2: "Create a game"**
   - ✅ Check: 199 requests remaining → Allowed
   - ✅ Track: -1 request
   - Balance: 198/200

4. **... 197 more messages ...**

5. **User sends message #200: "Help with code"**
   - ✅ Check: 1 request remaining → Allowed
   - ✅ Track: -1 request
   - Balance: 0/200

6. **User sends message #201: "Another question"**
   - ❌ Check: 0 requests remaining → BLOCKED
   - 🚫 Request not processed
   - 💳 Shows pricing dialog: "Upgrade to Pro for 500 requests/month"

## 💡 Key Points

### ✅ What Autumn Tracks:
- Total chat messages sent
- Per user (by user ID)
- Per month (resets monthly)
- Across all sessions

### ❌ What Autumn Does NOT Track:
- Message content (text)
- Which AI model used
- Token counts
- Response quality

### 📊 Where Each User's Count is Stored:

**In Autumn's Database:**
```json
{
  "customer_id": "google_12345678",
  "product": "free",
  "features": {
    "ai-requests": {
      "balance": 196,     // Requests remaining
      "usage": 4,         // Requests used this month
      "limit": 200,       // Monthly limit
      "next_reset": "2025-12-15"
    }
  }
}
```

## 🎯 Summary

**YES - Autumn tracks:**
- ✅ Every chat message
- ✅ Count per user
- ✅ Monthly totals
- ✅ Remaining balance

**NO - It does NOT track:**
- ❌ Message content
- ❌ In Supabase
- ❌ Model details
- ❌ Token usage

**Simple answer:** Every chat message = 1 request counted by Autumn! 💬➡️📊
