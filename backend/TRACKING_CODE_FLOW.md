# Exact Code Flow: How Autumn Tracks Chat Messages

## 📁 Files Involved

### 1. `backend/app.py` (Main endpoint)
### 2. `backend/autumn_integration.py` (Autumn service)
### 3. Autumn API (External - their servers)

---

## 🔍 Step-by-Step Code Trace

### STEP 1: User Sends Chat Message

**File:** `backend/app.py` (Line 3352)

```python
@app.route('/chat', methods=['POST'])
def chat():
    # Verify who the user is
    user, error_response, status_code = verify_authentication()
    if error_response:
        return error_response, status_code
    
    # ⬇️ THIS IS WHERE TRACKING HAPPENS ⬇️
    # Line 3375-3382
    allowed, pricing_info = pricing_service.check_and_track_usage(user['id'])
    if not allowed:
        return jsonify({
            "error": "Monthly request limit exceeded",
            "pricing_info": pricing_info,
            "upgrade_url": f"{request.host_url}pricing",
            "success": False
        }), 429
    
    # Only continue if allowed...
```

**What happens here:**
- Gets `user['id']` (e.g., "google_12345678")
- Calls `check_and_track_usage(user['id'])`
- If blocked → Returns 429 error immediately
- If allowed → Continues to process message

---

### STEP 2: Check and Track Usage

**File:** `backend/autumn_integration.py` (Line 129-169)

```python
def check_and_track_usage(self, user_id: str) -> Tuple[bool, Dict]:
    """
    Check if user can make request and track usage atomically
    """
    # First: CHECK if user has requests left
    allowed, check_info = self.check_usage(user_id)
    
    # Auto-assign free tier to new users
    if check_info.get("balance") is None:
        self.attach_free_tier(user_id)
        allowed, check_info = self.check_usage(user_id)
    
    # If not allowed, return error
    if not allowed:
        return False, {
            "error": "Request limit exceeded",
            "upgrade_available": True
        }
    
    # If allowed: TRACK the usage
    track_success = self.track_usage(user_id)
    
    return True, {"usage_tracked": track_success}
```

**What happens here:**
- Calls `check_usage()` → Makes API call to Autumn
- If new user → Assigns Free tier
- If allowed → Calls `track_usage()` → Makes another API call to Autumn
- Returns whether user can proceed

---

### STEP 3: Check Usage (API Call #1)

**File:** `backend/autumn_integration.py` (Line 36-67)

```python
def check_usage(self, user_id: str) -> Tuple[bool, Dict]:
    """
    Check if user has access to make a request
    """
    response = requests.post(
        "https://api.useautumn.com/v1/check",  # ← Autumn's API
        headers={'Authorization': f'Bearer {self.api_key}'},
        json={
            "customer_id": user_id,      # ← Your user ID
            "feature_id": "ai-requests"  # ← The feature we're tracking
        }
    )
    
    data = response.json()
    # Returns: {"allowed": true, "balance": 197, "usage": 3}
    
    return data.get("allowed", False), data
```

**What happens here:**
- Makes HTTP POST to Autumn API
- Sends: user ID + feature ID
- Gets back: allowed (true/false), balance, usage
- Autumn checks their database

---

### STEP 4: Track Usage (API Call #2)

**File:** `backend/autumn_integration.py` (Line 69-97)

```python
def track_usage(self, user_id: str, value: int = 1) -> bool:
    """
    Track usage for a user
    """
    response = requests.post(
        "https://api.useautumn.com/v1/track",  # ← Autumn's API
        headers={'Authorization': f'Bearer {self.api_key}'},
        json={
            "customer_id": user_id,      # ← Your user ID
            "feature_id": "ai-requests", # ← The feature
            "value": 1                   # ← Count this as 1 request
        }
    )
    
    return response.status_code == 200
```

**What happens here:**
- Makes HTTP POST to Autumn API
- Tells Autumn: "User just made 1 request"
- Autumn updates their database
- Decrements balance: 197 → 196

---

## 📊 Complete Message Flow

```
USER ACTION:
  User types: "Create a player script"
  User clicks: Send
       ↓
       
BACKEND FILE: app.py
  Line 3370: verify_authentication()
    → Returns: user['id'] = "google_12345678"
       ↓
       
  Line 3375: pricing_service.check_and_track_usage(user['id'])
       ↓
       
BACKEND FILE: autumn_integration.py
  Line 140: check_usage(user_id)
    → HTTP POST to Autumn: /check
    → Autumn checks: google_12345678 has 197 left
    → Returns: allowed=true, balance=197
       ↓
       
  Line 159: track_usage(user_id)
    → HTTP POST to Autumn: /track
    → Autumn updates: google_12345678 usage +1
    → New balance: 196
       ↓
       
BACK TO: app.py
  Line 3376: if not allowed → Block (429)
  Line 3384+: If allowed → Process AI message
       ↓
       
AI PROCESSES MESSAGE AND RESPONDS
```

## 📁 File Summary

| File | Role | What It Does |
|------|------|--------------|
| `app.py` | Main endpoint | Receives chat, checks pricing, processes if allowed |
| `autumn_integration.py` | Autumn API client | Makes HTTP calls to Autumn to check/track |
| Autumn API | External service | Stores all usage data in their database |

## 🔢 What Gets Tracked

For user "google_12345678":

**In Autumn's Database:**
```
Month: November 2025
User: google_12345678
Tier: Free
Limit: 200
Usage: 4
Balance: 196
Last request: 2025-11-14 10:23:45
Next reset: 2025-12-01
```

**NOT in your code or Supabase!**

## 🎯 Simple Answer

**2 files do the tracking:**
1. **`app.py`** - Calls the tracking function
2. **`autumn_integration.py`** - Makes HTTP API calls to Autumn

**Autumn's servers** store and manage all the usage data!

---

**TL;DR:** Every chat message triggers 2 HTTP calls to Autumn API (check + track), Autumn stores the counts on their servers.
