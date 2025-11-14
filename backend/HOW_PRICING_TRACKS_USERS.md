# How Pricing Tracks Users - Technical Explanation

## 🎯 Overview

The pricing system **does NOT use Supabase** for user identification. It uses your **existing AuthManager** system.

## 📊 User Identification Flow

### 1. User Authentication (AuthManager)

When a request comes to `/chat`:

```python
# Step 1: Get user from existing auth
user, error_response, status_code = verify_authentication()
# Returns: {"id": "user_123", "name": "John", "email": "john@example.com"}

# Step 2: Pass user ID to Autumn
allowed, pricing_info = pricing_service.check_and_track_usage(user['id'])
```

### 2. Where User IDs Come From

Your existing `verify_authentication()` function supports **3 methods**:

#### Method 1: OAuth Users (Google/GitHub/Microsoft)
```python
# User logs in via OAuth
# AuthManager creates session with user data
user = {
    "id": "google_12345678",  # From OAuth provider
    "name": "John Doe",
    "email": "john@gmail.com",
    "provider": "google"
}
```

#### Method 2: Guest Users
```python
# User uses editor without login
# Machine ID is used to create guest session
user = {
    "id": "guest_machine_abc123",  # From machine_id
    "name": "Guest User",
    "email": "guest@orca.local",
    "provider": "guest"
}
```

#### Method 3: Dev Mode
```python
# In development (DEV_MODE=true)
user = {
    "id": "dev_machine_xyz789",  # From machine_id
    "name": "Dev User",
    "email": "dev@example.com",
    "provider": "dev_mode"
}
```

## 🔄 Complete Request Flow

```
1. User makes AI request from Orca Editor
   ↓
2. Request sent to backend /chat endpoint
   ↓
3. verify_authentication() checks user
   ↓ Returns user object with 'id'
   ↓
4. pricing_service.check_and_track_usage(user['id'])
   ↓ Calls Autumn API with user_id as customer_id
   ↓
5. Autumn checks: Does this user have requests left?
   ↓ YES: Allow & decrement balance
   ↓ NO: Block & return error
   ↓
6. If blocked → Return 429 error
   ↓
7. Frontend shows pricing dialog
```

## 🔑 Key Points

### ✅ User IDs are Unique Per User
- **OAuth users**: `google_123`, `github_456`, `microsoft_789`
- **Guest users**: `guest_machine_abc` (per machine)
- **Dev users**: `dev_machine_xyz` (per machine)

### ✅ Autumn Uses These IDs
Each unique `user['id']` becomes a `customer_id` in Autumn:

```python
# Backend code
user_id = user['id']  # From AuthManager
pricing_service.check_and_track_usage(user_id)

# Inside pricing service
requests.post("https://api.useautumn.com/v1/check", json={
    "customer_id": user_id,  # ← Same ID from auth
    "feature_id": "ai-requests"
})
```

### ✅ No Supabase for User Tracking
- **Supabase** is used for: Logging, analytics, crash reports
- **NOT used for**: User identification, pricing, rate limiting
- **AuthManager** handles: All user sessions and IDs
- **Autumn** handles: All pricing, limits, billing

## 🗂️ Data Storage Breakdown

| Data Type | Stored Where | Purpose |
|-----------|--------------|---------|
| **User Sessions** | AuthManager (.auth_sessions.pkl) | Authentication |
| **User IDs** | AuthManager | Unique user identification |
| **Pricing Limits** | Autumn Database | Rate limiting data |
| **Usage Counts** | Autumn Database | Request tracking |
| **Subscriptions** | Autumn Database | Which tier user has |
| **Payments** | Stripe (via Autumn) | Billing |
| **LLM Logs** | Supabase | Analytics only |

## 🧪 Example User Journey

### New User (Guest Mode):
```
1. User opens Orca Engine
   → Machine ID: abc123
   → AuthManager creates: guest_abc123

2. User sends AI message
   → Backend gets user_id: guest_abc123
   → Autumn checks: guest_abc123 usage
   → First time: Auto-assigns Free tier (200 requests)
   → Balance: 200 → 199

3. After 200 requests:
   → Autumn returns: allowed=false
   → Frontend shows pricing dialog
   → User clicks "Upgrade to Pro"
   → Redirects to Stripe checkout

4. After payment:
   → Autumn updates: guest_abc123 → Pro tier
   → New balance: 500 requests/month
```

### Logged In User (OAuth):
```
1. User logs in with Google
   → OAuth returns: google_12345678
   → AuthManager stores session

2. User sends AI message
   → Backend gets user_id: google_12345678
   → Autumn checks: google_12345678 usage
   → Tracks usage per Google account
```

## 💡 Key Insight

**Each user is tracked by their unique ID:**
- OAuth users → tracked by OAuth provider ID
- Guest users → tracked by machine ID
- This ID is passed to Autumn as `customer_id`
- Autumn manages all pricing/limits for that customer_id
- No additional database needed!

## 🔐 Security

User IDs are:
- ✅ Unique per user/machine
- ✅ Persistent across sessions (OAuth) or machine (Guest)
- ✅ Secure (OAuth tokens, session tokens)
- ✅ Never exposed to frontend
- ✅ Managed by AuthManager

Pricing limits are:
- ✅ Enforced server-side only
- ✅ Cannot be bypassed from frontend
- ✅ Stored in Autumn's database
- ✅ Synced with Stripe payments

---

**Summary:** Your existing AuthManager provides user IDs → Autumn tracks pricing per ID → No Supabase needed for pricing!
