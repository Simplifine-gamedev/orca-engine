# How Users Upgrade Their Plans

## 🎯 Complete Upgrade Flow

Users can upgrade in 2 ways:

### Method 1: Hit Rate Limit (Automatic)
### Method 2: Manual Upgrade (Proactive)

---

## 🚫 Method 1: Hit Rate Limit (Automatic Prompt)

### What Happens:

```
1. User sends message #201 (limit: 200)
   ↓
2. Backend blocks with 429 error
   ↓
3. Frontend shows Pricing Dialog
   ↓
4. User clicks "Upgrade to Pro"
   ↓
5. Stripe checkout opens
   ↓
6. User pays $20/month
   ↓
7. Autumn updates: 200 → 500 requests
   ↓
8. User can continue chatting
```

### Code Implementation:

**Backend blocks request:**
```python
# app.py line 3375-3382
allowed, pricing_info = pricing_service.check_and_track_usage(user['id'])
if not allowed:
    return jsonify({
        "error": "Monthly request limit exceeded",
        "pricing_info": pricing_info,
        "upgrade_url": f"{request.host_url}pricing"
    }), 429  # ← 429 status triggers frontend dialog
```

**Frontend detects 429 and shows dialog:**
```cpp
// ai_chat_dock.cpp line 4010-4012
if (error_category == "rate_limit") {
    if (response_data.has("pricing_info")) {
        _on_rate_limit_exceeded(pricing_info);  // ← Shows pricing dialog
    }
}
```

**Pricing dialog appears:**
```cpp
// ai_chat_dock.cpp line 16683-16706
void AIChatDock::_on_rate_limit_exceeded(const Dictionary &error_data) {
    String message = "Monthly request limit exceeded!\n\n";
    message += "You've reached your plan limit. Upgrade to continue.";
    
    rate_limit_popup->add_button("Upgrade Plan", false, "upgrade");
    rate_limit_popup->popup_centered();
}

// When user clicks "Upgrade Plan":
void AIChatDock::_on_rate_limit_upgrade_pressed(const String &action) {
    if (action == "upgrade") {
        pricing_dialog->show_rate_limit_dialog(Dictionary());
        // ↑ Shows full pricing options
    }
}
```

**User selects a plan:**
```cpp
// pricing_dialog.cpp line 143-160
void PricingDialog::_on_upgrade_pressed(const String &product_id) {
    // Make request to backend for checkout URL
    String url = backend_url + "/pricing/checkout";
    
    Dictionary request_data;
    request_data["product_id"] = product_id;  // "pro" or "proplus"
    
    checkout_http_request->request(url, headers, HTTPClient::METHOD_POST, json_string);
}
```

**Backend generates Stripe checkout:**
```python
# app.py line 9042-9059
@app.route('/pricing/checkout', methods=['POST'])
def create_checkout():
    user = verify_authentication()
    product_id = request.json.get('product_id')  # "pro" or "proplus"
    
    # Call Autumn to get Stripe checkout URL
    checkout_data = pricing_service.get_checkout_url(user['id'], product_id)
    
    return jsonify({"checkout": checkout_data})
```

**Autumn returns Stripe URL:**
```python
# autumn_integration.py line 171-189
def get_checkout_url(self, user_id: str, product_id: str):
    response = requests.post(
        "https://api.useautumn.com/v1/checkout",
        json={
            "customer_id": user_id,
            "product_id": product_id  # "pro" or "proplus"
        }
    )
    
    # Returns: {"checkout_url": "https://checkout.stripe.com/..."}
    return response.json()
```

**Frontend opens Stripe:**
```cpp
// pricing_dialog.cpp line 229-238
void PricingDialog::_on_checkout_response(...) {
    Dictionary checkout_data = response_data.get("checkout", Dictionary());
    
    if (checkout_data.has("checkout_url")) {
        String checkout_url = checkout_data.get("checkout_url", "");
        OS::get_singleton()->shell_open(checkout_url);  // ← Opens browser
    }
}
```

---

## 💳 Method 2: Manual Upgrade (Proactive)

### Future Implementation (Not Yet Built):

You can add a "Pricing" menu item in the editor:

```cpp
// In editor menu
Menu → Subscription → View Plans
  ↓
Shows pricing_dialog with all tiers
  ↓
User clicks "Upgrade to Pro+"
  ↓
Same checkout flow as above
```

---

## 🔄 What Happens After Payment

### Stripe Payment Complete:

```
1. User completes payment on Stripe
   ↓
2. Stripe sends webhook to Autumn
   ↓
3. Autumn updates customer:
      - Tier: Free → Pro
      - Limit: 200 → 500
      - Balance: Resets to 500
   ↓
4. User returns to Orca Engine
   ↓
5. Next request: Autumn sees Pro tier
   ↓
6. User gets 500 requests/month! 🎉
```

### No Code Needed for Webhooks:
- ✅ Autumn handles ALL Stripe webhooks
- ✅ Automatically updates user tier
- ✅ Synchronizes subscription status
- ✅ Handles failed payments
- ✅ Manages downgrades/cancellations

---

## 📁 Files Involved in Upgrade Flow

### Backend Files:
1. **`app.py`** (Lines 9042-9059)
   - `/pricing/checkout` endpoint
   - Gets checkout URL from Autumn

2. **`autumn_integration.py`** (Lines 171-189)
   - `get_checkout_url()` function
   - Makes API call to Autumn

### Frontend Files:
1. **`editor/docks/ai_chat_dock.cpp`** (Lines 16683-16716)
   - Detects rate limit (429 error)
   - Shows upgrade button
   - Opens pricing dialog

2. **`editor/pricing/pricing_dialog.cpp`** (All)
   - Shows pricing tiers
   - Handles upgrade button clicks
   - Opens Stripe checkout URL in browser

---

## 🎮 User Experience

### Current User Flow:

```
User at 200/200 requests (Free tier)
     ↓
Sends message #201
     ↓
[Dialog appears]
┌─────────────────────────────────────┐
│ Request Limit Exceeded!             │
│                                     │
│ You've used all 200 requests        │
│                                     │
│ [OK]  [Upgrade Plan]                │
└─────────────────────────────────────┘
     ↓ (clicks Upgrade Plan)
     ↓
[Pricing Dialog appears]
┌─────────────────────────────────────┐
│ Upgrade Your Orca Engine Plan       │
│                                     │
│ ┌─────────────────────────────────┐ │
│ │ Free - $0/month                 │ │
│ │ 200 requests/month              │ │
│ │ [Current Plan]                  │ │
│ └─────────────────────────────────┘ │
│                                     │
│ ┌─────────────────────────────────┐ │
│ │ Pro - $20/month                 │ │
│ │ 500 requests/month              │ │
│ │ [Upgrade] ← User clicks         │ │
│ └─────────────────────────────────┘ │
│                                     │
│ ┌─────────────────────────────────┐ │
│ │ Pro+ - $60/month                │ │
│ │ 1500 requests/month             │ │
│ │ [Upgrade]                       │ │
│ └─────────────────────────────────┘ │
└─────────────────────────────────────┘
     ↓
Browser opens Stripe checkout
     ↓
User enters payment details
     ↓
Payment successful
     ↓
Autumn updates tier automatically
     ↓
User returns to Orca Engine
     ↓
Can now send 500 messages/month! 🎉
```

---

## 🧪 Test the Upgrade Flow

```bash
# Terminal 1: Start backend
cd /Users/egekaanduman/orca/orca-engine/backend
python3 app.py

# Terminal 2: Test checkout endpoint
curl -X POST http://localhost:8080/pricing/checkout \
  -H "Content-Type: application/json" \
  -H "X-Machine-ID: test-machine" \
  -H "X-Allow-Guest: true" \
  -d '{"product_id": "pro"}'

# Should return:
# {"checkout": {"checkout_url": "https://checkout.stripe.com/..."}}
```

---

## 🎯 Summary

**Users buy plans by:**
1. ✅ Hitting rate limit → Dialog appears → Upgrade button
2. ✅ Manual upgrade (future: add menu item)
3. ✅ Clicking upgrade → Stripe checkout opens
4. ✅ Completing payment → Autumn auto-updates tier
5. ✅ Returning to editor → New limits active

**All handled by:**
- `app.py` - Checkout endpoint
- `autumn_integration.py` - Autumn API calls
- `pricing_dialog.cpp` - Frontend UI
- Autumn + Stripe - Payment processing

**No manual webhook handling needed!** 🚀
