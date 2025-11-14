# Fix: Set Free Tier as Default

## Problem
New users are currently getting Pro tier (500 requests) by default.
They should get Free tier (200 requests) by default.

## Solution: Configure Free as Default Product in Autumn

### In Your Autumn Dashboard:

1. **Go to Products** → Find your **Free** product

2. **Set as Default Product:**
   - Look for "Default Product" or "Is Default" setting
   - Enable this for the **Free** product
   - Make sure **Pro** and **Proplus** have this disabled

3. **Or in Product Settings:**
   - Free product → Mark as "Default plan for new customers"
   - Pro product → Uncheck "Default"
   - Proplus product → Uncheck "Default"

### Expected Behavior:

**Before Fix:**
```json
{
  "balance": 500,  // Pro tier ❌
  "included_usage": 500
}
```

**After Fix:**
```json
{
  "balance": 200,  // Free tier ✅
  "included_usage": 200
}
```

## Alternative: Auto-Assign Free Tier in Backend

If Autumn doesn't auto-assign a default product, we can modify the backend to explicitly attach the Free product when a new user is created.

Let me know if you need help with this approach!

