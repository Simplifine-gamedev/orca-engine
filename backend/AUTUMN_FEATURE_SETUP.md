# Setting Up the ai-requests Feature in Autumn

## Problem Detected
Your test showed: `feature with id ai-requests not found`

This means you need to create the `ai-requests` feature in Autumn to track usage.

## Step-by-Step Setup in Autumn Dashboard

### 1. Go to Features Section
1. Login to [https://app.useautumn.com](https://app.useautumn.com)
2. Navigate to **Features** section

### 2. Create the ai-requests Feature
Click "Create Feature" and configure:

**Feature Settings:**
- **Feature ID**: `ai-requests`
- **Feature Name**: `AI Requests`
- **Feature Type**: `Metered` (for usage tracking)
- **Description**: "AI chat requests usage tracking"

### 3. Add Feature to Each Product

Now add this feature to your three products:

#### Free Product:
1. Open your "Free" product
2. Click "Add Feature Item"
3. Select feature: `ai-requests`
4. Set **Included Usage**: `200`
5. Set **Reset Interval**: `Monthly` (usage resets every month)
6. **Save**

#### Pro Product:
1. Open your "Pro" product  
2. Click "Add Feature Item"
3. Select feature: `ai-requests`
4. Set **Included Usage**: `500`
5. Set **Reset Interval**: `Monthly`
6. **Save**

#### Proplus Product:
1. Open your "Proplus" product
2. Click "Add Feature Item"
3. Select feature: `ai-requests`
4. Set **Included Usage**: `1500`
5. Set **Reset Interval**: `Monthly`
6. **Save**

## Verify the Setup

After adding the feature, run the test again:

```bash
cd /Users/egekaanduman/orca/orca-engine/backend
python3 test_live_pricing.py
```

You should see:
- ✅ Autumn API: PASSED
- ✅ Feature usage tracking working

## Alternative: Using Autumn CLI (if available)

If you prefer using the CLI:

```bash
# Create the feature
autumn features create \
  --id ai-requests \
  --name "AI Requests" \
  --type metered

# Add to Free product
autumn products update free \
  --add-feature ai-requests:200:monthly

# Add to Pro product  
autumn products update pro \
  --add-feature ai-requests:500:monthly

# Add to Proplus product
autumn products update proplus \
  --add-feature ai-requests:1500:monthly
```

## Expected Result

Once configured, API calls will return:

```json
{
  "allowed": true,
  "balance": 200,
  "usage": 0,
  "limit": 200
}
```

And usage will be tracked against the monthly limits!

