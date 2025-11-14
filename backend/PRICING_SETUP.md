# Orca Engine Pricing System Setup Guide

## Overview

The Orca Engine now includes a comprehensive 3-tier pricing system powered by Autumn AI, providing:

- **Free**: 200 AI requests per month
- **Pro**: 500 AI requests per month ($20/month)
- **Pro+**: 1500 AI requests per month ($60/month)

## Setup Instructions

### 1. Autumn Account Setup

1. Go to [https://app.useautumn.com/sandbox/dev](https://app.useautumn.com/sandbox/dev)
2. Create an account and generate a TEST SECRET API key
3. Add the API key to your environment variables:

```bash
export AUTUMN_SECRET_KEY=am_sk_your_test_key_here
```

### 2. Configure Pricing Products in Autumn Dashboard

Create these three products in your Autumn dashboard:

**Free Tier:**
- Product ID: `free`
- Price: $0/month
- Feature: `ai-requests` with 200 monthly limit (resets monthly)

**Pro Tier:**
- Product ID: `pro`
- Price: $20/month
- Feature: `ai-requests` with 500 monthly limit (resets monthly)

**Pro+ Tier:**
- Product ID: `proplus`
- Price: $60/month
- Feature: `ai-requests` with 1500 monthly limit (resets monthly)

### 3. Stripe Configuration

1. Connect your Stripe account in the Autumn dashboard
2. Configure your webhook endpoints (handled automatically by Autumn)
3. Set up payment methods and billing cycles

### 4. Backend Configuration

The backend is already configured to handle pricing. Ensure these environment variables are set:

```bash
# Required
export AUTUMN_SECRET_KEY=am_sk_your_api_key_here

# Optional (defaults shown)
export BACKEND_URL=http://127.0.0.1:8080
```

### 5. Testing the Integration

Run the pricing integration tests:

```bash
cd backend
python3 test_pricing_integration.py
```

### 6. User Flow

1. User makes AI requests through the Orca Engine editor
2. Backend checks Autumn for usage limits before processing
3. If limit exceeded, user sees pricing dialog with upgrade options
4. User can upgrade through Stripe checkout (handled by Autumn)
5. Usage tracking and billing managed automatically

## API Endpoints

The following endpoints are now available:

- `GET /pricing/tiers` - Get available pricing tiers
- `GET /pricing/customer` - Get user's subscription info
- `POST /pricing/checkout` - Create Stripe checkout for upgrades

## Fallback Mode

If Autumn is not configured (`AUTUMN_SECRET_KEY` not set), the system runs in fallback mode with unlimited requests for development.

## Troubleshooting

1. **Pricing dialog not showing**: Check that `AUTUMN_SECRET_KEY` is set
2. **API errors**: Verify products are configured in Autumn dashboard
3. **Checkout failures**: Ensure Stripe is connected in Autumn
4. **Rate limiting not working**: Check backend logs for Autumn API errors

## Production Deployment

For production:
1. Use live Autumn API keys (`am_sk_live_...`)
2. Connect live Stripe account
3. Set production webhook URLs
4. Test the complete checkout flow

## Support

- Autumn Documentation: [https://docs.useautumn.com](https://docs.useautumn.com)
- Stripe Integration: Handled automatically by Autumn
- Discord Support: [https://discord.gg/STqxY92zuS](https://discord.gg/STqxY92zuS)
