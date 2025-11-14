# Pricing System Deployment Checklist

## Pre-Deployment Setup

### ✅ Autumn Account Configuration

- [ ] Create Autumn account at [app.useautumn.com](https://app.useautumn.com)
- [ ] Generate TEST API key for staging
- [ ] Generate LIVE API key for production
- [ ] Connect Stripe account in Autumn dashboard
- [ ] Configure webhook endpoints (automatic)

### ✅ Product Configuration in Autumn Dashboard

Create these exact products:

- [ ] **Free Tier**
  - Product ID: `free`
  - Price: $0/month
  - Feature: `ai-requests` (200 limit, monthly reset)

- [ ] **Pro Tier**
  - Product ID: `pro`
  - Price: $20/month  
  - Feature: `ai-requests` (500 limit, monthly reset)

- [ ] **Pro+ Tier**
  - Product ID: `proplus`
  - Price: $60/month
  - Feature: `ai-requests` (1500 limit, monthly reset)

### ✅ Stripe Configuration

- [ ] Connect live Stripe account to Autumn
- [ ] Test payment flows in Stripe test mode
- [ ] Configure payment methods (card, Apple Pay, Google Pay)
- [ ] Set up billing portal for customers
- [ ] Configure tax settings (if required)

## Environment Configuration

### ✅ Development Environment

```bash
export AUTUMN_SECRET_KEY=am_sk_test_your_test_key_here
export DEV_MODE=true
export BACKEND_URL=http://127.0.0.1:8080
```

### ✅ Production Environment

```bash
export AUTUMN_SECRET_KEY=am_sk_live_your_live_key_here
export DEV_MODE=false
export BACKEND_URL=https://your-production-domain.com
```

## Testing Checklist

### ✅ Backend Integration Tests

- [ ] Run `python3 test_pricing_integration.py` (should pass all tests)
- [ ] Test fallback mode (no API key) - should allow unlimited requests
- [ ] Test API key validation - should connect to Autumn
- [ ] Test rate limiting endpoints - should return proper responses

### ✅ Frontend Integration Tests

- [ ] Compile Orca Engine with pricing integration
- [ ] Test pricing dialog UI appearance
- [ ] Test rate limit popup functionality
- [ ] Test upgrade button navigation

### ✅ End-to-End User Flow

- [ ] Make AI requests until rate limit hit
- [ ] Verify pricing dialog appears with correct tiers
- [ ] Test checkout flow (use Stripe test cards)
- [ ] Verify usage resets after upgrade
- [ ] Test monthly usage reset functionality

## Deployment Steps

### ✅ Backend Deployment

1. **Update Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set Environment Variables**
   - Add `AUTUMN_SECRET_KEY` to production environment
   - Set `DEV_MODE=false`
   - Configure other required variables

3. **Deploy Backend**
   - Deploy to your platform (Railway, Vercel, GCP, etc.)
   - Verify pricing endpoints are accessible
   - Check logs for Autumn connection status

### ✅ Frontend Deployment

1. **Compile Orca Engine**
   ```bash
   scons platform=macos target=editor dev_build=yes
   ```

2. **Test Pricing Integration**
   - Launch editor with backend connection
   - Verify pricing dialog loads
   - Test upgrade flow

3. **Create Distribution**
   - Build production binaries
   - Include pricing system in release notes
   - Update user documentation

## Monitoring & Maintenance

### ✅ Monitoring Setup

- [ ] Monitor Autumn API usage in dashboard
- [ ] Set up alerts for API failures
- [ ] Track conversion rates (free → paid)
- [ ] Monitor monthly usage patterns

### ✅ Customer Support

- [ ] Document common pricing issues
- [ ] Train support team on billing flows
- [ ] Set up billing portal access for customers
- [ ] Create upgrade/downgrade procedures

### ✅ Regular Maintenance

- [ ] Review and rotate API keys quarterly
- [ ] Monitor Stripe webhook health
- [ ] Update pricing tiers as needed
- [ ] Review usage analytics monthly

## Rollback Plan

### ✅ Emergency Rollback

If pricing system fails:

1. **Immediate Fallback**
   ```bash
   # Remove API key to enable fallback mode
   unset AUTUMN_SECRET_KEY
   # or set to empty
   export AUTUMN_SECRET_KEY=""
   ```

2. **Revert to Previous Version**
   ```bash
   git checkout main  # or previous stable branch
   # Redeploy without pricing system
   ```

3. **Communication Plan**
   - Notify users of temporary unlimited access
   - Provide timeline for fix
   - Update status page

## Success Criteria

### ✅ Deployment Success Metrics

- [ ] All pricing tests pass
- [ ] Rate limiting works correctly
- [ ] Payment flows complete successfully  
- [ ] No increase in error rates
- [ ] Customer complaints < 1% of users

### ✅ Business Metrics (Week 1)

- [ ] Free users hit rate limits as expected
- [ ] Conversion rate to paid plans > 2%
- [ ] Payment processing success rate > 98%
- [ ] Support tickets about billing < 5% of total

## Post-Deployment

### ✅ Week 1 Tasks

- [ ] Monitor error logs daily
- [ ] Review user feedback
- [ ] Track conversion metrics
- [ ] Optimize pricing dialog UX based on usage

### ✅ Month 1 Tasks

- [ ] Analyze usage patterns
- [ ] Consider pricing tier adjustments
- [ ] Implement feature usage analytics
- [ ] Plan pricing experiments (A/B tests)

---

**Deployment Team Sign-off:**

- [ ] Backend Developer: ________________
- [ ] Frontend Developer: ________________  
- [ ] DevOps Engineer: ________________
- [ ] Product Manager: ________________
- [ ] QA Engineer: ________________

**Deployment Date:** ________________

**Go-Live Approved By:** ________________
