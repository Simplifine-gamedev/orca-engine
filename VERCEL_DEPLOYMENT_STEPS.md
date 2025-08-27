# Deploy Orca Engine to editor.orcaengine.ai

## Manual Steps for Vercel Dashboard:

1. **Go to Vercel Dashboard:**
   - Visit https://vercel.com/dashboard
   - Click on your `orca-engine` project

2. **Project Settings:**
   - Go to Settings → General
   - Under "Root Directory", leave it empty (or `.`)
   - Under "Build & Output Settings":
     - Build Command: `echo 'Using prebuilt files'`
     - Output Directory: `.`
     - Install Command: Leave empty

3. **Add Custom Domain:**
   - Go to Settings → Domains
   - Click "Add Domain"
   - Enter: `editor.orcaengine.ai`
   - Follow the DNS setup instructions (usually add a CNAME record pointing to `cname.vercel-dns.com`)

4. **Redeploy:**
   - Go to the Deployments tab
   - Click the three dots on the latest deployment
   - Click "Redeploy"
   - Or trigger by pushing to GitHub again

## Files Already Set Up:

✅ **Web export files** - All Orca web files are in project root
✅ **vercel.json** - Configured with proper headers for WebAssembly
✅ **.vercelignore** - Ignoring unnecessary source files

## Current Status:
- Files are ready in the repository
- Configuration is correct
- Just needs manual domain setup in Vercel Dashboard

## Domain DNS Settings:
Add this to your DNS provider for orcaengine.ai:
- Type: CNAME
- Name: editor
- Value: cname.vercel-dns.com

Or if using Vercel DNS:
- Type: A
- Name: editor
- Value: 76.76.21.21
