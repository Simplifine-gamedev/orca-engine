# Orca Engine Cloud IDE

## Overview
Orca Engine Cloud IDE brings the full Orca game engine editor to the web browser, allowing developers to create games from anywhere without installing software.

## Live URL
- **Production**: [editor.orcaengine.ai](https://editor.orcaengine.ai)

## Architecture
The cloud IDE runs the actual Orca Engine binary on cloud servers and streams the interface through VNC/noVNC to web browsers.

### Components
- **Backend**: FastAPI server managing editor instances
- **Frontend**: Next.js React app with Monaco editor  
- **Editor**: Real Orca Engine running in Docker with VNC streaming
- **Infrastructure**: Cloud Run, Cloud Build, Container Registry

## Features
- ✅ Full Orca Engine editor in browser
- ✅ Real-time VNC streaming
- ✅ File management and code editing
- ✅ Project persistence
- ✅ Multi-user support

## Deployment

### Prerequisites
- Google Cloud Platform account
- Docker
- gcloud CLI configured

### Quick Deploy
```bash
# Set your project ID
export PROJECT_ID=your-project-id
export REGION=us-central1

# Deploy everything
gcloud builds submit --config=cloudbuild.yaml --project=$PROJECT_ID --region=$REGION

# Deploy just the Orca editor
gcloud builds submit --config=cloudbuild-orca-deploy.yaml --project=$PROJECT_ID --region=$REGION
```

### Build Orca for Linux
```bash
# Build the Linux binary
gcloud builds submit --config=cloudbuild-orca-compile.yaml --project=$PROJECT_ID --region=$REGION
```

## Development

### Local Testing
```bash
# Backend
cd cloud-ide/backend
pip install -r requirements.txt
uvicorn server_simple:app --reload --port 8000

# Frontend  
cd cloud-ide/frontend
npm install
npm run dev
```

## Docker Images

### Available Dockerfiles
- `docker/Dockerfile.orca-real-vnc` - Production Orca editor with VNC
- `docker/Dockerfile.orca-vnc` - Demo editor for testing
- `docker/Dockerfile.orca-builder` - Build environment for Linux compilation

## Configuration Files
- `cloudbuild.yaml` - Main deployment pipeline
- `cloudbuild-orca-compile.yaml` - Orca Linux compilation
- `cloudbuild-orca-deploy.yaml` - Deploy Orca editor
- `docker/supervisord-real.conf` - Process management

## Tech Stack
- **Cloud**: Google Cloud Platform
- **Frontend**: Next.js, React, TypeScript, Tailwind CSS
- **Backend**: Python, FastAPI, WebSockets
- **Streaming**: VNC, noVNC, Xvfb
- **Build**: SCons, Docker

## License
See LICENSE.txt for details.

## Contributing
Contributions are welcome! Please read our contributing guidelines before submitting PRs.

## Support
For issues and questions, please use the GitHub issue tracker.