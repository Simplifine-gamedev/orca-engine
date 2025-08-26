# Orca Cloud IDE

Full cloud-based IDE for Orca Engine - no WebAssembly, everything runs on the server!

## Architecture

- **Frontend**: Next.js app with Monaco editor + VNC viewport
- **Backend**: FastAPI server managing Docker containers  
- **Engine**: Orca/Godot running headlessly in Docker with VNC
- **Hosting**: Google Cloud Platform (Cloud Run + GCE for GPU)

## Quick Start (Local Testing)

```bash
# Test locally with Docker
./cloud-ide/test-local.sh

# Open browser to http://localhost:3000
```

## Deploy to Google Cloud

```bash
# Make sure you're authenticated
gcloud auth login
gcloud config set project new-website-85890

# Deploy everything
./cloud-ide/deploy-gcp.sh
```

## How It Works

1. **User opens browser** → Connects to Next.js frontend
2. **Creates project** → Backend spins up Docker container with Orca
3. **VNC streaming** → 3D viewport streamed via noVNC in iframe  
4. **Code editing** → Monaco editor sends changes via WebSocket
5. **Orca processes** → Commands in headless mode with virtual display

## Key Features

- ✅ Full Orca Engine running in cloud (no WebAssembly limits)
- ✅ Real-time 3D viewport streaming via VNC
- ✅ Code editor with GDScript support
- ✅ File persistence in cloud storage
- ✅ Can compile and export projects
- ✅ Multi-user support (one container per user)

## Tech Stack

- **Frontend**: Next.js, Monaco Editor, Tailwind CSS
- **Backend**: FastAPI, Docker, WebSockets  
- **Streaming**: noVNC, x11vnc, Xvfb
- **Infrastructure**: GCP Cloud Run, Artifact Registry
- **Engine**: Orca/Godot headless mode

## Costs

- **Small scale** (100 users): ~$265/month
- **Medium scale** (1000 users): ~$3500/month  
- **With GPU**: +$200-2000/month depending on usage

## Next Steps

1. Build Linux version of Orca for better performance
2. Add GPU support for 3D rendering
3. Implement proper auth and user management
4. Add collaboration features
5. Optimize container startup times
