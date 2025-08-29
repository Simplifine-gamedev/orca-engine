# Orca Cloud IDE Architecture (Like Lovable/Replit)

## The Problem with Current Web Build
The current WebAssembly approach runs EVERYTHING in the user's browser:
- Limited by browser memory (2-4GB)
- No real file system access
- Can't run native code
- Performance issues with large projects
- No collaboration features
- Can't compile/export games

## The Solution: Cloud-Based IDE Architecture

### Architecture Overview
```
┌─────────────────────────┐
│   User's Browser        │
│  (Thin Client)          │
│  - UI Components        │
│  - Code Editor          │
│  - WebSocket Client     │
└───────────┬─────────────┘
            │ WebSocket/WebRTC
            │ 
┌───────────┴─────────────┐
│   Cloud Infrastructure  │
├─────────────────────────┤
│   API Gateway           │
│   (Vercel Edge)         │
└───────────┬─────────────┘
            │
┌───────────┴─────────────┐
│   Backend Services      │
├─────────────────────────┤
│ • Container Orchestra    │
│ • Orca Engine Instances │
│ • File Storage (S3)     │
│ • Database (PostgreSQL) │
│ • Redis (Sessions)      │
└─────────────────────────┘
```

## Implementation Plan

### 1. Frontend (Vercel) - Thin Client
```typescript
// Next.js 14 App
/app
  /editor
    /[projectId]
      page.tsx        // Main editor UI
  /api
    /projects
    /compile
    /export
```

**Technologies:**
- **Next.js 14** - React framework
- **Monaco Editor** - VS Code editor in browser
- **WebSockets** - Real-time communication
- **WebRTC** - Video streaming for 3D viewport
- **React Three Fiber** - 3D scene preview

### 2. Backend Architecture

#### Option A: Container-per-User (Like GitHub Codespaces)
```yaml
# Docker container for each user
FROM ubuntu:22.04
RUN apt-get update && apt-get install -y \
    build-essential \
    scons \
    pkg-config \
    libx11-dev \
    libxrandr-dev \
    libxi-dev \
    libxinerama-dev \
    libglew-dev \
    libasound2-dev \
    libpulse-dev \
    xvfb \
    x11vnc

# Install Orca Engine
COPY ./bin/orca.linux.headless /usr/local/bin/orca
```

**Pros:**
- Full Orca functionality
- Isolated environments
- Can run any code

**Cons:**
- Expensive (~$50-200/month per active user)
- Complex orchestration
- Slow cold starts

#### Option B: Serverless Functions + Remote Rendering
```javascript
// Vercel Edge Function
export async function POST(request) {
  const { code, action } = await request.json();
  
  // Send to GPU-enabled worker
  const result = await fetch('https://gpu-worker.railway.app/compile', {
    method: 'POST',
    body: JSON.stringify({ code, projectId })
  });
  
  return new Response(result.body, {
    headers: { 'Content-Type': 'application/octet-stream' }
  });
}
```

**Services Needed:**
- **Vercel** - Frontend & API
- **Railway/Render** - GPU workers for rendering
- **Supabase** - Database & Auth
- **Cloudflare R2** - Asset storage

#### Option C: Hybrid Approach (RECOMMENDED)
Combination of client-side and server-side:

**Client-Side (WebAssembly):**
- 2D editing
- Code editing
- Simple previews
- UI interactions

**Server-Side (Cloud):**
- 3D rendering (streamed)
- Compilation
- Export/Build
- Asset processing
- Multiplayer sync

### 3. Core Services Implementation

#### 3.1 Project Service
```typescript
// /api/projects/[id]/route.ts
import { createClient } from '@supabase/supabase-js'

export async function GET(req, { params }) {
  const project = await supabase
    .from('projects')
    .select('*')
    .eq('id', params.id)
    .single()
    
  // Stream project files from S3
  const files = await s3.listObjects({
    Bucket: 'orca-projects',
    Prefix: `${params.id}/`
  })
  
  return NextResponse.json({ project, files })
}
```

#### 3.2 Live Collaboration
```javascript
// WebSocket server (separate service)
import { Server } from 'socket.io'
import { Redis } from 'ioredis'

const io = new Server()
const redis = new Redis()

io.on('connection', (socket) => {
  socket.on('join-project', async (projectId) => {
    socket.join(projectId)
    
    // Sync project state
    const state = await redis.get(`project:${projectId}`)
    socket.emit('project-state', state)
  })
  
  socket.on('code-change', (data) => {
    // Broadcast to all users in project
    socket.to(data.projectId).emit('code-update', data)
    
    // Save to Redis for persistence
    redis.set(`project:${data.projectId}`, data.content)
  })
})
```

#### 3.3 GPU Rendering Service
```python
# GPU worker (Python/FastAPI on Railway)
from fastapi import FastAPI
import subprocess
import tempfile

app = FastAPI()

@app.post("/render")
async def render_scene(scene_data: dict):
    with tempfile.NamedTemporaryFile(suffix='.tscn') as f:
        f.write(scene_data['content'].encode())
        f.flush()
        
        # Run headless Orca to render
        result = subprocess.run([
            '/usr/local/bin/orca',
            '--headless',
            '--render-frame', f.name,
            '--output', '/tmp/frame.png'
        ], capture_output=True)
        
        # Return rendered frame
        with open('/tmp/frame.png', 'rb') as img:
            return StreamingResponse(img, media_type="image/png")
```

### 4. Technology Stack

#### Frontend (Vercel)
- **Next.js 14** - Framework
- **Tailwind CSS** - Styling
- **Zustand** - State management
- **Monaco Editor** - Code editing
- **Three.js** - 3D preview
- **Socket.io Client** - Real-time sync

#### Backend Services
- **Supabase** - Auth & Database
- **Railway/Render** - GPU compute
- **Cloudflare R2/S3** - File storage
- **Redis** - Session & cache
- **Docker** - Containerization
- **Kubernetes** - Orchestration (optional)

#### Third-Party APIs
- **GitHub** - Version control
- **Stripe** - Payments
- **SendGrid** - Emails
- **Cloudflare Stream** - Video encoding

### 5. Deployment Strategy

#### Phase 1: MVP (1-2 months)
- Basic editor UI (Monaco)
- File management
- Simple 2D editing
- Project storage in S3
- User auth with Supabase

#### Phase 2: Core Features (2-3 months)
- Live collaboration
- GPU-accelerated 3D preview
- Asset pipeline
- Git integration
- Export to platforms

#### Phase 3: Scale (3-6 months)
- Multi-region deployment
- CDN for assets
- Kubernetes orchestration
- Team features
- Marketplace

### 6. Cost Estimates

#### Small Scale (100 users)
- **Vercel Pro**: $20/month
- **Supabase**: $25/month
- **Railway (GPU)**: $200/month
- **Cloudflare R2**: $20/month
- **Total**: ~$265/month

#### Medium Scale (1000 users)
- **Vercel Enterprise**: $500/month
- **Supabase Pro**: $599/month
- **Railway (Multiple GPUs)**: $2000/month
- **Cloudflare R2**: $200/month
- **Redis Cloud**: $200/month
- **Total**: ~$3500/month

#### Large Scale (10k+ users)
- **Custom infrastructure on AWS/GCP**
- **Kubernetes cluster**: $5000+/month
- **GPU instances**: $10000+/month
- **CDN & Storage**: $2000+/month
- **Total**: ~$20000+/month

### 7. Alternative: Streaming Desktop App

Instead of rewriting everything, stream the desktop app:

#### Using Parsec/Moonlight
```javascript
// Simple approach: Remote desktop streaming
const spawnOrcaInstance = async (userId) => {
  // Spawn EC2 instance with Orca
  const instance = await aws.runInstances({
    ImageId: 'ami-orca-desktop',
    InstanceType: 'g4dn.xlarge', // GPU instance
    UserData: Buffer.from(`
      #!/bin/bash
      systemctl start orca
      systemctl start parsec
    `).toString('base64')
  })
  
  // Return Parsec connection details
  return {
    ip: instance.PublicIpAddress,
    sessionId: generateSessionId()
  }
}
```

### 8. Quickest Path to Production

**Use Existing Solutions:**

1. **CodeSandbox/StackBlitz SDK**
   - They provide embeddable cloud IDEs
   - Can customize for Orca
   - ~$500-2000/month

2. **Gitpod/GitHub Codespaces**
   - Run Orca in cloud containers
   - Users connect via browser
   - ~$20-50/user/month

3. **Google Cloud Workstations**
   - Managed developer environments
   - GPU support built-in
   - ~$100-300/user/month

### 9. Recommended Architecture for Orca

**Best approach for game engine IDE:**

```
Frontend (Vercel)          Backend (Railway/Render)
┌────────────────┐        ┌─────────────────────┐
│                │        │                     │
│  Next.js App   │<------>│  Headless Orca     │
│  - File Tree   │  WS    │  - Scene Processing│
│  - Code Editor │        │  - Asset Compiler  │
│  - Properties  │        │  - Physics Sim     │
│                │        │                     │
└────────────────┘        └─────────────────────┘
         │                          │
         └──────────┬───────────────┘
                    │
            ┌───────▼────────┐
            │                │
            │  GPU Renderer  │
            │  - WebRTC      │
            │  - H.264 Stream│
            │                │
            └────────────────┘
```

### 10. Start Simple: Progressive Enhancement

**Step 1: Enhanced Web Editor (Current)**
- Keep WebAssembly version
- Add cloud save/load
- Add asset CDN

**Step 2: Hybrid Mode**
- Client handles 2D/UI
- Server handles 3D/compile

**Step 3: Full Cloud IDE**
- Streaming 3D viewport
- Cloud compilation
- Multi-user collaboration

## Implementation Example

### Quick Start with Next.js + Supabase

```bash
# 1. Create Next.js app
npx create-next-app@latest orca-cloud --typescript --tailwind

# 2. Install dependencies
npm install @supabase/supabase-js socket.io-client monaco-editor

# 3. Create basic editor
```

```typescript
// app/editor/page.tsx
'use client'

import { useEffect, useState } from 'react'
import MonacoEditor from '@monaco-editor/react'
import { createClient } from '@supabase/supabase-js'

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!
)

export default function Editor() {
  const [files, setFiles] = useState([])
  const [currentFile, setCurrentFile] = useState('')
  const [viewport, setViewport] = useState(null)

  useEffect(() => {
    // Connect to GPU renderer
    const ws = new WebSocket('wss://gpu.orca.dev/render')
    
    ws.onmessage = (event) => {
      // Display rendered frame
      const blob = new Blob([event.data], { type: 'image/png' })
      const url = URL.createObjectURL(blob)
      setViewport(url)
    }

    return () => ws.close()
  }, [])

  return (
    <div className="flex h-screen">
      {/* File tree */}
      <div className="w-64 bg-gray-900 p-4">
        {files.map(file => (
          <div key={file.name} onClick={() => setCurrentFile(file)}>
            {file.name}
          </div>
        ))}
      </div>

      {/* Code editor */}
      <div className="flex-1">
        <MonacoEditor
          height="50%"
          language="gdscript"
          theme="vs-dark"
          value={currentFile}
        />
        
        {/* 3D viewport (streamed) */}
        <div className="h-1/2 bg-black">
          {viewport && <img src={viewport} className="w-full h-full object-contain" />}
        </div>
      </div>

      {/* Properties panel */}
      <div className="w-80 bg-gray-900 p-4">
        {/* Inspector UI */}
      </div>
    </div>
  )
}
```

## Conclusion

To build a Lovable-like experience for Orca:

1. **Don't use WebAssembly for everything** - It's too limited
2. **Split rendering between client and server** - UI on client, heavy compute on server
3. **Use GPU instances for 3D** - Stream rendered frames
4. **Start with hybrid approach** - Progressively enhance
5. **Consider existing solutions** - Gitpod, CodeSandbox, etc.

The full implementation would take 3-6 months and cost $3000-20000/month to operate at scale.
