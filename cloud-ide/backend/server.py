"""
Orca Cloud IDE Backend Server
Manages Orca instances running in Docker containers
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
import uuid
import subprocess
import os
from typing import Dict, Optional
import docker
from pydantic import BaseModel

app = FastAPI()

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Docker client
docker_client = docker.from_env()

# Active Orca instances
active_instances: Dict[str, dict] = {}


class ProjectCreate(BaseModel):
    name: str
    template: Optional[str] = "empty"


class OrcaInstance:
    def __init__(self, user_id: str, project_id: str):
        self.user_id = user_id
        self.project_id = project_id
        self.container = None
        self.port = None
        
    async def start(self):
        """Start a new Orca container for this user"""
        try:
            # Find available port
            import socket
            sock = socket.socket()
            sock.bind(('', 0))
            self.port = sock.getsockname()[1]
            sock.close()
            
            # Run Docker container
            self.container = docker_client.containers.run(
                "orca-cloud:latest",
                detach=True,
                ports={
                    '6080/tcp': self.port,  # noVNC
                    '8080/tcp': self.port + 1  # API
                },
                environment={
                    'USER_ID': self.user_id,
                    'PROJECT_ID': self.project_id
                },
                volumes={
                    f'/tmp/orca-projects/{self.project_id}': {
                        'bind': '/workspace',
                        'mode': 'rw'
                    }
                },
                remove=True,
                name=f"orca-{self.user_id}-{self.project_id}"
            )
            
            # Wait for container to be ready
            await asyncio.sleep(3)
            
            return {
                'vnc_url': f'http://localhost:{self.port}/vnc.html',
                'api_url': f'http://localhost:{self.port + 1}',
                'container_id': self.container.id
            }
            
        except Exception as e:
            print(f"Error starting container: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def stop(self):
        """Stop this Orca instance"""
        if self.container:
            self.container.stop()
            self.container.remove()


@app.post("/api/projects")
async def create_project(project: ProjectCreate):
    """Create a new project and start an Orca instance"""
    project_id = str(uuid.uuid4())
    user_id = "demo-user"  # TODO: Get from auth
    
    # Create project directory
    os.makedirs(f'/tmp/orca-projects/{project_id}', exist_ok=True)
    
    # Start Orca instance
    instance = OrcaInstance(user_id, project_id)
    urls = await instance.start()
    
    # Store instance
    active_instances[project_id] = {
        'instance': instance,
        'urls': urls,
        'name': project.name
    }
    
    return {
        'project_id': project_id,
        'name': project.name,
        **urls
    }


@app.get("/api/projects/{project_id}")
async def get_project(project_id: str):
    """Get project details"""
    if project_id not in active_instances:
        # Try to restart instance
        instance = OrcaInstance("demo-user", project_id)
        urls = await instance.start()
        active_instances[project_id] = {
            'instance': instance,
            'urls': urls
        }
    
    return active_instances[project_id]['urls']


@app.delete("/api/projects/{project_id}")
async def delete_project(project_id: str):
    """Stop project instance"""
    if project_id in active_instances:
        await active_instances[project_id]['instance'].stop()
        del active_instances[project_id]
    
    return {"status": "deleted"}


@app.websocket("/ws/{project_id}")
async def websocket_endpoint(websocket: WebSocket, project_id: str):
    """WebSocket for real-time communication with Orca instance"""
    await websocket.accept()
    
    try:
        while True:
            # Receive message from frontend
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # TODO: Forward to Orca instance via its API
            
            # Send response back
            await websocket.send_text(json.dumps({
                'type': 'response',
                'data': 'processed'
            }))
            
    except WebSocketDisconnect:
        print(f"Client disconnected from project {project_id}")


@app.on_event("startup")
async def startup_event():
    """Build Docker image on startup"""
    print("Building Orca Cloud Docker image...")
    try:
        # Build the Docker image
        subprocess.run([
            "docker", "build", 
            "-f", "docker/Dockerfile.orca-cloud", 
            "-t", "orca-cloud:latest",
            "."
        ], check=True)
        print("Docker image built successfully")
    except subprocess.CalledProcessError as e:
        print(f"Failed to build Docker image: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
