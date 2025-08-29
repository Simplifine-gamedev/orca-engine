"""
Simplified Orca Cloud IDE Backend Server for Cloud Run
This version runs without Docker management for initial deployment
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
import uuid
from typing import Dict, Optional
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

# In-memory storage for demo (use Cloud Storage/Firestore in production)
projects = {}
project_files = {}

class ProjectCreate(BaseModel):
    name: str
    template: Optional[str] = "empty"

class FileContent(BaseModel):
    path: str
    content: str

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "ok", "service": "Orca Cloud IDE Backend"}

@app.post("/api/projects")
async def create_project(project: ProjectCreate):
    """Create a new project"""
    project_id = str(uuid.uuid4())
    
    # Store project metadata
    projects[project_id] = {
        "id": project_id,
        "name": project.name,
        "template": project.template,
        "files": {}
    }
    
    # Initialize with template files
    if project.template == "empty":
        project_files[project_id] = {
            "main.gd": """extends Node

# Welcome to Orca Engine!
func _ready():
    print("Hello, Orca!")
""",
            "project.godot": """[application]
config/name="{}".format(project.name)
run/main_scene="res://main.tscn"
"""
        }
    
    return {
        "project_id": project_id,
        "name": project.name,
        "vnc_url": f"https://demo-vnc.orcaengine.ai/{project_id}",  # Placeholder
        "api_url": f"https://orca-backend-umkiqffckq-uc.a.run.app"
    }

@app.get("/api/projects/{project_id}")
async def get_project(project_id: str):
    """Get project details"""
    if project_id not in projects:
        raise HTTPException(status_code=404, detail="Project not found")
    
    return {
        "project": projects[project_id],
        "vnc_url": f"https://demo-vnc.orcaengine.ai/{project_id}",  # Placeholder
        "api_url": f"https://orca-backend-umkiqffckq-uc.a.run.app"
    }

@app.get("/api/projects/{project_id}/files")
async def list_files(project_id: str):
    """List project files"""
    if project_id not in project_files:
        raise HTTPException(status_code=404, detail="Project not found")
    
    return {
        "files": [{"name": name, "path": name} for name in project_files[project_id].keys()]
    }

@app.get("/api/projects/{project_id}/files/{file_path:path}")
async def get_file(project_id: str, file_path: str):
    """Get file content"""
    if project_id not in project_files:
        raise HTTPException(status_code=404, detail="Project not found")
    
    if file_path not in project_files[project_id]:
        raise HTTPException(status_code=404, detail="File not found")
    
    return {
        "path": file_path,
        "content": project_files[project_id][file_path]
    }

@app.put("/api/projects/{project_id}/files/{file_path:path}")
async def save_file(project_id: str, file_path: str, content: FileContent):
    """Save file content"""
    if project_id not in project_files:
        project_files[project_id] = {}
    
    project_files[project_id][file_path] = content.content
    
    return {"status": "saved", "path": file_path}

@app.delete("/api/projects/{project_id}")
async def delete_project(project_id: str):
    """Delete project"""
    if project_id in projects:
        del projects[project_id]
    if project_id in project_files:
        del project_files[project_id]
    
    return {"status": "deleted"}

@app.websocket("/ws/{project_id}")
async def websocket_endpoint(websocket: WebSocket, project_id: str):
    """WebSocket for real-time communication"""
    await websocket.accept()
    
    try:
        while True:
            # Receive message from frontend
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Handle different message types
            if message.get("type") == "load-file":
                file_path = message.get("path")
                if project_id in project_files and file_path in project_files[project_id]:
                    await websocket.send_text(json.dumps({
                        "type": "file-content",
                        "path": file_path,
                        "content": project_files[project_id][file_path]
                    }))
            
            elif message.get("type") == "save-file":
                file_path = message.get("path")
                content = message.get("content")
                if project_id not in project_files:
                    project_files[project_id] = {}
                project_files[project_id][file_path] = content
                
                await websocket.send_text(json.dumps({
                    "type": "file-saved",
                    "path": file_path
                }))
            
            elif message.get("type") == "run-project":
                # Placeholder for running project
                await websocket.send_text(json.dumps({
                    "type": "output",
                    "data": "Project running... (simulation)\nHello, Orca!"
                }))
            
            # Echo back for other types
            else:
                await websocket.send_text(json.dumps({
                    "type": "response",
                    "data": "processed"
                }))
            
    except WebSocketDisconnect:
        print(f"Client disconnected from project {project_id}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
