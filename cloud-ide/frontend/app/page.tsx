'use client'

import { useState, useEffect, useRef } from 'react'
import MonacoEditor from '@monaco-editor/react'
import io from 'socket.io-client'
import axios from 'axios'

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

export default function OrcaCloudIDE() {
  const [projectId, setProjectId] = useState<string | null>(null)
  const [vncUrl, setVncUrl] = useState<string | null>(null)
  const [socket, setSocket] = useState<any>(null)
  const [files, setFiles] = useState<any[]>([])
  const [currentFile, setCurrentFile] = useState('')
  const [code, setCode] = useState('')
  const [loading, setLoading] = useState(false)
  const iframeRef = useRef<HTMLIFrameElement>(null)

  // Create or load project on mount
  useEffect(() => {
    initProject()
  }, [])

  const initProject = async () => {
    setLoading(true)
    try {
      // Create new project (or load existing from URL params)
      const response = await axios.post(`${API_URL}/api/projects`, {
        name: 'My Orca Project',
        template: 'empty'
      })
      
      setProjectId(response.data.project_id)
      setVncUrl(response.data.vnc_url)
      
      // Connect WebSocket
      const ws = io(`${API_URL}/ws/${response.data.project_id}`)
      ws.on('connect', () => {
        console.log('Connected to backend')
      })
      ws.on('file-update', (data: any) => {
        // Handle file updates from other users
        if (data.file === currentFile) {
          setCode(data.content)
        }
      })
      setSocket(ws)
      
    } catch (error) {
      console.error('Failed to initialize project:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleCodeChange = (value: string | undefined) => {
    if (!value) return
    setCode(value)
    
    // Send to backend
    if (socket) {
      socket.emit('code-change', {
        file: currentFile,
        content: value
      })
    }
  }

  const handleFileSelect = (file: any) => {
    setCurrentFile(file.path)
    // Load file content from backend
    if (socket) {
      socket.emit('load-file', { path: file.path })
    }
  }

  const handleRunProject = () => {
    if (socket) {
      socket.emit('run-project')
    }
  }

  const handleBuildProject = () => {
    if (socket) {
      socket.emit('build-project')
    }
  }

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center h-screen bg-gray-900 text-white">
        <img src="/logo-light.png" alt="Orca" className="h-16 w-auto mb-4 animate-pulse" />
        <div className="text-2xl">Starting Orca Engine...</div>
      </div>
    )
  }

  return (
    <div className="flex h-screen bg-gray-900 text-white">
      {/* File Explorer */}
      <div className="w-64 bg-gray-800 border-r border-gray-700 p-4">
        <div className="mb-4">
          <h2 className="text-lg font-semibold mb-2">Project Files</h2>
          <button 
            className="w-full bg-blue-600 hover:bg-blue-700 px-3 py-1 rounded text-sm"
            onClick={() => {/* TODO: Add new file */}}
          >
            + New File
          </button>
        </div>
        <div className="space-y-1">
          {files.map((file, idx) => (
            <div
              key={idx}
              className="px-2 py-1 hover:bg-gray-700 cursor-pointer rounded"
              onClick={() => handleFileSelect(file)}
            >
              {file.name}
            </div>
          ))}
        </div>
      </div>

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col">
        {/* Toolbar */}
        <div className="bg-gray-800 border-b border-gray-700 p-2 flex items-center gap-2">
          <img src="/logo-light.png" alt="Orca" className="h-6 w-auto mr-2" />
          <button 
            className="bg-green-600 hover:bg-green-700 px-4 py-1 rounded text-sm"
            onClick={handleRunProject}
          >
            ▶ Run
          </button>
          <button 
            className="bg-blue-600 hover:bg-blue-700 px-4 py-1 rounded text-sm"
            onClick={handleBuildProject}
          >
            Build
          </button>
          <div className="ml-auto text-sm text-gray-400">
            {currentFile || 'No file selected'}
          </div>
        </div>

        <div className="flex-1 flex">
          {/* Code Editor */}
          <div className="w-1/2 border-r border-gray-700">
            <MonacoEditor
              height="100%"
              language="gdscript"
              theme="vs-dark"
              value={code}
              onChange={handleCodeChange}
              options={{
                minimap: { enabled: false },
                fontSize: 14
              }}
            />
          </div>

          {/* 3D Viewport (VNC) */}
          <div className="w-1/2 bg-black relative">
            {vncUrl ? (
              <iframe
                ref={iframeRef}
                src={`${vncUrl}?autoconnect=true&resize=scale&quality=6`}
                className="w-full h-full"
                style={{ border: 'none' }}
              />
            ) : (
              <div className="flex items-center justify-center h-full text-gray-500">
                <div className="text-center">
                  <div className="text-xl mb-2">Viewport</div>
                  <div className="text-sm">Connecting to Orca Engine...</div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Properties Panel */}
        <div className="h-48 bg-gray-800 border-t border-gray-700 p-4">
          <h3 className="text-sm font-semibold mb-2">Properties</h3>
          <div className="text-sm text-gray-400">
            Select an object to view properties
          </div>
        </div>
      </div>

      {/* Right Panel - Inspector */}
      <div className="w-80 bg-gray-800 border-l border-gray-700 p-4">
        <h2 className="text-lg font-semibold mb-4">Inspector</h2>
        <div className="space-y-4">
          <div>
            <h3 className="text-sm font-semibold mb-2">Transform</h3>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span>Position</span>
                <span className="text-gray-400">0, 0, 0</span>
              </div>
              <div className="flex justify-between">
                <span>Rotation</span>
                <span className="text-gray-400">0, 0, 0</span>
              </div>
              <div className="flex justify-between">
                <span>Scale</span>
                <span className="text-gray-400">1, 1, 1</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
