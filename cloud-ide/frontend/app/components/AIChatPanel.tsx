'use client'

import { useState, useRef, useEffect } from 'react'
import axios from 'axios'

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

interface Message {
  role: 'user' | 'assistant'
  content: string
  timestamp: Date
}

interface AIChatPanelProps {
  mode: 'ask' | 'agent'
  projectId: string | null
}

export default function AIChatPanel({ mode, projectId }: AIChatPanelProps) {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const sendMessage = async () => {
    if (!input.trim() || loading) return

    const userMessage: Message = {
      role: 'user',
      content: input,
      timestamp: new Date()
    }

    setMessages(prev => [...prev, userMessage])
    setInput('')
    setLoading(true)

    try {
      const response = await axios.post(`${API_URL}/chat`, {
        messages: [
          ...messages.map(m => ({ role: m.role, content: m.content })),
          { role: 'user', content: input }
        ],
        mode: mode,
        project_id: projectId
      })

      // Handle streaming response
      const reader = response.data.getReader()
      let assistantMessage = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        const chunk = new TextDecoder().decode(value)
        const lines = chunk.split('\n').filter(line => line.trim())

        for (const line of lines) {
          try {
            const data = JSON.parse(line)
            if (data.content) {
              assistantMessage += data.content
            }
          } catch (e) {
            // Skip invalid JSON lines
          }
        }
      }

      const aiMessage: Message = {
        role: 'assistant',
        content: assistantMessage || 'No response received',
        timestamp: new Date()
      }

      setMessages(prev => [...prev, aiMessage])
    } catch (error) {
      console.error('Failed to send message:', error)
      
      const errorMessage: Message = {
        role: 'assistant',
        content: 'Sorry, I encountered an error. Please try again.',
        timestamp: new Date()
      }
      
      setMessages(prev => [...prev, errorMessage])
    } finally {
      setLoading(false)
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  const modeColor = mode === 'ask' ? 'blue' : 'purple'

  return (
    <div className="flex flex-col h-full bg-gray-800 border-l border-gray-700">
      {/* Header */}
      <div className={`px-4 py-3 bg-${modeColor}-900 border-b border-gray-700`}>
        <h2 className="text-lg font-semibold text-white flex items-center gap-2">
          <span className={`w-3 h-3 rounded-full bg-${modeColor}-500 animate-pulse`}></span>
          AI Assistant
          <span className="text-xs font-normal text-gray-300 ml-2">
            ({mode === 'ask' ? 'Ask Mode' : 'Agent Mode'})
          </span>
        </h2>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 ? (
          <div className="flex items-center justify-center h-full text-gray-500">
            <div className="text-center">
              <div className="text-xl mb-2">
                {mode === 'ask' ? '💬 Ask me anything' : '🤖 Ready to help'}
              </div>
              <div className="text-sm">
                {mode === 'ask' 
                  ? 'I can answer questions and help you understand your code'
                  : 'I can write code, edit files, and help build your project'
                }
              </div>
            </div>
          </div>
        ) : (
          messages.map((msg, idx) => (
            <div
              key={idx}
              className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div
                className={`max-w-[80%] rounded-lg px-4 py-2 ${
                  msg.role === 'user'
                    ? 'bg-blue-600 text-white'
                    : `bg-gray-700 text-gray-100`
                }`}
              >
                <div className="text-sm whitespace-pre-wrap">{msg.content}</div>
                <div className="text-xs mt-1 opacity-70">
                  {msg.timestamp.toLocaleTimeString()}
                </div>
              </div>
            </div>
          ))
        )}
        
        {loading && (
          <div className="flex justify-start">
            <div className="bg-gray-700 rounded-lg px-4 py-2">
              <div className="flex items-center gap-2 text-gray-300">
                <div className="animate-pulse">Thinking...</div>
                <div className="flex gap-1">
                  <span className="animate-bounce" style={{ animationDelay: '0ms' }}>.</span>
                  <span className="animate-bounce" style={{ animationDelay: '150ms' }}>.</span>
                  <span className="animate-bounce" style={{ animationDelay: '300ms' }}>.</span>
                </div>
              </div>
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="p-4 border-t border-gray-700">
        <div className="flex gap-2">
          <textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={mode === 'ask' ? 'Ask a question...' : 'Tell me what to do...'}
            className="flex-1 bg-gray-700 text-white rounded-lg px-4 py-2 resize-none focus:outline-none focus:ring-2 focus:ring-blue-500"
            rows={3}
            disabled={loading}
          />
          <button
            onClick={sendMessage}
            disabled={loading || !input.trim()}
            className={`px-6 py-2 rounded-lg font-medium transition-colors ${
              loading || !input.trim()
                ? 'bg-gray-600 text-gray-400 cursor-not-allowed'
                : `bg-${modeColor}-600 hover:bg-${modeColor}-700 text-white`
            }`}
          >
            Send
          </button>
        </div>
        <div className="text-xs text-gray-400 mt-2">
          Press Enter to send, Shift+Enter for new line
        </div>
      </div>
    </div>
  )
}
