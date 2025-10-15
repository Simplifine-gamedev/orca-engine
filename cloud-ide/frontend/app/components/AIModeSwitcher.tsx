'use client'

import { useState } from 'react'

export type AIMode = 'ask' | 'agent'

interface AIModeSwithcerProps {
  currentMode: AIMode
  onModeChange: (mode: AIMode) => void
}

export default function AIModeSwitcher({ currentMode, onModeChange }: AIModeSwithcerProps) {
  return (
    <div className="flex items-center gap-2 bg-gray-800 rounded-lg p-1">
      <button
        onClick={() => onModeChange('ask')}
        className={`px-4 py-2 rounded-md text-sm font-medium transition-all ${
          currentMode === 'ask'
            ? 'bg-blue-600 text-white shadow-md'
            : 'text-gray-400 hover:text-gray-200'
        }`}
      >
        <span className="flex items-center gap-2">
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          Ask Mode
        </span>
      </button>
      
      <button
        onClick={() => onModeChange('agent')}
        className={`px-4 py-2 rounded-md text-sm font-medium transition-all ${
          currentMode === 'agent'
            ? 'bg-purple-600 text-white shadow-md'
            : 'text-gray-400 hover:text-gray-200'
        }`}
      >
        <span className="flex items-center gap-2">
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 4a2 2 0 114 0v1a1 1 0 001 1h3a1 1 0 011 1v3a1 1 0 01-1 1h-1a2 2 0 100 4h1a1 1 0 011 1v3a1 1 0 01-1 1h-3a1 1 0 01-1-1v-1a2 2 0 10-4 0v1a1 1 0 01-1 1H7a1 1 0 01-1-1v-3a1 1 0 00-1-1H4a2 2 0 110-4h1a1 1 0 001-1V7a1 1 0 011-1h3a1 1 0 001-1V4z" />
          </svg>
          Agent Mode
        </span>
      </button>
    </div>
  )
}

export function ModeDescription({ mode }: { mode: AIMode }) {
  return (
    <div className="text-xs text-gray-400 mt-2 px-1">
      {mode === 'ask' ? (
        <div className="flex items-start gap-2">
          <svg className="w-3 h-3 mt-0.5 flex-shrink-0 text-blue-500" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
          </svg>
          <span>AI can read files and answer questions but won't modify your code</span>
        </div>
      ) : (
        <div className="flex items-start gap-2">
          <svg className="w-3 h-3 mt-0.5 flex-shrink-0 text-purple-500" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M6.267 3.455a3.066 3.066 0 001.745-.723 3.066 3.066 0 013.976 0 3.066 3.066 0 001.745.723 3.066 3.066 0 012.812 2.812c.051.643.304 1.254.723 1.745a3.066 3.066 0 010 3.976 3.066 3.066 0 00-.723 1.745 3.066 3.066 0 01-2.812 2.812 3.066 3.066 0 00-1.745.723 3.066 3.066 0 01-3.976 0 3.066 3.066 0 00-1.745-.723 3.066 3.066 0 01-2.812-2.812 3.066 3.066 0 00-.723-1.745 3.066 3.066 0 010-3.976 3.066 3.066 0 00.723-1.745 3.066 3.066 0 012.812-2.812zm7.44 5.252a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
          </svg>
          <span>AI can read, write, and modify your project files autonomously</span>
        </div>
      )}
    </div>
  )
}
