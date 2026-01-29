'use client'

interface AIModeSwitcherProps {
  mode: 'ask' | 'agent'
  onChange: (mode: 'ask' | 'agent') => void
}

export default function AIModeSwitcher({ mode, onChange }: AIModeSwitcherProps) {
  return (
    <div className="flex items-center gap-3 px-4 py-2 bg-gray-800 rounded-lg border border-gray-700">
      <span className="text-sm font-medium text-gray-300">AI Mode:</span>
      
      <div className="flex gap-2">
        <button
          onClick={() => onChange('ask')}
          className={`px-4 py-2 rounded-md text-sm font-medium transition-all ${
            mode === 'ask'
              ? 'bg-blue-600 text-white shadow-lg'
              : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
          }`}
        >
          Ask Mode
        </button>
        
        <button
          onClick={() => onChange('agent')}
          className={`px-4 py-2 rounded-md text-sm font-medium transition-all ${
            mode === 'agent'
              ? 'bg-purple-600 text-white shadow-lg'
              : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
          }`}
        >
          Agent Mode
        </button>
      </div>
      
      <div className="text-xs text-gray-400 ml-2">
        {mode === 'ask' ? (
          <span>Read-only - AI answers questions without modifying files</span>
        ) : (
          <span>Full access - AI can write and edit code</span>
        )}
      </div>
    </div>
  )
}
