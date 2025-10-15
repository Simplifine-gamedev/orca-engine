# AI Ask/Agent Modes Implementation

## Overview
This feature implements two distinct AI interaction modes for the Orca Cloud IDE:

### 🔵 Ask Mode
- **Purpose**: Read-only AI assistant for code questions and guidance
- **Capabilities**: 
  - Read and analyze project files
  - Answer questions about code
  - Provide suggestions and explanations
  - Search through codebase
- **Restrictions**: Cannot modify, write, or delete any files

### 🟣 Agent Mode (Default)
- **Purpose**: Full-featured AI coding assistant
- **Capabilities**:
  - All Ask Mode capabilities PLUS:
  - Write and edit project files
  - Create new files and directories
  - Refactor code
  - Implement features autonomously
  - Make project-wide changes

## Components

### 1. AIModeSwitcher (`app/components/AIModeSwitcher.tsx`)
A toggle component that allows users to switch between Ask and Agent modes.

**Features:**
- Visual toggle with distinct colors (blue for Ask, purple for Agent)
- Icons for each mode
- Mode description tooltip
- Keyboard accessible

**Props:**
```typescript
interface AIModeSwithcerProps {
  currentMode: AIMode  // 'ask' | 'agent'
  onModeChange: (mode: AIMode) => void
}
```

### 2. AIChatPanel (`app/components/AIChatPanel.tsx`)
Chat interface for interacting with the AI assistant.

**Features:**
- Real-time chat interface
- Mode-aware UI (changes color based on mode)
- Message history
- Loading states
- Keyboard shortcuts (Enter to send, Shift+Enter for new line)
- Empty state with helpful prompts

**Props:**
```typescript
interface AIChatPanelProps {
  mode: AIMode
  projectId: string | null
}
```

### 3. Main IDE Integration (`app/page.tsx`)
The AI components are integrated into the main IDE interface:
- Mode switcher in the top toolbar
- Chat panel in the right sidebar (replacing Inspector)
- State management for AI mode
- WebSocket integration for backend communication

## Usage

### Switching Modes
1. Click the mode toggle in the top toolbar
2. Choose "Ask Mode" for read-only assistance
3. Choose "Agent Mode" for full AI capabilities

### Chatting with AI
1. Type your message in the input box
2. Press Enter to send (Shift+Enter for multi-line)
3. View AI responses in the chat panel

### Mode-Specific Behavior

**In Ask Mode:**
- AI can read and analyze your code
- Ask: "What does this function do?"
- Ask: "How can I improve this code?"
- Ask: "Where is the player movement logic?"

**In Agent Mode:**
- AI can make changes to your code
- Request: "Add error handling to this function"
- Request: "Refactor this component to use hooks"
- Request: "Create a new player class"

## Backend Integration

The frontend sends mode information to the backend via WebSocket:

```typescript
socket.emit('ai-mode-change', { mode: 'ask' | 'agent' })
```

The backend should:
1. Store the current mode per user/session
2. Restrict write operations in Ask mode
3. Allow all operations in Agent mode

## Styling

The design follows Cursor's style guide:
- **Ask Mode**: Blue theme (#3B82F6)
- **Agent Mode**: Purple theme (#9333EA)
- Dark mode UI with gray-800/900 backgrounds
- Smooth transitions and hover states
- Accessible contrast ratios

## Future Enhancements

1. **Mode Persistence**: Save user's mode preference
2. **Mode-specific suggestions**: Different prompts per mode
3. **Activity History**: Track what AI has done in Agent mode
4. **Undo/Redo**: Revert AI changes in Agent mode
5. **Approval Flow**: Require approval for sensitive operations
6. **File-level permissions**: Restrict certain files even in Agent mode

## Testing

To test the implementation:

```bash
cd /workspace/cloud-ide/frontend
npm install
npm run dev
```

Navigate to `http://localhost:3000` and:
1. Verify mode switcher appears in toolbar
2. Click between Ask and Agent modes
3. Open the AI chat panel
4. Send test messages
5. Verify mode indicator updates correctly

## Build

```bash
npm run build  # Production build
npm start      # Start production server
```

## Architecture Notes

- **State Management**: React hooks (useState) for local state
- **Communication**: Socket.io for real-time backend communication
- **Styling**: Tailwind CSS for responsive design
- **Type Safety**: TypeScript with proper type definitions
- **Component Structure**: Modular, reusable components

## Security Considerations

1. **Backend Validation**: Backend must enforce mode restrictions
2. **Frontend is UI Only**: Mode switching in frontend is for UX, not security
3. **Authentication Required**: All AI operations require valid auth
4. **Rate Limiting**: Prevent abuse of AI endpoints
5. **Audit Logging**: Track all file modifications in Agent mode
