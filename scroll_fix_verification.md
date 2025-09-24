# AI Chat Auto-Scroll Fix Verification

## Issue Description
The AI chat was forcing scroll to bottom during streaming responses, making it impossible for users to scroll up and read previous messages while the agent was still typing.

## Root Cause Analysis
The `_on_chat_content_min_size_changed()` function was using a cached `auto_scroll_at_bottom` state that wasn't updated during streaming content changes. This caused forced scrolling even when users had manually scrolled up.

## Fix Implementation
Modified `_on_chat_content_min_size_changed()` in `/workspace/editor/docks/ai_chat_dock.cpp` to:

1. **Real-time Position Check**: Instead of relying on cached state, check actual scroll position using `_is_at_bottom()`
2. **Conditional Scrolling**: Only auto-scroll if user is genuinely at the bottom
3. **State Synchronization**: Update cached `auto_scroll_at_bottom` to keep it in sync

## Code Changes
```cpp
// Line 3966-3976 in ai_chat_dock.cpp
void AIChatDock::_on_chat_content_min_size_changed() {
    // Only auto-scroll when the user is already at the bottom.
    // Check current position in real-time to avoid forcing scroll during streaming
    // when user has scrolled up to read previous messages.
    bool is_at_bottom = _is_at_bottom();
    if (is_at_bottom) {
        call_deferred("_scroll_to_bottom");
    }
    // Update the cached state to keep it in sync
    auto_scroll_at_bottom = is_at_bottom;
}
```

## Expected Behavior After Fix
- ✅ User sends message → Auto-scroll enabled (normal behavior)
- ✅ During streaming → Only scrolls if user is at bottom
- ✅ User scrolls up during streaming → Scroll position preserved
- ✅ User scrolls back to bottom → Auto-scroll re-enabled
- ✅ Streaming completes → Normal scrolling behavior maintained

## Testing Strategy
1. Start a conversation with the AI
2. Send a message that will generate a long response
3. While the response is streaming, try to scroll up to read previous messages
4. Verify that the scroll position is maintained and not forced back to bottom
5. Scroll back to bottom manually and verify auto-scroll resumes

## Files Modified
- `/workspace/editor/docks/ai_chat_dock.cpp` (lines 3966-3976)

## Linear Issue Resolution
This fix resolves Linear issue ORC-44: "ai chat force scrolls to the bottom when a new agent message appears"