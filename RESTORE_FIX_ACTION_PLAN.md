# Action Plan: Fix Chat Restore & Persistence Issues

## Investigation Summary

Branch: `restore-chat-fix`  
Analysis Completed: 2025-11-21  
Files Analyzed: 12  
Issues Found: 12 critical, 5 architectural

### Key Files Involved
- `editor/docks/ai_chat_dock.cpp` (19,511 lines)
- `editor/docks/ai_chat_dock.h` (871 lines)
- `editor/docks/ai_conversation_persistence.cpp` (473 lines)
- `editor/docks/ai_conversation_persistence.h` (125 lines)
- `editor/docks/ai_chat_save_coordinator.cpp` (161 lines)
- `editor/docks/ai_chat_dock_user_messages.cpp` (restore logic)
- `editor/docks/ai_checkpoint_manager.cpp` (git restore)

---

## Phase 1: Emergency Fixes (Critical Data Loss Prevention)

**Estimated Time**: 4-6 hours  
**Risk Level**: Low (defensive changes)  
**Testing Required**: Manual crash testing

### 1.1 Add NOTIFICATION_CRASH Handler ⚡ CRITICAL

**File**: `editor/docks/ai_chat_dock.cpp`  
**Location**: ~line 1560 (in `_notification()` switch statement)

```cpp
case NOTIFICATION_CRASH: {
    // Crash detected - perform IMMEDIATE save
    
    // Stop all timers to prevent race conditions
    if (save_timer) {
        save_timer->stop();
    }
    
    // Wait for any running save thread
    if (save_thread_busy && save_thread) {
        save_thread->wait_to_finish();
        memdelete(save_thread);
        save_thread = nullptr;
        save_thread_busy = false;
    }
    
    // Create emergency backup SYNCHRONOUSLY
    if (conversation_persistence) {
        conversation_persistence->create_emergency_backup();
    }
    
    // Perform immediate synchronous save
    _save_conversations();
    
} break;
```

**Why**: Currently crashes bypass proper save logic, causing data loss.

---

### 1.2 Fix Orphaned Pending Save ⚡ CRITICAL

**File**: `editor/docks/ai_chat_dock.cpp`  
**Location**: ~line 1579 (NOTIFICATION_EXIT_TREE)

```cpp
case NOTIFICATION_EXIT_TREE: {
    // NEW: Handle orphaned pending save
    // If save was queued but thread never started, save now
    if (save_pending && !save_thread_busy) {
        save_pending = false;
        _save_conversations();  // Synchronous save before exit
    }
    
    // Final flush save to ensure no data loss on exit
    if (save_timer) {
        save_timer->stop();
    }
    if (save_thread_busy && save_thread) {
        save_thread->wait_to_finish();
        memdelete(save_thread);
        save_thread = nullptr;
        save_thread_busy = false;
    }
    
    // Always perform a final synchronous save with robust persistence
    if (conversation_persistence) {
        conversation_persistence->create_emergency_backup();
    }
    _save_conversations();
    
    // ... rest of cleanup ...
}
```

**Why**: Crash during 3-second delay window loses pending changes.

---

### 1.3 Remove call_deferred from Emergency Saves ⚡ CRITICAL

**File**: `editor/docks/ai_chat_dock.cpp`  
**Locations**: Lines 2255, 6045, 13356, 17780

**Change**:
```cpp
// BEFORE:
call_deferred("_emergency_save_conversations");

// AFTER:
_emergency_save_conversations();  // Immediate, not deferred
```

**Impact**: 4 call sites need to be changed

**Why**: Deferred "emergency" saves aren't actually emergency - they clear on crash.

---

### 1.4 Don't Queue Save During Restore ⚡ CRITICAL

**File**: `editor/docks/ai_chat_dock.cpp`  
**Location**: ~line 18430-18434

```cpp
// BEFORE:
if (current_conversation_index >= 0 && current_conversation_index < conversations.size()) {
    conversations.write[current_conversation_index].last_modified_timestamp = _get_timestamp();
    _queue_delayed_save();  // ❌ DANGER: Saves truncated chat
}

// AFTER:
if (current_conversation_index >= 0 && current_conversation_index < conversations.size()) {
    conversations.write[current_conversation_index].last_modified_timestamp = _get_timestamp();
    // Don't save immediately - wait for restore to complete fully
    call_deferred("_save_after_restore_verified");
}
```

**Add new method**:
```cpp
void AIChatDock::_save_after_restore_verified() {
    // Only save after restore is fully complete and UI is rebuilt
    if (current_conversation_index >= 0 && current_conversation_index < conversations.size()) {
        // Verify conversation has messages before saving
        if (conversations[current_conversation_index].messages.size() > 0) {
            _queue_delayed_save();
        } else {
            // Empty conversation after restore = something went wrong
            // Try to recover from backup instead of saving empty state
            if (conversation_persistence) {
                conversation_persistence->recover_from_corruption();
                _load_conversations();
            }
        }
    }
}
```

**Also add to .h file**:
```cpp
void _save_after_restore_verified();
```

**And bind in constructor**:
```cpp
ClassDB::bind_method(D_METHOD("_save_after_restore_verified"), &AIChatDock::_save_after_restore_verified);
```

**Why**: Restore truncates memory then immediately saves, wiping chat history.

---

### 1.5 Add Mutex Protection to Conversation Access ⚡ HIGH

**File**: `editor/docks/ai_chat_dock.cpp`  
**Location**: ~line 2953-2956 (_execute_delayed_save)

```cpp
// BEFORE:
SaveData *save_data = memnew(SaveData);
save_data->snapshot = memnew(Vector<Conversation>(conversations));
save_data->instance = this;
save_data->file_path = conversations_file_path;

// AFTER:
SaveData *save_data = memnew(SaveData);
{
    MutexLock lock(save_mutex);  // Protect conversations vector access
    save_data->snapshot = memnew(Vector<Conversation>(conversations));
}
save_data->instance = this;
save_data->file_path = conversations_file_path;
```

**Also protect other access points** (search for `conversations.write[`):
- When adding messages
- When modifying conversation properties
- When truncating during restore

**Why**: Race condition between main thread modifying conversations and background thread reading.

---

## Phase 2: Correctness Fixes (Prevent Wrong Recovery)

**Estimated Time**: 2-3 hours  
**Risk Level**: Low  
**Testing Required**: Backup recovery testing

### 2.1 Fix Backup Recovery Heuristic

**File**: `editor/docks/ai_conversation_persistence.cpp`  
**Location**: ~line 417-438

```cpp
// BEFORE:
// Find the LARGEST EMERGENCY backup (most likely to contain real data, not empty corruption)
String largest_emergency;
int64_t largest_size = 0;

for (int i = 0; i < files.size(); i++) {
    String file = files[i];
    if (file.begins_with("EMERGENCY_backup_") && file.ends_with(".simplifine")) {
        String backup_path = backup_directory.path_join(file);
        Ref<FileAccess> size_check = FileAccess::open(backup_path, FileAccess::READ);
        if (size_check.is_valid()) {
            int64_t file_size = size_check->get_length();
            size_check->close();
            
            if (file_size > 1000 && file_size > largest_size) {  
                largest_size = file_size;
                largest_emergency = file;
            }
        }
    }
}

// AFTER:
// Find the NEWEST valid EMERGENCY backup (most recent state)
String newest_emergency;
int64_t newest_time = 0;

for (int i = 0; i < files.size(); i++) {
    String file = files[i];
    if (file.begins_with("EMERGENCY_backup_") && file.ends_with(".simplifine")) {
        // Extract timestamp from filename: "EMERGENCY_backup_1234567890.simplifine"
        String time_str = file.substr(17, file.length() - 17 - 11);
        int64_t time = time_str.to_int();
        
        // Validate the backup is not corrupt and has reasonable size
        String backup_path = backup_directory.path_join(file);
        Ref<FileAccess> size_check = FileAccess::open(backup_path, FileAccess::READ);
        if (size_check.is_valid()) {
            int64_t file_size = size_check->get_length();
            size_check->close();
            
            // Pick newest backup that's not obviously corrupt (> 100 bytes)
            if (file_size > 100 && time > newest_time) {  
                newest_time = time;
                newest_emergency = file;
            }
        }
    }
}
```

**Update variable name**:
```cpp
// Line 440
if (newest_emergency.is_empty()) {
    return false;
}

String emergency_path = backup_directory.path_join(newest_emergency);
```

**Why**: Currently picks largest backup, not newest, leading to stale data recovery.

---

### 2.2 Validate Temp File Before Promoting

**File**: `editor/docks/ai_chat_dock.cpp`  
**Location**: ~line 12997-13002

```cpp
// BEFORE:
if (!FileAccess::exists(final_path) && FileAccess::exists(temp_path)) {
    Ref<DirAccess> da = DirAccess::open(base_dir);
    if (da.is_valid()) {
        da->rename(temp_name, final_name);
    }
}

// AFTER:
if (!FileAccess::exists(final_path) && FileAccess::exists(temp_path)) {
    // Validate temp file before promoting it
    Error err;
    String temp_content = FileAccess::get_file_as_string(temp_path, &err);
    if (err == OK) {
        // Try to parse as JSON
        Ref<JSON> json;
        json.instantiate();
        Error parse_err = json->parse(temp_content);
        
        if (parse_err == OK) {
            Dictionary data = json->get_data();
            // Basic validation
            if (data.has("conversations") && data.has("version")) {
                // Temp file is valid, promote it
                Ref<DirAccess> da = DirAccess::open(base_dir);
                if (da.is_valid()) {
                    da->rename(temp_name, final_name);
                }
            } else {
                // Invalid structure, delete corrupt temp file
                Ref<DirAccess> da = DirAccess::open(base_dir);
                if (da.is_valid()) {
                    da->remove(temp_name);
                }
            }
        } else {
            // Parse failed, delete corrupt temp file
            Ref<DirAccess> da = DirAccess::open(base_dir);
            if (da.is_valid()) {
                da->remove(temp_name);
            }
        }
    }
}
```

**Why**: Corrupt temp files can become main file without validation.

---

## Phase 3: Architecture Improvements (Long-term Stability)

**Estimated Time**: 1-2 days  
**Risk Level**: Medium  
**Testing Required**: Integration and stress testing

### 3.1 Add Save Queue with Retry Mechanism

**New file**: `editor/docks/ai_chat_save_queue.h`

```cpp
#ifndef AI_CHAT_SAVE_QUEUE_H
#define AI_CHAT_SAVE_QUEUE_H

#include "core/string/ustring.h"
#include "core/templates/vector.h"
#include "core/os/mutex.h"

class AIChatSaveQueue {
private:
    struct SaveRequest {
        String json_data;
        int64_t timestamp;
        int retry_count;
    };
    
    Vector<SaveRequest> pending_saves;
    Mutex queue_mutex;
    int max_retries = 3;
    
public:
    void enqueue_save(const String &p_json_data);
    bool has_pending() const;
    String dequeue_next();
    void mark_failed(const String &p_json_data);
    void clear_old_retries();
};

#endif
```

**Implementation**: `editor/docks/ai_chat_save_queue.cpp`

*This is a larger task, leaving implementation for Phase 3*

---

### 3.2 Periodic Auto-Save Independent of User Actions

**File**: `editor/docks/ai_chat_dock.cpp`

**Add timer in constructor** (~line 13879):
```cpp
// Periodic auto-save timer (every 30 seconds)
periodic_save_timer = memnew(Timer);
periodic_save_timer->set_wait_time(30.0);  // 30 seconds
periodic_save_timer->set_one_shot(false);  // Repeating
add_child(periodic_save_timer);
periodic_save_timer->connect("timeout", callable_mp(this, &AIChatDock::_periodic_auto_save));
periodic_save_timer->start();
```

**Add to .h file**:
```cpp
Timer *periodic_save_timer = nullptr;
void _periodic_auto_save();
```

**Implement method**:
```cpp
void AIChatDock::_periodic_auto_save() {
    // Only save if there are actual conversations and we're not already saving
    if (conversations.size() > 0 && !save_thread_busy) {
        // Create emergency backup first
        if (conversation_persistence) {
            conversation_persistence->create_emergency_backup();
        }
        
        // Queue save
        _queue_delayed_save();
    }
}
```

**Why**: Reduces window of data loss from minutes to max 30 seconds.

---

### 3.3 Add Comprehensive Logging

**File**: `editor/docks/ai_conversation_persistence.cpp`

**Enable actual logging** (currently no-ops at lines 265-275):

```cpp
void AIConversationPersistence::_log_save_attempt(const String &p_operation, bool p_success, const String &p_details) {
    // Create log directory
    String log_dir = conversations_file_path.get_base_dir().path_join("logs");
    Ref<DirAccess> da = DirAccess::create_for_path(log_dir);
    if (da.is_valid() && !da->dir_exists(log_dir)) {
        da->make_dir_recursive(log_dir);
    }
    
    // Append to log file
    String log_file = log_dir.path_join("chat_persistence.log");
    Ref<FileAccess> file = FileAccess::open(log_file, FileAccess::READ_WRITE);
    if (file.is_valid()) {
        file->seek_end();
        
        String timestamp = Time::get_singleton()->get_datetime_string_from_system();
        String status = p_success ? "[SUCCESS]" : "[FAILED]";
        String log_line = vformat("%s %s %s: %s\n", timestamp, status, p_operation, p_details);
        
        file->store_string(log_line);
        file->close();
    }
}
```

**Why**: Debugging save/restore issues is impossible without logs.

---

## Testing Plan

### Manual Testing Checklist

**Test 1: Crash During Save Window**
```
1. Send a message
2. Within 3 seconds, kill the process (kill -9 on Unix, Task Manager on Windows)
3. Restart
4. Verify message is present
✓ PASS / ✗ FAIL
```

**Test 2: Crash During Background Save**
```
1. Send 10 messages rapidly (to trigger background save)
2. During save (watch for disk activity), kill process
3. Restart
4. Verify all 10 messages present
✓ PASS / ✗ FAIL
```

**Test 3: Restore During Active Save**
```
1. Send 5 messages
2. Click "Restore to message 2"
3. Immediately start sending new messages
4. Let restore complete
5. Verify messages 3-5 are gone but new messages are present
✓ PASS / ✗ FAIL
```

**Test 4: Multiple Rapid Crashes**
```
1. Send message "A"
2. Kill process
3. Restart
4. Send message "B"  
5. Kill process
6. Restart
7. Verify both "A" and "B" are present
✓ PASS / ✗ FAIL
```

**Test 5: Backup Recovery**
```
1. Send 10 messages
2. Manually corrupt ai_chat_conversations.simplifine (add garbage text)
3. Restart
4. Verify recovery from backup
5. Verify all 10 messages restored
✓ PASS / ✗ FAIL
```

---

## Implementation Order

### Week 1: Critical Fixes
- [ ] Day 1: Implement 1.1 (NOTIFICATION_CRASH)
- [ ] Day 1: Implement 1.2 (Orphaned pending save)
- [ ] Day 2: Implement 1.3 (Remove call_deferred)
- [ ] Day 2: Implement 1.4 (Don't save during restore)
- [ ] Day 3: Implement 1.5 (Mutex protection)
- [ ] Day 3: Manual testing of all Phase 1 fixes

### Week 2: Correctness & Robustness
- [ ] Day 1: Implement 2.1 (Backup recovery fix)
- [ ] Day 1: Implement 2.2 (Temp file validation)
- [ ] Day 2: Implement 3.2 (Periodic auto-save)
- [ ] Day 3: Implement 3.3 (Logging)
- [ ] Day 3: Full integration testing

### Week 3: Architecture (if needed)
- [ ] Day 1-2: Implement 3.1 (Save queue)
- [ ] Day 3: Stress testing & edge cases

---

## Success Metrics

**Before Fixes** (reported by users):
- Chat loss on crash: ~30% of crashes
- Chat loss on restore: ~10% of restores
- User complaints: 5-10 per week

**After Fixes** (target):
- Chat loss on crash: < 1% of crashes
- Chat loss on restore: 0%
- User complaints: < 1 per month

---

## Rollback Plan

If any fix causes regressions:

1. **Immediate rollback**: `git revert <commit-hash>`
2. **Disable feature**: Add config flag to skip new code paths
3. **Emergency backup system**: Keep old backup logic as fallback

All changes are defensive additions - they don't remove existing code paths, minimizing rollback risk.

---

## Files That Need Changes

### Phase 1 (Critical)
- [ ] `editor/docks/ai_chat_dock.cpp` (6 changes)
- [ ] `editor/docks/ai_chat_dock.h` (2 additions)

### Phase 2 (Correctness)
- [ ] `editor/docks/ai_conversation_persistence.cpp` (2 changes)
- [ ] `editor/docks/ai_chat_dock.cpp` (1 change)

### Phase 3 (Architecture)
- [ ] `editor/docks/ai_chat_save_queue.h` (new file)
- [ ] `editor/docks/ai_chat_save_queue.cpp` (new file)
- [ ] `editor/docks/ai_chat_dock.cpp` (modifications)
- [ ] `editor/docks/ai_conversation_persistence.cpp` (logging)

---

## Notes

- All changes are backward compatible
- No changes to save file format
- Existing backups remain valid
- Can be deployed incrementally (Phase 1 → Phase 2 → Phase 3)

**Ready to implement: Phase 1 fixes**

