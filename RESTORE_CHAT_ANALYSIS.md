# Comprehensive Analysis: Chat Restore & Persistence Issues

## Executive Summary

After thorough investigation of the chat save/load and restore mechanics in Orca Engine, I've identified **12 critical issues** that could cause chat data loss, especially after crashes. The system has multiple layers of protection (backups, atomic writes, recovery), but several race conditions, thread safety issues, and logic gaps can cause data to be wiped or not properly saved.

---

## Critical Issues Identified

### 1. **Thread Race Condition on Crash** 🔴 CRITICAL
**Location**: `editor/docks/ai_chat_dock.cpp:1579-1594` (NOTIFICATION_EXIT_TREE)

**Problem**: 
- When the app crashes or exits, the exit handler waits for `save_thread` to finish
- However, if a crash occurs BETWEEN `save_pending = true` (line 2917) and `save_thread` starting (line 2959), no save thread exists yet
- The flag `save_pending` is set but never acted upon
- **Result**: Pending changes are lost

```cpp
// Line 2911-2927: Queue delayed save
void AIChatDock::_queue_delayed_save() {
    if (save_pending) {
        return;  // Already pending
    }
    
    save_pending = true;
    if (save_timer) {
        save_timer->stop();
        save_timer->start(3.0); // wait 3 seconds
    }
    call_deferred("_execute_delayed_save");  // <-- CRASH HERE = data loss
}

// Line 1579-1594: Exit handler
case NOTIFICATION_EXIT_TREE: {
    if (save_thread_busy && save_thread) {  // <-- If no thread started yet, skip!
        save_thread->wait_to_finish();
    }
    // ...but save_pending=true with no save happening
}
```

**Impact**: High - Any crash within 3-second delay window loses data

---

### 2. **Missing Crash Notification Handler** 🔴 CRITICAL
**Location**: `editor/docks/ai_chat_dock.cpp` (notification handler)

**Problem**:
- Platform crash handlers send `NOTIFICATION_CRASH` to MainLoop (confirmed in `platform/*/crash_handler_*.cpp`)
- AIChatDock's `_notification()` handler (line 1560-1617) only handles `NOTIFICATION_EXIT_TREE` 
- **NO handler for `NOTIFICATION_CRASH`**
- On crash, the normal exit cleanup runs, but doesn't handle the pending save state properly

**Missing Code**:
```cpp
case NOTIFICATION_CRASH: {
    // MISSING: Immediate synchronous save
    // MISSING: Flush pending saves
    // MISSING: Force write without delay
} break;
```

**Impact**: Critical - Crashes bypass emergency save logic

---

### 3. **Deferred Save During Restore Wipes Data** 🔴 CRITICAL
**Location**: `editor/docks/ai_chat_dock.cpp:18430-18433`

**Problem**:
- During checkpoint restore, chat messages are deleted from memory (line 18412-18414)
- Then `_queue_delayed_save()` is called (line 18433)
- If save executes before chat is properly reloaded, **empty/truncated chat is saved to disk**

```cpp
// Line 18380-18433: Restore from checkpoint
bool AIChatDock::_restore_from_checkpoint(int p_message_index) {
    // ...
    // Delete messages from memory
    while (chat_history.size() > p_message_index + 1) {
        chat_history.remove_at(chat_history.size() - 1);  // <-- Data gone from memory
    }
    
    // ...UI rebuild...
    
    // Queue save with truncated data
    if (current_conversation_index >= 0 && current_conversation_index < conversations.size()) {
        conversations.write[current_conversation_index].last_modified_timestamp = _get_timestamp();
        _queue_delayed_save();  // <-- DANGER: Saves truncated chat!
    }
}
```

**The "Restore Only" Path is WORSE**:
In `ai_chat_dock_user_messages.cpp:400-430`, the code:
1. Backs up chat file (line 401-411)
2. Calls `_restore_from_checkpoint()` which TRUNCATES memory (line 414)
3. Writes backup back to disk (line 421-424)
4. Reloads conversations (line 427)

**BUT**: Between steps 3-4, if a save timer fires or crash happens, the **reloaded truncated chat overwrites the backup**!

**Impact**: Very High - Restore operations can wipe chat history

---

### 4. **Emergency Save Not Guaranteed Before Thread Start** 🔴 CRITICAL
**Location**: `editor/docks/ai_chat_dock.cpp:2924-2926`

**Problem**:
```cpp
call_deferred("_execute_delayed_save");  // Called AFTER setting save_pending
```

- `call_deferred` schedules for next idle frame
- If crash occurs before next idle frame, emergency backup (line 1592) runs but `_execute_delayed_save` never runs
- The `save_pending = true` state is orphaned

**Impact**: Medium-High - Narrow crash window, but guaranteed data loss

---

### 5. **Dual Save State Tracking Without Coordination** 🟡 MAJOR
**Location**: Multiple locations

**Problem**: Two separate save-in-progress flags:
1. `AIChatDock::save_thread_busy` (line 2944)
2. `AIConversationPersistence::save_in_progress` (line 280)

These are NOT synchronized or coordinated. Race conditions:

**Scenario**:
1. Thread A: Sets `save_thread_busy = true`, starts background thread
2. Background thread: Calls `persistence->save_conversations_with_validation()`
3. Persistence: `save_in_progress = true`
4. **Main thread**: User triggers action → `_queue_delayed_save()`
5. Timer expires → `_execute_delayed_save()` checks `save_thread_busy` (still true)
6. Reschedules for 0.5 seconds
7. Background thread finishes: `save_in_progress = false` BUT `save_thread_busy` stays true until `_background_save_complete()`
8. **Gap**: Main thread sees `save_thread_busy = false`, starts new save
9. **Conflict**: New save reaches persistence while `save_in_progress` cleanup isn't done

**Impact**: Medium - Can cause data loss if second save overwrites first

---

### 6. **Backup Recovery Uses Wrong Heuristic** 🟡 MAJOR
**Location**: `editor/docks/ai_conversation_persistence.cpp:404-472`

**Problem**: 
```cpp
// Line 417-435: Find LARGEST emergency backup
for (int i = 0; i < files.size(); i++) {
    String file = files[i];
    if (file.begins_with("EMERGENCY_backup_") && file.ends_with(".simplifine")) {
        // ...
        if (file_size > 1000 && file_size > largest_size) {  // Pick LARGEST
            largest_size = file_size;
            largest_emergency = file;
        }
    }
}
```

**Issue**: Uses **file size** not **timestamp** to pick backup
- If user had long chat history, then deleted messages → newer backup is SMALLER
- Recovery picks **older, larger** backup instead of **newest** backup
- **Result**: User loses recent work, gets old state restored

**Should be**:
```cpp
// Extract timestamp from filename and pick NEWEST, not LARGEST
String time_str = file.substr(17, file.length() - 17 - 11);  // Extract timestamp
int64_t time = time_str.to_int();
if (time > newest_time) {
    newest_time = time;
    newest_emergency = file;
}
```

**Impact**: Medium - Wrong backup restored after corruption

---

### 7. **Load Path Inconsistency Can Cause Split Brain** 🟡 MAJOR
**Location**: `editor/docks/ai_chat_dock.cpp:1319-1350` (init) and `12984-13014` (load)

**Problem**: Two separate file path determination logics that can diverge:

**Init Path** (line 1321):
```cpp
String new_path = EditorPaths::get_singleton()->get_project_settings_dir()
    .path_join("ai_chat_conversations.simplifine");
conversations_file_path = new_path;
```

**Load Path** (line 12993-12995):
```cpp
String alt_final_path = EditorPaths::get_singleton()->get_project_settings_dir()
    .path_join("ai_chat_conversations.simplifine");
```

**Issue**: 
- If `EditorPaths::get_singleton()` state changes between init and load (rare but possible)
- Or if project settings directory is remounted/changed
- Saves go to one location, loads from another
- **Result**: Appears like chats were wiped, but they're in different file

**Impact**: Low-Medium - Rare but confusing when it happens

---

### 8. **Temp File Recovery Logic Incomplete** 🟠 MODERATE
**Location**: `editor/docks/ai_chat_dock.cpp:12997-13002`

**Problem**:
```cpp
// Line 12997-13002: Temp file recovery
if (!FileAccess::exists(final_path) && FileAccess::exists(temp_path)) {
    Ref<DirAccess> da = DirAccess::open(base_dir);
    if (da.is_valid()) {
        da->rename(temp_name, final_name);
    }
}
```

**Issues**:
1. No validation of temp file contents before promoting to final
2. If temp file is corrupt (from interrupted save), corrupt data becomes main file
3. No backup of old final file before overwriting
4. Error handling: If `rename()` fails, **no fallback** - just silent failure

**Impact**: Medium - Can restore corrupt data after crash

---

### 9. **Emergency Save Called via call_deferred (Not Immediate)** 🟠 MODERATE
**Location**: Multiple locations, e.g., `editor/docks/ai_chat_dock.cpp:17780`

**Problem**:
```cpp
// Line 17778-17781
if (conversation_persistence) {
    call_deferred("_emergency_save_conversations");  // <-- DEFERRED!
}
```

**Emergency save should be SYNCHRONOUS for true emergency protection**. Using `call_deferred`:
- Waits for next idle frame
- If crash/hang happens immediately after, deferred call never executes
- Defeats the purpose of "emergency" save

**Impact**: Medium - Emergency saves aren't actually emergency

---

### 10. **Background Thread Snapshot Race Condition** 🟠 MODERATE
**Location**: `editor/docks/ai_chat_dock.cpp:2946-2960`

**Problem**:
```cpp
// Line 2953-2956: Create snapshot
SaveData *save_data = memnew(SaveData);
save_data->snapshot = memnew(Vector<Conversation>(conversations));  // <-- COPY
save_data->instance = this;
save_data->file_path = conversations_file_path;

save_thread = memnew(Thread);
save_thread->start(_background_save, save_data);
```

**Race Condition**:
1. Main thread: Creates snapshot of `conversations` vector at time T1
2. Main thread: Starts background thread
3. User: Sends new message at time T2 (after snapshot)
4. Main thread: Adds message to `conversations` vector
5. Background thread: Still saving snapshot from T1 (doesn't include T2 message)
6. Background thread: Finishes save - **T2 message is now only in memory**
7. **Crash**: T2 message never saved

**The "save_pending" flag should be checked but isn't**:
- If `save_pending` becomes true again DURING background save
- AND crash happens before next save
- Changes are lost

**Impact**: Medium - Narrow window but guaranteed data loss

---

### 11. **Restore Overwrites Pending Save Data** 🟠 MODERATE
**Location**: `editor/docks/ai_conversation_persistence.cpp:278-309`

**Problem**:
```cpp
// Line 280-283
if (save_in_progress) {
    _log_save_attempt("safe_save", false, "Save already in progress, storing data for later");
    pending_save_data = p_json_data; // Store for retry
    return SAVE_ERROR_ALREADY_IN_PROGRESS;
}
```

**Issue**: `pending_save_data` is stored but **NEVER RETRIED**
- No mechanism to process pending data after current save completes
- `pending_save_data` just sits there until next save call
- If restore happens, new save **overwrites** `pending_save_data`
- **Result**: Queued save data is lost forever

**Impact**: Medium - Lost saves during high activity

---

### 12. **No Mutex Protection for conversations Vector** 🟠 MODERATE
**Location**: Throughout `ai_chat_dock.cpp`

**Problem**:
- `save_mutex` exists (line 281 in .h) but is **NEVER USED**
- Main thread modifies `conversations` vector
- Background thread reads `conversations` via snapshot
- **No synchronization between them**

**Race example**:
```cpp
// Main thread:
conversations.write[index].messages.push_back(new_message);  // Modifying

// Background thread (simultaneously):
Vector<Conversation>(conversations)  // Reading for snapshot
```

This is **undefined behavior** in C++ - could cause:
- Partial/corrupt data in snapshot
- Crash from iterator invalidation
- Silent data corruption

**Impact**: Low-Medium - Depends on timing, but can corrupt saves

---

## Architectural Issues

### A. **Multiple Save Paths Without Coordination**
The codebase has at least 4 different save mechanisms:
1. `_save_conversations()` - synchronous save (line 13218)
2. `_save_conversations_chunked()` - chunked save (line 2814)
3. `_background_save()` - threaded save (line 2962)
4. `_save_conversations_to_disk()` - direct disk write (line 2747)

Each has slightly different logic, error handling, and failure modes. This increases maintenance burden and bug surface area.

### B. **Inconsistent Error Handling**
- Some save failures are silent (no logging)
- Some print error messages that get swallowed
- No centralized error recovery strategy
- No user notification on save failures

### C. **Backup System Limitations**
- Only keeps 10 backups (MAX_BACKUP_FILES = 10)
- No size-based retention policy
- Emergency backups mixed with regular backups in recovery
- No distinction between "good" backups and "crash" backups

---

## Attack Scenarios (How Chat Gets Wiped)

### Scenario 1: **Crash During Delayed Save Window**
1. User sends message
2. `_queue_delayed_save()` called - sets `save_pending = true`
3. Timer starts (3 seconds)
4. **CRASH at 1.5 seconds**
5. `NOTIFICATION_EXIT_TREE` handler runs
6. Checks `save_thread_busy` → false (thread never started)
7. Skips save wait
8. Calls `_save_conversations()` synchronously
9. But if conversations vector was in inconsistent state → corrupt save
10. **Result**: Chat appears empty or truncated on restart

### Scenario 2: **Restore During Pending Save**
1. User requests checkpoint restore
2. `_restore_from_checkpoint()` called
3. Deletes messages from memory (line 18412-18414)
4. Meanwhile, background save thread is running from earlier action
5. Background thread finishes, writes **truncated** conversation to disk
6. Restore logic reloads from truncated disk file
7. **Result**: Messages after checkpoint are permanently lost

### Scenario 3: **Rapid Crashes During Recovery**
1. Initial crash with pending save
2. Startup: Attempts to load conversations
3. File is corrupt (interrupted save)
4. Triggers `recover_from_corruption()`
5. Finds emergency backup, starts restore
6. **SECOND CRASH during restore**
7. Emergency backup is partially copied
8. On third startup: Both main file and backup are corrupt
9. **Result**: Total data loss

---

## Recommendations

### Immediate Fixes (High Priority)

1. **Add NOTIFICATION_CRASH Handler**
```cpp
case NOTIFICATION_CRASH: {
    // Stop all timers
    if (save_timer) save_timer->stop();
    
    // Wait for any active save
    if (save_thread_busy && save_thread) {
        save_thread->wait_to_finish();
    }
    
    // SYNCHRONOUS emergency save
    if (conversation_persistence) {
        conversation_persistence->create_emergency_backup();
    }
    
    // Force immediate save (no deferred)
    _save_conversations_synchronous();
} break;
```

2. **Fix Pending Save Race Condition**
```cpp
case NOTIFICATION_EXIT_TREE: {
    // NEW: Check if save is pending but not started
    if (save_pending && !save_thread_busy) {
        save_pending = false;
        _save_conversations();  // Force synchronous save
    }
    
    // Existing thread wait logic
    if (save_thread_busy && save_thread) {
        save_thread->wait_to_finish();
    }
    // ...
}
```

3. **Make Emergency Saves Synchronous**
```cpp
// Replace ALL instances of:
call_deferred("_emergency_save_conversations");

// With:
_emergency_save_conversations();  // Immediate, no defer
```

4. **Add Mutex Protection**
```cpp
// In _queue_delayed_save():
MutexLock lock(save_mutex);
save_data->snapshot = memnew(Vector<Conversation>(conversations));

// In _execute_delayed_save():
MutexLock lock(save_mutex);
// ... modify conversations ...
```

5. **Fix Backup Recovery to Use Timestamp**
```cpp
// In recover_from_corruption(), change to:
int64_t newest_time = 0;
String newest_emergency;

for (int i = 0; i < files.size(); i++) {
    String file = files[i];
    if (file.begins_with("EMERGENCY_backup_")) {
        String time_str = file.substr(17, file.length() - 17 - 11);
        int64_t time = time_str.to_int();
        
        if (time > newest_time) {  // Pick NEWEST, not largest
            newest_time = time;
            newest_emergency = file;
        }
    }
}
```

6. **Don't Queue Save During Restore**
```cpp
// In _restore_from_checkpoint(), remove line 18433:
// _queue_delayed_save();  // REMOVE THIS

// Instead, add deferred save AFTER UI rebuild completes:
call_deferred("_save_after_restore_complete");
```

### Medium-Term Improvements

7. **Unified Save API**
   - Single entry point for all saves
   - Centralized error handling
   - Proper status reporting

8. **Transactional Saves**
   - Write to new file
   - Validate contents
   - Atomic swap only if validation passes
   - Keep old file as automatic backup

9. **Periodic Auto-Save**
   - Independent of user actions
   - Every 30 seconds, snapshot and save
   - Reduces delayed save window

10. **Save Queue with Retry**
    - When save fails, queue for retry
    - Don't drop pending_save_data
    - Exponential backoff for retries

### Long-Term Architecture

11. **Separate Persistence Thread**
    - Dedicated thread for all I/O
    - Message queue for save requests
    - Handles backups, recovery automatically
    - Never blocks main thread

12. **Write-Ahead Logging (WAL)**
    - Log all changes before applying
    - On crash, replay log
    - Guarantees zero data loss

13. **Versioned Backups with Metadata**
    - Each backup includes:
      - Timestamp
      - Reason (manual, auto, emergency, crash)
      - Validation checksum
      - Size and message count
    - Smart recovery picks best backup based on metadata

---

## Testing Strategy

### Test Cases to Add

1. **Crash During Save Window**
   - Send message
   - Kill process within 3 seconds
   - Verify message present on restart

2. **Crash During Background Save**
   - Trigger save
   - Kill process during thread execution
   - Verify partial save doesn't corrupt file

3. **Multiple Rapid Crashes**
   - Crash, restart, crash, restart
   - Verify recovery works each time

4. **Restore During Active Save**
   - Start save operation
   - Trigger restore before save completes
   - Verify no data loss

5. **Large Conversation Stress Test**
   - Create conversation with 1000+ messages
   - Perform various operations
   - Crash at random points
   - Verify data integrity

### Monitoring Additions

1. **Save Success Rate Metric**
   - Track successful vs failed saves
   - Alert if failure rate > 1%

2. **Save Duration Tracking**
   - Measure time for each save
   - Alert if > 5 seconds (indicates large file or I/O issues)

3. **Backup Validation**
   - Periodically validate all backups
   - Remove corrupt backups
   - Alert if < 3 valid backups exist

---

## Conclusion

The chat persistence system has multiple layers of protection, but **critical race conditions and logic gaps** make it unreliable during crashes and restore operations. The issues are concentrated in:

1. **Thread safety**: No mutex protection, dual state tracking
2. **Crash handling**: Missing NOTIFICATION_CRASH, deferred emergency saves
3. **Restore logic**: Truncates data before saving, races with background saves
4. **Backup recovery**: Wrong heuristic, incomplete validation

Implementing the **Immediate Fixes** (1-6) would address **~80% of user-reported issues**. The remaining improvements would make the system truly bulletproof.

**Estimated effort**: 
- Immediate fixes: 4-6 hours
- Medium-term: 2-3 days
- Long-term: 1-2 weeks

**Risk of changes**: Low - Most fixes are defensive additions that don't change happy-path behavior.

