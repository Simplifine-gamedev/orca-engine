# Critical Race Conditions - Visual Analysis

## Race Condition #1: Crash During Delayed Save Window

```
Timeline of Data Loss Scenario:

T=0.0s: User sends message "Hello World"
        ├─> conversations[0].messages.push_back(message)
        └─> _queue_delayed_save()
            ├─> save_pending = true
            ├─> save_timer->start(3.0)  // Wait 3 seconds
            └─> call_deferred("_execute_delayed_save")

T=0.5s: call_deferred callback arrives
        └─> _execute_delayed_save()
            └─> save_thread_busy check: false
                └─> Ready to start save thread...

T=1.2s: ⚠️  CRASH! (e.g., segfault, force quit, power loss)

T=1.2s: NOTIFICATION_EXIT_TREE fires
        ├─> save_timer->stop()  ✓
        ├─> Check: save_thread_busy? 
        │   └─> NO (thread never started yet!)
        │       └─> SKIP wait_to_finish()  ❌
        └─> create_emergency_backup()
            ├─> Reads current file (OLD state, no "Hello World")
            └─> Saves as EMERGENCY_backup_xxx.simplifine

T=1.3s: _save_conversations() synchronous call
        └─> But conversations vector may be in inconsistent state
            └─> If corrupted: saves garbage
            └─> If OK: saves but with race condition artifacts

T=1.4s: Process exits

RESTART:

T=5.0s: App starts, calls _load_conversations()
        ├─> Reads ai_chat_conversations.simplifine
        │   └─> File is either:
        │       ├─> Missing "Hello World" (never saved)
        │       ├─> Corrupt (from inconsistent state)
        │       └─> Or missing entirely (write failed)
        │
        └─> If corrupt: recover_from_corruption()
            └─> Uses EMERGENCY backup (which has OLD state)
                └─> "Hello World" message is GONE

RESULT: User's message disappeared! ❌
```

---

## Race Condition #2: Checkpoint Restore Wipes Chat

```
Timeline of Restore-Induced Data Loss:

INITIAL STATE:
  conversations[0].messages = [
    {id: 1, content: "First message"},
    {id: 2, content: "Second message"}, 
    {id: 3, content: "Third message"},
    {id: 4, content: "Fourth message"}
  ]

T=0.0s: User clicks "Restore to message 2"
        └─> _restore_from_checkpoint(2)

T=0.1s: Truncate messages in memory
        ├─> while (chat_history.size() > 2 + 1)
        │   └─> chat_history.remove_at(3)  // Remove "Fourth message"
        │   └─> chat_history.remove_at(2)  // Remove "Third message"
        │
        └─> IN MEMORY NOW:
            conversations[0].messages = [
              {id: 1, content: "First message"},
              {id: 2, content: "Second message"}
            ]

T=0.2s: UI rebuild starts
        └─> _rebuild_conversation_ui(chat_history)

T=0.3s: Save queued!
        └─> _queue_delayed_save()
            ├─> save_pending = true
            └─> save_timer->start(3.0)

T=0.5s: Background save from EARLIER action completes
        └─> _background_save_complete()
            └─> save_thread_busy = false  // Now available

T=0.6s: _execute_delayed_save() fires (from deferred call)
        ├─> save_thread_busy? NO
        ├─> save_thread_busy = true
        └─> Snapshot conversations vector
            └─> Includes TRUNCATED messages (only 2 messages!)

T=1.0s: Background thread saves to disk
        └─> Writes truncated conversation:
            {
              "conversations": [{
                "messages": [
                  {"id": 1, "content": "First message"},
                  {"id": 2, "content": "Second message"}
                ]
              }]
            }
            
T=1.5s: ⚠️  CRASH! (power loss, OS kill)

RESTART:

T=5.0s: _load_conversations()
        └─> Loads saved file:
            └─> Only 2 messages! "Third" and "Fourth" are GONE! ❌

RESULT: Restore operation permanently deleted chat history! ❌
```

---

## Race Condition #3: Background Thread Snapshot Misses New Data

```
Timeline of Snapshot Race:

T=0.0s: User sends "Message A"
        ├─> conversations[0].messages.push_back("Message A")
        └─> _queue_delayed_save()
            └─> save_pending = true

T=0.5s: _execute_delayed_save()
        ├─> save_pending = false
        ├─> save_thread_busy = true
        └─> Create snapshot:
            snapshot = conversations  // Includes "Message A" ✓

T=1.0s: Background thread starts serializing snapshot

T=1.5s: User sends "Message B" (while save is running!)
        ├─> conversations[0].messages.push_back("Message B")
        └─> _queue_delayed_save()
            ├─> save_pending = true
            └─> save_timer->start(3.0)

T=2.0s: Background thread finishes writing snapshot
        └─> File now has: ["Message A"]
        └─> Memory has: ["Message A", "Message B"]
        └─> save_thread_busy = false

T=2.1s: User sends "Message C"
        ├─> conversations[0].messages.push_back("Message C")
        └─> _queue_delayed_save()
            └─> save_pending already true, skip

T=2.5s: ⚠️  CRASH!

NOTIFICATION_EXIT_TREE:
  ├─> save_thread_busy? NO (background save finished)
  ├─> save_pending? YES
  │   └─> But NO CODE PATH handles this! ❌
  │
  └─> Calls _save_conversations() synchronously
      └─> Saves ["Message A", "Message B", "Message C"] ✓

BUT: If crash is hard (SIGKILL, power loss):
  └─> NOTIFICATION_EXIT_TREE never fires
      └─> Only "Message A" is on disk
      └─> "Message B" and "Message C" are LOST ❌
```

---

## Race Condition #4: Dual State Tracking Deadlock

```
Timeline of State Desynchronization:

INITIAL STATE:
  AIChatDock::save_thread_busy = false
  AIConversationPersistence::save_in_progress = false

T=0.0s: User action triggers save #1
        └─> _execute_delayed_save()
            ├─> save_thread_busy = true
            └─> save_thread starts

T=0.1s: Background thread calls persistence
        └─> conversation_persistence->save_conversations_with_validation(json)
            └─> save_conversations_safe(json)
                ├─> Check: save_in_progress? NO
                ├─> save_in_progress = true ✓
                └─> _atomic_write_file()

T=0.5s: User action triggers save #2
        └─> _execute_delayed_save()
            ├─> Check: save_thread_busy? YES
            └─> Reschedule for 0.5s later ✓ (CORRECT)

T=1.0s: Background thread finishes atomic write
        └─> save_in_progress = false ✓
        └─> Returns to _background_save()

T=1.2s: _background_save() completes
        └─> Schedules _background_save_complete() deferred

T=1.3s: Save #2 timer fires (from reschedule)
        └─> _execute_delayed_save()
            ├─> Check: save_thread_busy? 
            │   └─> STILL TRUE! (deferred completion not run yet)
            └─> Reschedule AGAIN for 0.5s

T=1.5s: Deferred _background_save_complete() fires
        └─> save_thread_busy = false

T=1.8s: Save #2 timer fires AGAIN
        └─> _execute_delayed_save()
            ├─> save_thread_busy? NO ✓
            ├─> save_pending = false
            ├─> save_thread_busy = true
            └─> Start new background thread

T=1.9s: User sends Message Z
        └─> _queue_delayed_save()
            ├─> Check: save_pending? NO (was just cleared!)
            ├─> save_pending = true
            └─> Schedule for 3.0s

T=2.0s: New background thread reaches persistence
        └─> conversation_persistence->save_conversations_safe(json)
            ├─> Check: save_in_progress? NO ✓
            └─> save_in_progress = true

T=2.1s: ⚠️  HARD CRASH (SIGKILL)

PROBLEM: 
  ├─> save_pending = true (for Message Z)
  ├─> save_thread_busy = true (background save in progress)
  ├─> save_in_progress = true (persistence is writing)
  └─> Message Z is in memory but NOT in the snapshot being saved! ❌

RESULT: Message Z is LOST on restart! ❌
```

---

## Race Condition #5: Emergency Save via call_deferred is Not Emergency

```
Timeline of "Emergency" Save Failure:

T=0.0s: User performs critical action (adds pending edit)
        └─> _add_pending_edit(tool_id, file_path)
            ├─> pending_apply_edits[tool_id] = file_path
            └─> call_deferred("_emergency_save_conversations")  ⚠️

T=0.001s: ⚠️  IMMEDIATE CRASH (before deferred call processes!)

NOTIFICATION_EXIT_TREE:
  └─> Deferred calls are CLEARED (never execute!)
      └─> _emergency_save_conversations() NEVER RUNS ❌

RESTART:
  └─> pending_apply_edits is EMPTY
      └─> User's pending edits are LOST ❌

---

WHAT SHOULD HAPPEN:

T=0.0s: User performs critical action
        └─> _add_pending_edit(tool_id, file_path)
            ├─> pending_apply_edits[tool_id] = file_path
            └─> _emergency_save_conversations()  // IMMEDIATE, NOT DEFERRED!
                ├─> create_emergency_backup()
                └─> _save_conversations()

T=0.001s: ⚠️  CRASH

NOTIFICATION_EXIT_TREE:
  └─> Emergency backup and save already completed ✓

RESTART:
  └─> pending_apply_edits restored from backup ✓
```

---

## Summary of Root Causes

| Issue | Root Cause | Impact | Frequency |
|-------|-----------|--------|-----------|
| **Pending save orphaned** | `save_pending` set but never cleared if crash before thread start | HIGH | Common (3-sec window) |
| **Restore truncates** | Save queued with truncated data in memory | CRITICAL | Every restore |
| **Snapshot races** | No mutex on conversations vector | MEDIUM | Occasional |
| **Dual state tracking** | Two separate flags without coordination | MEDIUM | Rare but severe |
| **Deferred emergency** | call_deferred clears on crash | HIGH | Common |
| **No NOTIFICATION_CRASH** | Missing crash handler | CRITICAL | Every crash |

---

## Mitigation Priority

### 🔴 P0 - Fix Immediately (Data Loss Guaranteed)
1. Add NOTIFICATION_CRASH handler
2. Fix pending save check in EXIT_TREE
3. Make emergency saves synchronous (remove call_deferred)
4. Don't queue save during restore

### 🟡 P1 - Fix Soon (Data Loss Likely)
5. Add mutex protection to conversations vector
6. Coordinate save_thread_busy and save_in_progress
7. Fix backup recovery to use timestamp not size

### 🟢 P2 - Fix When Time Permits (Data Loss Rare)
8. Validate temp files before promoting
9. Implement proper pending_save_data retry
10. Add comprehensive error handling and logging

