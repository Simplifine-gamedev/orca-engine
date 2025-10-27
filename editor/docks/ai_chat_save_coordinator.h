/**************************************************************************/
/*  ai_chat_save_coordinator.h                                            */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/

#ifndef AI_CHAT_SAVE_COORDINATOR_H
#define AI_CHAT_SAVE_COORDINATOR_H

#include "core/string/ustring.h"

// Forward declaration to avoid circular includes
class AIConversationPersistence;

// Safety coordinator to prevent catastrophic conversation deletion
// This replaces ALL dangerous direct FileAccess::WRITE operations
// Internal class - no script binding needed
class AIChatSaveCoordinator {

private:
    AIConversationPersistence* persistence;
    bool initialized;
    
    // Prevent multiple saves from racing
    bool emergency_save_pending;
    
public:
    AIChatSaveCoordinator();
    ~AIChatSaveCoordinator();
    
    void initialize(AIConversationPersistence* p_persistence);
    
    // SAFE save methods - replace all the dangerous ones in ai_chat_dock.cpp
    enum SaveStatus {
        SAVE_SUCCESS_SYNC,
        SAVE_SUCCESS_ASYNC, 
        SAVE_FAILED_VALIDATION,
        SAVE_FAILED_PERSISTENCE,
        SAVE_SKIPPED_RACING
    };
    
    // Main replacement for _save_conversations()
    SaveStatus save_conversations_safe_sync(const String& p_json_data);
    
    // Main replacement for _save_conversations_async() and _save_conversations_to_disk()  
    SaveStatus save_conversations_safe_async(const String& p_json_data);
    
    // Main replacement for _save_conversations_chunked()
    SaveStatus save_conversations_chunked_safe(const String& p_json_data);
    
    // Emergency save with proper coordination
    void emergency_save_safe(const String& p_json_data);
    
    // Validation and status
    bool is_ready() const { return initialized && persistence != nullptr; }
    String get_status_message() const;
    
    // CRITICAL: Static method to detect and prevent dangerous save patterns
    static bool is_dangerous_save_pattern(const String& p_method_name);
    
private:
    void _log_save_coordinator(const String &p_operation, bool p_success, const String &p_details = "");
};

#endif // AI_CHAT_SAVE_COORDINATOR_H
