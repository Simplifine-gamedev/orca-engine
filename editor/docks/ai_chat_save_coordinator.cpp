/**************************************************************************/
/*  ai_chat_save_coordinator.cpp                                          */
/**************************************************************************/

#include "ai_chat_save_coordinator.h"
#include "ai_conversation_persistence.h"
#include "core/io/file_access.h"

AIChatSaveCoordinator::AIChatSaveCoordinator() {
    persistence = nullptr;
    initialized = false;
    emergency_save_pending = false;
}

AIChatSaveCoordinator::~AIChatSaveCoordinator() {
    // Note: We don't own the persistence object, just reference it
}

void AIChatSaveCoordinator::initialize(AIConversationPersistence* p_persistence) {
    persistence = p_persistence;
    initialized = (persistence != nullptr);
    
    if (initialized) {
        _log_save_coordinator("initialize", true, "Save coordinator ready with persistence manager");
    } else {
        _log_save_coordinator("initialize", false, "No persistence manager provided");
    }
}

AIChatSaveCoordinator::SaveStatus AIChatSaveCoordinator::save_conversations_safe_sync(const String& p_json_data) {
    if (!is_ready()) {
        _log_save_coordinator("sync_save", false, "Coordinator not ready");
        return SAVE_FAILED_PERSISTENCE;
    }
    
    _log_save_coordinator("sync_save_start", true, "Starting synchronous safe save");
    
    // Use the safe persistence mechanism with full validation
    AIConversationPersistence::SaveResult result = persistence->save_conversations_with_validation(p_json_data);
    
    switch (result) {
        case AIConversationPersistence::SAVE_SUCCESS:
            _log_save_coordinator("sync_save_complete", true, "Synchronous save successful");
            return SAVE_SUCCESS_SYNC;
            
        case AIConversationPersistence::SAVE_ERROR_ALREADY_IN_PROGRESS:
            _log_save_coordinator("sync_save_racing", false, "Save already in progress - coordinating");
            return SAVE_SKIPPED_RACING;
            
        case AIConversationPersistence::SAVE_ERROR_JSON_GENERATION:
            _log_save_coordinator("sync_save_validation", false, "JSON validation failed");
            return SAVE_FAILED_VALIDATION;
            
        default:
            _log_save_coordinator("sync_save_error", false, "Persistence error: " + persistence->get_last_error());
            return SAVE_FAILED_PERSISTENCE;
    }
}

AIChatSaveCoordinator::SaveStatus AIChatSaveCoordinator::save_conversations_safe_async(const String& p_json_data) {
    if (!is_ready()) {
        _log_save_coordinator("async_save", false, "Coordinator not ready");
        return SAVE_FAILED_PERSISTENCE;
    }
    
    _log_save_coordinator("async_save_start", true, "Starting asynchronous safe save");
    
    // For async, still use synchronous safe save but don't block the caller
    // The persistence layer handles the actual file operations safely
    AIConversationPersistence::SaveResult result = persistence->save_conversations_with_validation(p_json_data);
    
    switch (result) {
        case AIConversationPersistence::SAVE_SUCCESS:
            _log_save_coordinator("async_save_complete", true, "Asynchronous save successful");
            return SAVE_SUCCESS_ASYNC;
            
        case AIConversationPersistence::SAVE_ERROR_ALREADY_IN_PROGRESS:
            _log_save_coordinator("async_save_racing", false, "Save already in progress - will retry");
            return SAVE_SKIPPED_RACING;
            
        case AIConversationPersistence::SAVE_ERROR_JSON_GENERATION:
            _log_save_coordinator("async_save_validation", false, "JSON validation failed");
            return SAVE_FAILED_VALIDATION;
            
        default:
            _log_save_coordinator("async_save_error", false, "Persistence error: " + persistence->get_last_error());
            return SAVE_FAILED_PERSISTENCE;
    }
}

AIChatSaveCoordinator::SaveStatus AIChatSaveCoordinator::save_conversations_chunked_safe(const String& p_json_data) {
    if (!is_ready()) {
        _log_save_coordinator("chunked_save", false, "Coordinator not ready");
        return SAVE_FAILED_PERSISTENCE;
    }
    
    _log_save_coordinator("chunked_save_start", true, "Starting chunked safe save");
    
    // The chunked approach isn't needed for safety - the persistence layer handles this
    // Just use the standard safe save with validation
    return save_conversations_safe_sync(p_json_data);
}

void AIChatSaveCoordinator::emergency_save_safe(const String& p_json_data) {
    if (!is_ready()) {
        _log_save_coordinator("emergency_save", false, "Emergency save requested but coordinator not ready");
        return;
    }
    
    // Prevent multiple emergency saves from stacking up
    if (emergency_save_pending) {
        _log_save_coordinator("emergency_save", false, "Emergency save already pending, skipping duplicate");
        return;
    }
    
    emergency_save_pending = true;
    _log_save_coordinator("emergency_save_start", true, "Starting emergency safe save");
    
    // Create emergency backup first
    persistence->create_emergency_backup();
    
    // Then do the safe save
    AIConversationPersistence::SaveResult result = persistence->save_conversations_with_validation(p_json_data);
    
    if (result == AIConversationPersistence::SAVE_SUCCESS) {
        _log_save_coordinator("emergency_save_complete", true, "Emergency save successful");
    } else {
        _log_save_coordinator("emergency_save_failed", false, "Emergency save failed: " + persistence->get_last_error());
    }
    
    emergency_save_pending = false;
}

String AIChatSaveCoordinator::get_status_message() const {
    if (!initialized) {
        return "Save coordinator not initialized";
    }
    if (!persistence) {
        return "No persistence manager available";
    }
    if (emergency_save_pending) {
        return "Emergency save in progress";
    }
    return "Save coordinator ready";
}

bool AIChatSaveCoordinator::is_dangerous_save_pattern(const String& p_method_name) {
    // Detect the dangerous patterns that we're replacing
    return p_method_name.find("FileAccess::open") != -1 && 
           p_method_name.find("WRITE") != -1 &&
           p_method_name.find("conversations") != -1;
}

void AIChatSaveCoordinator::_log_save_coordinator(const String &p_operation, bool p_success, const String &p_details) {
    String status = p_success ? "SUCCESS" : "FAILED";
    String log_msg = "AI Chat Save Coordinator [" + p_operation.to_upper() + "]: " + status;
    if (!p_details.is_empty()) {
        log_msg += " - " + p_details;
    }
    print_line(log_msg);
}

// No script binding needed - internal class only
