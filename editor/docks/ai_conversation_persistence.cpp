/**************************************************************************/
/*  ai_conversation_persistence.cpp                                       */
/**************************************************************************/

#include "ai_conversation_persistence.h"

AIConversationPersistence::AIConversationPersistence() {
    save_in_progress = false;
}

void AIConversationPersistence::initialize(const String &p_conversations_file_path) {
    conversations_file_path = p_conversations_file_path;
    backup_directory = conversations_file_path.get_base_dir().path_join("ai_chat_backups");
    
    // Ensure backup directory exists
    Ref<DirAccess> da = DirAccess::create_for_path(backup_directory);
    if (da.is_valid() && !da->dir_exists(backup_directory)) {
        da->make_dir_recursive(backup_directory);
    }
    
    _log_save_attempt("initialize", true, "Backup dir: " + backup_directory);
}

// Simplified implementation - just focus on backup and validation for now
// Full save/load will use existing AIChatDock methods

bool AIConversationPersistence::_create_backup_before_save(const String &p_file_path) {
    if (!FileAccess::exists(p_file_path)) {
        return true; // No file to backup
    }
    
    String timestamp = String::num_int64(Time::get_singleton()->get_unix_time_from_system());
    String backup_name = "conversations_backup_" + timestamp + ".simplifine";
    String backup_path = backup_directory.path_join(backup_name);
    
    Error err;
    String content = FileAccess::get_file_as_string(p_file_path, &err);
    if (err != OK) {
        return false;
    }
    
    Ref<FileAccess> backup_file = FileAccess::open(backup_path, FileAccess::WRITE, &err);
    if (err != OK) {
        return false;
    }
    
    backup_file->store_string(content);
    backup_file->close();
    
    return true;
}

AIConversationPersistence::SaveResult AIConversationPersistence::_atomic_write_file(const String &p_file_path, const String &p_content) {
    String base_dir = p_file_path.get_base_dir();
    String final_name = p_file_path.get_file();
    String temp_name = final_name + ".tmp";
    String temp_path = base_dir.path_join(temp_name);
    
    // Ensure directory exists
    Ref<DirAccess> da_mk = DirAccess::create_for_path(base_dir);
    if (da_mk.is_valid() && !da_mk->dir_exists(base_dir)) {
        da_mk->make_dir_recursive(base_dir);
    }
    
    // Write to temp file
    Error err;
    Ref<FileAccess> tmp = FileAccess::open(temp_path, FileAccess::WRITE, &err);
    if (err != OK) {
        last_error = "Cannot create temp file: " + String::num_int64(err);
        return SAVE_ERROR_FILE_ACCESS;
    }
    
    tmp->store_string(p_content);
    tmp->close();
    
    // Verify temp file was written correctly
    String verify_content = FileAccess::get_file_as_string(temp_path, &err);
    if (err != OK || verify_content != p_content) {
        last_error = "Temp file verification failed";
        return SAVE_ERROR_DISK_FULL;
    }
    
    // Atomic rename
    Ref<DirAccess> da = DirAccess::open(base_dir);
    if (!da.is_valid()) {
        last_error = "Cannot access directory: " + base_dir;
        return SAVE_ERROR_FILE_ACCESS;
    }
    
    if (da->file_exists(final_name)) {
        da->remove(final_name);
    }
    
    Error rename_err = da->rename(temp_name, final_name);
    if (rename_err != OK) {
        last_error = "Atomic rename failed: " + String::num_int64(rename_err);
        return SAVE_ERROR_FILE_ACCESS;
    }
    
    return SAVE_SUCCESS;
}

bool AIConversationPersistence::_validate_json_structure(const Dictionary &p_data) {
    if (!p_data.has("conversations")) {
        return false;
    }
    
    if (!p_data.has("version")) {
        return false;
    }
    
    Array conversations = p_data["conversations"];
    for (int i = 0; i < conversations.size(); i++) {
        Dictionary conv = conversations[i];
        if (!conv.has("id") || !conv.has("messages")) {
            return false;
        }
    }
    
    return true;
}

bool AIConversationPersistence::_attempt_recovery_from_backup() {
    
    Ref<DirAccess> da = DirAccess::open(backup_directory);
    if (!da.is_valid()) {
        return false;
    }
    
    // Find the most recent backup
    PackedStringArray files = da->get_files();
    String newest_backup;
    int64_t newest_time = 0;
    
    for (int i = 0; i < files.size(); i++) {
        String file = files[i];
        if (file.begins_with("conversations_backup_") && file.ends_with(".simplifine")) {
            String time_str = file.substr(21, file.length() - 21 - 11); // Extract timestamp
            int64_t time = time_str.to_int();
            if (time > newest_time) {
                newest_time = time;
                newest_backup = file;
            }
        }
    }
    
    if (newest_backup.is_empty()) {
        return false;
    }
    
    String backup_path = backup_directory.path_join(newest_backup);
    
    Error err;
    String backup_content = FileAccess::get_file_as_string(backup_path, &err);
    if (err != OK) {
        return false;
    }
    
    // Try to parse backup
    Ref<JSON> json;
    json.instantiate();
    Error parse_err = json->parse(backup_content);
    if (parse_err != OK) {
        return false;
    }
    
    Dictionary data = json->get_data();
    if (!_validate_json_structure(data)) {
        return false;
    }
    
    // Recovery successful - copy backup to main file
    SaveResult copy_result = _atomic_write_file(conversations_file_path, backup_content);
    if (copy_result == SAVE_SUCCESS) {
        return true;
    }
    
    return false;
}

void AIConversationPersistence::_cleanup_old_backups() {
    Ref<DirAccess> da = DirAccess::open(backup_directory);
    if (!da.is_valid()) {
        return;
    }
    
    // Get all backup files with timestamps
    PackedStringArray files = da->get_files();
    Vector<String> backup_files;
    
    for (int i = 0; i < files.size(); i++) {
        String file = files[i];
        if (file.begins_with("conversations_backup_") && file.ends_with(".simplifine")) {
            backup_files.push_back(file);
        }
    }
    
    // Sort by timestamp (newest first)
    backup_files.sort();
    backup_files.reverse();
    
    // Remove old backups beyond limit
    for (int i = MAX_BACKUP_FILES; i < backup_files.size(); i++) {
        String old_backup = backup_directory.path_join(backup_files[i]);
        da->remove(backup_files[i]);
    }
}

void AIConversationPersistence::create_emergency_backup() {
    if (!FileAccess::exists(conversations_file_path)) {
        return;
    }
    
    String emergency_name = "EMERGENCY_backup_" + String::num_int64(Time::get_singleton()->get_unix_time_from_system()) + ".simplifine";
    String emergency_path = backup_directory.path_join(emergency_name);
    
    Error err;
    String content = FileAccess::get_file_as_string(conversations_file_path, &err);
    if (err == OK) {
        Ref<FileAccess> emergency_file = FileAccess::open(emergency_path, FileAccess::WRITE);
        if (emergency_file.is_valid()) {
            emergency_file->store_string(content);
            emergency_file->close();
        }
    }
}

bool AIConversationPersistence::validate_conversations_file() {
    if (!FileAccess::exists(conversations_file_path)) {
        return false;
    }
    
    Error err;
    String content = FileAccess::get_file_as_string(conversations_file_path, &err);
    if (err != OK) {
        return false;
    }
    
    Ref<JSON> json;
    json.instantiate();
    Error parse_err = json->parse(content);
    if (parse_err != OK) {
        return false;
    }
    
    Dictionary data = json->get_data();
    return _validate_json_structure(data);
}

int64_t AIConversationPersistence::get_file_size() const {
    if (!FileAccess::exists(conversations_file_path)) {
        return 0;
    }
    
    Ref<FileAccess> file = FileAccess::open(conversations_file_path, FileAccess::READ);
    if (!file.is_valid()) {
        return 0;
    }
    
    int64_t size = file->get_length();
    file->close();
    return size;
}

void AIConversationPersistence::_log_save_attempt(const String &p_operation, bool p_success, const String &p_details) {
	(void)p_operation;
	(void)p_success;
	(void)p_details;
}

void AIConversationPersistence::_log_load_attempt(const String &p_operation, bool p_success, const String &p_details) {
	(void)p_operation;
	(void)p_success;
	(void)p_details;
}

// CRITICAL: Safe save method - prevents race conditions and data corruption
AIConversationPersistence::SaveResult AIConversationPersistence::save_conversations_safe(const String &p_json_data) {
    // Prevent race conditions - only one save at a time
    if (save_in_progress) {
        _log_save_attempt("safe_save", false, "Save already in progress, storing data for later");
        pending_save_data = p_json_data; // Store for retry
        return SAVE_ERROR_ALREADY_IN_PROGRESS;
    }
    
    save_in_progress = true;
    _log_save_attempt("safe_save_start", true, "Beginning safe save operation");
    
    // Always create backup before any write operation
    bool backup_created = _create_backup_before_save(conversations_file_path);
    if (!backup_created && FileAccess::exists(conversations_file_path)) {
        _log_save_attempt("backup_creation", false, "Failed to create backup - ABORTING save for safety");
        save_in_progress = false;
        return SAVE_ERROR_FILE_ACCESS;
    }
    
    // Use atomic write to prevent corruption
    SaveResult result = _atomic_write_file(conversations_file_path, p_json_data);
    
    if (result == SAVE_SUCCESS) {
        _log_save_attempt("safe_save_complete", true, "Safe save successful");
        _cleanup_old_backups();
    } else {
        _log_save_attempt("safe_save_failed", false, "Safe save failed: " + last_error);
    }
    
    save_in_progress = false;
    return result;
}

AIConversationPersistence::SaveResult AIConversationPersistence::save_conversations_with_validation(const String &p_json_data) {
    // First validate the JSON structure before attempting any save
    Ref<JSON> json;
    json.instantiate();
    Error parse_err = json->parse(p_json_data);
    if (parse_err != OK) {
        _log_save_attempt("validation", false, "JSON parse error before save");
        return SAVE_ERROR_JSON_GENERATION;
    }
    
    Dictionary data = json->get_data();
    if (!_validate_json_structure(data)) {
        _log_save_attempt("validation", false, "Invalid conversation structure before save");
        return SAVE_ERROR_JSON_GENERATION;
    }
    
    // Check size safety
    if (!is_size_safe_for_save(p_json_data)) {
        _log_save_attempt("size_check", false, "File size unsafe - attempting safe handling");
        if (!handle_large_file_safely(p_json_data)) {
            return SAVE_ERROR_DISK_FULL;
        }
    }
    
    return save_conversations_safe(p_json_data);
}

bool AIConversationPersistence::is_size_safe_for_save(const String &p_json_data) const {
    return p_json_data.length() < MAX_FILE_SIZE;
}

bool AIConversationPersistence::handle_large_file_safely(const String &p_json_data) {
    // NEVER just delete conversations! Instead, create large file backup and suggest cleanup
    _log_save_attempt("large_file_handling", true, "Creating large file backup instead of deletion");
    
    String timestamp = String::num_int64(Time::get_singleton()->get_unix_time_from_system());
    String large_backup = backup_directory.path_join("LARGE_FILE_backup_" + timestamp + ".simplifine");
    
    Ref<FileAccess> large_file = FileAccess::open(large_backup, FileAccess::WRITE);
    if (!large_file.is_valid()) {
        _log_save_attempt("large_backup", false, "Failed to create large file backup");
        return false;
    }
    
    large_file->store_string(p_json_data);
    large_file->close();
    
    _log_save_attempt("large_backup", true, "Large conversation data backed up to: " + large_backup);
    
    // Try to create a reasonable subset of conversations for main file
    // Parse and keep only recent conversations to stay under size limit
    Ref<JSON> json;
    json.instantiate();
    Error parse_err = json->parse(p_json_data);
    if (parse_err != OK) {
        return false;
    }
    
    Dictionary data = json->get_data();
    if (!data.has("conversations")) {
        return false;
    }
    
    Array conversations = data["conversations"];
    Array trimmed_conversations;
    int64_t estimated_size = 1000; // Base JSON overhead
    
    // Keep most recent conversations that fit in size limit - SAFER iteration
    for (int i = conversations.size() - 1; i >= 0 && estimated_size < (MAX_FILE_SIZE / 2); i--) {
        if (i < 0 || i >= conversations.size()) break; // Safety check
        
        Dictionary conv = conversations[i];
        Ref<JSON> temp_json;
        temp_json.instantiate();
        String conv_json = temp_json->stringify(conv);
        int64_t conv_size = conv_json.length();
        
        // Prevent runaway memory usage
        if (estimated_size + conv_size < (MAX_FILE_SIZE / 2)) {
            trimmed_conversations.push_front(conv);
            estimated_size += conv_size;
        } else {
            break; // Stop adding if we'd exceed limit
        }
    }
    
    data["conversations"] = trimmed_conversations;
    String trimmed_json = JSON().stringify(data, "  ");
    
    // Save the trimmed version safely
    return save_conversations_safe(trimmed_json) == SAVE_SUCCESS;
}

bool AIConversationPersistence::recover_from_corruption() {
    
    // First try regular backup recovery
    if (_attempt_recovery_from_backup()) {
        return true;
    }
    
    // Try emergency backups as fallback
    Ref<DirAccess> da = DirAccess::open(backup_directory);
    if (!da.is_valid()) {
        return false;
    }
    
    // Find the LARGEST EMERGENCY backup (most likely to contain real data, not empty corruption)
    PackedStringArray files = da->get_files();
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
                
                // Skip tiny files (likely empty conversations or corruption)
                if (file_size > 1000 && file_size > largest_size) {  
                    largest_size = file_size;
                    largest_emergency = file;
                }
            }
        }
    }
    
    if (largest_emergency.is_empty()) {
        return false;
    }
    
    String emergency_path = backup_directory.path_join(largest_emergency);
    
    Error err;
    String backup_content = FileAccess::get_file_as_string(emergency_path, &err);
    if (err != OK) {
        return false;
    }
    
    // Validate emergency backup
    Ref<JSON> json;
    json.instantiate();
    Error parse_err = json->parse(backup_content);
    if (parse_err != OK) {
        return false;
    }
    
    Dictionary data = json->get_data();
    if (!_validate_json_structure(data)) {
        return false;
    }
    
    // Recovery successful - restore from emergency backup
    SaveResult copy_result = _atomic_write_file(conversations_file_path, backup_content);
    if (copy_result == SAVE_SUCCESS) {
        return true;
    }
    
    return false;
}
