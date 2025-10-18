/**************************************************************************/
/*  ai_conversation_persistence.h                                         */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#pragma once

#include "core/io/file_access.h"
#include "core/io/dir_access.h"
#include "core/io/json.h"
#include "core/os/os.h"
#include "core/os/time.h"
#include "core/templates/vector.h"
#include "core/templates/hash_map.h"
#include "core/string/ustring.h"

// Forward declarations to avoid circular includes
class AIChatDock;

class AIConversationPersistence {
public:
    // Save/Load results
    enum SaveResult {
        SAVE_SUCCESS,
        SAVE_ERROR_FILE_ACCESS,
        SAVE_ERROR_JSON_GENERATION,
        SAVE_ERROR_DISK_FULL,
        SAVE_ERROR_PERMISSION
    };
    
    enum LoadResult {
        LOAD_SUCCESS,
        LOAD_ERROR_FILE_NOT_FOUND,
        LOAD_ERROR_JSON_PARSE,
        LOAD_ERROR_CORRUPTED_DATA,
        LOAD_ERROR_FILE_TOO_LARGE
    };

private:
    String conversations_file_path;
    String backup_directory;
    
    // Safety limits
    static const int64_t MAX_FILE_SIZE = 5 * 1024 * 1024; // 5MB
    static const int MAX_BACKUP_FILES = 10;
    
    // Corruption recovery
    bool _attempt_recovery_from_backup();
    bool _create_backup_before_save(const String &p_file_path);
    void _cleanup_old_backups();
    
    // Safe file operations
    SaveResult _atomic_write_file(const String &p_file_path, const String &p_content);
    bool _validate_json_structure(const Dictionary &p_data);
    
    // Logging
    void _log_save_attempt(const String &p_operation, bool p_success, const String &p_details = "");
    void _log_load_attempt(const String &p_operation, bool p_success, const String &p_details = "");

public:
    AIConversationPersistence();
    
    void initialize(const String &p_conversations_file_path);
    
    // Main save/load operations - use AIChatDock types directly
    template<typename ConversationType>
    SaveResult save_conversations_generic(const Vector<ConversationType> &p_conversations);
    
    template<typename ConversationType>
    LoadResult load_conversations_generic(Vector<ConversationType> &r_conversations);
    
    // Emergency operations
    void create_emergency_backup();
    bool recover_from_corruption();
    
    // Validation
    bool validate_conversations_file();
    int64_t get_file_size() const;
    
    // Status
    String get_last_error() const { return last_error; }
    String get_conversations_file_path() const { return conversations_file_path; }
    
private:
    String last_error;
};
