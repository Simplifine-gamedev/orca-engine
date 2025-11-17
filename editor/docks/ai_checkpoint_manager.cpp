/**************************************************************************/
/*  ai_checkpoint_manager.cpp                                             */
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

/*
 * © 2025 Simplifine Corp.
 * Personal Non‑Commercial License applies to this file as an original contribution to this Godot fork.
 * See LICENSES/COMPANY-NONCOMMERCIAL.md for terms. Commercial use requires a separate license from Simplifine.
 */

#include "ai_checkpoint_manager.h"
#include "core/io/dir_access.h"
#include "core/io/file_access.h"
#include "core/os/os.h"
#include "core/os/time.h"
#include "core/config/project_settings.h"
#include "core/io/resource_loader.h"
#include "core/io/resource_saver.h"
#include "core/object/script_language.h"
#include "scene/gui/control.h"
#include "scene/main/window.h"
#include "editor/editor_node.h"
#include "editor/editor_interface.h"
#include "editor/file_system/editor_paths.h"
#include "editor/docks/filesystem_dock.h"
#include "editor/file_system/editor_file_system.h"
#include "editor/script/script_editor_plugin.h"
#include "editor/ai/editor_tools.h"
#include "modules/gdscript/gdscript.h"
#include "modules/gdscript/gdscript_cache.h"
#include "servers/display_server.h"

AICheckpointManager::CheckpointResult AICheckpointManager::create_comprehensive_checkpoint(const String &p_project_root, const String &p_message, int p_message_index) {
	CheckpointResult result;
	result.success = false;
	
	
	// Get isolated checkpoint directory (separate from project git)
	String checkpoint_dir = _get_checkpoint_directory(p_project_root);
	
	// Ensure project .gitignore excludes .ai-checkpoints directory
	_ensure_project_gitignore_excludes_checkpoints(p_project_root);
	
	// Initialize dedicated checkpoint Git repo
	if (!_init_checkpoint_git_repo(checkpoint_dir)) {
		result.message = "Failed to initialize checkpoint Git repository";
		return result;
	}
	
	// Copy all project files to checkpoint directory
	if (!_copy_project_to_checkpoint(p_project_root, checkpoint_dir)) {
		result.message = "Failed to copy project files to checkpoint directory";
		return result;
	}
	
	// Stage ALL files in checkpoint directory
	if (!_stage_all_files(checkpoint_dir)) {
		result.message = "Failed to stage files in checkpoint directory";
		return result;
	}
	
	// Create commit in checkpoint directory
	if (!_create_commit(checkpoint_dir, p_message, p_message_index)) {
		// Try to create tag anyway (might be no changes)
		if (!_create_checkpoint_tag(checkpoint_dir, p_message_index)) {
			result.message = "Failed to create checkpoint commit and tag";
			return result;
		}
	} else {
		// Create tag for easy reference
		if (!_create_checkpoint_tag(checkpoint_dir, p_message_index)) {
			result.message = "Commit created but tag creation failed";
			return result;
		}
	}
	
	// Success!
	result.success = true;
	result.checkpoint_tag = _generate_checkpoint_name(p_message_index);
	result.message = "Isolated checkpoint created successfully";
	
	
	return result;
}

AICheckpointManager::RestoreResult AICheckpointManager::restore_to_checkpoint(const String &p_project_root, int p_message_index) {
	RestoreResult result;
	result.success = false;
	
	
	// Get isolated checkpoint directory
	String checkpoint_dir = _get_checkpoint_directory(p_project_root);
	
	// STEP 1: Capture current editor state (for reopening after restore)
	result.restored_scene_path = _get_current_scene_path();
	result.restored_scripts = _get_open_scripts();
	
	
	// STEP 2: Clear ALL editor state to prevent crashes during restore
	_clear_editor_state();
	
	// STEP 3: Find and validate checkpoint tag in checkpoint directory
	String checkpoint_tag = _find_checkpoint_tag(checkpoint_dir, p_message_index);
	
	if (checkpoint_tag.is_empty()) {
		result.message = "No checkpoint found for message index " + String::num_int64(p_message_index);
		return result;
	}
	
	
	// STEP 4: Reset checkpoint directory to the desired tag
	if (!_git_reset_to_tag(checkpoint_dir, checkpoint_tag)) {
		result.message = "Checkpoint git reset failed";
		return result;
	}
	
	// STEP 5: Copy files from checkpoint directory to project
	if (!_restore_project_from_checkpoint(checkpoint_dir, p_project_root)) {
		result.message = "Failed to restore project files from checkpoint";
		return result;
	}
	
	
	// VERIFICATION: List restored files
	Ref<DirAccess> verify_dir = DirAccess::open(p_project_root);
	if (verify_dir.is_valid()) {
		verify_dir->list_dir_begin();
		String file_name = verify_dir->get_next();
		int file_count = 0;
		int dir_count = 0;
		while (!file_name.is_empty()) {
			if (verify_dir->current_is_dir()) {
				if (!file_name.begins_with(".")) {
					dir_count++;
				}
			} else {
				file_count++;
			}
			file_name = verify_dir->get_next();
		}
		verify_dir->list_dir_end();
	}
	
	// Success!
	result.success = true;
	result.message = "Isolated checkpoint restored successfully";
	

	return result;
}

void AICheckpointManager::refresh_editor_completely(const String &p_restored_scene_path, const Vector<String> &p_restored_scripts) {
	
	// Phase 1: Resources (immediate)
	_refresh_phase_1_resources();
	
	// Phase 2: Scenes (0.2s delay for resource loading)
	// Note: We can't use SceneTreeTimer here since this is a static class
	// The caller (ai_chat_dock) should handle the phased delays
	
}

// ============================================================================
// PRIVATE IMPLEMENTATION METHODS
// ============================================================================

String AICheckpointManager::_get_checkpoint_directory(const String &p_project_root) {
	// Prefer storing checkpoints OUTSIDE the project tree so Godot's FileSystem
	// and importers never try to treat them as regular project resources.
	String checkpoint_dir;

	if (EditorPaths::get_singleton() && EditorPaths::get_singleton()->are_paths_valid()) {
		// Project settings dir is a RESOURCES path (e.g. "res://.godot/editor").
		// Globalize it to an OS filesystem path before using it with Git/DirAccess.
		String settings_dir_res = EditorPaths::get_singleton()->get_project_settings_dir();
		String settings_dir_fs = ProjectSettings::get_singleton()->globalize_path(settings_dir_res);
		checkpoint_dir = settings_dir_fs.path_join("ai_checkpoints");
	} else {
		// Fallback: old behavior (inside the project root on disk).
		checkpoint_dir = p_project_root.path_join(".ai-checkpoints");
	}
	
	// Create checkpoint directory if it doesn't exist
	if (!DirAccess::exists(checkpoint_dir)) {
		
		Ref<DirAccess> da = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
		if (da.is_valid()) {
			Error err = da->make_dir_recursive(checkpoint_dir);
			if (err != OK && err != ERR_ALREADY_EXISTS) {
				return String(); // Return empty string on failure
			}
		}
	}
	
	return checkpoint_dir;
}

bool AICheckpointManager::_copy_project_to_checkpoint(const String &p_project_root, const String &p_checkpoint_dir) {
	
	// Remove all existing files in checkpoint directory (clean slate)
	_remove_directory_contents_recursive(p_checkpoint_dir, true); // true = preserve .git
	
	// Copy all project files except .godot and .ai-checkpoints
	return _copy_directory_recursive(p_project_root, p_checkpoint_dir, true);
}

bool AICheckpointManager::_restore_project_from_checkpoint(const String &p_checkpoint_dir, const String &p_project_root) {
	
	// Don't remove .godot folder - we need to preserve it!
	// Only remove project files, not editor state
	_remove_directory_contents_recursive(p_project_root, false); // false = preserve .godot, .ai-checkpoints, .git
	
	// Copy files from checkpoint to project
	return _copy_directory_recursive(p_checkpoint_dir, p_project_root, false);
}

bool AICheckpointManager::_remove_directory_contents_recursive(const String &p_directory, bool p_preserve_git_only) {
	Ref<DirAccess> da = DirAccess::open(p_directory);
	if (!da.is_valid()) {
		return false;
	}
	
	da->list_dir_begin();
	String file_name = da->get_next();
	
	while (!file_name.is_empty()) {
		if (file_name != "." && file_name != "..") {
			// Determine what to preserve based on context
			bool should_preserve = false;
			
			if (p_preserve_git_only) {
				// Only preserve .git (checkpoint directory cleanup)
				should_preserve = (file_name == ".git");
			} else {
				// Preserve .godot, .ai-checkpoints, and .git (project directory cleanup)
				should_preserve = (file_name == ".godot" || file_name == ".ai-checkpoints" || file_name == ".git");
			}
			
			if (!should_preserve) {
				if (!p_preserve_git_only && file_name == "project.godot") {
					should_preserve = true;
				}
			}

			if (!should_preserve) {
				String full_path = p_directory.path_join(file_name);
				
				if (da->current_is_dir()) {
					// Recursively remove directory contents then remove the directory itself
					_remove_directory_contents_recursive(full_path, false); // Don't preserve anything in subdirectories
					Error err = da->remove(file_name);
					if (err != OK) {
					}
				} else {
					// Remove file
					Error err = da->remove(file_name);
					if (err != OK) {
					}
				}
			}
		}
		file_name = da->get_next();
	}
	da->list_dir_end();
	
	return true;
}

bool AICheckpointManager::_copy_directory_recursive(const String &p_source, const String &p_destination, bool p_is_to_checkpoint) {
	Ref<DirAccess> source_da = DirAccess::open(p_source);
	if (!source_da.is_valid()) {
		return false;
	}
	
	source_da->list_dir_begin();
	String file_name = source_da->get_next();
	int files_copied = 0;
	
	while (!file_name.is_empty()) {
		if (file_name != "." && file_name != "..") {
			// Skip certain directories/files based on context
			bool should_skip = false;
			
			if (p_is_to_checkpoint) {
				// When copying TO checkpoint, skip .godot, .ai-checkpoints, .git
				// Also skip project.godot to avoid fighting the running editor's project state.
				should_skip = (file_name == ".godot" || file_name == ".ai-checkpoints" || file_name == ".git" || file_name == "project.godot");
			} else {
				// When copying FROM checkpoint back into the project, skip .git and project.godot.
				// We intentionally do NOT restore project.godot to avoid reload prompts and editor state drift.
				should_skip = (file_name == ".git" || file_name == "project.godot");
			}
			
			if (!should_skip) {
				String source_path = p_source.path_join(file_name);
				String dest_path = p_destination.path_join(file_name);
				bool is_dir = source_da->current_is_dir();
				
				if (is_dir) {
					Ref<DirAccess> fs_dir = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
					if (fs_dir.is_valid()) {
						Error make_dir_err = fs_dir->make_dir_recursive(dest_path);
						if (make_dir_err != OK && make_dir_err != ERR_ALREADY_EXISTS) {
						}
					}
					_copy_directory_recursive(source_path, dest_path, p_is_to_checkpoint);
				} else {
					Ref<DirAccess> fs_dir = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
					if (fs_dir.is_valid()) {
						Error ensure_dir_err = fs_dir->make_dir_recursive(dest_path.get_base_dir());
						if (ensure_dir_err != OK && ensure_dir_err != ERR_ALREADY_EXISTS) {
						}
					}
					
					Error copy_err = DirAccess::copy_absolute(source_path, dest_path);
					if (copy_err != OK) {
					} else {
						files_copied++;
					}
				}
			}
		}
		file_name = source_da->get_next();
	}
	source_da->list_dir_end();
	
	if (files_copied > 0) {
	}
	
	return true;
}

void AICheckpointManager::_ensure_project_gitignore_excludes_checkpoints(const String &p_project_root) {
	String gitignore_path = p_project_root.path_join(".gitignore");
	String checkpoint_ignore_line = ".ai-checkpoints/";
	
	// Check if .ai-checkpoints is already ignored
	bool already_ignored = false;
	if (FileAccess::exists(gitignore_path)) {
		Ref<FileAccess> file = FileAccess::open(gitignore_path, FileAccess::READ);
		if (file.is_valid()) {
			String content = file->get_as_text();
			if (content.find(checkpoint_ignore_line) != -1) {
				already_ignored = true;
			}
			file->close();
		}
	}
	
	if (!already_ignored) {
		
		// Append to existing .gitignore or create new one
		Ref<FileAccess> file = FileAccess::open(gitignore_path, FileAccess::WRITE_READ);
		if (file.is_valid()) {
			file->seek_end();
			file->store_line("");
			file->store_line("# AI Chat Checkpoints (isolated storage)");
			file->store_line(".ai-checkpoints/");
			file->close();
		}
	}
}

bool AICheckpointManager::_git_exec(const String &p_project_root, const List<String> &p_args, String &r_output, int &r_exitcode) {
	List<String> args;
	args.push_back("-C");
	args.push_back(p_project_root);
	for (const List<String>::Element *E = p_args.front(); E; E = E->next()) {
		args.push_back(E->get());
	}

	// read_stderr=true so we capture Git's error messages into r_output as well.
	Error err = OS::get_singleton()->execute("git", args, &r_output, &r_exitcode, true, nullptr, false);

	if (err != OK) {
	}
	if (r_exitcode != 0) {
	}

	return err == OK && r_exitcode == 0;
}

bool AICheckpointManager::_init_checkpoint_git_repo(const String &p_checkpoint_dir) {
	String git_dir = p_checkpoint_dir.path_join(".git");
	
	// Check if Git repo already exists
	if (DirAccess::exists(git_dir)) {
		return true; // Already initialized
	}
	
	
	List<String> init_args; init_args.push_back("init");
	String output; int exitcode;
	if (!_git_exec(p_checkpoint_dir, init_args, output, exitcode)) {
		return false;
	}
	
	
	// Set git config for AI checkpoints
	List<String> config_name_args; config_name_args.push_back("config"); config_name_args.push_back("user.name"); config_name_args.push_back("AI Chat Checkpoints");
	_git_exec(p_checkpoint_dir, config_name_args, output, exitcode);
	List<String> config_email_args; config_email_args.push_back("config"); config_email_args.push_back("user.email"); config_email_args.push_back("ai-chat@orcaengine.local");
	_git_exec(p_checkpoint_dir, config_email_args, output, exitcode);
	
	
	// Create minimal .gitignore for checkpoint directory
	_create_checkpoint_gitignore(p_checkpoint_dir);
	_ensure_gitignore_committed(p_checkpoint_dir);
	
	
	return true;
}

void AICheckpointManager::_create_checkpoint_gitignore(const String &p_checkpoint_dir) {
	String gitignore_path = p_checkpoint_dir.path_join(".gitignore");
	
	// Only create if it doesn't exist
	if (FileAccess::exists(gitignore_path)) {
		return;
	}
	
	
	Ref<FileAccess> gitignore = FileAccess::open(gitignore_path, FileAccess::WRITE);
	if (gitignore.is_valid()) {
		// Simple .gitignore for checkpoint directory - track everything except temp files
		gitignore->store_line("# AI Chat Checkpoint Repository");
		gitignore->store_line("# This is an isolated git repository for AI checkpoints");
		gitignore->store_line("# Track ALL project files for complete restoration");
		gitignore->store_line("");
		gitignore->store_line("# Ignore only temporary or OS files");
		gitignore->store_line("*.tmp");
		gitignore->store_line("*.temp");
		gitignore->store_line(".DS_Store");
		gitignore->store_line("Thumbs.db");
		gitignore->close();
		
	}
}

void AICheckpointManager::_ensure_gitignore_committed(const String &p_checkpoint_dir) {
	// Check if .gitignore is already tracked/committed
	List<String> ls_files_args;
	ls_files_args.push_back("ls-files");
	ls_files_args.push_back(".gitignore");
	
	String output;
	int exitcode;
	_git_exec(p_checkpoint_dir, ls_files_args, output, exitcode);
	
	if (!output.strip_edges().is_empty()) {
		// .gitignore is already tracked
		return;
	}
	
	
	// Stage .gitignore
	List<String> add_gitignore_args;
	add_gitignore_args.push_back("add");
	add_gitignore_args.push_back(".gitignore");
	
	if (!_git_exec(p_checkpoint_dir, add_gitignore_args, output, exitcode)) {
		return;
	}
	
	// Commit .gitignore
	List<String> commit_gitignore_args;
	commit_gitignore_args.push_back("commit");
	commit_gitignore_args.push_back("-m");
	commit_gitignore_args.push_back("Add .gitignore for checkpoint repository");
	
	if (!_git_exec(p_checkpoint_dir, commit_gitignore_args, output, exitcode)) {
		// Check if it's just "nothing to commit"
		if (output.find("nothing to commit") == -1) {
		}
		return;
	}
	
}

bool AICheckpointManager::_stage_all_files(const String &p_checkpoint_dir) {
	
	// Add ALL files in checkpoint directory (this is an isolated git repo)
	List<String> add_all_args;
	add_all_args.push_back("add");
	add_all_args.push_back("-A"); // Add all modifications, deletions, and new files
	
	String output;
	int exitcode;
	if (!_git_exec(p_checkpoint_dir, add_all_args, output, exitcode)) {
		return false;
	}
	
	// Verify what was staged
	List<String> status_args;
	status_args.push_back("status");
	status_args.push_back("--short");
	
	String status_output;
	_git_exec(p_checkpoint_dir, status_args, status_output, exitcode);
	
	PackedStringArray status_lines = status_output.split("\n");
	int staged_count = 0;
	for (int i = 0; i < status_lines.size(); i++) {
		String line = status_lines[i].strip_edges();
		if (!line.is_empty() && (line[0] == 'A' || line[0] == 'M' || line[0] == 'D')) {
			staged_count++;
		}
	}
	
	
	return true;
}

bool AICheckpointManager::_create_commit(const String &p_checkpoint_dir, const String &p_message, int p_message_index) {
	String checkpoint_name = _generate_checkpoint_name(p_message_index);
	String commit_message = "AI Chat Checkpoint: " + checkpoint_name + " - " + p_message.substr(0, 50);
	if (p_message.length() > 50) {
		commit_message += "...";
	}
	
	
	List<String> commit_args; commit_args.push_back("commit"); commit_args.push_back("--allow-empty"); commit_args.push_back("-m"); commit_args.push_back(commit_message);
	String output; int exitcode;
	if (!_git_exec(p_checkpoint_dir, commit_args, output, exitcode)) {
		// Check if this is a "nothing to commit" case
		if (output.find("nothing to commit") != -1 || output.find("no changes added") != -1) {
			return false; // Caller will handle tag creation
		}
		
		return false;
	}
	
	return true;
}

bool AICheckpointManager::_create_checkpoint_tag(const String &p_checkpoint_dir, int p_message_index) {
	String tag_name = _generate_checkpoint_name(p_message_index);
	String tag_message = "AI Chat checkpoint for message " + String::num_int64(p_message_index) + " - Project state BEFORE AI response";
	
	
	// Clean up old checkpoint tags for this message index
	
	List<String> list_old_tags_args;
	list_old_tags_args.push_back("tag");
	list_old_tags_args.push_back("--list");
	list_old_tags_args.push_back("msg_" + String::num_int64(p_message_index) + "_*");
	
	String old_tags_output;
	int old_tags_exitcode;
	if (_git_exec(p_checkpoint_dir, list_old_tags_args, old_tags_output, old_tags_exitcode)) {
		PackedStringArray old_tags = old_tags_output.strip_edges().split("\n");
		int deleted_count = 0;
		for (int i = 0; i < old_tags.size(); i++) {
			String old_tag = old_tags[i].strip_edges();
			if (!old_tag.is_empty()) {
				
				List<String> delete_tag_args;
				delete_tag_args.push_back("tag");
				delete_tag_args.push_back("-d");
				delete_tag_args.push_back(old_tag);
				
				String delete_output;
				if (_git_exec(p_checkpoint_dir, delete_tag_args, delete_output, old_tags_exitcode)) {
					deleted_count++;
				}
			}
		}
		if (deleted_count > 0) {
		}
	}
	
	List<String> tag_args; 
	tag_args.push_back("tag"); 
	tag_args.push_back("-f"); // Force in case we missed one
	tag_args.push_back(tag_name); 
	tag_args.push_back("-m"); 
	tag_args.push_back(tag_message);
	
	String output; 
	int exitcode;
	if (!_git_exec(p_checkpoint_dir, tag_args, output, exitcode)) {
		return false;
	}
	
	return true;
}

String AICheckpointManager::_find_checkpoint_tag(const String &p_checkpoint_dir, int p_message_index) {
	
	// Select the LATEST (NEWEST) checkpoint for this message index
	List<String> tag_args; 
	tag_args.push_back("tag"); 
	tag_args.push_back("--list"); 
	tag_args.push_back("msg_" + String::num_int64(p_message_index) + "_*"); 
	tag_args.push_back("--sort=-creatordate"); // Descending = newest first
	
	String output; 
	int exitcode;
	if (!_git_exec(p_checkpoint_dir, tag_args, output, exitcode) || output.strip_edges().is_empty()) {
		return String();
	}
	
	// First line is the NEWEST "before AI" checkpoint
	PackedStringArray tags = output.strip_edges().split("\n");
	if (tags.is_empty() || tags[0].strip_edges().is_empty()) {
		return String();
	}
	
	String found_tag = tags[0].strip_edges();
	
	// Show all matching tags for debugging
	if (tags.size() > 1) {
		for (int i = 0; i < tags.size(); i++) {
		}
		
		// Clean up old checkpoints to avoid clutter (keep only the newest)
		for (int i = 1; i < tags.size(); i++) {
			String old_tag = tags[i].strip_edges();
			if (!old_tag.is_empty()) {
				List<String> delete_old_args;
				delete_old_args.push_back("tag");
				delete_old_args.push_back("-d");
				delete_old_args.push_back(old_tag);
				
				String delete_output;
				_git_exec(p_checkpoint_dir, delete_old_args, delete_output, exitcode);
			}
		}
	}
	
	return found_tag;
}

bool AICheckpointManager::_git_reset_to_tag(const String &p_checkpoint_dir, const String &p_tag) {
	
	// Perform hard reset in checkpoint directory
	List<String> reset_args; reset_args.push_back("reset"); reset_args.push_back("--hard"); reset_args.push_back(p_tag);
	String output; int exitcode;
	if (!_git_exec(p_checkpoint_dir, reset_args, output, exitcode)) {
		return false;
	}
	
	
	return true;
}

void AICheckpointManager::_clear_editor_state() {
	
	// CRITICAL FIX: Clear ALL preview overlays FIRST
	// This is the most important step - preview overlays mask the actual disk content!
	EditorTools::clear_all_preview_overlays();
	
	// Close ALL script tabs to force complete reload
	if (ScriptEditor::get_singleton()) {
		ScriptEditor *se = ScriptEditor::get_singleton();
		
		// Get list of open scripts
		const Vector<Ref<Script>> &scripts = se->get_open_scripts();
		
		// Scripts will be closed during phase 3 refresh
		for (int i = 0; i < scripts.size(); i++) {
			if (scripts[i].is_valid()) {
				String script_path = scripts[i]->get_path();
			}
		}
	}
	
	// Close current scene
	if (EditorNode::get_singleton()) {
		EditorNode::get_singleton()->new_scene();
	}
	
	// Clear GDScript cache to force recompilation
	GDScriptCache::clear();
	
}

Vector<String> AICheckpointManager::_get_open_scripts() {
	Vector<String> open_scripts;
	
	if (ScriptEditor::get_singleton()) {
		const Vector<Ref<Script>> &scripts = ScriptEditor::get_singleton()->get_open_scripts();
		for (int i = 0; i < scripts.size(); i++) {
			if (scripts[i].is_valid() && !scripts[i]->get_path().is_empty()) {
				open_scripts.push_back(scripts[i]->get_path());
			}
		}
	}
	
	return open_scripts;
}

String AICheckpointManager::_get_current_scene_path() {
	if (EditorNode::get_singleton()) {
		Node *edited_scene = EditorNode::get_singleton()->get_edited_scene();
		if (edited_scene) {
			return edited_scene->get_scene_file_path();
		}
	}
	return String();
}

String AICheckpointManager::_generate_checkpoint_name(int p_message_index) {
	String timestamp = Time::get_singleton()->get_datetime_string_from_system();
	timestamp = timestamp.replace(":", "-").replace(" ", "_");
	return "msg_" + String::num_int64(p_message_index) + "_" + timestamp;
}

// ============================================================================
// REFRESH PHASE IMPLEMENTATIONS
// ============================================================================

void AICheckpointManager::_refresh_phase_1_resources() {
	
	// Clear GDScript cache (force recompilation from disk when accessed)
	GDScriptCache::clear();
	
	// SKIP file system scan - causes crashes during active streaming
	// Git reset already restored files on disk
	// Resources will reload when accessed
	
}

void AICheckpointManager::_refresh_phase_2_scenes() {
	
	if (EditorNode::get_singleton()) {
		Node *edited_scene = EditorNode::get_singleton()->get_edited_scene();
		String scene_path;
		if (edited_scene) {
			scene_path = edited_scene->get_scene_file_path();
		}
		
		if (!scene_path.is_empty() && FileAccess::exists(scene_path)) {
			EditorNode::get_singleton()->new_scene();
			
			EditorInterface::get_singleton()->open_scene_from_path(scene_path);
		}
	}
	
}

void AICheckpointManager::_refresh_phase_3_scripts(const Vector<String> &p_script_paths) {
	
	if (ScriptEditor::get_singleton()) {
		ScriptEditor *se = ScriptEditor::get_singleton();
		
		// CLOSE all currently open scripts to clear in-memory state
		se->call("close_all");
		
		// CRITICAL: We can't clear ResourceCache directly (it's private)
		// Instead, we'll rely on CACHE_MODE_IGNORE when loading resources
		// and forcibly set source code from disk
		
		// REOPEN from restored disk files (bypassing ALL caches)
		for (int i = 0; i < p_script_paths.size(); i++) {
			String script_path = p_script_paths[i];
			
			// Verify file actually exists on disk after git reset
			if (!FileAccess::exists(script_path)) {
				continue;
			}
			
			// CRITICAL: Read the actual file content from disk DIRECTLY
			// Don't rely on ResourceLoader cache at all
			Error err;
			String disk_content = FileAccess::get_file_as_string(script_path, &err);
			if (err != OK) {
				continue;
			}
			
			
			// Load fresh from disk with CACHE_MODE_IGNORE to force fresh read
			Ref<Resource> resource = ResourceLoader::load(script_path, "", ResourceFormatLoader::CACHE_MODE_IGNORE);
			Ref<Script> script = resource;
			if (script.is_valid()) {
				// FORCE the script to use the disk content (override any cached content)
				script->set_source_code(disk_content);
				script->reload(true); // Force full reload
				
				se->edit(script);
			} else {
			}
		}
	}
	
}

void AICheckpointManager::_refresh_phase_4_ui() {
	
	// Force main window refresh
	if (EditorNode::get_singleton()) {
		Window *main_window = EditorNode::get_singleton()->get_window();
		if (main_window) {
			main_window->grab_focus();
			main_window->move_to_foreground();
			
			if (DisplayServer::get_singleton()) {
				DisplayServer::get_singleton()->process_events();
				DisplayServer::get_singleton()->force_process_and_drop_events();
			}
		}
		
		// Force complete redraw by triggering theme change notification
		EditorNode::get_singleton()->get_tree()->get_root()->propagate_notification(Control::NOTIFICATION_THEME_CHANGED);
	}
	
}

void AICheckpointManager::_refresh_phase_5_docks() {
	
	// CRITICAL FIX: Git restore changes files on disk, but Godot's FileSystem doesn't know!
	// We MUST force a filesystem scan or files won't appear in the navigator
	
	if (EditorFileSystem::get_singleton()) {
		// NUCLEAR OPTION: Full filesystem scan to rebuild file tree
		// This is necessary because Git changed files behind Godot's back
		EditorFileSystem::get_singleton()->scan();
		
		// Also scan for changes (more lightweight, catches modifications)
		EditorFileSystem::get_singleton()->scan_changes();
	} else {
	}
	
	// Force FileSystem dock to refresh its view
	if (FileSystemDock::get_singleton()) {
		// Navigate to current path to force refresh
		FileSystemDock::get_singleton()->navigate_to_path("res://");
	}
	
}

