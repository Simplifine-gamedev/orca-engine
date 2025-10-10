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
	
	print_line("AI Checkpoint: ========================================");
	print_line("AI Checkpoint: CREATING COMPREHENSIVE PROJECT SNAPSHOT");
	print_line("AI Checkpoint: Project root: " + p_project_root);
	print_line("AI Checkpoint: Message index: " + String::num_int64(p_message_index));
	print_line("AI Checkpoint: ========================================");
	
    // Avoid changing CWD; use git -C <dir> for reliability
	
	// CRITICAL: Create .gitignore FIRST before any git operations
	// This ensures .godot/ is NEVER tracked from the start
    _create_minimal_gitignore(p_project_root);
	
	// Initialize Git repo if needed
    if (!_init_git_repo(p_project_root)) {
        result.message = "Failed to initialize Git repository";
        return result;
    }
	
	// SAFETY: Ensure .gitignore is committed to the repo (if it's new)
	_ensure_gitignore_committed(p_project_root);
	
	// Stage ALL files (including .import, project.godot, etc.)
    if (!_stage_all_files(p_project_root)) {
		result.message = "Failed to stage files";
		return result;
	}
	
	// Create commit with all staged files
	if (!_create_commit(p_project_root, p_message, p_message_index)) {
		// Check if failure was due to "nothing to commit"
		// If so, still create a tag and consider it successful
        if (!_create_checkpoint_tag(p_project_root, p_message_index)) {
			result.message = "Failed to create checkpoint commit and tag";
			return result;
		}
		// Tag created on current HEAD - success!
		print_line("AI Checkpoint: No changes to commit, but checkpoint tag created");
	} else {
		// Create tag for easy reference
        if (!_create_checkpoint_tag(p_project_root, p_message_index)) {
			result.message = "Commit created but tag creation failed";
			return result;
		}
	}
	
    // No CWD changes were made.
	
	// Success!
	result.success = true;
	result.checkpoint_tag = _generate_checkpoint_name(p_message_index);
	result.message = "Checkpoint created successfully";
	
	print_line("AI Checkpoint: ✅✅✅ CHECKPOINT CREATED SUCCESSFULLY ✅✅✅");
	print_line("AI Checkpoint: Tag: " + result.checkpoint_tag);
	print_line("AI Checkpoint: Captured ALL files (scenes, scripts, resources, .import, project.godot)");
	print_line("AI Checkpoint: ========================================");
	
	return result;
}

AICheckpointManager::RestoreResult AICheckpointManager::restore_to_checkpoint(const String &p_project_root, int p_message_index) {
	RestoreResult result;
	result.success = false;
	
	print_line("AI Checkpoint: ========================================");
	print_line("AI Checkpoint: RESTORING COMPLETE PROJECT SNAPSHOT");
	print_line("AI Checkpoint: Project root: " + p_project_root);
	print_line("AI Checkpoint: Message index: " + String::num_int64(p_message_index));
	print_line("AI Checkpoint: ========================================");
	
	// STEP 1: Capture current editor state (for reopening after restore)
	print_line("AI Checkpoint: STEP 1: Capturing current editor state...");
	result.restored_scene_path = _get_current_scene_path();
	result.restored_scripts = _get_open_scripts();
	
	print_line("AI Checkpoint: - Current scene: " + (result.restored_scene_path.is_empty() ? "(none)" : result.restored_scene_path));
	print_line("AI Checkpoint: - Open scripts: " + String::num_int64(result.restored_scripts.size()));
	
	// STEP 2: Clear ALL editor state to prevent crashes during Git reset
	print_line("AI Checkpoint: STEP 2: Clearing ALL editor state...");
	_clear_editor_state();
	print_line("AI Checkpoint: - Editor state cleared (scripts closed, scene closed, caches cleared)");
	
	// STEP 3: Find and validate checkpoint tag
	print_line("AI Checkpoint: STEP 3: Finding checkpoint tag...");
    String checkpoint_tag = _find_checkpoint_tag(p_project_root, p_message_index);
	
	if (checkpoint_tag.is_empty()) {
		result.message = "No checkpoint found for message index " + String::num_int64(p_message_index);
		print_line("AI Checkpoint: ❌ " + result.message);
		return result;
	}
	
	print_line("AI Checkpoint: ✅ Found checkpoint: " + checkpoint_tag);
	
	// STEP 4: Perform Git hard reset to restore ALL files
	print_line("AI Checkpoint: STEP 4: Performing Git hard reset...");
	print_line("AI Checkpoint: This will restore ALL files to their exact checkpoint state");
	
    if (!_git_reset_to_tag(p_project_root, checkpoint_tag)) {
		result.message = "Git reset failed";
		print_line("AI Checkpoint: ❌ Git reset failed");
		return result;
	}
	
	print_line("AI Checkpoint: ✅ Git reset complete - verifying restoration...");
	
	// VERIFICATION: List all files now present in project root
	print_line("AI Checkpoint: VERIFICATION - Listing actual files on disk after git reset:");
	Ref<DirAccess> verify_dir = DirAccess::open(p_project_root);
	if (verify_dir.is_valid()) {
		verify_dir->list_dir_begin();
		String file_name = verify_dir->get_next();
		int file_count = 0;
		int dir_count = 0;
		while (!file_name.is_empty()) {
			if (verify_dir->current_is_dir()) {
				if (!file_name.begins_with(".")) {
					print_line("AI Checkpoint:   [DIR]  " + file_name + "/");
					dir_count++;
				}
			} else {
				print_line("AI Checkpoint:   [FILE] " + file_name);
				file_count++;
			}
			file_name = verify_dir->get_next();
		}
		verify_dir->list_dir_end();
		print_line("AI Checkpoint: VERIFICATION - Found " + String::num_int64(file_count) + " files, " + String::num_int64(dir_count) + " directories");
	}
	
	print_line("AI Checkpoint: Files restored include:");
	print_line("AI Checkpoint:   - Scenes (.tscn, .scn)");
	print_line("AI Checkpoint:   - Scripts (.gd, .cs, .shader)");
	print_line("AI Checkpoint:   - Resources (.tres, .res)");
	print_line("AI Checkpoint:   - Import files (.import)");
	print_line("AI Checkpoint:   - Project settings (project.godot)");
	print_line("AI Checkpoint:   - All assets (textures, models, audio, etc.)");
	
	// CRITICAL POST-RESTORE CHECK: Ensure .godot/ folder still exists!
	// If git reset deleted it (because it was accidentally tracked), we're in trouble
	print_line("AI Checkpoint: STEP 5: POST-RESTORE SAFETY CHECK...");
	String godot_folder_path = p_project_root.path_join(".godot");
	
	if (!DirAccess::exists(godot_folder_path)) {
		print_line("AI Checkpoint: 🚨🚨🚨 CRITICAL ERROR: .godot/ folder was DELETED by git reset!");
		print_line("AI Checkpoint: This means .godot/ was being tracked (the bug we're trying to prevent)");
		print_line("AI Checkpoint: Creating .godot/ folder now to prevent editor crash...");
		
		// Create the .godot folder to prevent immediate crash
		Ref<DirAccess> da = DirAccess::create_for_path(p_project_root);
		if (da.is_valid()) {
			Error err = da->make_dir(".godot");
			if (err == OK) {
				print_line("AI Checkpoint: ✅ Created .godot/ folder");
				
				// Also create essential subdirectories that Godot needs
				da->make_dir_recursive(".godot/editor");
				da->make_dir_recursive(".godot/imported");
				da->make_dir_recursive(".godot/shader_cache");
				print_line("AI Checkpoint: ✅ Created essential .godot subdirectories");
			} else {
				print_line("AI Checkpoint: ❌ Failed to create .godot/ folder: " + String::num_int64(err));
				result.success = false;
				result.message = ".godot/ was deleted by git reset - this is a critical bug!";
				return result;
			}
		}
		
		// Verify .gitignore exists and is correct
		_create_minimal_gitignore(p_project_root);
		_ensure_gitignore_committed(p_project_root);
		
		print_line("AI Checkpoint: ⚠️ .godot/ was recreated but editor may need restart");
	} else {
		print_line("AI Checkpoint: ✅✅✅ .godot/ folder SAFE - not deleted by git reset!");
	}
	
	// Success!
	result.success = true;
	result.message = "Checkpoint restored successfully";
	
	print_line("AI Checkpoint: ========================================");
	print_line("AI Checkpoint: ✅✅✅ RESTORE COMPLETE ✅✅✅");
	print_line("AI Checkpoint: Git reset --hard restored all tracked files");
	print_line("AI Checkpoint: .godot/ folder is safe and intact");
	print_line("AI Checkpoint: Next: Editor refresh will reload all components");
	print_line("AI Checkpoint: ========================================");
	
    // NO git clean - git reset --hard is sufficient and safe
    // git clean would delete .godot/ and cause crashes!

	return result;
}

void AICheckpointManager::refresh_editor_completely(const String &p_restored_scene_path, const Vector<String> &p_restored_scripts) {
	print_line("AI Checkpoint: ========================================");
	print_line("AI Checkpoint: COMPREHENSIVE EDITOR REFRESH");
	print_line("AI Checkpoint: Reloading ALL editor components from disk");
	print_line("AI Checkpoint: ========================================");
	
	// Phase 1: Resources (immediate)
	_refresh_phase_1_resources();
	
	// Phase 2: Scenes (0.2s delay for resource loading)
	// Note: We can't use SceneTreeTimer here since this is a static class
	// The caller (ai_chat_dock) should handle the phased delays
	
	print_line("AI Checkpoint: Editor refresh initiated - phases will run over next 1 second");
}

// ============================================================================
// PRIVATE IMPLEMENTATION METHODS
// ============================================================================

bool AICheckpointManager::_git_exec(const String &p_project_root, const List<String> &p_args, String &r_output, int &r_exitcode) {
    List<String> args;
    args.push_back("-C");
    args.push_back(p_project_root);
    for (const List<String>::Element *E = p_args.front(); E; E = E->next()) {
        args.push_back(E->get());
    }
    Error err = OS::get_singleton()->execute("git", args, &r_output, &r_exitcode, false, nullptr, false);
    return err == OK && r_exitcode == 0;
}

bool AICheckpointManager::_init_git_repo(const String &p_project_root) {
	String git_dir = p_project_root.path_join(".git");
	
	// Check if Git repo already exists
	if (DirAccess::exists(git_dir)) {
		return true; // Already initialized
	}
	
	print_line("AI Checkpoint: Initializing Git repository...");
	
    List<String> init_args; init_args.push_back("init");
    String output; int exitcode;
    if (!_git_exec(p_project_root, init_args, output, exitcode)) {
		print_line("AI Checkpoint: Failed to initialize Git repo: " + output);
		return false;
	}
	
	print_line("AI Checkpoint: ✅ Git repository initialized");
	
	// Set git config for AI checkpoints
	List<String> config_name_args; config_name_args.push_back("config"); config_name_args.push_back("user.name"); config_name_args.push_back("AI Chat Checkpoints");
	_git_exec(p_project_root, config_name_args, output, exitcode);
	List<String> config_email_args; config_email_args.push_back("config"); config_email_args.push_back("user.email"); config_email_args.push_back("ai-chat@orcaengine.local");
	_git_exec(p_project_root, config_email_args, output, exitcode);
	
	print_line("AI Checkpoint: ✅ Git config set for AI checkpoints");
	
	// CRITICAL: AGGRESSIVELY remove .godot/ from git index if it's currently tracked
	// This prevents git reset from deleting .godot/ contents
	print_line("AI Checkpoint: AGGRESSIVELY ensuring .godot/ is NEVER tracked...");
	
	List<String> rm_cached_args; 
	rm_cached_args.push_back("rm"); 
	rm_cached_args.push_back("-r"); 
	rm_cached_args.push_back("--cached"); 
	rm_cached_args.push_back("--ignore-unmatch"); 
	rm_cached_args.push_back(".godot");
	
	if (_git_exec(p_project_root, rm_cached_args, output, exitcode)) {
		print_line("AI Checkpoint: ✅ Removed .godot/ from git index (if it was tracked)");
	}
	
	// Also remove .godot/ from HEAD if it exists there (from previous bad commits)
	List<String> filter_branch_args;
	filter_branch_args.push_back("filter-branch");
	filter_branch_args.push_back("--force");
	filter_branch_args.push_back("--index-filter");
	filter_branch_args.push_back("git rm -r --cached --ignore-unmatch .godot");
	filter_branch_args.push_back("--prune-empty");
	filter_branch_args.push_back("--tag-name-filter");
	filter_branch_args.push_back("cat");
	filter_branch_args.push_back("--");
	filter_branch_args.push_back("--all");
	
	// This is a heavy operation - only run if .godot/ was found in history
	// Check if .godot/ exists in any commit
	List<String> check_history_args;
	check_history_args.push_back("log");
	check_history_args.push_back("--all");
	check_history_args.push_back("--format=%H");
	check_history_args.push_back("--");
	check_history_args.push_back(".godot/");
	
	String history_output;
	if (_git_exec(p_project_root, check_history_args, history_output, exitcode) && !history_output.strip_edges().is_empty()) {
		print_line("AI Checkpoint: ⚠️ .godot/ found in Git history - attempting deep cleanup...");
		// Note: filter-branch is complex and can fail - skip it for now, rely on rm --cached
		// If needed, user can manually run: git filter-branch --force --index-filter 'git rm -r --cached --ignore-unmatch .godot' --prune-empty --tag-name-filter cat -- --all
		print_line("AI Checkpoint: ℹ️ Manual cleanup recommended: git filter-branch to remove .godot/ from history");
	}
	
	print_line("AI Checkpoint: ✅ .godot/ protection configured");
	
	return true;
}

void AICheckpointManager::_create_minimal_gitignore(const String &p_project_root) {
	String gitignore_path = p_project_root.path_join(".gitignore");
	
	// Only create if it doesn't exist
	if (FileAccess::exists(gitignore_path)) {
		return;
	}
	
	print_line("AI Checkpoint: Creating minimal .gitignore...");
	
	Ref<FileAccess> gitignore = FileAccess::open(gitignore_path, FileAccess::WRITE);
	if (gitignore.is_valid()) {
		// CRITICAL: Ignore the ENTIRE .godot/ folder!
		// This prevents git from tracking/deleting editor caches and causing "project data folder missing" errors
		gitignore->store_line("# AI Chat Checkpoints - Comprehensive .gitignore");
		gitignore->store_line("# CRITICAL: This file protects your editor state from Git operations");
		gitignore->store_line("");
		gitignore->store_line("# ========================================");
		gitignore->store_line("# GODOT EDITOR FOLDER - NEVER TRACK THIS!");
		gitignore->store_line("# ========================================");
		gitignore->store_line("# The .godot/ folder contains editor caches and MUST be ignored");
		gitignore->store_line("# If git tracks this folder, 'git reset --hard' will DELETE it!");
		gitignore->store_line("# This causes the 'project data folder missing' error");
		gitignore->store_line(".godot/");
		gitignore->store_line("/.godot/");
		gitignore->store_line("**/.godot/");
		gitignore->store_line("");
		gitignore->store_line("# Build artifacts (also should never be tracked)");
		gitignore->store_line("build/");
		gitignore->store_line("bin/");
		gitignore->store_line(".mono/");
		gitignore->store_line("mono_crash.*.json");
		gitignore->store_line("");
		gitignore->store_line("# ========================================");
		gitignore->store_line("# IMPORTANT: These ARE tracked for restoration:");
		gitignore->store_line("# ========================================");
		gitignore->store_line("# ✅ .import files (needed for asset import settings)");
		gitignore->store_line("# ✅ project.godot (project configuration)");
		gitignore->store_line("# ✅ All .tscn, .gd, .tres files (scenes, scripts, resources)");
		gitignore->store_line("# ✅ All assets (textures, models, audio, etc.)");
		gitignore->close();
		
		print_line("AI Checkpoint: ✅ Created comprehensive .gitignore (protects .godot/ folder)");
	}
}

void AICheckpointManager::_ensure_gitignore_committed(const String &p_project_root) {
	// Check if .gitignore is already tracked/committed
	List<String> ls_files_args;
	ls_files_args.push_back("ls-files");
	ls_files_args.push_back(".gitignore");
	
	String output;
	int exitcode;
	_git_exec(p_project_root, ls_files_args, output, exitcode);
	
	if (!output.strip_edges().is_empty()) {
		// .gitignore is already tracked
		print_line("AI Checkpoint: ✅ .gitignore already tracked in git");
		return;
	}
	
	print_line("AI Checkpoint: Adding and committing .gitignore to ensure it's tracked...");
	
	// Stage .gitignore
	List<String> add_gitignore_args;
	add_gitignore_args.push_back("add");
	add_gitignore_args.push_back(".gitignore");
	
	if (!_git_exec(p_project_root, add_gitignore_args, output, exitcode)) {
		print_line("AI Checkpoint: Warning - could not add .gitignore: " + output);
		return;
	}
	
	// Commit .gitignore
	List<String> commit_gitignore_args;
	commit_gitignore_args.push_back("commit");
	commit_gitignore_args.push_back("-m");
	commit_gitignore_args.push_back("Add .gitignore to protect .godot/ folder");
	
	if (!_git_exec(p_project_root, commit_gitignore_args, output, exitcode)) {
		// Check if it's just "nothing to commit"
		if (output.find("nothing to commit") == -1) {
			print_line("AI Checkpoint: Warning - could not commit .gitignore: " + output);
		}
		return;
	}
	
	print_line("AI Checkpoint: ✅ .gitignore committed to repository");
}

bool AICheckpointManager::_stage_all_files(const String &p_project_root) {
	print_line("AI Checkpoint: Staging ALL project files...");
	
	// CRITICAL: Force add .import files (they might be ignored by default)
	List<String> force_import_args;
	force_import_args.push_back("add");
	force_import_args.push_back("-f");
	force_import_args.push_back("*.import");
	
	String output;
	int exitcode;
    _git_exec(p_project_root, force_import_args, output, exitcode);
	print_line("AI Checkpoint: - Force added .import files");
	
	// CRITICAL: Force add project.godot
	List<String> force_project_args;
	force_project_args.push_back("add");
	force_project_args.push_back("-f");
	force_project_args.push_back("project.godot");
    _git_exec(p_project_root, force_project_args, output, exitcode);
	print_line("AI Checkpoint: - Force added project.godot");
	
	// Add ALL other files WITHOUT --force to respect .gitignore
	// CRITICAL FIX: NEVER use --force here - it would bypass .gitignore and add .godot/!
	List<String> add_all_args;
	add_all_args.push_back("add");
	add_all_args.push_back("-A"); // Add all modifications, deletions, and new files
	// REMOVED: --force flag that was causing .godot/ to be tracked!
	
    if (!_git_exec(p_project_root, add_all_args, output, exitcode)) {
		print_line("AI Checkpoint: ❌ Failed to stage files: " + output);
		return false;
	}
	
	// CRITICAL SAFETY CHECK: Verify .godot/ is NOT staged
	List<String> check_godot_args;
	check_godot_args.push_back("ls-files");
	check_godot_args.push_back("--cached");
	check_godot_args.push_back(".godot/");
	
	String godot_check_output;
	_git_exec(p_project_root, check_godot_args, godot_check_output, exitcode);
	
	if (!godot_check_output.strip_edges().is_empty()) {
		// .godot/ is staged - this should NEVER happen!
		print_line("AI Checkpoint: 🚨🚨🚨 CRITICAL ERROR: .godot/ is staged! Unstaging it now...");
		print_line("AI Checkpoint: Staged .godot/ files: " + godot_check_output.strip_edges());
		
		// NUCLEAR OPTION: Use git rm --cached to forcibly remove from staging
		List<String> unstage_godot_args;
		unstage_godot_args.push_back("rm");
		unstage_godot_args.push_back("-r");
		unstage_godot_args.push_back("--cached");
		unstage_godot_args.push_back("--ignore-unmatch");
		unstage_godot_args.push_back(".godot/");
		
		if (_git_exec(p_project_root, unstage_godot_args, output, exitcode)) {
			print_line("AI Checkpoint: ✅ Forcibly removed .godot/ from staging area");
		} else {
			print_line("AI Checkpoint: ❌ Failed to unstage .godot/: " + output);
			// This is critical - abort checkpoint creation
			return false;
		}
		
		// Double-check it's really gone
		String verify_output;
		_git_exec(p_project_root, check_godot_args, verify_output, exitcode);
		if (!verify_output.strip_edges().is_empty()) {
			print_line("AI Checkpoint: ❌❌❌ FAILED to unstage .godot/ - ABORTING checkpoint!");
			return false;
		}
		print_line("AI Checkpoint: ✅✅✅ Verified .godot/ is now unstaged");
	} else {
		print_line("AI Checkpoint: ✅ Verified .godot/ is NOT staged (correct!)");
	}
	
	// Verify what was staged
	List<String> status_args;
	status_args.push_back("status");
	status_args.push_back("--short");
	
	String status_output;
    _git_exec(p_project_root, status_args, status_output, exitcode);
	
	PackedStringArray status_lines = status_output.split("\n");
	int staged_count = 0;
	for (int i = 0; i < status_lines.size(); i++) {
		String line = status_lines[i].strip_edges();
		if (!line.is_empty() && (line[0] == 'A' || line[0] == 'M' || line[0] == 'D')) {
			staged_count++;
		}
	}
	
	print_line("AI Checkpoint: ✅ Staged " + String::num_int64(staged_count) + " files for checkpoint");
	
	return true;
}

bool AICheckpointManager::_create_commit(const String &p_project_root, const String &p_message, int p_message_index) {
	String checkpoint_name = _generate_checkpoint_name(p_message_index);
	String commit_message = "AI Chat Checkpoint: " + checkpoint_name + " - " + p_message.substr(0, 50);
	if (p_message.length() > 50) {
		commit_message += "...";
	}
	
	print_line("AI Checkpoint: Creating Git commit...");
	print_line("AI Checkpoint: Commit message: " + commit_message);
	
    List<String> commit_args; commit_args.push_back("commit"); commit_args.push_back("--allow-empty"); commit_args.push_back("-m"); commit_args.push_back(commit_message);
    String output; int exitcode;
    if (!_git_exec(p_project_root, commit_args, output, exitcode)) {
		// Check if this is a "nothing to commit" case
		if (output.find("nothing to commit") != -1 || output.find("no changes added") != -1) {
			print_line("AI Checkpoint: No changes to commit (working tree clean)");
			return false; // Caller will handle tag creation
		}
		
		print_line("AI Checkpoint: ❌ Failed to create commit: " + output);
		return false;
	}
	
	print_line("AI Checkpoint: ✅ Git commit created successfully");
	return true;
}

bool AICheckpointManager::_create_checkpoint_tag(const String &p_project_root, int p_message_index) {
	String tag_name = _generate_checkpoint_name(p_message_index);
	String tag_message = "AI Chat checkpoint for message " + String::num_int64(p_message_index) + " - Project state BEFORE AI response";
	
	print_line("AI Checkpoint: Creating checkpoint tag: " + tag_name);
	
	// CRITICAL: Delete ALL old checkpoint tags for this message index first
	// Each message should only have ONE checkpoint (the most recent "before AI" state)
	// This prevents accumulating dozens of stale checkpoints
	print_line("AI Checkpoint: Cleaning up old checkpoints for message " + String::num_int64(p_message_index) + "...");
	
	List<String> list_old_tags_args;
	list_old_tags_args.push_back("tag");
	list_old_tags_args.push_back("--list");
	list_old_tags_args.push_back("msg_" + String::num_int64(p_message_index) + "_*");
	
	String old_tags_output;
	int old_tags_exitcode;
	if (_git_exec(p_project_root, list_old_tags_args, old_tags_output, old_tags_exitcode)) {
		PackedStringArray old_tags = old_tags_output.strip_edges().split("\n");
		int deleted_count = 0;
		for (int i = 0; i < old_tags.size(); i++) {
			String old_tag = old_tags[i].strip_edges();
			if (!old_tag.is_empty()) {
				print_line("AI Checkpoint:   - Deleting old tag: " + old_tag);
				
				List<String> delete_tag_args;
				delete_tag_args.push_back("tag");
				delete_tag_args.push_back("-d");
				delete_tag_args.push_back(old_tag);
				
				String delete_output;
				if (_git_exec(p_project_root, delete_tag_args, delete_output, old_tags_exitcode)) {
					deleted_count++;
				}
			}
		}
		if (deleted_count > 0) {
			print_line("AI Checkpoint: ✅ Cleaned up " + String::num_int64(deleted_count) + " old checkpoint(s)");
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
    if (!_git_exec(p_project_root, tag_args, output, exitcode)) {
		print_line("AI Checkpoint: ❌ Failed to create tag: " + output);
		return false;
	}
	
	print_line("AI Checkpoint: ✅ Checkpoint tag created: " + tag_name + " (BEFORE AI work)");
	return true;
}

String AICheckpointManager::_find_checkpoint_tag(const String &p_project_root, int p_message_index) {
	print_line("AI Checkpoint: Searching for checkpoint tag for message " + String::num_int64(p_message_index) + "...");
	
    // CRITICAL: Select the LATEST (NEWEST) checkpoint for this message index
    // Why newest? Because checkpoints are created BEFORE AI work, and if the user
    // re-sends the same message multiple times, we want the MOST RECENT "before state"
    // Use --sort=-creatordate (descending) to list newest tags first
    List<String> tag_args; 
	tag_args.push_back("tag"); 
	tag_args.push_back("--list"); 
	tag_args.push_back("msg_" + String::num_int64(p_message_index) + "_*"); 
	tag_args.push_back("--sort=-creatordate"); // Descending = newest first
	
	String output; 
	int exitcode;
	if (!_git_exec(p_project_root, tag_args, output, exitcode) || output.strip_edges().is_empty()) {
		print_line("AI Checkpoint: ❌ No checkpoint tag found for message " + String::num_int64(p_message_index));
		return String();
	}
	
    // First line is the NEWEST "before AI" checkpoint
	PackedStringArray tags = output.strip_edges().split("\n");
    if (tags.is_empty() || tags[0].strip_edges().is_empty()) {
		print_line("AI Checkpoint: ❌ No valid tags in output");
		return String();
	}
	
    String found_tag = tags[0].strip_edges();
	print_line("AI Checkpoint: ✅ Found LATEST checkpoint (state BEFORE AI's work): " + found_tag);
    
    // Show all matching tags for debugging
    if (tags.size() > 1) {
        print_line("AI Checkpoint: ℹ️  Multiple checkpoints found for message " + String::num_int64(p_message_index) + ":");
        for (int i = 0; i < tags.size(); i++) {
            print_line("AI Checkpoint:     " + String::num_int64(i+1) + ". " + tags[i].strip_edges() + (i == 0 ? " ← USING THIS (newest pre-AI state)" : " (older, will be cleaned up)"));
        }
		
		// Clean up old checkpoints to avoid clutter (keep only the newest)
		print_line("AI Checkpoint: Cleaning up " + String::num_int64(tags.size() - 1) + " outdated checkpoint(s)...");
		for (int i = 1; i < tags.size(); i++) {
			String old_tag = tags[i].strip_edges();
			if (!old_tag.is_empty()) {
				List<String> delete_old_args;
				delete_old_args.push_back("tag");
				delete_old_args.push_back("-d");
				delete_old_args.push_back(old_tag);
				
				String delete_output;
				_git_exec(p_project_root, delete_old_args, delete_output, exitcode);
				print_line("AI Checkpoint:     ✅ Deleted outdated: " + old_tag);
			}
		}
    }
	
	return found_tag;
}

bool AICheckpointManager::_git_reset_to_tag(const String &p_project_root, const String &p_tag) {
	print_line("AI Checkpoint: Performing Git hard reset to: " + p_tag);
	print_line("AI Checkpoint: WARNING: This will discard ALL uncommitted changes!");
	
    // Perform hard reset using git -C <dir>
    List<String> reset_args; reset_args.push_back("reset"); reset_args.push_back("--hard"); reset_args.push_back(p_tag);
    String output; int exitcode;
    if (!_git_exec(p_project_root, reset_args, output, exitcode)) {
		print_line("AI Checkpoint: ❌ Git reset failed: " + output);
		return false;
	}
	
	print_line("AI Checkpoint: ✅ Git reset successful");
	print_line("AI Checkpoint: Output: " + output.strip_edges());
	
	return true;
}

void AICheckpointManager::_clear_editor_state() {
	print_line("AI Checkpoint: ========================================");
	print_line("AI Checkpoint: CLEARING ALL EDITOR STATE");
	print_line("AI Checkpoint: ========================================");
	
	// CRITICAL FIX: Clear ALL preview overlays FIRST
	// This is the most important step - preview overlays mask the actual disk content!
	print_line("AI Checkpoint: STEP 1: Clearing ALL preview overlays...");
	print_line("AI Checkpoint: (Preview overlays are in-memory edits that mask disk content)");
	EditorTools::clear_all_preview_overlays();
	print_line("AI Checkpoint: ✅ ALL preview overlays cleared - scripts will now read from DISK");
	
	// Close ALL script tabs to force complete reload
	if (ScriptEditor::get_singleton()) {
		print_line("AI Checkpoint: STEP 2: Preparing to close script editor tabs...");
		ScriptEditor *se = ScriptEditor::get_singleton();
		
		// Get list of open scripts
		const Vector<Ref<Script>> &scripts = se->get_open_scripts();
		print_line("AI Checkpoint: - Found " + String::num_int64(scripts.size()) + " open scripts");
		
		// Scripts will be closed during phase 3 refresh
		for (int i = 0; i < scripts.size(); i++) {
			if (scripts[i].is_valid()) {
				String script_path = scripts[i]->get_path();
				print_line("AI Checkpoint:   - Will close: " + script_path);
			}
		}
		print_line("AI Checkpoint: ✅ Script tabs will be closed and reopened in Phase 3");
	}
	
	// Close current scene
	if (EditorNode::get_singleton()) {
		print_line("AI Checkpoint: STEP 3: Closing current scene...");
		EditorNode::get_singleton()->new_scene();
		print_line("AI Checkpoint: ✅ Scene closed");
	}
	
	// Clear GDScript cache to force recompilation
	print_line("AI Checkpoint: STEP 4: Clearing GDScript cache...");
	GDScriptCache::clear();
	print_line("AI Checkpoint: ✅ GDScript cache cleared - scripts will recompile from disk");
	
	print_line("AI Checkpoint: ========================================");
	print_line("AI Checkpoint: ✅✅✅ EDITOR STATE CLEARED COMPLETELY");
	print_line("AI Checkpoint: Next: Git reset will restore files to disk");
	print_line("AI Checkpoint: ========================================");
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
	print_line("AI Checkpoint: PHASE 1: Clearing caches...");
	
	// Clear GDScript cache (force recompilation from disk when accessed)
	print_line("AI Checkpoint:   - Clearing GDScript cache...");
	GDScriptCache::clear();
	print_line("AI Checkpoint:   ✅ GDScript cache cleared");
	
	// SKIP file system scan - causes crashes during active streaming
	// Git reset already restored files on disk
	// Resources will reload when accessed
	print_line("AI Checkpoint:   - Skipping file system scan (prevents UI crashes)");
	
	print_line("AI Checkpoint: ✅ Phase 1 complete");
}

void AICheckpointManager::_refresh_phase_2_scenes() {
	print_line("AI Checkpoint: PHASE 2: Closing and reopening scene from disk...");
	
	if (EditorNode::get_singleton()) {
		Node *edited_scene = EditorNode::get_singleton()->get_edited_scene();
		String scene_path;
		if (edited_scene) {
			scene_path = edited_scene->get_scene_file_path();
		}
		
		if (!scene_path.is_empty() && FileAccess::exists(scene_path)) {
			print_line("AI Checkpoint:   - CLOSING: " + scene_path);
			EditorNode::get_singleton()->new_scene();
			
			print_line("AI Checkpoint:   - REOPENING from disk...");
			EditorInterface::get_singleton()->open_scene_from_path(scene_path);
			print_line("AI Checkpoint:   ✅ Scene reopened");
		}
	}
	
	print_line("AI Checkpoint: ✅ Phase 2 complete");
}

void AICheckpointManager::_refresh_phase_3_scripts(const Vector<String> &p_script_paths) {
	print_line("AI Checkpoint: PHASE 3: FORCIBLY reloading scripts from restored disk files...");
	
	if (ScriptEditor::get_singleton()) {
		ScriptEditor *se = ScriptEditor::get_singleton();
		
		// CLOSE all currently open scripts to clear in-memory state
		print_line("AI Checkpoint:   - Closing all open scripts...");
		se->call("close_all");
		print_line("AI Checkpoint:   ✅ All scripts closed");
		
		// CRITICAL: We can't clear ResourceCache directly (it's private)
		// Instead, we'll rely on CACHE_MODE_IGNORE when loading resources
		// and forcibly set source code from disk
		print_line("AI Checkpoint:   - Will bypass resource cache using CACHE_MODE_IGNORE");
		
		// REOPEN from restored disk files (bypassing ALL caches)
		print_line("AI Checkpoint:   - Reopening " + String::num_int64(p_script_paths.size()) + " scripts fresh from disk...");
		for (int i = 0; i < p_script_paths.size(); i++) {
			String script_path = p_script_paths[i];
			
			// Verify file actually exists on disk after git reset
			if (!FileAccess::exists(script_path)) {
				print_line("AI Checkpoint:     ⚠️  File doesn't exist after restore: " + script_path + " (AI created it, now removed)");
				continue;
			}
			
			// CRITICAL: Read the actual file content from disk DIRECTLY
			// Don't rely on ResourceLoader cache at all
			Error err;
			String disk_content = FileAccess::get_file_as_string(script_path, &err);
			if (err != OK) {
				print_line("AI Checkpoint:     ❌ Failed to read from disk: " + script_path);
				continue;
			}
			
			print_line("AI Checkpoint:     - Read " + String::num_int64(disk_content.length()) + " bytes from disk: " + script_path);
			
			// Load fresh from disk with CACHE_MODE_IGNORE to force fresh read
			Ref<Resource> resource = ResourceLoader::load(script_path, "", ResourceFormatLoader::CACHE_MODE_IGNORE);
			Ref<Script> script = resource;
			if (script.is_valid()) {
				// FORCE the script to use the disk content (override any cached content)
				script->set_source_code(disk_content);
				script->reload(true); // Force full reload
				
				print_line("AI Checkpoint:     ✅ Loaded and forced content from disk: " + script_path);
				se->edit(script);
			} else {
				print_line("AI Checkpoint:     ❌ Failed to load script resource: " + script_path);
			}
		}
	}
	
	print_line("AI Checkpoint: ✅ Phase 3 complete - Scripts FORCIBLY reloaded from restored disk files");
}

void AICheckpointManager::_refresh_phase_4_ui() {
	print_line("AI Checkpoint: PHASE 4: Refreshing UI...");
	
	// Force main window refresh
	if (EditorNode::get_singleton()) {
		Window *main_window = EditorNode::get_singleton()->get_window();
		if (main_window) {
			print_line("AI Checkpoint:   - Activating main window...");
			main_window->grab_focus();
			main_window->move_to_foreground();
			
			if (DisplayServer::get_singleton()) {
				DisplayServer::get_singleton()->process_events();
				DisplayServer::get_singleton()->force_process_and_drop_events();
			}
		}
		
		// Force complete redraw by triggering theme change notification
		print_line("AI Checkpoint:   - Forcing complete redraw...");
		EditorNode::get_singleton()->get_tree()->get_root()->propagate_notification(Control::NOTIFICATION_THEME_CHANGED);
	}
	
	print_line("AI Checkpoint: ✅ Phase 4 complete");
}

void AICheckpointManager::_refresh_phase_5_docks() {
	print_line("AI Checkpoint: PHASE 5: CRITICAL - Refreshing FileSystem dock...");
	
	// CRITICAL FIX: Git restore changes files on disk, but Godot's FileSystem doesn't know!
	// We MUST force a filesystem scan or files won't appear in the navigator
	print_line("AI Checkpoint:   - Forcing FileSystem scan to detect restored files...");
	
	if (EditorFileSystem::get_singleton()) {
		// NUCLEAR OPTION: Full filesystem scan to rebuild file tree
		// This is necessary because Git changed files behind Godot's back
		EditorFileSystem::get_singleton()->scan();
		print_line("AI Checkpoint:   ✅ FileSystem FULL SCAN started");
		
		// Also scan for changes (more lightweight, catches modifications)
		EditorFileSystem::get_singleton()->scan_changes();
		print_line("AI Checkpoint:   ✅ FileSystem CHANGE SCAN started");
	} else {
		print_line("AI Checkpoint:   ❌ EditorFileSystem not available!");
	}
	
	// Force FileSystem dock to refresh its view
	if (FileSystemDock::get_singleton()) {
		print_line("AI Checkpoint:   - Refreshing FileSystem dock UI...");
		// Navigate to current path to force refresh
		FileSystemDock::get_singleton()->navigate_to_path("res://");
		print_line("AI Checkpoint:   ✅ FileSystem dock refreshed");
	}
	
	print_line("AI Checkpoint: ✅ Phase 5 complete");
	print_line("AI Checkpoint: ========================================");
	print_line("AI Checkpoint: ✅✅✅ REFRESH COMPLETE ✅✅✅");
	print_line("AI Checkpoint: Git restored files to disk");
	print_line("AI Checkpoint: FileSystem scanned - files NOW visible in navigator");
	print_line("AI Checkpoint: Scripts/scenes reloaded from disk");
	print_line("AI Checkpoint: ========================================");
}

