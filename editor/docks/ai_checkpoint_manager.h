

/*
 * © 2025 Simplifine Corp.
 * Personal Non‑Commercial License applies to this file as an original contribution to this Godot fork.
 * See LICENSES/COMPANY-NONCOMMERCIAL.md for terms. Commercial use requires a separate license from Simplifine.
 */

#pragma once

#include "core/string/ustring.h"
#include "core/templates/vector.h"
#include "core/templates/list.h"

class EditorNode;
class ScriptEditor;
class FileSystemDock;

/**
 * AICheckpointManager - Handles comprehensive project snapshots for AI Chat restore functionality
 * 
 * This class manages Git-based checkpoints that capture EVERYTHING in a Godot project:
 * - All scene files (.tscn)
 * - All script files (.gd, .cs, .shader, etc.)
 * - All resource files (.tres, .res)
 * - All import files (.import) - critical for proper asset restoration
 * - Project settings (project.godot)
 * - All assets (textures, models, audio, etc.)
 * 
 * When a checkpoint is restored, EVERYTHING is restored to its exact state,
 * and the editor UI is completely refreshed to reflect the changes.
 */
class AICheckpointManager {
public:
	struct CheckpointResult {
		bool success = false;
		String message;
		String checkpoint_tag;
		int files_captured = 0;
	};

	struct RestoreResult {
		bool success = false;
		String message;
		int files_restored = 0;
		Vector<String> restored_scripts;
		String restored_scene_path;
	};

	/**
	 * Create a comprehensive checkpoint of the entire project
	 * @param p_project_root Absolute path to project root
	 * @param p_message User message to include in commit
	 * @param p_message_index Message index for checkpoint naming
	 * @return CheckpointResult with success status and details
	 */
	static CheckpointResult create_comprehensive_checkpoint(const String &p_project_root, const String &p_message, int p_message_index);

	/**
	 * Get or create the isolated checkpoint directory path
	 * Returns path to .ai-checkpoints directory with its own git repo
	 * @param p_project_root Project root directory
	 * @return Path to checkpoint storage directory
	 */
	static String _get_checkpoint_directory(const String &p_project_root);

	/**
	 * Restore project to a specific checkpoint
	 * @param p_project_root Absolute path to project root
	 * @param p_message_index Message index to restore to
	 * @return RestoreResult with success status and details
	 */
	static RestoreResult restore_to_checkpoint(const String &p_project_root, int p_message_index);

	/**
	 * Refresh ALL editor components after checkpoint restoration
	 * This is a 5-phase process that ensures EVERYTHING is reloaded
	 * @param p_restored_scene_path Path to scene that should be reopened (if any)
	 * @param p_restored_scripts Paths to scripts that should be reopened
	 */
	static void refresh_editor_completely(const String &p_restored_scene_path, const Vector<String> &p_restored_scripts);

private:
    // Run a git command in the specified project root using `git -C <dir> ...`.
    // Returns true on success (exitcode == 0). Captures stdout/stderr into output.
    static bool _git_exec(const String &p_project_root, const List<String> &p_args, String &r_output, int &r_exitcode);

	/**
	 * Initialize Git repository in checkpoint directory (separate from project)
	 */
	static bool _init_checkpoint_git_repo(const String &p_checkpoint_dir);

	/**
	 * Copy all project files to checkpoint directory
	 */
	static bool _copy_project_to_checkpoint(const String &p_project_root, const String &p_checkpoint_dir);

	/**
	 * Restore all project files from checkpoint directory
	 */
	static bool _restore_project_from_checkpoint(const String &p_checkpoint_dir, const String &p_project_root);

	/**
	 * Create project .gitignore to ignore .ai-checkpoints directory
	 */
	static void _ensure_project_gitignore_excludes_checkpoints(const String &p_project_root);

	/**
	 * Recursively copy directory contents
	 */
	static bool _copy_directory_recursive(const String &p_source, const String &p_destination, bool p_is_to_checkpoint);

	/**
	 * Recursively remove directory contents (with preservation options)
	 */
	static bool _remove_directory_contents_recursive(const String &p_directory, bool p_preserve_git_only);

	/**
	 * Create minimal .gitignore in checkpoint directory
	 */
	static void _create_checkpoint_gitignore(const String &p_checkpoint_dir);

	/**
	 * Ensure .gitignore is committed to checkpoint repository
	 */
	static void _ensure_gitignore_committed(const String &p_checkpoint_dir);

	/**
	 * Stage ALL files for commit in checkpoint directory
	 */
	static bool _stage_all_files(const String &p_checkpoint_dir);

	/**
	 * Create Git commit with all staged files in checkpoint directory
	 */
	static bool _create_commit(const String &p_checkpoint_dir, const String &p_message, int p_message_index);

	/**
	 * Create Git tag for easy checkpoint reference in checkpoint directory
	 */
	static bool _create_checkpoint_tag(const String &p_checkpoint_dir, int p_message_index);

	/**
	 * Find checkpoint tag for given message index in checkpoint directory
	 */
	static String _find_checkpoint_tag(const String &p_checkpoint_dir, int p_message_index);

	/**
	 * Perform Git hard reset to checkpoint tag in checkpoint directory
	 */
	static bool _git_reset_to_tag(const String &p_checkpoint_dir, const String &p_tag);

	/**
	 * Clear all editor state before restoration (prevents crashes)
	 */
	static void _clear_editor_state();

	/**
	 * Get list of currently open scripts (before clearing)
	 */
	static Vector<String> _get_open_scripts();

	/**
	 * Get currently edited scene path (before clearing)
	 */
	static String _get_current_scene_path();

	/**
	 * Generate checkpoint name from message index and timestamp
	 */
	static String _generate_checkpoint_name(int p_message_index);

	// Refresh phase implementations
	static void _refresh_phase_1_resources();
	static void _refresh_phase_2_scenes();
	static void _refresh_phase_3_scripts(const Vector<String> &p_script_paths);
	static void _refresh_phase_4_ui();
	static void _refresh_phase_5_docks();
};

