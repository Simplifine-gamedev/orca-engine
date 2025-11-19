/**************************************************************************/
/*  git_manager.h                                                         */
/**************************************************************************/

#pragma once

#include "core/os/os.h"
#include "core/config/project_settings.h"
#include "core/io/file_access.h"
#include "core/io/dir_access.h"

class GitManager {
public:
	struct GitStatus {
		bool is_repo = false;
		bool is_clean = true;
		Vector<String> modified_files;
		Vector<String> added_files;
		Vector<String> deleted_files;
		Vector<String> untracked_files;
		String current_branch;
		String error_message;
	};

	struct GitRemote {
		String name;
		String url;
	};

	struct GitResult {
		bool success = false;
		String message;
		String output;
		int exit_code = 0;
	};

	// Core Git operations
	static GitResult initialize_repository(const String &p_project_path);
	static GitStatus get_repository_status(const String &p_project_path);
	static GitResult add_files(const String &p_project_path, const String &p_pattern = ".");
	static GitResult commit_changes(const String &p_project_path, const String &p_message);
	static GitResult push_changes(const String &p_project_path, const String &p_remote = "origin", const String &p_branch = "");
	static GitResult pull_changes(const String &p_project_path, const String &p_remote = "origin", const String &p_branch = "");
	
	// Remote management
	static Vector<GitRemote> get_remotes(const String &p_project_path);
	static GitResult add_remote(const String &p_project_path, const String &p_name, const String &p_url);
	static GitResult remove_remote(const String &p_project_path, const String &p_name);
	
	// Branch management
	static Vector<String> get_branches(const String &p_project_path);
	static String get_current_branch(const String &p_project_path);
	static GitResult create_branch(const String &p_project_path, const String &p_branch_name);
	static GitResult checkout_branch(const String &p_project_path, const String &p_branch_name);
	
	// Utility functions
	static bool is_git_available();
	static bool is_git_repository(const String &p_project_path);
	static void create_gitignore(const String &p_project_path);
	static void create_gitattributes(const String &p_project_path);

	// Execute Git command with proper error handling (public for GitCredentials)
	static GitResult execute_git_command(const String &p_project_path, const List<String> &p_args);

private:
	static GitResult _execute_command(const String &p_executable, const List<String> &p_args, const String &p_working_dir = "");
	static String _get_git_executable();
};
