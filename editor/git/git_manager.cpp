/**************************************************************************/
/*  git_manager.cpp                                                       */
/**************************************************************************/

#include "git_manager.h"

GitManager::GitResult GitManager::initialize_repository(const String &p_project_path) {
	GitResult result;
	
	// Check if already a Git repository
	if (is_git_repository(p_project_path)) {
		result.success = false;
		result.message = "Git repository already exists";
		return result;
	}
	
	// Initialize Git repository
	List<String> init_args;
	init_args.push_back("init");
	result = execute_git_command(p_project_path, init_args);
	
	if (!result.success) {
		result.message = "Failed to initialize Git repository. Make sure Git is installed.";
		return result;
	}
	
	// Create .gitignore and .gitattributes
	create_gitignore(p_project_path);
	create_gitattributes(p_project_path);
	
	// Add all files
	GitResult add_result = add_files(p_project_path, ".");
	if (!add_result.success) {
		result.message = "Git initialized but failed to add files: " + add_result.message;
		return result;
	}
	
	// Create initial commit
	GitResult commit_result = commit_changes(p_project_path, "Initial commit - Orca Engine project");
	if (!commit_result.success) {
		result.message = "Git initialized but failed to create initial commit: " + commit_result.message;
		return result;
	}
	
	result.success = true;
	result.message = "Git repository initialized successfully";
	return result;
}

GitManager::GitStatus GitManager::get_repository_status(const String &p_project_path) {
	GitStatus status;
	
	// Check if Git repo exists
	if (!is_git_repository(p_project_path)) {
		status.is_repo = false;
		status.error_message = "Not a Git repository";
		return status;
	}
	
	status.is_repo = true;
	
	// Get current branch
	status.current_branch = get_current_branch(p_project_path);
	
	// Get status with porcelain format
	List<String> status_args;
	status_args.push_back("status");
	status_args.push_back("--porcelain");
	
	GitResult result = execute_git_command(p_project_path, status_args);
	if (!result.success) {
		status.error_message = "Failed to get Git status: " + result.message;
		return status;
	}
	
	// Debug: Print the raw Git output
	print_line("GIT DEBUG: Raw output: '" + result.output + "'");
	print_line("GIT DEBUG: Output length: " + String::num_int64(result.output.length()));
	
	// Parse status output
	Vector<String> lines = result.output.split("\n");
	print_line("GIT DEBUG: Number of lines: " + String::num_int64(lines.size()));
	
	// Check if repository is clean
	bool has_changes = false;
	for (const String &line : lines) {
		if (!line.strip_edges().is_empty()) {
			has_changes = true;
			break;
		}
	}
	status.is_clean = !has_changes;
	
	for (int i = 0; i < lines.size(); i++) {
		String line = lines[i];
		print_line("GIT DEBUG: Line " + String::num_int64(i) + ": '" + line + "'");
		
		if (line.strip_edges().is_empty()) continue;
		
		if (line.length() < 3) {
			print_line("GIT DEBUG: Line too short, skipping");
			continue;
		}
		
		String status_chars = line.left(2);
		String file_path = line.substr(3);
		
		print_line("GIT DEBUG: Status chars: '" + status_chars + "', File: '" + file_path + "'");
		
		if (status_chars.begins_with("M") || status_chars.contains("M")) {
			status.modified_files.push_back(file_path);
			print_line("GIT DEBUG: Added to modified: " + file_path);
		} else if (status_chars.begins_with("A") || status_chars.contains("A")) {
			status.added_files.push_back(file_path);
			print_line("GIT DEBUG: Added to added: " + file_path);
		} else if (status_chars.begins_with("D") || status_chars.contains("D")) {
			status.deleted_files.push_back(file_path);
			print_line("GIT DEBUG: Added to deleted: " + file_path);
		} else if (status_chars.begins_with("??")) {
			status.untracked_files.push_back(file_path);
			print_line("GIT DEBUG: Added to untracked: " + file_path);
		} else {
			print_line("GIT DEBUG: Unknown status: '" + status_chars + "'");
			// Add to untracked as fallback
			status.untracked_files.push_back(file_path);
		}
	}
	
	print_line("GIT DEBUG: Final status - Clean: " + String(status.is_clean ? "true" : "false"));
	print_line("GIT DEBUG: Modified: " + String::num_int64(status.modified_files.size()));
	print_line("GIT DEBUG: Added: " + String::num_int64(status.added_files.size()));
	print_line("GIT DEBUG: Deleted: " + String::num_int64(status.deleted_files.size()));
	print_line("GIT DEBUG: Untracked: " + String::num_int64(status.untracked_files.size()));
	
	return status;
}

GitManager::GitResult GitManager::add_files(const String &p_project_path, const String &p_pattern) {
	List<String> add_args;
	add_args.push_back("add");
	add_args.push_back(p_pattern);
	
	GitResult result = execute_git_command(p_project_path, add_args);
	if (result.success) {
		result.message = "Files staged successfully";
	}
	return result;
}

GitManager::GitResult GitManager::commit_changes(const String &p_project_path, const String &p_message) {
	List<String> commit_args;
	commit_args.push_back("commit");
	commit_args.push_back("-m");
	commit_args.push_back(p_message);
	
	GitResult result = execute_git_command(p_project_path, commit_args);
	if (result.success) {
		result.message = "Changes committed successfully";
	}
	return result;
}

GitManager::GitResult GitManager::push_changes(const String &p_project_path, const String &p_remote, const String &p_branch) {
	List<String> push_args;
	push_args.push_back("push");
	
	if (!p_remote.is_empty()) {
		push_args.push_back(p_remote);
		if (!p_branch.is_empty()) {
			push_args.push_back(p_branch);
		}
	}
	
	GitResult result = execute_git_command(p_project_path, push_args);
	if (result.success) {
		result.message = "Changes pushed successfully";
	}
	return result;
}

GitManager::GitResult GitManager::pull_changes(const String &p_project_path, const String &p_remote, const String &p_branch) {
	List<String> pull_args;
	pull_args.push_back("pull");
	
	if (!p_remote.is_empty()) {
		pull_args.push_back(p_remote);
		if (!p_branch.is_empty()) {
			pull_args.push_back(p_branch);
		}
	}
	
	GitResult result = execute_git_command(p_project_path, pull_args);
	if (result.success) {
		result.message = "Changes pulled successfully";
	}
	return result;
}

Vector<GitManager::GitRemote> GitManager::get_remotes(const String &p_project_path) {
	Vector<GitRemote> remotes;
	
	List<String> remote_args;
	remote_args.push_back("remote");
	remote_args.push_back("-v");
	
	GitResult result = execute_git_command(p_project_path, remote_args);
	if (!result.success) {
		return remotes;
	}
	
	Vector<String> lines = result.output.split("\n");
	for (const String &line : lines) {
		if (line.strip_edges().is_empty()) continue;
		
		Vector<String> parts = line.split("\t");
		if (parts.size() >= 2 && parts[1].ends_with(" (fetch)")) {
			GitRemote remote;
			remote.name = parts[0].strip_edges();
			remote.url = parts[1].replace(" (fetch)", "").strip_edges();
			remotes.push_back(remote);
		}
	}
	
	return remotes;
}

GitManager::GitResult GitManager::add_remote(const String &p_project_path, const String &p_name, const String &p_url) {
	List<String> remote_args;
	remote_args.push_back("remote");
	remote_args.push_back("add");
	remote_args.push_back(p_name);
	remote_args.push_back(p_url);
	
	GitResult result = execute_git_command(p_project_path, remote_args);
	if (result.success) {
		result.message = "Remote '" + p_name + "' added successfully";
	}
	return result;
}

GitManager::GitResult GitManager::remove_remote(const String &p_project_path, const String &p_name) {
	List<String> remote_args;
	remote_args.push_back("remote");
	remote_args.push_back("remove");
	remote_args.push_back(p_name);
	
	GitResult result = execute_git_command(p_project_path, remote_args);
	if (result.success) {
		result.message = "Remote '" + p_name + "' removed successfully";
	}
	return result;
}

Vector<String> GitManager::get_branches(const String &p_project_path) {
	Vector<String> branches;
	
	List<String> branch_args;
	branch_args.push_back("branch");
	
	GitResult result = execute_git_command(p_project_path, branch_args);
	if (!result.success) {
		return branches;
	}
	
	Vector<String> lines = result.output.split("\n");
	for (const String &line : lines) {
		String branch_name = line.strip_edges();
		if (branch_name.begins_with("* ")) {
			branch_name = branch_name.substr(2);
		}
		if (!branch_name.is_empty()) {
			branches.push_back(branch_name);
		}
	}
	
	return branches;
}

String GitManager::get_current_branch(const String &p_project_path) {
	List<String> branch_args;
	branch_args.push_back("rev-parse");
	branch_args.push_back("--abbrev-ref");
	branch_args.push_back("HEAD");
	
	GitResult result = execute_git_command(p_project_path, branch_args);
	if (result.success) {
		return result.output.strip_edges();
	}
	return "";
}

bool GitManager::is_git_available() {
	List<String> version_args;
	version_args.push_back("--version");
	
	String output;
	int exit_code;
	Error err = OS::get_singleton()->execute("git", version_args, &output, &exit_code, true, nullptr, false);
	
	return err == OK && exit_code == 0;
}

bool GitManager::is_git_repository(const String &p_project_path) {
	String git_dir = p_project_path.path_join(".git");
	return DirAccess::exists(git_dir);
}

void GitManager::create_gitignore(const String &p_project_path) {
	String gitignore_path = p_project_path.path_join(".gitignore");
	Ref<FileAccess> file = FileAccess::open(gitignore_path, FileAccess::WRITE);
	if (file.is_valid()) {
		file->store_line("# Godot 4+ specific ignores");
		file->store_line(".godot/");
		file->store_line(".import/");
		file->store_line("");
		file->store_line("# Godot-specific ignores");
		file->store_line("export.cfg");
		file->store_line("export_presets.cfg");
		file->store_line("");
		file->store_line("# Imported translations (automatically generated from CSV files)");
		file->store_line("*.translation");
		file->store_line("");
		file->store_line("# Mono-specific ignores");
		file->store_line(".mono/");
		file->store_line("data_*/");
		file->store_line("mono_crash.*.json");
		file->close();
	}
}

void GitManager::create_gitattributes(const String &p_project_path) {
	String gitattributes_path = p_project_path.path_join(".gitattributes");
	Ref<FileAccess> file = FileAccess::open(gitattributes_path, FileAccess::WRITE);
	if (file.is_valid()) {
		file->store_line("# Normalize EOL for all files that Git considers text files.");
		file->store_line("* text=auto eol=lf");
		file->close();
	}
}

GitManager::GitResult GitManager::execute_git_command(const String &p_project_path, const List<String> &p_args) {
	GitResult result;
	
	// Prepend -C argument to specify working directory
	List<String> final_args;
	final_args.push_back("-C");
	final_args.push_back(p_project_path);
	
	// Add user arguments
	for (const String &arg : p_args) {
		final_args.push_back(arg);
	}
	
	String output;
	int exit_code;
	Error err = OS::get_singleton()->execute("git", final_args, &output, &exit_code, true, nullptr, false);
	
	result.output = output;
	result.exit_code = exit_code;
	result.success = (err == OK && exit_code == 0);
	
	if (!result.success) {
		result.message = "Git command failed (exit code: " + String::num_int64(exit_code) + ")";
		if (!output.is_empty()) {
			result.message += ": " + output;
		}
	}
	
	return result;
}

GitManager::GitResult GitManager::_execute_command(const String &p_executable, const List<String> &p_args, const String &p_working_dir) {
	GitResult result;
	
	String output;
	int exit_code;
	Error err = OS::get_singleton()->execute(p_executable, p_args, &output, &exit_code, true, nullptr, false);
	
	result.output = output;
	result.exit_code = exit_code;
	result.success = (err == OK && exit_code == 0);
	
	if (!result.success) {
		result.message = "Command failed (exit code: " + String::num_int64(exit_code) + ")";
		if (!output.is_empty()) {
			result.message += ": " + output;
		}
	}
	
	return result;
}
