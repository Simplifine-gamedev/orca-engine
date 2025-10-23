/**************************************************************************/
/*  git_gui_dialog.cpp                                                    */
/**************************************************************************/

#include "git_gui_dialog.h"
#include "scene/gui/separator.h"
#include "editor/editor_node.h"

void GitGuiDialog::_bind_methods() {
	// No binds needed for internal dialog
}

void GitGuiDialog::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_READY: {
			_setup_ui();
		} break;
	}
}

GitGuiDialog::GitGuiDialog() {
	set_title("Git Repository Manager");
	set_min_size(Size2(700, 500));
}

GitGuiDialog::~GitGuiDialog() {
}

void GitGuiDialog::set_project_path(const String &p_path) {
	project_path = p_path;
	refresh_all();
}

void GitGuiDialog::_setup_ui() {
	// Main tab container
	main_tabs = memnew(TabContainer);
	main_tabs->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	main_tabs->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	add_child(main_tabs);
	
	_create_status_tab();
	_create_actions_tab();
	_create_remote_tab();
	_create_github_tab();
}

void GitGuiDialog::_create_status_tab() {
	status_tab = memnew(VBoxContainer);
	status_tab->set_name("Status");
	main_tabs->add_child(status_tab);
	
	// Repository info
	repo_info_label = memnew(Label);
	status_tab->add_child(repo_info_label);
	
	branch_info_label = memnew(Label);
	status_tab->add_child(branch_info_label);
	
	status_tab->add_child(memnew(HSeparator));
	
	// Status display
	status_display = memnew(RichTextLabel);
	status_display->set_use_bbcode(true);
	status_display->set_selection_enabled(true);
	status_display->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	status_tab->add_child(status_display);
	
	// Refresh button
	refresh_status_button = memnew(Button);
	refresh_status_button->set_text("Refresh Status");
	refresh_status_button->connect("pressed", callable_mp(this, &GitGuiDialog::_on_refresh_pressed));
	status_tab->add_child(refresh_status_button);
}

void GitGuiDialog::_create_actions_tab() {
	actions_tab = memnew(VBoxContainer);
	actions_tab->set_name("Actions");
	main_tabs->add_child(actions_tab);
	
	// Stage files section
	Label *stage_label = memnew(Label);
	stage_label->set_text("Stage Files:");
	actions_tab->add_child(stage_label);
	
	stage_container = memnew(HBoxContainer);
	actions_tab->add_child(stage_container);
	
	add_all_button = memnew(Button);
	add_all_button->set_text("Add All Files");
	add_all_button->set_tooltip_text("git add .");
	add_all_button->connect("pressed", callable_mp(this, &GitGuiDialog::_on_add_all_pressed));
	stage_container->add_child(add_all_button);
	
	add_current_button = memnew(Button);
	add_current_button->set_text("Add Current Scene");
	add_current_button->set_tooltip_text("Add currently edited scene");
	add_current_button->connect("pressed", callable_mp(this, &GitGuiDialog::_on_add_current_pressed));
	stage_container->add_child(add_current_button);
	
	actions_tab->add_child(memnew(HSeparator));
	
	// Commit section
	Label *commit_label = memnew(Label);
	commit_label->set_text("Commit Changes:");
	actions_tab->add_child(commit_label);
	
	commit_container = memnew(VBoxContainer);
	actions_tab->add_child(commit_container);
	
	commit_message_input = memnew(LineEdit);
	commit_message_input->set_placeholder("Enter commit message");
	commit_message_input->connect("text_changed", callable_mp(this, &GitGuiDialog::_on_commit_message_changed));
	commit_container->add_child(commit_message_input);
	
	commit_button = memnew(Button);
	commit_button->set_text("Commit Changes");
	commit_button->set_disabled(true);
	commit_button->connect("pressed", callable_mp(this, &GitGuiDialog::_on_commit_pressed));
	commit_container->add_child(commit_button);
	
	actions_tab->add_child(memnew(HSeparator));
	
	// Push/Pull section
	Label *sync_label = memnew(Label);
	sync_label->set_text("Sync with Remote:");
	actions_tab->add_child(sync_label);
	
	push_pull_container = memnew(HBoxContainer);
	actions_tab->add_child(push_pull_container);
	
	pull_button = memnew(Button);
	pull_button->set_text("Pull from Remote");
	pull_button->set_tooltip_text("git pull origin");
	pull_button->connect("pressed", callable_mp(this, &GitGuiDialog::_on_pull_pressed));
	push_pull_container->add_child(pull_button);
	
	push_button = memnew(Button);
	push_button->set_text("Push to Remote");
	push_button->set_tooltip_text("git push origin");
	push_button->connect("pressed", callable_mp(this, &GitGuiDialog::_on_push_pressed));
	push_pull_container->add_child(push_button);
}

void GitGuiDialog::_create_remote_tab() {
	remote_tab = memnew(VBoxContainer);
	remote_tab->set_name("Remotes");
	main_tabs->add_child(remote_tab);
	
	Label *remotes_label = memnew(Label);
	remotes_label->set_text("Current Remotes:");
	remote_tab->add_child(remotes_label);
	
	remotes_list_container = memnew(VBoxContainer);
	remote_tab->add_child(remotes_list_container);
	
	remote_tab->add_child(memnew(HSeparator));
	
	// Add new remote section
	Label *add_remote_label = memnew(Label);
	add_remote_label->set_text("Add New Remote:");
	remote_tab->add_child(add_remote_label);
	
	add_remote_container = memnew(VBoxContainer);
	remote_tab->add_child(add_remote_container);
	
	remote_name_input = memnew(LineEdit);
	remote_name_input->set_placeholder("Remote name (e.g., origin)");
	add_remote_container->add_child(remote_name_input);
	
	remote_url_input = memnew(LineEdit);
	remote_url_input->set_placeholder("Remote URL (e.g., https://github.com/username/repo.git)");
	add_remote_container->add_child(remote_url_input);
	
	add_remote_button = memnew(Button);
	add_remote_button->set_text("Add Remote");
	add_remote_button->connect("pressed", callable_mp(this, &GitGuiDialog::_on_add_remote_pressed));
	add_remote_container->add_child(add_remote_button);
}

void GitGuiDialog::_create_github_tab() {
	github_tab = memnew(VBoxContainer);
	github_tab->set_name("GitHub");
	main_tabs->add_child(github_tab);
	
	Label *creds_label = memnew(Label);
	creds_label->set_text("GitHub Credentials:");
	github_tab->add_child(creds_label);
	
	credentials_container = memnew(VBoxContainer);
	github_tab->add_child(credentials_container);
	
	username_input = memnew(LineEdit);
	username_input->set_placeholder("GitHub username");
	credentials_container->add_child(username_input);
	
	email_input = memnew(LineEdit);
	email_input->set_placeholder("Email address");
	credentials_container->add_child(email_input);
	
	github_token_input = memnew(LineEdit);
	github_token_input->set_placeholder("GitHub Personal Access Token (ghp_...)");
	github_token_input->set_secret(true);
	credentials_container->add_child(github_token_input);
	
	save_credentials_check = memnew(CheckButton);
	save_credentials_check->set_text("Save credentials securely");
	save_credentials_check->set_pressed(true);
	credentials_container->add_child(save_credentials_check);
	
	save_credentials_button = memnew(Button);
	save_credentials_button->set_text("Save Credentials");
	save_credentials_button->connect("pressed", callable_mp(this, &GitGuiDialog::_on_save_credentials_pressed));
	credentials_container->add_child(save_credentials_button);
	
	credentials_status_label = memnew(Label);
	credentials_status_label->set_modulate(Color(0.8, 0.8, 0.8));
	credentials_container->add_child(credentials_status_label);
	
	github_tab->add_child(memnew(HSeparator));
	
	// GitHub repository setup
	Label *github_repo_label = memnew(Label);
	github_repo_label->set_text("Quick GitHub Setup:");
	github_tab->add_child(github_repo_label);
	
	github_actions_container = memnew(VBoxContainer);
	github_tab->add_child(github_actions_container);
	
	github_repo_input = memnew(LineEdit);
	github_repo_input->set_placeholder("GitHub repository (username/repo-name)");
	github_actions_container->add_child(github_repo_input);
	
	add_github_remote_button = memnew(Button);
	add_github_remote_button->set_text("Add as Origin Remote");
	add_github_remote_button->connect("pressed", callable_mp(this, &GitGuiDialog::_on_add_github_remote_pressed));
	github_actions_container->add_child(add_github_remote_button);
}

void GitGuiDialog::refresh_all() {
	if (project_path.is_empty()) {
		return;
	}
	
	_refresh_status();
	_refresh_remotes();
	_load_credentials();
	_update_ui_state();
}

void GitGuiDialog::_refresh_status() {
	if (!repo_info_label || project_path.is_empty()) {
		return;
	}
	
	GitManager::GitStatus status = GitManager::get_repository_status(project_path);
	
	repo_info_label->set_text("Repository: " + project_path);
	
	if (!status.is_repo) {
		branch_info_label->set_text("Not a Git repository");
		status_display->clear();
		status_display->append_text("[color=red]This directory is not a Git repository.[/color]\n");
		status_display->append_text("Would you like to initialize Git for this project?\n\n");
		
		// Add initialization button if not a repo and not already added
		if (!initialize_git_button) {
			initialize_git_button = memnew(Button);
			initialize_git_button->set_text("Initialize Git Repository");
			initialize_git_button->connect("pressed", callable_mp(this, &GitGuiDialog::_on_initialize_git_pressed));
			status_tab->add_child(initialize_git_button);
		}
		initialize_git_button->set_visible(true);
		return;
	}
	
	branch_info_label->set_text("Branch: " + status.current_branch);
	
	// Hide initialization button if repo exists
	if (initialize_git_button) {
		initialize_git_button->set_visible(false);
	}
	
	status_display->clear();
	
	if (status.is_clean) {
		status_display->append_text("[color=green][b]Repository is clean[/b][/color]\n");
		status_display->append_text("No uncommitted changes detected.");
	} else {
		status_display->append_text("[color=orange][b]Uncommitted changes detected:[/b][/color]\n\n");
		
		if (!status.modified_files.is_empty()) {
			status_display->append_text("[b]Modified files:[/b]\n");
			for (const String &file : status.modified_files) {
				status_display->append_text("[color=yellow]M [/color]" + file + "\n");
			}
			status_display->append_text("\n");
		}
		
		if (!status.added_files.is_empty()) {
			status_display->append_text("[b]Added files:[/b]\n");
			for (const String &file : status.added_files) {
				status_display->append_text("[color=green]A [/color]" + file + "\n");
			}
			status_display->append_text("\n");
		}
		
		if (!status.deleted_files.is_empty()) {
			status_display->append_text("[b]Deleted files:[/b]\n");
			for (const String &file : status.deleted_files) {
				status_display->append_text("[color=red]D [/color]" + file + "\n");
			}
			status_display->append_text("\n");
		}
		
		if (!status.untracked_files.is_empty()) {
			status_display->append_text("[b]Untracked files:[/b]\n");
			for (const String &file : status.untracked_files) {
				status_display->append_text("[color=lightblue]? [/color]" + file + "\n");
			}
		}
	}
}

void GitGuiDialog::_refresh_remotes() {
	if (!remotes_list_container || project_path.is_empty()) {
		return;
	}
	
	// Clear existing remote display
	for (int i = remotes_list_container->get_child_count() - 1; i >= 0; i--) {
		remotes_list_container->get_child(i)->queue_free();
	}
	
	Vector<GitManager::GitRemote> remotes = GitManager::get_remotes(project_path);
	
	if (remotes.is_empty()) {
		Label *no_remotes = memnew(Label);
		no_remotes->set_text("No remote repositories configured");
		no_remotes->set_modulate(Color(0.7, 0.7, 0.7));
		remotes_list_container->add_child(no_remotes);
	} else {
		for (const GitManager::GitRemote &remote : remotes) {
			HBoxContainer *remote_container = memnew(HBoxContainer);
			remotes_list_container->add_child(remote_container);
			
			Label *remote_label = memnew(Label);
			remote_label->set_text(remote.name + ": " + remote.url);
			remote_label->set_h_size_flags(Control::SIZE_EXPAND_FILL);
			remote_container->add_child(remote_label);
			
			Button *remove_button = memnew(Button);
			remove_button->set_text("Remove");
			remove_button->connect("pressed", callable_mp(this, &GitGuiDialog::_on_remove_remote_pressed).bind(remote.name));
			remote_container->add_child(remove_button);
		}
	}
}

void GitGuiDialog::_load_credentials() {
	GitCredentials::Credentials creds = GitCredentials::load_credentials();
	
	if (username_input) username_input->set_text(creds.username);
	if (email_input) email_input->set_text(creds.email);
	if (github_token_input) github_token_input->set_text(creds.github_token);
	if (save_credentials_check) save_credentials_check->set_pressed(creds.save_credentials);
	
	if (credentials_status_label) {
		if (GitCredentials::has_valid_credentials()) {
			credentials_status_label->set_text("Valid credentials loaded");
			credentials_status_label->set_modulate(Color(0.4, 0.8, 0.4));
		} else {
			credentials_status_label->set_text("No valid credentials found");
			credentials_status_label->set_modulate(Color(0.8, 0.8, 0.4));
		}
	}
}

void GitGuiDialog::_on_refresh_pressed() {
	_refresh_status();
	_refresh_remotes();
}

void GitGuiDialog::_on_add_all_pressed() {
	GitManager::GitResult result = GitManager::add_files(project_path, ".");
	_show_operation_result(result, "Add All Files");
	_refresh_status();
}

void GitGuiDialog::_on_add_current_pressed() {
	// Add currently edited scene/script
	_show_operation_result({true, "Feature coming soon", "", 0}, "Add Current File");
}

void GitGuiDialog::_on_commit_pressed() {
	String message = commit_message_input->get_text().strip_edges();
	if (message.is_empty()) {
		_show_operation_result({false, "Commit message cannot be empty", "", 1}, "Commit");
		return;
	}
	
	GitManager::GitResult result = GitManager::commit_changes(project_path, message);
	_show_operation_result(result, "Commit");
	
	if (result.success) {
		commit_message_input->clear();
		_refresh_status();
	}
}

void GitGuiDialog::_on_push_pressed() {
	GitManager::GitResult result = GitManager::push_changes(project_path);
	_show_operation_result(result, "Push");
}

void GitGuiDialog::_on_pull_pressed() {
	GitManager::GitResult result = GitManager::pull_changes(project_path);
	_show_operation_result(result, "Pull");
	_refresh_status();
}

void GitGuiDialog::_on_add_remote_pressed() {
	String name = remote_name_input->get_text().strip_edges();
	String url = remote_url_input->get_text().strip_edges();
	
	if (name.is_empty() || url.is_empty()) {
		_show_operation_result({false, "Remote name and URL are required", "", 1}, "Add Remote");
		return;
	}
	
	GitManager::GitResult result = GitManager::add_remote(project_path, name, url);
	_show_operation_result(result, "Add Remote");
	
	if (result.success) {
		remote_name_input->clear();
		remote_url_input->clear();
		_refresh_remotes();
	}
}

void GitGuiDialog::_on_remove_remote_pressed(const String &p_remote_name) {
	GitManager::GitResult result = GitManager::remove_remote(project_path, p_remote_name);
	_show_operation_result(result, "Remove Remote");
	_refresh_remotes();
}

void GitGuiDialog::_on_save_credentials_pressed() {
	GitCredentials::Credentials creds;
	creds.username = username_input->get_text().strip_edges();
	creds.email = email_input->get_text().strip_edges();
	creds.github_token = github_token_input->get_text().strip_edges();
	creds.save_credentials = save_credentials_check->is_pressed();
	
	// Validate GitHub token
	if (!creds.github_token.is_empty() && !GitCredentials::validate_github_token(creds.github_token)) {
		_show_operation_result({false, "Invalid GitHub token format. Token should start with 'ghp_', 'github_pat_', etc.", "", 1}, "Save Credentials");
		return;
	}
	
	GitCredentials::save_credentials(creds);
	GitCredentials::configure_git_credentials(project_path, creds);
	
	_show_operation_result({true, "Credentials saved successfully", "", 0}, "Save Credentials");
	_load_credentials();
}

void GitGuiDialog::_on_add_github_remote_pressed() {
	String repo = github_repo_input->get_text().strip_edges();
	if (repo.is_empty()) {
		_show_operation_result({false, "GitHub repository format required (username/repo-name)", "", 1}, "Add GitHub Remote");
		return;
	}
	
	// Create GitHub URL
	String github_url = GitCredentials::create_github_https_url("", repo);
	
	GitManager::GitResult result = GitManager::add_remote(project_path, "origin", github_url);
	_show_operation_result(result, "Add GitHub Remote");
	
	if (result.success) {
		github_repo_input->clear();
		_refresh_remotes();
	}
}

void GitGuiDialog::_on_commit_message_changed(const String &p_text) {
	if (commit_button) {
		commit_button->set_disabled(p_text.strip_edges().is_empty());
	}
}

void GitGuiDialog::_update_ui_state() {
	bool is_repo = GitManager::is_git_repository(project_path);
	
	// Enable/disable buttons based on repository state
	if (add_all_button) add_all_button->set_disabled(!is_repo);
	if (add_current_button) add_current_button->set_disabled(!is_repo);
	if (commit_button) commit_button->set_disabled(!is_repo || commit_message_input->get_text().strip_edges().is_empty());
	if (push_button) push_button->set_disabled(!is_repo);
	if (pull_button) pull_button->set_disabled(!is_repo);
}

void GitGuiDialog::_on_initialize_git_pressed() {
	GitManager::GitResult result = GitManager::initialize_repository(project_path);
	_show_operation_result(result, "Initialize Git");
	
	if (result.success) {
		refresh_all();
	}
}

void GitGuiDialog::_show_operation_result(const GitManager::GitResult &p_result, const String &p_operation) {
	AcceptDialog *result_dialog = memnew(AcceptDialog);
	result_dialog->set_title(p_operation + " Result");
	
	String message = p_operation + ": ";
	if (p_result.success) {
		message += "SUCCESS\n\n" + p_result.message;
		if (!p_result.output.is_empty()) {
			message += "\n\nOutput:\n" + p_result.output;
		}
	} else {
		message += "FAILED\n\n" + p_result.message;
		if (!p_result.output.is_empty()) {
			message += "\n\nError details:\n" + p_result.output;
		}
	}
	
	result_dialog->set_text(message);
	get_parent()->add_child(result_dialog);
	result_dialog->popup_centered();
}
