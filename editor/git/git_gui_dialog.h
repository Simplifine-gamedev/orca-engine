/**************************************************************************/
/*  git_gui_dialog.h                                                      */
/**************************************************************************/

#pragma once

#include "git_manager.h"
#include "git_credentials.h"
#include "scene/gui/dialogs.h"
#include "scene/gui/box_container.h"
#include "scene/gui/button.h"
#include "scene/gui/label.h"
#include "scene/gui/line_edit.h"
#include "scene/gui/text_edit.h"
#include "scene/gui/rich_text_label.h"
#include "scene/gui/option_button.h"
#include "scene/gui/check_button.h"
#include "scene/gui/tab_container.h"
#include "scene/gui/separator.h"

class GitGuiDialog : public AcceptDialog {
	GDCLASS(GitGuiDialog, AcceptDialog);

private:
	String project_path;
	TabContainer *main_tabs = nullptr;
	
	// Status Tab
	VBoxContainer *status_tab = nullptr;
	Label *repo_info_label = nullptr;
	Label *branch_info_label = nullptr;
	RichTextLabel *status_display = nullptr;
	Button *refresh_status_button = nullptr;
	Button *initialize_git_button = nullptr;
	
	// Actions Tab
	VBoxContainer *actions_tab = nullptr;
	HBoxContainer *stage_container = nullptr;
	Button *add_all_button = nullptr;
	Button *add_current_button = nullptr;
	VBoxContainer *commit_container = nullptr;
	LineEdit *commit_message_input = nullptr;
	Button *commit_button = nullptr;
	HSeparator *push_pull_separator = nullptr;
	HBoxContainer *push_pull_container = nullptr;
	Button *push_button = nullptr;
	Button *pull_button = nullptr;
	
	// Remote Tab
	VBoxContainer *remote_tab = nullptr;
	VBoxContainer *remotes_list_container = nullptr;
	HSeparator *add_remote_separator = nullptr;
	VBoxContainer *add_remote_container = nullptr;
	LineEdit *remote_name_input = nullptr;
	LineEdit *remote_url_input = nullptr;
	Button *add_remote_button = nullptr;
	
	// GitHub Tab
	VBoxContainer *github_tab = nullptr;
	VBoxContainer *credentials_container = nullptr;
	LineEdit *username_input = nullptr;
	LineEdit *email_input = nullptr;
	LineEdit *github_token_input = nullptr;
	CheckButton *save_credentials_check = nullptr;
	Button *save_credentials_button = nullptr;
	Label *credentials_status_label = nullptr;
	HSeparator *github_actions_separator = nullptr;
	VBoxContainer *github_actions_container = nullptr;
	LineEdit *github_repo_input = nullptr;
	Button *add_github_remote_button = nullptr;
	
	void _setup_ui();
	void _create_status_tab();
	void _create_actions_tab();
	void _create_remote_tab();
	void _create_github_tab();
	
	void _refresh_status();
	void _refresh_remotes();
	void _load_credentials();
	
	// Event handlers
	void _on_refresh_pressed();
	void _on_add_all_pressed();
	void _on_add_current_pressed();
	void _on_commit_pressed();
	void _on_push_pressed();
	void _on_pull_pressed();
	void _on_add_remote_pressed();
	void _on_remove_remote_pressed(const String &p_remote_name);
	void _on_save_credentials_pressed();
	void _on_add_github_remote_pressed();
	void _on_initialize_git_pressed();
	void _on_commit_message_changed(const String &p_text);
	
	void _update_ui_state();
	void _show_operation_result(const GitManager::GitResult &p_result, const String &p_operation);

protected:
	static void _bind_methods();
	void _notification(int p_what);

public:
	void set_project_path(const String &p_path);
	void refresh_all();
	
	GitGuiDialog();
	~GitGuiDialog();
};
