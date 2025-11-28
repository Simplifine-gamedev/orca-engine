/**************************************************************************/
/*  auth_dialog.h                                                         */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             ORCA ENGINE                                */
/**************************************************************************/

#ifndef AUTH_DIALOG_H
#define AUTH_DIALOG_H

#include "scene/gui/dialogs.h"
#include "scene/gui/box_container.h"
#include "scene/gui/button.h"
#include "scene/gui/label.h"
#include "scene/gui/line_edit.h"
#include "scene/gui/texture_rect.h"

class AuthDialog : public AcceptDialog {
	GDCLASS(AuthDialog, AcceptDialog);

private:
	VBoxContainer *main_container = nullptr;
	Label *title_label = nullptr;
	Label *description_label = nullptr;
	Label *status_label = nullptr;
	
	// Google OAuth buttons (existing)
	Button *sign_in_button = nullptr;
	Button *create_account_button = nullptr;
	Button *copy_url_button = nullptr;
	String pending_login_url;
	
	// Email/Password UI (new)
	LineEdit *email_input = nullptr;
	LineEdit *password_input = nullptr;
	LineEdit *name_input = nullptr;
	Button *email_sign_in_button = nullptr;
	Button *email_sign_up_button = nullptr;
	Button *toggle_auth_mode_button = nullptr;
	Button *toggle_signup_mode_button = nullptr;
	
	// Mode control
	bool show_email_mode = false;
	bool show_signup_mode = false;
	
	bool waiting_for_callback = false;
	bool auth_successful = false;
	bool should_load_project_manager = false;

	// Existing handlers
	void _on_sign_in_pressed();
	void _on_create_account_pressed();
	void _on_copy_url_pressed();
	void _reset_copy_button_text();
	void _check_auth_status();
	
	// New email/password handlers
	void _on_email_sign_in_pressed();
	void _on_email_sign_up_pressed();
	void _on_toggle_auth_mode_pressed();
	void _toggle_signup_mode();
	void _set_signup_mode(bool p_signup_mode);
	
	// UI helpers
	void _setup_google_oauth_ui();
	void _setup_email_auth_ui();
	void _show_auth_mode(bool p_email_mode);

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	void show_login();
	void show_waiting();
	void show_success();
	void show_error(const String &p_message);
	void show_info(const String &p_message); // For informational messages (not errors)
	void show_url_fallback(const String &p_url); // For when browser can't open
	
	bool is_waiting_for_callback() const { return waiting_for_callback; }
	bool is_auth_successful() const { return auth_successful; }
	bool should_continue_to_project_manager() const { return should_load_project_manager; }

	AuthDialog();
	~AuthDialog();
};

#endif // AUTH_DIALOG_H

