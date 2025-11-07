/**************************************************************************/
/*  auth_dialog.cpp                                                       */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             ORCA ENGINE                                */
/**************************************************************************/

#include "auth_dialog.h"
#include "auth_manager.h"

#include "editor/themes/editor_scale.h"
#include "scene/gui/margin_container.h"
#include "scene/gui/separator.h"

void AuthDialog::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_on_sign_in_pressed"), &AuthDialog::_on_sign_in_pressed);
	ClassDB::bind_method(D_METHOD("_on_create_account_pressed"), &AuthDialog::_on_create_account_pressed);
	ClassDB::bind_method(D_METHOD("_check_auth_status"), &AuthDialog::_check_auth_status);
	ClassDB::bind_method(D_METHOD("_on_email_sign_in_pressed"), &AuthDialog::_on_email_sign_in_pressed);
	ClassDB::bind_method(D_METHOD("_on_email_sign_up_pressed"), &AuthDialog::_on_email_sign_up_pressed);
	ClassDB::bind_method(D_METHOD("_on_toggle_auth_mode_pressed"), &AuthDialog::_on_toggle_auth_mode_pressed);
}

void AuthDialog::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_READY: {
			if (waiting_for_callback) {
				set_process(true);
			}
		} break;
		
		case NOTIFICATION_PROCESS: {
			if (waiting_for_callback) {
				_check_auth_status();
			}
		} break;
	}
}

AuthDialog::AuthDialog() {
	set_title("Sign in to Orca");
	set_min_size(Size2(500, 400) * EDSCALE);
	set_max_size(Size2(800, 700) * EDSCALE);
	set_size(Size2(550, 450) * EDSCALE);
	set_ok_button_text("Close");
	set_hide_on_ok(false);
	get_ok_button()->set_visible(false);
	set_exclusive(true);
	set_flag(Window::FLAG_POPUP, false);
	// Allow resizing so users can see all content, especially in email mode
	set_flag(Window::FLAG_RESIZE_DISABLED, false);
	
	// Main container
	main_container = memnew(VBoxContainer);
	add_child(main_container);
	
	// Top spacer
	Control *top_spacer = memnew(Control);
	top_spacer->set_custom_minimum_size(Size2(0, 20) * EDSCALE);
	main_container->add_child(top_spacer);
	
	// Title
	title_label = memnew(Label);
	title_label->set_text("Welcome to Orca");
	title_label->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_CENTER);
	title_label->add_theme_font_size_override("font_size", 24 * EDSCALE);
	main_container->add_child(title_label);
	
	Control *spacer1 = memnew(Control);
	spacer1->set_custom_minimum_size(Size2(0, 10) * EDSCALE);
	main_container->add_child(spacer1);
	
	// Description
	description_label = memnew(Label);
	description_label->set_text("Choose your sign in method");
	description_label->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_CENTER);
	description_label->set_autowrap_mode(TextServer::AUTOWRAP_WORD_SMART);
	main_container->add_child(description_label);
	
	Control *spacer2 = memnew(Control);
	spacer2->set_custom_minimum_size(Size2(0, 20) * EDSCALE);
	main_container->add_child(spacer2);
	
	// Status label (for errors/success/waiting)
	status_label = memnew(Label);
	status_label->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_CENTER);
	status_label->set_autowrap_mode(TextServer::AUTOWRAP_WORD_SMART);
	status_label->set_visible(false);
	main_container->add_child(status_label);
	
	Control *spacer3 = memnew(Control);
	spacer3->set_custom_minimum_size(Size2(0, 10) * EDSCALE);
	main_container->add_child(spacer3);
	
	// Setup both UI modes
	_setup_google_oauth_ui();
	_setup_email_auth_ui();
	
	// Register this dialog with AuthManager for callbacks
	AuthManager *auth = AuthManager::get_singleton();
	if (auth) {
		auth->set_auth_dialog(this);
	}
	
	// Start with Google OAuth mode
	_show_auth_mode(false);
}

AuthDialog::~AuthDialog() {
}

void AuthDialog::_on_sign_in_pressed() {
	AuthManager *auth = AuthManager::get_singleton();
	if (auth) {
		auth->open_web_login();
		show_waiting();
	}
}

void AuthDialog::_on_create_account_pressed() {
	// Open browser to signup page
	OS::get_singleton()->shell_open("https://orcaengine.ai/desktop-login.html?tab=email");
	show_waiting();
}

void AuthDialog::_check_auth_status() {
	AuthManager *auth = AuthManager::get_singleton();
	
	if (!auth) {
		return;
	}
	
	if (auth->get_is_authenticated()) {
		show_success();
		waiting_for_callback = false;
		set_process(false);
		should_load_project_manager = true;
		
		// Show the hidden ProjectManager
		if (has_meta("project_manager")) {
			Control *pm = Object::cast_to<Control>(get_meta("project_manager"));
			if (pm) {
				pm->show();
			}
		}
		
		// Close dialog
		call_deferred("hide");
	}
}

void AuthDialog::show_login() {
	waiting_for_callback = false;
	auth_successful = false;
	
	title_label->set_text("Welcome to Orca");
	description_label->set_visible(true);
	status_label->set_visible(false);
	
	// Reset to Google OAuth mode
	_show_auth_mode(false);
	show_signup_mode = false;
	
	// Clear email/password fields
	if (email_input) email_input->set_text("");
	if (password_input) password_input->set_text("");
	if (name_input) name_input->set_text("");
	
	print_line("ORCA AUTH: Showing login dialog as exclusive popup");
	
	// Defer the popup until next frame when dialog is fully in the tree
	call_deferred("popup_centered");
}

void AuthDialog::show_waiting() {
	waiting_for_callback = true;
	
	if (show_email_mode) {
		status_label->set_text("Signing in...\nPlease wait while we authenticate your account.");
	} else {
		status_label->set_text("Waiting for sign in...\nPlease complete the sign in process in your browser.");
	}
	status_label->set_visible(true);
	
	// Hide all buttons during authentication
	sign_in_button->set_visible(false);
	create_account_button->set_visible(false);
	email_sign_in_button->set_visible(false);
	email_sign_up_button->set_visible(false);
	if (toggle_signup_mode_button) {
		toggle_signup_mode_button->set_visible(false);
	}
	
	set_process(true);
}

void AuthDialog::show_success() {
	waiting_for_callback = false;
	auth_successful = true;
	
	AuthManager *auth = AuthManager::get_singleton();
	String user_name = auth ? auth->get_user_name() : "";
	if (user_name.is_empty()) {
		user_name = auth ? auth->get_user_email() : "User";
	}
	
	title_label->set_text("Sign in Successful!");
	description_label->set_visible(false);
	status_label->set_text("Welcome, " + user_name + "!");
	status_label->set_visible(true);
	
	// Hide all buttons and input fields
	sign_in_button->set_visible(false);
	create_account_button->set_visible(false);
	email_sign_in_button->set_visible(false);
	email_sign_up_button->set_visible(false);
	toggle_auth_mode_button->set_visible(false);
	if (toggle_signup_mode_button) {
		toggle_signup_mode_button->set_visible(false);
	}
	
	// Hide email input fields
	Control *email_container = Object::cast_to<Control>(email_input->get_parent()->get_parent());
	if (email_container) {
		email_container->set_visible(false);
	}
	
	get_ok_button()->set_visible(true);
	
	// Close dialog and show project manager after a short delay
	should_load_project_manager = true;
	if (has_meta("project_manager")) {
		Control *pm = Object::cast_to<Control>(get_meta("project_manager"));
		if (pm) {
			pm->show();
		}
	}
	
	// Close dialog after showing success message briefly
	call_deferred("hide");
}

void AuthDialog::show_error(const String &p_message) {
	waiting_for_callback = false;
	
	title_label->set_text("Sign in Failed");
	description_label->set_visible(false);
	status_label->set_text(p_message);
	status_label->set_visible(true);
	
	// Show appropriate buttons based on current mode
	if (show_email_mode) {
		email_sign_in_button->set_visible(true);
		if (show_signup_mode) {
			email_sign_up_button->set_visible(true);
		}
		if (toggle_signup_mode_button) {
			toggle_signup_mode_button->set_visible(true);
		}
	} else {
		sign_in_button->set_visible(true);
		create_account_button->set_visible(true);
	}
}

void AuthDialog::show_info(const String &p_message) {
	waiting_for_callback = false;
	
	title_label->set_text("Account Created");
	description_label->set_visible(false);
	status_label->set_text(p_message);
	status_label->set_visible(true);
	
	// Show sign-in button so user can try signing in after confirming email
	if (show_email_mode) {
		// Switch to sign-in mode (hide name field, show sign-in button)
		_set_signup_mode(false);
		email_sign_in_button->set_visible(true);
		email_sign_up_button->set_visible(false);
		if (toggle_signup_mode_button) {
			toggle_signup_mode_button->set_visible(true);
		}
	} else {
		sign_in_button->set_visible(true);
		create_account_button->set_visible(false);
	}
}

// ===== NEW EMAIL/PASSWORD AUTHENTICATION METHODS =====

void AuthDialog::_setup_google_oauth_ui() {
	// Google OAuth Sign In button
	sign_in_button = memnew(Button);
	sign_in_button->set_text("Sign in with Google");
	sign_in_button->set_custom_minimum_size(Size2(300, 45) * EDSCALE);
	sign_in_button->add_theme_color_override("font_color", Color(1, 1, 1));
	sign_in_button->connect("pressed", callable_mp(this, &AuthDialog::_on_sign_in_pressed));
	
	HBoxContainer *signin_center = memnew(HBoxContainer);
	signin_center->set_alignment(BoxContainer::ALIGNMENT_CENTER);
	signin_center->add_child(sign_in_button);
	main_container->add_child(signin_center);
	
	Control *spacer4 = memnew(Control);
	spacer4->set_custom_minimum_size(Size2(0, 10) * EDSCALE);
	main_container->add_child(spacer4);
	
	// Create Account button
	create_account_button = memnew(Button);
	create_account_button->set_text("Create Account");
	create_account_button->set_custom_minimum_size(Size2(300, 45) * EDSCALE);
	create_account_button->connect("pressed", callable_mp(this, &AuthDialog::_on_create_account_pressed));
	
	HBoxContainer *signup_center = memnew(HBoxContainer);
	signup_center->set_alignment(BoxContainer::ALIGNMENT_CENTER);
	signup_center->add_child(create_account_button);
	main_container->add_child(signup_center);
	
	Control *spacer5 = memnew(Control);
	spacer5->set_custom_minimum_size(Size2(0, 20) * EDSCALE);
	main_container->add_child(spacer5);
	
	// Toggle to Email mode button
	toggle_auth_mode_button = memnew(Button);
	toggle_auth_mode_button->set_text("or Sign in with Email");
	toggle_auth_mode_button->set_flat(true);
	toggle_auth_mode_button->connect("pressed", callable_mp(this, &AuthDialog::_on_toggle_auth_mode_pressed));
	
	HBoxContainer *toggle_center = memnew(HBoxContainer);
	toggle_center->set_alignment(BoxContainer::ALIGNMENT_CENTER);
	toggle_center->add_child(toggle_auth_mode_button);
	main_container->add_child(toggle_center);
}

void AuthDialog::_setup_email_auth_ui() {
	// Email input
	VBoxContainer *email_container = memnew(VBoxContainer);
	main_container->add_child(email_container);
	
	Label *email_label = memnew(Label);
	email_label->set_text("Email:");
	email_container->add_child(email_label);
	
	email_input = memnew(LineEdit);
	email_input->set_placeholder("Enter your email");
	email_input->set_custom_minimum_size(Size2(350, 35) * EDSCALE);
	
	HBoxContainer *email_center = memnew(HBoxContainer);
	email_center->set_alignment(BoxContainer::ALIGNMENT_CENTER);
	email_center->add_child(email_input);
	email_container->add_child(email_center);
	
	Control *email_spacer = memnew(Control);
	email_spacer->set_custom_minimum_size(Size2(0, 10) * EDSCALE);
	email_container->add_child(email_spacer);
	
	// Password input
	Label *password_label = memnew(Label);
	password_label->set_text("Password:");
	email_container->add_child(password_label);
	
	password_input = memnew(LineEdit);
	password_input->set_placeholder("Enter your password");
	password_input->set_secret(true);
	password_input->set_custom_minimum_size(Size2(350, 35) * EDSCALE);
	
	HBoxContainer *password_center = memnew(HBoxContainer);
	password_center->set_alignment(BoxContainer::ALIGNMENT_CENTER);
	password_center->add_child(password_input);
	email_container->add_child(password_center);
	
	Control *password_spacer = memnew(Control);
	password_spacer->set_custom_minimum_size(Size2(0, 10) * EDSCALE);
	email_container->add_child(password_spacer);
	
	// Name input (for sign up)
	Label *name_label = memnew(Label);
	name_label->set_text("Full Name:");
	email_container->add_child(name_label);
	
	name_input = memnew(LineEdit);
	name_input->set_placeholder("Enter your full name");
	name_input->set_custom_minimum_size(Size2(350, 35) * EDSCALE);
	
	HBoxContainer *name_center = memnew(HBoxContainer);
	name_center->set_alignment(BoxContainer::ALIGNMENT_CENTER);
	name_center->add_child(name_input);
	email_container->add_child(name_center);
	
	Control *name_spacer = memnew(Control);
	name_spacer->set_custom_minimum_size(Size2(0, 20) * EDSCALE);
	email_container->add_child(name_spacer);
	
	// Sign In button
	email_sign_in_button = memnew(Button);
	email_sign_in_button->set_text("Sign In");
	email_sign_in_button->set_custom_minimum_size(Size2(300, 45) * EDSCALE);
	email_sign_in_button->connect("pressed", callable_mp(this, &AuthDialog::_on_email_sign_in_pressed));
	
	HBoxContainer *email_signin_center = memnew(HBoxContainer);
	email_signin_center->set_alignment(BoxContainer::ALIGNMENT_CENTER);
	email_signin_center->add_child(email_sign_in_button);
	email_container->add_child(email_signin_center);
	
	Control *button_spacer = memnew(Control);
	button_spacer->set_custom_minimum_size(Size2(0, 10) * EDSCALE);
	email_container->add_child(button_spacer);
	
	// Sign Up button
	email_sign_up_button = memnew(Button);
	email_sign_up_button->set_text("Create Account");
	email_sign_up_button->set_custom_minimum_size(Size2(300, 45) * EDSCALE);
	email_sign_up_button->connect("pressed", callable_mp(this, &AuthDialog::_on_email_sign_up_pressed));
	
	HBoxContainer *email_signup_center = memnew(HBoxContainer);
	email_signup_center->set_alignment(BoxContainer::ALIGNMENT_CENTER);
	email_signup_center->add_child(email_sign_up_button);
	email_container->add_child(email_signup_center);
	
	Control *toggle_spacer = memnew(Control);
	toggle_spacer->set_custom_minimum_size(Size2(0, 10) * EDSCALE);
	email_container->add_child(toggle_spacer);
	
	// Toggle between sign-in and sign-up mode button
	toggle_signup_mode_button = memnew(Button);
	toggle_signup_mode_button->set_text("Don't have an account? Create one");
	toggle_signup_mode_button->set_flat(true);
	toggle_signup_mode_button->connect("pressed", callable_mp(this, &AuthDialog::_toggle_signup_mode));
	
	HBoxContainer *toggle_signup_center = memnew(HBoxContainer);
	toggle_signup_center->set_alignment(BoxContainer::ALIGNMENT_CENTER);
	toggle_signup_center->add_child(toggle_signup_mode_button);
	email_container->add_child(toggle_signup_center);
	
	// Initially hide email UI
	email_container->set_visible(false);
}

void AuthDialog::_show_auth_mode(bool p_email_mode) {
	show_email_mode = p_email_mode;
	
	// Toggle visibility of Google OAuth UI
	sign_in_button->set_visible(!p_email_mode);
	create_account_button->set_visible(!p_email_mode);
	
	// Toggle visibility of Email/Password UI
	Control *email_container = Object::cast_to<Control>(email_input->get_parent()->get_parent());
	if (email_container) {
		email_container->set_visible(p_email_mode);
	}
	
	// Update toggle button text
	if (p_email_mode) {
		toggle_auth_mode_button->set_text("Back to Google");
		description_label->set_text("Sign in with email and password");
		// Set to sign-in mode (not signup) when switching to email mode
		_set_signup_mode(false);
		// Increase dialog size to accommodate email fields
		set_size(Size2(550, 600) * EDSCALE);
		set_min_size(Size2(500, 550) * EDSCALE);
	} else {
		toggle_auth_mode_button->set_text("or Sign in with Email");
		description_label->set_text("Choose your sign in method");
		// Reset to smaller size for Google OAuth mode
		set_size(Size2(550, 450) * EDSCALE);
		set_min_size(Size2(500, 400) * EDSCALE);
	}
}

void AuthDialog::_toggle_signup_mode() {
	_set_signup_mode(!show_signup_mode);
}

void AuthDialog::_set_signup_mode(bool p_signup_mode) {
	show_signup_mode = p_signup_mode;
	
	// Show/hide name field and adjust buttons
	Control *name_container = Object::cast_to<Control>(name_input->get_parent());
	if (name_container) {
		name_container->set_visible(show_signup_mode);
	}
	
	if (show_signup_mode) {
		email_sign_in_button->set_visible(false);
		email_sign_up_button->set_visible(true);
		email_sign_up_button->set_text("Create Account");
		if (toggle_signup_mode_button) {
			toggle_signup_mode_button->set_text("Already have an account? Sign in");
		}
		// Increase size further when showing full name field
		set_size(Size2(550, 650) * EDSCALE);
		set_min_size(Size2(500, 600) * EDSCALE);
	} else {
		email_sign_in_button->set_visible(true);
		email_sign_up_button->set_visible(false);
		if (toggle_signup_mode_button) {
			toggle_signup_mode_button->set_text("Don't have an account? Create one");
		}
		// Reset to smaller size for sign in mode
		set_size(Size2(550, 600) * EDSCALE);
		set_min_size(Size2(500, 550) * EDSCALE);
	}
}

void AuthDialog::_on_email_sign_in_pressed() {
	String email = email_input->get_text().strip_edges();
	String password = password_input->get_text();
	
	// Basic validation
	if (email.is_empty() || password.is_empty()) {
		show_error("Please enter both email and password");
		return;
	}
	
	if (!email.contains("@") || !email.contains(".")) {
		show_error("Please enter a valid email address");
		return;
	}
	
	if (password.length() < 6) {
		show_error("Password must be at least 6 characters");
		return;
	}
	
	// Call AuthManager
	AuthManager *auth = AuthManager::get_singleton();
	if (auth) {
		show_waiting();
		auth->sign_in_with_email(email, password);
	}
}

void AuthDialog::_on_email_sign_up_pressed() {
	String email = email_input->get_text().strip_edges();
	String password = password_input->get_text();
	String name = name_input->get_text().strip_edges();
	
	// Basic validation
	if (email.is_empty() || password.is_empty()) {
		show_error("Please enter email and password");
		return;
	}
	
	if (!email.contains("@") || !email.contains(".")) {
		show_error("Please enter a valid email address");
		return;
	}
	
	if (password.length() < 6) {
		show_error("Password must be at least 6 characters");
		return;
	}
	
	if (name.is_empty()) {
		name = email.get_slice("@", 0); // Use email prefix as fallback
	}
	
	// Call AuthManager
	AuthManager *auth = AuthManager::get_singleton();
	if (auth) {
		show_waiting();
		auth->sign_up_with_email(email, password, name);
	}
}

void AuthDialog::_on_toggle_auth_mode_pressed() {
	_show_auth_mode(!show_email_mode);
}

