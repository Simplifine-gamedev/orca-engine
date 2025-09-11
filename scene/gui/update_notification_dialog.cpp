/*
© 2025 Simplifine Corp. Auto-update notification dialog for Orca Engine.
Personal Non‑Commercial License applies. Commercial use requires a separate license from Simplifine.
See LICENSES/COMPANY-NONCOMMERCIAL.md.
*/

#include "update_notification_dialog.h"

#include "scene/gui/box_container.h"
#include "scene/gui/label.h"
#include "scene/gui/separator.h"
#include "core/auto_update_manager.h"

UpdateNotificationDialog::UpdateNotificationDialog() {
	set_title("Orca Engine Update Available");
	set_ok_button_text("Install Now");
	set_size(Size2(500, 400));
	set_flag(FLAG_RESIZE_DISABLED, true);
	
	_setup_ui();
}

UpdateNotificationDialog::~UpdateNotificationDialog() {
}

void UpdateNotificationDialog::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_update_info", "update_info"), &UpdateNotificationDialog::set_update_info);
	ClassDB::bind_method(D_METHOD("set_current_version", "version"), &UpdateNotificationDialog::set_current_version);
	ClassDB::bind_method(D_METHOD("set_latest_version", "version"), &UpdateNotificationDialog::set_latest_version);
	ClassDB::bind_method(D_METHOD("set_release_notes", "notes"), &UpdateNotificationDialog::set_release_notes);
	ClassDB::bind_method(D_METHOD("set_download_url", "url"), &UpdateNotificationDialog::set_download_url);
	
	ClassDB::bind_method(D_METHOD("set_download_progress", "progress"), &UpdateNotificationDialog::set_download_progress);
	ClassDB::bind_method(D_METHOD("set_downloading", "downloading"), &UpdateNotificationDialog::set_downloading);
	ClassDB::bind_method(D_METHOD("set_installing", "installing"), &UpdateNotificationDialog::set_installing);
	
	ClassDB::bind_method(D_METHOD("get_selected_action"), &UpdateNotificationDialog::get_selected_action);
	ClassDB::bind_method(D_METHOD("get_current_version"), &UpdateNotificationDialog::get_current_version);
	ClassDB::bind_method(D_METHOD("get_latest_version"), &UpdateNotificationDialog::get_latest_version);
	ClassDB::bind_method(D_METHOD("get_download_url"), &UpdateNotificationDialog::get_download_url);
	ClassDB::bind_method(D_METHOD("is_update_ready"), &UpdateNotificationDialog::is_update_ready);
	
	ClassDB::bind_method(D_METHOD("update_status_text", "status"), &UpdateNotificationDialog::update_status_text);
	ClassDB::bind_method(D_METHOD("show_error", "error_message"), &UpdateNotificationDialog::show_error);
	ClassDB::bind_method(D_METHOD("show_success", "message"), &UpdateNotificationDialog::show_success);
	
	ClassDB::bind_method(D_METHOD("_on_install_now_pressed"), &UpdateNotificationDialog::_on_install_now_pressed);
	ClassDB::bind_method(D_METHOD("_on_install_later_pressed"), &UpdateNotificationDialog::_on_install_later_pressed);
	ClassDB::bind_method(D_METHOD("_on_skip_version_pressed"), &UpdateNotificationDialog::_on_skip_version_pressed);
	
	// Signals
	ADD_SIGNAL(MethodInfo("update_action_selected", PropertyInfo(Variant::INT, "action")));
	ADD_SIGNAL(MethodInfo("install_now_requested"));
	ADD_SIGNAL(MethodInfo("install_later_requested"));
	ADD_SIGNAL(MethodInfo("skip_version_requested"));
	
	// Enums
	BIND_ENUM_CONSTANT(UPDATE_ACTION_NONE);
	BIND_ENUM_CONSTANT(UPDATE_ACTION_INSTALL_NOW);
	BIND_ENUM_CONSTANT(UPDATE_ACTION_INSTALL_LATER);
	BIND_ENUM_CONSTANT(UPDATE_ACTION_SKIP_VERSION);
}

void UpdateNotificationDialog::_setup_ui() {
	// Create main container
	VBoxContainer *main_vbox = memnew(VBoxContainer);
	add_child(main_vbox);
	
	// Version info label
	Label *version_label = memnew(Label);
	version_label->set_text("A new version of Orca Engine is available!");
	version_label->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_CENTER);
	main_vbox->add_child(version_label);
	
	// Version details
	HBoxContainer *version_hbox = memnew(HBoxContainer);
	main_vbox->add_child(version_hbox);
	
	Label *current_label = memnew(Label);
	current_label->set_text("Current: ");
	version_hbox->add_child(current_label);
	
	Label *current_version_label = memnew(Label);
	current_version_label->set_name("CurrentVersionLabel");
	version_hbox->add_child(current_version_label);
	
	version_hbox->add_child(memnew(VSeparator));
	
	Label *latest_label = memnew(Label);
	latest_label->set_text("Latest: ");
	version_hbox->add_child(latest_label);
	
	Label *latest_version_label = memnew(Label);
	latest_version_label->set_name("LatestVersionLabel");
	latest_version_label->add_theme_color_override("font_color", Color(0.4, 0.8, 0.4)); // Green color
	version_hbox->add_child(latest_version_label);
	
	// Separator
	main_vbox->add_child(memnew(HSeparator));
	
	// Release notes section
	Label *notes_title = memnew(Label);
	notes_title->set_text("What's New:");
	main_vbox->add_child(notes_title);
	
	release_notes_label = memnew(RichTextLabel);
	release_notes_label->set_custom_minimum_size(Size2(460, 200));
	release_notes_label->set_fit_content(true);
	release_notes_label->set_scroll_active(true);
	release_notes_label->set_bbcode_enabled(true);
	main_vbox->add_child(release_notes_label);
	
	// Progress bar (initially hidden)
	download_progress = memnew(ProgressBar);
	download_progress->set_visible(false);
	download_progress->set_custom_minimum_size(Size2(460, 24));
	main_vbox->add_child(download_progress);
	
	// Status label
	Label *status_label = memnew(Label);
	status_label->set_name("StatusLabel");
	status_label->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_CENTER);
	status_label->set_visible(false);
	main_vbox->add_child(status_label);
	
	// Custom buttons
	install_later_button = add_button("Install Later", false, "install_later");
	skip_version_button = add_button("Skip This Version", false, "skip_version");
	
	// Connect button signals
	install_later_button->connect("pressed", callable_mp(this, &UpdateNotificationDialog::_on_install_later_pressed));
	skip_version_button->connect("pressed", callable_mp(this, &UpdateNotificationDialog::_on_skip_version_pressed));
}

void UpdateNotificationDialog::_notification(int p_what) {
	AcceptDialog::_notification(p_what);
	
	switch (p_what) {
		case NOTIFICATION_READY: {
			// Connect to AutoUpdateManager signals if available
			AutoUpdateManager *update_manager = AutoUpdateManager::get_singleton();
			if (update_manager) {
				update_manager->connect("update_progress", callable_mp(this, &UpdateNotificationDialog::set_download_progress));
				update_manager->connect("update_downloaded", callable_mp(this, &UpdateNotificationDialog::show_success));
				update_manager->connect("update_error", callable_mp(this, &UpdateNotificationDialog::show_error));
			}
		} break;
	}
}

void UpdateNotificationDialog::ok_pressed() {
	_on_install_now_pressed();
}

void UpdateNotificationDialog::_on_install_now_pressed() {
	selected_action = UPDATE_ACTION_INSTALL_NOW;
	
	// Update UI to show progress
	set_downloading(true);
	update_status_text("Starting download...");
	
	// Emit signal
	emit_signal("install_now_requested");
	emit_signal("update_action_selected", UPDATE_ACTION_INSTALL_NOW);
	
	// Start the actual download/install process
	AutoUpdateManager *update_manager = AutoUpdateManager::get_singleton();
	if (update_manager && !download_url.is_empty()) {
		update_manager->download_update(download_url);
	}
}

void UpdateNotificationDialog::_on_install_later_pressed() {
	selected_action = UPDATE_ACTION_INSTALL_LATER;
	
	emit_signal("install_later_requested");
	emit_signal("update_action_selected", UPDATE_ACTION_INSTALL_LATER);
	
	hide();
}

void UpdateNotificationDialog::_on_skip_version_pressed() {
	selected_action = UPDATE_ACTION_SKIP_VERSION;
	
	emit_signal("skip_version_requested");
	emit_signal("update_action_selected", UPDATE_ACTION_SKIP_VERSION);
	
	hide();
}

void UpdateNotificationDialog::set_update_info(const Dictionary &update_info) {
	if (update_info.has("current_version")) {
		set_current_version(update_info["current_version"]);
	}
	
	if (update_info.has("latest_version")) {
		set_latest_version(update_info["latest_version"]);
	}
	
	if (update_info.has("release_notes")) {
		set_release_notes(update_info["release_notes"]);
	}
	
	if (update_info.has("download_url")) {
		set_download_url(update_info["download_url"]);
	}
}

void UpdateNotificationDialog::set_current_version(const String &version) {
	current_version = version;
	
	Label *label = get_node<Label>("CurrentVersionLabel");
	if (label) {
		label->set_text(version);
	}
}

void UpdateNotificationDialog::set_latest_version(const String &version) {
	latest_version = version;
	
	Label *label = get_node<Label>("LatestVersionLabel");
	if (label) {
		label->set_text(version);
	}
}

void UpdateNotificationDialog::set_release_notes(const String &notes) {
	release_notes = notes;
	
	if (release_notes_label) {
		// Convert markdown-style formatting to BBCode
		String formatted_notes = notes;
		formatted_notes = formatted_notes.replace("**", "[b]");
		formatted_notes = formatted_notes.replace("*", "[i]");
		formatted_notes = formatted_notes.replace("##", "[size=16][b]");
		formatted_notes = formatted_notes.replace("#", "[size=18][b]");
		
		release_notes_label->set_text(formatted_notes);
	}
}

void UpdateNotificationDialog::set_download_url(const String &url) {
	download_url = url;
}

void UpdateNotificationDialog::set_download_progress(float progress) {
	if (download_progress) {
		download_progress->set_value(progress);
		
		if (progress > 0 && !download_progress->is_visible()) {
			download_progress->set_visible(true);
		}
	}
}

void UpdateNotificationDialog::set_downloading(bool downloading) {
	is_downloading = downloading;
	
	// Update UI state
	if (downloading) {
		get_ok_button()->set_disabled(true);
		install_later_button->set_disabled(true);
		skip_version_button->set_disabled(true);
		
		if (download_progress) {
			download_progress->set_visible(true);
		}
	} else {
		get_ok_button()->set_disabled(false);
		install_later_button->set_disabled(false);
		skip_version_button->set_disabled(false);
		
		if (download_progress) {
			download_progress->set_visible(false);
		}
	}
}

void UpdateNotificationDialog::set_installing(bool installing) {
	is_installing = installing;
	
	if (installing) {
		update_status_text("Installing update...");
		get_ok_button()->set_disabled(true);
		install_later_button->set_disabled(true);
		skip_version_button->set_disabled(true);
	}
}

UpdateNotificationDialog::UpdateAction UpdateNotificationDialog::get_selected_action() const {
	return selected_action;
}

String UpdateNotificationDialog::get_current_version() const {
	return current_version;
}

String UpdateNotificationDialog::get_latest_version() const {
	return latest_version;
}

String UpdateNotificationDialog::get_download_url() const {
	return download_url;
}

bool UpdateNotificationDialog::is_update_ready() const {
	return !download_url.is_empty() && !latest_version.is_empty();
}

void UpdateNotificationDialog::update_status_text(const String &status) {
	Label *status_label = get_node<Label>("StatusLabel");
	if (status_label) {
		status_label->set_text(status);
		status_label->set_visible(!status.is_empty());
	}
}

void UpdateNotificationDialog::show_error(const String &error_message) {
	update_status_text("Error: " + error_message);
	
	Label *status_label = get_node<Label>("StatusLabel");
	if (status_label) {
		status_label->add_theme_color_override("font_color", Color(0.8, 0.2, 0.2)); // Red color
	}
	
	set_downloading(false);
	set_installing(false);
}

void UpdateNotificationDialog::show_success(const String &message) {
	update_status_text(message);
	
	Label *status_label = get_node<Label>("StatusLabel");
	if (status_label) {
		status_label->add_theme_color_override("font_color", Color(0.2, 0.8, 0.2)); // Green color
	}
	
	// If installation is complete, close the dialog after a short delay
	if (message.contains("complete") || message.contains("installed")) {
		// Schedule close after 3 seconds
		get_tree()->create_timer(3.0)->connect("timeout", callable_mp((Window *)this, &Window::hide));
	}
}

void UpdateNotificationDialog::show_update_notification(Node *parent, const Dictionary &update_info) {
	UpdateNotificationDialog *dialog = memnew(UpdateNotificationDialog);
	dialog->set_update_info(update_info);
	
	parent->add_child(dialog);
	dialog->popup_centered();
}