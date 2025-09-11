/*
© 2025 Simplifine Corp. Auto-update service for Orca Engine startup integration.
Personal Non‑Commercial License applies. Commercial use requires a separate license from Simplifine.
See LICENSES/COMPANY-NONCOMMERCIAL.md.
*/

#include "auto_update_service.h"

#include "core/auto_update_manager.h"
#include "scene/gui/update_notification_dialog.h"
#include "core/config/project_settings.h"
#include "core/os/os.h"

AutoUpdateService *AutoUpdateService::singleton = nullptr;

AutoUpdateService *AutoUpdateService::get_singleton() {
	return singleton;
}

AutoUpdateService::AutoUpdateService() {
	singleton = this;
	set_name("AutoUpdateService");
	set_process_mode(PROCESS_MODE_ALWAYS);
	
	_load_user_preferences();
}

AutoUpdateService::~AutoUpdateService() {
	singleton = nullptr;
}

void AutoUpdateService::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_auto_check_on_startup", "enabled"), &AutoUpdateService::set_auto_check_on_startup);
	ClassDB::bind_method(D_METHOD("is_auto_check_on_startup"), &AutoUpdateService::is_auto_check_on_startup);
	
	ClassDB::bind_method(D_METHOD("set_background_checking_enabled", "enabled"), &AutoUpdateService::set_background_checking_enabled);
	ClassDB::bind_method(D_METHOD("is_background_checking_enabled"), &AutoUpdateService::is_background_checking_enabled);
	
	ClassDB::bind_method(D_METHOD("set_check_interval_hours", "hours"), &AutoUpdateService::set_check_interval_hours);
	ClassDB::bind_method(D_METHOD("get_check_interval_hours"), &AutoUpdateService::get_check_interval_hours);
	
	ClassDB::bind_method(D_METHOD("check_for_updates_now"), &AutoUpdateService::check_for_updates_now);
	ClassDB::bind_method(D_METHOD("show_update_dialog_if_available"), &AutoUpdateService::show_update_dialog_if_available);
	ClassDB::bind_method(D_METHOD("skip_version", "version"), &AutoUpdateService::skip_version);
	
	ClassDB::bind_method(D_METHOD("is_update_available"), &AutoUpdateService::is_update_available);
	ClassDB::bind_method(D_METHOD("get_last_update_info"), &AutoUpdateService::get_last_update_info);
	ClassDB::bind_method(D_METHOD("has_startup_check_completed"), &AutoUpdateService::has_startup_check_completed);
	
	ClassDB::bind_method(D_METHOD("reset_skipped_versions"), &AutoUpdateService::reset_skipped_versions);
	ClassDB::bind_method(D_METHOD("get_skipped_version"), &AutoUpdateService::get_skipped_version);
	
	// Internal callbacks
	ClassDB::bind_method(D_METHOD("_on_update_check_timer_timeout"), &AutoUpdateService::_on_update_check_timer_timeout);
	ClassDB::bind_method(D_METHOD("_on_update_available", "version", "notes"), &AutoUpdateService::_on_update_available);
	ClassDB::bind_method(D_METHOD("_on_update_error", "error"), &AutoUpdateService::_on_update_error);
	ClassDB::bind_method(D_METHOD("_on_update_downloaded", "file_path"), &AutoUpdateService::_on_update_downloaded);
	
	ClassDB::bind_method(D_METHOD("_on_dialog_install_now_requested"), &AutoUpdateService::_on_dialog_install_now_requested);
	ClassDB::bind_method(D_METHOD("_on_dialog_install_later_requested"), &AutoUpdateService::_on_dialog_install_later_requested);
	ClassDB::bind_method(D_METHOD("_on_dialog_skip_version_requested"), &AutoUpdateService::_on_dialog_skip_version_requested);
	
	// Properties
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "auto_check_on_startup"), "set_auto_check_on_startup", "is_auto_check_on_startup");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "background_checking_enabled"), "set_background_checking_enabled", "is_background_checking_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "check_interval_hours"), "set_check_interval_hours", "get_check_interval_hours");
	
	// Signals
	ADD_SIGNAL(MethodInfo("update_check_completed", PropertyInfo(Variant::DICTIONARY, "update_info")));
	ADD_SIGNAL(MethodInfo("update_notification_shown", PropertyInfo(Variant::STRING, "version")));
	ADD_SIGNAL(MethodInfo("update_install_started"));
	ADD_SIGNAL(MethodInfo("update_install_completed"));
}

void AutoUpdateService::_ready() {
	// Get reference to AutoUpdateManager
	update_manager = AutoUpdateManager::get_singleton();
	if (!update_manager) {
		print_error("AutoUpdateService: AutoUpdateManager singleton not found!");
		return;
	}
	
	// Connect to AutoUpdateManager signals
	update_manager->connect("update_available", callable_mp(this, &AutoUpdateService::_on_update_available));
	update_manager->connect("update_error", callable_mp(this, &AutoUpdateService::_on_update_error));
	update_manager->connect("update_downloaded", callable_mp(this, &AutoUpdateService::_on_update_downloaded));
	
	// Setup background checking timer
	_setup_background_checking();
	
	// Perform startup check after a short delay to allow the application to fully initialize
	if (auto_check_on_startup) {
		call_deferred("_perform_startup_check");
	}
}

void AutoUpdateService::_notification(int p_what) {
	Node::_notification(p_what);
	
	switch (p_what) {
		case NOTIFICATION_WM_CLOSE_REQUEST: {
			_save_user_preferences();
		} break;
	}
}

void AutoUpdateService::_perform_startup_check() {
	if (startup_check_completed || !update_manager) {
		return;
	}
	
	print_line("AutoUpdateService: Performing startup update check...");
	update_manager->check_for_updates();
	startup_check_completed = true;
}

void AutoUpdateService::_setup_background_checking() {
	if (!background_checking_enabled) {
		return;
	}
	
	// Create timer for background checks
	check_timer = memnew(Timer);
	check_timer->set_wait_time(check_interval_hours * 3600.0); // Convert hours to seconds
	check_timer->set_autostart(true);
	check_timer->connect("timeout", callable_mp(this, &AutoUpdateService::_on_update_check_timer_timeout));
	add_child(check_timer);
}

void AutoUpdateService::_on_update_check_timer_timeout() {
	if (update_manager) {
		print_line("AutoUpdateService: Background update check triggered");
		update_manager->check_for_updates();
	}
}

void AutoUpdateService::_on_update_available(const String &version, const String &notes) {
	Dictionary update_info;
	update_info["latest_version"] = version;
	update_info["release_notes"] = notes;
	update_info["update_available"] = true;
	
	if (update_manager) {
		Dictionary full_info = update_manager->get_update_info();
		for (int i = 0; i < full_info.size(); i++) {
			Variant key = full_info.get_key_at_index(i);
			update_info[key] = full_info[key];
		}
	}
	
	last_update_info = update_info;
	
	emit_signal("update_check_completed", update_info);
	
	// Show notification if appropriate
	if (_should_show_notification(update_info)) {
		call_deferred("_show_update_notification", update_info);
	}
}

void AutoUpdateService::_on_update_error(const String &error) {
	print_error("AutoUpdateService: Update check failed: " + error);
	
	Dictionary error_info;
	error_info["error"] = error;
	error_info["update_available"] = false;
	
	emit_signal("update_check_completed", error_info);
}

void AutoUpdateService::_on_update_downloaded(const String &file_path) {
	print_line("AutoUpdateService: Update downloaded to: " + file_path);
	
	// Auto-install if dialog is still open
	if (notification_dialog && notification_dialog->is_visible()) {
		notification_dialog->set_installing(true);
		
		if (update_manager) {
			update_manager->install_update(file_path);
		}
	}
}

void AutoUpdateService::_show_update_notification(const Dictionary &update_info) {
	// Don't show if we already have a dialog open
	if (notification_dialog && notification_dialog->is_visible()) {
		return;
	}
	
	// Create new notification dialog
	notification_dialog = memnew(UpdateNotificationDialog);
	notification_dialog->set_update_info(update_info);
	
	// Connect dialog signals
	notification_dialog->connect("install_now_requested", callable_mp(this, &AutoUpdateService::_on_dialog_install_now_requested));
	notification_dialog->connect("install_later_requested", callable_mp(this, &AutoUpdateService::_on_dialog_install_later_requested));
	notification_dialog->connect("skip_version_requested", callable_mp(this, &AutoUpdateService::_on_dialog_skip_version_requested));
	
	// Add to scene and show
	get_tree()->get_current_scene()->add_child(notification_dialog);
	notification_dialog->popup_centered();
	
	String version = update_info.get("latest_version", "unknown");
	emit_signal("update_notification_shown", version);
	
	print_line("AutoUpdateService: Update notification shown for version " + version);
}

void AutoUpdateService::_on_dialog_install_now_requested() {
	emit_signal("update_install_started");
	print_line("AutoUpdateService: User requested immediate update installation");
}

void AutoUpdateService::_on_dialog_install_later_requested() {
	print_line("AutoUpdateService: User chose to install update later");
	
	if (notification_dialog) {
		notification_dialog->queue_free();
		notification_dialog = nullptr;
	}
}

void AutoUpdateService::_on_dialog_skip_version_requested() {
	if (notification_dialog) {
		String version = notification_dialog->get_latest_version();
		skip_version(version);
		
		notification_dialog->queue_free();
		notification_dialog = nullptr;
		
		print_line("AutoUpdateService: User skipped version " + version);
	}
}

bool AutoUpdateService::_should_show_notification(const Dictionary &update_info) {
	// Don't show if update is not available
	if (!update_info.get("update_available", false)) {
		return false;
	}
	
	// Don't show if user has skipped this version
	String latest_version = update_info.get("latest_version", "");
	if (!latest_version.is_empty() && latest_version == skipped_version) {
		return false;
	}
	
	// Don't show if we don't have download info
	if (!update_info.has("download_url") || update_info["download_url"].operator String().is_empty()) {
		return false;
	}
	
	return true;
}

void AutoUpdateService::_save_user_preferences() {
	// Save preferences to project settings or user data
	ProjectSettings *ps = ProjectSettings::get_singleton();
	if (ps) {
		ps->set_setting("application/auto_update/skipped_version", skipped_version);
		ps->set_setting("application/auto_update/auto_check_on_startup", auto_check_on_startup);
		ps->set_setting("application/auto_update/background_checking_enabled", background_checking_enabled);
		ps->set_setting("application/auto_update/check_interval_hours", check_interval_hours);
		
		// Save to user://settings.cfg or similar
		ps->save_custom(OS::get_singleton()->get_user_data_dir() + "/auto_update_settings.cfg");
	}
}

void AutoUpdateService::_load_user_preferences() {
	// Load preferences from project settings
	ProjectSettings *ps = ProjectSettings::get_singleton();
	if (ps) {
		String settings_path = OS::get_singleton()->get_user_data_dir() + "/auto_update_settings.cfg";
		if (FileAccess::file_exists(settings_path)) {
			// Load custom settings file
			Ref<ConfigFile> config;
			config.instantiate();
			Error err = config->load(settings_path);
			
			if (err == OK) {
				skipped_version = config->get_value("auto_update", "skipped_version", "");
				auto_check_on_startup = config->get_value("auto_update", "auto_check_on_startup", true);
				background_checking_enabled = config->get_value("auto_update", "background_checking_enabled", true);
				check_interval_hours = config->get_value("auto_update", "check_interval_hours", 24);
			}
		}
	}
}

// Public API methods

void AutoUpdateService::set_auto_check_on_startup(bool enabled) {
	auto_check_on_startup = enabled;
	_save_user_preferences();
}

bool AutoUpdateService::is_auto_check_on_startup() const {
	return auto_check_on_startup;
}

void AutoUpdateService::set_background_checking_enabled(bool enabled) {
	background_checking_enabled = enabled;
	
	if (check_timer) {
		check_timer->set_paused(!enabled);
	}
	
	_save_user_preferences();
}

bool AutoUpdateService::is_background_checking_enabled() const {
	return background_checking_enabled;
}

void AutoUpdateService::set_check_interval_hours(int hours) {
	check_interval_hours = MAX(1, hours); // Minimum 1 hour
	
	if (check_timer) {
		check_timer->set_wait_time(check_interval_hours * 3600.0);
	}
	
	_save_user_preferences();
}

int AutoUpdateService::get_check_interval_hours() const {
	return check_interval_hours;
}

void AutoUpdateService::check_for_updates_now() {
	if (update_manager) {
		print_line("AutoUpdateService: Manual update check requested");
		update_manager->check_for_updates();
	}
}

void AutoUpdateService::show_update_dialog_if_available() {
	if (is_update_available()) {
		_show_update_notification(last_update_info);
	}
}

void AutoUpdateService::skip_version(const String &version) {
	skipped_version = version;
	_save_user_preferences();
}

bool AutoUpdateService::is_update_available() const {
	return last_update_info.get("update_available", false);
}

Dictionary AutoUpdateService::get_last_update_info() const {
	return last_update_info;
}

bool AutoUpdateService::has_startup_check_completed() const {
	return startup_check_completed;
}

void AutoUpdateService::reset_skipped_versions() {
	skipped_version = "";
	_save_user_preferences();
}

String AutoUpdateService::get_skipped_version() const {
	return skipped_version;
}