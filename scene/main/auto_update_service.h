/*
© 2025 Simplifine Corp. Auto-update service for Orca Engine startup integration.
Personal Non‑Commercial License applies. Commercial use requires a separate license from Simplifine.
See LICENSES/COMPANY-NONCOMMERCIAL.md.
*/

#pragma once

#include "scene/main/node.h"
#include "scene/main/timer.h"
#include "core/variant/dictionary.h"

class UpdateNotificationDialog;
class AutoUpdateManager;

class AutoUpdateService : public Node {
	GDCLASS(AutoUpdateService, Node);

private:
	static AutoUpdateService *singleton;
	
	AutoUpdateManager *update_manager = nullptr;
	UpdateNotificationDialog *notification_dialog = nullptr;
	Timer *check_timer = nullptr;
	
	bool auto_check_on_startup = true;
	bool background_checking_enabled = true;
	int check_interval_hours = 24;
	bool startup_check_completed = false;
	
	Dictionary last_update_info;
	String skipped_version;
	
	void _on_update_check_timer_timeout();
	void _on_update_available(const String &version, const String &notes);
	void _on_update_error(const String &error);
	void _on_update_downloaded(const String &file_path);
	
	void _on_dialog_install_now_requested();
	void _on_dialog_install_later_requested();
	void _on_dialog_skip_version_requested();
	
	void _show_update_notification(const Dictionary &update_info);
	void _perform_startup_check();
	void _setup_background_checking();
	bool _should_show_notification(const Dictionary &update_info);
	void _save_user_preferences();
	void _load_user_preferences();

protected:
	void _ready() override;
	void _notification(int p_what);
	static void _bind_methods();

public:
	static AutoUpdateService *get_singleton();
	
	AutoUpdateService();
	~AutoUpdateService();
	
	// Configuration
	void set_auto_check_on_startup(bool enabled);
	bool is_auto_check_on_startup() const;
	
	void set_background_checking_enabled(bool enabled);
	bool is_background_checking_enabled() const;
	
	void set_check_interval_hours(int hours);
	int get_check_interval_hours() const;
	
	// Manual operations
	void check_for_updates_now();
	void show_update_dialog_if_available();
	void skip_version(const String &version);
	
	// Status
	bool is_update_available() const;
	Dictionary get_last_update_info() const;
	bool has_startup_check_completed() const;
	
	// Utility
	void reset_skipped_versions();
	String get_skipped_version() const;
};
