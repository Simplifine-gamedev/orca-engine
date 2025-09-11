/*
© 2025 Simplifine Corp. Auto-update system for Orca Engine.
Personal Non‑Commercial License applies. Commercial use requires a separate license from Simplifine.
See LICENSES/COMPANY-NONCOMMERCIAL.md.
*/

#pragma once

#include "core/object/object.h"
#include "core/object/ref_counted.h"
#include "core/variant/dictionary.h"
#include "core/string/ustring.h"
#include "core/templates/vector.h"
#include "core/io/http_client.h"

class AutoUpdateManager : public Object {
	GDCLASS(AutoUpdateManager, Object);

public:
	enum UpdateStatus {
		UPDATE_STATUS_CHECKING,
		UPDATE_STATUS_AVAILABLE,
		UPDATE_STATUS_UP_TO_DATE,
		UPDATE_STATUS_ERROR,
		UPDATE_STATUS_DOWNLOADING,
		UPDATE_STATUS_INSTALLING
	};

private:
	static AutoUpdateManager *singleton;
	
	String current_version;
	String latest_version;
	String backend_url;
	UpdateStatus status;
	bool auto_check_enabled;
	int check_interval_hours;
	uint64_t last_check_time;
	
	Dictionary cached_update_info;
	bool cache_valid;
	uint64_t cache_timestamp;
	static const uint64_t CACHE_TIMEOUT_MS = 3600000; // 1 hour
	
	// HTTP client for API requests
	Ref<HTTPClient> http_client;
	bool is_checking;
	
	void _on_update_check_completed(Dictionary result);
	void _parse_update_response(const String &response_body);
	bool _is_cache_valid() const;
	void _cache_update_info(const Dictionary &info);
	Dictionary _get_cached_info() const;

protected:
	static void _bind_methods();

public:
	static AutoUpdateManager *get_singleton();
	
	AutoUpdateManager();
	~AutoUpdateManager();
	
	// Core update functionality
	void check_for_updates();
	Dictionary get_update_info();
	void download_update(const String &download_url);
	void install_update(const String &file_path);
	
	// Configuration
	void set_backend_url(const String &url);
	String get_backend_url() const;
	
	void set_current_version(const String &version);
	String get_current_version() const;
	
	void set_auto_check_enabled(bool enabled);
	bool is_auto_check_enabled() const;
	
	void set_check_interval_hours(int hours);
	int get_check_interval_hours() const;
	
	// Status and info
	UpdateStatus get_status() const;
	String get_latest_version() const;
	uint64_t get_last_check_time() const;
	bool is_update_available() const;
	
	// Utility functions
	void clear_cache();
	Dictionary get_system_info();
	
	// Background checking
	void start_background_checking();
	void stop_background_checking();
	
	// Signals
	void emit_update_available(const String &version, const String &notes);
	void emit_update_downloaded(const String &file_path);
	void emit_update_error(const String &error_message);
};

VARIANT_ENUM_CAST(AutoUpdateManager::UpdateStatus);