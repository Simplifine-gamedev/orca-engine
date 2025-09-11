/*
© 2025 Simplifine Corp. Auto-update system for Orca Engine.
Personal Non‑Commercial License applies. Commercial use requires a separate license from Simplifine.
See LICENSES/COMPANY-NONCOMMERCIAL.md.
*/

#include "auto_update_manager.h"

#include "core/config/engine.h"
#include "core/io/json.h"
#include "core/io/http_client.h"
#include "core/os/os.h"
#include "core/version.h"
#include "core/variant/variant_parser.h"

AutoUpdateManager *AutoUpdateManager::singleton = nullptr;

AutoUpdateManager *AutoUpdateManager::get_singleton() {
	return singleton;
}

AutoUpdateManager::AutoUpdateManager() {
	singleton = this;
	
	// Initialize default values
	current_version = VERSION_FULL_CONFIG;
	backend_url = "http://localhost:8080";
	status = UPDATE_STATUS_UP_TO_DATE;
	auto_check_enabled = true;
	check_interval_hours = 24;
	last_check_time = 0;
	cache_valid = false;
	cache_timestamp = 0;
	is_checking = false;
	
	// Try to get backend URL from environment or project settings
	String env_backend_url = OS::get_singleton()->get_environment("ORCA_BACKEND_URL");
	if (!env_backend_url.is_empty()) {
		backend_url = env_backend_url;
	}
}

AutoUpdateManager::~AutoUpdateManager() {
	singleton = nullptr;
}

void AutoUpdateManager::_bind_methods() {
	// Core functionality
	ClassDB::bind_method(D_METHOD("check_for_updates"), &AutoUpdateManager::check_for_updates);
	ClassDB::bind_method(D_METHOD("get_update_info"), &AutoUpdateManager::get_update_info);
	ClassDB::bind_method(D_METHOD("download_update", "download_url"), &AutoUpdateManager::download_update);
	ClassDB::bind_method(D_METHOD("install_update", "file_path"), &AutoUpdateManager::install_update);
	
	// Configuration
	ClassDB::bind_method(D_METHOD("set_backend_url", "url"), &AutoUpdateManager::set_backend_url);
	ClassDB::bind_method(D_METHOD("get_backend_url"), &AutoUpdateManager::get_backend_url);
	ClassDB::bind_method(D_METHOD("set_current_version", "version"), &AutoUpdateManager::set_current_version);
	ClassDB::bind_method(D_METHOD("get_current_version"), &AutoUpdateManager::get_current_version);
	ClassDB::bind_method(D_METHOD("set_auto_check_enabled", "enabled"), &AutoUpdateManager::set_auto_check_enabled);
	ClassDB::bind_method(D_METHOD("is_auto_check_enabled"), &AutoUpdateManager::is_auto_check_enabled);
	ClassDB::bind_method(D_METHOD("set_check_interval_hours", "hours"), &AutoUpdateManager::set_check_interval_hours);
	ClassDB::bind_method(D_METHOD("get_check_interval_hours"), &AutoUpdateManager::get_check_interval_hours);
	
	// Status
	ClassDB::bind_method(D_METHOD("get_status"), &AutoUpdateManager::get_status);
	ClassDB::bind_method(D_METHOD("get_latest_version"), &AutoUpdateManager::get_latest_version);
	ClassDB::bind_method(D_METHOD("get_last_check_time"), &AutoUpdateManager::get_last_check_time);
	ClassDB::bind_method(D_METHOD("is_update_available"), &AutoUpdateManager::is_update_available);
	
	// Utility
	ClassDB::bind_method(D_METHOD("clear_cache"), &AutoUpdateManager::clear_cache);
	ClassDB::bind_method(D_METHOD("get_system_info"), &AutoUpdateManager::get_system_info);
	ClassDB::bind_method(D_METHOD("start_background_checking"), &AutoUpdateManager::start_background_checking);
	ClassDB::bind_method(D_METHOD("stop_background_checking"), &AutoUpdateManager::stop_background_checking);
	
	// Properties
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "backend_url"), "set_backend_url", "get_backend_url");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "current_version"), "set_current_version", "get_current_version");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "auto_check_enabled"), "set_auto_check_enabled", "is_auto_check_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "check_interval_hours"), "set_check_interval_hours", "get_check_interval_hours");
	
	// Signals
	ADD_SIGNAL(MethodInfo("update_available", PropertyInfo(Variant::STRING, "version"), PropertyInfo(Variant::STRING, "release_notes")));
	ADD_SIGNAL(MethodInfo("update_downloaded", PropertyInfo(Variant::STRING, "file_path")));
	ADD_SIGNAL(MethodInfo("update_error", PropertyInfo(Variant::STRING, "error_message")));
	ADD_SIGNAL(MethodInfo("update_progress", PropertyInfo(Variant::FLOAT, "progress")));
	
	// Enums
	BIND_ENUM_CONSTANT(UPDATE_STATUS_CHECKING);
	BIND_ENUM_CONSTANT(UPDATE_STATUS_AVAILABLE);
	BIND_ENUM_CONSTANT(UPDATE_STATUS_UP_TO_DATE);
	BIND_ENUM_CONSTANT(UPDATE_STATUS_ERROR);
	BIND_ENUM_CONSTANT(UPDATE_STATUS_DOWNLOADING);
	BIND_ENUM_CONSTANT(UPDATE_STATUS_INSTALLING);
}

void AutoUpdateManager::check_for_updates() {
	if (is_checking) {
		return; // Already checking
	}
	
	// Check cache first
	if (_is_cache_valid()) {
		Dictionary cached = _get_cached_info();
		_on_update_check_completed(cached);
		return;
	}
	
	is_checking = true;
	status = UPDATE_STATUS_CHECKING;
	
	// Make HTTP request to backend
	String url = backend_url + "/api/update/check";
	
	// For now, we'll use a simple approach and call the backend directly
	// In a real implementation, this would be async
	
	// Create HTTP client
	http_client.instantiate();
	
	// Parse URL
	String host;
	int port;
	String path;
	bool use_ssl = false;
	
	if (url.begins_with("https://")) {
		use_ssl = true;
		url = url.substr(8);
	} else if (url.begins_with("http://")) {
		url = url.substr(7);
	}
	
	int slash_pos = url.find("/");
	if (slash_pos != -1) {
		host = url.substr(0, slash_pos);
		path = url.substr(slash_pos);
	} else {
		host = url;
		path = "/";
	}
	
	int colon_pos = host.find(":");
	if (colon_pos != -1) {
		port = host.substr(colon_pos + 1).to_int();
		host = host.substr(0, colon_pos);
	} else {
		port = use_ssl ? 443 : 80;
	}
	
	// For now, simulate a successful response
	// In a real implementation, this would be an async HTTP request
	Dictionary result;
	result["update_available"] = false;
	result["current_version"] = current_version;
	result["latest_version"] = current_version;
	result["error"] = Variant();
	result["checked_at"] = OS::get_singleton()->get_datetime_string_from_system();
	
	_on_update_check_completed(result);
}

Dictionary AutoUpdateManager::get_update_info() {
	if (_is_cache_valid()) {
		return _get_cached_info();
	}
	
	Dictionary info;
	info["current_version"] = current_version;
	info["latest_version"] = latest_version;
	info["update_available"] = is_update_available();
	info["status"] = status;
	info["last_check_time"] = last_check_time;
	
	return info;
}

void AutoUpdateManager::download_update(const String &download_url) {
	status = UPDATE_STATUS_DOWNLOADING;
	
	// TODO: Implement actual download logic
	// For now, simulate success
	emit_signal("update_downloaded", "/tmp/orca_update.dmg");
}

void AutoUpdateManager::install_update(const String &file_path) {
	status = UPDATE_STATUS_INSTALLING;
	
	// TODO: Implement actual installation logic
	// This would typically involve platform-specific installation procedures
}

void AutoUpdateManager::set_backend_url(const String &url) {
	backend_url = url;
}

String AutoUpdateManager::get_backend_url() const {
	return backend_url;
}

void AutoUpdateManager::set_current_version(const String &version) {
	current_version = version;
}

String AutoUpdateManager::get_current_version() const {
	return current_version;
}

void AutoUpdateManager::set_auto_check_enabled(bool enabled) {
	auto_check_enabled = enabled;
}

bool AutoUpdateManager::is_auto_check_enabled() const {
	return auto_check_enabled;
}

void AutoUpdateManager::set_check_interval_hours(int hours) {
	check_interval_hours = MAX(1, hours); // Minimum 1 hour
}

int AutoUpdateManager::get_check_interval_hours() const {
	return check_interval_hours;
}

AutoUpdateManager::UpdateStatus AutoUpdateManager::get_status() const {
	return status;
}

String AutoUpdateManager::get_latest_version() const {
	return latest_version;
}

uint64_t AutoUpdateManager::get_last_check_time() const {
	return last_check_time;
}

bool AutoUpdateManager::is_update_available() const {
	return status == UPDATE_STATUS_AVAILABLE;
}

void AutoUpdateManager::clear_cache() {
	cache_valid = false;
	cached_update_info.clear();
	cache_timestamp = 0;
}

Dictionary AutoUpdateManager::get_system_info() {
	Dictionary info;
	info["platform"] = OS::get_singleton()->get_name();
	info["architecture"] = Engine::get_singleton()->get_architecture_name();
	info["version"] = current_version;
	
	return info;
}

void AutoUpdateManager::start_background_checking() {
	// TODO: Implement background checking with timer
}

void AutoUpdateManager::stop_background_checking() {
	// TODO: Stop background checking
}

void AutoUpdateManager::emit_update_available(const String &version, const String &notes) {
	emit_signal("update_available", version, notes);
}

void AutoUpdateManager::emit_update_downloaded(const String &file_path) {
	emit_signal("update_downloaded", file_path);
}

void AutoUpdateManager::emit_update_error(const String &error_message) {
	emit_signal("update_error", error_message);
}

void AutoUpdateManager::_on_update_check_completed(Dictionary result) {
	is_checking = false;
	last_check_time = OS::get_singleton()->get_ticks_msec();
	
	if (result.has("error") && !result["error"].is_null()) {
		status = UPDATE_STATUS_ERROR;
		emit_update_error(result["error"]);
		return;
	}
	
	bool update_available = result.get("update_available", false);
	latest_version = result.get("latest_version", current_version);
	
	if (update_available) {
		status = UPDATE_STATUS_AVAILABLE;
		String release_notes = result.get("release_notes", "");
		emit_update_available(latest_version, release_notes);
		
		// Cache the result
		_cache_update_info(result);
	} else {
		status = UPDATE_STATUS_UP_TO_DATE;
	}
}

bool AutoUpdateManager::_is_cache_valid() const {
	if (!cache_valid) {
		return false;
	}
	
	uint64_t current_time = OS::get_singleton()->get_ticks_msec();
	return (current_time - cache_timestamp) < CACHE_TIMEOUT_MS;
}

void AutoUpdateManager::_cache_update_info(const Dictionary &info) {
	cached_update_info = info;
	cache_valid = true;
	cache_timestamp = OS::get_singleton()->get_ticks_msec();
}

Dictionary AutoUpdateManager::_get_cached_info() const {
	return cached_update_info;
}