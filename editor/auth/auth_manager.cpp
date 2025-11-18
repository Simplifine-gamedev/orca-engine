/**************************************************************************/
/*  auth_manager.cpp                                                      */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             ORCA ENGINE                                */
/**************************************************************************/

#include "auth_manager.h"
#include "auth_dialog.h"

#include "core/config/project_settings.h"
#include "core/io/http_client.h"
#include "core/io/json.h"
#include "core/os/os.h"
#include "core/io/file_access.h"
#include "core/io/dir_access.h"
#include "core/crypto/crypto.h"
#include "editor/settings/editor_settings.h"

#ifdef WINDOWS_ENABLED
#include <windows.h>
#include <wincred.h>
#endif

#ifdef LINUXBSD_ENABLED
// For Linux we'll use a simple encrypted file approach
// In production, you'd use libsecret
#endif

AuthManager *AuthManager::singleton = nullptr;

void AuthManager::_bind_methods() {
	ClassDB::bind_method(D_METHOD("open_web_login"), &AuthManager::open_web_login);
	ClassDB::bind_method(D_METHOD("handle_deep_link", "url"), &AuthManager::handle_deep_link);
	ClassDB::bind_method(D_METHOD("sign_out"), &AuthManager::sign_out);
	ClassDB::bind_method(D_METHOD("get_is_authenticated"), &AuthManager::get_is_authenticated);
	ClassDB::bind_method(D_METHOD("get_user_id"), &AuthManager::get_user_id);
	ClassDB::bind_method(D_METHOD("get_user_email"), &AuthManager::get_user_email);
	ClassDB::bind_method(D_METHOD("get_user_name"), &AuthManager::get_user_name);
	ADD_SIGNAL(MethodInfo("auth_state_changed"));
}

AuthManager::AuthManager() {
	singleton = this;
}

AuthManager::~AuthManager() {
	_stop_loopback_server();
	singleton = nullptr;
}

void AuthManager::open_web_login() {
	String login_url = LOGIN_URL;

	if (_start_loopback_server()) {
		String redirect = vformat("http://127.0.0.1:%d/auth/callback", loopback_port);
		String redirect_param = "redirect=" + redirect.uri_encode();
		login_url += login_url.contains("?") ? "&" + redirect_param : "?" + redirect_param;
		print_line(vformat("ORCA AUTH: Loopback redirect active on %s", redirect));
	} else {
		print_line("ORCA AUTH: Loopback redirect unavailable, falling back to custom scheme only");
	}

	print_line("Opening web login at: " + login_url);
	OS::get_singleton()->shell_open(login_url);
}

bool AuthManager::handle_deep_link(const String &p_url) {
	print_line("Received deep link: " + p_url);

	// Parse URL: orca://auth?access_token=XXX&refresh_token=YYY&user_id=ZZZ&email=...&name=...
	if (!p_url.begins_with("orca://")) {
		return false;
	}

	String path = p_url.substr(7); // Remove "orca://"
	int query_start = path.find("?");
	
	if (query_start == -1) {
		return false;
	}

	String host = path.substr(0, query_start);
	if (host != "auth") {
		return false;
	}

	// Parse query parameters
	String query_string = path.substr(query_start + 1);
	PackedStringArray params = query_string.split("&");
	
	HashMap<String, String> query_params;
	for (int i = 0; i < params.size(); i++) {
		PackedStringArray kv = params[i].split("=");
		if (kv.size() == 2) {
			query_params[kv[0]] = kv[1].uri_decode();
		}
	}

	// Extract tokens
	if (!query_params.has("access_token") || !query_params.has("refresh_token") || !query_params.has("user_id")) {
		print_error("AuthManager: Missing required parameters in deep link");
		return false;
	}

	String new_access_token = query_params["access_token"];
	String new_refresh_token = query_params["refresh_token"];
	String new_user_id = query_params["user_id"];
	String new_email = query_params.has("email") ? query_params["email"] : "";
	String new_name = query_params.has("name") ? query_params["name"] : "";

	// Store tokens
	store_tokens(new_access_token, new_refresh_token, new_user_id, new_email, new_name);
	
	is_authenticated = true;
	print_line("Authentication successful for user: " + user_email);
	_stop_loopback_server();
	
	// Initialize Autumn account after successful login
	initialize_autumn_account();
	
	// Notify the auth dialog if it exists
	if (auth_dialog) {
		auth_dialog->call_deferred("_check_auth_status");
	}

	return true;
}

bool AuthManager::try_auto_login() {
	// Try to load stored tokens, but fail silently if keychain access is denied
	// This prevents keychain prompts from blocking app startup
	if (load_stored_tokens()) {
		// TODO: Verify tokens are still valid by making a test API call
		is_authenticated = true;
		print_line("Auto-login successful for user: " + user_email);
		
		// Initialize Autumn account after auto-login
		initialize_autumn_account();
		
		return true;
	}
	// Don't print error - this is expected if user hasn't logged in yet or denied keychain access
	return false;
}

void AuthManager::sign_out() {
	clear_stored_tokens();
	is_authenticated = false;
	access_token = "";
	refresh_token = "";
	user_id = "";
	user_email = "";
	user_name = "";
	_stop_loopback_server();
	print_line("User signed out");
	emit_signal("auth_state_changed");
}

void AuthManager::sign_in_with_email(const String &p_email, const String &p_password) {
	print_line("AuthManager: Signing in with email: " + p_email);
	
	// Make HTTP request to Supabase auth endpoint
	Ref<HTTPClient> http = HTTPClient::create();
	// Remove https:// prefix from SUPABASE_URL for connect_to_host
	String host = SUPABASE_URL.replace("https://", "").replace("http://", "");
	Error err = http->connect_to_host(host, 443, TLSOptions::client());
	if (err != OK) {
		print_error("Failed to connect to Supabase for email sign-in");
		return;
	}
	
	// Wait for connection
	while (http->get_status() == HTTPClient::STATUS_CONNECTING || http->get_status() == HTTPClient::STATUS_RESOLVING) {
		http->poll();
		OS::get_singleton()->delay_usec(10000);
	}
	
	if (http->get_status() != HTTPClient::STATUS_CONNECTED) {
		print_error("Could not connect to Supabase");
		return;
	}
	
	// Prepare request body
	Dictionary body_dict;
	body_dict["email"] = p_email;
	body_dict["password"] = p_password;
	String body_json = JSON::stringify(body_dict);
	
	// Prepare headers
	Vector<String> headers;
	headers.push_back("Content-Type: application/json");
	headers.push_back("apikey: " + SUPABASE_ANON_KEY);
	
	// Make POST request to /auth/v1/token?grant_type=password
	CharString body_utf8 = body_json.utf8();
	err = http->request(HTTPClient::METHOD_POST, "/auth/v1/token?grant_type=password", headers, (const uint8_t *)body_utf8.get_data(), body_utf8.length());
	
	if (err != OK) {
		print_error("Failed to make sign-in request");
		return;
	}
	
	// Wait for response
	while (http->get_status() == HTTPClient::STATUS_REQUESTING) {
		http->poll();
		OS::get_singleton()->delay_usec(10000);
	}
	
	// Read response
	if (http->has_response()) {
		PackedByteArray rb;
		while (http->get_status() == HTTPClient::STATUS_BODY) {
			http->poll();
			PackedByteArray chunk = http->read_response_body_chunk();
			if (chunk.size() == 0) {
				OS::get_singleton()->delay_usec(10000);
			} else {
				rb.append_array(chunk);
			}
		}
		
		String response_text = String::utf8((const char *)rb.ptr(), rb.size());
		print_line("Email sign-in response: " + response_text);
		
		// Check HTTP status code
		int status_code = http->get_response_code();
		if (status_code >= 400) {
			// Parse error response
			JSON json;
			Error parse_err = json.parse(response_text);
			if (parse_err == OK) {
				Dictionary error_response = json.get_data();
				String error_msg = "Sign-in failed";
				
				// Handle different error response formats
				if (error_response.has("msg")) {
					error_msg = error_response["msg"];
				} else if (error_response.has("error_code")) {
					String error_code = error_response["error_code"];
					if (error_code == "email_not_confirmed") {
						error_msg = "Email not confirmed. Please check your email and click the confirmation link before signing in.";
					} else {
						error_msg = error_response.get("msg", error_code);
					}
				} else if (error_response.has("error")) {
					error_msg = error_response.get("error_description", error_response.get("error", "Unknown error"));
				} else {
					error_msg = "Sign-in failed (HTTP " + String::num(status_code) + ")";
				}
				
				print_error("Sign-in failed: " + error_msg);
				_notify_auth_error(error_msg);
				return;
			} else {
				_notify_auth_error("Sign-in failed (HTTP " + String::num(status_code) + ")");
				return;
			}
		}
		
		// Parse JSON response
		JSON json;
		Error parse_err = json.parse(response_text);
		if (parse_err != OK) {
			print_error("Failed to parse sign-in response");
			_notify_auth_error("Invalid response from server");
			return;
		}
		
		Dictionary response = json.get_data();
		
		// Check for error in response (legacy format)
		if (response.has("error")) {
			String error_msg = response.get("error_description", response.get("error", "Unknown error"));
			print_error("Sign-in failed: " + error_msg);
			_notify_auth_error(error_msg);
			return;
		}
		
		// Extract tokens and user info
		if (response.has("access_token") && response.has("refresh_token")) {
			String new_access_token = response["access_token"];
			String new_refresh_token = response["refresh_token"];
			
			Dictionary user_data = response.get("user", Dictionary());
			String new_user_id = user_data.get("id", "");
			String new_email = user_data.get("email", p_email);
			
			// Get name from user metadata
			String new_name = "";
			if (user_data.has("user_metadata")) {
				Dictionary user_metadata = user_data["user_metadata"];
				new_name = user_metadata.get("name", p_email.get_slice("@", 0));
			} else {
				new_name = p_email.get_slice("@", 0);
			}
			
			// Store tokens and mark as authenticated
			store_tokens(new_access_token, new_refresh_token, new_user_id, new_email, new_name);
			is_authenticated = true;
			
			print_line("Email sign-in successful!");
			
			// Initialize Autumn account after successful login
			initialize_autumn_account();
			
			_notify_auth_success();
		} else {
			print_error("Invalid sign-in response format");
			_notify_auth_error("Invalid response from server");
		}
	} else {
		_notify_auth_error("Network error occurred");
	}
}

void AuthManager::sign_up_with_email(const String &p_email, const String &p_password, const String &p_name) {
	print_line("AuthManager: Signing up with email: " + p_email);
	
	// Make HTTP request to Supabase auth endpoint
	Ref<HTTPClient> http = HTTPClient::create();
	// Remove https:// prefix from SUPABASE_URL for connect_to_host
	String host = SUPABASE_URL.replace("https://", "").replace("http://", "");
	Error err = http->connect_to_host(host, 443, TLSOptions::client());
	if (err != OK) {
		print_error("Failed to connect to Supabase for email sign-up");
		return;
	}
	
	// Wait for connection
	while (http->get_status() == HTTPClient::STATUS_CONNECTING || http->get_status() == HTTPClient::STATUS_RESOLVING) {
		http->poll();
		OS::get_singleton()->delay_usec(10000);
	}
	
	if (http->get_status() != HTTPClient::STATUS_CONNECTED) {
		print_error("Could not connect to Supabase");
		return;
	}
	
	// Prepare request body with user metadata
	Dictionary user_metadata;
	user_metadata["name"] = p_name;
	user_metadata["source"] = "desktop_app";
	
	Dictionary body_dict;
	body_dict["email"] = p_email;
	body_dict["password"] = p_password;
	body_dict["data"] = user_metadata;
	
	String body_json = JSON::stringify(body_dict);
	
	// Prepare headers
	Vector<String> headers;
	headers.push_back("Content-Type: application/json");
	headers.push_back("apikey: " + SUPABASE_ANON_KEY);
	
	// Make POST request to /auth/v1/signup
	CharString body_utf8 = body_json.utf8();
	err = http->request(HTTPClient::METHOD_POST, "/auth/v1/signup", headers, (const uint8_t *)body_utf8.get_data(), body_utf8.length());
	
	if (err != OK) {
		print_error("Failed to make sign-up request");
		return;
	}
	
	// Wait for response
	while (http->get_status() == HTTPClient::STATUS_REQUESTING) {
		http->poll();
		OS::get_singleton()->delay_usec(10000);
	}
	
	// Read response
	if (http->has_response()) {
		PackedByteArray rb;
		while (http->get_status() == HTTPClient::STATUS_BODY) {
			http->poll();
			PackedByteArray chunk = http->read_response_body_chunk();
			if (chunk.size() == 0) {
				OS::get_singleton()->delay_usec(10000);
			} else {
				rb.append_array(chunk);
			}
		}
		
		String response_text = String::utf8((const char *)rb.ptr(), rb.size());
		print_line("Email sign-up response: " + response_text);
		
		// Check HTTP status code
		int status_code = http->get_response_code();
		if (status_code >= 400) {
			// Parse error response
			JSON json;
			Error parse_err = json.parse(response_text);
			if (parse_err == OK) {
				Dictionary error_response = json.get_data();
				String error_msg = "Sign-up failed";
				
				// Handle different error response formats
				if (error_response.has("msg")) {
					error_msg = error_response["msg"];
				} else if (error_response.has("error_code")) {
					String error_code = error_response["error_code"];
					error_msg = error_response.get("msg", error_code);
				} else if (error_response.has("error")) {
					error_msg = error_response.get("error_description", error_response.get("error", "Unknown error"));
				} else {
					error_msg = "Sign-up failed (HTTP " + String::num(status_code) + ")";
				}
				
				print_error("Sign-up failed: " + error_msg);
				_notify_auth_error("Sign-up failed: " + error_msg);
				return;
			} else {
				_notify_auth_error("Sign-up failed (HTTP " + String::num(status_code) + ")");
				return;
			}
		}
		
		// Parse JSON response
		JSON json;
		Error parse_err = json.parse(response_text);
		if (parse_err != OK) {
			print_error("Failed to parse sign-up response");
			_notify_auth_error("Invalid response from server");
			return;
		}
		
		Dictionary response = json.get_data();
		
		// Check for error in response (legacy format)
		if (response.has("error")) {
			String error_msg = response.get("error_description", response.get("error", "Unknown error"));
			print_error("Sign-up failed: " + error_msg);
			_notify_auth_error("Sign-up failed: " + error_msg);
			return;
		}
		
		// Check if email confirmation is required
		if (!response.has("session") || response["session"] == Variant()) {
			print_line("Account created! Please check your email to confirm your account.");
			_notify_auth_info("Account created successfully!\n\nPlease check your email (" + p_email + ") and click the confirmation link.\n\nAfter confirming, you can sign in with your email and password.");
			return;
		}
		
		// Auto-login if no email confirmation needed
		Dictionary session = response["session"];
		if (session.has("access_token") && session.has("refresh_token")) {
			String new_access_token = session["access_token"];
			String new_refresh_token = session["refresh_token"];
			
			Dictionary user_data = response.get("user", Dictionary());
			String new_user_id = user_data.get("id", "");
			String new_email = user_data.get("email", p_email);
			
			// Store tokens and mark as authenticated
			store_tokens(new_access_token, new_refresh_token, new_user_id, new_email, p_name);
			is_authenticated = true;
			
			print_line("Email sign-up and auto-login successful!");
			
			// Initialize Autumn account after successful sign-up
			initialize_autumn_account();
			
			_notify_auth_success();
		} else {
			_notify_auth_error("Sign-up succeeded but auto-login failed");
		}
	} else {
		_notify_auth_error("Network error during sign-up");
	}
}

void AuthManager::store_tokens(const String &p_access_token, const String &p_refresh_token, const String &p_user_id, const String &p_email, const String &p_name) {
	access_token = p_access_token;
	refresh_token = p_refresh_token;
	user_id = p_user_id;
	user_email = p_email;
	user_name = p_name;

	// Store securely in keychain
	_store_token_secure("orca_access_token", p_access_token);
	_store_token_secure("orca_refresh_token", p_refresh_token);
	_store_token_secure("orca_user_id", p_user_id);
	_store_token_secure("orca_user_email", p_email);
	_store_token_secure("orca_user_name", p_name);

	emit_signal("auth_state_changed");
}

bool AuthManager::load_stored_tokens() {
	access_token = _retrieve_token_secure("orca_access_token");
	refresh_token = _retrieve_token_secure("orca_refresh_token");
	user_id = _retrieve_token_secure("orca_user_id");
	user_email = _retrieve_token_secure("orca_user_email");
	user_name = _retrieve_token_secure("orca_user_name");

	return !access_token.is_empty() && !refresh_token.is_empty() && !user_id.is_empty();
}

void AuthManager::clear_stored_tokens() {
	_delete_token_secure("orca_access_token");
	_delete_token_secure("orca_refresh_token");
	_delete_token_secure("orca_user_id");
	_delete_token_secure("orca_user_email");
	_delete_token_secure("orca_user_name");
}

// ===== UI NOTIFICATION METHODS =====

void AuthManager::_notify_auth_success() {
	if (auth_dialog) {
		auth_dialog->show_success();
	}
}

void AuthManager::_notify_auth_error(const String &p_message) {
	if (auth_dialog) {
		auth_dialog->show_error(p_message);
	} else {
		print_error("Auth error (no dialog): " + p_message);
	}
}

void AuthManager::_notify_auth_info(const String &p_message) {
	if (auth_dialog) {
		auth_dialog->show_info(p_message);
	} else {
		print_line("Auth info (no dialog): " + p_message);
	}
}

void AuthManager::poll_loopback_server() {
	if (!loopback_server.is_valid()) {
		return;
	}

	if (!loopback_server->is_listening()) {
		_stop_loopback_server();
		return;
	}

	if (loopback_deadline_msec > 0 && OS::get_singleton()->get_ticks_msec() > loopback_deadline_msec) {
		print_line("ORCA AUTH: Loopback login timed out, closing listener");
		_stop_loopback_server();
		return;
	}

	if (!loopback_server->is_connection_available()) {
		return;
	}

	Ref<StreamPeerTCP> client = loopback_server->take_connection();
	if (client.is_null()) {
		return;
	}

	_handle_loopback_connection(client);
}

String AuthManager::_get_secure_storage_key(const String &p_key) const {
	return "ai.orcaengine." + p_key;
}

String AuthManager::_get_auth_storage_dir() const {
#ifdef MACOS_ENABLED
	return OS::get_singleton()->get_config_path().path_join("Orca").path_join("auth");
#elif defined(LINUXBSD_ENABLED)
	return OS::get_singleton()->get_config_path().path_join("Orca").path_join("auth");
#else
	return OS::get_singleton()->get_user_data_dir().path_join("auth");
#endif
}

String AuthManager::_get_legacy_auth_storage_dir() const {
	return OS::get_singleton()->get_user_data_dir().path_join("auth");
}

void AuthManager::_ensure_auth_storage_dir_exists(const String &p_dir) const {
	Ref<DirAccess> dir = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
	if (dir.is_valid() && !dir->dir_exists(p_dir)) {
		dir->make_dir_recursive(p_dir);
	}
}

void AuthManager::_remove_token_file(const String &p_dir, const String &p_key) const {
	if (p_dir.is_empty()) {
		return;
	}

	String file_path = p_dir.path_join(_get_secure_storage_key(p_key));
	if (FileAccess::exists(file_path)) {
		Ref<DirAccess> dir = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
		if (dir.is_valid()) {
			dir->remove(file_path);
		}
	}
}

#ifdef MACOS_ENABLED
// Simple file-based storage - no keychain prompts!
// Stored in ~/Library/Application Support/Orca/auth/
bool AuthManager::_store_token_secure(const String &p_key, const String &p_value) {
	String auth_dir = _get_auth_storage_dir();
	_ensure_auth_storage_dir_exists(auth_dir);
	FileAccess::set_unix_permissions(auth_dir,
			FileAccess::UNIX_READ_OWNER | FileAccess::UNIX_WRITE_OWNER | FileAccess::UNIX_EXECUTE_OWNER);
	
	// Store in file
	String file_path = auth_dir.path_join(_get_secure_storage_key(p_key));
	Ref<FileAccess> file = FileAccess::open(file_path, FileAccess::WRITE);
	if (!file.is_valid()) {
		print_error("Failed to store token in file: " + file_path);
		return false;
	}
	
	file->store_string(p_value);
	file->close();

	FileAccess::set_unix_permissions(file_path, FileAccess::UNIX_READ_OWNER | FileAccess::UNIX_WRITE_OWNER);

	// Remove any legacy copies once we successfully write to the new location.
	String legacy_dir = _get_legacy_auth_storage_dir();
	if (legacy_dir != auth_dir) {
		_remove_token_file(legacy_dir, p_key);
	}

	return true;
}

String AuthManager::_retrieve_token_secure(const String &p_key) {
	String auth_dir = _get_auth_storage_dir();
	String file_path = auth_dir.path_join(_get_secure_storage_key(p_key));

	if (!FileAccess::exists(file_path)) {
		// Attempt legacy location migration.
		String legacy_path = _get_legacy_auth_storage_dir().path_join(_get_secure_storage_key(p_key));
		if (FileAccess::exists(legacy_path)) {
			String legacy_value;
			Ref<FileAccess> legacy_file = FileAccess::open(legacy_path, FileAccess::READ);
			if (legacy_file.is_valid()) {
				legacy_value = legacy_file->get_as_text();
				legacy_file->close();
			}

			if (!legacy_value.is_empty()) {
				_store_token_secure(p_key, legacy_value);
				return legacy_value;
			}
		}

		return String();
	}
	
	Ref<FileAccess> file = FileAccess::open(file_path, FileAccess::READ);
	if (!file.is_valid()) {
		return String();
	}
	
	String value = file->get_as_text();
	file->close();
	
	return value;
}

void AuthManager::_delete_token_secure(const String &p_key) {
	_remove_token_file(_get_auth_storage_dir(), p_key);

	String legacy_dir = _get_legacy_auth_storage_dir();
	if (legacy_dir != _get_auth_storage_dir()) {
		_remove_token_file(legacy_dir, p_key);
	}
}
#endif

#ifdef WINDOWS_ENABLED
bool AuthManager::_store_token_secure(const String &p_key, const String &p_value) {
	String target_name = _get_secure_storage_key(p_key);
	
	CREDENTIALW cred = { 0 };
	cred.Type = CRED_TYPE_GENERIC;
	cred.TargetName = (LPWSTR)target_name.utf16().get_data();
	cred.CredentialBlobSize = p_value.utf8().length();
	cred.CredentialBlob = (LPBYTE)p_value.utf8().get_data();
	cred.Persist = CRED_PERSIST_LOCAL_MACHINE;
	cred.UserName = (LPWSTR)L"Orca";
	
	BOOL result = CredWriteW(&cred, 0);
	if (!result) {
		print_error("Failed to store token in Windows Credential Manager");
		return false;
	}
	
	return true;
}

String AuthManager::_retrieve_token_secure(const String &p_key) {
	String target_name = _get_secure_storage_key(p_key);
	
	PCREDENTIALW cred = nullptr;
	BOOL result = CredReadW((LPCWSTR)target_name.utf16().get_data(), CRED_TYPE_GENERIC, 0, &cred);
	
	if (!result || !cred) {
		return String();
	}
	
	String value = String::utf8((const char *)cred->CredentialBlob, cred->CredentialBlobSize);
	CredFree(cred);
	
	return value;
}

void AuthManager::_delete_token_secure(const String &p_key) {
	String target_name = _get_secure_storage_key(p_key);
	CredDeleteW((LPCWSTR)target_name.utf16().get_data(), CRED_TYPE_GENERIC, 0);
}
#endif

#ifdef LINUXBSD_ENABLED
// Simple file-based storage for Linux
// In production, you should use libsecret
bool AuthManager::_store_token_secure(const String &p_key, const String &p_value) {
	String secure_dir = _get_auth_storage_dir();
	_ensure_auth_storage_dir_exists(secure_dir);
	FileAccess::set_unix_permissions(secure_dir,
			FileAccess::UNIX_READ_OWNER | FileAccess::UNIX_WRITE_OWNER | FileAccess::UNIX_EXECUTE_OWNER);
	
	String file_path = secure_dir.path_join(_get_secure_storage_key(p_key));
	Ref<FileAccess> file = FileAccess::open(file_path, FileAccess::WRITE);
	if (file.is_null()) {
		print_error("Failed to store token in file: " + file_path);
		return false;
	}
	
	file->store_string(p_value);
	file->close();

	FileAccess::set_unix_permissions(file_path, FileAccess::UNIX_READ_OWNER | FileAccess::UNIX_WRITE_OWNER);
	
	return true;
}

String AuthManager::_retrieve_token_secure(const String &p_key) {
	String file_path = _get_auth_storage_dir().path_join(_get_secure_storage_key(p_key));
	
	if (!FileAccess::exists(file_path)) {
		String legacy_path = _get_legacy_auth_storage_dir().path_join(_get_secure_storage_key(p_key));
		if (FileAccess::exists(legacy_path)) {
			String legacy_value;
			Ref<FileAccess> legacy_file = FileAccess::open(legacy_path, FileAccess::READ);
			if (legacy_file.is_valid()) {
				legacy_value = legacy_file->get_as_text();
				legacy_file->close();
			}
			if (!legacy_value.is_empty()) {
				_store_token_secure(p_key, legacy_value);
				return legacy_value;
			}
		}
		return String();
	}
	
	Ref<FileAccess> file = FileAccess::open(file_path, FileAccess::READ);
	if (file.is_null()) {
		return String();
	}
	
	String value = file->get_as_text();
	file->close();
	
	return value;
}

void AuthManager::_delete_token_secure(const String &p_key) {
	String file_path = _get_auth_storage_dir().path_join(_get_secure_storage_key(p_key));
	
	if (FileAccess::exists(file_path)) {
		Ref<DirAccess> dir = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
		if (dir.is_valid()) {
			dir->remove(file_path);
		}
	}
}
#endif

Error AuthManager::make_supabase_request(const String &p_endpoint, const String &p_method, const String &p_body, String &r_response) {
	Ref<HTTPClient> http_client = HTTPClient::create();
	
	// Parse the Supabase URL - remove https:// prefix for connect_to_host
	String host = SUPABASE_URL.replace("https://", "").replace("http://", "");
	Error err = http_client->connect_to_host(host, 443, TLSOptions::client());
	if (err != OK) {
		print_error("Failed to connect to Supabase: " + itos(err));
		return err;
	}
	
	// Wait for connection
	while (http_client->get_status() == HTTPClient::STATUS_CONNECTING || 
	       http_client->get_status() == HTTPClient::STATUS_RESOLVING) {
		http_client->poll();
		OS::get_singleton()->delay_usec(10000);
	}
	
	if (http_client->get_status() != HTTPClient::STATUS_CONNECTED) {
		print_error("Could not connect to Supabase");
		return FAILED;
	}
	
	// Prepare headers
	Vector<String> headers;
	headers.push_back("apikey: " + SUPABASE_ANON_KEY);
	headers.push_back("Content-Type: application/json");
	
	if (!access_token.is_empty()) {
		headers.push_back("Authorization: Bearer " + access_token);
	}
	
	// Make request
	HTTPClient::Method method = HTTPClient::METHOD_GET;
	if (p_method == "POST") {
		method = HTTPClient::METHOD_POST;
	} else if (p_method == "PUT") {
		method = HTTPClient::METHOD_PUT;
	} else if (p_method == "DELETE") {
		method = HTTPClient::METHOD_DELETE;
	}
	
	// Convert body string to byte array
	CharString body_utf8 = p_body.utf8();
	err = http_client->request(method, p_endpoint, headers, (const uint8_t *)body_utf8.get_data(), body_utf8.length());
	if (err != OK) {
		print_error("Failed to make request: " + itos(err));
		return err;
	}
	
	// Wait for response
	while (http_client->get_status() == HTTPClient::STATUS_REQUESTING) {
		http_client->poll();
		OS::get_singleton()->delay_usec(10000);
	}
	
	if (http_client->get_status() != HTTPClient::STATUS_BODY && 
	    http_client->get_status() != HTTPClient::STATUS_CONNECTED) {
		print_error("Request failed with status: " + itos(http_client->get_status()));
		return FAILED;
	}
	
	// Read response
	if (http_client->has_response()) {
		PackedByteArray rb;
		while (http_client->get_status() == HTTPClient::STATUS_BODY) {
			http_client->poll();
			PackedByteArray chunk = http_client->read_response_body_chunk();
			if (chunk.size() == 0) {
				OS::get_singleton()->delay_usec(10000);
			} else {
				rb.append_array(chunk);
			}
		}
		
		r_response = String::utf8((const char *)rb.ptr(), rb.size());
	}
	
	return OK;
}

void AuthManager::initialize_autumn_account() {
	// Only initialize if user is authenticated
	if (!is_authenticated || user_id.is_empty()) {
		print_line("ORCA AUTUMN: Skipping initialization - user not authenticated");
		return;
	}
	
	print_line("ORCA AUTUMN: Initializing Autumn account for user: " + user_id);
	
	// Call website API directly: https://orcaengine.ai/api/autumn/customer
	// This creates customer + assigns Free plan if new user
	Ref<HTTPClient> http = HTTPClient::create();
	Error err = http->connect_to_host("orcaengine.ai", 443, TLSOptions::client());
	if (err != OK) {
		print_error("ORCA AUTUMN: Failed to connect to website API");
		return;
	}
	
	// Wait for connection
	while (http->get_status() == HTTPClient::STATUS_CONNECTING || http->get_status() == HTTPClient::STATUS_RESOLVING) {
		http->poll();
		OS::get_singleton()->delay_usec(10000);
	}
	
	if (http->get_status() != HTTPClient::STATUS_CONNECTED) {
		print_error("ORCA AUTUMN: Could not connect to website API");
		return;
	}
	
	// Prepare headers - send user_id as Bearer token (as per website API spec)
	Vector<String> headers;
	headers.push_back("Content-Type: application/json");
	headers.push_back("Authorization: Bearer " + user_id);
	if (!user_email.is_empty()) {
		headers.push_back("X-User-Email: " + user_email);
	}
	
	// Prepare request body for /check endpoint (auto-creates customer with default Free plan)
	Dictionary body_dict;
	body_dict["feature_id"] = "ai-requests";
	body_dict["required_quantity"] = 0;  // Just checking, not consuming quota
	String body_json = JSON::stringify(body_dict);
	CharString body_utf8 = body_json.utf8();
	
	// Make POST request to /api/autumn/check (auto-creates customer if new)
	err = http->request(HTTPClient::METHOD_POST, "/api/autumn/check", headers, (const uint8_t *)body_utf8.get_data(), body_utf8.length());
	
	if (err != OK) {
		print_error("ORCA AUTUMN: Failed to make initialization request");
		return;
	}
	
	// Wait for response
	while (http->get_status() == HTTPClient::STATUS_REQUESTING) {
		http->poll();
		OS::get_singleton()->delay_usec(10000);
	}
	
	// Read response
	if (http->has_response()) {
		PackedByteArray rb;
		while (http->get_status() == HTTPClient::STATUS_BODY) {
			http->poll();
			PackedByteArray chunk = http->read_response_body_chunk();
			if (chunk.size() == 0) {
				OS::get_singleton()->delay_usec(10000);
			} else {
				rb.append_array(chunk);
			}
		}
		
		String response_text = String::utf8((const char *)rb.ptr(), rb.size());
		
		// Parse JSON response from /check endpoint
		JSON json;
		Error parse_err = json.parse(response_text);
		if (parse_err == OK) {
			Dictionary check_response = json.get_data();
			if (!check_response.has("error")) {
				print_line("ORCA AUTUMN: Successfully initialized Autumn account");
				// Log quota info from /check response
				if (check_response.has("balance")) {
					int balance = check_response.get("balance", 0);
					int included = check_response.get("included_usage", 0);
					bool allowed = check_response.get("allowed", false);
					print_line(vformat("ORCA AUTUMN: Balance: %d/%d AI requests, Allowed: %s", balance, included, allowed ? "Yes" : "No"));
				}
			} else {
				String error_msg = check_response.get("error", "Unknown error");
				print_line("ORCA AUTUMN: Initialization failed: " + error_msg);
			}
		} else {
			print_line("ORCA AUTUMN: Failed to parse response: " + response_text.substr(0, 200));
		}
	} else {
		print_line("ORCA AUTUMN: No response from website API");
	}
}

bool AuthManager::_start_loopback_server() {
	_stop_loopback_server();

	loopback_server.instantiate();
	IPAddress bind_addr("127.0.0.1");

	for (int port = 42100; port <= 42200; port++) {
		if (loopback_server->listen(port, bind_addr) == OK) {
			loopback_port = port;
			loopback_deadline_msec = OS::get_singleton()->get_ticks_msec() + 120000; // 2 minutes
			return true;
		}
	}

	print_error("ORCA AUTH: Unable to start loopback server on localhost");
	loopback_server.unref();
	loopback_port = 0;
	loopback_deadline_msec = 0;
	return false;
}

void AuthManager::_stop_loopback_server() {
	if (loopback_server.is_valid()) {
		loopback_server->stop();
		loopback_server.unref();
	}

	loopback_port = 0;
	loopback_deadline_msec = 0;
}

String AuthManager::_read_http_request(const Ref<StreamPeerTCP> &p_client) {
	String request;
	const uint64_t start = OS::get_singleton()->get_ticks_msec();

	while (OS::get_singleton()->get_ticks_msec() - start < 5000) {
		int available = p_client->get_available_bytes();
		if (available > 0) {
			PackedByteArray buffer;
			Error resize_err = buffer.resize(available);
			if (resize_err != OK) {
				break;
			}
			Error read_err = p_client->get_data(buffer.ptrw(), available);
			if (read_err != OK) {
				break;
			}
			request += String::utf8((const char *)buffer.ptr(), buffer.size());

			if (request.find("\r\n\r\n") != -1 || request.find("\n\n") != -1) {
				break;
			}
		} else {
			OS::get_singleton()->delay_usec(1000);
		}
	}

	return request;
}

void AuthManager::_send_loopback_response(const Ref<StreamPeerTCP> &p_client, bool p_success) {
	String body;
	body = p_success
			? "<html><body><h2>Login successful</h2><p>You can close this tab.</p></body></html>"
			: "<html><body><h2>Login failed</h2><p>Please try again.</p></body></html>";

	PackedByteArray body_bytes = body.to_utf8_buffer();
	String header = vformat("HTTP/1.1 %s\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: %d\r\nConnection: close\r\n\r\n",
			p_success ? "200 OK" : "400 Bad Request",
			body_bytes.size());

	PackedByteArray header_bytes = header.to_utf8_buffer();
	p_client->put_data(header_bytes.ptr(), header_bytes.size());
	p_client->put_data(body_bytes.ptr(), body_bytes.size());
	p_client->disconnect_from_host();
}

bool AuthManager::_handle_loopback_connection(const Ref<StreamPeerTCP> &p_client) {
	String request = _read_http_request(p_client);
	bool success = false;

	if (!request.is_empty()) {
		int line_end = request.find("\r\n");
		if (line_end == -1) {
			line_end = request.find("\n");
		}

		if (line_end != -1) {
			String first_line = request.substr(0, line_end);
			PackedStringArray parts = first_line.split(" ");
			if (parts.size() >= 2) {
				String method = parts[0];
				String path = parts[1];
				if (method == "GET" && path.begins_with("/auth/callback")) {
					int query_pos = path.find("?");
					if (query_pos != -1 && query_pos + 1 < path.length()) {
						String query = path.substr(query_pos + 1);
						success = handle_deep_link("orca://auth?" + query);
					}
				}
			}
		}
	}

	_send_loopback_response(p_client, success);
	return success;
}

