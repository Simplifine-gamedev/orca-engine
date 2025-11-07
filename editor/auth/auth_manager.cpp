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
}

AuthManager::AuthManager() {
	singleton = this;
}

AuthManager::~AuthManager() {
	singleton = nullptr;
}

void AuthManager::open_web_login() {
	print_line("Opening web login at: " + LOGIN_URL);
	OS::get_singleton()->shell_open(LOGIN_URL);
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
	print_line("User signed out");
}

void AuthManager::sign_in_with_email(const String &p_email, const String &p_password) {
	print_line("AuthManager: Signing in with email: " + p_email);
	
	// Make HTTP request to Supabase auth endpoint
	Ref<HTTPClient> http = HTTPClient::create();
	Error err = http->connect_to_host(SUPABASE_URL, 443, TLSOptions::client());
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
				Dictionary metadata = user_data["user_metadata"];
				new_name = metadata.get("name", p_email.get_slice("@", 0));
			} else {
				new_name = p_email.get_slice("@", 0);
			}
			
			// Store tokens and mark as authenticated
			store_tokens(new_access_token, new_refresh_token, new_user_id, new_email, new_name);
			is_authenticated = true;
			
			print_line("Email sign-in successful!");
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
	Error err = http->connect_to_host(SUPABASE_URL, 443, TLSOptions::client());
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
	Dictionary metadata;
	metadata["name"] = p_name;
	metadata["source"] = "desktop_app";
	
	Dictionary body_dict;
	body_dict["email"] = p_email;
	body_dict["password"] = p_password;
	body_dict["data"] = metadata;
	
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

String AuthManager::_get_secure_storage_key(const String &p_key) const {
	return "ai.orcaengine." + p_key;
}

#ifdef MACOS_ENABLED
// Simple file-based storage - no keychain prompts!
// Stored in ~/Library/Application Support/Orca/auth/
bool AuthManager::_store_token_secure(const String &p_key, const String &p_value) {
	String config_dir = OS::get_singleton()->get_user_data_dir();
	String auth_dir = config_dir.path_join("auth");
	
	// Create auth directory if it doesn't exist
	Ref<DirAccess> dir = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
	if (dir.is_valid() && !dir->dir_exists(auth_dir)) {
		dir->make_dir_recursive(auth_dir);
	}
	
	// Store in file
	String file_path = auth_dir.path_join(_get_secure_storage_key(p_key));
	Ref<FileAccess> file = FileAccess::open(file_path, FileAccess::WRITE);
	if (!file.is_valid()) {
		print_error("Failed to store token in file: " + file_path);
		return false;
	}
	
	file->store_string(p_value);
	file->close();
	
	return true;
}

String AuthManager::_retrieve_token_secure(const String &p_key) {
	String config_dir = OS::get_singleton()->get_user_data_dir();
	String file_path = config_dir.path_join("auth").path_join(_get_secure_storage_key(p_key));
	
	if (!FileAccess::exists(file_path)) {
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
	String config_dir = OS::get_singleton()->get_user_data_dir();
	String file_path = config_dir.path_join("auth").path_join(_get_secure_storage_key(p_key));
	
	if (FileAccess::exists(file_path)) {
		Ref<DirAccess> dir = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
		if (dir.is_valid()) {
			dir->remove(file_path);
		}
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
	String config_dir = OS::get_singleton()->get_user_data_dir();
	String secure_dir = config_dir.path_join(".orca_auth");
	
	DirAccess *dir = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
	if (!dir->dir_exists(secure_dir)) {
		dir->make_dir(secure_dir);
	}
	memdelete(dir);
	
	String file_path = secure_dir.path_join(_get_secure_storage_key(p_key));
	FileAccess *file = FileAccess::open(file_path, FileAccess::WRITE);
	if (!file) {
		print_error("Failed to store token in file: " + file_path);
		return false;
	}
	
	file->store_string(p_value);
	file->close();
	memdelete(file);
	
	return true;
}

String AuthManager::_retrieve_token_secure(const String &p_key) {
	String config_dir = OS::get_singleton()->get_user_data_dir();
	String file_path = config_dir.path_join(".orca_auth").path_join(_get_secure_storage_key(p_key));
	
	if (!FileAccess::exists(file_path)) {
		return String();
	}
	
	FileAccess *file = FileAccess::open(file_path, FileAccess::READ);
	if (!file) {
		return String();
	}
	
	String value = file->get_as_text();
	file->close();
	memdelete(file);
	
	return value;
}

void AuthManager::_delete_token_secure(const String &p_key) {
	String config_dir = OS::get_singleton()->get_user_data_dir();
	String file_path = config_dir.path_join(".orca_auth").path_join(_get_secure_storage_key(p_key));
	
	if (FileAccess::exists(file_path)) {
		DirAccess *dir = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
		dir->remove(file_path);
		memdelete(dir);
	}
}
#endif

Error AuthManager::make_supabase_request(const String &p_endpoint, const String &p_method, const String &p_body, String &r_response) {
	Ref<HTTPClient> http_client = HTTPClient::create();
	
	// Parse the Supabase URL
	Error err = http_client->connect_to_host(SUPABASE_URL, 443, TLSOptions::client());
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

