/**************************************************************************/
/*  git_credentials.cpp                                                   */
/**************************************************************************/

#include "git_credentials.h"
#include "git_manager.h"

const String GitCredentials::SETTINGS_PREFIX = "git_integration/";

GitCredentials::Credentials GitCredentials::load_credentials() {
	Credentials creds;
	
	EditorSettings *settings = EditorSettings::get_singleton();
	if (!settings) {
		return creds;
	}
	
	String username_key = SETTINGS_PREFIX + "username";
	creds.username = settings->has_setting(username_key) ? settings->get_setting(username_key) : "";
	
	String email_key = SETTINGS_PREFIX + "email";
	creds.email = settings->has_setting(email_key) ? settings->get_setting(email_key) : "";
	
	String token_key = SETTINGS_PREFIX + "github_token_encrypted";
	creds.github_token = settings->has_setting(token_key) ? _decrypt_token(settings->get_setting(token_key)) : "";
	
	String use_token_key = SETTINGS_PREFIX + "use_token";
	creds.use_token = settings->has_setting(use_token_key) ? (bool)settings->get_setting(use_token_key) : true;
	
	String save_key = SETTINGS_PREFIX + "save_credentials";
	creds.save_credentials = settings->has_setting(save_key) ? (bool)settings->get_setting(save_key) : true;
	
	return creds;
}

void GitCredentials::save_credentials(const Credentials &p_credentials) {
	EditorSettings *settings = EditorSettings::get_singleton();
	if (!settings) {
		return;
	}
	
	if (p_credentials.save_credentials) {
		settings->set_setting(SETTINGS_PREFIX + "username", p_credentials.username);
		settings->set_setting(SETTINGS_PREFIX + "email", p_credentials.email);
		settings->set_setting(SETTINGS_PREFIX + "github_token_encrypted", _encrypt_token(p_credentials.github_token));
		settings->set_setting(SETTINGS_PREFIX + "use_token", p_credentials.use_token);
		settings->set_setting(SETTINGS_PREFIX + "save_credentials", p_credentials.save_credentials);
	} else {
		// Clear saved credentials if user doesn't want them saved
		clear_credentials();
	}
	
	settings->save();
}

void GitCredentials::clear_credentials() {
	EditorSettings *settings = EditorSettings::get_singleton();
	if (!settings) {
		return;
	}
	
	settings->erase(SETTINGS_PREFIX + "username");
	settings->erase(SETTINGS_PREFIX + "email");
	settings->erase(SETTINGS_PREFIX + "github_token_encrypted");
	settings->erase(SETTINGS_PREFIX + "use_token");
	settings->save();
}

bool GitCredentials::validate_github_token(const String &p_token) {
	// Basic validation - GitHub tokens start with specific prefixes
	return p_token.length() >= 20 && 
		   (p_token.begins_with("ghp_") || 
			p_token.begins_with("github_pat_") ||
			p_token.begins_with("gho_") ||
			p_token.begins_with("ghu_"));
}

bool GitCredentials::has_valid_credentials() {
	Credentials creds = load_credentials();
	return !creds.username.is_empty() && !creds.email.is_empty() && 
		   (creds.use_token ? validate_github_token(creds.github_token) : true);
}

void GitCredentials::configure_git_user(const String &p_project_path, const String &p_name, const String &p_email) {
	// Set Git user name
	List<String> name_args;
	name_args.push_back("config");
	name_args.push_back("user.name");
	name_args.push_back(p_name);
	GitManager::execute_git_command(p_project_path, name_args);
	
	// Set Git user email
	List<String> email_args;
	email_args.push_back("config");
	email_args.push_back("user.email");
	email_args.push_back(p_email);
	GitManager::execute_git_command(p_project_path, email_args);
}

void GitCredentials::configure_git_credentials(const String &p_project_path, const Credentials &p_credentials) {
	configure_git_user(p_project_path, p_credentials.username, p_credentials.email);
	
	if (p_credentials.use_token && !p_credentials.github_token.is_empty()) {
		// Configure Git to use token authentication for GitHub
		List<String> cred_args;
		cred_args.push_back("config");
		cred_args.push_back("credential.helper");
		cred_args.push_back("store");
		GitManager::execute_git_command(p_project_path, cred_args);
	}
}

String GitCredentials::extract_github_repo_from_url(const String &p_url) {
	// Extract username/repo from GitHub URLs
	String url = p_url.to_lower();
	if (url.contains("github.com")) {
		// Handle both HTTPS and SSH URLs
		String path;
		if (url.begins_with("https://github.com/")) {
			path = p_url.substr(19); // Remove "https://github.com/"
		} else if (url.begins_with("git@github.com:")) {
			path = p_url.substr(15); // Remove "git@github.com:"
		}
		
		if (path.ends_with(".git")) {
			path = path.substr(0, path.length() - 4);
		}
		
		return path;
	}
	return "";
}

String GitCredentials::create_github_https_url(const String &p_username, const String &p_repo) {
	return "https://github.com/" + p_username + "/" + p_repo + ".git";
}

String GitCredentials::create_github_token_url(const String &p_token, const String &p_username, const String &p_repo) {
	return "https://" + p_token + "@github.com/" + p_username + "/" + p_repo + ".git";
}

String GitCredentials::_encrypt_token(const String &p_token) {
	// Simple XOR encryption with the user's system identifier
	if (p_token.is_empty()) {
		return "";
	}
	
	String system_id = OS::get_singleton()->get_unique_id();
	String encrypted = "";
	
	for (int i = 0; i < p_token.length(); i++) {
		char32_t c = p_token[i];
		char32_t key = system_id[i % system_id.length()];
		encrypted += String::chr(c ^ key);
	}
	
	PackedByteArray bytes = encrypted.to_utf8_buffer();
	return String::hex_encode_buffer(bytes.ptr(), bytes.size());
}

String GitCredentials::_decrypt_token(const String &p_encrypted_token) {
	// Decrypt XOR encrypted token
	if (p_encrypted_token.is_empty()) {
		return "";
	}
	
	PackedByteArray bytes = p_encrypted_token.hex_decode();
	if (bytes.is_empty()) {
		return "";
	}
	
	String system_id = OS::get_singleton()->get_unique_id();
	String decrypted = "";
	
	for (int i = 0; i < bytes.size(); i++) {
		char32_t c = bytes[i];
		char32_t key = system_id[i % system_id.length()];
		decrypted += String::chr(c ^ key);
	}
	
	return decrypted;
}
