/**************************************************************************/
/*  git_credentials.h                                                     */
/**************************************************************************/

#pragma once

#include "core/string/ustring.h"
#include "editor/settings/editor_settings.h"

class GitCredentials {
public:
	struct Credentials {
		String username;
		String email;
		String github_token;
		bool use_token = true; // Prefer token over username/password
		bool save_credentials = true;
	};

	// Load/Save credentials to EditorSettings
	static Credentials load_credentials();
	static void save_credentials(const Credentials &p_credentials);
	static void clear_credentials();
	
	// Credential validation
	static bool validate_github_token(const String &p_token);
	static bool has_valid_credentials();
	
	// Git config management
	static void configure_git_user(const String &p_project_path, const String &p_name, const String &p_email);
	static void configure_git_credentials(const String &p_project_path, const Credentials &p_credentials);
	
	// GitHub URL helpers
	static String extract_github_repo_from_url(const String &p_url);
	static String create_github_https_url(const String &p_username, const String &p_repo);
	static String create_github_token_url(const String &p_token, const String &p_username, const String &p_repo);
	
private:
	static const String SETTINGS_PREFIX;
	static String _encrypt_token(const String &p_token);
	static String _decrypt_token(const String &p_encrypted_token);
};
