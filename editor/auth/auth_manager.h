/**************************************************************************/
/*  auth_manager.h                                                        */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             ORCA ENGINE                                */
/**************************************************************************/

#ifndef AUTH_MANAGER_H
#define AUTH_MANAGER_H

#include "core/object/object.h"
#include "core/object/ref_counted.h"
#include "core/string/ustring.h"
#include "core/templates/hash_map.h"
#include "core/io/http_client.h"

class AuthManager : public Object {
	GDCLASS(AuthManager, Object);

private:
	static AuthManager *singleton;

	// Authentication state
	bool is_authenticated = false;
	String access_token;
	String refresh_token;
	String user_id;
	String user_email;
	String user_name;
	
	// UI reference
	class AuthDialog *auth_dialog = nullptr;
	
	// Supabase configuration
	const String SUPABASE_URL = "https://ximaifnuynbswxtnpiid.supabase.co";
	const String SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InhpbWFpZm51eW5ic3d4dG5waWlkIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTU2NzYwNTcsImV4cCI6MjA3MTI1MjA1N30.B27Tmf0xD13quZOPb1jBD-m5MIq1ZKbJJFNAVCi6_s0";
	const String LOGIN_URL = "https://orcaengine.ai/desktop-login.html";

	// Secure storage paths
	String _get_secure_storage_key(const String &p_key) const;
	bool _store_token_secure(const String &p_key, const String &p_value);
	String _retrieve_token_secure(const String &p_key);
	void _delete_token_secure(const String &p_key);

protected:
	static void _bind_methods();

public:
	static AuthManager *get_singleton() { return singleton; }

	// Authentication flow
	void open_web_login(); // Google OAuth (opens browser)
	bool handle_deep_link(const String &p_url);
	bool try_auto_login();
	void sign_out();
	
	// Email/Password authentication (native)
	void sign_in_with_email(const String &p_email, const String &p_password);
	void sign_up_with_email(const String &p_email, const String &p_password, const String &p_name);
	
	// UI callback handling
	void set_auth_dialog(class AuthDialog *p_dialog) { auth_dialog = p_dialog; }
	void _notify_auth_success();
	void _notify_auth_error(const String &p_message);
	void _notify_auth_info(const String &p_message);

	// Token management
	void store_tokens(const String &p_access_token, const String &p_refresh_token, const String &p_user_id, const String &p_email, const String &p_name);
	bool load_stored_tokens();
	void clear_stored_tokens();

	// Getters
	bool get_is_authenticated() const { return is_authenticated; }
	String get_access_token() const { return access_token; }
	String get_refresh_token() const { return refresh_token; }
	String get_user_id() const { return user_id; }
	String get_user_email() const { return user_email; }
	String get_user_name() const { return user_name; }

	// HTTP utilities for Supabase API
	Error make_supabase_request(const String &p_endpoint, const String &p_method, const String &p_body, String &r_response);

	AuthManager();
	~AuthManager();
};

#endif // AUTH_MANAGER_H

