/*
 * © 2025 Simplifine Corp.
 * Personal Non‑Commercial License applies to this file as an original contribution to this Godot fork.
 * See LICENSES/COMPANY-NONCOMMERCIAL.md for terms. Commercial use requires a separate license from Simplifine.
 */

#pragma once

#include "scene/main/http_request.h"
#include "scene/main/timer.h"
#include "scene/gui/button.h"
#include "scene/gui/label.h"

class AIChatDock;

class AIChatDockAuth {
public:
	// Setup authentication UI
	static void setup_authentication_ui(AIChatDock *p_dock);
	
	// Login/logout handlers
	static void on_login_button_pressed(AIChatDock *p_dock);
	static void on_auth_provider_selected(AIChatDock *p_dock, int p_id);
	static void logout_user(AIChatDock *p_dock);
	
	// Authentication request handlers
	static void check_authentication_status(AIChatDock *p_dock);
	static void on_auth_request_completed(AIChatDock *p_dock, int p_result, int p_code, const PackedStringArray &p_headers, const PackedByteArray &p_body);
	static void on_auth_providers_request_completed(AIChatDock *p_dock, int p_result, int p_code, const PackedStringArray &p_headers, const PackedByteArray &p_body);
	static void on_auth_dialog_action(AIChatDock *p_dock, const StringName &p_action);
	
	// Auto-verification and polling
	static void auto_verify_saved_credentials(AIChatDock *p_dock);
	static void start_login_polling(AIChatDock *p_dock);
	static void poll_login_status(AIChatDock *p_dock);
	static void stop_login_polling(AIChatDock *p_dock);
	
	// User status management
	static void update_user_status(AIChatDock *p_dock);
	static bool is_user_authenticated(AIChatDock *p_dock);
	
	// Utility methods
	static String get_machine_id();
};
