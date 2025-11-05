/*
 * © 2025 Simplifine Corp.
 * Personal Non‑Commercial License applies to this file as an original contribution to this Godot fork.
 * See LICENSES/COMPANY-NONCOMMERCIAL.md for terms. Commercial use requires a separate license from Simplifine.
 */

#include "ai_chat_authentication_ui.h"
#include "ai_chat_dock.h"
#include "editor/editor_string_names.h"

void AIChatAuthenticationUI::setup_authentication_ui(VBoxContainer *p_parent, AIChatDock *p_chat_dock) {
	if (!p_parent || !p_chat_dock) {
		return;
	}

	// Toolbar for import/export/snapshot buttons (auth removed - handled by AuthManager)
	HBoxContainer *toolbar_container = _create_toolbar_container(p_chat_dock);
	p_parent->add_child(toolbar_container);
	
	// Import conversation button
	Button *import_button = _create_import_button(p_chat_dock);
	toolbar_container->add_child(import_button);
	p_chat_dock->import_button = import_button;
	
	// Export conversation button  
	Button *export_button = _create_export_button(p_chat_dock);
	toolbar_container->add_child(export_button);
	p_chat_dock->export_button = export_button;
	
	// Manual Snapshot button
	Button *snapshot_button = _create_snapshot_button(p_chat_dock);
	toolbar_container->add_child(snapshot_button);
	p_chat_dock->snapshot_button = snapshot_button;
	
	// Restore Snapshot button
	Button *restore_button = _create_restore_button(p_chat_dock);
	toolbar_container->add_child(restore_button);
	p_chat_dock->restore_snapshot_button = restore_button;
	
	// Create HTTP requests for authentication
	HTTPRequest *auth_request = _create_auth_request(p_chat_dock);
	p_parent->add_child(auth_request);
	p_chat_dock->auth_request = auth_request;

	HTTPRequest *auth_providers_request = _create_auth_providers_request(p_chat_dock);
	p_parent->add_child(auth_providers_request);
	p_chat_dock->auth_providers_request = auth_providers_request;
}

void AIChatAuthenticationUI::update_user_status(Label *p_user_status_label, Button *p_login_button, bool p_is_authenticated, const String &p_user_name, AIChatDock *p_chat_dock) {
	if (!p_user_status_label || !p_login_button || !p_chat_dock) {
		print_line("AI Chat: update_user_status called before UI elements created, skipping");
		return;
	}
	
	if (p_is_authenticated) {
		p_user_status_label->set_text(p_user_name);
		p_login_button->set_text("Logout");
		p_login_button->add_theme_icon_override("icon", p_chat_dock->get_theme_icon(SNAME("Unlock"), SNAME("EditorIcons")));
	} else {
		p_user_status_label->set_text("Not logged in");
		p_login_button->set_text("Login");
		p_login_button->add_theme_icon_override("icon", p_chat_dock->get_theme_icon(SNAME("Key"), SNAME("EditorIcons")));
	}
}

HBoxContainer *AIChatAuthenticationUI::_create_toolbar_container(AIChatDock *p_chat_dock) {
	return memnew(HBoxContainer);
}

Button *AIChatAuthenticationUI::_create_import_button(AIChatDock *p_chat_dock) {
	if (!p_chat_dock) {
		return nullptr;
	}

	Button *import_button = memnew(Button);
	import_button->set_text("Import");
	import_button->add_theme_icon_override("icon", p_chat_dock->get_theme_icon(SNAME("Load"), SNAME("EditorIcons")));
	import_button->connect("pressed", callable_mp(p_chat_dock, &AIChatDock::_on_import_button_pressed));
	import_button->set_tooltip_text("Import conversation from JSON file");
	return import_button;
}

Button *AIChatAuthenticationUI::_create_export_button(AIChatDock *p_chat_dock) {
	if (!p_chat_dock) {
		return nullptr;
	}

	Button *export_button = memnew(Button);
	export_button->set_text("Export");
	export_button->add_theme_icon_override("icon", p_chat_dock->get_theme_icon(SNAME("Save"), SNAME("EditorIcons")));
	export_button->connect("pressed", callable_mp(p_chat_dock, &AIChatDock::_on_export_button_pressed));
	export_button->set_tooltip_text("Export current conversation to JSON file");
	return export_button;
}

Button *AIChatAuthenticationUI::_create_snapshot_button(AIChatDock *p_chat_dock) {
	if (!p_chat_dock) {
		return nullptr;
	}

	Button *snapshot_button = memnew(Button);
	snapshot_button->set_text("Save Snapshot");
	snapshot_button->add_theme_icon_override("icon", p_chat_dock->get_theme_icon(SNAME("Favorites"), SNAME("EditorIcons")));
	snapshot_button->connect("pressed", callable_mp(p_chat_dock, &AIChatDock::_on_save_snapshot_pressed));
	snapshot_button->set_tooltip_text("Save a named snapshot of your entire project");
	return snapshot_button;
}

Button *AIChatAuthenticationUI::_create_restore_button(AIChatDock *p_chat_dock) {
	if (!p_chat_dock) {
		return nullptr;
	}

	Button *restore_button = memnew(Button);
	restore_button->set_text("Snapshots");
	restore_button->add_theme_icon_override("icon", p_chat_dock->get_theme_icon(SNAME("History"), SNAME("EditorIcons")));
	restore_button->connect("pressed", callable_mp(p_chat_dock, &AIChatDock::_on_view_snapshots_pressed));
	restore_button->set_tooltip_text("View and restore manual snapshots");
	return restore_button;
}

HTTPRequest *AIChatAuthenticationUI::_create_auth_request(AIChatDock *p_chat_dock) {
	if (!p_chat_dock) {
		return nullptr;
	}

	HTTPRequest *auth_request = memnew(HTTPRequest);
	auth_request->connect("request_completed", callable_mp(p_chat_dock, &AIChatDock::_on_auth_request_completed));
	return auth_request;
}

HTTPRequest *AIChatAuthenticationUI::_create_auth_providers_request(AIChatDock *p_chat_dock) {
	if (!p_chat_dock) {
		return nullptr;
	}

	HTTPRequest *auth_providers_request = memnew(HTTPRequest);
	auth_providers_request->connect("request_completed", callable_mp(p_chat_dock, &AIChatDock::_on_auth_providers_request_completed));
	return auth_providers_request;
}
