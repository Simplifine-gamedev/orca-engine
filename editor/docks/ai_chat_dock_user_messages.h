/*
 * © 2025 Simplifine Corp.
 * Personal Non‑Commercial License applies to this file as an original contribution to this Godot fork.
 * See LICENSES/COMPANY-NONCOMMERCIAL.md for terms. Commercial use requires a separate license from Simplifine.
 */

#pragma once

#include "core/object/class_db.h"
#include "scene/gui/panel_container.h"
#include "scene/gui/dialogs.h"

class AIChatDock;
class TextEdit;
class RichTextLabel;
class VBoxContainer;

// Handler for user message interactions in AI chat
class UserMessageHandler : public Object {
	GDCLASS(UserMessageHandler, Object);
	
private:
	AIChatDock *chat_dock = nullptr;
	ConfirmationDialog *restore_send_dialog = nullptr;
	int clicked_message_index = -1;
	String edited_message_content;
	
	void _on_bubble_gui_input(const Ref<InputEvent> &p_event, int p_message_index);
	void _on_restore_send_option(bool p_restore);
	void _restore_and_send(int p_message_index, const String &p_content);
	void _send_without_restoring(const String &p_content);
	
protected:
	static void _bind_methods();
	
public:
	UserMessageHandler();
	~UserMessageHandler();
	
	void initialize(AIChatDock *p_chat_dock);
	
	// Create user message bubble without header (just clickable content)
	void create_user_message_bubble(VBoxContainer *p_message_vbox, const String &p_content, int p_message_index);
	
	// Handle user bubble click
	void on_user_bubble_clicked(int p_message_index);
	
	// Check if sent message matches old message and handle accordingly
	// Returns true if handler intercepted the send, false if normal send should proceed
	bool handle_send_request(const String &p_message_content);
};

