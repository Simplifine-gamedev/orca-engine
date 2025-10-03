/*
 * © 2025 Simplifine Corp.
 * Personal Non‑Commercial License applies to this file as an original contribution to this Godot fork.
 * See LICENSES/COMPANY-NONCOMMERCIAL.md for terms. Commercial use requires a separate license from Simplifine.
 */

#include "ai_chat_dock_user_messages.h"
#include "ai_chat_dock.h"
#include "scene/gui/rich_text_label.h"
#include "scene/gui/box_container.h"
#include "scene/gui/text_edit.h"
#include "scene/gui/button.h"
#include "scene/gui/label.h"
#include "scene/gui/dialogs.h"
#include "scene/gui/panel_container.h"
#include "scene/main/viewport.h"
#include "core/input/input_event.h"
#include "editor/editor_node.h"

void UserMessageHandler::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_on_restore_send_option", "restore"), &UserMessageHandler::_on_restore_send_option);
	ClassDB::bind_method(D_METHOD("_on_bubble_gui_input", "event", "message_index"), &UserMessageHandler::_on_bubble_gui_input);
}

UserMessageHandler::UserMessageHandler() {
	restore_send_dialog = memnew(ConfirmationDialog);
	restore_send_dialog->set_title("Send Edited Message");
	restore_send_dialog->set_text("You've edited a previous message.\n\nWould you like to:");
	
	// Add custom buttons
	restore_send_dialog->get_ok_button()->set_text("Restore & Send");
	restore_send_dialog->add_cancel_button("Send Without Restoring");
	
	// The OK button (Restore & Send) will trigger confirmed signal
	restore_send_dialog->connect("confirmed", callable_mp(this, &UserMessageHandler::_on_restore_send_option).bind(true));
	restore_send_dialog->connect("canceled", callable_mp(this, &UserMessageHandler::_on_restore_send_option).bind(false));
}

UserMessageHandler::~UserMessageHandler() {
	if (restore_send_dialog && restore_send_dialog->is_inside_tree()) {
		restore_send_dialog->queue_free();
	}
}

void UserMessageHandler::initialize(AIChatDock *p_chat_dock) {
	chat_dock = p_chat_dock;
	
	// Add dialog to the chat dock's scene tree
	if (chat_dock && restore_send_dialog && !restore_send_dialog->is_inside_tree()) {
		chat_dock->add_child(restore_send_dialog);
	}
}

void UserMessageHandler::create_user_message_bubble(VBoxContainer *p_message_vbox, const String &p_content, int p_message_index) {
	if (!p_message_vbox || !chat_dock) return;
	
	// Get the parent PanelContainer (the actual bubble)
	PanelContainer *bubble_panel = Object::cast_to<PanelContainer>(p_message_vbox->get_parent());
	if (!bubble_panel) return;
	
	// Make the bubble clickable
	bubble_panel->set_mouse_filter(Control::MOUSE_FILTER_STOP);
	
	// Connect click event
	bubble_panel->connect("gui_input", callable_mp(this, &UserMessageHandler::_on_bubble_gui_input).bind(p_message_index));
	
	// Just add the content label - no header, no buttons
	RichTextLabel *content_label = memnew(RichTextLabel);
	content_label->set_fit_content(true);
	content_label->set_selection_enabled(true);
	content_label->set_use_bbcode(true);
	content_label->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	content_label->set_mouse_filter(Control::MOUSE_FILTER_IGNORE); // Let clicks pass through to bubble
	
	// Use friend access to call private markdown converter
	String bbcode_content = chat_dock->_markdown_to_bbcode(p_content);
	if (!bbcode_content.is_empty()) {
		content_label->set_text(bbcode_content);
	}
	
	p_message_vbox->add_child(content_label);
}

void UserMessageHandler::_on_bubble_gui_input(const Ref<InputEvent> &p_event, int p_message_index) {
	Ref<InputEventMouseButton> mb = p_event;
	if (mb.is_valid() && mb->is_pressed() && mb->get_button_index() == MouseButton::LEFT) {
		// User clicked the bubble
		on_user_bubble_clicked(p_message_index);
	}
}

void UserMessageHandler::on_user_bubble_clicked(int p_message_index) {
	if (!chat_dock) return;
	
	print_line("AI Chat: User message bubble clicked - index: " + String::num_int64(p_message_index));
	
	// Get messages as array
	Array messages = chat_dock->_get_messages_as_array();
	if (p_message_index < 0 || p_message_index >= messages.size()) return;
	
	Dictionary msg = messages[p_message_index];
	String content = msg.get("content", "");
	String role = msg.get("role", "");
	
	if (role != "user" || content.is_empty()) return;
	
	// Get the input field
	TextEdit *input_field = chat_dock->get_input_field();
	if (!input_field) {
		print_line("AI Chat: Could not find input field");
		return;
	}
	
	// Set the message content in the input field
	input_field->set_text(content);
	input_field->grab_focus();
	
	// Store the clicked message index for later comparison
	clicked_message_index = p_message_index;
	
	print_line("AI Chat: Populated input field with message from index " + String::num_int64(p_message_index));
}

bool UserMessageHandler::handle_send_request(const String &p_message_content) {
	if (!chat_dock) return false;
	
	// Get current chat history
	Array messages = chat_dock->_get_messages_as_array();
	
	// Find if this message matches any previous user message
	int matching_message_index = -1;
	for (int i = 0; i < messages.size(); i++) {
		Dictionary msg = messages[i];
		if (msg.get("role", "") == "user" && msg.get("content", "") == p_message_content) {
			matching_message_index = i;
			break;
		}
	}
	
	if (matching_message_index >= 0) {
		// Message matches an old message exactly - restore to that point
		print_line("AI Chat: Message matches existing message at index " + String::num_int64(matching_message_index) + " - restoring");
		_restore_and_send(matching_message_index, p_message_content);
		return true; // Handler took over - stop normal send flow
	} else if (clicked_message_index >= 0 && clicked_message_index < messages.size()) {
		// User edited a message - ask what they want to do
		Dictionary clicked_msg = messages[clicked_message_index];
		String original_content = clicked_msg.get("content", "");
		
		if (original_content != p_message_content) {
			// Message was edited
			print_line("AI Chat: Message was edited from index " + String::num_int64(clicked_message_index) + " - asking user preference");
			edited_message_content = p_message_content;
			
			if (restore_send_dialog) {
				restore_send_dialog->popup_centered(Size2(450, 200));
			}
			return true; // Wait for user choice - stop normal send flow
		}
	}
	
	// Reset clicked index if we're proceeding with normal send
	clicked_message_index = -1;
	
	// Let chat dock handle the send normally
	return false;
}

void UserMessageHandler::_on_restore_send_option(bool p_restore) {
	if (!chat_dock) return;
	
	if (p_restore) {
		// Restore to clicked message and send
		_restore_and_send(clicked_message_index, edited_message_content);
	} else {
		// Send without restoring
		_send_without_restoring(edited_message_content);
	}
	
	// Reset state
	clicked_message_index = -1;
	edited_message_content = "";
}

void UserMessageHandler::_restore_and_send(int p_message_index, const String &p_content) {
	if (!chat_dock) return;
	
	print_line("AI Chat: Restoring to message index " + String::num_int64(p_message_index) + " and sending");
	
	// Truncate conversation at this point
	chat_dock->_truncate_conversation_at(p_message_index);
	
	// Update the message content if it was edited
	chat_dock->_update_message_content_at(p_message_index, p_content);
	
	// Rebuild UI
	chat_dock->_rebuild_current_conversation_ui();
	
	// Clear input field
	TextEdit *input_field = chat_dock->get_input_field();
	if (input_field) {
		input_field->set_text("");
	}
	
	// Reset state
	clicked_message_index = -1;
	edited_message_content = "";
	
	// Trigger waiting state and send
	chat_dock->call("_process_send_request_async");
	
	// Call private method through friend access to set waiting state
	chat_dock->is_waiting_for_response = true;
	chat_dock->call("_update_ui_state");
}

void UserMessageHandler::_send_without_restoring(const String &p_content) {
	if (!chat_dock) return;
	
	print_line("AI Chat: Sending new message without restoring conversation");
	
	// Reset state
	clicked_message_index = -1;
	edited_message_content = "";
	
	// The message is already in the input field, so we don't need to set it again
	// Just let the normal send flow continue by NOT calling anything
	// The original _on_send_button_pressed will continue after handle_send_request returns
}

