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
#include "editor/editor_string_names.h"

void UserMessageHandler::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_on_restore_send_option", "restore"), &UserMessageHandler::_on_restore_send_option);
	ClassDB::bind_method(D_METHOD("_on_bubble_gui_input", "event", "message_index"), &UserMessageHandler::_on_bubble_gui_input);
	ClassDB::bind_method(D_METHOD("_on_edit_send_pressed", "edit_field", "message_index"), &UserMessageHandler::_on_edit_send_pressed);
	ClassDB::bind_method(D_METHOD("_on_edit_cancel_pressed", "message_index"), &UserMessageHandler::_on_edit_cancel_pressed);
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
		// User clicked the bubble - transform it into editable field
		on_user_bubble_clicked(p_message_index);
	}
}

void UserMessageHandler::on_user_bubble_clicked(int p_message_index) {
	if (!chat_dock) return;
	
	print_line("AI Chat: User message bubble clicked - transforming to edit field at index: " + String::num_int64(p_message_index));
	
	// Get messages
	Array messages = chat_dock->_get_messages_as_array();
	if (p_message_index < 0 || p_message_index >= messages.size()) return;
	
	Dictionary msg = messages[p_message_index];
	String content = msg.get("content", "");
	String role = msg.get("role", "");
	
	if (role != "user" || content.is_empty()) {
		print_line("AI Chat: Message is not a user message or is empty");
		return;
	}
	
	// Access chat_container directly through friend access
	VBoxContainer *chat_container = chat_dock->chat_container;
	if (!chat_container) {
		print_line("AI Chat: chat_container is null");
		return;
	}
	
	print_line("AI Chat: Searching for message_panel_" + String::num_int64(p_message_index) + " in " + String::num_int64(chat_container->get_child_count()) + " children");
	
	PanelContainer *bubble_panel = Object::cast_to<PanelContainer>(
		chat_container->find_child("message_panel_" + String::num_int64(p_message_index), true, false)
	);
	
	if (!bubble_panel) {
		print_line("AI Chat: Could not find bubble panel for editing");
		return;
	}
	
	print_line("AI Chat: Found bubble panel, transforming to edit field");
	
	// Store original content and index
	original_message_content = content;
	editing_message_index = p_message_index;
	
	// Replace bubble with edit field
	_replace_bubble_with_edit_field(bubble_panel, content, p_message_index);
}

void UserMessageHandler::_replace_bubble_with_edit_field(PanelContainer *p_bubble, const String &p_content, int p_message_index) {
	if (!p_bubble || !chat_dock) return;
	
	print_line("AI Chat: Replacing bubble with edit field - bubble has " + String::num_int64(p_bubble->get_child_count()) + " children");
	
	// Clear the bubble's content immediately (not queue_free - that's deferred)
	while (p_bubble->get_child_count() > 0) {
		Node *child = p_bubble->get_child(0);
		p_bubble->remove_child(child);
		memdelete(child); // Immediate deletion
	}
	
	print_line("AI Chat: Cleared bubble children, creating edit UI");
	
	// Create edit UI
	VBoxContainer *edit_vbox = memnew(VBoxContainer);
	p_bubble->add_child(edit_vbox);
	
	// Editable text field
	TextEdit *edit_field = memnew(TextEdit);
	edit_field->set_text(p_content);
	edit_field->set_custom_minimum_size(Size2(0, 100));
	edit_field->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	edit_field->set_line_wrapping_mode(TextEdit::LINE_WRAPPING_BOUNDARY);
	edit_field->set_fit_content_height_enabled(false);
	edit_vbox->add_child(edit_field);
	
	print_line("AI Chat: Added TextEdit field");
	
	// Buttons
	HBoxContainer *button_container = memnew(HBoxContainer);
	edit_vbox->add_child(button_container);
	
	Button *send_button = memnew(Button);
	send_button->set_text("Send");
	send_button->add_theme_icon_override("icon", chat_dock->get_theme_icon(SNAME("Play"), SNAME("EditorIcons")));
	send_button->connect("pressed", callable_mp(this, &UserMessageHandler::_on_edit_send_pressed).bind(edit_field, p_message_index));
	button_container->add_child(send_button);
	
	Button *cancel_button = memnew(Button);
	cancel_button->set_text("Cancel");
	cancel_button->add_theme_icon_override("icon", chat_dock->get_theme_icon(SNAME("Stop"), SNAME("EditorIcons")));
	cancel_button->connect("pressed", callable_mp(this, &UserMessageHandler::_on_edit_cancel_pressed).bind(p_message_index));
	button_container->add_child(cancel_button);
	
	print_line("AI Chat: Added buttons");
	
	// Make bubble visible and force redraw
	p_bubble->set_visible(true);
	p_bubble->queue_redraw();
	
	// Focus the edit field with a slight delay to ensure UI is ready
	edit_field->call_deferred("grab_focus");
	
	print_line("AI Chat: Transformed bubble to edit field successfully");
}

void UserMessageHandler::_on_edit_send_pressed(TextEdit *p_edit_field, int p_message_index) {
	if (!p_edit_field || !chat_dock) return;
	
	String new_content = p_edit_field->get_text().strip_edges();
	if (new_content.is_empty()) {
		print_line("AI Chat: Cannot send empty message");
		return;
	}
	
	print_line("AI Chat: Edit send pressed for message " + String::num_int64(p_message_index));
	
	// Get total messages to calculate how many will be lost
	Array messages = chat_dock->_get_messages_as_array();
	int messages_after = messages.size() - p_message_index - 1;
	
	print_line("AI Chat: This will restore to message " + String::num_int64(p_message_index) + ", losing " + String::num_int64(messages_after) + " newer messages");
	
	// ALWAYS show warning dialog when sending from a previous message
	// User needs to know they're about to rollback the conversation and project state
	editing_message_index = p_message_index;
	
	if (!restore_send_dialog) {
		print_line("AI Chat: ERROR - restore_send_dialog is null!");
		return;
	}
	
	// Update dialog text to be more informative
	bool content_changed = (new_content != original_message_content);
	String dialog_text = "You're about to restore the conversation to this message.\n\n";
	dialog_text += "This will:\n";
	dialog_text += "- Remove " + String::num_int64(messages_after) + " newer message" + (messages_after != 1 ? "s" : "") + "\n";
	dialog_text += "- Restore your project to the checkpoint at this message\n";
	dialog_text += "- Lose any changes made after this point\n\n";
	
	if (content_changed) {
		dialog_text += "You can also keep the full conversation and send this as a new message instead.";
	} else {
		dialog_text += "This action cannot be undone.";
	}
	
	restore_send_dialog->set_text(dialog_text);
	restore_send_dialog->set_meta("edited_content", new_content);
	restore_send_dialog->set_meta("content_changed", content_changed);
	
	// Update button labels based on whether content changed
	if (content_changed) {
		restore_send_dialog->get_ok_button()->set_text("Restore & Send");
		// Cancel button text is "Send Without Restoring" by default
	} else {
		restore_send_dialog->get_ok_button()->set_text("Restore & Send");
		restore_send_dialog->get_cancel_button()->set_text("Cancel");
	}
	
	print_line("AI Chat: Showing confirmation dialog...");
	restore_send_dialog->popup_centered(Size2(500, 300));
	print_line("AI Chat: Dialog shown");
}

void UserMessageHandler::_on_edit_cancel_pressed(int p_message_index) {
	print_line("AI Chat: Edit cancelled for message " + String::num_int64(p_message_index));
	
	// Reset state
	editing_message_index = -1;
	original_message_content = "";
	
	// Replace edit field back with bubble (without scrolling)
	_replace_edit_field_with_bubble(p_message_index, false);
}

void UserMessageHandler::_replace_edit_field_with_bubble(int p_message_index, bool p_scroll_to_bottom) {
	if (!chat_dock) return;
	
	// Get current scroll position before rebuild
	ScrollContainer *chat_scroll = chat_dock->chat_scroll;
	float scroll_position = 0.0f;
	if (chat_scroll && !p_scroll_to_bottom) {
		VScrollBar *vbar = chat_scroll->get_v_scroll_bar();
		if (vbar) {
			scroll_position = vbar->get_value();
		}
	}
	
	// Rebuild the conversation UI to restore all bubbles
	chat_dock->_rebuild_current_conversation_ui();
	
	// Restore scroll position if we're not scrolling to bottom
	if (chat_scroll && !p_scroll_to_bottom) {
		VScrollBar *vbar = chat_scroll->get_v_scroll_bar();
		if (vbar) {
			vbar->call_deferred("set_value", scroll_position);
		}
	}
}

void UserMessageHandler::_on_restore_send_option(bool p_restore) {
	if (!chat_dock || !restore_send_dialog) return;
	
	String edited_content = restore_send_dialog->get_meta("edited_content", "");
	bool content_changed = restore_send_dialog->get_meta("content_changed", false);
	
	if (p_restore) {
		// Restore to message and send
		print_line("AI Chat: User chose to restore & send");
		_restore_and_send(editing_message_index, edited_content);
	} else {
		// User clicked cancel/second option
		if (content_changed) {
			// Content was changed - "Send Without Restoring" was clicked
			print_line("AI Chat: User chose to send without restoring");
			_send_without_restoring(editing_message_index, edited_content);
		} else {
			// Content unchanged - "Cancel" was clicked, just close the edit field
			print_line("AI Chat: User cancelled the restore operation");
			_replace_edit_field_with_bubble(editing_message_index, false);
		}
	}
	
	// Reset state
	editing_message_index = -1;
	original_message_content = "";
}

void UserMessageHandler::_restore_and_send(int p_message_index, const String &p_content) {
	if (!chat_dock) return;
	
	print_line("AI Chat: Restoring to message index " + String::num_int64(p_message_index) + " and sending");
	
	// Truncate conversation at this point
	chat_dock->_truncate_conversation_at(p_message_index);
	
	// Update the message content
	chat_dock->_update_message_content_at(p_message_index, p_content);
	
	// Rebuild UI
	chat_dock->_rebuild_current_conversation_ui();
	
	// Reset state
	editing_message_index = -1;
	original_message_content = "";
	
	// Trigger waiting state and send
	chat_dock->is_waiting_for_response = true;
	chat_dock->call("_update_ui_state");
	chat_dock->call("_process_send_request_async");
}

void UserMessageHandler::_send_without_restoring(int p_message_index, const String &p_content) {
	if (!chat_dock) return;
	
	print_line("AI Chat: Sending edited message without restoring conversation");
	
	// Replace edit field with normal bubble first (scroll to bottom since adding new message)
	_replace_edit_field_with_bubble(p_message_index, true);
	
	// Add the edited content as a new user message
	chat_dock->call("_add_message_to_chat", "user", p_content, Array());
	
	// Reset state
	editing_message_index = -1;
	original_message_content = "";
	
	// Trigger send
	chat_dock->is_waiting_for_response = true;
	chat_dock->call("_update_ui_state");
	chat_dock->call("_process_send_request_async");
}

