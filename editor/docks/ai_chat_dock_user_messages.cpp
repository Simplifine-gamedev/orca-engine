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
#include "scene/gui/scroll_container.h"
#include "scene/resources/style_box_flat.h"
#include "scene/resources/style_box.h"
#include "scene/main/viewport.h"
#include "core/input/input_event.h"
#include "core/io/file_access.h"
#include "editor/editor_node.h"
#include "editor/editor_string_names.h"

void UserMessageHandler::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_on_dialog_option", "option"), &UserMessageHandler::_on_dialog_option);
	ClassDB::bind_method(D_METHOD("_on_dialog_custom_action", "action"), &UserMessageHandler::_on_dialog_custom_action);
	ClassDB::bind_method(D_METHOD("_on_bubble_gui_input", "event", "message_index"), &UserMessageHandler::_on_bubble_gui_input);
	ClassDB::bind_method(D_METHOD("_on_chat_container_gui_input", "event", "editing_message_index"), &UserMessageHandler::_on_chat_container_gui_input);
	ClassDB::bind_method(D_METHOD("_on_edit_send_pressed", "edit_field", "message_index"), &UserMessageHandler::_on_edit_send_pressed);
	ClassDB::bind_method(D_METHOD("_on_edit_cancel_pressed", "message_index"), &UserMessageHandler::_on_edit_cancel_pressed);
	ClassDB::bind_method(D_METHOD("_on_restore_only_pressed", "message_index"), &UserMessageHandler::_on_restore_only_pressed);
}

UserMessageHandler::UserMessageHandler() {
	restore_send_dialog = memnew(ConfirmationDialog);
	restore_send_dialog->set_title("Send Edited Message");
	restore_send_dialog->set_text("This will remove newer messages.\n\nChoose an option:");
	
	// Use standard dialog buttons at bottom
	restore_send_dialog->get_ok_button()->set_text("Restore & Send");
	restore_send_dialog->get_cancel_button()->set_text("Cancel");
	
	// Add custom "Send Without Restoring" button 
	restore_send_dialog->add_button("Send Without Restoring", false, "send_safe");
	
	// Connect signals
	restore_send_dialog->connect("confirmed", callable_mp(this, &UserMessageHandler::_on_dialog_option).bind(0));
	restore_send_dialog->connect("canceled", callable_mp(this, &UserMessageHandler::_on_dialog_option).bind(2)); 
	restore_send_dialog->connect("custom_action", callable_mp(this, &UserMessageHandler::_on_dialog_custom_action));
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

void UserMessageHandler::_on_chat_container_gui_input(const Ref<InputEvent> &p_event, int p_editing_message_index) {
	if (!chat_dock || editing_message_index < 0) return;
	
	Ref<InputEventMouseButton> mb = p_event;
	if (mb.is_valid() && mb->is_pressed() && mb->get_button_index() == MouseButton::LEFT) {
		// Get the edit panel
		VBoxContainer *chat_container = chat_dock->chat_container;
		if (!chat_container) return;
		
		PanelContainer *edit_panel = Object::cast_to<PanelContainer>(
			chat_container->find_child("message_panel_" + String::num_int64(editing_message_index), true, false)
		);
		
		if (!edit_panel || !edit_panel->get_meta("is_editing", false)) return;
		
		// Check if click is outside the edit panel
		Vector2 local_pos = edit_panel->get_local_mouse_position();
		Rect2 panel_rect = Rect2(Vector2(), edit_panel->get_size());
		
		if (!panel_rect.has_point(local_pos)) {
			// Clicked outside - cancel edit mode
			_on_edit_cancel_pressed(editing_message_index);
			
			// Disconnect the click-outside handler
			if (chat_dock->chat_scroll && chat_dock->chat_scroll->is_connected("gui_input", callable_mp(this, &UserMessageHandler::_on_chat_container_gui_input))) {
				chat_dock->chat_scroll->disconnect("gui_input", callable_mp(this, &UserMessageHandler::_on_chat_container_gui_input));
			}
		}
	}
}

void UserMessageHandler::on_user_bubble_clicked(int p_message_index) {
	if (!chat_dock) return;
	
	
	// Get messages
	Array messages = chat_dock->_get_messages_as_array();
	if (p_message_index < 0 || p_message_index >= messages.size()) return;
	
	Dictionary msg = messages[p_message_index];
	String content = msg.get("content", "");
	String role = msg.get("role", "");
	
	if (role != "user" || content.is_empty()) {
		return;
	}
	
	// Access chat_container directly through friend access
	VBoxContainer *chat_container = chat_dock->chat_container;
	if (!chat_container) {
		return;
	}
	
	
	PanelContainer *bubble_panel = Object::cast_to<PanelContainer>(
		chat_container->find_child("message_panel_" + String::num_int64(p_message_index), true, false)
	);
	
	if (!bubble_panel) {
		
		// Message not loaded due to performance optimization - validate index first
		Array all_messages = chat_dock->_get_messages_as_array();
		if (p_message_index >= all_messages.size()) {
			return;
		}
		
		// Save scroll position before rebuild
		ScrollContainer *chat_scroll = chat_dock->chat_scroll;
		float scroll_position = 0.0f;
		if (chat_scroll) {
			VScrollBar *vbar = chat_scroll->get_v_scroll_bar();
			if (vbar) {
				scroll_position = vbar->get_value();
			}
		}
		
		// Clear and rebuild UI with ALL messages loaded (no lazy loading)
		if (chat_container) {
			for (int i = chat_container->get_child_count() - 1; i >= 0; i--) {
				Node *child = chat_container->get_child(i);
				if (child != chat_dock->pending_edits_banner) {
					chat_container->remove_child(child);
					child->queue_free();
				}
			}
		}
		
		// Use full rebuild to load all messages
		chat_dock->_rebuild_conversation_ui_full();
		
		// Restore scroll position
		if (chat_scroll) {
			VScrollBar *vbar = chat_scroll->get_v_scroll_bar();
			if (vbar) {
				vbar->call_deferred("set_value", scroll_position);
			}
		}
		
		// Try to find the panel again after rebuild
		bubble_panel = Object::cast_to<PanelContainer>(
			chat_container->find_child("message_panel_" + String::num_int64(p_message_index), true, false)
		);
		
		if (!bubble_panel) {
			return;
		}
		
	}
	
	
	// Store original content and index
	original_message_content = content;
	editing_message_index = p_message_index;
	
	// Replace bubble with edit field
	_replace_bubble_with_edit_field(bubble_panel, content, p_message_index);
}

void UserMessageHandler::_replace_bubble_with_edit_field(PanelContainer *p_bubble, const String &p_content, int p_message_index) {
	if (!p_bubble || !chat_dock) return;
	
	
	// Clear the bubble's content immediately (not queue_free - that's deferred)
	while (p_bubble->get_child_count() > 0) {
		Node *child = p_bubble->get_child(0);
		p_bubble->remove_child(child);
		memdelete(child); // Immediate deletion
	}
	
	
	// Update bubble styling to match normal user message background (dark gray)
	Ref<StyleBoxFlat> edit_panel_style = memnew(StyleBoxFlat);
	edit_panel_style->set_content_margin_all(12);
	edit_panel_style->set_corner_radius_all(8);
	// Use the same dark gray as normal user message bubbles
	edit_panel_style->set_bg_color(chat_dock->get_theme_color(SNAME("accent_color"), SNAME("Editor")).lightened(0.8) * Color(1, 1, 1, 0.15));
	edit_panel_style->set_border_width_all(1);
	edit_panel_style->set_border_color(chat_dock->get_theme_color(SNAME("accent_color"), SNAME("Editor")) * Color(1, 1, 1, 0.35));
	p_bubble->add_theme_style_override("panel", edit_panel_style);
	
	// Create edit UI
	VBoxContainer *edit_vbox = memnew(VBoxContainer);
	edit_vbox->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	p_bubble->add_child(edit_vbox);
	
	// Editable text field with transparent background
	TextEdit *edit_field = memnew(TextEdit);
	edit_field->set_text(p_content);
	edit_field->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	edit_field->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	edit_field->set_line_wrapping_mode(TextEdit::LINE_WRAPPING_BOUNDARY);
	
	// Make text field auto-size to content
	edit_field->set_fit_content_height_enabled(true);
	
	// Remove background styling from TextEdit
	Ref<StyleBoxEmpty> transparent_style = memnew(StyleBoxEmpty);
	edit_field->add_theme_style_override("normal", transparent_style);
	edit_field->add_theme_style_override("focus", transparent_style);
	
	// Store reference to edit field for click-outside detection
	p_bubble->set_meta("edit_field", edit_field);
	p_bubble->set_meta("is_editing", true);
	
	edit_vbox->add_child(edit_field);
	
	
	// Buttons container aligned to right
	HBoxContainer *button_container = memnew(HBoxContainer);
	button_container->set_alignment(BoxContainer::ALIGNMENT_END);
	button_container->add_theme_constant_override("separation", 8);
	edit_vbox->add_child(button_container);
	
	Button *send_button = memnew(Button);
	send_button->set_text("Send");
	send_button->add_theme_icon_override("icon", chat_dock->get_theme_icon(SNAME("Play"), SNAME("EditorIcons")));
	send_button->connect("pressed", callable_mp(this, &UserMessageHandler::_on_edit_send_pressed).bind(edit_field, p_message_index));
	button_container->add_child(send_button);
	
	Button *restore_button = memnew(Button);
	restore_button->set_text("Restore");
	restore_button->add_theme_icon_override("icon", chat_dock->get_theme_icon(SNAME("Reload"), SNAME("EditorIcons")));
	restore_button->connect("pressed", callable_mp(this, &UserMessageHandler::_on_restore_only_pressed).bind(p_message_index));
	button_container->add_child(restore_button);
	
	Button *cancel_button = memnew(Button);
	cancel_button->set_text("Cancel");
	cancel_button->add_theme_icon_override("icon", chat_dock->get_theme_icon(SNAME("Close"), SNAME("EditorIcons")));
	cancel_button->connect("pressed", callable_mp(this, &UserMessageHandler::_on_edit_cancel_pressed).bind(p_message_index));
	button_container->add_child(cancel_button);
	
	
	// Enable click-outside detection on the chat scroll container
	if (chat_dock->chat_scroll) {
		// If not already connected, connect gui_input for click-outside
		if (!chat_dock->chat_scroll->is_connected("gui_input", callable_mp(this, &UserMessageHandler::_on_chat_container_gui_input))) {
			chat_dock->chat_scroll->connect("gui_input", callable_mp(this, &UserMessageHandler::_on_chat_container_gui_input).bind(p_message_index));
			chat_dock->chat_scroll->set_mouse_filter(Control::MOUSE_FILTER_PASS);
		}
	}
	
	// Make bubble visible and force redraw
	p_bubble->set_visible(true);
	p_bubble->queue_redraw();
	
	// Focus the edit field with a slight delay to ensure UI is ready
	edit_field->call_deferred("grab_focus");
	
}

void UserMessageHandler::_on_edit_send_pressed(TextEdit *p_edit_field, int p_message_index) {
	if (!p_edit_field || !chat_dock) return;
	
	String new_content = p_edit_field->get_text().strip_edges();
	if (new_content.is_empty()) {
		return;
	}
	
	
	// Disconnect click-outside handler if connected
	if (chat_dock->chat_scroll && chat_dock->chat_scroll->is_connected("gui_input", callable_mp(this, &UserMessageHandler::_on_chat_container_gui_input))) {
		chat_dock->chat_scroll->disconnect("gui_input", callable_mp(this, &UserMessageHandler::_on_chat_container_gui_input));
	}
	
	// Get total messages to calculate how many will be lost
	Array messages = chat_dock->_get_messages_as_array();
	int messages_after = messages.size() - p_message_index - 1;
	
	
	// ALWAYS show warning dialog when sending from a previous message
	// User needs to know they're about to rollback the conversation and project state
	editing_message_index = p_message_index;
	
	if (!restore_send_dialog) {
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
	
	// Dialog already has custom buttons configured in constructor
	
	restore_send_dialog->popup_centered(Size2(500, 300));
}

void UserMessageHandler::_on_edit_cancel_pressed(int p_message_index) {
	
	// Disconnect click-outside handler if connected
	if (chat_dock && chat_dock->chat_scroll && chat_dock->chat_scroll->is_connected("gui_input", callable_mp(this, &UserMessageHandler::_on_chat_container_gui_input))) {
		chat_dock->chat_scroll->disconnect("gui_input", callable_mp(this, &UserMessageHandler::_on_chat_container_gui_input));
	}
	
	// Reset state
	editing_message_index = -1;
	original_message_content = "";
	
	// Replace edit field back with bubble (without scrolling)
	_replace_edit_field_with_bubble(p_message_index, false);
}

void UserMessageHandler::_on_restore_only_pressed(int p_message_index) {
	if (!chat_dock) return;
	
	
	// Disconnect click-outside handler if connected
	if (chat_dock->chat_scroll && chat_dock->chat_scroll->is_connected("gui_input", callable_mp(this, &UserMessageHandler::_on_chat_container_gui_input))) {
		chat_dock->chat_scroll->disconnect("gui_input", callable_mp(this, &UserMessageHandler::_on_chat_container_gui_input));
	}
	
	// Save scroll position BEFORE any operations
	ScrollContainer *chat_scroll = chat_dock->chat_scroll;
	float saved_scroll_position = 0.0f;
	if (chat_scroll) {
		VScrollBar *vbar = chat_scroll->get_v_scroll_bar();
		if (vbar) {
			saved_scroll_position = vbar->get_value();
		}
	}
	
	// Backup the ENTIRE chat history file before git restore
	String chat_file_path = chat_dock->conversations_file_path;
	String chat_backup_content;
	bool has_backup = false;
	
	if (FileAccess::exists(chat_file_path)) {
		Error err;
		chat_backup_content = FileAccess::get_file_as_string(chat_file_path, &err);
		if (err == OK) {
			has_backup = true;
		}
	}
	
	// Call the Git restore function directly (UserMessageHandler is friend of AIChatDock)
	bool success = chat_dock->_restore_from_checkpoint(p_message_index);
	
	if (success) {
		
		// Restore the chat history file that we backed up (keep ALL messages)
		if (has_backup) {
			Error err;
			Ref<FileAccess> file = FileAccess::open(chat_file_path, FileAccess::WRITE, &err);
			if (err == OK && file.is_valid()) {
				file->store_string(chat_backup_content);
				file->close();
				
				// Reload conversations from the restored file
				chat_dock->_load_conversations();
			} else {
			}
		}
	} else {
	}
	
	// Reset state
	editing_message_index = -1;
	original_message_content = "";
	
	// Exit edit mode - replace edit field back with bubble (without scrolling)
	_replace_edit_field_with_bubble(p_message_index, false);
	
	// Restore scroll position after everything is rebuilt
	if (chat_scroll) {
		VScrollBar *vbar = chat_scroll->get_v_scroll_bar();
		if (vbar) {
			vbar->call_deferred("set_value", saved_scroll_position);
		}
	}
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
	chat_dock->_rebuild_current_conversation_ui(p_scroll_to_bottom);
	
	// Restore scroll position if we're not scrolling to bottom
	if (chat_scroll && !p_scroll_to_bottom) {
		VScrollBar *vbar = chat_scroll->get_v_scroll_bar();
		if (vbar) {
			vbar->call_deferred("set_value", scroll_position);
		}
	}
}

void UserMessageHandler::_on_dialog_option(int p_option) {
	if (!chat_dock || !restore_send_dialog) return;
	
	String edited_content = restore_send_dialog->get_meta("edited_content", "");
	
	// Disconnect click-outside handler if connected
	if (chat_dock->chat_scroll && chat_dock->chat_scroll->is_connected("gui_input", callable_mp(this, &UserMessageHandler::_on_chat_container_gui_input))) {
		chat_dock->chat_scroll->disconnect("gui_input", callable_mp(this, &UserMessageHandler::_on_chat_container_gui_input));
	}
	
	switch (p_option) {
		case 0: // Restore & Send
			_restore_and_send(editing_message_index, edited_content);
			// Reset state
			editing_message_index = -1;
			original_message_content = "";
			break;
			
		case 2: // Cancel
			// Don't reset state - user continues editing
			break;
	}
}

void UserMessageHandler::_on_dialog_custom_action(const StringName &p_action) {
	if (p_action == "send_safe") {
		// Send Without Restoring
		
		String edited_content = restore_send_dialog->get_meta("edited_content", "");
		
		// CRITICAL: Close the dialog first!
		restore_send_dialog->hide();
		
		// Disconnect click-outside handler if connected
		if (chat_dock->chat_scroll && chat_dock->chat_scroll->is_connected("gui_input", callable_mp(this, &UserMessageHandler::_on_chat_container_gui_input))) {
			chat_dock->chat_scroll->disconnect("gui_input", callable_mp(this, &UserMessageHandler::_on_chat_container_gui_input));
		}
		
		_send_without_restoring(editing_message_index, edited_content);
		// Reset state
		editing_message_index = -1;
		original_message_content = "";
	}
}

void UserMessageHandler::_restore_and_send(int p_message_index, const String &p_content) {
	if (!chat_dock) return;
	
	
    // Perform ACTUAL project restore (files + conversation) to the checkpoint for this message.
    // This was missing in this path and caused restore to be UI-only.
    bool ok = chat_dock->_restore_from_checkpoint(p_message_index);
    if (!ok) {
        // Fallback: at least truncate the conversation so sending proceeds
        chat_dock->_truncate_conversation_at(p_message_index);
    }
    
    // Ensure the edited content is applied to this message before sending
    chat_dock->_update_message_content_at(p_message_index, p_content);
    
    // Rebuild UI to reflect the updated content
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
	
	
	// EXACTLY what you want: 
	// 1. Delete all messages after the edited message
	// 2. Update the edited message content  
	// 3. Send the conversation up to that point
	
	Vector<AIChatDock::ChatMessage> &chat_history = chat_dock->_get_current_chat_history();
	
	// Update the message content
	if (p_message_index >= 0 && p_message_index < chat_history.size()) {
		chat_history.write[p_message_index].content = p_content;
		chat_history.write[p_message_index].timestamp = chat_dock->_get_timestamp();
	}
	
	// Truncate conversation - remove everything after this message
	chat_history.resize(p_message_index + 1);
	
	
	// Replace edit field with normal bubble
	_replace_edit_field_with_bubble(p_message_index, false);
	
	// Reset state
	editing_message_index = -1;
	original_message_content = "";
	
	// Rebuild the UI to reflect truncated conversation (use full rebuild since we can't pass Vector through call())
	chat_dock->call("_rebuild_conversation_ui_full");
	
	// Send the truncated conversation
	chat_dock->is_waiting_for_response = true;
	chat_dock->call("_update_ui_state");
	chat_dock->call("_process_send_request_async");
	
}

