/*
 * © 2025 Simplifine Corp.
 * Personal Non‑Commercial License applies to this file as an original contribution to this Godot fork.
 * See LICENSES/COMPANY-NONCOMMERCIAL.md for terms. Commercial use requires a separate license from Simplifine.
 */

#include "ai_chat_history_manager.h"
#include "scene/gui/button.h"
#include "scene/gui/label.h"
#include "scene/gui/popup_menu.h"
#include "scene/gui/popup.h"
#include "scene/gui/scroll_container.h"
#include "scene/gui/dialogs.h"
#include "scene/gui/line_edit.h"
#include "scene/gui/margin_container.h"
#include "scene/gui/panel_container.h"
#include "scene/resources/style_box_flat.h"
#include "editor/editor_string_names.h"
#include "editor/editor_interface.h"
#include "core/input/input_event.h"

void AIChatHistoryManager::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_on_conversation_selected", "index"), &AIChatHistoryManager::_on_conversation_selected);
	ClassDB::bind_method(D_METHOD("_on_new_conversation_pressed"), &AIChatHistoryManager::_on_new_conversation_pressed);
	ClassDB::bind_method(D_METHOD("_on_dropdown_button_pressed"), &AIChatHistoryManager::_on_dropdown_button_pressed);
	ClassDB::bind_method(D_METHOD("_on_conversation_item_selected", "index"), &AIChatHistoryManager::_on_conversation_item_selected);
	ClassDB::bind_method(D_METHOD("_on_edit_button_pressed", "index"), &AIChatHistoryManager::_on_edit_button_pressed);
	ClassDB::bind_method(D_METHOD("_on_delete_button_pressed", "index"), &AIChatHistoryManager::_on_delete_button_pressed);
	ClassDB::bind_method(D_METHOD("_on_edit_name_confirmed"), &AIChatHistoryManager::_on_edit_name_confirmed);
	ClassDB::bind_method(D_METHOD("_on_edit_name_cancelled"), &AIChatHistoryManager::_on_edit_name_cancelled);
	ClassDB::bind_method(D_METHOD("_on_delete_confirmed"), &AIChatHistoryManager::_on_delete_confirmed);
	ClassDB::bind_method(D_METHOD("_on_delete_cancelled"), &AIChatHistoryManager::_on_delete_cancelled);

	// Signals
	ADD_SIGNAL(MethodInfo("conversation_selected", PropertyInfo(Variant::INT, "index")));
	ADD_SIGNAL(MethodInfo("new_conversation_requested"));
	ADD_SIGNAL(MethodInfo("conversation_rename_requested", PropertyInfo(Variant::INT, "index"), PropertyInfo(Variant::STRING, "new_name")));
	ADD_SIGNAL(MethodInfo("conversation_delete_requested", PropertyInfo(Variant::INT, "index")));
}

AIChatHistoryManager::AIChatHistoryManager() {
	_setup_ui();
	_setup_custom_dropdown();
	_setup_edit_dialog();
	_setup_delete_dialog();
}

AIChatHistoryManager::~AIChatHistoryManager() {
}

void AIChatHistoryManager::_notification(int p_notification) {
	switch (p_notification) {
		case NOTIFICATION_READY:
			// Connect signals after the scene tree is ready
			break;
	}
}

void AIChatHistoryManager::_setup_ui() {
	// Create conversation history label
	Label *history_label = memnew(Label);
	history_label->set_text("Conversation:");
	add_child(history_label);

	// Create custom dropdown button (shows current conversation)
	conversation_dropdown_button = memnew(Button);
	conversation_dropdown_button->set_text("Select Conversation");
	conversation_dropdown_button->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	conversation_dropdown_button->set_clip_contents(true);
	conversation_dropdown_button->connect("pressed", callable_mp(this, &AIChatHistoryManager::_on_dropdown_button_pressed));
	add_child(conversation_dropdown_button);

	// Create new conversation button
	new_conversation_button = memnew(Button);
	new_conversation_button->set_text("New");
	new_conversation_button->set_tooltip_text("Start a new conversation");
	new_conversation_button->connect("pressed", callable_mp(this, &AIChatHistoryManager::_on_new_conversation_pressed));
	add_child(new_conversation_button);
	
	// Create hidden OptionButton for backward compatibility
	conversation_dropdown = memnew(OptionButton);
	conversation_dropdown->set_visible(false);
	add_child(conversation_dropdown);
}

void AIChatHistoryManager::_setup_custom_dropdown() {
	// Create popup panel for custom dropdown
	dropdown_popup = memnew(PopupPanel);
	dropdown_popup->set_size(Size2(400, 300));
	
	// Create scroll container for conversations list
	ScrollContainer *scroll = memnew(ScrollContainer);
	scroll->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	scroll->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	dropdown_popup->add_child(scroll);
	
	// Create container for conversation items
	conversations_container = memnew(VBoxContainer);
	conversations_container->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	scroll->add_child(conversations_container);
	
	add_child(dropdown_popup);
}

void AIChatHistoryManager::_setup_edit_dialog() {
	edit_name_dialog = memnew(AcceptDialog);
	edit_name_dialog->set_title("Edit Conversation Name");
	edit_name_dialog->set_size(Size2(350, 120));
	
	VBoxContainer *vbox = memnew(VBoxContainer);
	edit_name_dialog->add_child(vbox);
	
	Label *label = memnew(Label);
	label->set_text("Enter new conversation name:");
	vbox->add_child(label);
	
	name_edit_field = memnew(LineEdit);
	name_edit_field->set_placeholder("Conversation name");
	vbox->add_child(name_edit_field);
	
	edit_name_dialog->connect("confirmed", callable_mp(this, &AIChatHistoryManager::_on_edit_name_confirmed));
	edit_name_dialog->connect("cancelled", callable_mp(this, &AIChatHistoryManager::_on_edit_name_cancelled));
	
	add_child(edit_name_dialog);
}

void AIChatHistoryManager::_setup_delete_dialog() {
	delete_confirmation_dialog = memnew(ConfirmationDialog);
	delete_confirmation_dialog->set_title("Delete Conversation");
	delete_confirmation_dialog->get_label()->set_text("Are you sure you want to delete this conversation?\nThis action cannot be undone.");
	
	delete_confirmation_dialog->connect("confirmed", callable_mp(this, &AIChatHistoryManager::_on_delete_confirmed));
	delete_confirmation_dialog->connect("cancelled", callable_mp(this, &AIChatHistoryManager::_on_delete_cancelled));
	
	add_child(delete_confirmation_dialog);
}

void AIChatHistoryManager::_on_conversation_selected(int p_index) {
	emit_signal("conversation_selected", p_index);
}

void AIChatHistoryManager::_on_new_conversation_pressed() {
	emit_signal("new_conversation_requested");
}

void AIChatHistoryManager::_on_dropdown_button_pressed() {
	if (dropdown_popup && conversation_dropdown_button) {
		_populate_dropdown_conversations();
		
		// Calculate button position relative to screen
		Vector2 button_global_pos = conversation_dropdown_button->get_global_position();
		Vector2 button_size = conversation_dropdown_button->get_size();
		
		// Get the parent window to convert to screen coordinates
		Window *parent_window = conversation_dropdown_button->get_window();
		if (parent_window) {
			Vector2 window_pos = parent_window->get_position();
			Vector2 popup_screen_pos = window_pos + button_global_pos + Vector2(0, button_size.y + 2);
			
			// Use popup with explicit rect
			Rect2i popup_rect = Rect2i(popup_screen_pos, Size2(400, 300));
			dropdown_popup->popup(popup_rect);
		} else {
			// Fallback to simple positioning
			dropdown_popup->set_size(Size2(400, 300));
			dropdown_popup->set_position(button_global_pos + Vector2(0, button_size.y + 2));
			dropdown_popup->popup();
		}
	}
}

void AIChatHistoryManager::_on_conversation_item_selected(int p_index) {
	dropdown_popup->hide();
	emit_signal("conversation_selected", p_index);
}

void AIChatHistoryManager::_on_edit_button_pressed(int p_index) {
	context_menu_conversation_index = p_index;
	
	// Get current conversation name (from button text, remove timestamp)
	if (conversations_container && p_index < conversations_container->get_child_count()) {
		HBoxContainer *row = Object::cast_to<HBoxContainer>(conversations_container->get_child(p_index));
		if (row && row->get_child_count() > 0) {
			Button *conversation_btn = Object::cast_to<Button>(row->get_child(0));
			if (conversation_btn) {
				String current_name = conversation_btn->get_text();
				// Remove the timestamp part if it exists
				int paren_pos = current_name.find(" (");
				if (paren_pos > 0) {
					current_name = current_name.substr(0, paren_pos);
				}
				name_edit_field->set_text(current_name);
				name_edit_field->select_all();
			}
		}
	}
	
	dropdown_popup->hide();
	edit_name_dialog->popup_centered();
	name_edit_field->grab_focus();
}

void AIChatHistoryManager::_on_delete_button_pressed(int p_index) {
	context_menu_conversation_index = p_index;
	dropdown_popup->hide();
	delete_confirmation_dialog->popup_centered();
}

void AIChatHistoryManager::_on_edit_name_confirmed() {
	String new_name = name_edit_field->get_text().strip_edges();
	if (!new_name.is_empty()) {
		emit_signal("conversation_rename_requested", context_menu_conversation_index, new_name);
	}
	context_menu_conversation_index = -1;
}

void AIChatHistoryManager::_on_edit_name_cancelled() {
	context_menu_conversation_index = -1;
}

void AIChatHistoryManager::_on_delete_confirmed() {
	emit_signal("conversation_delete_requested", context_menu_conversation_index);
	context_menu_conversation_index = -1;
}

void AIChatHistoryManager::_on_delete_cancelled() {
	context_menu_conversation_index = -1;
}

void AIChatHistoryManager::_populate_dropdown_conversations() {
	if (!conversations_container) {
		return;
	}
	
	// Clear existing items
	for (int i = conversations_container->get_child_count() - 1; i >= 0; i--) {
		Node *child = conversations_container->get_child(i);
		child->queue_free();
	}
	
	// Get conversation titles from the hidden OptionButton for compatibility
	if (!conversation_dropdown) {
		return;
	}
	
	// Create conversation rows with edit/delete buttons
	for (int i = 0; i < conversation_dropdown->get_item_count(); i++) {
		String conversation_text = conversation_dropdown->get_item_text(i);
		bool is_current = (i == current_conversation_index);
		
		// Create row container
		HBoxContainer *row = memnew(HBoxContainer);
		row->set_h_size_flags(Control::SIZE_EXPAND_FILL);
		
		// Add background highlight for current conversation
		if (is_current) {
			// Create a panel for background color
			PanelContainer *highlight_panel = memnew(PanelContainer);
			highlight_panel->set_h_size_flags(Control::SIZE_EXPAND_FILL);
			
			// Create a StyleBox for the current conversation
			Ref<StyleBoxFlat> style_box = memnew(StyleBoxFlat);
			style_box->set_bg_color(Color(0.2, 0.4, 0.8, 0.3)); // Light blue highlight
			style_box->set_corner_radius_all(3);
			highlight_panel->add_theme_style_override("panel", style_box);
			
			// Move the row into the highlight panel
			highlight_panel->add_child(row);
			conversations_container->add_child(highlight_panel);
		} else {
			conversations_container->add_child(row);
		}
		
		// Create conversation button (main clickable area)
		Button *conversation_btn = memnew(Button);
		conversation_btn->set_text(conversation_text);
		conversation_btn->set_h_size_flags(Control::SIZE_EXPAND_FILL);
		conversation_btn->set_flat(true);
		conversation_btn->set_text_alignment(HORIZONTAL_ALIGNMENT_LEFT);
		
		// Highlight text color for current conversation
		if (is_current) {
			conversation_btn->add_theme_color_override("font_color", Color(1.0, 1.0, 1.0, 1.0)); // White text
			conversation_btn->add_theme_color_override("font_pressed_color", Color(0.9, 0.9, 0.9, 1.0));
		}
		
		conversation_btn->connect("pressed", callable_mp(this, &AIChatHistoryManager::_on_conversation_item_selected).bind(i));
		row->add_child(conversation_btn);
		
		// Create edit button
		Button *edit_btn = memnew(Button);
		edit_btn->set_text("Edit");
		edit_btn->set_tooltip_text("Edit conversation name");
		edit_btn->set_custom_minimum_size(Size2(50, 0));
		edit_btn->connect("pressed", callable_mp(this, &AIChatHistoryManager::_on_edit_button_pressed).bind(i));
		row->add_child(edit_btn);
		
		// Create delete button
		Button *delete_btn = memnew(Button);
		delete_btn->set_text("Delete");
		delete_btn->set_tooltip_text("Delete this conversation");
		delete_btn->set_custom_minimum_size(Size2(60, 0));
		delete_btn->add_theme_color_override("font_color", Color(0.8, 0.3, 0.3)); // Red color
		delete_btn->connect("pressed", callable_mp(this, &AIChatHistoryManager::_on_delete_button_pressed).bind(i));
		row->add_child(delete_btn);
	}
}

void AIChatHistoryManager::update_conversations(const Vector<String> &p_conversation_titles) {
	if (!conversation_dropdown) {
		return;
	}
	
	// Update the hidden OptionButton for compatibility
	conversation_dropdown->clear();
	for (int i = 0; i < p_conversation_titles.size(); i++) {
		conversation_dropdown->add_item(p_conversation_titles[i]);
	}
}

void AIChatHistoryManager::set_current_conversation(int p_index) {
	if (!conversation_dropdown || !conversation_dropdown_button) {
		return;
	}
	
	// Store current index for highlighting
	current_conversation_index = p_index;
	
	// Update the hidden OptionButton selection for compatibility
	if (p_index >= 0 && p_index < conversation_dropdown->get_item_count()) {
		conversation_dropdown->select(p_index);
		
		// Update the visible button text to show current conversation
		String current_text = conversation_dropdown->get_item_text(p_index);
		// Truncate if too long
		if (current_text.length() > 25) {
			current_text = current_text.substr(0, 22) + "...";
		}
		conversation_dropdown_button->set_text(current_text);
	} else {
		current_conversation_index = -1;
		conversation_dropdown_button->set_text("Select Conversation");
	}
}

int AIChatHistoryManager::get_selected_conversation_index() const {
	if (!conversation_dropdown) {
		return -1;
	}
	
	return conversation_dropdown->get_selected();
}
