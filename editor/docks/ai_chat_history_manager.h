/**************************************************************************/
/*  ai_chat_history_manager.h                                             */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#pragma once

#include "scene/gui/box_container.h"
#include "scene/gui/option_button.h"
#include "scene/gui/popup_menu.h"
#include "scene/gui/dialogs.h"
#include "scene/gui/line_edit.h"

class Button;
class ConfirmationDialog;
class AcceptDialog;
class PopupPanel;
class VBoxContainer;
class ScrollContainer;

class AIChatHistoryManager : public HBoxContainer {
	GDCLASS(AIChatHistoryManager, HBoxContainer);

private:
	// UI Components
	Button *conversation_dropdown_button = nullptr;
	PopupPanel *dropdown_popup = nullptr;
	VBoxContainer *conversations_container = nullptr;
	Button *new_conversation_button = nullptr;
	OptionButton *conversation_dropdown = nullptr; // Keep for backward compatibility
	
	// Dialogs
	AcceptDialog *edit_name_dialog = nullptr;
	LineEdit *name_edit_field = nullptr;
	ConfirmationDialog *delete_confirmation_dialog = nullptr;
	
	// State
	int context_menu_conversation_index = -1;
	int current_conversation_index = -1;

	void _setup_ui();
	void _setup_custom_dropdown();
	void _setup_edit_dialog();
	void _setup_delete_dialog();
	void _populate_dropdown_conversations();
	
	// Event handlers
	void _on_conversation_selected(int p_index);
	void _on_new_conversation_pressed();
	void _on_dropdown_button_pressed();
	void _on_conversation_item_selected(int p_index);
	void _on_edit_button_pressed(int p_index);
	void _on_delete_button_pressed(int p_index);
	void _on_edit_name_confirmed();
	void _on_edit_name_cancelled();
	void _on_delete_confirmed();
	void _on_delete_cancelled();

protected:
	void _notification(int p_notification);
	static void _bind_methods();

public:
	AIChatHistoryManager();
	~AIChatHistoryManager();
	
	// Public interface for parent dock
    void update_conversations(const Vector<String> &p_conversation_titles);
    void set_current_conversation(int p_index);
    // Mark a conversation as busy (shows an orange dot in the dropdown list)
    void mark_conversation_busy(int p_index, bool p_busy);
	int get_selected_conversation_index() const;
	
	// For backward compatibility
	OptionButton *get_conversation_dropdown() const { return conversation_dropdown; }
	Button *get_new_conversation_button() const { return new_conversation_button; }
	Button *get_conversation_dropdown_button() const { return conversation_dropdown_button; }
	
	// Signals
	// conversation_selected(int index)
	// new_conversation_requested()
	// conversation_rename_requested(int index, String new_name)
	// conversation_delete_requested(int index)
};
