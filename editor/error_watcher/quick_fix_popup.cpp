/**************************************************************************/
/*  quick_fix_popup.cpp                                                   */
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

#include "error_watcher.h"

#include "scene/gui/box_container.h"
#include "scene/gui/button.h"
#include "scene/gui/label.h"
#include "scene/gui/rich_text_label.h"
#include "scene/gui/option_button.h"

void QuickFixPopup::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_fix_selected", "index"), &QuickFixPopup::_fix_selected);
	ClassDB::bind_method(D_METHOD("_apply_fix"), &QuickFixPopup::_apply_fix);
	ClassDB::bind_method(D_METHOD("_preview_fix"), &QuickFixPopup::_preview_fix);
	ClassDB::bind_method(D_METHOD("_cancel_fix"), &QuickFixPopup::_cancel_fix);
}

void QuickFixPopup::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_THEME_CHANGED: {
			if (apply_button) {
				apply_button->set_icon(get_theme_icon(SNAME("ImportCheck"), SNAME("EditorIcons")));
			}
			if (preview_button) {
				preview_button->set_icon(get_theme_icon(SNAME("PreviewViewport"), SNAME("EditorIcons")));
			}
			if (cancel_button) {
				cancel_button->set_icon(get_theme_icon(SNAME("Close"), SNAME("EditorIcons")));
			}
		} break;
	}
}

void QuickFixPopup::_fix_selected(int p_index) {
	selected_fix_index = p_index;
	_update_preview();
}

void QuickFixPopup::_apply_fix() {
	if (selected_fix_index >= 0 && selected_fix_index < available_fixes.size()) {
		QuickFixAction fix = available_fixes[selected_fix_index];
		
		if (ErrorWatcher::get_singleton()->apply_quick_fix(fix)) {
			// Success - hide popup
			hide();
			
			// Show confirmation
			// TODO: Add toast notification
		} else {
			// Show error message
			// TODO: Add error dialog
		}
	}
}

void QuickFixPopup::_preview_fix() {
	if (selected_fix_index >= 0 && selected_fix_index < available_fixes.size()) {
		_update_preview();
	}
}

void QuickFixPopup::_cancel_fix() {
	hide();
}

void QuickFixPopup::_update_preview() {
	if (!preview_text || selected_fix_index < 0 || selected_fix_index >= available_fixes.size()) {
		return;
	}
	
	QuickFixAction fix = available_fixes[selected_fix_index];
	String preview = ErrorWatcher::get_singleton()->preview_quick_fix(fix);
	
	preview_text->clear();
	preview_text->append_text("[b]Preview:[/b]\n");
	preview_text->append_text(preview);
	
	if (apply_button) {
		apply_button->set_disabled(false);
	}
}

void QuickFixPopup::show_fixes(const ErrorWatcherError &p_error, const Vector<QuickFixAction> &p_fixes) {
	current_error = p_error;
	available_fixes = p_fixes;
	selected_fix_index = -1;
	
	if (!error_label || !fixes_container) {
		return;
	}
	
	// Update error description
	error_label->set_text(vformat("Error: %s (Line %d)", p_error.message, p_error.line));
	
	// Clear existing fix options
	for (int i = fixes_container->get_child_count() - 1; i >= 0; i--) {
		Node *child = fixes_container->get_child(i);
		fixes_container->remove_child(child);
		child->queue_free();
	}
	
	// Add fix options
	for (int i = 0; i < p_fixes.size(); i++) {
		const QuickFixAction &fix = p_fixes[i];
		
		Button *fix_button = memnew(Button);
		fix_button->set_text(fix.description);
		fix_button->set_h_size_flags(Control::SIZE_EXPAND_FILL);
		fix_button->connect("pressed", callable_mp(this, &QuickFixPopup::_fix_selected).bind(i));
		fixes_container->add_child(fix_button);
	}
	
	// Select first fix by default
	if (!p_fixes.is_empty()) {
		selected_fix_index = 0;
		_update_preview();
	}
	
	// Show popup
	popup_centered(Size2(500, 400));
}

void QuickFixPopup::hide_popup() {
	hide();
}

QuickFixPopup::QuickFixPopup() {
	set_title("Quick Fix");
	set_resizable(true);
	
	main_container = memnew(VBoxContainer);
	add_child(main_container);
	main_container->set_anchors_and_offsets_preset(Control::PRESET_FULL_RECT);
	main_container->set_custom_minimum_size(Size2(400, 300));
	
	// Error description
	error_label = memnew(Label);
	error_label->set_autowrap_mode(TextServer::AUTOWRAP_WORD_SMART);
	main_container->add_child(error_label);
	
	main_container->add_child(memnew(HSeparator));
	
	// Fix options
	Label *fixes_label = memnew(Label);
	fixes_label->set_text("Available fixes:");
	main_container->add_child(fixes_label);
	
	fixes_container = memnew(VBoxContainer);
	main_container->add_child(fixes_container);
	
	main_container->add_child(memnew(HSeparator));
	
	// Preview area
	Label *preview_label = memnew(Label);
	preview_label->set_text("Preview:");
	main_container->add_child(preview_label);
	
	preview_text = memnew(RichTextLabel);
	preview_text->set_use_bbcode(true);
	preview_text->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	preview_text->set_custom_minimum_size(Size2(0, 100));
	main_container->add_child(preview_text);
	
	// Buttons
	button_container = memnew(HBoxContainer);
	main_container->add_child(button_container);
	
	button_container->add_spacer();
	
	cancel_button = memnew(Button);
	cancel_button->set_text("Cancel");
	cancel_button->connect("pressed", callable_mp(this, &QuickFixPopup::_cancel_fix));
	button_container->add_child(cancel_button);
	
	preview_button = memnew(Button);
	preview_button->set_text("Preview");
	preview_button->connect("pressed", callable_mp(this, &QuickFixPopup::_preview_fix));
	button_container->add_child(preview_button);
	
	apply_button = memnew(Button);
	apply_button->set_text("Apply Fix");
	apply_button->connect("pressed", callable_mp(this, &QuickFixPopup::_apply_fix));
	apply_button->set_disabled(true);
	button_container->add_child(apply_button);
}

QuickFixPopup::~QuickFixPopup() {
}