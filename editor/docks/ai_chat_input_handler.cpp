/**************************************************************************/
/*  ai_chat_input_handler.cpp                                             */
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
/* included in all copies or substantial portions of the Software.         */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "ai_chat_input_handler.h"
#include "core/input/input_event.h"
#include "scene/gui/text_edit.h"

bool AIChatInputHandler::should_send_on_enter(const Ref<InputEvent> &p_event) {
	Ref<InputEventKey> key_event = p_event;
	if (!key_event.is_valid() || !key_event->is_pressed()) {
		return false;
	}
	
	Key keycode = key_event->get_keycode();
	if (keycode != Key::ENTER && keycode != Key::KP_ENTER) {
		return false;
	}
	
	// Only send if Shift is NOT pressed (Shift+Enter = newline)
	return !key_event->is_shift_pressed();
}

bool AIChatInputHandler::handle_input_event(const Ref<InputEvent> &p_event, TextEdit *p_input_field, const Callable &p_send_callback) {
	if (!p_input_field) {
		return false;
	}
	
	Ref<InputEventKey> key_event = p_event;
	if (!key_event.is_valid() || !key_event->is_pressed() || key_event->is_echo()) {
		return false;
	}
	
	Key keycode = key_event->get_keycode();
	
	// Handle Enter key
	if (keycode == Key::ENTER || keycode == Key::KP_ENTER) {
		if (!key_event->is_shift_pressed()) {
			// Enter without Shift: send the message
			if (p_send_callback.is_valid()) {
				p_send_callback.call();
			}
			return true; // Event handled
		}
		// Shift+Enter: allow TextEdit to handle it normally (create newline)
		// Don't mark as handled so TextEdit can process it
		return false;
	}
	
	return false; // Event not handled by us
}


