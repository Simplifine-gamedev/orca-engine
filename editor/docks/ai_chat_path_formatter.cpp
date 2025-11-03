/**************************************************************************/
/*  ai_chat_path_formatter.cpp                                           */
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

#include "ai_chat_path_formatter.h"
#include "core/config/project_settings.h"

String AIChatPathFormatter::format_path_for_display(const String &p_path) {
	String path = p_path;
	
	// Strip res:// prefix for display purposes
	if (path.begins_with("res://")) {
		path = path.substr(6); // Remove "res://" prefix (6 characters)
	}
	
	return path;
}

String AIChatPathFormatter::format_path_for_operation(const String &p_path) {
	String path = p_path;
	
	// If not an engine path and not absolute, assume project-relative and prefix with res://
	bool is_engine_path = path.begins_with("res://") || path.begins_with("user://");
	
	if (!is_engine_path && !path.is_absolute_path()) {
		// Normalize relative path (remove ./ and ../)
		String normalized_rel = path.replace("\\", "/");
		while (normalized_rel.begins_with("./")) {
			normalized_rel = normalized_rel.substr(2);
		}
		path = String("res://") + normalized_rel;
	}
	
	// If path is absolute, try to convert to res:// when possible
	if (path.is_absolute_path() && !is_engine_path) {
		String project_root = ProjectSettings::get_singleton()->globalize_path("res://");
		String rel = ProjectSettings::get_singleton()->localize_path(path);
		if (!rel.is_empty() && rel != path) {
			path = String("res://") + rel;
		}
	}
	
	return path;
}


