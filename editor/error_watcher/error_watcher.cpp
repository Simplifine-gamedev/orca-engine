/**************************************************************************/
/*  error_watcher.cpp                                                     */
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

#include "core/io/file_access.h"
#include "core/string/string_utils.h"
#include "editor/editor_node.h"
#include "editor/gui/code_editor.h"
#include "editor/script/script_text_editor.h"
#include "scene/gui/box_container.h"
#include "scene/gui/button.h"
#include "scene/gui/label.h"
#include "scene/gui/rich_text_label.h"

ErrorWatcher *ErrorWatcher::singleton = nullptr;

// ErrorWatcherPanel implementation

void ErrorWatcherPanel::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_clear_all_errors"), &ErrorWatcherPanel::_clear_all_errors);
	ClassDB::bind_method(D_METHOD("_error_item_clicked", "meta"), &ErrorWatcherPanel::_error_item_clicked);
}

void ErrorWatcherPanel::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_READY: {
			_update_error_display();
		} break;
		case NOTIFICATION_THEME_CHANGED: {
			if (clear_all_button) {
				clear_all_button->set_icon(get_theme_icon(SNAME("Clear"), SNAME("EditorIcons")));
			}
		} break;
	}
}

void ErrorWatcherPanel::_clear_all_errors() {
	current_errors.clear();
	_update_error_display();
	
	if (ErrorWatcher::get_singleton()) {
		ErrorWatcher::get_singleton()->clear_errors();
	}
}

void ErrorWatcherPanel::_error_item_clicked(const Variant &p_meta) {
	if (p_meta.get_type() != Variant::STRING) {
		return;
	}
	
	String meta_str = p_meta;
	if (meta_str.begins_with("goto:")) {
		Vector<String> parts = meta_str.split(":");
		if (parts.size() >= 3) {
			String file_path = parts[1];
			int line = parts[2].to_int();
			
			// Open file and go to line
			EditorNode *editor_node = EditorNode::get_singleton();
			if (editor_node) {
				editor_node->load_resource(file_path);
				// TODO: Navigate to specific line
			}
		}
	} else if (meta_str.begins_with("fix:")) {
		Vector<String> parts = meta_str.split(":");
		if (parts.size() >= 2) {
			int error_index = parts[1].to_int();
			if (error_index >= 0 && error_index < current_errors.size()) {
				ErrorWatcherError error = current_errors[error_index];
				Vector<QuickFixAction> fixes = ErrorWatcher::get_singleton()->get_quick_fixes(error);
				
				if (!fixes.is_empty()) {
					// Show quick fix popup
					// TODO: Implement QuickFixPopup
					ErrorWatcher::get_singleton()->_record_telemetry("auto_fix_shown", String::num_int64((int)error.type));
				}
			}
		}
	}
}

void ErrorWatcherPanel::_update_error_display() {
	if (!error_count_label || !error_list) {
		return;
	}
	
	error_count_label->set_text(vformat("Errors: %d", current_errors.size()));
	error_list->clear();
	
	if (current_errors.is_empty()) {
		error_list->append_text("No errors detected.");
		return;
	}
	
	for (int i = 0; i < current_errors.size(); i++) {
		const ErrorWatcherError &error = current_errors[i];
		
		String error_type_str;
		switch (error.type) {
			case ErrorWatcherError::DUPLICATE_VARIABLE:
				error_type_str = "Duplicate Variable";
				break;
			case ErrorWatcherError::MISSING_IMPORT:
				error_type_str = "Missing Import";
				break;
			case ErrorWatcherError::UNDEFINED_VARIABLE:
				error_type_str = "Undefined Variable";
				break;
			case ErrorWatcherError::SYNTAX_ERROR:
				error_type_str = "Syntax Error";
				break;
			case ErrorWatcherError::TYPE_MISMATCH:
				error_type_str = "Type Mismatch";
				break;
			default:
				error_type_str = "Error";
				break;
		}
		
		String file_display = error.file_path.get_file();
		if (file_display.is_empty()) {
			file_display = "Unknown";
		}
		
		error_list->append_text(vformat("[b]%s[/b] in [url=goto:%s:%d]%s:%d[/url]\n", 
			error_type_str, error.file_path, error.line, file_display, error.line));
		error_list->append_text(vformat("  %s\n", error.message));
		
		if (error.has_quick_fix) {
			error_list->append_text(vformat("  [color=blue][url=fix:%d]⚡ Quick Fix Available[/url][/color]\n", i));
		}
		
		if (i < current_errors.size() - 1) {
			error_list->append_text("\n");
		}
	}
}

void ErrorWatcherPanel::add_error(const ErrorWatcherError &p_error) {
	current_errors.push_back(p_error);
	_update_error_display();
}

void ErrorWatcherPanel::clear_errors() {
	current_errors.clear();
	_update_error_display();
}

void ErrorWatcherPanel::set_errors(const Vector<ErrorWatcherError> &p_errors) {
	current_errors = p_errors;
	_update_error_display();
}

ErrorWatcherPanel::ErrorWatcherPanel() {
	set_name("ErrorWatcher");
	set_custom_minimum_size(Size2(300, 200));
	
	main_container = memnew(VBoxContainer);
	add_child(main_container);
	main_container->set_anchors_and_offsets_preset(Control::PRESET_FULL_RECT);
	
	// Header with error count and clear button
	header_container = memnew(HBoxContainer);
	main_container->add_child(header_container);
	
	error_count_label = memnew(Label);
	error_count_label->set_text("Errors: 0");
	header_container->add_child(error_count_label);
	
	header_container->add_spacer();
	
	clear_all_button = memnew(Button);
	clear_all_button->set_text("Clear All");
	clear_all_button->connect("pressed", callable_mp(this, &ErrorWatcherPanel::_clear_all_errors));
	header_container->add_child(clear_all_button);
	
	// Error list
	error_list = memnew(RichTextLabel);
	error_list->set_use_bbcode(true);
	error_list->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	error_list->set_selection_enabled(true);
	error_list->connect("meta_clicked", callable_mp(this, &ErrorWatcherPanel::_error_item_clicked));
	main_container->add_child(error_list);
}

ErrorWatcherPanel::~ErrorWatcherPanel() {
}

// ErrorWatcher implementation

void ErrorWatcher::_bind_methods() {
	ClassDB::bind_method(D_METHOD("process_compiler_output", "output"), &ErrorWatcher::process_compiler_output);
	ClassDB::bind_method(D_METHOD("process_runtime_error", "file", "line", "column", "message"), &ErrorWatcher::process_runtime_error);
	ClassDB::bind_method(D_METHOD("get_telemetry_data"), &ErrorWatcher::get_telemetry_data);
	ClassDB::bind_method(D_METHOD("reset_telemetry"), &ErrorWatcher::reset_telemetry);
}

ErrorWatcherError::Type ErrorWatcher::_classify_error(const String &p_message) {
	String msg_lower = p_message.to_lower();
	
	if (_is_duplicate_variable_error(p_message)) {
		return ErrorWatcherError::DUPLICATE_VARIABLE;
	}
	
	if (_is_undefined_variable_error(p_message)) {
		return ErrorWatcherError::UNDEFINED_VARIABLE;
	}
	
	if (_is_missing_import_error(p_message)) {
		return ErrorWatcherError::MISSING_IMPORT;
	}
	
	if (msg_lower.contains("syntax") || msg_lower.contains("unexpected") || 
		msg_lower.contains("expected") || msg_lower.contains("invalid")) {
		return ErrorWatcherError::SYNTAX_ERROR;
	}
	
	if (msg_lower.contains("type") && (msg_lower.contains("mismatch") || msg_lower.contains("cannot convert"))) {
		return ErrorWatcherError::TYPE_MISMATCH;
	}
	
	return ErrorWatcherError::UNKNOWN;
}

bool ErrorWatcher::_is_duplicate_variable_error(const String &p_message) {
	String msg_lower = p_message.to_lower();
	return msg_lower.contains("already") && (msg_lower.contains("declared") || msg_lower.contains("defined")) ||
		   msg_lower.contains("duplicate") && (msg_lower.contains("variable") || msg_lower.contains("identifier"));
}

bool ErrorWatcher::_is_undefined_variable_error(const String &p_message) {
	String msg_lower = p_message.to_lower();
	return msg_lower.contains("undefined") || msg_lower.contains("not defined") || 
		   msg_lower.contains("unknown identifier") || msg_lower.contains("undeclared");
}

bool ErrorWatcher::_is_missing_import_error(const String &p_message) {
	String msg_lower = p_message.to_lower();
	return msg_lower.contains("not found") || msg_lower.contains("cannot find") ||
		   msg_lower.contains("no module") || msg_lower.contains("import");
}

ErrorWatcherError ErrorWatcher::_parse_compiler_error(const String &p_file, int p_line, int p_column, const String &p_message) {
	ErrorWatcherError error;
	error.file_path = p_file;
	error.line = p_line;
	error.column = p_column;
	error.message = p_message;
	error.original_error = p_message;
	error.type = _classify_error(p_message);
	
	// Check if we can provide a quick fix for this error type
	Vector<QuickFixAction> fixes = _generate_quick_fixes(error);
	error.has_quick_fix = !fixes.is_empty();
	
	if (error.has_quick_fix && !fixes.is_empty()) {
		error.suggested_fix = fixes[0].description;
		error.fix_preview = fixes[0].preview_text;
	}
	
	return error;
}

ErrorWatcherError ErrorWatcher::_parse_runtime_error(const String &p_file, int p_line, int p_column, const String &p_message) {
	return _parse_compiler_error(p_file, p_line, p_column, p_message);
}

Vector<QuickFixAction> ErrorWatcher::_generate_quick_fixes(const ErrorWatcherError &p_error) {
	Vector<QuickFixAction> fixes;
	
	switch (p_error.type) {
		case ErrorWatcherError::DUPLICATE_VARIABLE: {
			QuickFixAction fix = _create_rename_variable_fix(p_error);
			if (!fix.description.is_empty()) {
				fixes.push_back(fix);
			}
		} break;
		
		case ErrorWatcherError::UNDEFINED_VARIABLE: {
			// Try to suggest an import
			QuickFixAction import_fix = _create_add_import_fix(p_error);
			if (!import_fix.description.is_empty()) {
				fixes.push_back(import_fix);
			}
			
			// Or suggest adding a declaration
			QuickFixAction declare_fix = _create_add_declaration_fix(p_error);
			if (!declare_fix.description.is_empty()) {
				fixes.push_back(declare_fix);
			}
		} break;
		
		case ErrorWatcherError::MISSING_IMPORT: {
			QuickFixAction fix = _create_add_import_fix(p_error);
			if (!fix.description.is_empty()) {
				fixes.push_back(fix);
			}
		} break;
		
		case ErrorWatcherError::SYNTAX_ERROR: {
			QuickFixAction fix = _create_syntax_fix(p_error);
			if (!fix.description.is_empty()) {
				fixes.push_back(fix);
			}
		} break;
		
		default:
			break;
	}
	
	return fixes;
}

QuickFixAction ErrorWatcher::_create_rename_variable_fix(const ErrorWatcherError &p_error) {
	QuickFixAction action;
	action.type = QuickFixAction::RENAME_VARIABLE;
	action.file_path = p_error.file_path;
	action.line = p_error.line;
	action.column = p_error.column;
	
	String var_name = _extract_variable_name(p_error.message);
	if (var_name.is_empty()) {
		return action;
	}
	
	// Suggest a new name by appending a number
	String new_name = var_name + "2";
	action.description = vformat("Rename '%s' to '%s'", var_name, new_name);
	action.old_text = var_name;
	action.new_text = new_name;
	action.preview_text = vformat("Change '%s' to '%s' on line %d", var_name, new_name, p_error.line);
	
	return action;
}

QuickFixAction ErrorWatcher::_create_add_import_fix(const ErrorWatcherError &p_error) {
	QuickFixAction action;
	action.type = QuickFixAction::ADD_IMPORT;
	action.file_path = p_error.file_path;
	action.line = 1; // Add import at top of file
	action.column = 1;
	
	String symbol = _extract_variable_name(p_error.message);
	if (symbol.is_empty()) {
		return action;
	}
	
	String import_suggestion = _suggest_import_for_symbol(symbol);
	if (import_suggestion.is_empty()) {
		return action;
	}
	
	action.description = vformat("Add import: %s", import_suggestion);
	action.old_text = "";
	action.new_text = import_suggestion + "\n";
	action.preview_text = vformat("Add '%s' at the top of the file", import_suggestion);
	
	return action;
}

QuickFixAction ErrorWatcher::_create_add_declaration_fix(const ErrorWatcherError &p_error) {
	QuickFixAction action;
	action.type = QuickFixAction::ADD_VARIABLE_DECLARATION;
	action.file_path = p_error.file_path;
	action.line = p_error.line;
	action.column = p_error.column;
	
	String var_name = _extract_variable_name(p_error.message);
	if (var_name.is_empty()) {
		return action;
	}
	
	String declaration = vformat("var %s", var_name);
	action.description = vformat("Declare variable '%s'", var_name);
	action.old_text = "";
	action.new_text = declaration + "\n";
	action.preview_text = vformat("Add '%s' before line %d", declaration, p_error.line);
	
	return action;
}

QuickFixAction ErrorWatcher::_create_syntax_fix(const ErrorWatcherError &p_error) {
	QuickFixAction action;
	action.type = QuickFixAction::FIX_SYNTAX;
	action.file_path = p_error.file_path;
	action.line = p_error.line;
	action.column = p_error.column;
	
	String msg_lower = p_error.message.to_lower();
	
	// Common syntax fixes
	if (msg_lower.contains("missing") && msg_lower.contains(":")) {
		action.description = "Add missing colon";
		action.preview_text = "Add ':' at the end of the line";
	} else if (msg_lower.contains("missing") && msg_lower.contains("(")) {
		action.description = "Add missing opening parenthesis";
		action.preview_text = "Add '(' before the expression";
	} else if (msg_lower.contains("missing") && msg_lower.contains(")")) {
		action.description = "Add missing closing parenthesis";
		action.preview_text = "Add ')' after the expression";
	} else {
		return action; // No specific fix available
	}
	
	return action;
}

String ErrorWatcher::_extract_variable_name(const String &p_error_message) {
	// Simple regex-like extraction for common patterns
	Vector<String> words = p_error_message.split(" ");
	
	for (int i = 0; i < words.size(); i++) {
		String word = words[i];
		// Remove quotes and punctuation
		word = word.strip_edges().trim_prefix("'").trim_suffix("'").trim_prefix("\"").trim_suffix("\"");
		word = word.trim_suffix(",").trim_suffix(".").trim_suffix(":").trim_suffix(";");
		
		// Check if this looks like an identifier
		if (word.length() > 0 && (word[0].is_ascii_identifier_char() || word[0] == '_')) {
			bool is_identifier = true;
			for (int j = 1; j < word.length(); j++) {
				if (!word[j].is_ascii_identifier_char() && word[j] != '_') {
					is_identifier = false;
					break;
				}
			}
			if (is_identifier && !word.is_numeric()) {
				return word;
			}
		}
	}
	
	return "";
}

String ErrorWatcher::_suggest_import_for_symbol(const String &p_symbol) {
	// Basic import suggestions for common Godot classes
	HashMap<String, String> common_imports;
	common_imports["Vector2"] = "# Vector2 is built-in";
	common_imports["Vector3"] = "# Vector3 is built-in";
	common_imports["Node"] = "# Node is built-in";
	common_imports["Control"] = "# Control is built-in";
	common_imports["RigidBody2D"] = "# RigidBody2D is built-in";
	common_imports["CharacterBody2D"] = "# CharacterBody2D is built-in";
	
	if (common_imports.has(p_symbol)) {
		return common_imports[p_symbol];
	}
	
	// For custom classes, suggest a preload
	return vformat("const %s = preload(\"res://path/to/%s.gd\")", p_symbol, p_symbol.to_lower());
}

String ErrorWatcher::_get_file_content(const String &p_file_path) {
	Ref<FileAccess> file = FileAccess::open(p_file_path, FileAccess::READ);
	if (file.is_null()) {
		return "";
	}
	return file->get_as_text();
}

Vector<String> ErrorWatcher::_get_file_lines(const String &p_file_path) {
	String content = _get_file_content(p_file_path);
	return content.split("\n");
}

void ErrorWatcher::_record_telemetry(const String &p_event, const String &p_error_type) {
	if (p_event == "auto_fix_shown") {
		telemetry.auto_fix_shown++;
	} else if (p_event == "auto_fix_applied") {
		telemetry.auto_fix_applied++;
	} else if (p_event == "auto_fix_undone") {
		telemetry.auto_fix_undone++;
	}
	
	if (!p_error_type.is_empty()) {
		if (!telemetry.error_type_counts.has(p_error_type)) {
			telemetry.error_type_counts[p_error_type] = 0;
		}
		telemetry.error_type_counts[p_error_type]++;
	}
}

void ErrorWatcher::initialize(ErrorWatcherPanel *p_panel) {
	error_panel = p_panel;
}

void ErrorWatcher::shutdown() {
	if (error_panel) {
		error_panel = nullptr;
	}
	detected_errors.clear();
	file_errors.clear();
}

void ErrorWatcher::process_compiler_output(const String &p_output) {
	Vector<String> lines = p_output.split("\n");
	
	for (const String &line : lines) {
		// Parse common error formats: "file:line:column: error: message"
		if (line.contains(": error:") || line.contains(": warning:")) {
			int first_colon = line.find(":");
			int second_colon = line.find(":", first_colon + 1);
			int third_colon = line.find(":", second_colon + 1);
			
			if (first_colon != -1 && second_colon != -1 && third_colon != -1) {
				String file_path = line.substr(0, first_colon);
				String line_str = line.substr(first_colon + 1, second_colon - first_colon - 1);
				String col_str = line.substr(second_colon + 1, third_colon - second_colon - 1);
				String message = line.substr(third_colon + 1).strip_edges();
				
				if (line_str.is_numeric() && col_str.is_numeric()) {
					int line_num = line_str.to_int();
					int col_num = col_str.to_int();
					
					ErrorWatcherError error = _parse_compiler_error(file_path, line_num, col_num, message);
					add_error(error);
				}
			}
		}
	}
}

void ErrorWatcher::process_runtime_error(const String &p_file, int p_line, int p_column, const String &p_message) {
	ErrorWatcherError error = _parse_runtime_error(p_file, p_line, p_column, p_message);
	add_error(error);
}

void ErrorWatcher::process_script_errors(const List<ScriptLanguage::ScriptError> &p_errors, const String &p_file_path) {
	for (const ScriptLanguage::ScriptError &script_error : p_errors) {
		ErrorWatcherError error = _parse_compiler_error(
			p_file_path.is_empty() ? script_error.path : p_file_path,
			script_error.line,
			script_error.column,
			script_error.message
		);
		add_error(error);
	}
}

void ErrorWatcher::add_error(const ErrorWatcherError &p_error) {
	detected_errors.push_back(p_error);
	
	if (!file_errors.has(p_error.file_path)) {
		file_errors[p_error.file_path] = Vector<ErrorWatcherError>();
	}
	file_errors[p_error.file_path].push_back(p_error);
	
	if (error_panel) {
		error_panel->add_error(p_error);
	}
	
	// Record telemetry
	_record_telemetry("error_detected", String::num_int64((int)p_error.type));
}

void ErrorWatcher::clear_errors() {
	detected_errors.clear();
	file_errors.clear();
	
	if (error_panel) {
		error_panel->clear_errors();
	}
}

void ErrorWatcher::clear_file_errors(const String &p_file_path) {
	if (file_errors.has(p_file_path)) {
		file_errors.erase(p_file_path);
	}
	
	// Remove from global list
	for (int i = detected_errors.size() - 1; i >= 0; i--) {
		if (detected_errors[i].file_path == p_file_path) {
			detected_errors.remove_at(i);
		}
	}
	
	if (error_panel) {
		error_panel->set_errors(detected_errors);
	}
}

Vector<ErrorWatcherError> ErrorWatcher::get_errors_for_file(const String &p_file_path) {
	if (file_errors.has(p_file_path)) {
		return file_errors[p_file_path];
	}
	return Vector<ErrorWatcherError>();
}

Vector<QuickFixAction> ErrorWatcher::get_quick_fixes(const ErrorWatcherError &p_error) {
	return _generate_quick_fixes(p_error);
}

bool ErrorWatcher::apply_quick_fix(const QuickFixAction &p_action) {
	Ref<FileAccess> file = FileAccess::open(p_action.file_path, FileAccess::READ);
	if (file.is_null()) {
		return false;
	}
	
	String content = file->get_as_text();
	file->close();
	
	Vector<String> lines = content.split("\n");
	if (p_action.line <= 0 || p_action.line > lines.size()) {
		return false;
	}
	
	// Apply the fix based on action type
	bool success = false;
	
	switch (p_action.type) {
		case QuickFixAction::RENAME_VARIABLE: {
			if (p_action.line <= lines.size()) {
				String line = lines[p_action.line - 1];
				String new_line = line.replace(p_action.old_text, p_action.new_text);
				if (new_line != line) {
					lines.write[p_action.line - 1] = new_line;
					success = true;
				}
			}
		} break;
		
		case QuickFixAction::ADD_IMPORT: {
			lines.insert(0, p_action.new_text.strip_edges());
			success = true;
		} break;
		
		case QuickFixAction::ADD_VARIABLE_DECLARATION: {
			if (p_action.line <= lines.size()) {
				lines.insert(p_action.line - 1, p_action.new_text.strip_edges());
				success = true;
			}
		} break;
		
		default:
			break;
	}
	
	if (success) {
		// Write the modified content back
		file = FileAccess::open(p_action.file_path, FileAccess::WRITE);
		if (file.is_valid()) {
			String new_content = String("\n").join(lines);
			file->store_string(new_content);
			file->close();
			
			_record_telemetry("auto_fix_applied", String::num_int64((int)p_action.type));
			return true;
		}
	}
	
	return false;
}

bool ErrorWatcher::apply_quick_fix_with_undo(const QuickFixAction &p_action, CodeTextEditor *p_editor) {
	if (!p_editor) {
		return apply_quick_fix(p_action);
	}
	
	CodeEdit *text_editor = p_editor->get_text_editor();
	if (!text_editor) {
		return apply_quick_fix(p_action);
	}
	
	// Begin an undo group for the fix operation
	text_editor->begin_complex_operation();
	
	bool success = false;
	
	switch (p_action.type) {
		case QuickFixAction::RENAME_VARIABLE: {
			// Find and replace the variable name in the current line
			int line_idx = p_action.line - 1;
			if (line_idx >= 0 && line_idx < text_editor->get_line_count()) {
				String line_text = text_editor->get_line(line_idx);
				String new_line = line_text.replace(p_action.old_text, p_action.new_text);
				if (new_line != line_text) {
					text_editor->set_line(line_idx, new_line);
					success = true;
				}
			}
		} break;
		
		case QuickFixAction::ADD_IMPORT: {
			// Insert import at the top of the file
			text_editor->insert_line_at(0, p_action.new_text.strip_edges());
			success = true;
		} break;
		
		case QuickFixAction::ADD_VARIABLE_DECLARATION: {
			// Insert variable declaration before the error line
			int line_idx = p_action.line - 1;
			if (line_idx >= 0 && line_idx <= text_editor->get_line_count()) {
				text_editor->insert_line_at(line_idx, p_action.new_text.strip_edges());
				success = true;
			}
		} break;
		
		case QuickFixAction::FIX_SYNTAX: {
			// Apply basic syntax fixes
			int line_idx = p_action.line - 1;
			if (line_idx >= 0 && line_idx < text_editor->get_line_count()) {
				String line_text = text_editor->get_line(line_idx);
				String fixed_line = line_text;
				
				// Apply common syntax fixes
				if (p_action.description.contains("colon")) {
					fixed_line = line_text.rstrip(" \t") + ":";
				} else if (p_action.description.contains("parenthesis")) {
					if (p_action.description.contains("opening")) {
						fixed_line = "(" + line_text;
					} else if (p_action.description.contains("closing")) {
						fixed_line = line_text + ")";
					}
				}
				
				if (fixed_line != line_text) {
					text_editor->set_line(line_idx, fixed_line);
					success = true;
				}
			}
		} break;
		
		default:
			break;
	}
	
	// End the undo group
	text_editor->end_complex_operation();
	
	if (success) {
		_record_telemetry("auto_fix_applied", String::num_int64((int)p_action.type));
		
		// Trigger validation to update error markers
		if (p_editor) {
			p_editor->validate_script();
		}
	}
	
	return success;
}

String ErrorWatcher::preview_quick_fix(const QuickFixAction &p_action) {
	return p_action.preview_text;
}

Dictionary ErrorWatcher::get_telemetry_data() {
	Dictionary data;
	data["auto_fix_shown"] = telemetry.auto_fix_shown;
	data["auto_fix_applied"] = telemetry.auto_fix_applied;
	data["auto_fix_undone"] = telemetry.auto_fix_undone;
	
	Dictionary error_types;
	for (const KeyValue<String, int> &kv : telemetry.error_type_counts) {
		error_types[kv.key] = kv.value;
	}
	data["error_type_counts"] = error_types;
	
	return data;
}

void ErrorWatcher::reset_telemetry() {
	telemetry.auto_fix_shown = 0;
	telemetry.auto_fix_applied = 0;
	telemetry.auto_fix_undone = 0;
	telemetry.error_type_counts.clear();
}

void ErrorWatcher::integrate_with_script_editor(ScriptTextEditor *p_editor) {
	// TODO: Add integration hooks
}

void ErrorWatcher::integrate_with_code_editor(CodeTextEditor *p_editor) {
	// TODO: Add integration hooks
}

ErrorWatcher::ErrorWatcher() {
	singleton = this;
}

ErrorWatcher::~ErrorWatcher() {
	if (singleton == this) {
		singleton = nullptr;
	}
}