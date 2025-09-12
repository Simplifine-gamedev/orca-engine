/**************************************************************************/
/*  error_watcher.h                                                       */
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

#include "core/object/object.h"
#include "core/string/ustring.h"
#include "core/variant/variant.h"
#include "scene/gui/control.h"

class CodeTextEditor;
class ScriptTextEditor;
class RichTextLabel;
class VBoxContainer;
class HBoxContainer;
class Button;
class Label;
class PopupPanel;

struct ErrorWatcherError {
	enum Type {
		DUPLICATE_VARIABLE,
		MISSING_IMPORT,
		UNDEFINED_VARIABLE,
		SYNTAX_ERROR,
		TYPE_MISMATCH,
		UNKNOWN
	};

	Type type = UNKNOWN;
	String file_path;
	int line = -1;
	int column = -1;
	String message;
	String original_error;
	bool has_quick_fix = false;
	String suggested_fix;
	String fix_preview;
};

struct QuickFixAction {
	enum ActionType {
		RENAME_VARIABLE,
		ADD_IMPORT,
		ADD_VARIABLE_DECLARATION,
		FIX_SYNTAX,
		REMOVE_DUPLICATE
	};

	ActionType type;
	String description;
	String file_path;
	int line = -1;
	int column = -1;
	String old_text;
	String new_text;
	String preview_text;
};

class ErrorWatcherPanel : public Control {
	GDCLASS(ErrorWatcherPanel, Control);

private:
	VBoxContainer *main_container = nullptr;
	HBoxContainer *header_container = nullptr;
	Label *error_count_label = nullptr;
	Button *clear_all_button = nullptr;
	RichTextLabel *error_list = nullptr;

	Vector<ErrorWatcherError> current_errors;

	void _clear_all_errors();
	void _error_item_clicked(const Variant &p_meta);
	void _update_error_display();

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	void add_error(const ErrorWatcherError &p_error);
	void clear_errors();
	void set_errors(const Vector<ErrorWatcherError> &p_errors);
	Vector<ErrorWatcherError> get_errors() const { return current_errors; }
	int get_error_count() const { return current_errors.size(); }

	ErrorWatcherPanel();
	~ErrorWatcherPanel();
};

class ErrorWatcher : public Object {
	GDCLASS(ErrorWatcher, Object);

private:
	static ErrorWatcher *singleton;
	
	ErrorWatcherPanel *error_panel = nullptr;
	Vector<ErrorWatcherError> detected_errors;
	HashMap<String, Vector<ErrorWatcherError>> file_errors;

	// Telemetry tracking
	struct TelemetryData {
		int auto_fix_shown = 0;
		int auto_fix_applied = 0;
		int auto_fix_undone = 0;
		HashMap<String, int> error_type_counts;
	} telemetry;

	// Error parsing methods
	ErrorWatcherError::Type _classify_error(const String &p_message);
	ErrorWatcherError _parse_compiler_error(const String &p_file, int p_line, int p_column, const String &p_message);
	ErrorWatcherError _parse_runtime_error(const String &p_file, int p_line, int p_column, const String &p_message);
	Vector<QuickFixAction> _generate_quick_fixes(const ErrorWatcherError &p_error);

	// Quick fix implementations
	QuickFixAction _create_rename_variable_fix(const ErrorWatcherError &p_error);
	QuickFixAction _create_add_import_fix(const ErrorWatcherError &p_error);
	QuickFixAction _create_add_declaration_fix(const ErrorWatcherError &p_error);
	QuickFixAction _create_syntax_fix(const ErrorWatcherError &p_error);

	// File analysis helpers
	String _get_file_content(const String &p_file_path);
	Vector<String> _get_file_lines(const String &p_file_path);
	String _extract_variable_name(const String &p_error_message);
	String _suggest_import_for_symbol(const String &p_symbol);
	bool _is_duplicate_variable_error(const String &p_message);
	bool _is_undefined_variable_error(const String &p_message);
	bool _is_missing_import_error(const String &p_message);

	void _record_telemetry(const String &p_event, const String &p_error_type = "");

protected:
	static void _bind_methods();

public:
	static ErrorWatcher *get_singleton() { return singleton; }

	void initialize(ErrorWatcherPanel *p_panel);
	void shutdown();

	// Main API methods
	void process_compiler_output(const String &p_output);
	void process_runtime_error(const String &p_file, int p_line, int p_column, const String &p_message);
	void process_script_errors(const List<ScriptLanguage::ScriptError> &p_errors, const String &p_file_path);
	
	void add_error(const ErrorWatcherError &p_error);
	void clear_errors();
	void clear_file_errors(const String &p_file_path);
	
	Vector<ErrorWatcherError> get_errors_for_file(const String &p_file_path);
	Vector<ErrorWatcherError> get_all_errors() const { return detected_errors; }
	
	// Quick fix methods
	Vector<QuickFixAction> get_quick_fixes(const ErrorWatcherError &p_error);
	bool apply_quick_fix(const QuickFixAction &p_action);
	bool apply_quick_fix_with_undo(const QuickFixAction &p_action, CodeTextEditor *p_editor);
	String preview_quick_fix(const QuickFixAction &p_action);
	
	// Telemetry methods
	Dictionary get_telemetry_data();
	void reset_telemetry();

	// Integration with existing error systems
	void integrate_with_script_editor(ScriptTextEditor *p_editor);
	void integrate_with_code_editor(CodeTextEditor *p_editor);

	ErrorWatcher();
	~ErrorWatcher();
};

class QuickFixPopup : public PopupPanel {
	GDCLASS(QuickFixPopup, PopupPanel);

private:
	VBoxContainer *main_container = nullptr;
	Label *error_label = nullptr;
	VBoxContainer *fixes_container = nullptr;
	HBoxContainer *button_container = nullptr;
	Button *apply_button = nullptr;
	Button *preview_button = nullptr;
	Button *cancel_button = nullptr;
	RichTextLabel *preview_text = nullptr;

	ErrorWatcherError current_error;
	Vector<QuickFixAction> available_fixes;
	int selected_fix_index = -1;

	void _fix_selected(int p_index);
	void _apply_fix();
	void _preview_fix();
	void _cancel_fix();
	void _update_preview();

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	void show_fixes(const ErrorWatcherError &p_error, const Vector<QuickFixAction> &p_fixes);
	void hide_popup();

	QuickFixPopup();
	~QuickFixPopup();
};