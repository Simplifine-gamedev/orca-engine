/**************************************************************************/
/*  editor_terminal.h                                                     */
/**************************************************************************/

#pragma once

#include "core/os/os.h"
#include "core/config/project_settings.h"
#include "scene/gui/box_container.h"
#include "scene/gui/button.h"
#include "scene/gui/label.h"
#include "scene/gui/line_edit.h"
#include "scene/gui/rich_text_label.h"

class EditorTerminal : public VBoxContainer {
	GDCLASS(EditorTerminal, VBoxContainer);

private:
	RichTextLabel *output = nullptr;
	HBoxContainer *input_container = nullptr;
	Label *prompt_label = nullptr;
	LineEdit *input_field = nullptr;
	Button *send_button = nullptr;

	// Security - allowed commands
	HashSet<String> allowed_commands;
	
	// Command history
	Vector<String> command_history;
	int history_index = -1;
	String current_input;

	void _setup_ui();
	void _setup_allowed_commands();
	void _on_send_pressed();
	void _on_input_submitted(const String &p_text);
	void _execute_command(const String &p_command);
	void _add_output(const String &p_text, bool p_is_error = false);
	
	// History navigation
	void _navigate_history_up();
	void _navigate_history_down();
	void _on_input_gui_input(const Ref<InputEvent> &p_event);
	
	// Focus management
	void _ensure_input_focus();
	void _delayed_focus();

protected:
	static void _bind_methods();
	void _notification(int p_what);

public:
	EditorTerminal();
	~EditorTerminal();
	
	void grab_focus();
};
