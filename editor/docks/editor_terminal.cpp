/**************************************************************************/
/*  editor_terminal.cpp                                                   */
/**************************************************************************/

#include "editor_terminal.h"

void EditorTerminal::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_delayed_focus"), &EditorTerminal::_delayed_focus);
}

void EditorTerminal::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_READY: {
			_setup_ui();
		} break;
	}
}

void EditorTerminal::_setup_ui() {
	set_name("Terminal");
	
	// Output area
	output = memnew(RichTextLabel);
	output->set_use_bbcode(true);
	output->set_scroll_follow(true);
	output->set_selection_enabled(true);
	output->set_context_menu_enabled(true);
	output->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	output->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	// Prevent output from grabbing focus
	output->set_focus_mode(Control::FOCUS_NONE);
	add_child(output);
	
	// Input container
	input_container = memnew(HBoxContainer);
	add_child(input_container);
	
	// Prompt
	prompt_label = memnew(Label);
	prompt_label->set_text("$ ");
	prompt_label->add_theme_color_override("font_color", Color(0.4, 0.8, 0.4));
	input_container->add_child(prompt_label);
	
	// Input field
	input_field = memnew(LineEdit);
	input_field->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	input_field->set_placeholder("Enter CLI commands (e.g., ls -la, grep pattern *.txt, git status, find . -name \"*.gd\")");
	input_field->connect("text_submitted", callable_mp(this, &EditorTerminal::_on_input_submitted));
	input_field->connect("gui_input", callable_mp(this, &EditorTerminal::_on_input_gui_input));
	input_container->add_child(input_field);
	
	// Send button
	send_button = memnew(Button);
	send_button->set_text("Run");
	send_button->connect("pressed", callable_mp(this, &EditorTerminal::_on_send_pressed));
	input_container->add_child(send_button);
	
	// Setup allowed commands
	_setup_allowed_commands();
	
	// Welcome message
	_add_output("[b]Terminal Ready[/b] - Standard CLI commands available", false);
	_add_output("Try: [color=lightblue]ls -la[/color], [color=lightblue]git status[/color], [color=lightblue]grep -r \"pattern\" .[/color], [color=lightblue]find . -name \"*.txt\"[/color]", false);
	
	// Set initial focus on input field (use call_deferred to ensure UI is ready)
	input_field->call_deferred("grab_focus");
}

void EditorTerminal::_setup_allowed_commands() {
	allowed_commands.clear();
	// File and directory operations
	allowed_commands.insert("ls");
	allowed_commands.insert("dir");
	allowed_commands.insert("pwd");
	allowed_commands.insert("cd");
	allowed_commands.insert("cat");
	allowed_commands.insert("head");
	allowed_commands.insert("tail");
	allowed_commands.insert("find");
	allowed_commands.insert("grep");
	allowed_commands.insert("tree");
	allowed_commands.insert("cp");
	allowed_commands.insert("mv");
	allowed_commands.insert("mkdir");
	allowed_commands.insert("rm");
	allowed_commands.insert("touch");
	allowed_commands.insert("chmod");
	// Text processing
	allowed_commands.insert("sort");
	allowed_commands.insert("uniq");
	allowed_commands.insert("wc");
	allowed_commands.insert("diff");
	allowed_commands.insert("cut");
	allowed_commands.insert("awk");
	allowed_commands.insert("sed");
	// Version control
	allowed_commands.insert("git");
	// Network utilities
	allowed_commands.insert("curl");
	allowed_commands.insert("wget");
	allowed_commands.insert("ping");
	// System utilities
	allowed_commands.insert("echo");
	allowed_commands.insert("which");
	allowed_commands.insert("whoami");
	allowed_commands.insert("uname");
	allowed_commands.insert("date");
	allowed_commands.insert("uptime");
	// Process management
	allowed_commands.insert("ps");
	allowed_commands.insert("top");
	allowed_commands.insert("htop");
	allowed_commands.insert("kill");
	// Scripting
	allowed_commands.insert("python");
	allowed_commands.insert("python3");
	allowed_commands.insert("node");
	allowed_commands.insert("sh");
	allowed_commands.insert("bash");
}

void EditorTerminal::_on_send_pressed() {
	String command = input_field->get_text().strip_edges();
	if (!command.is_empty()) {
		_on_input_submitted(command);
	}
	// Ensure focus returns to input field after clicking send button
	_ensure_input_focus();
}

void EditorTerminal::_on_input_submitted(const String &p_text) {
	String command = p_text.strip_edges();
	if (command.is_empty()) {
		return;
	}
	
	// Store command in history (avoid duplicates of the last command)
	if (command_history.is_empty() || command_history[command_history.size() - 1] != command) {
		command_history.push_back(command);
		// Limit history size to prevent memory issues
		if (command_history.size() > 100) {
			command_history.remove_at(0);
		}
	}
	
	// Reset history navigation
	history_index = -1;
	current_input = "";
	
	// Clear input
	input_field->clear();
	
	// Keep focus on input field for continuous typing - use aggressive focus management
	_ensure_input_focus();
	
	// Show command in output
	_add_output("[b]$ " + command + "[/b]", false);
	
	// Execute command
	_execute_command(command);
	
	// Extra focus insurance after everything is done
	_ensure_input_focus();
}

void EditorTerminal::_execute_command(const String &p_command) {
	// Get project root for command context  
	String project_root = ProjectSettings::get_singleton()->get_resource_path();
	
	// Parse command into executable and arguments
	Vector<String> parts = p_command.split(" ");
	if (parts.is_empty()) {
		_add_output("Error: Empty command", true);
		return;
	}
	
	String executable = parts[0];
	List<String> arguments;
	for (int i = 1; i < parts.size(); i++) {
		arguments.push_back(parts[i]);
	}
	
	// Handle special commands that don't need execution
	if (executable == "cd") {
		_add_output("[color=orange]Info: All commands run in project root: " + project_root + "[/color]", false);
		return;
	}
	
	if (executable == "pwd") {
		_add_output(project_root, false);
		return;
	}
	
	// Security check - only allow safe commands
	if (!allowed_commands.has(executable)) {
		_add_output("[color=orange]Warning: Command '" + executable + "' not in allow list.[/color]", true);
		_add_output("[color=orange]Allowed: ls, pwd, cd, grep, find, git, cat, cp, mv, curl, python, etc.[/color]", true);
		return;
	}
	
	// For Git commands, use -C to specify working directory
	List<String> final_arguments;
	if (executable == "git") {
		final_arguments.push_back("-C");
		final_arguments.push_back(project_root);
		// Add original arguments
		for (const String &arg : arguments) {
			final_arguments.push_back(arg);
		}
	} else {
		// For other commands, use original arguments
		final_arguments = arguments;
	}
	
	// Execute command
	String output_text;
	int exit_code;
	Error err = OS::get_singleton()->execute(executable, final_arguments, &output_text, &exit_code, true, nullptr, false);
	
	if (err == OK) {
		if (exit_code == 0) {
			if (!output_text.is_empty()) {
				_add_output(output_text, false);
			} else {
				_add_output("[color=green]Command completed successfully (no output)[/color]", false);
			}
		} else {
			_add_output("[color=orange]Command exited with code " + String::num_int64(exit_code) + "[/color]", true);
			if (!output_text.is_empty()) {
				_add_output(output_text, true);
			}
		}
	} else {
		_add_output("[color=red]Failed to execute command: " + executable + "[/color]", true);
	}
	
	// Ensure focus returns to input field after command execution
	_ensure_input_focus();
}

void EditorTerminal::_add_output(const String &p_text, bool p_is_error) {
	if (!output) {
		return;
	}
	
	String formatted_text;
	if (p_is_error) {
		formatted_text = "[color=lightcoral]" + p_text + "[/color]";
	} else {
		formatted_text = p_text;
	}
	
	output->append_text(formatted_text + "\n");
}

void EditorTerminal::_on_input_gui_input(const Ref<InputEvent> &p_event) {
	Ref<InputEventKey> key_event = p_event;
	if (key_event.is_valid() && key_event->is_pressed()) {
		if (key_event->get_keycode() == Key::UP) {
			_navigate_history_up();
			// Prevent the event from being processed further
			input_field->accept_event();
		} else if (key_event->get_keycode() == Key::DOWN) {
			_navigate_history_down();
			// Prevent the event from being processed further
			input_field->accept_event();
		}
	}
}

void EditorTerminal::_navigate_history_up() {
	if (command_history.is_empty()) {
		return;
	}
	
	// Store current input if we're starting history navigation
	if (history_index == -1) {
		current_input = input_field->get_text();
	}
	
	// Move up in history (towards older commands)
	if (history_index < command_history.size() - 1) {
		history_index++;
		int actual_index = command_history.size() - 1 - history_index;
		input_field->set_text(command_history[actual_index]);
		input_field->set_caret_column(input_field->get_text().length());
		// Ensure focus stays on input field
		input_field->grab_focus();
	}
}

void EditorTerminal::_navigate_history_down() {
	if (history_index == -1) {
		return; // Not in history navigation mode
	}
	
	// Move down in history (towards newer commands)
	if (history_index > 0) {
		history_index--;
		int actual_index = command_history.size() - 1 - history_index;
		input_field->set_text(command_history[actual_index]);
		input_field->set_caret_column(input_field->get_text().length());
	} else {
		// Return to current input
		history_index = -1;
		input_field->set_text(current_input);
		input_field->set_caret_column(input_field->get_text().length());
	}
	// Ensure focus stays on input field
	input_field->grab_focus();
}

EditorTerminal::EditorTerminal() {
}

EditorTerminal::~EditorTerminal() {
}

void EditorTerminal::grab_focus() {
	if (input_field) {
		input_field->grab_focus();
	}
}

void EditorTerminal::_ensure_input_focus() {
	if (!input_field) {
		return;
	}
	
	// Multiple approaches to ensure focus stays
	input_field->grab_focus();
	input_field->call_deferred("grab_focus");
	
	// Also try a slightly delayed approach for stubborn cases
	call_deferred("_delayed_focus");
}

void EditorTerminal::_delayed_focus() {
	if (input_field) {
		input_field->grab_focus();
	}
}
