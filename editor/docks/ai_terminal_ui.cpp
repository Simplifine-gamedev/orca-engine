/**************************************************************************/
/*  ai_terminal_ui.cpp                                                    */
/**************************************************************************/
/*  Terminal UI components for AI chat tool results                      */
/**************************************************************************/

#include "ai_terminal_ui.h"
#include "scene/gui/button.h"
#include "scene/gui/label.h"
#include "scene/gui/rich_text_label.h"
#include "scene/gui/text_edit.h"
#include "scene/gui/panel_container.h"
#include "scene/gui/scroll_container.h"
#include "scene/gui/separator.h"
#include "scene/gui/grid_container.h"
#include "scene/resources/style_box_flat.h"
#include "editor/themes/editor_scale.h"
#include "editor/editor_node.h"

void AiTerminalUI::create_terminal_ui(VBoxContainer *p_content_vbox, const Dictionary &p_args, const Dictionary &p_result, bool p_success) {
	String op = p_args.get("op", "");
	
	if (op == "execute") {
		// Create terminal-like UI for command execution results
		String command = p_args.get("command", "");
		String working_dir = p_result.get("working_directory", "");
		int exit_code = p_result.get("exit_code", 0);
		bool shell_used = p_result.get("shell_used", false);
		String output = p_result.get("output", "");
		bool has_output = p_result.get("has_output", false);
		uint64_t exec_time = p_result.get("execution_time", 0);
		
		// DEBUG: Log what we're getting from the result Dictionary
		print_line("TERMINAL_UI_DEBUG: Retrieved output from Dictionary");
		print_line("TERMINAL_UI_DEBUG: output.length() = " + String::num_int64(output.length()));
		print_line("TERMINAL_UI_DEBUG: output content (first 100 chars) = '" + output.substr(0, 100) + "'");
		
		// Check if output contains newlines properly
		int newline_count = 0;
		for (int i = 0; i < output.length(); i++) {
			if (output[i] == '\n') newline_count++;
		}
		print_line("TERMINAL_UI_DEBUG: newline count = " + String::num_int64(newline_count));
		
		// Create command header
		_create_command_header(p_content_vbox, command, exit_code, shell_used);
		
		// Create output section if there's output
		if (has_output && !output.is_empty()) {
			_create_output_panel(p_content_vbox, output);
		} else {
			// No output - just show a note
			Label *no_output_label = memnew(Label);
			no_output_label->set_text("(no output)");
			no_output_label->add_theme_color_override("font_color", EditorNode::get_singleton()->get_editor_theme()->get_color(SNAME("font_color"), SNAME("Editor")) * Color(1,1,1,0.6));
			no_output_label->add_theme_font_size_override("font_size", 11);
			p_content_vbox->add_child(no_output_label);
		}
		
		// Add execution details
		_create_execution_details(p_content_vbox, working_dir, shell_used, exec_time);
		return;
	}
	
	// Handle other operations (pwd, status, etc.)
	if (op == "pwd" || op == "status") {
		Label *info_label = memnew(Label);
		String message = p_result.get("message", "");
		info_label->set_text(message);
		p_content_vbox->add_child(info_label);
		return;
	}
	
	if (op == "allowed_commands") {
		Array commands = p_result.get("allowed_commands", Array());
		
		Label *header_label = memnew(Label);
		header_label->set_text("Allowed Commands (" + String::num_int64(commands.size()) + "):");
		header_label->add_theme_color_override("font_color", EditorNode::get_singleton()->get_editor_theme()->get_color(SNAME("font_color"), SNAME("Editor")));
		p_content_vbox->add_child(header_label);
		
		// Create commands grid
		_create_commands_grid(p_content_vbox, commands);
		return;
	}
	
	// Fallback for other operations
	Label *result_label = memnew(Label);
	String message = p_result.get("message", "");
	result_label->set_text(message);
	p_content_vbox->add_child(result_label);
}

void AiTerminalUI::_create_command_header(VBoxContainer *p_content_vbox, const String &p_command, int p_exit_code, bool p_shell_used) {
	// Terminal header showing command info
	HBoxContainer *terminal_header = memnew(HBoxContainer);
	p_content_vbox->add_child(terminal_header);
	
	// Terminal icon
	Label *terminal_icon = memnew(Label);
	terminal_icon->set_text("$ ");
	terminal_icon->add_theme_color_override("font_color", Color(0.4, 0.8, 0.4));
	terminal_header->add_child(terminal_icon);
	
	// Command text
	Label *command_label = memnew(Label);
	command_label->set_text(p_command);
	command_label->add_theme_color_override("font_color", Color(1.0, 1.0, 1.0));
	terminal_header->add_child(command_label);
	
	// Spacer
	Control *spacer = memnew(Control);
	spacer->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	terminal_header->add_child(spacer);
	
	// Exit code indicator
	Label *exit_label = memnew(Label);
	if (p_exit_code == 0) {
		exit_label->set_text("✓");
		exit_label->add_theme_color_override("font_color", EditorNode::get_singleton()->get_editor_theme()->get_color(SNAME("success_color"), SNAME("Editor")));
	} else {
		exit_label->set_text("✗ " + String::num_int64(p_exit_code));
		exit_label->add_theme_color_override("font_color", EditorNode::get_singleton()->get_editor_theme()->get_color(SNAME("error_color"), SNAME("Editor")));
	}
	terminal_header->add_child(exit_label);
	
	// Shell indicator
	if (p_shell_used) {
		Label *shell_indicator = memnew(Label);
		shell_indicator->set_text("🐚");
		shell_indicator->set_tooltip_text("Executed through shell (supports pipes, wildcards)");
		terminal_header->add_child(shell_indicator);
	}
}

void AiTerminalUI::_create_output_panel(VBoxContainer *p_content_vbox, const String &p_output) {
	// Terminal output panel with dark background
	PanelContainer *output_panel = memnew(PanelContainer);
	p_content_vbox->add_child(output_panel);
	
	Ref<StyleBoxFlat> terminal_style = memnew(StyleBoxFlat);
	terminal_style->set_bg_color(Color(0.1, 0.1, 0.1)); // Dark terminal background
	terminal_style->set_border_width_all(1);
	terminal_style->set_border_color(Color(0.3, 0.3, 0.3));
	terminal_style->set_corner_radius_all(4);
	terminal_style->set_content_margin_all(8);
	output_panel->add_theme_style_override("panel", terminal_style);
	
	// DEBUG: Analyze the actual output string in detail
	print_line("TERMINAL_UI_DEBUG: _create_output_panel called");
	print_line("TERMINAL_UI_DEBUG: p_output.length() = " + String::num_int64(p_output.length()));
	
	// Check for character-by-character storage issue
	if (p_output.length() > 10) {
		print_line("TERMINAL_UI_DEBUG: First 10 characters as codes:");
		for (int i = 0; i < 10 && i < p_output.length(); i++) {
			print_line("  [" + String::num_int64(i) + "] = '" + String::chr(p_output[i]) + "' (code: " + String::num_int64(p_output[i]) + ")");
		}
	}
	
	// Try to clean the output string of any weird characters or formatting
	String cleaned_output = p_output;
	
	// Check if this looks like character-by-character corruption
	bool looks_corrupted = false;
	if (cleaned_output.length() > 6) {
		// If every other character is a newline, we have character corruption
		int consecutive_single_chars = 0;
		for (int i = 0; i < MIN(20, cleaned_output.length() - 1); i++) {
			if (cleaned_output[i] != '\n' && cleaned_output[i + 1] == '\n') {
				consecutive_single_chars++;
			}
		}
		looks_corrupted = consecutive_single_chars > 5; // More than 5 single-char lines = likely corruption
		print_line("TERMINAL_UI_DEBUG: Corruption check - consecutive single chars: " + String::num_int64(consecutive_single_chars) + ", looks_corrupted: " + String(looks_corrupted ? "YES" : "NO"));
	}
	
	// If corrupted, try to reconstruct by removing spurious newlines
	if (looks_corrupted) {
		print_line("TERMINAL_UI_DEBUG: Attempting to fix corrupted output");
		String fixed_output = "";
		for (int i = 0; i < cleaned_output.length(); i++) {
			char32_t c = cleaned_output[i];
			// Skip lone newlines that separate single characters
			if (c == '\n' && i > 0 && i < cleaned_output.length() - 1) {
				char32_t prev = cleaned_output[i - 1];  
				char32_t next = cleaned_output[i + 1];
				// If previous and next aren't newlines, this might be spurious
				if (prev != '\n' && next != '\n' && prev != ' ' && next != ' ') {
					continue; // Skip this newline
				}
			}
			fixed_output += String::chr(c);
		}
		cleaned_output = fixed_output;
		print_line("TERMINAL_UI_DEBUG: Fixed output length = " + String::num_int64(cleaned_output.length()));
		print_line("TERMINAL_UI_DEBUG: Fixed content = '" + cleaned_output.substr(0, 100) + "'");
	}
	
	// Scrollable output
	ScrollContainer *output_scroll = memnew(ScrollContainer);
	output_scroll->set_custom_minimum_size(Size2(0, 120));
	output_scroll->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	output_panel->add_child(output_scroll);
	
	// Output text - use simple Label for most reliable text display
	Label *output_text = memnew(Label);
	output_text->set_vertical_alignment(VERTICAL_ALIGNMENT_TOP);
	output_text->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_LEFT);
	output_text->set_autowrap_mode(TextServer::AUTOWRAP_OFF);
	output_text->add_theme_color_override("font_color", Color(0.9, 0.9, 0.9));
	
	// Set the cleaned text
	output_text->set_text(cleaned_output);
	output_scroll->add_child(output_text);
}

void AiTerminalUI::_create_execution_details(VBoxContainer *p_content_vbox, const String &p_working_dir, bool p_shell_used, uint64_t p_exec_time) {
	// Add execution details
	HSeparator *separator = memnew(HSeparator);
	p_content_vbox->add_child(separator);
	
	HBoxContainer *details_container = memnew(HBoxContainer);
	p_content_vbox->add_child(details_container);
	
	Label *details_label = memnew(Label);
	String details = "Working directory: " + p_working_dir;
	if (p_shell_used) {
		details += " | Shell execution";
	}
	if (p_exec_time > 0) {
		details += " | " + String::num_uint64(p_exec_time) + "ms";
	}
	details_label->set_text(details);
	details_label->add_theme_color_override("font_color", EditorNode::get_singleton()->get_editor_theme()->get_color(SNAME("font_color"), SNAME("Editor")) * Color(1,1,1,0.7));
	details_label->add_theme_font_size_override("font_size", 10);
	details_container->add_child(details_label);
}

void AiTerminalUI::_create_commands_grid(VBoxContainer *p_content_vbox, const Array &p_commands) {
	// Create a grid layout for commands
	GridContainer *commands_grid = memnew(GridContainer);
	commands_grid->set_columns(6); // 6 columns of commands
	p_content_vbox->add_child(commands_grid);
	
	for (int i = 0; i < p_commands.size(); i++) {
		String cmd = p_commands[i];
		Label *cmd_label = memnew(Label);
		cmd_label->set_text(cmd);
		cmd_label->add_theme_color_override("font_color", Color(0.7, 0.9, 1.0)); // Light blue
		commands_grid->add_child(cmd_label);
	}
}
