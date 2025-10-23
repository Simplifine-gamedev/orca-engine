/**************************************************************************/
/*  ai_terminal_ui.h                                                      */
/**************************************************************************/
/*  Terminal UI components for AI chat tool results                      */
/**************************************************************************/

#pragma once

#include "scene/gui/box_container.h"
#include "core/variant/dictionary.h"

class AiTerminalUI {
public:
	// Create terminal-like UI for terminal_manager tool results
	static void create_terminal_ui(VBoxContainer *p_content_vbox, const Dictionary &p_args, const Dictionary &p_result, bool p_success);
	
private:
	// Helper methods for creating terminal UI components
	static void _create_command_header(VBoxContainer *p_content_vbox, const String &p_command, int p_exit_code, bool p_shell_used);
	static void _create_output_panel(VBoxContainer *p_content_vbox, const String &p_output);
	static void _create_execution_details(VBoxContainer *p_content_vbox, const String &p_working_dir, bool p_shell_used, uint64_t p_exec_time);
	static void _create_commands_grid(VBoxContainer *p_content_vbox, const Array &p_commands);
};
