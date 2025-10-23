/**************************************************************************/
/*  editor_terminal_plugin.cpp                                           */
/**************************************************************************/

#include "editor_terminal_plugin.h"
#include "editor_terminal.h"
#include "editor/editor_node.h"
#include "editor/gui/editor_bottom_panel.h"
#include "editor/editor_string_names.h"
#include "editor/settings/editor_settings.h"

void EditorTerminalPlugin::make_visible(bool p_visible) {
	if (p_visible) {
		EditorNode::get_bottom_panel()->make_item_visible(terminal);
		// Focus the input field when terminal becomes visible
		if (terminal->is_visible_in_tree()) {
			// Use call_deferred to ensure the terminal is fully ready
			terminal->call_deferred("grab_focus");
		}
	} else {
		if (terminal->is_visible_in_tree()) {
			EditorNode::get_bottom_panel()->hide_bottom_panel();
		}
	}
}

EditorTerminalPlugin::EditorTerminalPlugin() {
	terminal = memnew(EditorTerminal);
	terminal_button = EditorNode::get_bottom_panel()->add_item("Terminal", terminal);
}

EditorTerminalPlugin::~EditorTerminalPlugin() {
	if (terminal) {
		EditorNode::get_bottom_panel()->remove_item(terminal);
	}
}
