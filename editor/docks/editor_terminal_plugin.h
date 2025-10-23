/**************************************************************************/
/*  editor_terminal_plugin.h                                             */
/**************************************************************************/

#pragma once

#include "editor/plugins/editor_plugin.h"

class EditorTerminal;
class Button;

class EditorTerminalPlugin : public EditorPlugin {
	GDCLASS(EditorTerminalPlugin, EditorPlugin);

private:
	EditorTerminal *terminal = nullptr;
	Button *terminal_button = nullptr;

public:
	void make_visible(bool p_visible) override;
	
	EditorTerminalPlugin();
	~EditorTerminalPlugin();
};
