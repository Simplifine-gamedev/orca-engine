/**************************************************************************/
/*  ai_chat_tool_styling.h                                                */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             ORCA ENGINE                                */
/**************************************************************************/

#ifndef AI_CHAT_TOOL_STYLING_H
#define AI_CHAT_TOOL_STYLING_H

#include "scene/gui/button.h"
#include "scene/gui/label.h"
#include "scene/gui/panel_container.h"
#include "scene/resources/style_box_flat.h"

class AIChatToolStyling {
public:
	// Style a tool label for an executing tool (blue -> monochrome with subtle animation)
	static void style_executing_tool_label(Label *p_label, Control *p_theme_source);
	
	// Style a tool result button (green/red -> monochrome)
	static void style_tool_result_button(Button *p_button, bool p_success, Control *p_theme_source);
	
	// Style a placeholder panel for tool results (transparent, no borders)
	static void style_tool_placeholder_panel(PanelContainer *p_panel, Control *p_theme_source);
	
	// Get monochrome color for tool status (no green/red)
	static Color get_tool_status_color(bool p_success, Control *p_theme_source);
};

#endif // AI_CHAT_TOOL_STYLING_H



