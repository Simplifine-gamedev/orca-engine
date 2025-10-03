/*
 * © 2025 Simplifine Corp.
 * Personal Non‑Commercial License applies to this file as an original contribution to this Godot fork.
 * See LICENSES/COMPANY-NONCOMMERCIAL.md for terms. Commercial use requires a separate license from Simplifine.
 */

#pragma once

#include "scene/gui/box_container.h"

class AIChatDock;
class TextEdit;
class Button;
class PanelContainer;

// Component for the chat input box UI (bottom of AI chat)
class AIChatInputBox {
public:
	static void create_input_ui(AIChatDock *p_chat_dock, VBoxContainer *p_parent_container);
	static void style_send_button(Button *p_send_button, AIChatDock *p_chat_dock);
	static void style_stop_button(Button *p_stop_button, AIChatDock *p_chat_dock, bool p_enabled);
};

