/*
 * © 2025 Simplifine Corp.
 * Personal Non‑Commercial License applies to this file as an original contribution to this Godot fork.
 * See LICENSES/COMPANY-NONCOMMERCIAL.md for terms. Commercial use requires a separate license from Simplifine.
 */

#pragma once

#include "ai_chat_dock_types.h"
#include "core/variant/dictionary.h"
#include "core/variant/array.h"
#include "scene/gui/box_container.h"
#include "scene/gui/panel_container.h"
#include "scene/gui/rich_text_label.h"
#include "scene/gui/tree.h"

class AIChatDock;
class Button;

class AIChatDockUI {
public:
	// Core UI notification handler
	static void handle_notification(AIChatDock *p_dock, int p_notification);
	
	// Message bubble creation
	static void create_message_bubble(AIChatDock *p_dock, const AIChatDockTypes::ChatMessage &p_message, int p_message_index = -1);
	static void build_message_content(PanelContainer *p_message_panel, const AIChatDockTypes::ChatMessage &p_message, int p_message_index);
	static void create_edit_message_bubble(AIChatDock *p_dock, const AIChatDockTypes::ChatMessage &p_message, int p_message_index);
	static PanelContainer *build_edit_message_panel(AIChatDock *p_dock, const AIChatDockTypes::ChatMessage &p_message, int p_message_index);
	
	// Message content management
	static void add_message_to_chat(AIChatDock *p_dock, const String &p_role, const String &p_content, const Array &p_tool_calls = Array());
	static RichTextLabel *get_or_create_current_assistant_message_label(AIChatDock *p_dock);
	
	// Conversation UI rebuild
	static void rebuild_conversation_ui(AIChatDock *p_dock, const Vector<AIChatDockTypes::ChatMessage> &p_messages);
	static void rebuild_conversation_ui_full(AIChatDock *p_dock);
	
	// Message editing handlers
	static void on_edit_message_pressed(AIChatDock *p_dock, int p_message_index);
	static void on_edit_send_button_pressed(AIChatDock *p_dock, Button *p_button);
	static void on_edit_message_send_pressed(AIChatDock *p_dock, int p_message_index, const String &p_new_content);
	static void on_edit_message_cancel_pressed(AIChatDock *p_dock, int p_message_index);
	static void on_edit_field_gui_input(AIChatDock *p_dock, const Ref<InputEvent> &p_event, Button *p_send_button);
	
	// Markdown processing
	static String markdown_to_bbcode(const String &p_markdown);
	static String process_inline_markdown(String p_line);
	
	// Scroll management
	static void scroll_to_bottom(AIChatDock *p_dock);
	static void perform_scroll(AIChatDock *p_dock);
	static bool is_at_bottom(AIChatDock *p_dock);
	static void on_chat_scroll_changed(AIChatDock *p_dock, float p_value);
	static void on_chat_content_min_size_changed(AIChatDock *p_dock);
	static void scroll_to_bottom_smooth(AIChatDock *p_dock);
	
	// UI state management
	static void update_ui_state(AIChatDock *p_dock);
	static bool is_busy(AIChatDock *p_dock);
	
	// Utility helpers
	static bool is_label_descendant_of_node(Node *p_label, Node *p_node);
	static String truncate_text_for_context(const String &p_text, int p_max_chars = 20000);
	
	// Performance helpers
	static int calculate_initial_message_start_index(AIChatDock *p_dock);
	static void apply_simplified_tool_result(AIChatDock *p_dock, const String &p_tool_call_id, const String &p_tool_name, const String &p_content);
	
	// Tree/hierarchy building
	static void build_hierarchy_tree_item(Tree *p_tree, TreeItem *p_parent, const Dictionary &p_node_data);
	
	// Related graph UI
	static void ensure_related_graph_ui(AIChatDock *p_dock);
	static void populate_related_graph(AIChatDock *p_dock, const Dictionary &p_graph);
	static void show_related_graph(AIChatDock *p_dock, const Dictionary &p_graph);
};
