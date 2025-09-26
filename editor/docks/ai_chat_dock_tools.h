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

class AIChatDock;
class PanelContainer;
class Button;

class AIChatDockTools {
public:
	// Tool execution management
	static void execute_tool_calls(AIChatDock *p_dock, const Array &p_tool_calls);
	static void execute_apply_edit_async(AIChatDock *p_dock, const String &p_tool_call_id, const Dictionary &p_args);
	static void execute_file_edit_deferred(AIChatDock *p_dock, const String &p_tool_call_id, const String &p_operation, const Dictionary &p_args);
	static void execute_frontend_tool_deferred(AIChatDock *p_dock, const String &p_tool_call_id, const String &p_function_name, const String &p_arguments_str);
	
	// Thread management for async tools
	static void apply_edit_thread(void *p_userdata);
	static void on_apply_edit_thread_done(AIChatDock *p_dock);
	
	// Tool UI creation and management
	static void create_tool_call_bubbles(AIChatDock *p_dock, const Array &p_tool_calls);
	static void update_tool_placeholder_with_result(AIChatDock *p_dock, const AIChatDockTypes::ChatMessage &p_tool_message);
	static void create_tool_specific_ui(VBoxContainer *p_content_vbox, const String &p_tool_name, const Dictionary &p_result, bool p_success, const Dictionary &p_args = Dictionary());
	
	// Tool placeholder management
	static void update_tool_placeholder_status(AIChatDock *p_dock, const String &p_tool_id, const String &p_tool_name, const String &p_status);
	static void update_tool_placeholder_with_description(AIChatDock *p_dock, const String &p_tool_id, const String &p_tool_name, const String &p_status, const String &p_description);
	static void create_backend_tool_placeholder(AIChatDock *p_dock, const String &p_tool_id, const String &p_tool_name);
	static void create_assistant_message_for_backend_tool(AIChatDock *p_dock, const String &p_tool_name);
	
	// Tool response handling
	static void add_tool_response_to_chat(AIChatDock *p_dock, const String &p_tool_call_id, const String &p_name, const Dictionary &p_args, const Dictionary &p_result);
	static void apply_tool_result_deferred(AIChatDock *p_dock, const String &p_tool_call_id, const String &p_tool_name, const String &p_content, const Array &p_tool_results);
	static void on_tool_result_retry_timeout(AIChatDock *p_dock, const String &p_tool_call_id, const String &p_tool_name, const String &p_content, const Array &p_tool_results);
	
	// Tool output management
	static void on_tool_output_toggled(Control *p_content);
	
	// Tool button management
	static void on_tool_call_accept_pressed(AIChatDock *p_dock, const String &p_tool_call_id, const String &p_file_path, const String &p_content);
	static void on_tool_call_reject_pressed(AIChatDock *p_dock, const String &p_tool_call_id, const String &p_file_path);
	static void on_tool_result_accept_pressed(AIChatDock *p_dock, const String &p_tool_call_id, const String &p_file_path, const String &p_content, const NodePath &p_btns_path, const NodePath &p_status_path);
	static void on_tool_result_reject_pressed(AIChatDock *p_dock, const String &p_tool_call_id, const String &p_file_path, const NodePath &p_btns_path, const NodePath &p_status_path);
	
	// Apply/discard edit handlers
	static void on_apply_preview_to_editor(AIChatDock *p_dock, const String &p_path, const String &p_content, const NodePath &p_btns_path = NodePath(), const NodePath &p_status_label_path = NodePath());
	static void on_discard_preview(AIChatDock *p_dock, const String &p_path, const NodePath &p_btns_path = NodePath(), const NodePath &p_status_label_path = NodePath());
	
	// Tool status generation
	static String generate_descriptive_tool_status(const String &p_tool_name, const Dictionary &p_args, const Dictionary &p_result, bool p_success);
	static String generate_executing_tool_message(const String &p_tool_name);
	static String get_immediate_tool_status(const String &p_tool_name, const String &p_arguments_str);
	
	// Tool UI button management
	static void update_tool_call_button_status(AIChatDock *p_dock, const String &p_tool_call_id, const String &p_status);
	static void update_tool_call_button_status_in_container(VBoxContainer *p_container, const String &p_tool_call_id, const String &p_status);
	static void add_apply_edit_buttons_to_tool_container(AIChatDock *p_dock, VBoxContainer *p_container, const String &p_tool_call_id, const Dictionary &p_args, const Dictionary &p_result);
	
	// Lazy UI helpers
	static void create_lazy_node_props_ui(VBoxContainer *p_content_vbox, const Dictionary &p_result, const Dictionary &p_args);
	static void on_load_node_props_pressed(Button *p_button, VBoxContainer *p_target_vbox);
	static void render_full_node_props(VBoxContainer *p_target_vbox, const Dictionary &p_full_result);
	
	// Tool result truncation and expansion
	static bool should_truncate_tool_result(const String &p_tool_name, const Dictionary &p_result);
	static void create_truncated_tool_ui(VBoxContainer *p_content_vbox, const String &p_tool_name, const Dictionary &p_result);
	static String generate_tool_result_summary(const String &p_tool_name, const Dictionary &p_result, bool p_success);
	static void expand_truncated_tool_result(Button *p_expand_button, VBoxContainer *p_content_vbox);
	static void expand_full_nodes_list(Button *p_expand_button, VBoxContainer *p_nodes_vbox);
	static void expand_full_files_tree(Button *p_expand_button, VBoxContainer *p_files_vbox);
	static void expand_full_search_results(Button *p_expand_button, VBoxContainer *p_search_vbox);
};
