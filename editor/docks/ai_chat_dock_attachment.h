/*
 * © 2025 Simplifine Corp.
 * Personal Non‑Commercial License applies to this file as an original contribution to this Godot fork.
 * See LICENSES/COMPANY-NONCOMMERCIAL.md for terms. Commercial use requires a separate license from Simplifine.
 */

#pragma once

#include "ai_chat_dock_types.h"
#include "core/variant/dictionary.h"
#include "core/variant/array.h"
#include "scene/main/node.h"

class AIChatDock;
class TreeItem;

class AIChatDockAttachment {
public:
	// Attachment menu handlers
	static void on_attachment_menu_item_pressed(AIChatDock *p_dock, int p_id);
	static void on_attach_files_pressed(AIChatDock *p_dock);
	static void on_attach_scene_nodes_pressed(AIChatDock *p_dock);
	static void on_attach_current_script_pressed(AIChatDock *p_dock);
	static void on_attach_resources_pressed(AIChatDock *p_dock);
	
	// File selection handlers
	static void on_files_selected(AIChatDock *p_dock, const Vector<String> &p_files);
	static void on_remove_attachment(AIChatDock *p_dock, const String &p_path);
	static void update_attached_files_display(AIChatDock *p_dock);
	static void clear_attachments(AIChatDock *p_dock);
	
	// Drag and drop support
	static bool can_drop_data(AIChatDock *p_dock, const Point2 &p_point, const Variant &p_data);
	static void drop_data(AIChatDock *p_dock, const Point2 &p_point, const Variant &p_data);
	static bool can_drop_data_fw(AIChatDock *p_dock, const Point2 &p_point, const Variant &p_data, Control *p_from);
	static void drop_data_fw(AIChatDock *p_dock, const Point2 &p_point, const Variant &p_data, Control *p_from);
	
	// File attachment helpers
	static void attach_dragged_files(AIChatDock *p_dock, const Vector<String> &p_files);
	static void attach_external_files(AIChatDock *p_dock, const Vector<String> &p_files);
	static void attach_dragged_nodes(AIChatDock *p_dock, const Array &p_nodes);
	static String get_file_type_icon(const AIChatDockTypes::AttachedFile &p_file);
	
	// Scene node attachment
	static void attach_scene_node(AIChatDock *p_dock, Node *p_node);
	static void attach_current_script(AIChatDock *p_dock);
	static void on_scene_tree_node_selected(AIChatDock *p_dock);
	
	// Tree population helpers
	static void populate_scene_tree_recursive(Node *p_node, TreeItem *p_parent);
	static String get_node_info_string(Node *p_node);
	
	// At-mention system
	static void update_at_mention_popup(AIChatDock *p_dock);
	static void populate_at_mention_tree(AIChatDock *p_dock, const String &p_filter = "");
	static void populate_tree_recursive(EditorFileSystemDirectory *p_dir, TreeItem *p_parent, const String &p_filter);
	static void on_at_mention_item_selected(AIChatDock *p_dock);
	
	// Clipboard support
	static void handle_clipboard_paste(AIChatDock *p_dock);
	static void on_input_field_gui_input(AIChatDock *p_dock, const Ref<InputEvent> &p_event);
	
	// External file attachment
	static void attach_external_file(AIChatDock *p_dock, const String &p_file_path);
};
