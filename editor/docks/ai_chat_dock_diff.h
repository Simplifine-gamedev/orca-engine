/*
 * © 2025 Simplifine Corp.
 * Personal Non‑Commercial License applies to this file as an original contribution to this Godot fork.
 * See LICENSES/COMPANY-NONCOMMERCIAL.md for terms. Commercial use requires a separate license from Simplifine.
 */

#pragma once

#include "core/variant/dictionary.h"

class AIChatDock;

class AIChatDockDiff {
public:
	// Diff handling system
	static void on_diff_accepted(AIChatDock *p_dock, const String &p_path, const String &p_content);
	static void apply_file_edit_immediate(AIChatDock *p_dock, const String &p_path, const String &p_content);
	
	// Script editor diff integration
	static void on_script_editor_diff_accepted(AIChatDock *p_dock, const String &p_path, const String &p_content);
	static void on_script_editor_diff_rejected(AIChatDock *p_dock, const String &p_path);
	static void on_script_editor_save(AIChatDock *p_dock, const String &p_path);
	static void connect_script_editor_signals(AIChatDock *p_dock);
	
	// Diff display methods
	static void show_diff_in_script_editor(AIChatDock *p_dock, const String &p_path, const String &p_original, const String &p_modified, const String &p_inline_diff = "");
	static void show_diff_in_script_editor_deferred(AIChatDock *p_dock, const String &p_path, const String &p_original, const String &p_modified, const String &p_inline_diff = "");
	static void show_cumulative_diff_for_file(AIChatDock *p_dock, const String &p_path, const String &p_original, const String &p_final, const String &p_inline_diff = "");
	static String generate_inline_diff(const String &p_original, const String &p_modified);
	
	// Pending edit tracking
	static void register_pending_edit(AIChatDock *p_dock, const String &p_path, const NodePath &p_btns_path, const NodePath &p_status_label_path);
	static void clear_pending_edit(AIChatDock *p_dock, const String &p_path);
	
	// Unified accept/reject system
	static void handle_apply_edit_accepted(AIChatDock *p_dock, const String &p_file_path, const String &p_content);
	static void handle_apply_edit_rejected(AIChatDock *p_dock, const String &p_file_path);
	
	// Pending edits banner system
	static void update_pending_edits_banner(AIChatDock *p_dock);
	static void add_pending_edit(AIChatDock *p_dock, const String &p_tool_call_id, const String &p_file_path);
	static void remove_pending_edit(AIChatDock *p_dock, const String &p_tool_call_id);
	static void remove_pending_edits_for_file(AIChatDock *p_dock, const String &p_file_path);
	static void on_banner_clicked(AIChatDock *p_dock);
};


