/*
 * © 2025 Simplifine Corp.
 * Personal Non‑Commercial License applies to this file as an original contribution to this Godot fork.
 * See LICENSES/COMPANY-NONCOMMERCIAL.md for terms. Commercial use requires a separate license from Simplifine.
 */

#pragma once

#include "core/variant/dictionary.h"

class AIChatDock;

class AIChatDockCheckpoint {
public:
	// Git checkpoints system (uses project git directly)
	static void init_checkpoints_repo(AIChatDock *p_dock);
	static bool create_checkpoint(AIChatDock *p_dock, const String &p_message, int p_message_index);
	static String get_checkpoint_name(int p_message_index);
	static bool restore_from_checkpoint(AIChatDock *p_dock, int p_message_index);
	
	// Checkpoint UI handlers
	static void on_checkpoint_message_pressed(AIChatDock *p_dock, int p_message_index);
	static void on_restore_checkpoint_confirmed(AIChatDock *p_dock);
	
	// Post-restore refresh helpers
	static void safely_reopen_scene_after_checkpoint(AIChatDock *p_dock, const String &p_scene_path);
	static void verify_scene_reopened(AIChatDock *p_dock, const String &p_expected_scene_path);
	static void force_editor_refresh_after_checkpoint(AIChatDock *p_dock);
	static void trigger_external_change_detection(AIChatDock *p_dock);
	static void final_ui_refresh_after_checkpoint(AIChatDock *p_dock);
	static void immediate_post_restore_refresh(AIChatDock *p_dock);
	
	// Scene context summarization helper
	static Dictionary summarize_scene_node_for_context(const Dictionary &p_node, int p_max_depth, int p_max_children);
};


