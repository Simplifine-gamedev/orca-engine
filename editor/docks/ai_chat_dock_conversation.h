/*
 * © 2025 Simplifine Corp.
 * Personal Non‑Commercial License applies to this file as an original contribution to this Godot fork.
 * See LICENSES/COMPANY-NONCOMMERCIAL.md for terms. Commercial use requires a separate license from Simplifine.
 */

#pragma once

#include "ai_chat_dock_types.h"
#include "core/variant/dictionary.h"
#include "core/variant/array.h"
#include "core/os/thread.h"
#include "core/os/mutex.h"

class AIChatDock;

class AIChatDockConversation {
public:
	// Conversation file management
	static void load_conversations(AIChatDock *p_dock);
	static void save_conversations(AIChatDock *p_dock);
	static void save_conversations_async(AIChatDock *p_dock);
	static void save_conversations_to_disk(AIChatDock *p_dock, const String &p_json_data);
	static void save_conversations_chunked(AIChatDock *p_dock, int p_start_index);
	static void finalize_conversations_save(AIChatDock *p_dock);
	
	// Background saving
	static void queue_delayed_save(AIChatDock *p_dock);
	static void execute_delayed_save(AIChatDock *p_dock);
	static void background_save(void *p_userdata);
	static void on_background_save_finished(AIChatDock *p_dock);
	
	// Conversation creation and management
	static void create_new_conversation(AIChatDock *p_dock);
	static void create_new_conversation_instant(AIChatDock *p_dock);
	static void switch_to_conversation(AIChatDock *p_dock, int p_index);
	static void update_conversation_dropdown(AIChatDock *p_dock);
	
	// Conversation UI handlers
	static void on_conversation_selected(AIChatDock *p_dock, int p_index);
	static void on_new_conversation_pressed(AIChatDock *p_dock);
	static void on_conversation_rename_requested(AIChatDock *p_dock, int p_index, const String &p_new_name);
	static void on_conversation_delete_requested(AIChatDock *p_dock, int p_index);
	
	// Export functionality
	static void on_export_button_pressed(AIChatDock *p_dock);
	static void on_export_file_selected(AIChatDock *p_dock, const String &p_file_path);
	
	// Utility methods
	static String generate_conversation_id();
	static String generate_conversation_title(const Vector<AIChatDockTypes::ChatMessage> &p_messages);
	
	// Token management and summarization
	static int estimate_token_count(const String &p_text);
	static int calculate_conversation_tokens(const Vector<AIChatDockTypes::ChatMessage> &p_messages);
	static int get_model_token_limit(const String &p_model);
	static void check_and_trigger_summarization(AIChatDock *p_dock);
	static void on_summarization_request_completed(AIChatDock *p_dock, int p_result, int p_code, const PackedStringArray &p_headers, const PackedByteArray &p_body);
};
