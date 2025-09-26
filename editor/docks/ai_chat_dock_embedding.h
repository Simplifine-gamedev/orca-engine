/*
 * © 2025 Simplifine Corp.
 * Personal Non‑Commercial License applies to this file as an original contribution to this Godot fork.
 * See LICENSES/COMPANY-NONCOMMERCIAL.md for terms. Commercial use requires a separate license from Simplifine.
 */

#pragma once

#include "scene/main/http_request.h"
#include "scene/main/timer.h"
#include "scene/gui/label.h"

class AIChatDock;

class AIChatDockEmbedding {
public:
	// Initialize the embedding system
	static void initialize_embedding_system(AIChatDock *p_dock);
	
	// Perform initial project indexing
	static void perform_initial_indexing(AIChatDock *p_dock);
	
	// Check if project is already indexed
	static void check_index_status_and_start_if_needed(AIChatDock *p_dock);
	
	// File system change handlers
	static void on_filesystem_changed(AIChatDock *p_dock);
	static void on_sources_changed(AIChatDock *p_dock, bool p_exist);
	static void on_editor_resource_saved(AIChatDock *p_dock, Object *p_res);
	static void on_editor_scene_saved(AIChatDock *p_dock, const String &p_path);
	
	// Embedding request handlers
	static void send_embedding_request(AIChatDock *p_dock, const String &p_action, const Dictionary &p_data = Dictionary());
	static void on_embedding_request_completed(AIChatDock *p_dock, int p_result, int p_code, const PackedStringArray &p_headers, const PackedByteArray &p_body);
	static void on_index_status_response(AIChatDock *p_dock, int p_result, int p_response_code, const PackedStringArray &p_headers, const PackedByteArray &p_body);
	
	// File indexing utilities
	static void scan_and_index_project_files(AIChatDock *p_dock);
	static void scan_directory_recursive(const String &p_dir_path, const String &p_project_root, Array &p_file_contents, int &p_files_processed, int &p_files_skipped);
	static Dictionary read_file_for_indexing(const String &p_file_path, const String &p_project_root);
	static String calculate_content_hash(const String &p_content);
	static void send_file_batch(AIChatDock *p_dock, const Array &p_all_files, int p_start_index, int p_batch_size, int p_current_batch, int p_total_batches);
	
	// File change tracking
	static void update_file_embedding(AIChatDock *p_dock, const String &p_file_path);
	static void remove_file_embedding(AIChatDock *p_dock, const String &p_file_path);
	static void on_embedding_poll_tick(AIChatDock *p_dock);
	
	// Project reindexing
	static void perform_project_reindex(AIChatDock *p_dock);
	static void on_reindex_response(AIChatDock *p_dock, int p_result, int p_response_code, const PackedStringArray &p_headers, const PackedByteArray &p_body);
	
	// Utility methods
	static String get_project_root_path();
	static String get_embed_base_url(AIChatDock *p_dock);
	static bool should_index_file(const String &p_file_path);
	static void set_embedding_status(AIChatDock *p_dock, const String &p_text, bool p_busy);
	static void on_embedding_status_tick(AIChatDock *p_dock);
	
	// File system change debouncing
	static void on_filesystem_debounced_scan(AIChatDock *p_dock, uint64_t p_scheduled_at);
	static void perform_filesystem_scan_changes(AIChatDock *p_dock);
	
	// Auto context attachment
	static void suggest_relevant_files(AIChatDock *p_dock, const String &p_query);
	static void auto_attach_relevant_context(AIChatDock *p_dock);
	static void ensure_project_indexing(AIChatDock *p_dock);
};
