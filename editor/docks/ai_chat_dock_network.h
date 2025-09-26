/*
 * © 2025 Simplifine Corp.
 * Personal Non‑Commercial License applies to this file as an original contribution to this Godot fork.
 * See LICENSES/COMPANY-NONCOMMERCIAL.md for terms. Commercial use requires a separate license from Simplifine.
 */

#pragma once

#include "ai_chat_dock_types.h"
#include "core/variant/dictionary.h"
#include "core/variant/array.h"
#include "core/io/http_client.h"

class AIChatDock;

class AIChatDockNetwork {
public:
	// Request processing
	static void process_send_request_async(AIChatDock *p_dock);
	static void send_chat_request(AIChatDock *p_dock);
	static void send_chat_request_chunked(AIChatDock *p_dock, int p_start_index);
	static Dictionary build_api_message(const AIChatDockTypes::ChatMessage &p_msg);
	static void finalize_chat_request(AIChatDock *p_dock);
	
	// Response handling
	static void handle_response_chunk(AIChatDock *p_dock, const PackedByteArray &p_chunk);
	static void process_ndjson_line(AIChatDock *p_dock, const String &p_line);
	static void request_completed(AIChatDock *p_dock);
	
	// Stop mechanism
	static void on_stop_button_pressed(AIChatDock *p_dock);
	static void send_stop_request(AIChatDock *p_dock);
	static void on_stop_request_completed(AIChatDock *p_dock, int p_result, int p_code, const PackedStringArray &p_headers, const PackedByteArray &p_body);
	static void reset_connection_error_flag(AIChatDock *p_dock);
	
	// Model selection
	static void on_model_selected(AIChatDock *p_dock, int p_index);
	static void populate_all_models(AIChatDock *p_dock);
	static void on_models_request_completed(AIChatDock *p_dock, int p_result, int p_code, const PackedStringArray &p_headers, const PackedByteArray &p_body);
	static String get_api_base_url();
	
	// Input handling
	static void on_send_button_pressed(AIChatDock *p_dock);
	static void on_input_text_changed(AIChatDock *p_dock);
	
	// Version compatibility
	static void add_version_headers_to_request(PackedStringArray &p_headers);
	static void check_version_compatibility_on_startup(AIChatDock *p_dock);
	static void on_version_check_completed(AIChatDock *p_dock, int p_result, int p_code, const PackedStringArray &p_headers, const PackedByteArray &p_body);
	static void handle_version_mismatch(AIChatDock *p_dock, const Dictionary &p_version_info);
	
	// Utility methods
	static String get_timestamp();
	static String convert_to_godot_path(const String &p_path);
};
