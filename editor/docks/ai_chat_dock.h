/**************************************************************************/
/*  ai_chat_dock.h                                                        */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#pragma once

#include "ai_tool_server.h"
#include "common.h"
#include "core/io/http_client.h"
#include "core/io/image.h"
#include "diff_viewer.h"
#include "scene/gui/box_container.h"
#include "scene/gui/dialogs.h"
#include "scene/gui/tab_container.h"
#include "scene/gui/texture_rect.h"
#include "scene/main/http_request.h"

class Button;
class MenuButton;
class ConfigFile;
class EditorFileDialog;
class EditorFileSystemDirectory;
class EditorTools;
class HTTPClient;
class HTTPRequest;
class Node;
class OptionButton;
class RichTextLabel;
class ScrollContainer;
class StyleBoxFlat;
class TextEdit;
class Tree;
class TreeItem;
class PopupPanel;
class PopupMenu;
class DiffViewer;
class ScriptEditor;
class ScriptTextEditor;
class HFlowContainer;

class AIChatDock : public VBoxContainer {
	GDCLASS(AIChatDock, VBoxContainer);

private:
	DiffViewer *diff_viewer;
	Ref<AIToolServer> tool_server;
	// Helper to find RichTextLabel recursively.
	static RichTextLabel *find_rich_text_label_in_children(Node *p_node) {
		if (!p_node) {
			return nullptr;
		}
		RichTextLabel *rt = Object::cast_to<RichTextLabel>(p_node);
		if (rt) {
			return rt;
		}
		for (int i = 0; i < p_node->get_child_count(); i++) {
			RichTextLabel *child_rt = find_rich_text_label_in_children(p_node->get_child(i));
			if (child_rt) {
				return child_rt;
			}
		}
		return nullptr;
	}

	struct AttachedFile {
		String path;
		String name;
		String content;
		bool is_image = false;
		String mime_type;
		String base64_data; // For images encoded for API
		Vector2i original_size = Vector2i(0, 0);
		Vector2i display_size = Vector2i(0, 0);
		bool was_downsampled = false;
		// Node support
		bool is_node = false;
		NodePath node_path;
		String node_type;
	};

	struct ChatMessage {
		String role; // "user", "assistant", or "tool"
		String content;
		String timestamp;
		// For assistant tool calls.
		Array tool_calls;
		// For tool responses.
		String tool_call_id;
		String name;
		// For attached files
		Vector<AttachedFile> attached_files;
		// For storing tool execution results (like generated images)
		Array tool_results;
	};

	// Track pending apply_edit operations: path -> {bubble_node_path, status_label_path, bubble_container}
	Dictionary pending_edits;

	struct Conversation {
		String id;
		String title;
		String created_timestamp;
		String last_modified_timestamp;
		Vector<ChatMessage> messages;
	};

	// Attachment safety limits to protect model context
	static const int64_t MAX_TEXT_ATTACHMENT_PREVIEW_BYTES = 64 * 1024; // Read at most 64 KiB from disk
	static const int MAX_TEXT_ATTACHMENT_PREVIEW_CHARS = 20000; // And cap decoded text length

	ScrollContainer *chat_scroll = nullptr;
	VBoxContainer *chat_container = nullptr;
	OptionButton *model_dropdown = nullptr;
	OptionButton *conversation_history_dropdown = nullptr;
	Button *new_conversation_button = nullptr;
    Button *index_button = nullptr; // Removed from UI; kept to avoid widespread ref changes
	TextEdit *input_field = nullptr;
	Button *send_button = nullptr;
	Button *stop_button = nullptr;
	MenuButton *attach_button = nullptr;
	HFlowContainer *attached_files_container = nullptr;
	EditorFileDialog *file_dialog = nullptr;
	EditorFileDialog *save_image_dialog = nullptr;
	AcceptDialog *image_warning_dialog = nullptr;

	// Embedding system state
	HTTPRequest *embedding_request = nullptr;
	bool embedding_system_initialized = false;
	bool initial_indexing_done = false;
	bool embedding_request_busy = false;
	// Periodic polling for indexing
	Timer *embedding_poll_timer = nullptr;
	int embedding_poll_seconds = 120; // 2 minutes
	uint64_t last_index_request_ms = 0;
	bool pending_fs_changes = false;
	// Accumulate changed files for periodic batching
	HashSet<String> pending_changed_files;
	// Embedding progress UI
	Label *embedding_status_label = nullptr;
	Timer *embedding_status_timer = nullptr;
	bool embedding_in_progress = false;
	int embedding_status_dots = 0;
	String embedding_status_base;
	Dictionary current_batch_info;

	// User authentication
	HTTPRequest *auth_request = nullptr;
	HTTPRequest *auth_providers_request = nullptr;
	Button *login_button = nullptr;
	Label *user_status_label = nullptr;
	String current_user_id;
	String current_user_name;
	String auth_token;
	String pending_login_provider;
	String provider_pending_login;

	// Login polling
	Timer *login_poll_timer = nullptr;
	int login_poll_attempts = 0;
	int login_poll_max_attempts = 30;
	PopupPanel *at_mention_popup = nullptr;
	Tree *at_mention_tree = nullptr;
	PopupPanel *scene_tree_popup = nullptr;
	Tree *scene_tree = nullptr;
	EditorFileDialog *resource_dialog = nullptr;

	// Related graph UI
	PopupPanel *related_graph_panel = nullptr;
	Tree *related_graph_tree = nullptr;
	TabContainer *graph_tabs = nullptr;

	// For saving images
	String pending_save_image_data; // Base64 image data to save
	String pending_save_image_format; // "png" or "jpg"

	// For delayed saving performance optimization
	bool save_pending = false;

	// Background saving to prevent UI freezing
	Thread *save_thread = nullptr;
	Mutex *save_mutex = nullptr;
	bool save_thread_busy = false;

  // Async tool execution tracking (prevents UI freeze for long tools like apply_edit)
  int pending_tool_tasks = 0;
  bool ui_rebuild_in_progress = false;
  bool editing_in_progress = false;

  // Incremental conversation rebuild state
  Vector<ChatMessage> rebuild_msgs;
  int rebuild_next_index = 0;
  Timer *rebuild_timer = nullptr;
  Timer *loading_anim_timer = nullptr;
  Control *loading_overlay = nullptr;
  Mutex *apply_edit_mutex = nullptr;
  Vector<void *> apply_edit_done; // stores ApplyEditTaskData* as opaque pointers

	// Streaming HTTP client (replaces HTTPRequest for streaming support)
	Ref<HTTPClient> http_client;
	// Separate HTTP request for stop requests (non-streaming)
	HTTPRequest *stop_http_request = nullptr;
	enum HTTPStatus {
		STATUS_IDLE,
		STATUS_CONNECTING,
		STATUS_REQUESTING,
		STATUS_BODY,
		STATUS_DONE
	};
	HTTPStatus http_status = STATUS_IDLE;
	String pending_request_path;
	PackedStringArray pending_request_headers;
	PackedByteArray pending_request_body;

	RichTextLabel *current_assistant_message_label = nullptr;
	String response_buffer;
	Array _chunked_messages; // For processing large conversations in chunks
	Array _chunked_conversations_array; // For async saving
	int _chunked_save_index = 0;

	Vector<Conversation> conversations;
	int current_conversation_index = -1;
	Vector<AttachedFile> current_attached_files;
	String conversations_file_path;
	String api_key;
    // Default API endpoint; will be overridden at runtime based on IS_DEV env
    String api_endpoint = "http://127.0.0.1:8000/chat";
	String model = "gpt-5";

	bool is_waiting_for_response = false;

	// Track displayed images to prevent duplication
	HashSet<String> current_displayed_images;

	// Stop mechanism
	String current_request_id;
	bool stop_requested = false;
	bool stream_completed_successfully = false;
	// Timer to defer heavy conversation save to idle time
	Timer *save_timer = nullptr;

	void _on_send_button_pressed();
	void _on_stop_button_pressed();
	void _on_show_project_graph_pressed();
	void _process_send_request_async();
	void _send_stop_request();
	void _on_stop_request_completed(int p_result, int p_code, const PackedStringArray &p_headers, const PackedByteArray &p_body);
	void _on_edit_message_pressed(int p_message_index);
	void _on_edit_message_send_pressed(int p_message_index, const String &p_new_content);
	void _on_edit_send_button_pressed(Button *p_button);
	void _on_edit_message_cancel_pressed(int p_message_index);
	void _on_edit_field_gui_input(const Ref<InputEvent> &p_event, Button *p_send_button);
  // Build-only helper that constructs an edit bubble panel for a message without
  // adding spacers or attaching it to the chat container.
  PanelContainer *_build_edit_message_panel(const ChatMessage &p_message, int p_message_index);
	void _create_edit_message_bubble(const ChatMessage &p_message, int p_message_index);
	void _save_conversations_async();
	void _on_input_text_changed();
	void _on_input_field_gui_input(const Ref<InputEvent> &p_event);
	void _update_at_mention_popup();
	void _populate_at_mention_tree(const String &p_filter = "");
	void _populate_tree_recursive(EditorFileSystemDirectory *p_dir, TreeItem *p_parent, const String &p_filter);
	void _on_at_mention_item_selected();
	void _on_model_selected(int p_index);
	void _on_index_button_pressed();
    void _on_editor_resource_saved(Object *p_res);
    void _on_editor_scene_saved(const String &p_path);
	void _on_tool_output_toggled(Control *p_content);
	void _on_attachment_menu_item_pressed(int p_id);
	void _on_attach_files_pressed();
	void _on_attach_scene_nodes_pressed();
	void _on_attach_current_script_pressed();
	void _on_attach_resources_pressed();
	void _on_reindex_project_pressed();
	void _perform_project_reindex();
	void _on_reindex_response(int p_result, int p_response_code, const PackedStringArray &p_headers, const PackedByteArray &p_body);
	void _on_scene_tree_node_selected();
	void _on_files_selected(const Vector<String> &p_files);
	void _on_remove_attachment(const String &p_path);
	void _on_conversation_selected(int p_index);
	void _on_new_conversation_pressed();
	void _on_save_image_pressed(const String &p_base64_data, const String &p_format);
	void _on_save_image_location_selected(const String &p_file_path);
	void _save_conversations_to_disk(const String &p_json_data);
	void _process_image_attachment_async(const String &p_file_path, const String &p_name, const String &p_mime_type);
	void _handle_response_chunk(const PackedByteArray &p_chunk);
	void _process_ndjson_line(const String &p_line);
	void _execute_tool_calls(const Array &p_tool_calls);
  // Async apply_edit helpers
  void _execute_apply_edit_async(const String &p_tool_call_id, const Dictionary &p_args);
  static void _apply_edit_thread(void *p_userdata);
  void _on_apply_edit_thread_done();
  void _finalize_simple_tool_async(const String &p_tool_call_id, const String &p_name, const Dictionary &p_args, const Dictionary &p_result);
  void _run_rig_tool_on_main(const String &p_tool_call_id, const String &p_name, const Dictionary &p_args);
  // Async auto-rig job helpers
  void _on_autorig_start_completed(int p_result, int p_code, const PackedStringArray &p_headers, const PackedByteArray &p_body, String p_tool_call_id, String p_name, Dictionary p_args, String p_base_url);
  void _on_autorig_status_completed(int p_result, int p_code, const PackedStringArray &p_headers, const PackedByteArray &p_body, String p_tool_call_id, String p_name, Dictionary p_args, String p_base_url, String p_job_id);
  void _on_autorig_result_completed(int p_result, int p_code, const PackedStringArray &p_headers, const PackedByteArray &p_body, String p_tool_call_id, String p_name, Dictionary p_args);
  void _on_rebuild_tick();
	RichTextLabel *_get_or_create_current_assistant_message_label();
	void _create_tool_call_bubbles(const Array &p_tool_calls);
	void _update_tool_placeholder_with_result(const ChatMessage &p_tool_message);
	void _create_tool_specific_ui(VBoxContainer *p_content_vbox, const String &p_tool_name, const Dictionary &p_result, bool p_success, const Dictionary &p_args = Dictionary());
	void _toggle_expand_label(RichTextLabel *p_label, Button *p_button, const String &p_full_text, const String &p_snippet_text);
	void _toggle_docs_card(Control *p_snippet_node, Control *p_full_node, Button *p_button, VBoxContainer *p_holder);
	void _rebuild_conversation_ui(const Vector<ChatMessage> &p_messages);
	void _apply_tool_result_deferred(const String &p_tool_call_id, const String &p_tool_name, const String &p_content, const Array &p_tool_results);
	void _build_hierarchy_tree_item(Tree *p_tree, TreeItem *p_parent, const Dictionary &p_node_data);

	void _ensure_related_graph_ui();
	void _populate_related_graph(const Dictionary &p_graph);
	void _show_related_graph(const Dictionary &p_graph);

	void _add_message_to_chat(const String &p_role, const String &p_content, const Array &p_tool_calls = Array());
	void _add_tool_response_to_chat(const String &p_tool_call_id, const String &p_name, const Dictionary &p_args, const Dictionary &p_result);
	void _send_chat_request();
	void _send_chat_request_chunked(int p_start_index);
	Dictionary _build_api_message(const ChatMessage &p_msg);
	void _finalize_chat_request();
	void _update_ui_state();
	void _create_message_bubble(const AIChatDock::ChatMessage &p_message, int p_message_index = -1);
	void _build_message_content(PanelContainer *p_message_panel, const AIChatDock::ChatMessage &p_message, int p_message_index);
	void _update_attached_files_display();
	void _clear_attachments();
	String _get_timestamp();
	void _scroll_to_bottom();

	// Helper to truncate overly large text before sending to the model
	String _truncate_text_for_context(const String &p_text, int p_max_chars = MAX_TEXT_ATTACHMENT_PREVIEW_CHARS);

	// Conversation management
	void _load_conversations();
	void _save_conversations();
	void _save_conversations_chunked(int p_start_index);
	void _finalize_conversations_save();
	void _queue_delayed_save();
	void _execute_delayed_save();
	static void _background_save(void *p_userdata);
	void _on_background_save_finished();
	void _create_new_conversation();
	void _create_new_conversation_instant();
	void _switch_to_conversation(int p_index);
	void _update_conversation_dropdown();
	String _generate_conversation_id();
	String _generate_conversation_title(const Vector<AIChatDock::ChatMessage> &p_messages);
	Vector<AIChatDock::ChatMessage> &_get_current_chat_history();

	void _on_tool_file_link_pressed(const String &p_path);
	String _convert_to_godot_path(const String &p_path);

	void _save_layout_to_config(Ref<ConfigFile> p_layout, const String &p_section) const;
	void _load_layout_from_config(Ref<ConfigFile> p_layout, const String &p_section);

	// Markdown to BBCode conversion
	String _markdown_to_bbcode(const String &p_markdown);
	String _process_inline_markdown(String p_line);

	void _on_diff_accepted(const String &p_path, const String &p_content);
	void _on_apply_preview_to_editor(const String &p_path, const String &p_content, const NodePath &p_btns_path = NodePath(), const NodePath &p_status_label_path = NodePath());
	void _on_discard_preview(const String &p_path, const NodePath &p_btns_path = NodePath(), const NodePath &p_status_label_path = NodePath());
	void _mark_apply_discard_buttons_hidden_and_status(Node *p_context, const String &p_status_text);
	void _on_script_editor_save(const String &p_path);
	void _register_pending_edit(const String &p_path, const NodePath &p_btns_path, const NodePath &p_status_label_path);
	void _clear_pending_edit(const String &p_path);
	void _connect_script_editor_signals();
	void _show_diff_in_script_editor(const String &p_path, const String &p_original, const String &p_modified);
	void _show_diff_in_script_editor_deferred(const String &p_path, const String &p_original, const String &p_modified);

	// Image processing methods
	bool _is_image_file(const String &p_path);
	String _get_mime_type_from_extension(const String &p_path);
	bool _process_image_attachment(AttachedFile &p_file);
	Vector2i _calculate_downsampled_size(const Vector2i &p_original, int p_max_dimension = 1024);

	// UI validation helpers
	bool _is_label_descendant_of_node(Node *p_label, Node *p_node);
	void _show_image_warning_dialog(const String &p_filename, const Vector2i &p_original, const Vector2i &p_new_size);

	// Image generation handling
	void _handle_generated_image(const String &p_base64_data, const String &p_id);
	void _display_generated_image_deferred(const String &p_base64_data, const String &p_id);
	void _display_generated_image_in_tool_result(VBoxContainer *p_container, const String &p_base64_data, const Dictionary &p_data);

	// Unified image display method for all image types
	void _display_image_unified(VBoxContainer *p_container, const String &p_base64_data, const Dictionary &p_metadata = Dictionary());

	// Utility: save base64-encoded image to a path (png/jpg based on extension)
	bool _save_base64_image_to_path(const String &p_base64_data, const String &p_file_path);

	// Apply edited content directly to disk for non-script resources (e.g., .tscn)
	void _apply_file_edit_immediate(const String &p_path, const String &p_content);

	// Drag and drop support
	bool can_drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from) const;
	void drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from);
	bool can_drop_data(const Point2 &p_point, const Variant &p_data) const override;
	void drop_data(const Point2 &p_point, const Variant &p_data) override;
	void _attach_dragged_files(const Vector<String> &p_files);
	void _attach_external_files(const Vector<String> &p_files);
	void _attach_dragged_nodes(const Array &p_nodes);
	String _get_file_type_icon(const AttachedFile &p_file);

	// Attachment helpers
	void _attach_scene_node(Node *p_node);
	void _attach_current_script();
	void _populate_scene_tree_recursive(Node *p_node, TreeItem *p_parent);
	String _get_node_info_string(Node *p_node);

	// Tool execution feedback
	void _update_tool_placeholder_status(const String &p_tool_id, const String &p_tool_name, const String &p_status);
	void _create_backend_tool_placeholder(const String &p_tool_id, const String &p_tool_name);
	void _create_assistant_message_for_backend_tool(const String &p_tool_name);

	// Track tool call handling to avoid duplicate UI updates when both
	// the editor and backend report the same tool execution.
	HashSet<String> tool_calls_executed_locally;
	HashSet<String> tool_calls_ui_applied;
	HashSet<String> tool_result_deferred_pending;
	Dictionary tool_result_retry_counts; // key: tool_call_id -> int tries

	// Global AI-busy flag for overlay scoping and UI state correctness
	int ai_busy_count = 0;
	Mutex ai_busy_mutex;

	void _schedule_tool_result_retry(const String &p_tool_call_id, const String &p_tool_name, const String &p_content, const Array &p_tool_results);
	void _on_tool_result_retry_timeout(const String &p_tool_call_id, const String &p_tool_name, const String &p_content, const Array &p_tool_results);


	// Embedding system for project indexing
	void _initialize_embedding_system();
	void _perform_initial_indexing();
	void _on_filesystem_changed();
	void _on_sources_changed(bool p_exist);
	void _update_file_embedding(const String &p_file_path);
	void _remove_file_embedding(const String &p_file_path);
	void _send_embedding_request(const String &p_action, const Dictionary &p_data = Dictionary());
	void _on_embedding_request_completed(int p_result, int p_code, const PackedStringArray &p_headers, const PackedByteArray &p_body);
	void _on_filesystem_debounced_scan(uint64_t p_scheduled_at);
	void _perform_filesystem_scan_changes();
	bool _should_index_file(const String &p_file_path);
	String _get_project_root_path();
	String _get_embed_base_url();
	void _set_embedding_status(const String &p_text, bool p_busy);
	void _on_embedding_status_tick();
	void _on_embedding_poll_tick();

	// Cloud-ready file indexing methods
	void _scan_and_index_project_files();
	void _scan_directory_recursive(const String &p_dir_path, const String &p_project_root, Array &p_file_contents, int &p_files_processed, int &p_files_skipped);
	Dictionary _read_file_for_indexing(const String &p_file_path, const String &p_project_root);
	String _calculate_content_hash(const String &p_content);
	void _send_file_batch(const Array &p_all_files, int p_start_index, int p_batch_size, int p_current_batch, int p_total_batches);
	
	// Smart context attachment based on embeddings
	void _suggest_relevant_files(const String &p_query);
	void _auto_attach_relevant_context();

	// User authentication methods
	void _setup_authentication_ui();
	void _on_login_button_pressed();
	void _on_auth_request_completed(int p_result, int p_code, const PackedStringArray &p_headers, const PackedByteArray &p_body);
	void _on_auth_dialog_action(const StringName &p_action);
	void _on_auth_provider_selected(int p_id);
	void _on_auth_providers_request_completed(int p_result, int p_code, const PackedStringArray &p_headers, const PackedByteArray &p_body);
	void _check_authentication_status();
	void _auto_verify_saved_credentials();
	void _start_login_polling();
	void _poll_login_status();
	void _stop_login_polling();
	void _ensure_project_indexing();
	void _update_user_status();
	void _logout_user();
	bool _is_user_authenticated() const;

protected:
	void _notification(int p_notification);
	static void _bind_methods();

public:
	AIChatDock();
	~AIChatDock();

	void clear_chat_history();
	void clear_current_conversation();
	void set_api_key(const String &p_api_key);
	void set_api_endpoint(const String &p_endpoint);
	void set_model(const String &p_model);
	void send_error_message(const String &p_error_text);

	// Authentication getters for EditorTools
	String get_current_user_id() const { return current_user_id; }
	String get_machine_id() const;
	String get_auth_token() const { return auth_token; }

	// Attach an external file (e.g., screenshot) to the current chat input attachments.
	void attach_external_file(const String &p_file_path);
};
