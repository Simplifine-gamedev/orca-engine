/*
 * © 2025 Simplifine Corp.
 * Personal Non‑Commercial License applies to this file as an original contribution to this Godot fork.
 * See LICENSES/COMPANY-NONCOMMERCIAL.md for terms. Commercial use requires a separate license from Simplifine.
 */

#include "ai_chat_dock_embedding.h"
#include "ai_chat_dock.h"
#include "ai_chat_dock_auth.h"
#include "core/io/dir_access.h"
#include "core/io/file_access.h"
#include "core/io/json.h"
#include "core/os/os.h"
#include "core/object/object.h"
#include "core/object/callable_method_pointer.h"
#include "core/config/project_settings.h"
#include "editor/editor_node.h"
#include "editor/file_system/editor_file_system.h"
#include "editor/settings/editor_settings.h"
#include "../ai/editor_tools.h"

// ========== EMBEDDING SYSTEM IMPLEMENTATION ==========

void AIChatDockEmbedding::initialize_embedding_system(AIChatDock *p_dock) {
    print_line("AI Chat: Initializing cloud-based embedding system");

    // Connect to editor file system signals for automatic reindexing
    // This connection does not require authentication; it only sets up callbacks.
    if (EditorFileSystem::get_singleton()) {
        if (!EditorFileSystem::get_singleton()->is_connected("filesystem_changed", callable_mp(p_dock, &AIChatDock::_on_filesystem_changed))) {
            EditorFileSystem::get_singleton()->connect("filesystem_changed", callable_mp(p_dock, &AIChatDock::_on_filesystem_changed));
        }
        if (!EditorFileSystem::get_singleton()->is_connected("sources_changed", callable_mp(p_dock, &AIChatDock::_on_sources_changed))) {
            EditorFileSystem::get_singleton()->connect("sources_changed", callable_mp(p_dock, &AIChatDock::_on_sources_changed));
        }
        print_line("AI Chat: Connected to EditorFileSystem change signals (filesystem_changed, sources_changed)");
    } else {
        print_line("AI Chat: EditorFileSystem not ready; change signals not connected");
    }

    // Connect to precise save signals from EditorNode to index only changed files
    if (EditorNode::get_singleton()) {
        if (!EditorNode::get_singleton()->is_connected("resource_saved", callable_mp(p_dock, &AIChatDock::_on_editor_resource_saved))) {
            EditorNode::get_singleton()->connect("resource_saved", callable_mp(p_dock, &AIChatDock::_on_editor_resource_saved), Object::CONNECT_DEFERRED);
        }
        if (!EditorNode::get_singleton()->is_connected("scene_saved", callable_mp(p_dock, &AIChatDock::_on_editor_scene_saved))) {
            EditorNode::get_singleton()->connect("scene_saved", callable_mp(p_dock, &AIChatDock::_on_editor_scene_saved), Object::CONNECT_DEFERRED);
        }
        print_line("AI Chat: Connected to EditorNode save signals (resource_saved, scene_saved)");
    }

    // Create HTTPRequest for embedding API calls
    if (!p_dock->embedding_request) {
        p_dock->embedding_request = memnew(HTTPRequest);
        p_dock->add_child(p_dock->embedding_request);
        p_dock->embedding_request->connect("request_completed", callable_mp(p_dock, &AIChatDock::_on_embedding_request_completed));
    }

    // Setup status timer for animated dots
    if (!p_dock->embedding_status_timer) {
        p_dock->embedding_status_timer = memnew(Timer);
        p_dock->embedding_status_timer->set_wait_time(0.5);
        p_dock->embedding_status_timer->set_one_shot(false);
        p_dock->embedding_status_timer->connect("timeout", callable_mp(p_dock, &AIChatDock::_on_embedding_status_tick));
        p_dock->add_child(p_dock->embedding_status_timer);
    }

    // Setup periodic poll timer for background indexing
    if (!p_dock->embedding_poll_timer) {
        p_dock->embedding_poll_timer = memnew(Timer);
        p_dock->embedding_poll_timer->set_wait_time(p_dock->embedding_poll_seconds);
        p_dock->embedding_poll_timer->set_one_shot(false);
        p_dock->embedding_poll_timer->connect("timeout", callable_mp(p_dock, &AIChatDock::_on_embedding_poll_tick));
        p_dock->add_child(p_dock->embedding_poll_timer);
        p_dock->embedding_poll_timer->start();
        print_line("AI Chat: Enabled periodic indexing poll every " + String::num_int64(p_dock->embedding_poll_seconds) + "s");
    }

    p_dock->embedding_system_initialized = true;
    // Keep status UI hidden by default
    p_dock->_set_embedding_status("", false);

    if (!p_dock->_is_user_authenticated()) {
        p_dock->current_user_id = "guest:" + p_dock->get_machine_id();
        p_dock->current_user_name = "Guest";
        p_dock->auth_token = "";
        p_dock->_update_user_status();
        print_line("AI Chat: Embedding system ready; indexing as guest session");
    }

    // Defer status/indexing to avoid overlapping requests right after init
    print_line("AI Chat: Embedding system initialized successfully");
}

// ========== EMBEDDING SYSTEM IMPLEMENTATION ==========

void AIChatDock::_initialize_embedding_system() {
    print_line("AI Chat: Initializing cloud-based embedding system");

    // Connect to editor file system signals for automatic reindexing
    // This connection does not require authentication; it only sets up callbacks.
    if (EditorFileSystem::get_singleton()) {
        if (!EditorFileSystem::get_singleton()->is_connected("filesystem_changed", callable_mp(this, &AIChatDock::_on_filesystem_changed))) {
            EditorFileSystem::get_singleton()->connect("filesystem_changed", callable_mp(this, &AIChatDock::_on_filesystem_changed));
        }
        if (!EditorFileSystem::get_singleton()->is_connected("sources_changed", callable_mp(this, &AIChatDock::_on_sources_changed))) {
            EditorFileSystem::get_singleton()->connect("sources_changed", callable_mp(this, &AIChatDock::_on_sources_changed));
        }
        print_line("AI Chat: Connected to EditorFileSystem change signals (filesystem_changed, sources_changed)");
    } else {
        print_line("AI Chat: EditorFileSystem not ready; change signals not connected");
    }

    // Connect to precise save signals from EditorNode to index only changed files
    if (EditorNode::get_singleton()) {
        if (!EditorNode::get_singleton()->is_connected("resource_saved", callable_mp(this, &AIChatDock::_on_editor_resource_saved))) {
            EditorNode::get_singleton()->connect("resource_saved", callable_mp(this, &AIChatDock::_on_editor_resource_saved), CONNECT_DEFERRED);
        }
        if (!EditorNode::get_singleton()->is_connected("scene_saved", callable_mp(this, &AIChatDock::_on_editor_scene_saved))) {
            EditorNode::get_singleton()->connect("scene_saved", callable_mp(this, &AIChatDock::_on_editor_scene_saved), CONNECT_DEFERRED);
        }
        print_line("AI Chat: Connected to EditorNode save signals (resource_saved, scene_saved)");
    }

    // Create HTTPRequest for embedding API calls
    if (!embedding_request) {
        embedding_request = memnew(HTTPRequest);
        add_child(embedding_request);
        embedding_request->connect("request_completed", callable_mp(this, &AIChatDock::_on_embedding_request_completed));
    }

    // Setup status timer for animated dots
    if (!embedding_status_timer) {
        embedding_status_timer = memnew(Timer);
        embedding_status_timer->set_wait_time(0.5);
        embedding_status_timer->set_one_shot(false);
        embedding_status_timer->connect("timeout", callable_mp(this, &AIChatDock::_on_embedding_status_tick));
        add_child(embedding_status_timer);
    }

    // Setup periodic poll timer for background indexing
    if (!embedding_poll_timer) {
        embedding_poll_timer = memnew(Timer);
        embedding_poll_timer->set_wait_time(embedding_poll_seconds);
        embedding_poll_timer->set_one_shot(false);
        embedding_poll_timer->connect("timeout", callable_mp(this, &AIChatDock::_on_embedding_poll_tick));
        add_child(embedding_poll_timer);
        embedding_poll_timer->start();
        print_line("AI Chat: Enabled periodic indexing poll every " + String::num_int64(embedding_poll_seconds) + "s");
    }

    embedding_system_initialized = true;
    // Keep status UI hidden by default
    _set_embedding_status("", false);

    if (!_is_user_authenticated()) {
        current_user_id = "guest:" + get_machine_id();
        current_user_name = "Guest";
        auth_token = "";
        _update_user_status();
        print_line("AI Chat: Embedding system ready; indexing as guest session");
    }

    // Defer status/indexing to avoid overlapping requests right after init
    print_line("AI Chat: Embedding system initialized successfully");
}
void AIChatDock::_perform_initial_indexing() {
	print_line("AI Chat: Starting project indexing...");
	print_line("AI Chat: DEBUG - embedding_system_initialized: " + String(embedding_system_initialized ? "true" : "false"));
	print_line("AI Chat: DEBUG - embedding_in_progress: " + String(embedding_in_progress ? "true" : "false"));
	
	if (!embedding_system_initialized) {
		print_line("AI Chat: Cannot start indexing - system not initialized");
		return;
	}
	
	if (embedding_in_progress) {
		print_line("AI Chat: Indexing already in progress");
		return;
	}
	
	print_line("AI Chat: All checks passed, starting file scan (silent UI)...");
	_set_embedding_status("", false);
	
	// Always use cloud-ready approach: scan files and send content
	// This works both locally and when deployed to cloud
	print_line("AI Chat: About to call _scan_and_index_project_files...");
	_scan_and_index_project_files();
}

void AIChatDock::_send_embedding_request(const String &p_action, const Dictionary &p_data) {
	if (!embedding_request || embedding_request_busy) {
		print_line("AI Chat: Cannot send embedding request - busy or not initialized");
		return;
	}
	
	String embed_url = _get_embed_base_url() + "/embed";
	
	Dictionary request_data;
	request_data["action"] = p_action;
	
	// Always include project_root for all embedding requests
	request_data["project_root"] = _get_project_root_path();
	
	if (!p_data.is_empty()) {
		for (const Variant *key = p_data.next(); key; key = p_data.next(key)) {
			request_data[*key] = p_data[*key];
		}
	}
	
	Ref<JSON> json;
	json.instantiate();
	String request_body = json->stringify(request_data);
	
	PackedStringArray headers;
	headers.push_back("Content-Type: application/json");
	_add_version_headers_to_request(headers);
	
	// Add authentication headers
	if (!auth_token.is_empty()) {
		headers.push_back("Authorization: Bearer " + auth_token);
	}
	headers.push_back("X-User-ID: " + current_user_id);
	headers.push_back("X-Machine-ID: " + get_machine_id());
	
    print_line("AI Chat: Sending embedding request: " + p_action + " to " + embed_url +
        " (project_root=" + _get_project_root_path() + ")");
	
	embedding_request_busy = true;
	Error err = embedding_request->request(embed_url, headers, HTTPClient::METHOD_POST, request_body);
	
	if (err != OK) {
		print_line("AI Chat: Failed to send embedding request: " + String::num_int64(err));
		embedding_request_busy = false;
		_set_embedding_status("Request failed", false);
	}
}
void AIChatDock::_on_embedding_request_completed(int p_result, int p_code, const PackedStringArray &p_headers, const PackedByteArray &p_body) {
	embedding_request_busy = false;
	
	print_line("AI Chat: Embedding request completed - Result: " + String::num_int64(p_result) + ", Code: " + String::num_int64(p_code));
	
	if (p_result != HTTPRequest::RESULT_SUCCESS || p_code != 200) {
		String error_msg = "Request failed (" + String::num_int64(p_code) + ")";
		print_line("AI Chat: " + error_msg);
		_set_embedding_status(error_msg, false);
		return;
	}
	
	String response_text = String::utf8((const char *)p_body.ptr(), p_body.size());
	
	Ref<JSON> json;
	json.instantiate();
	Error parse_err = json->parse(response_text);
	
	if (parse_err != OK) {
		print_line("AI Chat: Failed to parse embedding response");
		_set_embedding_status("Parse error", false);
		return;
	}
	
	Dictionary response = json->get_data();
	bool success = response.get("success", false);
	String action = response.get("action", "");
	
	if (!success) {
		String error = response.get("error", "Unknown error");
		print_line("AI Chat: Embedding request failed: " + error);
		_set_embedding_status("Error: " + error, false);
		return;
	}
	
	print_line("AI Chat: Embedding action '" + action + "' completed successfully");
	
	if (action == "index_project") {
		Dictionary stats = response.get("stats", Dictionary());
		int total = stats.get("total", 0);
		int indexed = stats.get("indexed", 0);
		int skipped = stats.get("skipped", 0);
		
		String status_text = "Indexed " + String::num_int64(indexed) + "/" + String::num_int64(total) + " files";
		if (skipped > 0) {
			status_text += " (" + String::num_int64(skipped) + " skipped)";
		}
		
		_set_embedding_status(status_text, false);
		initial_indexing_done = true;
		
		print_line("AI Chat: Project indexing completed - " + status_text);
		
	} else if (action == "index_files") {
		// Handle batch processing response
		Dictionary stats = response.get("stats", Dictionary());
		int batch_indexed = stats.get("indexed", 0);
		int batch_skipped = stats.get("skipped", 0);
		int batch_failed = stats.get("failed", 0);
		
		print_line("AI Chat: Batch completed - indexed: " + String::num_int64(batch_indexed) + ", skipped: " + String::num_int64(batch_skipped) + ", failed: " + String::num_int64(batch_failed));
		
		// Check if we need to send more batches
		if (current_batch_info.has("current_batch") && current_batch_info.has("total_batches")) {
			int current_batch = current_batch_info["current_batch"];
			int total_batches = current_batch_info["total_batches"];
			
			if (current_batch < total_batches) {
				// Send next batch
				int next_batch = current_batch + 1;
				int start_index = current_batch_info["start_index"];
				int batch_size = current_batch_info["batch_size"];
				Array all_files = current_batch_info["all_files"];
				
				int next_start_index = start_index + batch_size;
				
				// Keep indexing UI silent
				_set_embedding_status("", false);
				call_deferred("_send_file_batch", all_files, next_start_index, batch_size, next_batch, total_batches);
			} else {
				// All batches completed
				_set_embedding_status("All files indexed successfully", false);
				initial_indexing_done = true;
				print_line("AI Chat: All file batches completed successfully");
			}
		}
		
	} else if (action == "status") {
		Dictionary stats = response.get("stats", Dictionary());
		int files_indexed = stats.get("files_indexed", 0);
		int total_chunks = stats.get("total_chunks", 0);
		
		if (files_indexed > 0) {
			String status_text = String::num_int64(files_indexed) + " files indexed (" + String::num_int64(total_chunks) + " chunks)";
			_set_embedding_status(status_text, false);
			initial_indexing_done = true;
		} else {
			_set_embedding_status("No files indexed", false);
			initial_indexing_done = false;
		}
		
	} else if (action == "clear") {
		_set_embedding_status("Index cleared", false);
		initial_indexing_done = false;
	}
}

String AIChatDock::_get_project_root_path() {
	return ProjectSettings::get_singleton()->globalize_path("res://");
}

String AIChatDock::_get_embed_base_url() {
	// Use same endpoint as chat but for embedding operations
	String base_url = api_endpoint;
	
	// Remove /chat suffix if present and replace with embedding endpoint
	if (base_url.ends_with("/chat")) {
		base_url = base_url.substr(0, base_url.length() - 5);
	}
	
	return base_url;
}

void AIChatDock::_set_embedding_status(const String &p_text, bool p_busy) {
	if (!embedding_status_label) {
		return;
	}
	
	embedding_in_progress = p_busy;
	embedding_status_base = p_text;
	embedding_status_dots = 0;
	
	if (p_busy) {
		embedding_status_label->set_text(p_text + "...");
		embedding_status_label->set_modulate(Color(1.0, 0.8, 0.0)); // Yellow for in-progress
		if (embedding_status_timer) {
			embedding_status_timer->start();
		}
	} else {
		embedding_status_label->set_text(p_text);
		embedding_status_label->set_modulate(Color(0.7, 0.7, 0.7)); // Gray for idle
		if (embedding_status_timer) {
			embedding_status_timer->stop();
		}
	}
}

void AIChatDock::_on_embedding_status_tick() {
	if (!embedding_in_progress || !embedding_status_label) {
		return;
	}
	
	embedding_status_dots = (embedding_status_dots + 1) % 4;
	String dots = "";
	for (int i = 0; i < embedding_status_dots; i++) {
		dots += ".";
	}
	
	embedding_status_label->set_text(embedding_status_base + dots);
}

bool AIChatDock::_should_index_file(const String &p_file_path) {
	// Only allow textual/code formats; backend enforces too
	String ext = p_file_path.get_extension().to_lower();
	static const HashSet<String> allowed_text_ext = {
		"gd", "cs", "c", "cpp", "h", "hpp", "glsl", "shader", "gdshader",
		"tscn", "scn", "tres", "res", "godot", "import",
		"json", "cfg", "ini", "yaml", "yml", "xml", "md", "txt", "rst"
	};
	if (!allowed_text_ext.has(ext)) {
		return false;
	}
	String filename = p_file_path.get_file();
	if (filename.begins_with(".")) {
		return false;
	}
	return true;
}
void AIChatDock::_update_file_embedding(const String &p_file_path) {
	if (!embedding_system_initialized || !_should_index_file(p_file_path)) {
		return;
	}
	
	// Cloud-ready: read file content and send via index_files
	Dictionary file_data = _read_file_for_indexing(p_file_path, _get_project_root_path());
	if (file_data.is_empty()) {
		print_line("AI Chat: Failed to read file for embedding update: " + p_file_path);
		return;
	}
	
	Array files_arr;
	files_arr.push_back(file_data);
	
	Dictionary payload;
	payload["files"] = files_arr;
	Dictionary batch_info;
	batch_info["current"] = 1;
	batch_info["total"] = 1;
	batch_info["files_in_batch"] = 1;
	payload["batch_info"] = batch_info;
	
	_send_embedding_request("index_files", payload);
}
void AIChatDock::_remove_file_embedding(const String &p_file_path) {
	if (!embedding_system_initialized) {
		return;
	}
	
	Dictionary payload;
	payload["file_path"] = p_file_path;
	payload["project_root"] = _get_project_root_path();
	
	_send_embedding_request("remove_file", payload);
}

void AIChatDock::_on_filesystem_changed() {
    print_line("AI Chat: filesystem_changed signal received");
    // Only rely on per-file saved signals for accuracy; skip project-wide reindex on generic FS changes.
    pending_fs_changes = true;
}

void AIChatDock::_on_sources_changed(bool p_exist) {
    // Called when source files change
    print_line("AI Chat: sources_changed signal received (exist=" + String(p_exist ? "true" : "false") + ")");
    // Do nothing here; precise per-file handlers will trigger indexing.
}

void AIChatDock::_on_editor_resource_saved(Object *p_res) {
    if (!p_res) {
        return;
    }
    Ref<Resource> res = Ref<Resource>(Object::cast_to<Resource>(p_res));
    if (res.is_null()) {
        return;
    }
    String path = res->get_path();
    if (path.is_empty()) {
        return;
    }
    print_line("AI Chat: resource_saved -> " + path);
    // Clear overlay on save to ensure subsequent reads/checks use disk content
    EditorTools::clear_preview_overlay(path);
    // Always queue for periodic batch to avoid immediate requests
    String abs_path_res = ProjectSettings::get_singleton()->globalize_path(path);
    if (_should_index_file(abs_path_res)) {
        pending_changed_files.insert(abs_path_res);
        pending_fs_changes = true;
        // Optional: surface status as queued (non-blocking)
        _set_embedding_status("Queued file for indexing", false);
    }
}

void AIChatDock::_on_editor_scene_saved(const String &p_path) {
    print_line("AI Chat: scene_saved -> " + p_path);
    // Always queue for periodic batch to avoid immediate requests
    String abs_path = ProjectSettings::get_singleton()->globalize_path(p_path);
    if (_should_index_file(abs_path)) {
        pending_changed_files.insert(abs_path);
        pending_fs_changes = true;
        _set_embedding_status("Queued file for indexing", false);
    }
}

void AIChatDock::_on_embedding_poll_tick() {
    if (!embedding_system_initialized || !_is_user_authenticated() || embedding_request_busy) {
        return;
    }
    // Prefer batching changed files; otherwise do lightweight incremental project scan
    if (!pending_changed_files.is_empty()) {
        // Batch changed files using cloud-ready endpoint to minimize requests
        Array files_arr;
        for (const String &abs_path : pending_changed_files) {
            // Read file contents and hash (same as bulk flow)
            Dictionary file_data = _read_file_for_indexing(abs_path, _get_project_root_path());
            if (!file_data.is_empty()) {
                files_arr.push_back(file_data);
            }
        }
        pending_changed_files.clear();
        if (!files_arr.is_empty()) {
            Dictionary payload;
            payload["files"] = files_arr;
            Dictionary batch_info;
            batch_info["current"] = 1;
            batch_info["total"] = 1;
            batch_info["files_in_batch"] = files_arr.size();
            payload["batch_info"] = batch_info;
            _set_embedding_status("Indexing changed files", true);
            _send_embedding_request("index_files", payload);
            last_index_request_ms = OS::get_singleton()->get_ticks_msec();
            return;
        }
    }
    // If there were FS changes but nothing queued, avoid aggressive full project scan
    if (pending_fs_changes) {
        print_line("AI Chat: FS changes detected but no specific files queued. Skipping full project scan to avoid unnecessary re-indexing.");
        // Only clear the flag - don't trigger full project indexing for generic FS changes
        // Real changes should be caught by the specific save handlers above
        pending_fs_changes = false;
    }
}

void AIChatDock::_suggest_relevant_files(const String &p_query) {
	// TODO: Implement smart file suggestions based on embedding similarity
	print_line("AI Chat: Smart file suggestions not implemented yet for query: " + p_query);
}

void AIChatDock::_auto_attach_relevant_context() {
	// TODO: Implement automatic context attachment based on message content
	print_line("AI Chat: Auto context attachment not implemented yet");
}

void AIChatDock::_scan_and_index_project_files() {
	print_line("AI Chat: Scanning project files for indexing...");
	
	String project_root = _get_project_root_path();
	print_line("AI Chat: DEBUG - project_root: " + project_root);
	
	Array file_contents = Array();
	int files_processed = 0;
	int files_skipped = 0;
	
	// Get all files in project recursively
	print_line("AI Chat: Starting recursive directory scan...");
	_scan_directory_recursive(project_root, project_root, file_contents, files_processed, files_skipped);
	
	print_line("AI Chat: Scan complete - " + String::num_int64(files_processed) + " files to index, " + String::num_int64(files_skipped) + " skipped");
	print_line("AI Chat: DEBUG - file_contents.size(): " + String::num_int64(file_contents.size()));
	
	if (file_contents.size() == 0) {
		print_line("AI Chat: No files found to index!");
		_set_embedding_status("No files to index", false);
		return;
	}
	
	// Send files in batches to avoid huge HTTP requests
	int batch_size = 20; // Process 20 files at a time
	int total_batches = (file_contents.size() + batch_size - 1) / batch_size;
	
	print_line("AI Chat: Preparing " + String::num_int64(total_batches) + " batches of " + String::num_int64(batch_size) + " files each (silent UI)");
	_set_embedding_status("", false);
	_send_file_batch(file_contents, 0, batch_size, 1, total_batches);
}
void AIChatDock::_scan_directory_recursive(const String &p_dir_path, const String &p_project_root, Array &p_file_contents, int &p_files_processed, int &p_files_skipped) {
	Ref<DirAccess> dir = DirAccess::open(p_dir_path);
	if (dir.is_null()) {
		print_line("AI Chat: Cannot access directory: " + p_dir_path);
		return;
	}
	
	dir->list_dir_begin();
	String file_name = dir->get_next();
	
	while (!file_name.is_empty()) {
		String full_path = p_dir_path.path_join(file_name);
		
		if (dir->current_is_dir()) {
			// Skip hidden directories and common build/cache dirs
			if (!file_name.begins_with(".") && 
				file_name != "build" && 
				file_name != "bin" && 
				file_name != "obj" && 
				file_name != "__pycache__") {
				_scan_directory_recursive(full_path, p_project_root, p_file_contents, p_files_processed, p_files_skipped);
			} else {
				p_files_skipped++;
			}
		} else {
			// Check if we should index this file
			if (_should_index_file(full_path)) {
				Dictionary file_data = _read_file_for_indexing(full_path, p_project_root);
				if (!file_data.is_empty()) {
					p_file_contents.push_back(file_data);
					p_files_processed++;
				} else {
					p_files_skipped++;
				}
			} else {
				p_files_skipped++;
			}
		}
		
		file_name = dir->get_next();
	}
	
	dir->list_dir_end();
}

Dictionary AIChatDock::_read_file_for_indexing(const String &p_file_path, const String &p_project_root) {
	Ref<FileAccess> file = FileAccess::open(p_file_path, FileAccess::READ);
	if (file.is_null()) {
		print_line("AI Chat: Cannot read file: " + p_file_path);
		return Dictionary();
	}
	
	String content = file->get_as_text(true); // Skip BOM if present
	file->close();

	// Apply smart truncation for large arrays BEFORE other processing
	content = EditorTools::smart_truncate_for_ai_context(content, p_file_path);

	// Sanitize content to avoid invalid JSON: strip control chars except whitespace
	{
		String sanitized;
		for (int i = 0; i < content.length(); i++) {
			char32_t ch = content[i];
			if (ch >= 32 || ch == '\n' || ch == '\t' || ch == '\r') {
				sanitized += ch;
			}
		}
		content = sanitized;
	}

	// Cap per-file payload size
	const int64_t max_len = 200000; // ~200 KB
	if (content.length() > max_len) {
		content = content.substr(0, max_len);
	}
	
	// Skip empty files or files with only whitespace
	if (content.strip_edges().is_empty()) {
		return Dictionary();
	}
	
	// Get relative path from project root
	String relative_path = p_file_path.replace(p_project_root, "");
	if (relative_path.begins_with("/") || relative_path.begins_with("\\")) {
		relative_path = relative_path.substr(1);
	}
	
	// Calculate content hash for change detection
	String content_hash = _calculate_content_hash(content);
	
	Dictionary file_data;
	file_data["path"] = relative_path;
	file_data["content"] = content;
	file_data["hash"] = content_hash;
	file_data["size"] = content.length();
	
	return file_data;
}

String AIChatDock::_calculate_content_hash(const String &p_content) {
	// Simple hash calculation using Godot's built-in hash function
	uint32_t hash_value = p_content.hash();
	return String::num_uint64(hash_value, 16);
}

void AIChatDock::_send_file_batch(const Array &p_all_files, int p_start_index, int p_batch_size, int p_current_batch, int p_total_batches) {
	Array batch_files = Array();
	int end_index = MIN(p_start_index + p_batch_size, p_all_files.size());
	
	// Create batch
	for (int i = p_start_index; i < end_index; i++) {
		batch_files.push_back(p_all_files[i]);
	}
	
	// Prepare request data
	Dictionary batch_info_dict;
	batch_info_dict["current"] = p_current_batch;
	batch_info_dict["total"] = p_total_batches;
	batch_info_dict["files_in_batch"] = batch_files.size();
	
	Dictionary payload;
	payload["files"] = batch_files;
	payload["batch_info"] = batch_info_dict;
	
	// Store batch info for handling response
	current_batch_info["start_index"] = p_start_index;
	current_batch_info["batch_size"] = p_batch_size;
	current_batch_info["current_batch"] = p_current_batch;
	current_batch_info["total_batches"] = p_total_batches;
	current_batch_info["all_files"] = p_all_files;
	
	print_line("AI Chat: [BATCH] Sending batch " + String::num_int64(p_current_batch) + "/" + String::num_int64(p_total_batches) + " (" + String::num_int64(batch_files.size()) + " files)");
	
	_send_embedding_request("index_files", payload);
}

