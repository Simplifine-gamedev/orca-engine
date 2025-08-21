/*
 * © 2025 Simplifine Corp.
 * Personal Non‑Commercial License applies to this file as an original contribution to this Godot fork.
 * See LICENSES/COMPANY-NONCOMMERCIAL.md for terms. Commercial use requires a separate license from the Project Owner.
 */
#include "ai_tool_server.h"
#include "editor/ai/editor_tools.h"
#include "core/io/json.h"
#include "core/string/string_builder.h"
#include "core/config/project_settings.h"

void AIToolServer::_bind_methods() {
	ClassDB::bind_method(D_METHOD("listen", "port"), &AIToolServer::listen, DEFVAL(8001));
	ClassDB::bind_method(D_METHOD("stop"), &AIToolServer::stop);
	ClassDB::bind_method(D_METHOD("is_listening"), &AIToolServer::is_listening);
}

void AIToolServer::_server_thread_poll(void *data) {
	AIToolServer *server = static_cast<AIToolServer *>(data);
	server->_poll();
}

void AIToolServer::_clear_client() {
	peer.unref();
	tcp.unref();
	req_pos = 0;
	memset(req_buf, 0, sizeof(req_buf));
}

void AIToolServer::_send_response() {
	// This will be populated by _handle_tool_request
}

Dictionary AIToolServer::_handle_tool_request(const String &p_method, const String &p_path, const String &p_body) {
	Dictionary result;
	
	if (p_method != "POST") {
		result["error"] = "Method not allowed";
		return result;
	}
	
	// Parse JSON body
	Ref<JSON> json;
	json.instantiate();
	Error err = json->parse(p_body);
	if (err != OK) {
		result["error"] = "Invalid JSON";
		return result;
	}
	
	Dictionary request_data = json->get_data();
	String function_name = request_data.get("function_name", "");
	Dictionary args = request_data.get("arguments", Dictionary());
	
	// Handle tool execution
    // NOTE: apply_edit is now handled in the backend, not here
    if (function_name == "list_project_files") {
		result = EditorTools::list_project_files(args);
    } else if (function_name == "read_file") {
        result = EditorTools::read_file(args);
	} else if (function_name == "read_file_content") {
        // Deprecated: route to unified read
        result = EditorTools::read_file(args);
	} else if (function_name == "read_file_advanced") {
        // Deprecated: route to unified read
        result = EditorTools::read_file(args);
	} else if (function_name == "get_scene_info") {
		result = EditorTools::get_scene_info(args);
	} else if (function_name == "get_all_nodes") {
		result = EditorTools::get_all_nodes(args);
	} else if (function_name == "search_nodes_by_type") {
		result = EditorTools::search_nodes_by_type(args);
	} else if (function_name == "get_editor_selection") {
		result = EditorTools::get_editor_selection(args);
	} else if (function_name == "get_node_properties") {
		result = EditorTools::get_node_properties(args);
    } else if (function_name == "save_scene") {
        // Tool removed for safety: respond with a clear error
        result["success"] = false;
        result["message"] = "Tool 'save_scene' has been removed. Please save scenes manually.";
	} else if (function_name == "create_node") {
		result = EditorTools::create_node(args);
	} else if (function_name == "delete_node") {
		result = EditorTools::delete_node(args);
	} else if (function_name == "set_node_property") {
		result = EditorTools::set_node_property(args);
	} else if (function_name == "batch_set_node_properties") {
		result = EditorTools::batch_set_node_properties(args);
	} else if (function_name == "move_node") {
		result = EditorTools::move_node(args);
	} else if (function_name == "call_node_method") {
		result = EditorTools::call_node_method(args);
	} else if (function_name == "get_available_classes") {
		result = EditorTools::get_available_classes(args);
	} else if (function_name == "get_node_script") {
		result = EditorTools::get_node_script(args);
	} else if (function_name == "attach_script") {
		result = EditorTools::attach_script(args);
	} else if (function_name == "manage_scene") {
		result = EditorTools::manage_scene(args);
	} else if (function_name == "add_collision_shape") {
		result = EditorTools::add_collision_shape(args);
	} else if (function_name == "check_compilation_errors") {
		result = EditorTools::check_compilation_errors(args);
	} else if (function_name == "get_runtime_errors") {
		result = EditorTools::get_runtime_errors(args);
	} else if (function_name == "test_diff_and_errors") {
		// Test endpoint that simulates an edit to test diff and compilation error functionality
		String path = args.get("path", "");
		String mock_edit = args.get("mock_edit", "");
		
		if (path.is_empty()) {
			result["error"] = "Path is required";
		} else {
			// Read original file
			Error err;
			String original_content = FileAccess::get_file_as_string(path, &err);
			if (err != OK) {
				result["success"] = false;
				result["message"] = "Failed to read file: " + path;
				result["diff"] = "";
				result["compilation_errors"] = Array();
			} else {
				// Use provided mock edit or create a simple one
				String modified_content = mock_edit;
				if (modified_content.is_empty()) {
					// Create a simple mock edit for testing
					modified_content = original_content.replace("print(\"Hello World\")", "print(\"Hello from Test!\")");
				}
				
				// Test the diff generation
				String diff = EditorTools::_generate_unified_diff(original_content, modified_content, path);
				
				// Test compilation error checking (temporarily disabled to prevent crashes)
				Array compilation_errors;
				// compilation_errors = EditorTools::_check_compilation_errors(path, modified_content);
				
				result["success"] = true;
				result["message"] = "Test completed successfully";
				result["original_content"] = original_content;
				result["modified_content"] = modified_content;
				result["diff"] = diff;
				result["compilation_errors"] = compilation_errors;
				result["has_errors"] = compilation_errors.size() > 0;
			}
		}
	} else if (function_name == "run_scene") {
		result = EditorTools::run_scene(args);
	} else if (function_name == "get_scene_tree_hierarchy") {
		result = EditorTools::get_scene_tree_hierarchy(args);
	} else if (function_name == "inspect_physics_body") {
		result = EditorTools::inspect_physics_body(args);
	} else if (function_name == "get_camera_info") {
		result = EditorTools::get_camera_info(args);
	} else if (function_name == "take_screenshot") {
		result = EditorTools::take_screenshot(args);
	} else if (function_name == "check_node_in_scene_tree") {
		result = EditorTools::check_node_in_scene_tree(args);
	} else if (function_name == "inspect_animation_state") {
		result = EditorTools::inspect_animation_state(args);
    } else if (function_name == "get_layers_and_zindex") {
		result = EditorTools::get_layers_and_zindex(args);
    } else if (function_name == "search_across_project") {
        // Normalize defaults for agent and inject project root
        if (!args.has("max_results")) {
            args["max_results"] = 5;
        }
        if (!args.has("include_graph")) {
            args["include_graph"] = true;
        }
        // Always include the project root to avoid backend defaulting to old projects
        String project_root = ProjectSettings::get_singleton()->globalize_path("res://");
        args["project_root"] = project_root;
        print_line("AI Tool Server: search_across_project invoking with project_root=" + project_root);
		result = EditorTools::search_across_project(args);
	} else if (function_name == "search_across_godot_docs") {
		if (!args.has("max_results")) {
			args["max_results"] = 5;
		}
		print_line("AI Tool Server: search_across_godot_docs invoking");
		result = EditorTools::search_across_godot_docs(args);
	} else if (function_name == "editor_introspect") {
		result = EditorTools::editor_introspect(args);
	} else if (function_name == "slice_spritesheet") {
		// Forward to editor_introspect with slice_spritesheet operation
		args["operation"] = "slice_spritesheet";
		result = EditorTools::editor_introspect(args);
	} else if (function_name == "analyze_current_rig") {
		result = EditorTools::analyze_current_rig(args);
	} else if (function_name == "scan_project_rigs") {
		result = EditorTools::scan_project_rigs(args);
	} else if (function_name == "analyze_project_rigs") {
		result = EditorTools::analyze_project_rigs(args);
	} else if (function_name == "analyze_selected_rig") {
		result = EditorTools::analyze_selected_rig(args);
	} else if (function_name == "standardize_rig_structure") {
		result = EditorTools::standardize_rig_structure(args);
	} else if (function_name == "batch_fix_rigs") {
		result = EditorTools::batch_fix_rigs(args);
	} else if (function_name == "instantiate_rig_asset") {
		result = EditorTools::instantiate_rig_asset(args);
	} else if (function_name == "extract_mesh_data") {
		result = EditorTools::extract_mesh_data(args);
	} else {
		result["error"] = "Unknown function: " + function_name;
	}
	
	return result;
}

void AIToolServer::_poll() {
	while (!server_quit.is_set()) {
		if (!server->is_listening()) {
			OS::get_singleton()->delay_usec(100000); // 100ms
			continue;
		}
		
		if (tcp.is_null()) {
			if (!server->is_connection_available()) {
				OS::get_singleton()->delay_usec(10000); // 10ms delay
				continue;
			}
			tcp = server->take_connection();
			peer = tcp;
			time = OS::get_singleton()->get_ticks_msec();
		}
		
		if (peer.is_null()) {
			OS::get_singleton()->delay_usec(10000); // 10ms delay
			continue;
		}
	
	while (true) {
		// Read request data
		if (req_pos >= 4096) {
			_clear_client();
			break;
		}
		
		int received = 0;
		Error err = peer->get_partial_data(&req_buf[req_pos], 1, received);
		if (err != OK || received != 1) {
			break;
		}
		req_pos++;
		
		// Check for end of headers
		if (req_pos >= 4 && String::chr(req_buf[req_pos - 4]) + String::chr(req_buf[req_pos - 3]) + String::chr(req_buf[req_pos - 2]) + String::chr(req_buf[req_pos - 1]) == "\r\n\r\n") {
			String request_str = String::utf8((const char *)req_buf, req_pos);
			
			// Parse HTTP request
			PackedStringArray lines = request_str.split("\r\n");
			if (lines.size() == 0) {
				_clear_client();
				break;
			}
			
			PackedStringArray request_line = lines[0].split(" ");
			if (request_line.size() < 2) {
				_clear_client();
				break;
			}
			
			String method = request_line[0];
			String path = request_line[1];
			
			// Find Content-Length
			int content_length = 0;
			for (int i = 1; i < lines.size(); i++) {
				if (lines[i].begins_with("Content-Length:")) {
					content_length = lines[i].substr(15).strip_edges().to_int();
					break;
				}
			}
			
			// Read body if present
			String body = "";
			if (content_length > 0) {
				PackedByteArray body_bytes;
				body_bytes.resize(content_length);
				int body_read = 0;
				while (body_read < content_length) {
					int chunk_received = 0;
					Error chunk_err = peer->get_partial_data(&body_bytes.ptrw()[body_read], content_length - body_read, chunk_received);
					if (chunk_err != OK || chunk_received <= 0) break;
					body_read += chunk_received;
				}
				body = String::utf8((const char *)body_bytes.ptr(), body_read);
			}
			
			// Handle the request
			Dictionary response_data = _handle_tool_request(method, path, body);
			
			// Send response
			Ref<JSON> response_json;
			response_json.instantiate();
			String response_body = response_json->stringify(response_data);
			PackedByteArray body_bytes = response_body.to_utf8_buffer();
			
			StringBuilder response_builder;
			response_builder.append("HTTP/1.1 200 OK\r\n");
			response_builder.append("Content-Type: application/json\r\n");
			response_builder.append("Content-Length: " + itos(body_bytes.size()) + "\r\n");
			response_builder.append("Access-Control-Allow-Origin: *\r\n");
			response_builder.append("Access-Control-Allow-Methods: POST, OPTIONS\r\n");
			response_builder.append("Access-Control-Allow-Headers: Content-Type\r\n");
			response_builder.append("Connection: close\r\n");
			response_builder.append("\r\n");
			
			String headers = response_builder.as_string();
			PackedByteArray headers_bytes = headers.to_utf8_buffer();
			
			// Send headers first
			peer->put_data(headers_bytes.ptr(), headers_bytes.size());
			// Then send body
			peer->put_data(body_bytes.ptr(), body_bytes.size());
			
			_clear_client();
			break;
		}
		
		OS::get_singleton()->delay_usec(10000); // 10ms delay
	}
	}
}

Error AIToolServer::listen(int p_port) {
	server.instantiate();
	Error err = server->listen(p_port, IPAddress("127.0.0.1"));
	if (err != OK) {
		print_line("ERROR: Failed to start AI tool server on port " + itos(p_port));
		return err;
	}
	
	print_line("AI Tool Server: Started on port " + itos(p_port));
	
	server_quit.clear();
	server_thread.start(_server_thread_poll, this);
	
	return OK;
}

void AIToolServer::stop() {
	server_quit.set();
	if (server_thread.is_started()) {
		server_thread.wait_to_finish();
	}
	server.unref();
	_clear_client();
	
	print_line("AI Tool Server: Stopped");
}

bool AIToolServer::is_listening() const {
	return server.is_valid() && server->is_listening();
}

AIToolServer::AIToolServer() {
	req_pos = 0;
	memset(req_buf, 0, sizeof(req_buf));
}

AIToolServer::~AIToolServer() {
	stop();
} 