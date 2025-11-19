/*
 * © 2025 Simplifine Corp.
 * Enhanced Godot Graph Parser for Frontend
 * Personal Non-Commercial License applies to this file as an original contribution to this Godot fork.
 * See LICENSES/COMPANY-NONCOMMERCIAL.md for terms. Commercial use requires a separate license from Simplifine.
 */

#include "ai_enhanced_graph_parser.h"
#include "core/config/project_settings.h"
#include "core/io/config_file.h"
#include "core/os/time.h"

AIEnhancedGraphParser::AIEnhancedGraphParser() {
}

AIEnhancedGraphParser::~AIEnhancedGraphParser() {
}

void AIEnhancedGraphParser::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_project_root", "root"), &AIEnhancedGraphParser::set_project_root);
	ClassDB::bind_method(D_METHOD("parse_file", "file_path", "content"), &AIEnhancedGraphParser::parse_file);
	ClassDB::bind_method(D_METHOD("parse_project_file", "content"), &AIEnhancedGraphParser::parse_project_file);
	ClassDB::bind_method(D_METHOD("get_graph_data"), &AIEnhancedGraphParser::get_graph_data);
	ClassDB::bind_method(D_METHOD("get_context_for_file", "file_path"), &AIEnhancedGraphParser::get_context_for_file);
	ClassDB::bind_method(D_METHOD("enrich_file_context", "file_path"), &AIEnhancedGraphParser::enrich_file_context);
	ClassDB::bind_method(D_METHOD("clear"), &AIEnhancedGraphParser::clear);
}

void AIEnhancedGraphParser::set_project_root(const String &p_root) {
	project_root = p_root;
}

String AIEnhancedGraphParser::_normalize_path(const String &p_path) {
	String normalized = p_path;
	if (normalized.begins_with("res://")) {
		normalized = normalized.substr(6);
	}
	if (normalized.begins_with("\"") && normalized.ends_with("\"")) {
		normalized = normalized.substr(1, normalized.length() - 2);
	}
	if (normalized.begins_with("*")) {
		normalized = normalized.substr(1);
	}
	return normalized.strip_edges();
}

void AIEnhancedGraphParser::parse_file(const String &p_file_path, const String &p_content) {
	String ext = p_file_path.get_extension().to_lower();
	
	if (ext == "gd" || ext == "cs") {
		_parse_gdscript_file(p_file_path, p_content);
	} else if (ext == "tscn" || ext == "scn") {
		_parse_scene_file(p_file_path, p_content);
	} else if (ext == "godot") {
		_parse_project_settings(p_content);
	}
}

void AIEnhancedGraphParser::parse_project_file(const String &p_content) {
	_parse_project_settings(p_content);
}

void AIEnhancedGraphParser::_parse_gdscript_file(const String &p_file_path, const String &p_content) {
	Dictionary node_data;
	node_data["file_path"] = p_file_path;
	node_data["node_type"] = "script";
	
	// Extract signals
	Array signals_defined = _extract_signal_definitions(p_content);
	node_data["signals_defined"] = signals_defined;
	
	// Extract signal emissions
	Array signals_emitted = _extract_signal_emissions(p_content);
	node_data["signals_emitted"] = signals_emitted;
	
	// Extract functions WITH their signal emissions (CRITICAL for multi-hop tracing!)
	Array functions = _extract_function_definitions(p_content);
	node_data["functions"] = functions;
	
	// GAME-CHANGER: Map which functions emit which signals
	Array function_signal_map = _extract_function_signal_mappings(p_content, functions, signals_emitted);
	node_data["function_signals"] = function_signal_map;
	
	// Extract preloads
	Array preloads = _extract_preloads(p_content);
	node_data["preloads"] = preloads;
	
	// Extract extends
	Array extends = _extract_extends(p_content);
	node_data["extends"] = extends;
	
	// WORLD-CLASS: Extract method calls and node accesses
	Array method_calls = _extract_method_calls(p_content);
	node_data["method_calls"] = method_calls;
	
	Array node_accesses = _extract_node_accesses(p_content);
	node_data["node_accesses"] = node_accesses;
	
	Array property_mods = _extract_property_modifications(p_content);
	node_data["property_modifications"] = property_mods;
	
	// GAME-CHANGER: Export variables (cross-file dependencies)
	Array exports = _extract_export_variables(p_content);
	node_data["exports"] = exports;
	
	// Group memberships
	Array groups = _extract_group_memberships(p_content);
	node_data["groups"] = groups;
	
	// Dynamic scene loading (mob_scene.instantiate(), etc.)
	Array dynamic_loads = _extract_dynamic_scene_loads(p_content);
	node_data["dynamic_scene_loads"] = dynamic_loads;
	
	nodes[p_file_path] = node_data;
	
	// Track signal flows for each emitted signal
	for (int i = 0; i < signals_emitted.size(); i++) {
		Dictionary emission = signals_emitted[i];
		String signal_name = emission.get("signal_name", "");
		if (!signal_name.is_empty()) {
			if (!signal_flows.has(signal_name)) {
				signal_flows[signal_name] = Array();
			}
			Array flows = signal_flows[signal_name];
			Dictionary flow;
			flow["from_file"] = p_file_path;
			flow["line"] = emission.get("line", 0);
			flows.push_back(flow);
			signal_flows[signal_name] = flows;
		}
	}
	
	// Create connections for preloads
	for (int i = 0; i < preloads.size(); i++) {
		Dictionary preload_data = preloads[i];
		String preload_path = preload_data.get("path", "");
		if (!preload_path.is_empty()) {
			Dictionary connection;
			connection["source_file"] = p_file_path;
			connection["target_file"] = _normalize_path(preload_path);
			connection["connection_type"] = "preload";
			connections.push_back(connection);
		}
	}
	
	// GAME-CHANGER: Create connections for dynamic scene loads
	for (int i = 0; i < dynamic_loads.size(); i++) {
		Dictionary load_data = dynamic_loads[i];
		String scene_var = load_data.get("scene_variable", "");
		String scene_path = load_data.get("scene_path", "");
		
		if (!scene_path.is_empty()) {
			// Direct load() call with path
			Dictionary connection;
			connection["source_file"] = p_file_path;
			connection["target_file"] = _normalize_path(scene_path);
			connection["connection_type"] = "dynamic_scene_load";
			connection["load_type"] = "load";
			connection["line"] = load_data.get("line", 0);
			connections.push_back(connection);
		} else if (!scene_var.is_empty()) {
			// .instantiate() call - will be resolved by export variable mapping
			Dictionary connection;
			connection["source_file"] = p_file_path;
			connection["connection_type"] = "dynamic_scene_load";
			connection["load_type"] = "instantiate";
			connection["scene_variable"] = scene_var;
			connection["line"] = load_data.get("line", 0);
			connections.push_back(connection);
		}
	}
}

void AIEnhancedGraphParser::_parse_scene_file(const String &p_file_path, const String &p_content) {
	Dictionary node_data;
	node_data["file_path"] = p_file_path;
	node_data["node_type"] = "scene";
	
	// Extract external resources
	Array external_resources = _extract_external_resources(p_content);
	node_data["external_resources"] = external_resources;
	
	// Parse scene connections (signal flows within the scene)
	Array scene_connections = _parse_scene_connections(p_content);
	node_data["scene_connections"] = scene_connections;
	
	// WORLD-CLASS: Extract script attachments from scene nodes
	Array script_attachments;
	Vector<String> lines = p_content.split("\n");
	String current_node_name = "";
	
	for (int i = 0; i < lines.size(); i++) {
		String line = lines[i];
		
		// Track current node being defined
		if (line.contains("[node") && line.contains("name=\"")) {
			int name_pos = line.find("name=\"");
			if (name_pos >= 0) {
				int name_end = line.find("\"", name_pos + 6);
				if (name_end >= 0) {
					current_node_name = line.substr(name_pos + 6, name_end - name_pos - 6);
				}
			}
		}
		
		// Look for script attachments
		if (line.contains("script = ") && !current_node_name.is_empty()) {
			Dictionary attachment;
			attachment["node_name"] = current_node_name;
			
			// Extract script reference (ExtResource or direct path)
			if (line.contains("ExtResource(")) {
				int ext_pos = line.find("ExtResource(");
				int quote_start = line.find("\"", ext_pos);
				if (quote_start >= 0) {
					int quote_end = line.find("\"", quote_start + 1);
					if (quote_end >= 0) {
						String ext_id = line.substr(quote_start + 1, quote_end - quote_start - 1);
						attachment["ext_resource_id"] = ext_id;
						
						// Match using the short ID (now properly extracted from id="..." attribute)
						for (int j = 0; j < external_resources.size(); j++) {
							Dictionary ext_res = external_resources[j];
							String res_id = ext_res.get("id", "");
							String res_type = ext_res.get("type", "");
							String res_path = ext_res.get("path", "");
							
							// Simple exact match on the short ID
							if (res_id == ext_id && res_type == "Script" && !res_path.is_empty()) {
								attachment["script_path"] = res_path;
								break;
							}
						}
					}
				}
			}
			
			if (attachment.has("script_path")) {
				script_attachments.push_back(attachment);
			}
		}
	}
	
	node_data["script_attachments"] = script_attachments;
	
	// WORLD-CLASS: Extract scene instances (which scenes are instantiated)
	Array scene_instances = _extract_scene_instances(p_content);
	node_data["scene_instances"] = scene_instances;
	
	// GAME-CHANGER: Extract groups from scene nodes (groups=["mobs"])
	Array scene_groups = _extract_scene_groups(p_content);
	node_data["groups"] = scene_groups;
	
	// WORLD-CLASS: Extract export variable assignments (mob_scene = ExtResource("2"))
	Array export_assignments = _extract_export_assignments(p_content, external_resources, script_attachments);
	node_data["export_assignments"] = export_assignments;
	
	nodes[p_file_path] = node_data;
	
	// Create connections for script attachments (scene -> script relationships)
	for (int i = 0; i < script_attachments.size(); i++) {
		Dictionary attachment = script_attachments[i];
		String script_path = attachment.get("script_path", "");
		String node_name = attachment.get("node_name", "");
		
		if (!script_path.is_empty()) {
			Dictionary connection;
			connection["source_file"] = p_file_path;
			connection["target_file"] = _normalize_path(script_path);
			connection["connection_type"] = "script_attachment";
			connection["node_name"] = node_name;
			connections.push_back(connection);
		}
	}
	
	// Create connections for external resources
	for (int i = 0; i < external_resources.size(); i++) {
		Dictionary resource = external_resources[i];
		String resource_path = resource.get("path", "");
		String resource_type = resource.get("type", "");
		String resource_id = resource.get("id", "");
		
		if (!resource_path.is_empty()) {
			Dictionary connection;
			connection["source_file"] = p_file_path;
			connection["target_file"] = _normalize_path(resource_path);
			connection["connection_type"] = "external_resource";
			connection["resource_type"] = resource_type;
			connection["resource_id"] = resource_id;
			connections.push_back(connection);
			
			// Track resource dependencies
			if (!resource_dependencies.has(p_file_path)) {
				resource_dependencies[p_file_path] = Array();
			}
			Array deps = resource_dependencies[p_file_path];
			deps.push_back(_normalize_path(resource_path));
			resource_dependencies[p_file_path] = deps;
		}
	}
	
	// GAME-CHANGER: Create connections for export variable assignments
	// This maps @export var mob_scene: PackedScene → ExtResource("2") → mob.tscn
	for (int i = 0; i < export_assignments.size(); i++) {
		Dictionary assignment = export_assignments[i];
		String script_path = assignment.get("script_path", "");
		String export_var_name = assignment.get("export_var_name", "");
		String resource_path = assignment.get("resource_path", "");
		
		if (!script_path.is_empty() && !export_var_name.is_empty() && !resource_path.is_empty()) {
			Dictionary connection;
			connection["source_file"] = _normalize_path(script_path);
			connection["target_file"] = _normalize_path(resource_path);
			connection["connection_type"] = "export_variable";
			connection["export_var_name"] = export_var_name;
			connection["export_var_type"] = assignment.get("export_var_type", "");
			connection["scene_file"] = p_file_path;
			connection["node_name"] = assignment.get("node_name", "");
			connections.push_back(connection);
		}
	}
	
	// WORLD-CLASS: Create comprehensive signal flow connections
	for (int i = 0; i < scene_connections.size(); i++) {
		Dictionary conn = scene_connections[i];
		String signal_name = conn.get("signal_name", "");
		String from_node = conn.get("from_node", "");
		String to_node = conn.get("to_node", "");
		String method = conn.get("method", "");
		
		if (!signal_name.is_empty()) {
			// Track in signal flows
			if (!signal_flows.has(signal_name)) {
				signal_flows[signal_name] = Array();
			}
			Array flows = signal_flows[signal_name];
			Dictionary flow;
			flow["scene"] = p_file_path;
			flow["from_node"] = from_node;
			flow["to_node"] = to_node;
			flow["handler_method"] = method;
			flow["connection_type"] = "scene_signal_connection";
			flows.push_back(flow);
			signal_flows[signal_name] = flows;
			
			// GAME-CHANGER: Create detailed signal flow connection
			Dictionary signal_conn;
			signal_conn["connection_type"] = "signal_flow";
			signal_conn["signal_name"] = signal_name;
			signal_conn["handler_method"] = method;
			signal_conn["from_node"] = from_node;
			signal_conn["to_node"] = to_node;
			signal_conn["scene_file"] = p_file_path;
			
			// CRITICAL: Resolve source and target scripts from node attachments
			String source_script = "";
			String target_script = "";
			
			// Helper to match node names flexibly (handles ".", "./NodeName", "NodeName", etc.)
			auto matches_node = [](const String &node_path, const String &attachment_name) -> bool {
				if (node_path == attachment_name) return true;
				if (node_path == "." && (attachment_name == "." || attachment_name.is_empty())) return true;
				String cleaned_path = node_path.replace("./", "").strip_edges();
				if (cleaned_path == attachment_name) return true;
				return false;
			};
			
			// Find source and target scripts
			for (int j = 0; j < script_attachments.size(); j++) {
				Dictionary attachment = script_attachments[j];
				String node_name = attachment.get("node_name", "");
				String script_path = attachment.get("script_path", "");
				
				if (!script_path.is_empty()) {
					if (matches_node(from_node, node_name)) {
						source_script = _normalize_path(script_path);
					}
					if (matches_node(to_node, node_name)) {
						target_script = _normalize_path(script_path);
					}
				}
			}
			
			// Special fallback: if to_node is "." and no match, use first script (scene root)
			if (to_node == "." && target_script.is_empty() && script_attachments.size() > 0) {
				Dictionary first_attachment = script_attachments[0];
				String first_script = first_attachment.get("script_path", "");
				if (!first_script.is_empty()) {
					target_script = _normalize_path(first_script);
				}
			}
			
			// CRITICAL FIX: Set source and target to the SCRIPTS, not scene
			// The backend trace needs to follow script-to-script connections
			if (!source_script.is_empty()) {
				signal_conn["source_file"] = source_script;
				signal_conn["source_script"] = source_script;
			} else {
				// Fallback: emit from scene context (scene signals like Timer.timeout)
				signal_conn["source_file"] = p_file_path;
				signal_conn["source_script"] = "";
			}
			
			if (!target_script.is_empty()) {
				signal_conn["target_file"] = target_script;
				signal_conn["target_script"] = target_script;
			} else {
				// Fallback: handle in scene context
				signal_conn["target_file"] = p_file_path;
				signal_conn["target_script"] = "";
			}
			
			connections.push_back(signal_conn);
		}
	}
	
	// WORLD-CLASS: Create scene instantiation connections
	int instantiation_connections_created = 0;
	for (int i = 0; i < scene_instances.size(); i++) {
		Dictionary instance = scene_instances[i];
		String node_name = instance.get("node_name", "");
		String resource_id = instance.get("resource_id", "");
		
		if (!node_name.is_empty() && !resource_id.is_empty()) {
			// Find the actual scene file from external resources
			bool resource_found = false;
			for (int j = 0; j < external_resources.size(); j++) {
				Dictionary ext_res = external_resources[j];
				String ext_id = ext_res.get("id", "");
				String ext_type = ext_res.get("type", "");
				String ext_path = ext_res.get("path", "");
				
				if (ext_id == resource_id && ext_type == "PackedScene") {
					if (!ext_path.is_empty()) {
						Dictionary instantiation_conn;
						instantiation_conn["source_file"] = p_file_path;
						instantiation_conn["target_file"] = _normalize_path(ext_path);
						instantiation_conn["connection_type"] = "scene_instantiation";
						instantiation_conn["node_name"] = node_name;
						instantiation_conn["resource_id"] = resource_id;
						instantiation_conn["instantiated_as"] = node_name;
						connections.push_back(instantiation_conn);
						instantiation_connections_created++;
						resource_found = true;
					}
					break;
				}
			}
			
			// Silently skip if resource not found - common with UID-based references
			if (false && !resource_found) {
				for (int j = 0; j < external_resources.size(); j++) {
					Dictionary ext_res = external_resources[j];
					print_line("    - id=" + String(ext_res.get("id", "")) + ", type=" + String(ext_res.get("type", "")) + ", path=" + String(ext_res.get("path", "")));
				}
			}
		}
	}
	
	// Silently continue - logs removed for production
}

void AIEnhancedGraphParser::_parse_project_settings(const String &p_content) {
	// Parse project.godot for autoloads and input actions
	
	// Simple line-by-line parsing (Godot config format)
	Vector<String> lines = p_content.split("\n");
	String current_section = "";
	
	for (int i = 0; i < lines.size(); i++) {
		String line = lines[i].strip_edges();
		
		// Section headers
		if (line.begins_with("[") && line.ends_with("]")) {
			current_section = line.substr(1, line.length() - 2);
			continue;
		}
		
		// Autoloads
		if (current_section == "autoload" && line.contains("=")) {
			int eq_pos = line.find("=");
			String name = line.substr(0, eq_pos).strip_edges();
			String value = line.substr(eq_pos + 1).strip_edges();
			value = value.strip_edges().trim_prefix("\"").trim_suffix("\"");
			autoloads[name] = _normalize_path(value);
		}
		
		// Input actions
		if (current_section == "input" && line.contains("=")) {
			int eq_pos = line.find("=");
			String action_name = line.substr(0, eq_pos).strip_edges();
			String action_value = line.substr(eq_pos + 1).strip_edges();
			
			Dictionary action_data;
			action_data["name"] = action_name;
			action_data["definition"] = action_value;
			input_actions[action_name] = action_data;
		}
	}
}

Array AIEnhancedGraphParser::_extract_signal_definitions(const String &p_content) {
	Array signals;
	Vector<String> lines = p_content.split("\n");
	
	for (int i = 0; i < lines.size(); i++) {
		String line = lines[i].strip_edges();
		if (line.begins_with("signal ")) {
			Dictionary signal_data;
			String signal_decl = line.substr(7).strip_edges();
			
			// Extract signal name (before '(' or end of line)
			int paren_pos = signal_decl.find("(");
			String signal_name = paren_pos >= 0 ? signal_decl.substr(0, paren_pos).strip_edges() : signal_decl;
			
			signal_data["signal_name"] = signal_name;
			signal_data["line"] = i + 1;
			signal_data["declaration"] = signal_decl;
			signals.push_back(signal_data);
		}
	}
	
	return signals;
}

Array AIEnhancedGraphParser::_extract_signal_emissions(const String &p_content) {
	Array emissions;
	Vector<String> lines = p_content.split("\n");
	
	for (int i = 0; i < lines.size(); i++) {
		String line = lines[i];
		
		// Look for .emit() or emit_signal()
		if (line.contains(".emit(") || line.contains("emit_signal(")) {
			Dictionary emission;
			emission["line"] = i + 1;
			
			// Extract signal name
			String signal_name = "";
			if (line.contains(".emit(")) {
				int emit_pos = line.find(".emit(");
				// Work backwards to find signal name
				String before_emit = line.substr(0, emit_pos);
				Vector<String> parts = before_emit.split(" ");
				if (parts.size() > 0) {
					signal_name = parts[parts.size() - 1].strip_edges();
				}
			} else if (line.contains("emit_signal(")) {
				int emit_pos = line.find("emit_signal(");
				int quote_start = line.find("\"", emit_pos);
				if (quote_start >= 0) {
					int quote_end = line.find("\"", quote_start + 1);
					if (quote_end >= 0) {
						signal_name = line.substr(quote_start + 1, quote_end - quote_start - 1);
					}
				}
			}
			
			if (!signal_name.is_empty()) {
				emission["signal_name"] = signal_name;
				emission["code"] = line.strip_edges();
				emissions.push_back(emission);
			}
		}
	}
	
	return emissions;
}

Array AIEnhancedGraphParser::_extract_function_definitions(const String &p_content) {
	Array functions;
	Vector<String> lines = p_content.split("\n");
	
	for (int i = 0; i < lines.size(); i++) {
		String line = lines[i].strip_edges();
		if (line.begins_with("func ")) {
			Dictionary function_data;
			String function_decl = line.substr(5).strip_edges();
			
			// Extract function name (before '(')
			int paren_pos = function_decl.find("(");
			String function_name = paren_pos >= 0 ? function_decl.substr(0, paren_pos).strip_edges() : function_decl;
			
			function_data["function_name"] = function_name;
			function_data["line"] = i + 1;
			function_data["declaration"] = function_decl;
			functions.push_back(function_data);
		}
	}
	
	return functions;
}

Array AIEnhancedGraphParser::_extract_preloads(const String &p_content) {
	Array preloads;
	Vector<String> lines = p_content.split("\n");
	
	for (int i = 0; i < lines.size(); i++) {
		String line = lines[i];
		
		if (line.contains("preload(")) {
			Dictionary preload_data;
			preload_data["line"] = i + 1;
			
			// Extract path from preload("path")
			int preload_pos = line.find("preload(");
			int quote_start = line.find("\"", preload_pos);
			if (quote_start >= 0) {
				int quote_end = line.find("\"", quote_start + 1);
				if (quote_end >= 0) {
					String path = line.substr(quote_start + 1, quote_end - quote_start - 1);
					preload_data["path"] = _normalize_path(path);
					preloads.push_back(preload_data);
				}
			}
		}
	}
	
	return preloads;
}

Array AIEnhancedGraphParser::_extract_extends(const String &p_content) {
	Array extends_array;
	Vector<String> lines = p_content.split("\n");
	
	for (int i = 0; i < lines.size(); i++) {
		String line = lines[i].strip_edges();
		if (line.begins_with("extends ")) {
			Dictionary extends_data;
			String extends_target = line.substr(8).strip_edges();
			
			// Remove any comments
			int comment_pos = extends_target.find("#");
			if (comment_pos >= 0) {
				extends_target = extends_target.substr(0, comment_pos).strip_edges();
			}
			
			extends_data["target"] = extends_target;
			extends_data["line"] = i + 1;
			extends_array.push_back(extends_data);
			break; // Only one extends per file
		}
	}
	
	return extends_array;
}

Array AIEnhancedGraphParser::_extract_external_resources(const String &p_content) {
	Array resources;
	Vector<String> lines = p_content.split("\n");
	
	for (int i = 0; i < lines.size(); i++) {
		String line = lines[i];
		
		// Look for [ext_resource type="..." path="..." id="..." uid="..."]
		// Godot 4 format: [ext_resource type="Script" uid="uid://xxx" path="res://main.gd" id="1_abc"]
		if (line.contains("[ext_resource") && line.contains("type=") && line.contains("path=")) {
			Dictionary resource;
			
			// Extract type
			int type_pos = line.find("type=\"");
			if (type_pos >= 0) {
				int type_end = line.find("\"", type_pos + 6);
				if (type_end >= 0) {
					resource["type"] = line.substr(type_pos + 6, type_end - type_pos - 6);
				}
			}
			
			// Extract path
			int path_pos = line.find("path=\"");
			if (path_pos >= 0) {
				int path_end = line.find("\"", path_pos + 6);
				if (path_end >= 0) {
					String path = line.substr(path_pos + 6, path_end - path_pos - 6);
					resource["path"] = _normalize_path(path);
				}
			}
			
			// CRITICAL: Extract BOTH uid and id for Godot 4
			// uid = long UID like "uid://c4wt6ace7hycd"
			int uid_pos = line.find("uid=\"");
			if (uid_pos >= 0) {
				int uid_end = line.find("\"", uid_pos + 5);
				if (uid_end >= 0) {
					resource["uid"] = line.substr(uid_pos + 5, uid_end - uid_pos - 5);
				}
			}
			
			// id = short key like "1_0r6n5" (THIS is what ExtResource() refs use!)
			int id_pos = line.find("id=\"");
			if (id_pos >= 0) {
				int id_end = line.find("\"", id_pos + 4);
				if (id_end >= 0) {
					String short_id = line.substr(id_pos + 4, id_end - id_pos - 4);
					resource["id"] = short_id;  // Store short ID as primary lookup key
				}
			}
			
			if (resource.has("path")) {
				resources.push_back(resource);
			}
		}
	}
	
	return resources;
}

Array AIEnhancedGraphParser::_parse_scene_connections(const String &p_content) {
	Array connections_arr;
	Vector<String> lines = p_content.split("\n");
	
	for (int i = 0; i < lines.size(); i++) {
		String line = lines[i];
		
		// Look for [connection signal="..." from="..." to="..." method="..."]
		if (line.contains("[connection") && line.contains("signal=") && line.contains("from=") && line.contains("to=") && line.contains("method=")) {
			Dictionary connection;
			
			// Extract signal name
			int signal_pos = line.find("signal=\"");
			if (signal_pos >= 0) {
				int signal_end = line.find("\"", signal_pos + 8);
				if (signal_end >= 0) {
					connection["signal_name"] = line.substr(signal_pos + 8, signal_end - signal_pos - 8);
				}
			}
			
			// Extract from node
			int from_pos = line.find("from=\"");
			if (from_pos >= 0) {
				int from_end = line.find("\"", from_pos + 6);
				if (from_end >= 0) {
					connection["from_node"] = line.substr(from_pos + 6, from_end - from_pos - 6);
				}
			}
			
			// Extract to node
			int to_pos = line.find("to=\"");
			if (to_pos >= 0) {
				int to_end = line.find("\"", to_pos + 4);
				if (to_end >= 0) {
					connection["to_node"] = line.substr(to_pos + 4, to_end - to_pos - 4);
				}
			}
			
			// Extract method
			int method_pos = line.find("method=\"");
			if (method_pos >= 0) {
				int method_end = line.find("\"", method_pos + 8);
				if (method_end >= 0) {
					connection["method"] = line.substr(method_pos + 8, method_end - method_pos - 8);
				}
			}
			
			if (connection.has("signal_name") && connection.has("method")) {
				connections_arr.push_back(connection);
			}
		}
	}
	
	return connections_arr;
}

Dictionary AIEnhancedGraphParser::get_graph_data() {
	Dictionary graph;
	
	// Convert nodes HashMap to Array
	Array nodes_array;
	for (const KeyValue<String, Dictionary> &E : nodes) {
		Dictionary node = E.value;
		node["file_path"] = E.key; // Ensure file_path is set
		nodes_array.push_back(node);
	}
	graph["nodes"] = nodes_array;
	
	// Connections
	graph["connections"] = connections;
	
	// Signal flows
	Dictionary signal_flows_dict;
	for (const KeyValue<String, Array> &E : signal_flows) {
		signal_flows_dict[E.key] = E.value;
	}
	graph["signal_flows"] = signal_flows_dict;
	
	// Autoloads
	Dictionary autoloads_dict;
	for (const KeyValue<String, String> &E : autoloads) {
		autoloads_dict[E.key] = E.value;
	}
	graph["autoloads"] = autoloads_dict;
	
	// Input actions - convert HashMap to Dictionary
	Dictionary input_actions_dict;
	for (const KeyValue<String, Dictionary> &E : input_actions) {
		input_actions_dict[E.key] = E.value;
	}
	graph["input_actions"] = input_actions_dict;
	
	// Resource dependencies
	Dictionary deps_dict;
	for (const KeyValue<String, Array> &E : resource_dependencies) {
		deps_dict[E.key] = E.value;
	}
	graph["resource_dependencies"] = deps_dict;
	
	// Summary statistics
	Dictionary summary;
	summary["total_nodes"] = nodes_array.size();
	summary["total_connections"] = connections.size();
	summary["total_signals"] = signal_flows.size();
	summary["total_autoloads"] = autoloads.size();
	summary["total_input_actions"] = input_actions.size();
	summary["total_files"] = nodes_array.size(); // Backend expects this field
	graph["summary"] = summary;
	
	// CRITICAL: Add version and timestamp for backend compatibility
	graph["version"] = "2.0.0"; // Enhanced graph version
	graph["created_at"] = Time::get_singleton()->get_datetime_string_from_system(false, true); // ISO format
	
	return graph;
}

Dictionary AIEnhancedGraphParser::get_context_for_file(const String &p_file_path) {
	Dictionary context;
	
	if (!nodes.has(p_file_path)) {
		return context;
	}
	
	Dictionary node = nodes[p_file_path];
	context["node_data"] = node;
	
	// Find all connections involving this file
	Array file_connections;
	for (int i = 0; i < connections.size(); i++) {
		Dictionary conn = connections[i];
		String source = conn.get("source_file", "");
		String target = conn.get("target_file", "");
		
		if (source == p_file_path || target == p_file_path) {
			file_connections.push_back(conn);
		}
	}
	context["connections"] = file_connections;
	
	// Find signals this file emits or receives
	Array signals_emitted_to;
	Array signals_received_from;
	
	for (const KeyValue<String, Array> &E : signal_flows) {
		String signal_name = E.key;
		Array flows = E.value;
		
		for (int i = 0; i < flows.size(); i++) {
			Dictionary flow = flows[i];
			String from_file = flow.get("from_file", "");
			String scene = flow.get("scene", "");
			
			if (from_file == p_file_path || scene == p_file_path) {
				Dictionary signal_info;
				signal_info["signal_name"] = signal_name;
				signal_info["flow_data"] = flow;
				signals_emitted_to.push_back(signal_info);
			}
			
			// Check if this file receives the signal (via handler method in scene)
			String handler = flow.get("handler_method", "");
			if (!handler.is_empty() && scene == p_file_path) {
				Dictionary signal_info;
				signal_info["signal_name"] = signal_name;
				signal_info["handler_method"] = handler;
				signal_info["flow_data"] = flow;
				signals_received_from.push_back(signal_info);
			}
		}
	}
	
	context["signals_emitted_to"] = signals_emitted_to;
	context["signals_received_from"] = signals_received_from;
	
	// Dependencies
	if (resource_dependencies.has(p_file_path)) {
		context["dependencies"] = resource_dependencies[p_file_path];
	}
	
	// WORLD-CLASS: Add which scenes use this script
	Array scripts_attached_to;
	for (int i = 0; i < connections.size(); i++) {
		Dictionary conn = connections[i];
		if (conn.get("connection_type", "") == "script_attachment" && 
		    conn.get("target_file", "") == p_file_path) {
			Dictionary attachment_info;
			attachment_info["scene_file"] = conn.get("source_file", "");
			attachment_info["node_name"] = conn.get("node_name", "");
			scripts_attached_to.push_back(attachment_info);
		}
	}
	if (scripts_attached_to.size() > 0) {
		context["scripts_attached_to"] = scripts_attached_to;
	}
	
	// Add extends information (what this script inherits from)
	if (nodes.has(p_file_path)) {
		Dictionary node_info = nodes[p_file_path];
		if (node_info.has("extends")) {
			Array extends = node_info.get("extends", Array());
			if (extends.size() > 0) {
				context["extends_from"] = extends;
			}
		}
	}
	
	// Add which files preload this file
	Array preloaded_by;
	for (int i = 0; i < connections.size(); i++) {
		Dictionary conn = connections[i];
		if (conn.get("connection_type", "") == "preload" && 
		    conn.get("target_file", "") == p_file_path) {
			preloaded_by.push_back(conn.get("source_file", ""));
		}
	}
	if (preloaded_by.size() > 0) {
		context["preloaded_by"] = preloaded_by;
	}
	
	return context;
}

Array AIEnhancedGraphParser::get_signal_flows() {
	Array flows;
	for (const KeyValue<String, Array> &E : signal_flows) {
		Dictionary signal_data;
		signal_data["signal_name"] = E.key;
		signal_data["flows"] = E.value;
		flows.push_back(signal_data);
	}
	return flows;
}

Dictionary AIEnhancedGraphParser::get_autoloads() {
	Dictionary result;
	for (const KeyValue<String, String> &E : autoloads) {
		result[E.key] = E.value;
	}
	return result;
}

Dictionary AIEnhancedGraphParser::get_input_actions() {
	// Convert HashMap to Dictionary
	Dictionary result;
	for (const KeyValue<String, Dictionary> &E : input_actions) {
		result[E.key] = E.value;
	}
	return result;
}

Dictionary AIEnhancedGraphParser::enrich_file_context(const String &p_file_path) {
	Dictionary enriched = get_context_for_file(p_file_path);
	
	// Add user-friendly summary with ALL relationships
	String summary_text = "";
	
	// Signals emitted
	Array signals_emitted = enriched.get("signals_emitted_to", Array());
	if (signals_emitted.size() > 0) {
		summary_text += "Emits " + String::num_int64(signals_emitted.size()) + " signal(s): ";
		for (int i = 0; i < MIN(signals_emitted.size(), 3); i++) {
			Dictionary sig = signals_emitted[i];
			if (i > 0) summary_text += ", ";
			summary_text += String(sig.get("signal_name", ""));
		}
		if (signals_emitted.size() > 3) summary_text += "...";
		summary_text += "; ";
	}
	
	// Signals received
	Array signals_received = enriched.get("signals_received_from", Array());
	if (signals_received.size() > 0) {
		summary_text += "Receives " + String::num_int64(signals_received.size()) + " signal(s); ";
	}
	
	// Script attachments (which scenes use this script)
	Array attached_to = enriched.get("scripts_attached_to", Array());
	if (attached_to.size() > 0) {
		summary_text += "Used in " + String::num_int64(attached_to.size()) + " scene(s): ";
		for (int i = 0; i < MIN(attached_to.size(), 2); i++) {
			Dictionary attach = attached_to[i];
			if (i > 0) summary_text += ", ";
			String scene_file = String(attach.get("scene_file", ""));
			summary_text += scene_file.get_file();
		}
		if (attached_to.size() > 2) summary_text += "...";
		summary_text += "; ";
	}
	
	// Extends
	Array extends = enriched.get("extends_from", Array());
	if (extends.size() > 0) {
		Dictionary extend_data = extends[0];
		summary_text += "Extends: " + String(extend_data.get("target", "")) + "; ";
	}
	
	// Preloaded by
	Array preloaded_by = enriched.get("preloaded_by", Array());
	if (preloaded_by.size() > 0) {
		summary_text += "Preloaded by " + String::num_int64(preloaded_by.size()) + " file(s); ";
	}
	
	// Dependencies
	Array deps = enriched.get("dependencies", Array());
	if (deps.size() > 0) {
		summary_text += "Depends on " + String::num_int64(deps.size()) + " resource(s); ";
	}
	
	// Method calls (which nodes this script calls methods on)
	if (nodes.has(p_file_path)) {
		Dictionary node_data = nodes[p_file_path];
		Array method_calls = node_data.get("method_calls", Array());
		if (method_calls.size() > 0) {
			summary_text += "Calls " + String::num_int64(method_calls.size()) + " method(s) on nodes; ";
			enriched["method_calls"] = method_calls;
		}
		
		// Node accesses (which nodes this script references)
		Array node_accesses = node_data.get("node_accesses", Array());
		if (node_accesses.size() > 0) {
			summary_text += "Accesses " + String::num_int64(node_accesses.size()) + " node(s); ";
			enriched["node_accesses"] = node_accesses;
		}
	}
	
	// General connections
	Array conns = enriched.get("connections", Array());
	if (conns.size() > 0) {
		summary_text += String::num_int64(conns.size()) + " connection(s) total";
	}
	
	enriched["summary"] = summary_text.strip_edges();
	
	return enriched;
}

Array AIEnhancedGraphParser::_extract_method_calls(const String &p_content) {
	// Extract method calls like: $HUD.show_message(), get_node("Player").start()
	Array method_calls;
	Vector<String> lines = p_content.split("\n");
	
	for (int i = 0; i < lines.size(); i++) {
		String line = lines[i];
		
		// Look for $NodeName.method() or get_node().method() patterns
		if (line.contains("$.") || line.contains("get_node(")) {
			Dictionary call_data;
			call_data["line"] = i + 1;
			call_data["code"] = line.strip_edges();
			
			// Extract node and method
			if (line.contains("$.")) {
				int dollar_pos = line.find("$");
				int dot_pos = line.find(".", dollar_pos);
				if (dot_pos > dollar_pos) {
					String node_name = line.substr(dollar_pos + 1, dot_pos - dollar_pos - 1);
					int paren_pos = line.find("(", dot_pos);
					if (paren_pos > dot_pos) {
						String method_name = line.substr(dot_pos + 1, paren_pos - dot_pos - 1);
						call_data["node"] = node_name;
						call_data["method"] = method_name;
						method_calls.push_back(call_data);
					}
				}
			}
		}
	}
	
	return method_calls;
}

Array AIEnhancedGraphParser::_extract_node_accesses(const String &p_content) {
	// Extract node references like: $Player, get_node("Enemy"), %Singleton
	Array node_accesses;
	Vector<String> lines = p_content.split("\n");
	
	for (int i = 0; i < lines.size(); i++) {
		String line = lines[i];
		
		// $NodeName pattern
		if (line.contains("$")) {
			int dollar_pos = 0;
			while ((dollar_pos = line.find("$", dollar_pos)) != -1) {
				// Extract node name (until space, dot, or special char)
				int end_pos = dollar_pos + 1;
				while (end_pos < line.length()) {
					char32_t ch = line[end_pos];
					if (!((ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z') || 
					      (ch >= '0' && ch <= '9') || ch == '_')) {
						break;
					}
					end_pos++;
				}
				
				if (end_pos > dollar_pos + 1) {
					String node_name = line.substr(dollar_pos + 1, end_pos - dollar_pos - 1);
					Dictionary access;
					access["node_name"] = node_name;
					access["line"] = i + 1;
					access["type"] = "dollar_reference";
					node_accesses.push_back(access);
				}
				
				dollar_pos = end_pos;
			}
		}
		
		// get_node("NodeName") pattern
		if (line.contains("get_node(")) {
			int get_node_pos = line.find("get_node(");
			int quote_start = line.find("\"", get_node_pos);
			if (quote_start >= 0) {
				int quote_end = line.find("\"", quote_start + 1);
				if (quote_end >= 0) {
					String node_path = line.substr(quote_start + 1, quote_end - quote_start - 1);
					Dictionary access;
					access["node_name"] = node_path;
					access["line"] = i + 1;
					access["type"] = "get_node";
					node_accesses.push_back(access);
				}
			}
		}
	}
	
	return node_accesses;
}

Array AIEnhancedGraphParser::_extract_property_modifications(const String &p_content) {
	// Extract property assignments like: visible = false, position.x = 100
	Array modifications;
	Vector<String> lines = p_content.split("\n");
	
	for (int i = 0; i < lines.size(); i++) {
		String line = lines[i].strip_edges();
		
		// Look for simple assignments (not ==, !=, <=, >=)
		if (line.contains(" = ") && !line.contains("==") && !line.contains("!=") && 
		    !line.contains("<=") && !line.contains(">=")) {
			int eq_pos = line.find(" = ");
			if (eq_pos > 0) {
				String lhs = line.substr(0, eq_pos).strip_edges();
				// Skip var declarations
				if (!lhs.begins_with("var ") && !lhs.begins_with("const ")) {
					Dictionary mod;
					mod["property"] = lhs;
					mod["line"] = i + 1;
					modifications.push_back(mod);
				}
			}
		}
	}
	
	return modifications;
}

Array AIEnhancedGraphParser::_extract_scene_instances(const String &p_content) {
	// Extract scene instance references from .tscn files
	Array instances;
	Vector<String> lines = p_content.split("\n");
	
	for (int i = 0; i < lines.size(); i++) {
		String line = lines[i];
		
		// Look for [node ... instance=ExtResource(...)]
		if (line.contains("[node") && line.contains("instance=ExtResource(")) {
			Dictionary instance;
			
			// Extract node name
			int name_pos = line.find("name=\"");
			if (name_pos >= 0) {
				int name_end = line.find("\"", name_pos + 6);
				if (name_end >= 0) {
					instance["node_name"] = line.substr(name_pos + 6, name_end - name_pos - 6);
				}
			}
			
			// Extract instance resource ID
			int inst_pos = line.find("instance=ExtResource(\"");
			if (inst_pos >= 0) {
				int inst_end = line.find("\"", inst_pos + 22);
				if (inst_end >= 0) {
					instance["resource_id"] = line.substr(inst_pos + 22, inst_end - inst_pos - 22);
				}
			}
			
			if (instance.has("node_name")) {
				instances.push_back(instance);
			}
		}
	}
	
	return instances;
}

Array AIEnhancedGraphParser::_extract_export_variables(const String &p_content) {
	// Extract @export variables: @export var mob_scene: PackedScene
	Array exports;
	Vector<String> lines = p_content.split("\n");
	
	for (int i = 0; i < lines.size(); i++) {
		String line = lines[i].strip_edges();
		if (line.begins_with("@export ") || line.begins_with("export ")) {
			Dictionary export_data;
			export_data["line"] = i + 1;
			
			// Extract variable name and type
			String export_line = line.begins_with("@export ") ? line.substr(8) : line.substr(7);
			if (export_line.begins_with("var ")) {
				String var_part = export_line.substr(4).strip_edges();
				int colon_pos = var_part.find(":");
				if (colon_pos >= 0) {
					String var_name = var_part.substr(0, colon_pos).strip_edges();
					String var_type = var_part.substr(colon_pos + 1).strip_edges();
					// Remove assignment if present
					int eq_pos = var_type.find("=");
					if (eq_pos >= 0) {
						var_type = var_type.substr(0, eq_pos).strip_edges();
					}
					export_data["name"] = var_name;
					export_data["type"] = var_type;
					exports.push_back(export_data);
				}
			}
		}
	}
	
	return exports;
}

Array AIEnhancedGraphParser::_extract_group_memberships(const String &p_content) {
	// Extract add_to_group() calls, call_group(), and is_in_group() checks
	Array groups;
	Vector<String> lines = p_content.split("\n");
	
	for (int i = 0; i < lines.size(); i++) {
		String line = lines[i];
		
		// add_to_group("group_name")
		if (line.contains("add_to_group(")) {
			int group_pos = line.find("add_to_group(");
			int quote_start = line.find("\"", group_pos);
			if (quote_start >= 0) {
				int quote_end = line.find("\"", quote_start + 1);
				if (quote_end >= 0) {
					String group_name = line.substr(quote_start + 1, quote_end - quote_start - 1);
					Dictionary group_data;
					group_data["group_name"] = group_name;
					group_data["line"] = i + 1;
					group_data["action"] = "add_to_group";
					group_data["code"] = line.strip_edges();
					groups.push_back(group_data);
				}
			}
		}
		// get_tree().call_group("group_name", "method_name")
		else if (line.contains("call_group(")) {
			int group_pos = line.find("call_group(");
			int quote_start = line.find("\"", group_pos);
			if (quote_start >= 0) {
				int quote_end = line.find("\"", quote_start + 1);
				if (quote_end >= 0) {
					String group_name = line.substr(quote_start + 1, quote_end - quote_start - 1);
					
					// Extract method name (second parameter)
					String method_name = "";
					int method_quote_start = line.find("\"", quote_end + 1);
					if (method_quote_start >= 0) {
						int method_quote_end = line.find("\"", method_quote_start + 1);
						if (method_quote_end >= 0) {
							method_name = line.substr(method_quote_start + 1, method_quote_end - method_quote_start - 1);
						}
					}
					
					Dictionary group_data;
					group_data["group_name"] = group_name;
					group_data["line"] = i + 1;
					group_data["action"] = "call_group";
					group_data["method_called"] = method_name;
					group_data["code"] = line.strip_edges();
					groups.push_back(group_data);
				}
			}
		}
		// is_in_group("group_name")
		else if (line.contains("is_in_group(")) {
			int group_pos = line.find("is_in_group(");
			int quote_start = line.find("\"", group_pos);
			if (quote_start >= 0) {
				int quote_end = line.find("\"", quote_start + 1);
				if (quote_end >= 0) {
					String group_name = line.substr(quote_start + 1, quote_end - quote_start - 1);
					Dictionary group_data;
					group_data["group_name"] = group_name;
					group_data["line"] = i + 1;
					group_data["action"] = "is_in_group";
					group_data["code"] = line.strip_edges();
					groups.push_back(group_data);
				}
			}
		}
	}
	
	return groups;
}

Array AIEnhancedGraphParser::_extract_dynamic_scene_loads(const String &p_content) {
	// Extract .instantiate(), load(), and scene variable usage
	Array loads;
	Vector<String> lines = p_content.split("\n");
	
	for (int i = 0; i < lines.size(); i++) {
		String line = lines[i];
		
		// Look for .instantiate() calls on PackedScene variables
		if (line.contains(".instantiate()")) {
			int inst_pos = line.find(".instantiate()");
			// Work backwards to find variable name
			int var_start = inst_pos - 1;
			while (var_start >= 0) {
				char32_t ch = line[var_start];
				if (!((ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z') || 
				      (ch >= '0' && ch <= '9') || ch == '_')) {
					var_start++;
					break;
				}
				var_start--;
			}
			if (var_start < inst_pos) {
				String scene_var = line.substr(var_start, inst_pos - var_start);
				Dictionary load_data;
				load_data["scene_variable"] = scene_var.strip_edges();
				load_data["line"] = i + 1;
				load_data["type"] = "instantiate";
				loads.push_back(load_data);
			}
		}
		
		// Look for load() calls
		if (line.contains("load(\"res://") && line.contains(".tscn")) {
			int load_pos = line.find("load(\"");
			int path_start = load_pos + 6;
			int path_end = line.find("\"", path_start);
			if (path_end > path_start) {
				String scene_path = line.substr(path_start, path_end - path_start);
				Dictionary load_data;
				load_data["scene_path"] = scene_path;
				load_data["line"] = i + 1;
				load_data["type"] = "load";
				loads.push_back(load_data);
			}
		}
	}
	
	return loads;
}

Array AIEnhancedGraphParser::_extract_scene_groups(const String &p_content) {
	// Extract groups=["mobs"] from .tscn node definitions
	Array groups;
	Vector<String> lines = p_content.split("\n");
	String current_node_name = "";
	
	for (int i = 0; i < lines.size(); i++) {
		String line = lines[i];
		
		// Track current node being defined
		if (line.contains("[node") && line.contains("name=\"")) {
			int name_pos = line.find("name=\"");
			if (name_pos >= 0) {
				int name_end = line.find("\"", name_pos + 6);
				if (name_end >= 0) {
					current_node_name = line.substr(name_pos + 6, name_end - name_pos - 6);
				}
			}
		}
		
		// Look for groups=["group1", "group2"] or groups=PoolStringArray("group1", "group2")
		if (line.contains("groups=") && !current_node_name.is_empty()) {
			int groups_pos = line.find("groups=");
			String groups_part = line.substr(groups_pos + 7).strip_edges();
			
			// Handle groups=["mobs"] format
			if (groups_part.begins_with("[")) {
				int bracket_end = groups_part.find("]");
				if (bracket_end >= 0) {
					String groups_content = groups_part.substr(1, bracket_end - 1);
					// Extract quoted strings
					int quote_start = 0;
					while ((quote_start = groups_content.find("\"", quote_start)) != -1) {
						int quote_end = groups_content.find("\"", quote_start + 1);
						if (quote_end > quote_start) {
							String group_name = groups_content.substr(quote_start + 1, quote_end - quote_start - 1);
							Dictionary group_data;
							group_data["group_name"] = group_name;
							group_data["node_name"] = current_node_name;
							group_data["line"] = i + 1;
							groups.push_back(group_data);
							quote_start = quote_end + 1;
						} else {
							break;
						}
					}
				}
			}
			// Handle groups=PoolStringArray("group1", "group2") format
			else if (groups_part.begins_with("PoolStringArray(")) {
				int paren_start = groups_part.find("(");
				int paren_end = groups_part.find(")", paren_start);
				if (paren_end > paren_start) {
					String groups_content = groups_part.substr(paren_start + 1, paren_end - paren_start - 1);
					// Extract quoted strings
					int quote_start = 0;
					while ((quote_start = groups_content.find("\"", quote_start)) != -1) {
						int quote_end = groups_content.find("\"", quote_start + 1);
						if (quote_end > quote_start) {
							String group_name = groups_content.substr(quote_start + 1, quote_end - quote_start - 1);
							Dictionary group_data;
							group_data["group_name"] = group_name;
							group_data["node_name"] = current_node_name;
							group_data["line"] = i + 1;
							groups.push_back(group_data);
							quote_start = quote_end + 1;
						} else {
							break;
						}
					}
				}
			}
		}
	}
	
	return groups;
}

Array AIEnhancedGraphParser::_extract_export_assignments(const String &p_content, const Array &p_external_resources, const Array &p_script_attachments) {
	// Extract export variable assignments: mob_scene = ExtResource("2")
	// This connects @export var mob_scene: PackedScene → ExtResource("2") → mob.tscn
	Array assignments;
	Vector<String> lines = p_content.split("\n");
	String current_node_name = "";
	String current_script_path = "";
	
	// Build lookup maps
	Dictionary ext_resources_by_id;
	for (int i = 0; i < p_external_resources.size(); i++) {
		Dictionary ext_res = p_external_resources[i];
		String id = ext_res.get("id", "");
		if (!id.is_empty()) {
			ext_resources_by_id[id] = ext_res;
		}
	}
	
	Dictionary scripts_by_node;
	for (int i = 0; i < p_script_attachments.size(); i++) {
		Dictionary attachment = p_script_attachments[i];
		String node_name = attachment.get("node_name", "");
		String script_path = attachment.get("script_path", "");
		if (!node_name.is_empty() && !script_path.is_empty()) {
			scripts_by_node[node_name] = script_path;
		}
	}
	
	for (int i = 0; i < lines.size(); i++) {
		String line = lines[i];
		
		// Track current node being defined
		if (line.contains("[node") && line.contains("name=\"")) {
			int name_pos = line.find("name=\"");
			if (name_pos >= 0) {
				int name_end = line.find("\"", name_pos + 6);
				if (name_end >= 0) {
					current_node_name = line.substr(name_pos + 6, name_end - name_pos - 6);
					// Update current script for this node
					if (scripts_by_node.has(current_node_name)) {
						current_script_path = scripts_by_node[current_node_name];
					} else {
						current_script_path = "";
					}
				}
			}
		}
		
		// Look for export variable assignments: mob_scene = ExtResource("2")
		// Pattern: variable_name = ExtResource("id")
		if (line.contains(" = ExtResource(") && !current_node_name.is_empty() && !current_script_path.is_empty()) {
			int eq_pos = line.find(" = ");
			if (eq_pos > 0) {
				String var_name = line.substr(0, eq_pos).strip_edges();
				
				// Extract ExtResource ID
				int ext_pos = line.find("ExtResource(\"");
				if (ext_pos >= 0) {
					int id_start = ext_pos + 13; // Length of "ExtResource(\""
					int id_end = line.find("\"", id_start);
					if (id_end > id_start) {
						String ext_id = line.substr(id_start, id_end - id_start);
						
						// Resolve ExtResource ID to actual file path
						if (ext_resources_by_id.has(ext_id)) {
							Dictionary ext_res = ext_resources_by_id[ext_id];
							String resource_path = ext_res.get("path", "");
							String resource_type = ext_res.get("type", "");
							
							if (!resource_path.is_empty()) {
								// Check if the script has this export variable
								// We'll verify this by checking if script is already parsed
								Dictionary assignment;
								assignment["export_var_name"] = var_name;
								assignment["script_path"] = current_script_path;
								assignment["resource_path"] = resource_path;
								assignment["resource_type"] = resource_type;
								assignment["node_name"] = current_node_name;
								assignment["ext_resource_id"] = ext_id;
								assignment["line"] = i + 1;
								
								// Try to get export var type from script if already parsed
								if (nodes.has(current_script_path)) {
									Dictionary script_node = nodes[current_script_path];
									Array exports = script_node.get("exports", Array());
									for (int j = 0; j < exports.size(); j++) {
										Dictionary exp = exports[j];
										if (String(exp.get("name", "")) == var_name) {
											assignment["export_var_type"] = exp.get("type", "");
											break;
										}
									}
								}
								
								assignments.push_back(assignment);
							}
						}
					}
				}
			}
		}
	}
	
	return assignments;
}

Array AIEnhancedGraphParser::_extract_function_signal_mappings(const String &p_content, const Array &p_functions, const Array &p_signals_emitted) {
	/**
	 * GAME-CHANGER: Map which functions emit which signals
	 * This is CRITICAL for multi-hop signal cascade tracing
	 * 
	 * Example output:
	 * [
	 *   {"function_name": "game_over", "signals_emitted": ["game_over_started"], "line": 42},
	 *   {"function_name": "_on_hit", "signals_emitted": ["hit", "health_changed"], "line": 78}
	 * ]
	 */
	Array mappings;
	Vector<String> lines = p_content.split("\n");
	
	// Build a map of line numbers to signal emissions
	HashMap<int, String> emissions_by_line;
	for (int i = 0; i < p_signals_emitted.size(); i++) {
		Dictionary emission = p_signals_emitted[i];
		int line_num = emission.get("line", 0);
		String signal_name = emission.get("signal_name", "");
		if (line_num > 0 && !signal_name.is_empty()) {
			emissions_by_line[line_num] = signal_name;
		}
	}
	
	// For each function, find which signals it emits
	for (int i = 0; i < p_functions.size(); i++) {
		Dictionary func = p_functions[i];
		String func_name = func.get("function_name", "");
		int func_start_line = func.get("line", 0);
		
		if (func_name.is_empty() || func_start_line == 0) {
			continue;
		}
		
		// Find the end of this function (next function start or end of file)
		int func_end_line = lines.size();
		if (i + 1 < p_functions.size()) {
			Dictionary next_func = p_functions[i + 1];
			func_end_line = next_func.get("line", lines.size());
		}
		
		// Collect all signals emitted within this function's body
		Array signals_in_function;
		HashSet<String> unique_signals; // Prevent duplicates
		
		for (int line_num = func_start_line; line_num < func_end_line; line_num++) {
			if (emissions_by_line.has(line_num)) {
				String signal_name = emissions_by_line[line_num];
				if (!unique_signals.has(signal_name)) {
					unique_signals.insert(signal_name);
					signals_in_function.push_back(signal_name);
				}
			}
		}
		
		// Only create mapping if function emits signals
		if (signals_in_function.size() > 0) {
			Dictionary mapping;
			mapping["function_name"] = func_name;
			mapping["signals_emitted"] = signals_in_function;
			mapping["line_start"] = func_start_line;
			mapping["line_end"] = func_end_line - 1;
			mappings.push_back(mapping);
		}
	}
	
	return mappings;
}

void AIEnhancedGraphParser::clear() {
	nodes.clear();
	connections.clear();
	signal_flows.clear();
	autoloads.clear();
	input_actions.clear();
	scene_hierarchies.clear();
	resource_dependencies.clear();
}

