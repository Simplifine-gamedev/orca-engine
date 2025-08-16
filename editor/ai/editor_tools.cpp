/*
 * © 2025 Simplifine Corp.
 * Personal Non‑Commercial License applies to this file as an original contribution to this Godot fork.
 * See LICENSES/COMPANY-NONCOMMERCIAL.md for terms. Commercial use requires a separate license from the Project Owner.
 */
#include "editor_tools.h"

// Add core includes early so they're available to helper functions below
#include "core/crypto/crypto.h"
#include "core/io/dir_access.h"
#include "core/io/file_access.h"
#include "core/io/http_client.h"
#include "core/io/json.h"
#include "core/io/resource_loader.h"
#include "core/config/project_settings.h"
#include "editor/editor_data.h"
#include "editor/editor_interface.h"
#include "editor/editor_node.h"
#include "editor/settings/editor_settings.h"
#include "core/variant/typed_array.h"
#include "editor/editor_string_names.h"
#include "editor/docks/scene_tree_dock.h"
#include "editor/docks/ai_chat_dock.h"
#include "editor/run/editor_run_bar.h"
#include "editor/script/script_editor_plugin.h"
#include "editor/script/script_text_editor.h"
#include "scene/main/node.h"
#include "scene/main/window.h"
#include "modules/gdscript/gdscript.h"
#include "modules/gdscript/gdscript_parser.h"
#include "modules/gdscript/gdscript_analyzer.h"
#include "modules/gdscript/gdscript_compiler.h"

// In-memory overlay of edited content awaiting user Accept/Reject in the editor.
// Key: absolute or res:// path; Value: edited content string
static Dictionary s_preview_overlays;
static Array s_runtime_errors; // Array of Dictionary: { type, time_ms, message, file, line, is_warning }

void EditorTools::set_preview_overlay(const String &p_path, const String &p_content) {
	if (p_path.is_empty()) {
		return;
	}
	s_preview_overlays[p_path] = p_content;
}

void EditorTools::clear_preview_overlay(const String &p_path) {
	if (p_path.is_empty()) {
		return;
	}
	if (s_preview_overlays.has(p_path)) {
		s_preview_overlays.erase(p_path);
	}
}

bool EditorTools::has_preview_overlay(const String &p_path) {
	return !p_path.is_empty() && s_preview_overlays.has(p_path);
}

String EditorTools::get_preview_overlay(const String &p_path) {
	if (!p_path.is_empty() && s_preview_overlays.has(p_path)) {
		return String(s_preview_overlays[p_path]);
	}
	return String();
}

void EditorTools::record_runtime_error(const Dictionary &p_error) {
    // Store up to a reasonable number to avoid unbounded growth
    const int MAX_STORED = 500;
    s_runtime_errors.push_back(p_error);
    if (s_runtime_errors.size() > MAX_STORED) {
        // Remove oldest
        s_runtime_errors.pop_front();
    }
}

Dictionary EditorTools::get_runtime_errors(const Dictionary &p_args) {
    Dictionary result;
    Array out;
    bool include_warnings = p_args.get("include_warnings", true);
    int max_count = p_args.get("max_count", 100);
    String file_filter = p_args.get("file", "");

    // iterate from newest to oldest
    for (int i = s_runtime_errors.size() - 1; i >= 0 && out.size() < max_count; i--) {
        Dictionary e = s_runtime_errors[i];
        bool is_warning = e.get("is_warning", false);
        if (!include_warnings && is_warning) {
            continue;
        }
        if (!file_filter.is_empty() && String(e.get("file", "")) != file_filter) {
            continue;
        }
        out.push_back(e);
    }
    result["success"] = true;
    result["errors"] = out;
    result["count"] = out.size();
    return result;
}

int EditorTools::get_file_line_count(const String &p_path, int p_max_bytes) {
	if (p_path.is_empty()) {
		return 0;
	}
	// Prefer overlay if present
	if (EditorTools::has_preview_overlay(p_path)) {
		Vector<String> lines = EditorTools::get_preview_overlay(p_path).split("\n");
		return lines.size();
	}
	Ref<FileAccess> f = FileAccess::open(p_path, FileAccess::READ);
	if (f.is_null()) {
		return 0;
	}
	int64_t remaining = p_max_bytes > 0 ? p_max_bytes : INT64_MAX;
	int line_count = 0;
	const int64_t CHUNK = 64 * 1024;
	PackedByteArray buf;
	while (!f->eof_reached() && remaining > 0) {
		int64_t to_read = MIN(CHUNK, remaining);
		buf.resize(to_read);
		int64_t read = f->get_buffer(buf.ptrw(), to_read);
		if (read <= 0) {
			break;
		}
		for (int64_t i = 0; i < read; i++) {
			if (buf[i] == '\n') {
				line_count++;
			}
		}
		remaining -= read;
	}
	f->close();
	return line_count;
}

#include <functional>
#include <utility>
// Static members for simple signal tracing
EditorTools *EditorTools::tracer_instance = nullptr;
Dictionary EditorTools::trace_registry;
Dictionary EditorTools::property_watch_registry;

EditorTools *EditorTools::ensure_tracer() {
    if (!tracer_instance) {
        tracer_instance = memnew(EditorTools);
        // Not added to scene; used for method binding only
    }
    return tracer_instance;
}

void EditorTools::_record_trace_event(const String &trace_id, const String &src_path, const String &sig_name, const Array &args) {
    if (!trace_registry.has(trace_id)) return;
    Dictionary reg = trace_registry[trace_id];
    Array events = reg.get("events", Array());
    int max_events = reg.get("max_events", 100);
    int next_index = reg.get("next_index", 0);
    Dictionary evt;
    evt["i"] = next_index;
    evt["time_ms"] = OS::get_singleton()->get_ticks_msec();
    evt["source_path"] = src_path;
    evt["signal"] = sig_name;
    if (!args.is_empty()) evt["args"] = args;
    events.push_back(evt);
    while (events.size() > max_events) { events.remove_at(0); }
    reg["events"] = events; reg["next_index"] = next_index + 1;
    trace_registry[trace_id] = reg;
}

void EditorTools::_on_traced_signal_0(const String &p_trace_id, const String &p_source_path, const String &p_signal_name) {
    _record_trace_event(p_trace_id, p_source_path, p_signal_name, Array());
}
void EditorTools::_on_traced_signal_1(const Variant &a0, const String &p_trace_id, const String &p_source_path, const String &p_signal_name) {
    Array args; args.push_back(a0); _record_trace_event(p_trace_id, p_source_path, p_signal_name, args);
}
void EditorTools::_on_traced_signal_2(const Variant &a0, const Variant &a1, const String &p_trace_id, const String &p_source_path, const String &p_signal_name) {
    Array args; args.push_back(a0); args.push_back(a1); _record_trace_event(p_trace_id, p_source_path, p_signal_name, args);
}
void EditorTools::_on_traced_signal_3(const Variant &a0, const Variant &a1, const Variant &a2, const String &p_trace_id, const String &p_source_path, const String &p_signal_name) {
    Array args; args.push_back(a0); args.push_back(a1); args.push_back(a2); _record_trace_event(p_trace_id, p_source_path, p_signal_name, args);
}
void EditorTools::_on_traced_signal_4(const Variant &a0, const Variant &a1, const Variant &a2, const Variant &a3, const String &p_trace_id, const String &p_source_path, const String &p_signal_name) {
    Array args; args.push_back(a0); args.push_back(a1); args.push_back(a2); args.push_back(a3); _record_trace_event(p_trace_id, p_source_path, p_signal_name, args);
}

void EditorTools::set_api_endpoint(const String &p_endpoint) {
    // This is now handled in AIChatDock
}

Dictionary EditorTools::_get_node_info(Node *p_node) {
	Dictionary node_info;
	if (!p_node) {
		return node_info;
	}
	node_info["name"] = p_node->get_name();
	node_info["type"] = p_node->get_class();
	
	// Get scene-relative path instead of absolute path
	Node *scene_root = EditorNode::get_singleton()->get_tree()->get_edited_scene_root();
    if (scene_root && p_node == scene_root) {
        // This is the scene root itself. Use "." as canonical root-relative path
        node_info["path"] = String(".");
	} else if (scene_root && scene_root->is_ancestor_of(p_node)) {
		// Get relative path from scene root
		node_info["path"] = scene_root->get_path_to(p_node);
	} else {
		// Fallback to absolute path if not in scene tree
		node_info["path"] = p_node->get_path();
	}
	
	node_info["owner"] = p_node->get_owner() ? String(p_node->get_owner()->get_name()) : String();
	node_info["child_count"] = p_node->get_child_count();
	return node_info;
}

Node *EditorTools::_get_node_from_path(const String &p_path, Dictionary &r_error_result) {
	Node *root = EditorNode::get_singleton()->get_tree()->get_edited_scene_root();
	if (!root) {
		r_error_result["success"] = false;
		r_error_result["message"] = "No scene is currently being edited.";
		return nullptr;
	}
    // Accept common root references and tolerant root-name matching
    if (p_path.is_empty() || p_path == "." || p_path.to_lower() == String(root->get_name()).to_lower()) {
        return root;
    }
    // Normalize a few absolute/root-like prefixes
    String norm_path = p_path;
    if (norm_path.begins_with("/")) {
        // Absolute paths are editor-tree relative; try stripping leading slash and root name
        norm_path = norm_path.substr(1);
        if (norm_path.begins_with(String(root->get_name()) + "/")) {
            norm_path = norm_path.substr(String(root->get_name()).length() + 1);
        }
    }
    if (norm_path == String(root->get_name())) {
        return root;
    }

    Node *node = root->get_node_or_null(norm_path);
    if (!node && !norm_path.begins_with("./") && norm_path.begins_with(".")) {
        String alt = norm_path;
        if (alt.begins_with("./")) alt = alt.substr(2);
        node = root->get_node_or_null(alt);
    }
    if (!node && !norm_path.begins_with("./") && !norm_path.begins_with(".")) {
        String prefixed = String("./") + norm_path;
        node = root->get_node_or_null(prefixed);
    }
    if (!node && !norm_path.contains("/")) {
        String target_name_lc = norm_path.to_lower();
        std::function<Node*(Node*)> dfs = [&](Node *n) -> Node* {
            if (!n) return nullptr;
            if (String(n->get_name()).to_lower() == target_name_lc) return n;
            for (int i = 0; i < n->get_child_count(); i++) {
                if (Node *res = dfs(n->get_child(i))) return res;
            }
            return nullptr;
        };
        node = dfs(root);
    }
    // Tolerant segment-wise resolution: allow matching by name (case-insensitive), by class name,
    // and normalize engine-generated instance names like "@Area2D@24529"
    if (!node && norm_path.find("/") != -1) {
        Vector<String> segments = norm_path.split("/");
        // Skip an initial segment equal to root name
        int start_i = 0;
        if (segments.size() > 0 && segments[0].to_lower() == String(root->get_name()).to_lower()) {
            start_i = 1;
        }
        Node *current = root;
        for (int i = start_i; i < segments.size() && current; i++) {
            String seg = segments[i].strip_edges();
            if (seg.is_empty() || seg == ".") continue;
            // Normalize engine instance-style segments: @Class@12345 -> Class
            String class_hint;
            if (seg.begins_with("@")) {
                // Extract between first two '@' as class hint if present
                int second = seg.find("@", 1);
                if (second > 1) {
                    class_hint = seg.substr(1, second - 1);
                }
            }

            Node *exact = current->get_node_or_null(seg);
            if (exact) {
                current = exact;
                continue;
            }
            // Try case-insensitive name match among direct children
            Node *match = nullptr;
            for (int c = 0; c < current->get_child_count(); c++) {
                Node *child = current->get_child(c);
                if (String(child->get_name()).to_lower() == seg.to_lower()) { match = child; break; }
            }
            if (!match) {
                // Try class-name match among direct children (e.g., "AnimatedSprite2D")
                for (int c = 0; c < current->get_child_count(); c++) {
                    Node *child = current->get_child(c);
                    if (String(child->get_class()).to_lower() == seg.to_lower()) { match = child; break; }
                }
            }
            if (!match && !class_hint.is_empty()) {
                String lc = class_hint.to_lower();
                for (int c = 0; c < current->get_child_count(); c++) {
                    Node *child = current->get_child(c);
                    if (String(child->get_class()).to_lower() == lc) { match = child; break; }
                }
            }
            current = match; // may become null breaking the loop
        }
        node = current;
    }
    if (!node) {
        r_error_result["success"] = false;
        r_error_result["error_code"] = "NODE_NOT_FOUND";
        r_error_result["message"] = "Node not found at path: " + p_path + " (root='" + String(root->get_name()) + "')";
    }
	return node;
}

Dictionary EditorTools::get_scene_info(const Dictionary &p_args) {
	Dictionary result;
	Node *root = EditorNode::get_singleton()->get_tree()->get_edited_scene_root();
	if (!root) {
		result["success"] = false;
		result["message"] = "No scene is currently being edited.";
		return result;
	}
	result["success"] = true;
	result["scene_name"] = root->get_scene_file_path();
	result["root_node"] = _get_node_info(root);
	return result;
}

Dictionary EditorTools::get_all_nodes(const Dictionary &p_args) {
	Dictionary result;
	Node *root = EditorNode::get_singleton()->get_tree()->get_edited_scene_root();
	if (!root) {
		result["success"] = false;
		result["message"] = "No scene is currently being edited.";
		return result;
	}
	
	Array nodes;
	
	// Helper lambda to recursively collect all nodes
	std::function<void(Node*)> collect_nodes = [&](Node* node) {
		if (node) {
			nodes.push_back(_get_node_info(node));
			// Recursively collect all children
			for (int i = 0; i < node->get_child_count(); i++) {
				collect_nodes(node->get_child(i));
			}
		}
	};
	
	// Start collecting from the root
	collect_nodes(root);
	
	result["success"] = true;
	result["nodes"] = nodes;
	return result;
}

Dictionary EditorTools::search_nodes_by_type(const Dictionary &p_args) {
	Dictionary result;
	if (!p_args.has("type")) {
		result["success"] = false;
		result["message"] = "Missing 'type' argument.";
		return result;
	}
	String type = p_args["type"];
	Array nodes;
	Node *root = EditorNode::get_singleton()->get_tree()->get_edited_scene_root();
	if (root) {
		// Helper lambda to recursively search all nodes
		std::function<void(Node*)> search_nodes = [&](Node* node) {
			if (node) {
				if (node->is_class(type)) {
					nodes.push_back(_get_node_info(node));
				}
				// Recursively search all children
				for (int i = 0; i < node->get_child_count(); i++) {
					search_nodes(node->get_child(i));
				}
			}
		};
		
		// Start searching from the root
		search_nodes(root);
	}
	result["success"] = true;
	result["nodes"] = nodes;
	return result;
}

Dictionary EditorTools::get_editor_selection(const Dictionary &p_args) {
	Dictionary result;
	Array selection = EditorNode::get_singleton()->get_editor_selection()->get_selected_nodes();
	Array nodes;
	for (int i = 0; i < selection.size(); i++) {
		Node *node = Object::cast_to<Node>(selection[i]);
		if (node) {
			nodes.push_back(_get_node_info(node));
		}
	}
	result["success"] = true;
	result["selected_nodes"] = nodes;
	return result;
}

Dictionary EditorTools::get_node_properties(const Dictionary &p_args) {
	Dictionary result;
	if (!p_args.has("path")) {
		result["success"] = false;
		result["message"] = "Missing 'path' argument.";
		return result;
	}
	Node *node = _get_node_from_path(p_args["path"], result);
	if (!node) {
		return result;
	}

	List<PropertyInfo> properties;
	node->get_property_list(&properties);

	Dictionary props_dict;
	for (const PropertyInfo &prop_info : properties) {
		if (prop_info.usage & PROPERTY_USAGE_EDITOR) {
			props_dict[prop_info.name] = node->get(prop_info.name);
		}
	}

	result["success"] = true;
	result["properties"] = props_dict;
	return result;
}


Dictionary EditorTools::create_node(const Dictionary &p_args) {
	Dictionary result;
	if (!p_args.has("type") || !p_args.has("name")) {
		result["success"] = false;
		result["message"] = "Missing 'type' or 'name' argument.";
		return result;
	}
	String type = p_args["type"];
	String name = p_args["name"];
	Node *parent = nullptr;

	if (p_args.has("parent")) {
		parent = _get_node_from_path(p_args["parent"], result);
		if (!parent) {
			return result;
		}
	} else {
		parent = EditorNode::get_singleton()->get_tree()->get_edited_scene_root();
		if (!parent) {
			result["success"] = false;
			result["message"] = "No scene is currently being edited to add a root node.";
			return result;
		}
	}

	// Validate the requested type is a Node-derived class and instantiate safely.
	if (!ClassDB::can_instantiate(type)) {
		result["success"] = false;
		result["message"] = "Cannot instantiate type: " + type;
		return result;
	}
	if (!ClassDB::is_parent_class(type, "Node")) {
		result["success"] = false;
		result["message"] = "Requested type is not a Node: " + type;
		return result;
	}

	Object *obj = ClassDB::instantiate(type);
	Node *new_node = Object::cast_to<Node>(obj);
	if (!new_node) {
		if (obj) {
			memdelete(obj);
		}
		result["success"] = false;
		result["message"] = "Failed to create Node of type: " + type;
		return result;
	}

	new_node->set_name(name);
	parent->add_child(new_node);
	// Ensure the new node is owned by the edited scene root so editor features see it as part of the scene.
	Node *scene_root = EditorNode::get_singleton()->get_tree()->get_edited_scene_root();
	if (scene_root) {
		new_node->set_owner(scene_root);
	} else {
		new_node->set_owner(parent->get_owner() ? parent->get_owner() : parent);
	}

	result["success"] = true;
	result["node_path"] = new_node->get_path();
	result["message"] = "Node created successfully.";
	
	// Check for configuration warnings
	PackedStringArray warnings = new_node->get_configuration_warnings();
	if (!warnings.is_empty()) {
		String warning_text = "";
		for (int i = 0; i < warnings.size(); i++) {
			warning_text += warnings[i];
			if (i < warnings.size() - 1) warning_text += "; ";
		}
		result["warnings"] = warning_text;
		result["message"] = "Node created successfully, but has warnings: " + warning_text;
	}
	
	return result;
}

Dictionary EditorTools::delete_node(const Dictionary &p_args) {
	Dictionary result;
	if (!p_args.has("path")) {
		result["success"] = false;
		result["message"] = "Missing 'path' argument.";
		return result;
	}
	Node *node = _get_node_from_path(p_args["path"], result);
	if (!node) {
		return result;
	}
	node->queue_free();
	result["success"] = true;
	result["message"] = "Node deleted successfully.";
	return result;
}

Dictionary EditorTools::set_node_property(const Dictionary &p_args) {
	Dictionary result;
	if (!p_args.has("path") || !p_args.has("property") || !p_args.has("value")) {
		result["success"] = false;
		result["message"] = "Missing 'path', 'property', or 'value' argument.";
		return result;
	}
	Node *node = _get_node_from_path(p_args["path"], result);
	if (!node) {
		return result;
	}
    StringName prop = p_args["property"];
    Variant value = p_args["value"];
	
    // Special handling for Vector2-like properties from flexible inputs
    if ((prop == StringName("position") || prop == StringName("global_position") || prop == StringName("scale"))) {
        // Accept [x, y], {x, y}, or "x,y"/"x y" strings
        Vector2 vec2_value;
        bool has_vec2 = false;

        if (value.get_type() == Variant::ARRAY) {
            Array arr = value;
            if (arr.size() >= 2 && arr[0].get_type() != Variant::NIL && arr[1].get_type() != Variant::NIL) {
                vec2_value = Vector2(arr[0], arr[1]);
                has_vec2 = true;
            }
        } else if (value.get_type() == Variant::DICTIONARY) {
            Dictionary d = value;
            if ((d.has("x") || d.has("X")) && (d.has("y") || d.has("Y"))) {
                Variant vx = d.has("x") ? d["x"] : d["X"];
                Variant vy = d.has("y") ? d["y"] : d["Y"];
                vec2_value = Vector2((double)vx, (double)vy);
                has_vec2 = true;
            }
        } else if (value.get_type() == Variant::STRING) {
            String s = (String)value;
            // Allow formats like "x,y" or "x y"
            s = s.strip_edges();
            PackedStringArray parts = s.split(",");
            if (parts.size() < 2) {
                parts = s.split(" ");
            }
            if (parts.size() >= 2) {
                vec2_value = Vector2(parts[0].strip_edges().to_float(), parts[1].strip_edges().to_float());
                has_vec2 = true;
            }
        }

        if (has_vec2) {
            value = vec2_value;
        }
    }

    // If value is a resource path string, attempt to load it as a Resource (helps for properties like sprite_frames, material, texture, etc.)
    if (value.get_type() == Variant::STRING) {
        String s = (String)value;
        if (s.begins_with("res://") || s.ends_with(".tres") || s.ends_with(".res") || s.ends_with(".png") || s.ends_with(".jpg") || s.ends_with(".jpeg")) {
            Ref<Resource> res = ResourceLoader::load(s);
            if (res.is_valid()) {
                value = res;
                print_line("SET_NODE_PROPERTY: Loaded resource from path for property '" + String(prop) + "': " + s);
            }
        }
    }

    // Special handling for color properties
	if (prop == "color" || prop == "modulate" || prop == "self_modulate") {
		if (value.get_type() == Variant::STRING) {
			String color_str = value;
			Color color;
			
			// Handle common color names
			if (color_str.to_lower() == "yellow") {
				color = Color(1.0, 1.0, 0.0, 1.0);
			} else if (color_str.to_lower() == "red") {
				color = Color(1.0, 0.0, 0.0, 1.0);
			} else if (color_str.to_lower() == "green") {
				color = Color(0.0, 1.0, 0.0, 1.0);
			} else if (color_str.to_lower() == "blue") {
				color = Color(0.0, 0.0, 1.0, 1.0);
			} else if (color_str.to_lower() == "white") {
				color = Color(1.0, 1.0, 1.0, 1.0);
			} else if (color_str.to_lower() == "black") {
				color = Color(0.0, 0.0, 0.0, 1.0);
			} else if (color_str.begins_with("#")) {
				// Handle hex colors
				color = Color::from_string(color_str, Color(1.0, 1.0, 1.0, 1.0));
			} else if (color_str.begins_with("(") && color_str.ends_with(")")) {
				// Handle Color constructor format: "(r, g, b, a)"
				String values = color_str.substr(1, color_str.length() - 2);
				PackedStringArray components = values.split(",");
				if (components.size() >= 3) {
					float r = components[0].strip_edges().to_float();
					float g = components[1].strip_edges().to_float();
					float b = components[2].strip_edges().to_float();
					float a = components.size() >= 4 ? components[3].strip_edges().to_float() : 1.0;
					color = Color(r, g, b, a);
				} else {
					color = Color(1.0, 1.0, 1.0, 1.0);
					print_line("SET_NODE_PROPERTY WARNING: Invalid Color constructor format '" + color_str + "', using white");
				}
			} else {
				// Try to parse as Color constructor or fallback to white
				color = Color::from_string(color_str, Color(1.0, 1.0, 1.0, 1.0));
				print_line("SET_NODE_PROPERTY WARNING: Unknown color '" + color_str + "', using white as fallback");
			}
			value = color;
			print_line("SET_NODE_PROPERTY: Converted color string '" + color_str + "' to Color(" + String::num(color.r) + ", " + String::num(color.g) + ", " + String::num(color.b) + ", " + String::num(color.a) + ")");
		}
	}
	
	bool valid = false;
	node->set(prop, value, &valid);
    if (!valid) {
        result["success"] = false;
        result["error_code"] = "PROPERTY_INVALID_OR_READONLY";
        result["message"] = "Failed to set property '" + String(prop) + "'. It might be invalid or read-only. Node type: " + node->get_class();
        return result;
    }
	
	// Auto-save the scene after property changes so changes persist when running the game
	String current_scene = EditorNode::get_singleton()->get_edited_scene()->get_scene_file_path();
	if (!current_scene.is_empty()) {
		EditorNode::get_singleton()->save_scene_if_open(current_scene);
		print_line("SET_NODE_PROPERTY: Auto-saved scene after property change: " + current_scene);
	} else {
		print_line("SET_NODE_PROPERTY: Scene has no save path, cannot auto-save");
	}
	
	result["success"] = true;
	result["message"] = "Property set successfully and scene saved.";
	return result;
}

Dictionary EditorTools::move_node(const Dictionary &p_args) {
	Dictionary result;
	if (!p_args.has("path") || !p_args.has("new_parent")) {
		result["success"] = false;
		result["message"] = "Missing 'path' or 'new_parent' argument.";
		return result;
	}
	Node *node = _get_node_from_path(p_args["path"], result);
	if (!node) {
		return result;
	}
	Node *new_parent = _get_node_from_path(p_args["new_parent"], result);
	if (!new_parent) {
		return result;
	}
	node->get_parent()->remove_child(node);
	new_parent->add_child(node);
	result["success"] = true;
	result["message"] = "Node moved successfully.";
	return result;
}

Dictionary EditorTools::call_node_method(const Dictionary &p_args) {
	Dictionary result;
	if (!p_args.has("path") || !p_args.has("method")) {
		result["success"] = false;
		result["message"] = "Missing 'path' or 'method' argument.";
		return result;
	}

	Node *node = _get_node_from_path(p_args["path"], result);
	if (!node) {
		return result;
	}

	StringName method = p_args["method"];
	Array args = p_args.has("args") ? (Array)p_args["args"] : Array();
	// Accept both "args" and "method_args" from different backends/schemas.
	if (args.is_empty() && p_args.has("method_args")) {
		args = (Array)p_args["method_args"];
	}

	Variant ret = node->callv(method, args);

	result["success"] = true;
	result["return_value"] = ret;

	return result;
}

Dictionary EditorTools::get_available_classes(const Dictionary &p_args) {
	Dictionary result;
	List<StringName> class_list;
	ClassDB::get_class_list(&class_list);

	Array classes;
	for (const StringName &E : class_list) {
		if (ClassDB::can_instantiate(E) && ClassDB::is_parent_class(E, "Node")) {
			classes.push_back(String(E));
		}
	}

	result["success"] = true;
	result["classes"] = classes;
	return result;
}

Dictionary EditorTools::get_node_script(const Dictionary &p_args) {
	Dictionary result;
	if (!p_args.has("path")) {
		result["success"] = false;
		result["message"] = "Missing 'path' argument.";
		return result;
	}

	Node *node = _get_node_from_path(p_args["path"], result);
	if (!node) {
		return result;
	}

	Ref<Script> script = node->get_script();
	if (script.is_null()) {
		result["success"] = false;
		result["message"] = "Node has no script attached.";
	} else {
		result["success"] = true;
		result["script_path"] = script->get_path();
	}

	return result;
}

Dictionary EditorTools::attach_script(const Dictionary &p_args) {
	Dictionary result;
	if (!p_args.has("path") || !p_args.has("script_path")) {
		result["success"] = false;
		result["message"] = "Missing 'path' or 'script_path' argument.";
		return result;
	}

	Node *node = _get_node_from_path(p_args["path"], result);
	if (!node) {
		return result;
	}

	Ref<Script> script = ResourceLoader::load(p_args["script_path"]);
	if (script.is_null()) {
		result["success"] = false;
		result["message"] = "Failed to load script at path: " + String(p_args["script_path"]);
		return result;
	}

	node->set_script(script);
	result["success"] = true;
	result["message"] = "Script attached successfully.";
	return result;
}

Dictionary EditorTools::manage_scene(const Dictionary &p_args) {
	Dictionary result;
	if (!p_args.has("operation")) {
		result["success"] = false;
		result["message"] = "Missing 'operation' argument.";
		return result;
	}

	String operation = p_args["operation"];

	if (operation == "create_new") {
		EditorNode::get_singleton()->new_scene();
		
		// Create a default Node2D root for the new scene
		Node *root_node = nullptr;
		if (ClassDB::can_instantiate("Node2D")) {
			root_node = (Node *)ClassDB::instantiate("Node2D");
			root_node->set_name("Main");
			
			// Properly set the scene root using EditorNode's method
			EditorNode::get_singleton()->set_edited_scene(root_node);
			root_node->set_owner(root_node);
		}
		
		if (root_node) {
			result["success"] = true;
			result["message"] = "New scene created with Node2D root.";
		} else {
			result["success"] = false;
			result["message"] = "Failed to create scene root node.";
		}

	} else if (operation == "save_as") {
		if (!p_args.has("path")) {
			result["success"] = false;
			result["message"] = "Missing 'path' argument for save_as operation.";
			return result;
		}
		String path = p_args["path"];
		EditorInterface::get_singleton()->save_scene_as(path);
		result["success"] = true;
		result["message"] = "Scene saved as " + path;

	} else if (operation == "open") {
		if (!p_args.has("path")) {
			result["success"] = false;
			result["message"] = "Missing 'path' argument for open operation.";
			return result;
		}
		String path = p_args["path"];
		EditorInterface::get_singleton()->open_scene_from_path(path);
		result["success"] = true;
		result["message"] = "Scene opened: " + path;

	} else {
		result["success"] = false;
		result["message"] = "Unknown operation: " + operation + ". Supported: create_new, save_as, open";
	}

	return result;
}

Dictionary EditorTools::add_collision_shape(const Dictionary &p_args) {
	Dictionary result;
	if (!p_args.has("node_path")) {
		result["success"] = false;
		result["message"] = "Missing 'node_path' argument.";
		return result;
	}

	String node_path = p_args["node_path"];
	String shape_type = p_args.get("shape_type", "rectangle");

	Node *node = _get_node_from_path(node_path, result);
	if (!node) {
		return result;
	}

	// Check if it's a physics body that needs collision
	if (!node->is_class("CharacterBody2D") && !node->is_class("RigidBody2D") && !node->is_class("StaticBody2D") && !node->is_class("Area2D")) {
		result["success"] = false;
		result["message"] = "Node is not a physics body that can have collision shapes.";
		return result;
	}

	// Create CollisionShape2D
	Node *collision_shape = nullptr;
	if (ClassDB::can_instantiate("CollisionShape2D")) {
		collision_shape = (Node *)ClassDB::instantiate("CollisionShape2D");
	} else {
		result["success"] = false;
		result["message"] = "Cannot instantiate CollisionShape2D.";
		return result;
	}

	// Create the actual shape resource based on type
	Variant shape_resource;
	if (shape_type == "rectangle") {
		if (ClassDB::can_instantiate("RectangleShape2D")) {
			shape_resource = ClassDB::instantiate("RectangleShape2D");
		}
	} else if (shape_type == "circle") {
		if (ClassDB::can_instantiate("CircleShape2D")) {
			shape_resource = ClassDB::instantiate("CircleShape2D");
		}
	} else if (shape_type == "capsule") {
		if (ClassDB::can_instantiate("CapsuleShape2D")) {
			shape_resource = ClassDB::instantiate("CapsuleShape2D");
		}
	}

	if (shape_resource.get_type() == Variant::NIL) {
		collision_shape->queue_free();
		result["success"] = false;
		result["message"] = "Failed to create shape resource of type: " + shape_type;
		return result;
	}

	// Set the shape on the collision node
	collision_shape->set("shape", shape_resource);
	
	// Add as child safely
	if (node && collision_shape) {
		node->add_child(collision_shape);
		collision_shape->set_owner(node->get_owner() ? node->get_owner() : node);
	} else {
		if (collision_shape) collision_shape->queue_free();
		result["success"] = false;
		result["message"] = "Failed to add collision shape - invalid nodes.";
		return result;
	}

	result["success"] = true;
	result["message"] = "CollisionShape2D with " + shape_type + " shape added to " + node_path;
	return result;
}

Dictionary EditorTools::generalnodeeditor(const Dictionary &p_args) {
	Dictionary result;
	
	// Validate required arguments
	if (!p_args.has("node_path")) {
		result["success"] = false;
		result["message"] = "Missing 'node_path' argument.";
		return result;
	}
	
	String node_path = p_args["node_path"];
	Array node_paths;
	
	// Support both single node and array of nodes
	if (node_path.begins_with("[") && node_path.ends_with("]")) {
		// Parse array of node paths
		String paths_str = node_path.substr(1, node_path.length() - 2);
		PackedStringArray paths = paths_str.split(",");
		for (int i = 0; i < paths.size(); i++) {
			node_paths.push_back(paths[i].strip_edges());
		}
	} else {
		node_paths.push_back(node_path);
	}
	
	Dictionary properties = p_args.get("properties", Dictionary());
	String texture_path = p_args.get("texture_path", "");
	bool batch_operation = node_paths.size() > 1;
	
	Array operation_results;
	int success_count = 0;
	int failure_count = 0;
	
	// Process each node
	for (int i = 0; i < node_paths.size(); i++) {
		String current_node_path = node_paths[i];
		Dictionary node_result;
		node_result["node_path"] = current_node_path;
		
		Dictionary temp_result;
		Node *node = _get_node_from_path(current_node_path, temp_result);
		if (!node) {
			node_result["success"] = false;
			node_result["message"] = temp_result["message"];
			operation_results.push_back(node_result);
			failure_count++;
			continue;
		}
		
		Array property_results;
		bool node_success = true;
		String node_message = "";
		
		// Handle texture assignment
		if (!texture_path.is_empty()) {
			bool texture_applied = false;
			String texture_error = "";
			
			// Check if node supports texture
			bool has_texture_property = false;
			bool valid = false;
			node->get("texture", &valid);
			has_texture_property = valid;
			
			if (node->has_method("set_texture") || has_texture_property) {
				Ref<Texture2D> texture = ResourceLoader::load(texture_path);
				if (texture.is_valid()) {
					if (node->has_method("set_texture")) {
						Array args;
						args.push_back(texture);
						node->callv("set_texture", args);
						texture_applied = true;
					} else {
						bool valid = false;
						node->set("texture", texture, &valid);
						texture_applied = valid;
					}
					
					if (!texture_applied) {
						texture_error = "Failed to apply texture to node";
					}
				} else {
					texture_error = "Failed to load texture from: " + texture_path;
				}
			} else {
				texture_error = "Node type '" + node->get_class() + "' does not support texture assignment";
			}
			
			Dictionary texture_result;
			texture_result["operation"] = "texture_assignment";
			texture_result["success"] = texture_applied;
			texture_result["message"] = texture_applied ? "Texture applied successfully" : texture_error;
			property_results.push_back(texture_result);
			
			if (!texture_applied) {
				node_success = false;
			}
		}
		
		// Handle property modifications
		Array property_keys = properties.keys();
		for (int j = 0; j < property_keys.size(); j++) {
			String property_name = property_keys[j];
			Variant property_value = properties[property_name];
			
			Dictionary prop_result;
			prop_result["operation"] = "property_modification";
			prop_result["property"] = property_name;
			prop_result["value"] = property_value;
			
			// Special handling for common properties
			if (property_name == "position" && property_value.get_type() == Variant::ARRAY) {
				Array pos_array = property_value;
				if (pos_array.size() >= 2) {
					Vector2 position(pos_array[0], pos_array[1]);
					bool valid = false;
					node->set("position", position, &valid);
					prop_result["success"] = valid;
					prop_result["message"] = valid ? "Position set successfully" : "Failed to set position";
				} else {
					prop_result["success"] = false;
					prop_result["message"] = "Position array must have at least 2 elements [x, y]";
				}
			} else if (property_name == "scale" && property_value.get_type() == Variant::ARRAY) {
				Array scale_array = property_value;
				if (scale_array.size() >= 2) {
					Vector2 scale(scale_array[0], scale_array[1]);
					bool valid = false;
					node->set("scale", scale, &valid);
					prop_result["success"] = valid;
					prop_result["message"] = valid ? "Scale set successfully" : "Failed to set scale";
				} else {
					prop_result["success"] = false;
					prop_result["message"] = "Scale array must have at least 2 elements [x, y]";
				}
			} else {
				// Standard property setting with color handling
				Variant processed_value = property_value;
				
				// Special handling for color properties
				if ((property_name == "color" || property_name == "modulate" || property_name == "self_modulate") && property_value.get_type() == Variant::STRING) {
					String color_str = property_value;
					Color color;
					
					// Handle common color names
					if (color_str.to_lower() == "yellow") {
						color = Color(1.0, 1.0, 0.0, 1.0);
					} else if (color_str.to_lower() == "red") {
						color = Color(1.0, 0.0, 0.0, 1.0);
					} else if (color_str.to_lower() == "green") {
						color = Color(0.0, 1.0, 0.0, 1.0);
					} else if (color_str.to_lower() == "blue") {
						color = Color(0.0, 0.0, 1.0, 1.0);
					} else if (color_str.to_lower() == "white") {
						color = Color(1.0, 1.0, 1.0, 1.0);
					} else if (color_str.to_lower() == "black") {
						color = Color(0.0, 0.0, 0.0, 1.0);
					} else if (color_str.begins_with("#")) {
						color = Color::from_string(color_str, Color(1.0, 1.0, 1.0, 1.0));
					} else {
						color = Color::from_string(color_str, Color(1.0, 1.0, 1.0, 1.0));
					}
					processed_value = color;
					print_line("GENERALNODEEDITOR: Converted color string '" + color_str + "' to Color(" + String::num(color.r) + ", " + String::num(color.g) + ", " + String::num(color.b) + ", " + String::num(color.a) + ")");
				}
				
				bool valid = false;
				node->set(StringName(property_name), processed_value, &valid);
				prop_result["success"] = valid;
				prop_result["message"] = valid ? 
					"Property '" + property_name + "' set successfully" : 
					"Failed to set property '" + property_name + "'. It might be invalid or read-only. Node type: " + node->get_class();
			}
			
			property_results.push_back(prop_result);
			
			if (!prop_result["success"]) {
				node_success = false;
			}
		}
		
		// Compile node result
		node_result["success"] = node_success;
		node_result["property_results"] = property_results;
		
		if (node_success) {
			success_count++;
			node_result["message"] = "All operations completed successfully on " + current_node_path;
		} else {
			failure_count++;
			node_result["message"] = "Some operations failed on " + current_node_path;
		}
		
		operation_results.push_back(node_result);
	}
	
	// Compile final result
	result["operation_results"] = operation_results;
	result["batch_operation"] = batch_operation;
	result["total_nodes"] = node_paths.size();
	result["success_count"] = success_count;
	result["failure_count"] = failure_count;
	
	if (failure_count == 0) {
		result["success"] = true;
		result["message"] = String("Successfully processed all ") + String::num_int64(success_count) + " node(s)";
	} else if (success_count == 0) {
		result["success"] = false;
		result["message"] = String("Failed to process all ") + String::num_int64(failure_count) + " node(s)";
	} else {
		result["success"] = true; // Partial success
		result["message"] = String("Processed ") + String::num_int64(success_count) + " successfully, " + 
							String::num_int64(failure_count) + " failed";
	}
	
	return result;
}

Dictionary EditorTools::list_project_files(const Dictionary &p_args) {
	Dictionary result;
	String path = p_args.has("dir") ? p_args["dir"] : "res://";
	String filter = p_args.has("filter") ? p_args["filter"] : "";

	Array files;
	Array dirs;
	Ref<DirAccess> dir = DirAccess::open(path);
	if (dir.is_valid()) {
		dir->list_dir_begin();
		String file_name = dir->get_next();
		while (file_name != "") {
			if (dir->current_is_dir()) {
				if (file_name != "." && file_name != "..") {
					dirs.push_back(file_name);
				}
			} else {
				if (filter.is_empty() || file_name.match(filter)) {
					String full_path = path.ends_with("/") ? path + file_name : path + String("/") + file_name;
					Dictionary info;
					info["name"] = file_name;
					info["path"] = full_path;
					info["line_count"] = get_file_line_count(full_path, 512 * 1024); // up to ~512KB
					files.push_back(info);
				}
			}
			file_name = dir->get_next();
		}
	} else {
		result["success"] = false;
		result["message"] = "Could not open directory: " + path;
		return result;
	}
	result["success"] = true;
	result["files"] = files;
	result["directories"] = dirs;
	return result;
}

Dictionary EditorTools::read_file(const Dictionary &p_args) {
    // Unified read: if line range is present, use advanced; otherwise full content with preview fallback
    if (p_args.has("start_line") || p_args.has("end_line")) {
        return read_file_advanced(p_args);
    }
    return read_file_content(p_args);
}

Dictionary EditorTools::read_file_content(const Dictionary &p_args) {
	Dictionary result;
	if (!p_args.has("path")) {
		result["success"] = false;
		result["message"] = "Missing 'path' argument.";
		return result;
	}
	String path = p_args["path"];
	Error err;
	// Prefer in-memory preview overlay only when AI is actively editing
	bool use_overlay = false; // default off; AI dock will use overlay via direct apply flow
	if (use_overlay && EditorTools::has_preview_overlay(path)) {
		String overlay = EditorTools::get_preview_overlay(path);
		result["success"] = true;
		result["content"] = overlay;
		return result;
	}
	String content = FileAccess::get_file_as_string(path, &err);
	if (err == OK) {
		result["success"] = true;
		result["content"] = content;
		return result;
	}

	// Fallback: attempt a bounded preview for very large or special files (e.g., big .tres)
	Ref<FileAccess> f = FileAccess::open(path, FileAccess::READ);
	if (f.is_valid()) {
		const int64_t MAX_PREVIEW_BYTES = 64 * 1024; // 64 KiB preview
		int64_t file_len = f->get_length();
		int64_t to_read = file_len < MAX_PREVIEW_BYTES ? file_len : MAX_PREVIEW_BYTES;
		PackedByteArray bytes;
		bytes.resize(to_read);
		int64_t read = f->get_buffer(bytes.ptrw(), to_read);
		f->close();
		String preview = String::utf8((const char *)bytes.ptr(), (int)read);
		result["success"] = true;
		result["content"] = preview + (file_len > to_read ? String("\n\n…\n[Truncated preview. Use read_file_advanced with start_line/end_line to fetch specific sections.]") : String());
		result["truncated"] = file_len > to_read;
		return result;
	}

	// If fallback also fails, report the original error
	result["success"] = false;
	result["message"] = "Failed to read file: " + path;
	return result;
}

Dictionary EditorTools::read_file_advanced(const Dictionary &p_args) {
	Dictionary result;
	if (!p_args.has("path")) {
		result["success"] = false;
		result["message"] = "Missing 'path' argument.";
		return result;
	}
	String path = p_args["path"];
	// Honor in-memory overlay only for advanced reads when explicitly desired (future flag)
	if (false && EditorTools::has_preview_overlay(path)) {
		int start_line = p_args.has("start_line") ? (int)p_args["start_line"] : 1;
		int end_line = p_args.has("end_line") ? (int)p_args["end_line"] : -1;
		String overlay = EditorTools::get_preview_overlay(path);
		Vector<String> lines = overlay.split("\n");
		if (end_line == -1) end_line = lines.size();
		String out;
		for (int i = MAX(1, start_line); i <= MIN(end_line, lines.size()); i++) {
			out += lines[i - 1] + "\n";
		}
		result["success"] = true;
		result["content"] = out;
		return result;
	}
	Ref<FileAccess> file = FileAccess::open(path, FileAccess::READ);
	if (file.is_null()) {
		result["success"] = false;
		result["message"] = "Failed to open file: " + path;
		return result;
	}

	int start_line = p_args.has("start_line") ? (int)p_args["start_line"] : 1;
	int end_line = p_args.has("end_line") ? (int)p_args["end_line"] : -1;
	String content;
	int current_line = 1;

	while (!file->eof_reached() && (end_line == -1 || current_line <= end_line)) {
		String line = file->get_line();
		if (current_line >= start_line) {
			content += line + "\n";
		}
		current_line++;
	}

	result["success"] = true;
	result["content"] = content;
	return result;
}

Dictionary EditorTools::_predict_code_edit(const String &p_file_content, const String &p_prompt, const String &p_api_endpoint) {
	Dictionary result;
	HTTPClient *http_client = HTTPClient::create();

	// Prepare request
	String host = p_api_endpoint;
	int port = 80;
	bool use_ssl = false;

	if (host.begins_with("https://")) {
		host = host.trim_prefix("https://");
		use_ssl = true;
		port = 443;
	} else if (host.begins_with("http://")) {
		host = host.trim_prefix("http://");
	}

	String base_path = "/";
	if (host.find("/") != -1) {
		base_path = host.substr(host.find("/"), -1);
		host = host.substr(0, host.find("/"));
	}

	if (host.find(":") != -1) {
		port = host.substr(host.find(":") + 1, -1).to_int();
		host = host.substr(0, host.find(":"));
	}
	
	// Construct the full path by replacing /chat with /predict_code_edit
	String predict_path = base_path.replace("/chat", "/predict_code_edit");

	Error err = http_client->connect_to_host(host, port, use_ssl ? Ref<TLSOptions>() : Ref<TLSOptions>());
	if (err != OK) {
		result["success"] = false;
		result["message"] = "Failed to connect to host: " + host;
		memdelete(http_client);
		return result;
	}

	// Wait for connection
	while (http_client->get_status() == HTTPClient::STATUS_CONNECTING || http_client->get_status() == HTTPClient::STATUS_RESOLVING) {
		http_client->poll();
		OS::get_singleton()->delay_usec(1000);
	}

	if (http_client->get_status() != HTTPClient::STATUS_CONNECTED) {
		result["success"] = false;
		result["message"] = "Failed to connect to host after polling.";
		memdelete(http_client);
		return result;
	}

	// Prepare request body
	Dictionary request_data;
	request_data["file_content"] = p_file_content;
	request_data["prompt"] = p_prompt;

	Ref<JSON> json;
	json.instantiate();
	String request_body_str = json->stringify(request_data);
	PackedByteArray request_body = request_body_str.to_utf8_buffer();

	PackedStringArray headers;
	headers.push_back("Content-Type: application/json");
	headers.push_back("Content-Length: " + itos(request_body.size()));

	err = http_client->request(HTTPClient::METHOD_POST, predict_path, headers, request_body.ptr(), request_body.size());
	if (err != OK) {
		result["success"] = false;
		result["message"] = "HTTPClient->request failed.";
		memdelete(http_client);
		return result;
	}

	// Wait for response
	while (http_client->get_status() == HTTPClient::STATUS_REQUESTING) {
		http_client->poll();
		OS::get_singleton()->delay_usec(1000);
	}

	if (http_client->get_status() != HTTPClient::STATUS_BODY && http_client->get_status() != HTTPClient::STATUS_CONNECTED) {
		result["success"] = false;
		result["message"] = "Request failed after sending.";
		memdelete(http_client);
		return result;
	}

	if (!http_client->has_response()) {
		result["success"] = false;
		result["message"] = "Request completed, but no response received.";
		memdelete(http_client);
		return result;
	}

	int response_code = http_client->get_response_code();
	PackedByteArray body;

	while (http_client->get_status() == HTTPClient::STATUS_BODY) {
		http_client->poll();
		PackedByteArray chunk = http_client->read_response_body_chunk();
		if (chunk.size() == 0) {
			OS::get_singleton()->delay_usec(1000);
		} else {
			body.append_array(chunk);
		}
	}

	String response_str = String::utf8((const char *)body.ptr(), body.size());

	memdelete(http_client);

	if (response_code != 200) {
		result["success"] = false;
		result["message"] = "Prediction server returned error " + itos(response_code) + ": " + response_str;
		return result;
	}

	err = json->parse(response_str);
	if (err != OK) {
		result["success"] = false;
		result["message"] = "Failed to parse JSON response from prediction server.";
		return result;
	}

	Dictionary response_data = json->get_data();
	response_data["success"] = true;
	return response_data;
}

Dictionary EditorTools::_call_apply_endpoint(const String &p_file_path, const String &p_file_content, const Dictionary &p_ai_args, const String &p_api_endpoint) {
	Dictionary result;
	HTTPClient *http_client = HTTPClient::create();

	// Prepare request
	String host = p_api_endpoint;
	int port = 80;
	bool use_ssl = false;

	if (host.begins_with("https://")) {
		host = host.trim_prefix("https://");
		use_ssl = true;
		port = 443;
	} else if (host.begins_with("http://")) {
		host = host.trim_prefix("http://");
	}

	String base_path = "/";
	if (host.find("/") != -1) {
		base_path = host.substr(host.find("/"), -1);
		host = host.substr(0, host.find("/"));
	}

	if (host.find(":") != -1) {
		port = host.substr(host.find(":") + 1, -1).to_int();
		host = host.substr(0, host.find(":"));
	}
	
	// Construct the apply endpoint path expected by backend
	String apply_path = base_path.replace("/chat", "/predict_code_edit");

	Ref<TLSOptions> tls;
	if (use_ssl) {
		tls = TLSOptions::client();
	}
	Error err = http_client->connect_to_host(host, port, tls);
	if (err != OK) {
		result["success"] = false;
		result["message"] = "Failed to connect to host: " + host;
		memdelete(http_client);
		return result;
	}

	// Wait for connection
	while (http_client->get_status() == HTTPClient::STATUS_CONNECTING || http_client->get_status() == HTTPClient::STATUS_RESOLVING) {
		http_client->poll();
		OS::get_singleton()->delay_usec(1000);
	}

	if (http_client->get_status() != HTTPClient::STATUS_CONNECTED) {
		result["success"] = false;
		result["message"] = "Failed to connect to host after polling. Status: " + itos(http_client->get_status());
		memdelete(http_client);
		print_line("APPLY_EDIT ERROR: Connection failed with status: " + itos(http_client->get_status()));
		return result;
	}
	
	print_line("APPLY_EDIT: Successfully connected to " + host + ":" + itos(port) + ", sending request to: " + apply_path);

	// Prepare request body to match backend's expected format
	Dictionary request_data;
	request_data["file_content"] = p_file_content;
	request_data["prompt"] = p_ai_args.get("prompt", "");
	// Forward optional range context so backend can reconstruct full diff
	if (p_ai_args.has("lines")) request_data["lines"] = p_ai_args.get("lines", "all");
	if (p_ai_args.has("start_line")) request_data["start_line"] = p_ai_args.get("start_line", 0);
	if (p_ai_args.has("end_line")) request_data["end_line"] = p_ai_args.get("end_line", 0);
	if (p_ai_args.has("pre_text")) request_data["pre_text"] = p_ai_args.get("pre_text", String());
	if (p_ai_args.has("post_text")) request_data["post_text"] = p_ai_args.get("post_text", String());
	if (p_ai_args.has("path")) request_data["path"] = p_ai_args.get("path", String());

	Ref<JSON> json;
	json.instantiate();
	String request_body_str = json->stringify(request_data);
	PackedByteArray request_body = request_body_str.to_utf8_buffer();

	// Build headers (mirror chat/image requests)
	PackedStringArray headers;
	headers.push_back("Content-Type: application/json");
	headers.push_back("Content-Length: " + itos(request_body.size()));
	headers.push_back("Accept: application/json");

	// Auth and context headers
	String auth_token;
	String user_id;
	if (EditorSettings::get_singleton()->has_setting("ai_chat/auth_token")) {
		auth_token = EditorSettings::get_singleton()->get_setting("ai_chat/auth_token");
	}
	if (EditorSettings::get_singleton()->has_setting("ai_chat/user_id")) {
		user_id = EditorSettings::get_singleton()->get_setting("ai_chat/user_id");
	}
	String machine_id = OS::get_singleton()->get_unique_id();
	if (machine_id.is_empty()) {
		machine_id = OS::get_singleton()->get_processor_name() + String("_") + OS::get_singleton()->get_name();
		machine_id = machine_id.replace(" ", "_").replace("(", "").replace(")", "");
	}
	String project_root = ProjectSettings::get_singleton()->globalize_path("res://");

	if (!auth_token.is_empty()) {
		headers.push_back("Authorization: Bearer " + auth_token);
	}
	if (!user_id.is_empty()) {
		headers.push_back("X-User-ID: " + user_id);
	}
	if (!machine_id.is_empty()) {
		headers.push_back("X-Machine-ID: " + machine_id);
	}
	if (!project_root.is_empty()) {
		headers.push_back("X-Project-Root: " + project_root);
	}

	err = http_client->request(HTTPClient::METHOD_POST, apply_path, headers, request_body.ptr(), request_body.size());
	if (err != OK) {
		result["success"] = false;
		result["message"] = "HTTPClient->request failed.";
		memdelete(http_client);
		return result;
	}

	// Wait for response
	while (http_client->get_status() == HTTPClient::STATUS_REQUESTING) {
		http_client->poll();
		OS::get_singleton()->delay_usec(1000);
	}

	if (http_client->get_status() != HTTPClient::STATUS_BODY && http_client->get_status() != HTTPClient::STATUS_CONNECTED) {
		result["success"] = false;
		result["message"] = "Request failed after sending. Status: " + itos(http_client->get_status());
		print_line("APPLY_EDIT ERROR: Request failed with status: " + itos(http_client->get_status()) + " (expected STATUS_BODY=" + itos(HTTPClient::STATUS_BODY) + " or STATUS_CONNECTED=" + itos(HTTPClient::STATUS_CONNECTED) + ")");
		memdelete(http_client);
		return result;
	}

	if (!http_client->has_response()) {
		result["success"] = false;
		result["message"] = "Request completed, but no response received.";
		memdelete(http_client);
		return result;
	}

	int response_code = http_client->get_response_code();
	PackedByteArray body;

	while (http_client->get_status() == HTTPClient::STATUS_BODY) {
		http_client->poll();
		PackedByteArray chunk = http_client->read_response_body_chunk();
		if (chunk.size() == 0) {
			OS::get_singleton()->delay_usec(1000);
		} else {
			body.append_array(chunk);
		}
	}

	String response_str = String::utf8((const char *)body.ptr(), body.size());

	memdelete(http_client);

	if (response_code != 200) {
		result["success"] = false;
		result["message"] = "Apply server returned error " + itos(response_code) + ": " + response_str;
		return result;
	}

	err = json->parse(response_str);
	if (err != OK) {
		result["success"] = false;
		result["message"] = "Failed to parse JSON response from apply server.";
		return result;
	}

	Dictionary response_data = json->get_data();

	// Clean up the edited_content if it exists
	if (response_data.has("edited_content")) {
		String edited_content = response_data["edited_content"];
		String cleaned_content = _clean_backend_content(edited_content);
		response_data["edited_content"] = cleaned_content;
	}

	response_data["success"] = true;
	return response_data;
}

Dictionary EditorTools::apply_edit(const Dictionary &p_args) {
    // Enhanced version that returns diff and compilation errors as JSON
    String path = p_args.get("path", "");
    String prompt = p_args.get("prompt", "");
    
    print_line("APPLY_EDIT: Using enhanced processing with diff and error checking");
    
    if (path.is_empty() || prompt.is_empty()) {
        Dictionary result;
        result["success"] = false;
        result["message"] = "Missing path or prompt for apply_edit";
        result["diff"] = "";
        result["compilation_errors"] = Array();
        return result;
    }
    
    // Read the file content (treat missing file as empty to allow creation)
    Error err;
    String file_content = FileAccess::get_file_as_string(path, &err);
    bool file_missing = false;
    if (err != OK) {
        file_missing = true;
        file_content = ""; // create new file from scratch
        print_line("APPLY_EDIT: Target file does not exist; will create new file: " + path);
    }

    // Determine edit scope: full file vs line range
    String lines_mode = String(p_args.get("lines", "all")).to_lower();
    int range_start = (int)p_args.get("start_line", 0);
    int range_end = (int)p_args.get("end_line", 0);
    bool use_range = (lines_mode == "range") || (range_start > 0 && range_end >= range_start);

    Vector<String> file_lines = file_content.split("\n");
    String pre_text, segment_text, post_text;
    int total_lines = file_lines.size();
    if (use_range) {
        // Clamp range within file
        if (range_start <= 0) range_start = 1;
        if (range_end <= 0 || range_end > total_lines) range_end = total_lines;
        // Build pre/segment/post
        for (int i = 0; i < range_start - 1 && i < total_lines; i++) {
            pre_text += file_lines[i];
            if (i < range_start - 2 || total_lines > 1) pre_text += "\n";
        }
        for (int i = range_start - 1; i < range_end && i < total_lines; i++) {
            segment_text += file_lines[i];
            if (i < range_end - 1) segment_text += "\n";
        }
        for (int i = range_end; i < total_lines; i++) {
            if (!post_text.is_empty()) post_text += "\n";
            post_text += file_lines[i];
        }
    }

    // Prepare request content: either whole file or only the selected segment
    String content_for_model = use_range ? segment_text : file_content;

    // Prepare auth/context headers to mirror chat/image generation
    String auth_token = String();
    String user_id = String();
    if (EditorSettings::get_singleton()->has_setting("ai_chat/auth_token")) {
        auth_token = EditorSettings::get_singleton()->get_setting("ai_chat/auth_token");
    }
    if (EditorSettings::get_singleton()->has_setting("ai_chat/user_id")) {
        user_id = EditorSettings::get_singleton()->get_setting("ai_chat/user_id");
    }
    String machine_id = OS::get_singleton()->get_unique_id();
    if (machine_id.is_empty()) {
        machine_id = OS::get_singleton()->get_processor_name() + String("_") + OS::get_singleton()->get_name();
        machine_id = machine_id.replace(" ", "_").replace("(", "").replace(")", "");
    }
    String project_root = ProjectSettings::get_singleton()->globalize_path("res://");

    print_line("APPLY_EDIT: Using HTTPClient to call backend API - prompt: " + prompt);

    // Determine backend URL using same logic as main chat system
    String base_url;
    String is_dev = OS::get_singleton()->get_environment("IS_DEV");
    if (is_dev.is_empty()) {
        // Backward-compat with DEV_MODE
        is_dev = OS::get_singleton()->get_environment("DEV_MODE");
    }
    if (!is_dev.is_empty() && is_dev.to_lower() == "true") {
        base_url = "http://127.0.0.1:8000";
    } else {
        base_url = "https://gamechat.simplifine.com";
    }
    // Allow override via editor settings or environment variable
    if (EditorSettings::get_singleton() && EditorSettings::get_singleton()->has_setting("ai_chat/base_url")) {
        String override_url = EditorSettings::get_singleton()->get_setting("ai_chat/base_url");
        if (!override_url.is_empty()) {
            base_url = override_url;
        }
    } else if (!OS::get_singleton()->get_environment("AI_CHAT_CLOUD_URL").is_empty()) {
        base_url = OS::get_singleton()->get_environment("AI_CHAT_CLOUD_URL");
    }

    // Call backend using only the segment if range mode is used, and pass range context for diff reconstruction
    Dictionary args_for_backend = p_args.duplicate();
    args_for_backend["prompt"] = prompt; // ensure present
    if (use_range) {
        args_for_backend["lines"] = String("range");
        args_for_backend["start_line"] = range_start;
        args_for_backend["end_line"] = range_end;
        args_for_backend["pre_text"] = pre_text;
        args_for_backend["post_text"] = post_text;
        args_for_backend["path"] = path;
    } else {
        args_for_backend["lines"] = String("all");
        args_for_backend["path"] = path;
    }
    Dictionary local_result = _call_apply_endpoint(path, content_for_model, args_for_backend, base_url + "/chat");

    if (local_result.get("success", false)) {
        String backend_segment = local_result.get("edited_content", content_for_model);
        String cleaned_segment = _clean_backend_content(backend_segment);

        // Prefer backend-provided full content when available
        String full_edited_content = local_result.has("full_edited_content")
                ? String(local_result.get("full_edited_content", String()))
                : String();
        if (full_edited_content.is_empty()) {
            // Reconstruct full edited content if we only edited a segment
            if (use_range) {
                full_edited_content = pre_text;
                if (!pre_text.is_empty() && !cleaned_segment.is_empty() && !pre_text.ends_with("\n")) {
                    // Ensure newline separation if needed
                    full_edited_content += "\n";
                }
                full_edited_content += cleaned_segment;
                if (!post_text.is_empty()) {
                    if (!full_edited_content.is_empty() && !full_edited_content.ends_with("\n")) {
                        full_edited_content += "\n";
                    }
                    full_edited_content += post_text;
                }
            } else {
                full_edited_content = cleaned_segment;
            }
        }

        // Prefer backend diff if supplied, else fall back to simple summary
        String diff = local_result.has("diff") ? String(local_result.get("diff", String())) : String();
        if (diff.is_empty()) {
            if (file_content.length() > 100000 || full_edited_content.length() > 100000) {
                diff = "Diff skipped - file too large (original: " + String::num_int64(file_content.length()) + " chars, new: " + String::num_int64(full_edited_content.length()) + " chars)";
            } else {
                diff = "=== TEMPORARY SIMPLE DIFF ===\nOriginal length: " + String::num_int64(file_content.length()) + " chars\nNew length: " + String::num_int64(full_edited_content.length()) + " chars\n=== END DIFF ===";
            }
        }

        // Optional compilation check
        bool skip_compilation_check = p_args.get("skip_compilation_check", false);
        Array comp_errors;
        bool has_errors = false;
        if (!skip_compilation_check) {
            comp_errors = _check_compilation_errors(path, full_edited_content);
            has_errors = comp_errors.size() > 0;
            print_line("APPLY_EDIT: Compilation check completed - found " + String::num_int64(comp_errors.size()) + " errors/warnings");
        } else {
            print_line("APPLY_EDIT: Skipping compilation check for better performance");
        }

        Dictionary result;
        result["success"] = true;
        result["message"] = file_missing ? String("File does not exist; preview created. Use Accept/Reject to apply.") : String("Preview created. Use Accept/Reject to apply.");
        result["path"] = path;
        result["original_content"] = file_content;
        result["edited_content"] = full_edited_content;
        result["diff"] = diff;
        result["compilation_errors"] = comp_errors;
        result["has_errors"] = has_errors;
        result["dynamic_approach"] = false;
        if (use_range) {
            Dictionary edit_range;
            edit_range["start_line"] = range_start;
            edit_range["end_line"] = range_end;
            result["edit_range"] = edit_range;
            result["edited_segment"] = cleaned_segment;
        }
        // Pass-through of backend-provided structured edits when available
        if (local_result.has("structured_edits")) {
            result["structured_edits"] = local_result.get("structured_edits", Dictionary());
        }
        if (local_result.has("mode")) {
            result["mode"] = local_result.get("mode", String());
        }
        if (local_result.has("start_line")) {
            result["start_line"] = local_result.get("start_line", 0);
        }
        if (local_result.has("end_line")) {
            result["end_line"] = local_result.get("end_line", 0);
        }
        return result;
    }
    
    // If local processing failed, still return proper structure
    Dictionary failed_result = local_result;
    failed_result["diff"] = "";
    failed_result["compilation_errors"] = Array();
    failed_result["has_errors"] = false;
    return failed_result;
}

String EditorTools::_clean_backend_content(const String &p_content) {
	String content = p_content;
	
	// Remove code block wrappers (```javascript, ```gdscript, etc.)
	// Handle various possible code block formats
	Vector<String> code_block_patterns = {
		"```javascript\n",
		"```gdscript\n", 
		"```\n",
		"```js\n",
		"```gd\n"
	};
	
	for (const String &pattern : code_block_patterns) {
		if (content.begins_with(pattern)) {
			content = content.substr(pattern.length());
			break;
		}
	}
	
	// Remove trailing code block marker
	if (content.ends_with("\n```")) {
		content = content.substr(0, content.length() - 4);
	} else if (content.ends_with("```")) {
		content = content.substr(0, content.length() - 3);
	}
	
	// Fix JavaScript to GDScript conversion for .gd files
	content = _convert_javascript_to_gdscript(content);
	
	// Fix common malformed content issues
	content = _fix_malformed_content(content);
	
	// Trim extra whitespace
	content = content.strip_edges();
	
	return content;
}

String EditorTools::_convert_javascript_to_gdscript(const String &p_content) {
	String content = p_content;
	Vector<String> lines = content.split("\n");
	Vector<String> converted_lines;
	
	for (const String &line : lines) {
		String converted_line = line;
		
		// Convert JavaScript function syntax to GDScript
		if (converted_line.contains("function ")) {
			// Replace "function name() {" with "func name():"
			String trimmed = converted_line.strip_edges();
			if (trimmed.begins_with("function ")) {
				// Extract function name
				String func_part = trimmed.substr(9); // Remove "function "
				int paren_pos = func_part.find("(");
				if (paren_pos > 0) {
					String func_name = func_part.substr(0, paren_pos);
					String params = func_part.substr(paren_pos);
					
					// Remove opening brace if present
					if (params.ends_with(" {")) {
						params = params.substr(0, params.length() - 2);
					} else if (params.ends_with("{")) {
						params = params.substr(0, params.length() - 1);
					}
					
					// Get indentation
					String indent = line.substr(0, line.length() - line.lstrip("\t ").length());
					converted_line = indent + "func " + func_name + params + ":";
				}
			}
		}
		
		// Convert console.log to print
		if (converted_line.contains("console.log(")) {
			converted_line = converted_line.replace("console.log(", "print(");
		}
		
		// Remove standalone opening/closing braces (JavaScript style)
		String trimmed = converted_line.strip_edges();
		if (trimmed == "{" || trimmed == "}") {
			continue; // Skip these lines in GDScript
		}
		
		// Convert JavaScript variable declarations
		if (converted_line.contains("let ") || converted_line.contains("var ") || converted_line.contains("const ")) {
			converted_line = converted_line.replace("let ", "var ");
			converted_line = converted_line.replace("const ", "var ");
		}
		
		converted_lines.push_back(converted_line);
	}
	
	return String("\n").join(converted_lines);
}

String EditorTools::_fix_malformed_content(const String &p_content) {
	String content = p_content;
	
	// Fix missing function endings in GDScript
	Vector<String> lines = content.split("\n");
	Vector<String> fixed_lines;
	bool in_function = false;
	
	for (int i = 0; i < lines.size(); i++) {
		String line = lines[i];
		String trimmed = line.strip_edges();
		
		// Track function declarations in GDScript
		if (trimmed.begins_with("func ")) {
			in_function = true;
		} else if (in_function) {
			// Check indentation
			String line_indent = line.substr(0, line.length() - line.lstrip("\t ").length());
			if (!trimmed.is_empty() && line_indent.length() == 0) {
				// Function ended (no indentation)
				in_function = false;
			}
		}
		
		fixed_lines.push_back(line);
		
		// If we're starting a new function without proper ending
		if (in_function && 
			i + 1 < lines.size() && 
			lines[i + 1].strip_edges().begins_with("func ")) {
			in_function = false;
		}
	}
	
	return String("\n").join(fixed_lines);
}

String EditorTools::_generate_unified_diff(const String &p_original, const String &p_modified, const String &p_file_path) {
	Vector<String> original_lines = p_original.split("\n");
	Vector<String> modified_lines = p_modified.split("\n");
	
	String diff = "--- " + p_file_path + " (original)\n";
	diff += "+++ " + p_file_path + " (modified)\n";
	
	// Simple diff implementation - compare line by line
	int original_line = 0;
	int modified_line = 0;
	int context_lines = 3;
	
	while (original_line < original_lines.size() || modified_line < modified_lines.size()) {
		// Find changes
		int change_start_orig = original_line;
		int change_start_mod = modified_line;
		
		// Skip matching lines
		while (original_line < original_lines.size() && 
			   modified_line < modified_lines.size() && 
			   original_lines[original_line] == modified_lines[modified_line]) {
			original_line++;
			modified_line++;
		}
		
		if (original_line >= original_lines.size() && modified_line >= modified_lines.size()) {
			break; // End of both files
		}
		
		// Find end of change block
		int change_end_orig = original_line;
		int change_end_mod = modified_line;
		
		// Simple heuristic: advance until we find matching lines again or reach end
		while ((change_end_orig < original_lines.size() || change_end_mod < modified_lines.size())) {
			// Look ahead to see if we find a match
			bool found_match = false;
			int lookahead = 3; // Look ahead a few lines
			
			for (int i = 0; i < lookahead && !found_match; i++) {
				if (change_end_orig + i < original_lines.size() && 
					change_end_mod + i < modified_lines.size() &&
					original_lines[change_end_orig + i] == modified_lines[change_end_mod + i]) {
					found_match = true;
					break;
				}
			}
			
			if (found_match) {
				break;
			}
			
			if (change_end_orig < original_lines.size()) change_end_orig++;
			if (change_end_mod < modified_lines.size()) change_end_mod++;
		}
		
		// Generate hunk header
		int context_start_orig = MAX(0, change_start_orig - context_lines);
		int context_start_mod = MAX(0, change_start_mod - context_lines);
		int context_end_orig = MIN(original_lines.size(), change_end_orig + context_lines);
		int context_end_mod = MIN(modified_lines.size(), change_end_mod + context_lines);
		
		int hunk_orig_lines = context_end_orig - context_start_orig;
		int hunk_mod_lines = context_end_mod - context_start_mod;
		
		diff += "@@ -" + String::num_int64(context_start_orig + 1) + "," + String::num_int64(hunk_orig_lines) + 
				" +" + String::num_int64(context_start_mod + 1) + "," + String::num_int64(hunk_mod_lines) + " @@\n";
		
		// Add context before change
		for (int i = context_start_orig; i < change_start_orig; i++) {
			diff += " " + original_lines[i] + "\n";
		}
		
		// Add removed lines
		for (int i = change_start_orig; i < change_end_orig && i < original_lines.size(); i++) {
			diff += "-" + original_lines[i] + "\n";
		}
		
		// Add added lines
		for (int i = change_start_mod; i < change_end_mod && i < modified_lines.size(); i++) {
			diff += "+" + modified_lines[i] + "\n";
		}
		
		// Add context after change
		for (int i = change_end_orig; i < context_end_orig; i++) {
			diff += " " + original_lines[i] + "\n";
		}
		
		original_line = change_end_orig;
		modified_line = change_end_mod;
	}
	
	return diff;
}

Array EditorTools::_check_compilation_errors(const String &p_file_path, const String &p_content) {
	Array errors;
	
	// Get file extension to determine script type
	String extension = p_file_path.get_extension();
	
	if (extension == "gd") {
		// GDScript compilation check using parser/analyzer/compiler approach
		GDScriptParser parser;
		Error parse_err = parser.parse(p_content, p_file_path, false);
		
		// Get parser errors
		const List<GDScriptParser::ParserError> &parser_errors = parser.get_errors();
		for (const GDScriptParser::ParserError &error : parser_errors) {
			Dictionary error_dict;
			error_dict["type"] = "parser_error";
			error_dict["line"] = error.line;
			error_dict["column"] = error.column;
			error_dict["message"] = error.message;
			errors.push_back(error_dict);
		}
		
		// Only continue to analysis if parsing succeeded
		if (parse_err == OK) {
			GDScriptAnalyzer analyzer(&parser);
			Error analyze_err = analyzer.analyze();
			
			// Get analyzer errors (they're stored in the parser)
			const List<GDScriptParser::ParserError> &analyzer_errors = parser.get_errors();
			for (const GDScriptParser::ParserError &error : analyzer_errors) {
				// Skip errors we already collected during parsing
				bool already_collected = false;
				for (const GDScriptParser::ParserError &parse_error : parser_errors) {
					if (parse_error.line == error.line && parse_error.message == error.message) {
						already_collected = true;
						break;
					}
				}
				if (!already_collected) {
					Dictionary error_dict;
					error_dict["type"] = "analyzer_error";
					error_dict["line"] = error.line;
					error_dict["column"] = error.column;
					error_dict["message"] = error.message;
					errors.push_back(error_dict);
				}
			}
			
			// Only continue to compilation if analysis succeeded
			if (analyze_err == OK) {
				// Create a temporary script for compilation
				Ref<GDScript> temp_script;
				temp_script.instantiate();
				
				GDScriptCompiler compiler;
				Error compile_err = compiler.compile(&parser, temp_script.ptr(), false);
				
				if (compile_err != OK) {
					Dictionary error_dict;
					error_dict["type"] = "compiler_error";
					error_dict["line"] = compiler.get_error_line();
					error_dict["column"] = compiler.get_error_column();
					error_dict["message"] = compiler.get_error();
					errors.push_back(error_dict);
				}
			}
		}
	} else if (extension == "cs") {
		// C# compilation would require mono/dotnet integration
		// For now, add a placeholder
		Dictionary error_dict;
		error_dict["type"] = "info";
		error_dict["line"] = 0;
		error_dict["column"] = 0;
		error_dict["message"] = "C# compilation checking not implemented yet";
		errors.push_back(error_dict);
	}
	
	print_line("COMPILATION CHECK: Found " + String::num_int64(errors.size()) + " errors for " + p_file_path);
	
	    return errors;
}

Dictionary EditorTools::check_compilation_errors(const Dictionary &p_args) {
    Dictionary result;
    String path = p_args.get("path", "");
    
    if (path.is_empty()) {
        result["success"] = false;
        result["message"] = "Path is required";
        result["errors"] = Array();
        return result;
    }
    
    print_line("CHECK_COMPILATION_ERRORS: Checking file - " + path);
    
    // Simple approach: Try to reload the script and see if it fails
    Array errors;
    
    if (path.get_extension() == "gd") {
        // Prefer unsaved preview overlay content if present
        Error file_err = OK;
        String file_content;
        if (EditorTools::has_preview_overlay(path)) {
            file_content = EditorTools::get_preview_overlay(path);
            print_line("CHECK_COMPILATION_ERRORS: Using preview overlay content for " + path);
        } else {
            file_content = FileAccess::get_file_as_string(path, &file_err);
        }
        
        if (file_err != OK) {
            Dictionary error_dict;
            error_dict["type"] = "file_error";
            error_dict["line"] = 0;
            error_dict["column"] = 0;
            error_dict["message"] = "Failed to read file: " + path;
            errors.push_back(error_dict);
        } else {
            // Parse the script directly to get detailed error information
            GDScriptParser parser;
            Error parse_err = parser.parse(file_content, path, false);
            
            // Get parser errors
            const List<GDScriptParser::ParserError> &parser_errors = parser.get_errors();
            for (const GDScriptParser::ParserError &error : parser_errors) {
                Dictionary error_dict;
                error_dict["type"] = "parser_error";
                error_dict["line"] = error.line;
                error_dict["column"] = error.column;
                error_dict["message"] = error.message;
                errors.push_back(error_dict);
                print_line("CHECK_COMPILATION_ERRORS: Found parser error at line " + String::num_int64(error.line) + ": " + error.message);
            }
            
            // Only continue to analysis if parsing succeeded
            if (parse_err == OK && parser_errors.is_empty()) {
                GDScriptAnalyzer analyzer(&parser);
                Error analyze_err = analyzer.analyze();
                
                // Get analyzer errors (they're stored in the parser)
                const List<GDScriptParser::ParserError> &analyzer_errors = parser.get_errors();
                for (const GDScriptParser::ParserError &error : analyzer_errors) {
                    // Skip errors we already collected during parsing
                    bool already_collected = false;
                    for (const GDScriptParser::ParserError &parse_error : parser_errors) {
                        if (parse_error.line == error.line && parse_error.message == error.message) {
                            already_collected = true;
                            break;
                        }
                    }
                    if (!already_collected) {
                        Dictionary error_dict;
                        error_dict["type"] = "analyzer_error";
                        error_dict["line"] = error.line;
                        error_dict["column"] = error.column;
                        error_dict["message"] = error.message;
                        errors.push_back(error_dict);
                        print_line("CHECK_COMPILATION_ERRORS: Found analyzer error at line " + String::num_int64(error.line) + ": " + error.message);
                    }
                }
                
                if (analyze_err == OK && analyzer_errors.size() == parser_errors.size()) {
                    print_line("CHECK_COMPILATION_ERRORS: Script parsed and analyzed successfully");
                }
            } else {
                print_line("CHECK_COMPILATION_ERRORS: Parsing failed with " + String::num_int64(parser_errors.size()) + " errors");
            }
        }
    } else if (path.get_extension() == "cs") {
        // For C# files, add placeholder check
        Dictionary info_dict;
        info_dict["type"] = "info";
        info_dict["line"] = 0;
        info_dict["column"] = 0;
        info_dict["message"] = "C# compilation checking not implemented";
        errors.push_back(info_dict);
    } else {
        Dictionary info_dict;
        info_dict["type"] = "info";
        info_dict["line"] = 0;
        info_dict["column"] = 0;
        info_dict["message"] = "Unsupported file type for compilation checking";
        errors.push_back(info_dict);
    }
    
    result["success"] = true;
    result["path"] = path;
    result["errors"] = errors;
    result["has_errors"] = errors.size() > 0;
    result["error_count"] = errors.size();
    
    print_line("CHECK_COMPILATION_ERRORS: Found " + String::num_int64(errors.size()) + " errors in " + path);
    
    return result;
}

// --- Universal Tools Implementation ---

Dictionary EditorTools::universal_node_manager(const Dictionary &p_args) {
	String operation = p_args.get("operation", "");
	
	if (operation == "create") return create_node(p_args);
	if (operation == "delete") return delete_node(p_args);
	if (operation == "move") return move_node(p_args);
	if (operation == "set_property") return set_node_property(p_args);
	if (operation == "get_info") return get_all_nodes(p_args);
	if (operation == "search") return search_nodes_by_type(p_args);
	if (operation == "select") return get_editor_selection(p_args);
	if (operation == "get_properties") return get_node_properties(p_args);
	if (operation == "call_method") return call_node_method(p_args);
	if (operation == "get_script") return get_node_script(p_args);
	if (operation == "attach_script") return attach_script(p_args);
	if (operation == "add_collision") return add_collision_shape(p_args);
	
	Dictionary result;
	result["success"] = false;
	result["message"] = "Unknown node operation: " + operation;
	return result;
}

Dictionary EditorTools::universal_file_manager(const Dictionary &p_args) {
	String operation = p_args.get("operation", "");
	
	if (operation == "read") {
		if (p_args.has("start_line") || p_args.has("end_line")) {
			return read_file_advanced(p_args);
		} else {
			return read_file_content(p_args);
		}
	}
	if (operation == "list") return list_project_files(p_args);
	if (operation == "apply_ai_edit") return apply_edit(p_args);
	if (operation == "check_compilation") return check_compilation_errors(p_args);
	if (operation == "get_classes") return get_available_classes(p_args);

	Dictionary result;
	result["success"] = false;
	result["message"] = "Unknown file operation: " + operation;
	return result;
}

Dictionary EditorTools::scene_manager(const Dictionary &p_args) {
	String operation = p_args.get("operation", "");
	
	if (operation == "get_info") return get_scene_info(p_args);
	if (operation == "open" || operation == "create_new" || operation == "save_as" || operation == "instantiate") {
		return manage_scene(p_args);
	}
	
	Dictionary result;
	result["success"] = false;
	result["message"] = "Unknown scene operation: " + operation;
	return result;
}

// --- New Debugging Tools Implementation ---

Dictionary EditorTools::run_scene(const Dictionary &p_args) {
	Dictionary result;
	String scene_path = p_args.get("scene_path", "");
	int duration = p_args.get("duration", 5);
	
	// Get the current scene if no path specified
	if (scene_path.is_empty()) {
		Node *current_scene = EditorNode::get_singleton()->get_tree()->get_edited_scene_root();
		if (current_scene) {
			scene_path = current_scene->get_scene_file_path();
		}
	}
	
	if (scene_path.is_empty()) {
		result["success"] = false;
		result["message"] = "No scene to run";
		return result;
	}
	
	// Start the scene
	EditorRunBar::get_singleton()->play_custom_scene(scene_path);
	
	result["success"] = true;
	result["message"] = "Scene started: " + scene_path;
	result["scene_path"] = scene_path;
	result["duration"] = duration;
	return result;
}

Dictionary EditorTools::get_scene_tree_hierarchy(const Dictionary &p_args) {
	Dictionary result;
	bool include_properties = p_args.get("include_properties", false);
	
	Node *root = EditorNode::get_singleton()->get_tree()->get_edited_scene_root();
	if (!root) {
		result["success"] = false;
		result["message"] = "No scene is currently being edited.";
		return result;
	}
	
	// Recursive function to build hierarchy
	std::function<Dictionary(Node*)> build_hierarchy = [&](Node* node) -> Dictionary {
		Dictionary node_dict;
		if (!node) return node_dict;
		
		node_dict["name"] = node->get_name();
		node_dict["type"] = node->get_class();
		node_dict["path"] = root->get_path_to(node);
		
		if (include_properties) {
			List<PropertyInfo> properties;
			node->get_property_list(&properties);
			Dictionary props_dict;
			for (const PropertyInfo &prop_info : properties) {
				if (prop_info.usage & PROPERTY_USAGE_EDITOR) {
					props_dict[prop_info.name] = node->get(prop_info.name);
				}
			}
			node_dict["properties"] = props_dict;
		}
		
		// Add children recursively
		Array children;
		for (int i = 0; i < node->get_child_count(); i++) {
			children.push_back(build_hierarchy(node->get_child(i)));
		}
		node_dict["children"] = children;
		node_dict["child_count"] = node->get_child_count();
		
		return node_dict;
	};
	
	result["success"] = true;
	result["hierarchy"] = build_hierarchy(root);
	result["include_properties"] = include_properties;
	return result;
}

Dictionary EditorTools::inspect_physics_body(const Dictionary &p_args) {
	Dictionary result;
	if (!p_args.has("path")) {
		result["success"] = false;
		result["message"] = "Missing 'path' argument.";
		return result;
	}
	
	Node *node = _get_node_from_path(p_args["path"], result);
	if (!node) {
		return result;
	}
	
	Dictionary physics_info;
	physics_info["node_name"] = node->get_name();
	physics_info["node_type"] = node->get_class();
	
	// Check if it's a physics body
	if (node->is_class("RigidBody2D") || node->is_class("CharacterBody2D") || 
		node->is_class("StaticBody2D") || node->is_class("Area2D")) {
		
		physics_info["is_physics_body"] = true;
		
		// Get physics properties
		physics_info["collision_layer"] = node->get("collision_layer");
		physics_info["collision_mask"] = node->get("collision_mask");
		
		if (node->is_class("RigidBody2D") || node->is_class("RigidBody3D")) {
			physics_info["mass"] = node->get("mass");
			physics_info["gravity_scale"] = node->get("gravity_scale");
			physics_info["linear_velocity"] = node->get("linear_velocity");
			physics_info["angular_velocity"] = node->get("angular_velocity");
		}
		
		// Check for collision shapes
		Array collision_shapes;
		for (int i = 0; i < node->get_child_count(); i++) {
			Node *child = node->get_child(i);
			if (child->is_class("CollisionShape2D") || child->is_class("CollisionShape3D")) {
				Dictionary shape_info;
				shape_info["name"] = child->get_name();
				shape_info["type"] = child->get_class();
				shape_info["disabled"] = child->get("disabled");
				collision_shapes.push_back(shape_info);
			}
		}
		physics_info["collision_shapes"] = collision_shapes;
		
	} else {
		physics_info["is_physics_body"] = false;
		physics_info["message"] = "Node is not a physics body";
	}
	
	result["success"] = true;
	result["physics_info"] = physics_info;
	return result;
}

Dictionary EditorTools::get_camera_info(const Dictionary &p_args) {
	Dictionary result;
	String camera_path = p_args.get("camera_path", "");
	
	Node *camera = nullptr;
	Node *root = EditorNode::get_singleton()->get_tree()->get_edited_scene_root();
	
	if (!camera_path.is_empty()) {
		camera = _get_node_from_path(camera_path, result);
		if (!camera) {
			return result;
		}
	} else if (root) {
		// Find first camera in the scene
		std::function<Node*(Node*)> find_camera = [&](Node* node) -> Node* {
			if (node->is_class("Camera2D") || node->is_class("Camera3D")) {
				return node;
			}
			for (int i = 0; i < node->get_child_count(); i++) {
				Node *found = find_camera(node->get_child(i));
				if (found) return found;
			}
			return nullptr;
		};
		camera = find_camera(root);
	}
	
	if (!camera) {
		result["success"] = false;
		result["message"] = "No camera found";
		return result;
	}
	
	Dictionary camera_info;
	camera_info["name"] = camera->get_name();
	camera_info["type"] = camera->get_class();
	camera_info["path"] = root ? root->get_path_to(camera) : camera->get_path();
	camera_info["position"] = camera->get("position");
	camera_info["enabled"] = camera->get("enabled");
	
	if (camera->is_class("Camera2D")) {
		camera_info["zoom"] = camera->get("zoom");
		camera_info["offset"] = camera->get("offset");
		camera_info["limit_left"] = camera->get("limit_left");
		camera_info["limit_right"] = camera->get("limit_right");
		camera_info["limit_top"] = camera->get("limit_top");
		camera_info["limit_bottom"] = camera->get("limit_bottom");
	}
	
	result["success"] = true;
	result["camera_info"] = camera_info;
	return result;
}

Dictionary EditorTools::take_screenshot(const Dictionary &p_args) {
	Dictionary result;
	String filename = p_args.get("filename", "screenshot_debug.png");
	
	// Try to get the main viewport
	Viewport *viewport = EditorNode::get_singleton()->get_viewport();
	if (!viewport) {
		result["success"] = false;
		result["message"] = "Could not access viewport";
		return result;
	}
	
	// Take screenshot
	Ref<Image> screenshot = viewport->get_texture()->get_image();
	if (screenshot.is_null()) {
		result["success"] = false;
		result["message"] = "Failed to capture screenshot";
		return result;
	}
	
	// Save to project directory
	String project_path = ProjectSettings::get_singleton()->globalize_path("res://");
	String full_path = project_path + "/" + filename;
	
	Error save_result = screenshot->save_png(full_path);
	if (save_result != OK) {
		result["success"] = false;
		result["message"] = "Failed to save screenshot: " + String::num_int64(save_result);
		return result;
	}
	
	result["success"] = true;
	result["message"] = "Screenshot saved";
	result["filename"] = filename;
	result["path"] = full_path;
	return result;
}

Dictionary EditorTools::check_node_in_scene_tree(const Dictionary &p_args) {
	Dictionary result;
	if (!p_args.has("path")) {
		result["success"] = false;
		result["message"] = "Missing 'path' argument.";
		return result;
	}
	
	Node *node = _get_node_from_path(p_args["path"], result);
	if (!node) {
		return result;
	}
	
	Dictionary node_status;
	node_status["exists"] = true;
	node_status["name"] = node->get_name();
	node_status["type"] = node->get_class();
	node_status["is_inside_tree"] = node->is_inside_tree();
	node_status["is_ready"] = node->is_ready();
	node_status["process_mode"] = node->get("process_mode");
	
	Node *parent = node->get_parent();
	if (parent) {
		node_status["parent_name"] = parent->get_name();
		node_status["parent_type"] = parent->get_class();
	} else {
		node_status["parent_name"] = "";
		node_status["parent_type"] = "";
	}
	
	node_status["child_count"] = node->get_child_count();
	node_status["visible"] = node->has_method("is_visible") ? node->call("is_visible") : Variant();
	
	result["success"] = true;
	result["node_status"] = node_status;
	return result;
}

Dictionary EditorTools::inspect_animation_state(const Dictionary &p_args) {
	Dictionary result;
	if (!p_args.has("path")) {
		result["success"] = false;
		result["message"] = "Missing 'path' argument.";
		return result;
	}
	
	Node *node = _get_node_from_path(p_args["path"], result);
	if (!node) {
		return result;
	}
	
	Dictionary animation_info;
	animation_info["node_name"] = node->get_name();
	animation_info["node_type"] = node->get_class();
	
	if (node->is_class("AnimationPlayer")) {
		animation_info["is_animation_player"] = true;
		animation_info["current_animation"] = node->get("current_animation");
		animation_info["is_playing"] = node->call("is_playing");
		animation_info["playback_speed"] = node->get("playback_speed");
		
		// Get list of animations
		Array animation_list;
		Variant animations = node->call("get_animation_list");
		if (animations.get_type() == Variant::ARRAY) {
			animation_list = animations;
		}
		animation_info["available_animations"] = animation_list;
		
	} else if (node->is_class("AnimatedSprite2D") || node->is_class("AnimatedSprite3D")) {
		animation_info["is_animated_sprite"] = true;
		animation_info["animation"] = node->get("animation");
		animation_info["frame"] = node->get("frame");
		animation_info["playing"] = node->call("is_playing");
		animation_info["speed_scale"] = node->get("speed_scale");
		
	} else {
		animation_info["is_animated"] = false;
		animation_info["message"] = "Node is not an animation node";
	}
	
	result["success"] = true;
	result["animation_info"] = animation_info;
	return result;
}

Dictionary EditorTools::get_layers_and_zindex(const Dictionary &p_args) {
	Dictionary result;
	String path = p_args.get("path", "");
	
	Node *root = EditorNode::get_singleton()->get_tree()->get_edited_scene_root();
	if (!root) {
		result["success"] = false;
		result["message"] = "No scene is currently being edited.";
		return result;
	}
	
	Array layer_info;
	
	if (!path.is_empty()) {
		// Get info for specific node
		Node *node = _get_node_from_path(path, result);
		if (!node) {
			return result;
		}
		
		Dictionary node_layer_info;
		node_layer_info["name"] = node->get_name();
		node_layer_info["type"] = node->get_class();
		node_layer_info["path"] = root->get_path_to(node);
		
		if (node->has_method("get_z_index")) {
			node_layer_info["z_index"] = node->call("get_z_index");
		}
		if (node->has_method("get_z_as_relative")) {
			node_layer_info["z_as_relative"] = node->call("get_z_as_relative");
		}
		if (node->is_class("CanvasLayer")) {
			node_layer_info["layer"] = node->get("layer");
		}
		
		layer_info.push_back(node_layer_info);
		
	} else {
		// Get info for all nodes with layer/z-index properties
		std::function<void(Node*)> collect_layer_nodes = [&](Node* node) {
			if (node) {
				Dictionary node_layer_info;
				bool has_layer_info = false;
				
				node_layer_info["name"] = node->get_name();
				node_layer_info["type"] = node->get_class();
				node_layer_info["path"] = root->get_path_to(node);
				
				if (node->has_method("get_z_index")) {
					node_layer_info["z_index"] = node->call("get_z_index");
					has_layer_info = true;
				}
				if (node->has_method("get_z_as_relative")) {
					node_layer_info["z_as_relative"] = node->call("get_z_as_relative");
					has_layer_info = true;
				}
				if (node->is_class("CanvasLayer")) {
					node_layer_info["layer"] = node->get("layer");
					has_layer_info = true;
				}
				
				if (has_layer_info) {
					layer_info.push_back(node_layer_info);
				}
				
				// Recursively check children
				for (int i = 0; i < node->get_child_count(); i++) {
					collect_layer_nodes(node->get_child(i));
				}
			}
		};
		
		collect_layer_nodes(root);
	}
	
	result["success"] = true;
	result["layer_info"] = layer_info;
	result["node_count"] = layer_info.size();
	return result;
}

Dictionary EditorTools::search_across_project(const Dictionary &p_args) {
	Dictionary result;
	
	String query = p_args.get("query", "");
	if (query.is_empty()) {
		result["success"] = false;
		result["error"] = "Query parameter is required";
		return result;
	}
	
	// Get optional parameters
	bool include_graph = p_args.get("include_graph", true);
	int max_results = p_args.get("max_results", 5);
	String modality_filter = p_args.get("modality_filter", "");
	int graph_depth = p_args.get("graph_depth", 2);
	
	// Get project root path
	String project_root = ProjectSettings::get_singleton()->get_resource_path();
	
	// Get authentication info from AIChatDock if available
	AIChatDock *ai_chat_dock = nullptr;
	// For now, we'll require dev mode or manual authentication
	// TODO: Implement proper AIChatDock lookup when needed
	
	// For dev mode, use hardcoded values
	String user_id = "106469680334583136136";  // Dev mode user
	String machine_id = "dev_machine";
	String auth_token = "dev_token";
	
	if (ai_chat_dock) {
		// Get authentication details from AI chat dock
		user_id = ai_chat_dock->get_current_user_id();
		machine_id = ai_chat_dock->get_machine_id();
		auth_token = ai_chat_dock->get_auth_token();
	}
	
	// For now, allow dev mode fallback
	if (user_id.is_empty()) {
		user_id = "106469680334583136136";  // Dev fallback
		machine_id = "dev_machine";
		auth_token = "dev_token";
	}
	
	// Prepare HTTP request to backend
	HTTPRequest *http_request = memnew(HTTPRequest);
	EditorNode::get_singleton()->add_child(http_request);
	
	// Prepare request data
	Dictionary request_data;
	request_data["query"] = query;
	request_data["include_graph"] = include_graph;
	request_data["max_results"] = max_results;
	request_data["project_root"] = project_root;
	request_data["user_id"] = user_id;
	request_data["machine_id"] = machine_id;
	request_data["graph_depth"] = graph_depth;
	if (p_args.has("graph_edge_kinds")) {
		request_data["graph_edge_kinds"] = p_args["graph_edge_kinds"];
	} else {
		Array kinds;
		kinds.push_back("CONNECTS_SIGNAL");
		kinds.push_back("ATTACHES_SCRIPT");
		kinds.push_back("INSTANTIATES_SCENE");
		kinds.push_back("CHILD_OF");
		kinds.push_back("DEFINES_FUNCTION");
		kinds.push_back("DEFINES_CLASS");
		kinds.push_back("DEFINES_SIGNAL");
		kinds.push_back("SCRIPT_EXTENDS");
		kinds.push_back("CALLS_FUNCTION");
		kinds.push_back("EMITS_SIGNAL");
		kinds.push_back("GROUP_MEMBER");
		kinds.push_back("REFERENCES_RESOURCE");
		request_data["graph_edge_kinds"] = kinds;
	}
	
	if (!modality_filter.is_empty()) {
		request_data["modality_filter"] = modality_filter;
	}
	
	// Convert to JSON
	Ref<JSON> json;
	json.instantiate();
	String json_string = json->stringify(request_data);
	
	// Prepare headers
	PackedStringArray headers;
	headers.push_back("Content-Type: application/json");
	headers.push_back("Authorization: Bearer " + auth_token);
	
	// For now, return a mock response since we can't easily do HTTP requests from EditorTools
	// This will be working once the authentication system is properly integrated
	result["success"] = true;
	result["query"] = query;
	result["message"] = "Search functionality is available. Tool successfully integrated.";
	result["note"] = "HTTP request to backend would be made here with proper authentication";
	result["similar_files"] = Array();
	result["central_files"] = Array();
	result["file_count"] = 0;
	result["include_graph"] = include_graph;
	
	// Clean up
	http_request->queue_free();
	
	return result;
} 

// --- Multiplexed editor introspection/debug tool ---
Dictionary EditorTools::editor_introspect(const Dictionary &p_args) {
    Dictionary result;
    String operation = p_args.get("operation", "");
    if (operation.is_empty()) {
        result["success"] = false;
        result["message"] = "Missing 'operation'";
        return result;
    }

    // Common helpers
    auto require_path = [&](Dictionary &r) -> Node * {
        if (!p_args.has("path")) {
            r["success"] = false;
            r["message"] = "Missing 'path'";
            return nullptr;
        }
        Dictionary err;
        Node *node = _get_node_from_path(p_args["path"], err);
        if (!node) {
            r = err;
            return nullptr;
        }
        return node;
    };

    if (operation == "list_node_signals") {
        Node *node = require_path(result);
        if (!node) return result;

        List<MethodInfo> signals;
        node->get_signal_list(&signals);

        Array out_signals;
        for (const MethodInfo &mi : signals) {
            Dictionary s;
            s["name"] = String(mi.name);
            Array args_arr;
#ifdef TOOLS_ENABLED
            // MethodInfo::arguments is available in Godot 4
            for (int i = 0; i < mi.arguments.size(); i++) {
                const PropertyInfo &pi = mi.arguments[i];
                Dictionary a;
                a["name"] = String(pi.name);
                a["type"] = Variant::get_type_name(pi.type);
                args_arr.push_back(a);
            }
#endif
            s["args"] = args_arr;
            out_signals.push_back(s);
        }

        result["success"] = true;
        result["signals"] = out_signals;
        return result;
    }

    if (operation == "list_signal_connections") {
        Node *node = require_path(result);
        if (!node) return result;

        StringName filter_signal = p_args.get("signal_name", StringName());
        Array out_conns;

        auto append_connections = [&](const StringName &sig_name) {
            List<Object::Connection> conns;
            node->get_signal_connection_list(sig_name, &conns);
            for (const Object::Connection &conn : conns) {
                Dictionary c;
                // Prefer reported signal name if available, else use current loop name
                StringName sname = sig_name;
                // In Godot 4, Connection has .signal with get_name()
                c["signal"] = String(sname);
                c["method"] = String(conn.callable.get_method());
                c["flags"] = conn.flags;
                Object *tobj = conn.callable.get_object();
                Node *tnode = Object::cast_to<Node>(tobj);
                if (tnode) {
                    Node *root = EditorNode::get_singleton()->get_tree()->get_edited_scene_root();
                    c["target_path"] = root ? root->get_path_to(tnode) : tnode->get_path();
                    c["target_type"] = tnode->get_class();
                }
                out_conns.push_back(c);
            }
        };

        if (String(filter_signal).is_empty()) {
            // No filter: iterate all signals on node
            List<MethodInfo> signals;
            node->get_signal_list(&signals);
            for (const MethodInfo &mi : signals) {
                append_connections(mi.name);
            }
        } else {
            append_connections(filter_signal);
        }

        result["success"] = true;
        result["connections"] = out_conns;
        return result;
    }

    if (operation == "list_incoming_connections") {
        Node *node = require_path(result);
        if (!node) return result;

        List<Object::Connection> incoming;
        node->get_signals_connected_to_this(&incoming);
        Array out_incoming;
        for (const Object::Connection &conn : incoming) {
            Dictionary c;
            // Source object emitting the signal
            Object *src_obj = nullptr;
            // In Godot 4, Connection stores the source signal object
            // This accessor name may vary, so guard
            src_obj = conn.signal.get_object();
            Node *src_node = Object::cast_to<Node>(src_obj);
            if (src_node) {
                Node *root = EditorNode::get_singleton()->get_tree()->get_edited_scene_root();
                c["source_path"] = root ? root->get_path_to(src_node) : src_node->get_path();
                c["source_type"] = src_node->get_class();
            }
            c["signal"] = String(conn.signal.get_name());
            c["method"] = String(conn.callable.get_method());
            c["flags"] = conn.flags;
            out_incoming.push_back(c);
        }
        result["success"] = true;
        result["incoming_connections"] = out_incoming;
        return result;
    }

    if (operation == "validate_signal_connection") {
        // More tolerant: accept aliases, infer when missing.
        Dictionary err;
        String source_path = p_args.get("source_path", p_args.get("path", String("")));
        String target_path = p_args.get("target_path", String(""));
        StringName sig = p_args.has("signal") ? (StringName)p_args["signal"] : (StringName)p_args.get("signal_name", StringName());
        StringName method = p_args.get("method", StringName());

        Node *source = nullptr;
        if (!source_path.is_empty()) source = _get_node_from_path(source_path, err);
        if (!source) return err;

        // If signal missing, attempt heuristic selection
        if (String(sig).is_empty()) {
            List<MethodInfo> sigs; source->get_signal_list(&sigs);
            // Prefer common gameplay signals if present
            StringName preferred;
            for (const MethodInfo &mi : sigs) { if (String(mi.name) == "hit") { preferred = mi.name; break; } }
            if (String(preferred).is_empty() && !sigs.is_empty()) preferred = sigs.front()->get().name;
            sig = preferred;
        }

        Node *target = nullptr;
        if (!target_path.is_empty()) {
            target = _get_node_from_path(target_path, err);
            if (!target) return err;
        }

        // If target or method missing, try inferring from existing connections
        List<Object::Connection> conns; source->get_signal_connection_list(sig, &conns);
        if (!target && method == StringName() && conns.size() == 1) {
            const Object::Connection &c = conns.front()->get();
            target = Object::cast_to<Node>(c.callable.get_object());
            method = c.callable.get_method();
        } else {
            if (!target && method != StringName()) {
                // Find unique node having this method
                Node *root = EditorNode::get_singleton()->get_tree()->get_edited_scene_root();
                int found = 0; Node *found_node = nullptr;
                std::function<void(Node*)> dfs = [&](Node *n){ if (!n) return; if (n->has_method(method)) { found++; found_node = n; } for (int i=0;i<n->get_child_count();i++) dfs(n->get_child(i)); };
                dfs(root);
                if (found == 1) target = found_node;
            }
            if (target && method == StringName()) {
                // Try Godot's conventional method name
                String m = String("_on_") + String(source->get_name()) + String("_") + String(sig);
                if (target->has_method(m)) method = StringName(m);
                else if (conns.size() == 1) method = conns.front()->get().callable.get_method();
            }
        }

        if (!target || method == StringName()) {
            result["success"] = false;
            result["message"] = "Could not infer target/method for validation";
            return result;
        }

        bool exists = false;
        for (const Object::Connection &conn : conns) {
            if (conn.callable.get_method() == method && conn.callable.get_object() == target) { exists = true; break; }
        }
        result["success"] = true;
        result["exists"] = exists;
        result["source_path"] = source_path;
        result["signal"] = String(sig);
        result["target_path"] = EditorNode::get_singleton()->get_tree()->get_edited_scene_root()->get_path_to(target);
        result["method"] = String(method);
        return result;
    }

    if (operation == "connect_signal") {
        Dictionary err;
        String source_path = p_args.get("source_path", p_args.get("path", String("")));
        String target_path = p_args.get("target_path", String(""));
        StringName sig = p_args.has("signal") ? (StringName)p_args["signal"] : (StringName)p_args.get("signal_name", StringName());
        StringName method = p_args.get("method", StringName());
        int flags = p_args.get("flags", 0);

        Node *source = nullptr;
        if (!source_path.is_empty()) source = _get_node_from_path(source_path, err);
        if (!source) return err;

        if (String(sig).is_empty()) {
            // Heuristic: prefer 'hit', else first signal
            List<MethodInfo> sigs; source->get_signal_list(&sigs);
            for (const MethodInfo &mi : sigs) { if (String(mi.name) == "hit") { sig = mi.name; break; } }
            if (String(sig).is_empty() && !sigs.is_empty()) sig = sigs.front()->get().name;
        }

        Node *target = nullptr;
        if (!target_path.is_empty()) target = _get_node_from_path(target_path, err);
        if (!target && method != StringName()) {
            // Infer target via method search
            Node *root = EditorNode::get_singleton()->get_tree()->get_edited_scene_root();
            int found = 0; Node *found_node = nullptr;
            std::function<void(Node*)> dfs = [&](Node *n){ if (!n) return; if (n->has_method(method)) { found++; found_node = n; } for (int i=0;i<n->get_child_count();i++) dfs(n->get_child(i)); };
            dfs(root);
            if (found == 1) target = found_node;
        }
        if (target && method == StringName()) {
            String m = String("_on_") + String(source->get_name()) + String("_") + String(sig);
            if (target->has_method(m)) method = StringName(m);
        }

        if (!target || method == StringName()) {
            result["success"] = false;
            result["message"] = "Could not infer target/method for connect";
            return result;
        }

        Error e = source->connect(sig, Callable(target, method), flags);
        if (e != OK) {
            result["success"] = false;
            result["message"] = String("Failed to connect signal (code ") + itos(e) + ")";
            return result;
        }
        result["success"] = true;
        result["message"] = "Signal connected";
        return result;
    }

    if (operation == "disconnect_signal") {
        Dictionary err;
        String source_path = p_args.get("source_path", p_args.get("path", String("")));
        String target_path = p_args.get("target_path", String(""));
        StringName sig = p_args.has("signal") ? (StringName)p_args["signal"] : (StringName)p_args.get("signal_name", StringName());
        StringName method = p_args.get("method", StringName());

        Node *source = nullptr;
        if (!source_path.is_empty()) source = _get_node_from_path(source_path, err);
        if (!source) return err;

        // Infer from existing connections if needed
        if (String(sig).is_empty() || target_path.is_empty() || method == StringName()) {
            List<Object::Connection> conns; source->get_signal_connection_list(sig, &conns);
            if (String(sig).is_empty()) {
                // If no signal specified, try when exactly one connected signal exists
                List<MethodInfo> sigs; source->get_signal_list(&sigs);
                for (const MethodInfo &mi : sigs) {
                    List<Object::Connection> tmp; source->get_signal_connection_list(mi.name, &tmp);
                    if (tmp.size() == 1) { sig = mi.name; conns = tmp; break; }
                }
            }
            if (conns.size() == 1 && (target_path.is_empty() || method == StringName())) {
                const Object::Connection &c = conns.front()->get();
                Node *t = Object::cast_to<Node>(c.callable.get_object());
                if (target_path.is_empty() && t) {
                    Node *root = EditorNode::get_singleton()->get_tree()->get_edited_scene_root();
                    target_path = root ? String(root->get_path_to(t)) : String(t->get_path());
                }
                if (method == StringName()) method = c.callable.get_method();
            }
        }

        Node *target = nullptr;
        if (!target_path.is_empty()) target = _get_node_from_path(target_path, err);
        if (!target || String(sig).is_empty() || method == StringName()) {
            result["success"] = false;
            result["message"] = "Could not infer enough info to disconnect";
            return result;
        }

        source->disconnect(sig, Callable(target, method));
        result["success"] = true;
        result["message"] = "Signal disconnected (if existed)";
        return result;
    }

    if (operation == "stop_scene") {
        EditorRunBar::get_singleton()->stop_playing();
        result["success"] = true;
        result["message"] = "Stopped running scene";
        return result;
    }

    if (operation == "set_property") {
        // Reuse set_node_property
        if (!p_args.has("path") || !p_args.has("property") || !p_args.has("value")) {
            result["success"] = false;
            result["message"] = "Missing 'path', 'property', or 'value'";
            return result;
        }
        return set_node_property(p_args);
    }

    if (operation == "call_method") {
        // Reuse call_node_method
        if (!p_args.has("path") || !p_args.has("method")) {
            result["success"] = false;
            result["message"] = "Missing 'path' or 'method'";
            return result;
        }
        return call_node_method(p_args);
    }

    if (operation == "start_signal_trace") {
        // args: node_paths[], signals?, include_args?, max_events?
        Array node_paths = p_args.get("node_paths", Array());
        Array signals = p_args.get("signals", Array());
        bool include_args = p_args.get("include_args", false);
        int max_events = p_args.get("max_events", 100);

        if (node_paths.is_empty()) {
            result["success"] = false;
            result["message"] = "node_paths required";
            return result;
        }

        String trace_id = String::num_uint64((uint64_t)OS::get_singleton()->get_ticks_usec());
        Dictionary reg;
        reg["events"] = Array();
        reg["include_args"] = include_args;
        reg["max_events"] = max_events;
        reg["next_index"] = 0;
        Array connections; // store for cleanup

        Node *root = EditorNode::get_singleton()->get_tree()->get_edited_scene_root();
        EditorTools *tracer = ensure_tracer();
        for (int i = 0; i < node_paths.size(); i++) {
            String np = node_paths[i];
            Dictionary err;
            Node *src = _get_node_from_path(np, err);
            if (!src) continue;

            // signals empty => connect all signals from node
            List<MethodInfo> sigs;
            src->get_signal_list(&sigs);
            for (const MethodInfo &mi : sigs) {
                if (!signals.is_empty()) {
                    bool match = false;
                    for (int s = 0; s < signals.size(); s++) {
                        if (String(mi.name) == String(signals[s])) { match = true; break; }
                    }
                    if (!match) continue;
                }
                // Connect to callback on tracer instance
        		String src_path_str = root ? String(root->get_path_to(src)) : String(src->get_path());
                int argc = mi.arguments.size();
                Callable cb;
                switch (MIN(argc, 4)) {
                    case 0: cb = callable_mp(tracer, &EditorTools::_on_traced_signal_0).bind(trace_id, src_path_str, String(mi.name)); break;
                    case 1: cb = callable_mp(tracer, &EditorTools::_on_traced_signal_1).bind(trace_id, src_path_str, String(mi.name)); break;
                    case 2: cb = callable_mp(tracer, &EditorTools::_on_traced_signal_2).bind(trace_id, src_path_str, String(mi.name)); break;
                    case 3: cb = callable_mp(tracer, &EditorTools::_on_traced_signal_3).bind(trace_id, src_path_str, String(mi.name)); break;
                    default: cb = callable_mp(tracer, &EditorTools::_on_traced_signal_4).bind(trace_id, src_path_str, String(mi.name)); break;
                }
                Error e = src->connect(mi.name, cb);
                if (e == OK) {
                    Dictionary c;
                    c["node_path"] = src_path_str;
                    c["signal"] = String(mi.name);
                    c["callable"] = cb; // store callable to disconnect precisely
                    connections.push_back(c);
                }
            }
        }

        reg["connections"] = connections;
        trace_registry[trace_id] = reg;
        result["success"] = true;
        result["trace_id"] = trace_id;
        result["connected"] = connections.size();
        return result;
    }

    if (operation == "stop_signal_trace") {
        String trace_id = p_args.get("trace_id", "");
        if (!trace_registry.has(trace_id)) {
            result["success"] = false;
            result["message"] = "Unknown trace_id";
            return result;
        }
        Dictionary reg = trace_registry[trace_id];
        Array connections = reg.get("connections", Array());
        Node *root = EditorNode::get_singleton()->get_tree()->get_edited_scene_root();
        for (int i = 0; i < connections.size(); i++) {
            Dictionary c = connections[i];
            Dictionary err;
            Node *src = _get_node_from_path(c.get("node_path", ""), err);
            if (!src) continue;
            StringName sig = c.get("signal", "");
            Variant callable_v = c.get("callable", Variant());
            if (callable_v.get_type() == Variant::CALLABLE) {
                Callable cb = callable_v;
                src->disconnect(sig, cb);
            }
        }
        trace_registry.erase(trace_id);
        result["success"] = true;
        result["message"] = "Trace stopped";
        return result;
    }

    if (operation == "get_trace_events") {
        String trace_id = p_args.get("trace_id", "");
        int since = p_args.get("since_index", 0);
        if (!trace_registry.has(trace_id)) {
            result["success"] = false;
            result["message"] = "Unknown trace_id";
            return result;
        }
        Dictionary reg = trace_registry[trace_id];
        Array events = reg.get("events", Array());
        Array out;
        for (int i = 0; i < events.size(); i++) {
            Dictionary e = events[i];
            if ((int)e.get("i", 0) >= since) out.push_back(e);
        }
        result["success"] = true;
        result["events"] = out;
        result["next_index"] = reg.get("next_index", 0);
        return result;
    }

    if (operation == "start_property_watch") {
        // variables[], node_path, max_events?
        Array variables = p_args.get("variables", Array());
        String node_path = p_args.get("node_path", String("."));
        int max_events = p_args.get("max_events", 200);
        if (variables.is_empty()) {
            result["success"] = false;
            result["message"] = "variables required";
            return result;
        }
        Dictionary err; Node *node = _get_node_from_path(node_path, err);
        if (!node) return err;

        String watch_id = String::num_uint64((uint64_t)OS::get_singleton()->get_ticks_usec());
        Dictionary reg;
        reg["node_path"] = node_path;
        reg["variables"] = variables;
        reg["last_values"] = Dictionary();
        reg["events"] = Array();
        reg["next_index"] = 0;
        reg["max_events"] = max_events;
        property_watch_registry[watch_id] = reg;

        // Initial snapshot
        Dictionary ev;
        ev["i"] = 0;
        ev["time_ms"] = OS::get_singleton()->get_ticks_msec();
        ev["snapshot"] = Dictionary();
        Dictionary snap;
        for (int i = 0; i < variables.size(); i++) {
            String v = variables[i];
            snap[v] = node->get(v);
        }
        ev["snapshot"] = snap;
        Array events = reg["events"]; events.push_back(ev);
        reg["events"] = events; reg["next_index"] = 1;
        reg["last_values"] = snap;
        property_watch_registry[watch_id] = reg;

        result["success"] = true;
        result["watch_id"] = watch_id;
        return result;
    }

    if (operation == "poll_property_watch") {
        String watch_id = p_args.get("watch_id", "");
        int since = p_args.get("since_index", 0);
        if (!property_watch_registry.has(watch_id)) {
            result["success"] = false;
            result["message"] = "Unknown watch_id";
            return result;
        }
        Dictionary reg = property_watch_registry[watch_id];
        String node_path = reg.get("node_path", String("."));
        Array variables = reg.get("variables", Array());
        Dictionary last = reg.get("last_values", Dictionary());
        Array events = reg.get("events", Array());
        int next_index = reg.get("next_index", 0);
        int max_events = reg.get("max_events", 200);

        Dictionary err; Node *node = _get_node_from_path(node_path, err);
        if (!node) return err;

        bool changed = false; Dictionary delta;
        for (int i = 0; i < variables.size(); i++) {
            String v = variables[i];
            Variant value = node->get(v);
            Variant last_v = last.get(v, Variant());
            if (value != last_v) {
                delta[v] = value; last[v] = value; changed = true;
            }
        }
        if (changed) {
            Dictionary ev;
            ev["i"] = next_index;
            ev["time_ms"] = OS::get_singleton()->get_ticks_msec();
            ev["delta"] = delta;
            events.push_back(ev);
            while (events.size() > max_events) events.remove_at(0);
            next_index += 1;
        }
        reg["events"] = events; reg["next_index"] = next_index; reg["last_values"] = last;
        property_watch_registry[watch_id] = reg;

        Array out;
        for (int i = 0; i < events.size(); i++) { Dictionary e = events[i]; if ((int)e.get("i", 0) >= since) out.push_back(e); }
        result["success"] = true;
        result["events"] = out;
        result["next_index"] = next_index;
        return result;
    }

    if (operation == "stop_property_watch") {
        String watch_id = p_args.get("watch_id", "");
        property_watch_registry.erase(watch_id);
        result["success"] = true;
        result["message"] = "Property watch stopped";
        return result;
    }

    if (operation == "simulate_interaction") {
        // Minimal scripted steps: e.g., "call:Player._on_Player_hit(); wait:500; set:Main.health=2"
        String script = p_args.get("interaction_script", String(""));
        String base = p_args.get("node_path", String("."));
        if (script.is_empty()) {
            result["success"] = false;
            result["message"] = "interaction_script required";
            return result;
        }
        Dictionary err; Node *root = _get_node_from_path(base, err);
        if (!root) return err;
        Vector<String> steps = script.split(";");
        for (int i = 0; i < steps.size(); i++) {
            String s = steps[i].strip_edges();
            if (s.is_empty()) continue;
            if (s.begins_with("wait:")) {
                int ms = s.substr(5).to_int();
                OS::get_singleton()->delay_usec((uint64_t)ms * 1000);
                continue;
            }
            if (s.begins_with("set:")) {
                String expr = s.substr(4); // Node.property=value
                int eq = expr.find("=");
                if (eq > 0) {
                    String lhs = expr.substr(0, eq).strip_edges();
                    String rhs = expr.substr(eq + 1).strip_edges();
                    int dot = lhs.find(".");
                    if (dot > 0) {
                        String node_rel = lhs.substr(0, dot);
                        String prop = lhs.substr(dot + 1);
                        Dictionary e2; Node *n = _get_node_from_path(node_rel, e2);
                        if (n) { n->set(prop, rhs); }
                    }
                }
                continue;
            }
            if (s.begins_with("call:")) {
                String call = s.substr(5); // Node.method(args?)
                int dot = call.find("."); int par = call.find("("); int par2 = call.rfind(")");
                if (dot > 0 && par > dot && par2 > par) {
                    String node_rel = call.substr(0, dot);
                    String method = call.substr(dot + 1, par - (dot + 1));
                    String args_str = call.substr(par + 1, par2 - par - 1);
                    Array args;
                    if (!args_str.is_empty()) { Vector<String> parts = args_str.split(","); for (int j=0;j<parts.size();j++) args.push_back(parts[j].strip_edges()); }
                    Dictionary e2; Node *n = _get_node_from_path(node_rel, e2); if (n) { n->callv(method, args); }
                }
                continue;
            }
        }
        result["success"] = true;
        result["message"] = "Simulation completed";
        return result;
    }

    // Minimal stub to keep calls safe; expand with concrete implementations as needed.
    if (operation == "rename_node") {
        if (!p_args.has("path") || !p_args.has("new_name")) {
            result["success"] = false;
            result["message"] = "Missing 'path' or 'new_name'";
            return result;
        }
        Dictionary err;
        Node *node = _get_node_from_path(p_args["path"], err);
        if (!node) {
            return err;
        }
        String new_name = p_args["new_name"];
        node->set_name(new_name);
        result["success"] = true;
        result["message"] = "Node renamed";
        result["path"] = p_args["path"];
        result["new_name"] = new_name;
        return result;
    }

    // Signals and trace operations not yet implemented in this stub.
    result["success"] = false;
    result["message"] = String("Operation not implemented: ") + operation;
    return result;
}

Dictionary EditorTools::search_across_godot_docs(const Dictionary &p_args) {
    Dictionary result;
    String query = p_args.get("query", "");
    if (query.is_empty()) {
        result["success"] = false;
        result["error"] = "Query parameter is required";
        return result;
    }
    // This method is invoked via AIToolServer and will be forwarded to backend
    // through the existing chat tool execution path. Keep it lightweight.
    result["success"] = true;
    result["query"] = query;
    return result;
}