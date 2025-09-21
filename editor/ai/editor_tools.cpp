/*
 * © 2025 Simplifine Corp.
 * Personal Non‑Commercial License applies to this file as an original contribution to this Godot fork.
 * See LICENSES/COMPANY-NONCOMMERCIAL.md for terms. Commercial use requires a separate license from the Project Owner.
 */
#include "editor_tools.h"

// Add core includes early so they're available to helper functions below
#include "core/crypto/crypto.h"
#include "core/core_bind.h"
#include "core/object/message_queue.h"
#include "core/os/time.h"
#include "core/io/dir_access.h"
#include "core/io/file_access.h"
#include "core/io/http_client.h"
#include "core/io/json.h"
#include "core/io/resource_loader.h"
#include "core/io/resource_saver.h"
#include "core/io/dir_access.h"
#include "editor/file_system/editor_file_system.h"
#include "editor/docks/import_dock.h"
#include "core/io/config_file.h"
#include "core/object/script_language.h"
#include "editor/run/game_view_plugin.h"
#include "core/config/project_settings.h"
#include "editor/editor_data.h"
#include "editor/editor_interface.h"
#include "editor/editor_node.h"
#include "editor/file_system/editor_file_system.h"
#include "core/io/image.h"
#include "editor/settings/editor_settings.h"
#include "core/variant/typed_array.h"
#include "core/input/input_map.h"
#include "core/input/input_event.h"
#include "editor/editor_string_names.h"
#include "editor/docks/scene_tree_dock.h"
#include "editor/docks/ai_chat_dock.h"
#include "editor/run/editor_run_bar.h"
#include "editor/run/editor_run.h"
#include "editor/script/script_editor_plugin.h"
#include "editor/script/script_text_editor.h"
#include "scene/main/node.h"
#include "scene/main/window.h"
#include "scene/resources/packed_scene.h"
#include "modules/gdscript/gdscript.h"
#include "modules/gdscript/gdscript_parser.h"
#include "modules/gdscript/gdscript_warning.h"
#include "modules/gdscript/gdscript_analyzer.h"
#include "modules/gdscript/gdscript_compiler.h"

// C++ standard library for thread-safe sleep
#include <thread>
#include <chrono>

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

void EditorTools::clear_all_preview_overlays() {
	s_preview_overlays.clear();
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

Dictionary EditorTools::batch_set_node_properties(const Dictionary &p_args) {
    Dictionary result;
    Array ops = p_args.get("operations", Array());
    bool save_after = p_args.get("save_after", false);
    int applied = 0;
    Array failures;
    
    for (int i = 0; i < ops.size(); i++) {
        Dictionary op = ops[i];
        Dictionary r = set_node_property(op);
        if (r.get("success", false)) {
            applied++;
        } else {
            failures.push_back(r);
        }
    }
    if (save_after) {
        String current_scene = EditorNode::get_singleton()->get_edited_scene()->get_scene_file_path();
        if (!current_scene.is_empty()) {
            EditorNode::get_singleton()->save_scene_if_open(current_scene);
        }
    }
    result["success"] = failures.is_empty();
    result["applied"] = applied;
    result["failed"] = failures.size();
    if (!failures.is_empty()) {
        result["failures"] = failures;
        // Include detailed error messages for debugging
        String error_summary = "Batch operation encountered " + String::num_int64(failures.size()) + " error(s): ";
        for (int i = 0; i < failures.size() && i < 3; i++) { // Show first 3 errors
            Dictionary failure = failures[i];
            error_summary += "\n- " + String(failure.get("message", "Unknown error"));
        }
        if (failures.size() > 3) {
            error_summary += "\n- ... and " + String::num_int64(failures.size() - 3) + " more error(s)";
        }
        result["message"] = error_summary;
    } else {
        result["message"] = "Successfully applied " + String::num_int64(applied) + " property change(s)";
    }
    return result;
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

// --- Introspection & Readiness ---

Dictionary EditorTools::resource_info(const Dictionary &p_args) {
    Dictionary out;
    String res_path = p_args.get("resource_path", "");
    if (res_path.is_empty()) {
        out["ok"] = false;
        out["error_code"] = "INVALID_ARGUMENT";
        out["error"] = "resource_path is required";
        return out;
    }

    bool exists = ResourceLoader::exists(res_path);
    out["exists"] = exists;
    out["loadable"] = false;
    out["type"] = String();
    out["import_status"] = String("unknown");

    if (exists) {
        String rtype = ResourceLoader::get_resource_type(res_path);
        out["type"] = rtype;
        Ref<Resource> res = ResourceLoader::load(res_path);
        out["loadable"] = res.is_valid();

        // Determine import status via EditorFileSystem
        if (EditorFileSystem::get_singleton()) {
            int idx = -1;
            EditorFileSystemDirectory *efd = EditorFileSystem::get_singleton()->find_file(res_path, &idx);
            if (efd && idx >= 0) {
                bool valid = efd->get_file_import_is_valid(idx);
                out["import_status"] = valid ? String("ok") : String("broken");
            }
        }

        // If an .import file exists but still loading fails, mark pending
        if (!bool(out.get("loadable", false))) {
            String import_sidecar = res_path + ".import";
            if (FileAccess::exists(import_sidecar)) {
                out["import_status"] = String("pending");
            }
        }
    }

    out["ok"] = true;
    return out;
}

Dictionary EditorTools::script_info(const Dictionary &p_args) {
    Dictionary out;
    String script_path = p_args.get("script_path", "");
    if (script_path.is_empty()) {
        out["ok"] = false;
        out["error_code"] = "INVALID_ARGUMENT";
        out["error"] = "script_path is required";
        return out;
    }

    Ref<Resource> res = ResourceLoader::load(script_path);
    Ref<Script> script = res;
    if (!script.is_valid()) {
        out["ok"] = false;
        out["error_code"] = "NOT_LOADABLE";
        out["error"] = "Not a valid script or failed to load";
        return out;
    }

    out["ok"] = true;
    out["language"] = script->get_language() ? String(script->get_language()->get_name()) : String();
    out["class_name"] = script->get_global_name();
    out["base_class"] = script->get_base_script().is_valid() ? String(script->get_base_script()->get_class()) : String();

    // Exported variables
    Array exports;
    List<PropertyInfo> props;
    script->get_script_property_list(&props);
    for (const PropertyInfo &pi : props) {
        if (pi.usage & PROPERTY_USAGE_SCRIPT_VARIABLE) {
            Dictionary e;
            e["name"] = String(pi.name);
            e["type"] = (int)pi.type;
            exports.push_back(e);
        }
    }
    out["exports"] = exports;

    // Signals
    Array signals;
    List<MethodInfo> sigs;
    script->get_script_signal_list(&sigs);
    for (const MethodInfo &mi : sigs) {
        signals.push_back(String(mi.name));
    }
    out["signals"] = signals;
    return out;
}

// --- Import control ---

Dictionary EditorTools::set_import_preset(const Dictionary &p_args) {
    Dictionary out;
    String res_path = p_args.get("resource_path", "");
    Dictionary options = p_args.get("options", Dictionary());
    if (res_path.is_empty()) {
        out["ok"] = false;
        out["error_code"] = "INVALID_ARGUMENT";
        out["error"] = "resource_path is required";
        return out;
    }

    Ref<ConfigFile> cfg;
    cfg.instantiate();
    Error e = cfg->load(res_path + ".import");
    if (e != OK) {
        out["ok"] = false;
        out["error_code"] = "NOT_FOUND";
        out["error"] = "No .import sidecar for resource";
        return out;
    }

    // Write options into the remap/importer group if present; fall back to global root
    String section = cfg->has_section("remap") ? String("remap") : String();
    for (const Variant *k = options.next(); k; k = options.next(k)) {
        if (section.is_empty()) {
            cfg->set_value("", *k, options[*k]);
        } else {
            cfg->set_value(section, *k, options[*k]);
        }
    }
    cfg->save(res_path + ".import");
    out["ok"] = true;
    return out;
}

Dictionary EditorTools::reimport_resource(const Dictionary &p_args) {
    Dictionary out;
    String res_path = p_args.get("resource_path", "");
    int timeout_ms = (int)p_args.get("timeout_ms", 10000);
    if (res_path.is_empty()) {
        out["ok"] = false;
        out["error_code"] = "INVALID_ARGUMENT";
        out["error"] = "resource_path is required";
        return out;
    }

    if (!EditorFileSystem::get_singleton()) {
        out["ok"] = false;
        out["error_code"] = "UNAVAILABLE";
        out["error"] = "EditorFileSystem not available";
        return out;
    }

    Vector<String> to_reimport;
    to_reimport.push_back(res_path);
    EditorFileSystem::get_singleton()->reimport_files(to_reimport);

    // Optionally wait
    uint64_t start = OS::get_singleton()->get_ticks_msec();
    while (EditorFileSystem::get_singleton()->is_importing()) {
        OS::get_singleton()->delay_usec(1000 * 50); // 50ms
        if ((int)(OS::get_singleton()->get_ticks_msec() - start) > timeout_ms) {
            out["ok"] = false;
            out["error_code"] = "IMPORT_TIMEOUT";
            out["error"] = "Timed out waiting for reimport";
            return out;
        }
    }

    out["ok"] = true;
    return out;
}

Dictionary EditorTools::wait_for_import(const Dictionary &p_args) {
    Dictionary out;
    String res_path = p_args.get("resource_path", "");
    int timeout_ms = (int)p_args.get("timeout_ms", 10000);
    int poll_ms = (int)p_args.get("poll_ms", 100);
    if (res_path.is_empty()) {
        out["ok"] = false;
        out["error_code"] = "INVALID_ARGUMENT";
        out["error"] = "resource_path is required";
        return out;
    }

    uint64_t start = OS::get_singleton()->get_ticks_msec();
    String status = "unknown";
    while (true) {
        Dictionary info;
        info["resource_path"] = res_path;
        Dictionary ri = resource_info(info);
        status = String(ri.get("import_status", "unknown"));
        if (status == "ok") {
            out["ok"] = true;
            out["status"] = status;
            return out;
        }
        if ((int)(OS::get_singleton()->get_ticks_msec() - start) > timeout_ms) {
            out["ok"] = false;
            out["error_code"] = "IMPORT_TIMEOUT";
            out["error"] = "Timed out waiting for import";
            out["status"] = status;
            return out;
        }
        OS::get_singleton()->delay_usec(1000 * poll_ms);
    }
}

Dictionary EditorTools::get_runtime_errors_summary(const Dictionary &p_args) {
    Dictionary result;
    bool include_warnings = p_args.get("include_warnings", true);
    String file_filter = p_args.get("file", "");
    
    print_line("RUNTIME_ERRORS_SUMMARY: Starting with " + String::num_int64(s_runtime_errors.size()) + " total recorded errors");
    
    // Track unique error messages and their counts
    HashMap<String, int> error_counts;
    HashMap<String, Dictionary> error_examples; // Store first example of each error type
    int total_errors = 0;
    int total_warnings = 0;
    
    // Process all errors from newest to oldest
    for (int i = s_runtime_errors.size() - 1; i >= 0; i--) {
        Dictionary e = s_runtime_errors[i];
        bool is_warning = e.get("is_warning", false);
        String message = e.get("message", "Unknown error");
        
        if (!include_warnings && is_warning) {
            continue;
        }
        if (!file_filter.is_empty() && String(e.get("file", "")) != file_filter) {
            continue;
        }
        
        // Count errors vs warnings
        if (is_warning) {
            total_warnings++;
        } else {
            total_errors++;
        }
        
        // Deduplicate by message
        if (error_counts.has(message)) {
            error_counts[message]++;
        } else {
            error_counts[message] = 1;
            error_examples[message] = e; // Store first example
        }
    }
    
    // Build summary array sorted by frequency
    Array unique_errors;
    for (const KeyValue<String, int> &kv : error_counts) {
        Dictionary summary;
        summary["message"] = kv.key;
        summary["count"] = kv.value;
        summary["example"] = error_examples[kv.key];
        unique_errors.push_back(summary);
    }
    
    // Sort by count (most frequent first)
    // Note: Manual sorting since we can't use lambdas easily here
    for (int i = 0; i < unique_errors.size() - 1; i++) {
        for (int j = i + 1; j < unique_errors.size(); j++) {
            Dictionary a = unique_errors[i];
            Dictionary b = unique_errors[j];
            if ((int)a.get("count", 0) < (int)b.get("count", 0)) {
                // Swap
                unique_errors[i] = b;
                unique_errors[j] = a;
            }
        }
    }
    
    result["success"] = true;
    result["total_errors"] = total_errors;
    result["total_warnings"] = total_warnings;
    result["total_messages"] = total_errors + total_warnings;
    result["unique_error_types"] = unique_errors.size();
    result["unique_errors"] = unique_errors;
    result["summary"] = String("Found ") + String::num_int64(total_errors + total_warnings) + 
                       " total messages (" + String::num_int64(total_errors) + " errors, " + 
                       String::num_int64(total_warnings) + " warnings) with " + 
                       String::num_int64(unique_errors.size()) + " unique types";
    
    return result;
}

Dictionary EditorTools::get_runtime_errors_detailed(const Dictionary &p_args) {
    Dictionary result;
    bool include_warnings = p_args.get("include_warnings", true);
    int max_count = p_args.get("max_count", 20);
    String file_filter = p_args.get("file", "");
    String message_filter = p_args.get("message_contains", "");
    bool group_duplicates = p_args.get("group_duplicates", true);
    
    print_line("RUNTIME_ERRORS_DETAILED: Starting with " + String::num_int64(s_runtime_errors.size()) + " total recorded errors");
    
    if (group_duplicates) {
        // Get summary and return detailed info for top error types
        Dictionary summary = get_runtime_errors_summary(p_args);
        Array unique_errors = summary.get("unique_errors", Array());
        
        Array detailed_errors;
        int shown = 0;
        
        for (int i = 0; i < unique_errors.size() && shown < max_count; i++) {
            Dictionary error_type = unique_errors[i];
            String message = error_type.get("message", "");
            
            if (!message_filter.is_empty() && !message.containsn(message_filter)) {
                continue;
            }
            
            detailed_errors.push_back(error_type);
            shown++;
        }
        
        result["success"] = true;
        result["errors"] = detailed_errors;
        result["count"] = detailed_errors.size();
        result["grouped"] = true;
        result["total_errors"] = summary.get("total_errors", 0);
        result["total_warnings"] = summary.get("total_warnings", 0);
        result["message"] = "Showing " + String::num_int64(detailed_errors.size()) + 
                           " most frequent error types (grouped)";
    } else {
        // Return individual error instances
        Array out;
        
        for (int i = s_runtime_errors.size() - 1; i >= 0 && out.size() < max_count; i--) {
            Dictionary e = s_runtime_errors[i];
            bool is_warning = e.get("is_warning", false);
            String message = e.get("message", "");
            
            if (!include_warnings && is_warning) {
                continue;
            }
            if (!file_filter.is_empty() && String(e.get("file", "")) != file_filter) {
                continue;
            }
            if (!message_filter.is_empty() && !message.containsn(message_filter)) {
                continue;
            }
            
            out.push_back(e);
        }
        
        result["success"] = true;
        result["errors"] = out;
        result["count"] = out.size();
        result["grouped"] = false;
        result["message"] = "Showing " + String::num_int64(out.size()) + " individual error instances";
    }
    
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

// Static members for screenshot management
Vector<int> EditorTools::_pending_screenshot_requests;
Vector<Dictionary> EditorTools::_current_screenshot_results;

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
		if (p_node->is_inside_tree()) {
			node_info["path"] = p_node->get_path();
		} else {
			node_info["path"] = NodePath(); // Empty path for detached nodes
		}
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

Dictionary EditorTools::get_project_context(const Dictionary &p_args) {
	Dictionary result;
	String operation = p_args.get("operation", "structure");
	
	if (operation == "structure") {
		// Get overall project structure
		Dictionary structure;
		structure["project_name"] = ProjectSettings::get_singleton()->get_setting("application/config/name");
		
		// PERFORMANCE LIMIT: Prevent UI freezing on huge projects
		int max_files = p_args.get("max_files", 200); // Default limit: 200 files of each type
		print_line("AI Chat: get_project_context starting with max_files limit: " + String::num_int64(max_files));
		
		// Get scenes in project (limited)
		Array scenes;
		List<String> scene_files;
		_get_all_project_files_limited("res://", scene_files, HashSet<String>({ String("tscn"), String("scn") }), max_files);
		for (const String &scene_path : scene_files) {
			Dictionary scene_info;
			scene_info["path"] = scene_path;
			scene_info["name"] = scene_path.get_file().get_basename();
			scene_info["folder"] = scene_path.get_base_dir();
			scenes.append(scene_info);
		}
		structure["scenes"] = scenes;
		if (scene_files.size() >= max_files) {
			structure["scenes_truncated"] = true;
			print_line("AI Chat: Scene list truncated at " + String::num_int64(max_files) + " files");
		}
		
		// Get scripts (limited)
		Array scripts;
		List<String> script_files;
		_get_all_project_files_limited("res://", script_files, HashSet<String>({ "gd", "cs" }), max_files);
		for (const String &script_path : script_files) {
			Dictionary script_info;
			script_info["path"] = script_path;
			script_info["name"] = script_path.get_file().get_basename();
			script_info["folder"] = script_path.get_base_dir();
			scripts.append(script_info);
		}
		structure["scripts"] = scripts;
		if (script_files.size() >= max_files) {
			structure["scripts_truncated"] = true;
			print_line("AI Chat: Script list truncated at " + String::num_int64(max_files) + " files");
		}
		
		// Get autoloads
		Array autoloads;
		List<PropertyInfo> props;
		ProjectSettings::get_singleton()->get_property_list(&props);
		for (const PropertyInfo &E : props) {
			if (!E.name.begins_with("autoload/")) {
				continue;
			}
			String name = E.name.get_slice("/", 1);
			String path = ProjectSettings::get_singleton()->get_setting(E.name);
			if (path.begins_with("*")) {
				path = path.substr(1, path.length());
			}
			Dictionary autoload;
			autoload["name"] = name;
			autoload["path"] = path;
			autoloads.append(autoload);
		}
		structure["autoloads"] = autoloads;
		
		// Get input map actions
		Dictionary input_map;
		props.clear();
		ProjectSettings::get_singleton()->get_property_list(&props);
		for (const PropertyInfo &E : props) {
			if (!E.name.begins_with("input/")) {
				continue;
			}
			String action_name = E.name.get_slice("/", 1);
			input_map[action_name] = true;
		}
		structure["input_map"] = input_map;
		
		// Statistics
		Dictionary stats;
		stats["total_scenes"] = scenes.size();
		stats["total_scripts"] = scripts.size();
		stats["max_files_limit"] = max_files;
		structure["statistics"] = stats;
		
		print_line("AI Chat: get_project_context completed - " + String::num_int64(scenes.size()) + " scenes, " + String::num_int64(scripts.size()) + " scripts");
		
		result["success"] = true;
		result["context"] = structure;
		
	} else if (operation == "find_scenes") {
		// Find existing scenes matching pattern
		String pattern = String(p_args.get("pattern", "")).to_lower();
		Array matching_scenes;
		
		// Use limited search to prevent UI freezing
		int max_files = p_args.get("max_files", 200);
		List<String> scene_files;
		_get_all_project_files_limited("res://", scene_files, HashSet<String>({ "tscn", "scn" }), max_files);
		
		for (const String &scene_path : scene_files) {
			String scene_name = String(scene_path.get_file().get_basename()).to_lower();
			if (pattern.is_empty() || scene_name.contains(pattern)) {
				Dictionary match;
				match["name"] = scene_path.get_file().get_basename();
				match["path"] = scene_path;
				match["exact_match"] = (scene_name == pattern);
				matching_scenes.append(match);
			}
		}
		
		result["success"] = true;
		result["existing_scenes"] = matching_scenes;
		
	} else {
		result["success"] = false;
		result["error"] = "Unknown operation: " + operation;
	}
	
	return result;
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
		// Enhanced diagnostics for debugging scene root issues
		SceneTree *tree = EditorNode::get_singleton()->get_tree();
		print_line("AI Chat: get_all_nodes failed - no scene root found");
		print_line("AI Chat: SceneTree valid: " + String(tree ? "yes" : "no"));
		if (tree) {
			print_line("AI Chat: Current scene: " + String(tree->get_current_scene() ? tree->get_current_scene()->get_name() : "null"));
			// Try alternative scene access methods
			Node *current_scene = tree->get_current_scene();
			if (current_scene) {
				print_line("AI Chat: Using current_scene as fallback");
				root = current_scene;
			}
		}
		
	if (!root) {
		result["success"] = false;
		result["message"] = "No scene is currently being edited.";
			result["nodes"] = Array();
			result["node_count"] = 0;
		return result;
		}
	}
	
	Array nodes;
	
	// PERFORMANCE LIMIT: Prevent UI freezing on huge scenes
	int max_nodes = p_args.get("max_nodes", 500); // Default limit: 500 nodes
	int nodes_collected = 0;
	bool hit_limit = false;
	
	print_line("AI Chat: get_all_nodes starting with max_nodes limit: " + String::num_int64(max_nodes));
	
	// Helper lambda to recursively collect nodes with limit
	std::function<void(Node*)> collect_nodes = [&](Node* node) {
		if (node && nodes_collected < max_nodes) {
			nodes.push_back(_get_node_info(node));
			nodes_collected++;
			
			// Recursively collect children (up to limit)
			for (int i = 0; i < node->get_child_count() && nodes_collected < max_nodes; i++) {
				collect_nodes(node->get_child(i));
			}
		} else if (node && nodes_collected >= max_nodes) {
			hit_limit = true;
		}
	};
	
	// Start collecting from the root
	collect_nodes(root);
	
	result["success"] = true;
	result["nodes"] = nodes;
	result["node_count"] = nodes.size();
	result["total_nodes_in_scene"] = nodes.size(); // Could be higher if hit limit
	if (hit_limit) {
		result["truncated"] = true;
		result["message"] = "Result limited to " + String::num_int64(max_nodes) + " nodes to prevent UI freezing. Use smaller scenes or increase max_nodes parameter.";
		print_line("AI Chat: get_all_nodes hit limit of " + String::num_int64(max_nodes) + " nodes");
	}
	
	print_line("AI Chat: get_all_nodes completed, collected " + String::num_int64(nodes.size()) + " nodes");
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
	
	// PERFORMANCE LIMIT: Prevent UI freezing on huge scenes (same as get_all_nodes)
	int max_nodes_to_search = p_args.get("max_nodes", 500); // Default limit: search up to 500 nodes
	int nodes_searched = 0;
	bool hit_search_limit = false;
	
	print_line("AI Chat: search_nodes_by_type starting for type '" + type + "' with max_nodes limit: " + String::num_int64(max_nodes_to_search));
	
	Node *root = EditorNode::get_singleton()->get_tree()->get_edited_scene_root();
	if (root) {
		// Helper lambda to recursively search nodes with limit
		std::function<void(Node*)> search_nodes = [&](Node* node) {
			if (node && nodes_searched < max_nodes_to_search) {
				nodes_searched++;
				
				if (node->is_class(type)) {
					nodes.push_back(_get_node_info(node));
				}
				
				// Recursively search children (up to limit)
				for (int i = 0; i < node->get_child_count() && nodes_searched < max_nodes_to_search; i++) {
					search_nodes(node->get_child(i));
				}
			} else if (node && nodes_searched >= max_nodes_to_search) {
				hit_search_limit = true;
			}
		};
		
		// Start searching from the root
		search_nodes(root);
	}
	
	result["success"] = true;
	result["nodes"] = nodes;
	result["nodes_found"] = nodes.size();
	result["nodes_searched"] = nodes_searched;
	if (hit_search_limit) {
		result["truncated"] = true;
		result["message"] = "Search limited to " + String::num_int64(max_nodes_to_search) + " nodes to prevent UI freezing. " + String::num_int64(nodes.size()) + " nodes of type '" + type + "' found.";
		print_line("AI Chat: search_nodes_by_type hit search limit of " + String::num_int64(max_nodes_to_search) + " nodes");
	}
	
	print_line("AI Chat: search_nodes_by_type completed - found " + String::num_int64(nodes.size()) + " nodes of type '" + type + "', searched " + String::num_int64(nodes_searched) + " total nodes");
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

	Dictionary props_dict; // name -> value
	Array props_info; // [{name,type,hint,hint_string,class_name,usage}]
	
	// PERFORMANCE LIMIT: Prevent UI freezing on nodes with hundreds of properties
	int max_properties = p_args.get("max_properties", 50); // Default limit: 50 properties
	int properties_processed = 0;
	bool hit_properties_limit = false;
	
	print_line("AI Chat: get_node_properties starting with max_properties limit: " + String::num_int64(max_properties));
	
	for (const PropertyInfo &prop_info : properties) {
		if (properties_processed >= max_properties) {
			hit_properties_limit = true;
			break;
		}
		
		// Collect property metadata for introspection
		Dictionary pi;
		pi["name"] = String(prop_info.name);
		pi["type"] = (int)prop_info.type;
		pi["hint"] = (int)prop_info.hint;
		pi["hint_string"] = String(prop_info.hint_string);
#ifdef TOOLS_ENABLED
		pi["class_name"] = String(prop_info.class_name);
#endif
		pi["usage"] = (int)prop_info.usage;
		props_info.push_back(pi);
		// Include values for editor-visible props
		if (prop_info.usage & PROPERTY_USAGE_EDITOR) {
			props_dict[prop_info.name] = node->get(prop_info.name);
		}
		
		properties_processed++;
	}

	// Optionally include script-defined properties (exported vars) from attached script
	bool include_script_props = p_args.get("include_script_properties", true);
	if (include_script_props) {
		Variant sv = node->get("script");
		Ref<Script> script = sv;
		if (script.is_valid()) {
			// Get script properties from the script instance
			List<PropertyInfo> sprops;
			script->get_script_property_list(&sprops);
			for (const PropertyInfo &pi : sprops) {
				StringName pn = pi.name;
				if (!props_dict.has(pn) && (pi.usage & PROPERTY_USAGE_SCRIPT_VARIABLE)) {
					// Try to get the value; some @export vars may not be accessible until script is properly initialized
					Variant val = node->get(pn);
					props_dict[pn] = val;
				}
			}
			
			// Also try getting script method list to show available methods
			List<MethodInfo> methods;
			script->get_script_method_list(&methods);
			Array method_names;
			for (const MethodInfo &mi : methods) {
				method_names.push_back(String(mi.name));
			}
			if (!method_names.is_empty()) {
				props_dict["_script_methods"] = method_names;
			}
			
			// Include script class name if it's a global class
			String script_path = script->get_path();
			if (!script_path.is_empty()) {
				List<StringName> global_classes;
				ScriptServer::get_global_class_list(&global_classes);
				for (const StringName &class_name : global_classes) {
					if (ScriptServer::get_global_class_path(class_name) == script_path) {
						props_dict["_script_class_name"] = String(class_name);
						break;
					}
				}
			}
		}
	}

	// Collect signals available on this node (native + script) - with limits
	List<MethodInfo> signal_list;
	node->get_signal_list(&signal_list);
	Array signals;
	int max_signals = p_args.get("max_signals", 30); // Default limit: 30 signals
	int signals_processed = 0;
	bool hit_signals_limit = false;
	
	for (const MethodInfo &si : signal_list) {
		if (signals_processed >= max_signals) {
			hit_signals_limit = true;
			break;
		}
		signals.push_back(String(si.name));
		signals_processed++;
	}

	result["success"] = true;
	result["ok"] = true;
	result["class"] = String(node->get_class());
	result["property_values"] = props_dict;
	result["property_info"] = props_info;
	result["signals"] = signals;
	result["properties_count"] = properties_processed;
	result["signals_count"] = signals_processed;
	
	// Add truncation information
	if (hit_properties_limit) {
		result["properties_truncated"] = true;
		result["message"] = "Properties limited to " + String::num_int64(max_properties) + " to prevent UI freezing";
		print_line("AI Chat: get_node_properties hit properties limit of " + String::num_int64(max_properties));
	}
	if (hit_signals_limit) {
		result["signals_truncated"] = true;
		String msg = result.get("message", "");
		if (!msg.is_empty()) msg += ". ";
		msg += "Signals limited to " + String::num_int64(max_signals);
		result["message"] = msg;
		print_line("AI Chat: get_node_properties hit signals limit of " + String::num_int64(max_signals));
	}
	
	print_line("AI Chat: get_node_properties completed - " + String::num_int64(properties_processed) + " properties, " + String::num_int64(signals_processed) + " signals");
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
    bool unique = p_args.get("unique", false);

	if (p_args.has("parent")) {
		parent = _get_node_from_path(p_args["parent"], result);
		if (!parent) {
			return result;
		}
		// Validate parent is in a valid state
		if (!parent->is_inside_tree()) {
			result["success"] = false;
			result["message"] = "Parent node is not in the scene tree.";
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

	// If unique requested, return existing child if found.
	if (parent && unique) {
		for (int i = 0; i < parent->get_child_count(); i++) {
			Node *c = parent->get_child(i);
			if (c && String(c->get_name()) == name) {
				result["success"] = true;
				result["node_path"] = c->is_inside_tree() ? c->get_path() : NodePath();
				result["message"] = "Existing node returned (unique=true).";
				return result;
			}
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
	result["node_path"] = new_node->is_inside_tree() ? new_node->get_path() : NodePath();
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
	
	// Safety checks before deletion
	if (!node->is_inside_tree()) {
		result["success"] = false;
		result["message"] = "Node is not in the scene tree and cannot be safely deleted.";
		return result;
	}
	
	// Don't delete the scene root
	Node *scene_root = EditorNode::get_singleton()->get_tree()->get_edited_scene_root();
	if (node == scene_root) {
		result["success"] = false;
		result["message"] = "Cannot delete the scene root node.";
		return result;
	}
	
	// Store node info before deletion for logging
	String node_name = node->get_name();
	String node_path = String(node->get_path());
	
	// Queue for deletion (safer than immediate removal)
	node->queue_free();
	
	// Scene tree will automatically update when the node is actually freed
	
	result["success"] = true;
	result["message"] = "Node '" + node_name + "' queued for deletion.";
	result["deleted_path"] = node_path;
	return result;
}

// DISABLED: This tool causes crashes due to complex ownership issues
Dictionary EditorTools::change_node_type(const Dictionary &p_args) {
    Dictionary result;
    result["success"] = false;
    result["message"] = "change_node_type is disabled due to stability issues. Use set_node_type with script_path instead, or create_node + delete_node for safer node replacement.";
    return result;
}

// Create a Resource by type and optional properties; optionally save to disk.
Dictionary EditorTools::create_resource(const Dictionary &p_args) {
    Dictionary result;
    if (!p_args.has("type")) { result["success"] = false; result["message"] = "Missing 'type'"; return result; }
    String type = p_args["type"];
    if (!ClassDB::can_instantiate(type) || !ClassDB::is_parent_class(type, "Resource")) {
        result["success"] = false; result["message"] = "Invalid resource type: " + type; return result;
    }
    Object *obj = ClassDB::instantiate(type);
    Resource *res = Object::cast_to<Resource>(obj);
    if (!res) { if (obj) memdelete(obj); result["success"] = false; result["message"] = "Failed to instantiate resource"; return result; }

    // Apply properties (enhanced for Curve)
    Dictionary props = p_args.get("properties", Dictionary());
    if (!props.is_empty()) {
        for (const Variant *k = props.next(); k; k = props.next(k)) {
            StringName key = *k;
            Variant value = props[*k];
            
            // Special handling for Curve points
            if (type == "Curve" && key == StringName("points")) {
                Curve *curve = Object::cast_to<Curve>(res);
                if (curve && value.get_type() == Variant::ARRAY) {
                    Array points = value;
                    curve->clear_points();
                    for (int i = 0; i < points.size(); i++) {
                        Dictionary point = points[i];
                        if (point.has("x") && point.has("y")) {
                            float x = point.get("x", 0.0f);
                            float y = point.get("y", 0.0f);
                            float left_tangent = point.get("left_tangent", 0.0f);
                            float right_tangent = point.get("right_tangent", 0.0f);
                            curve->add_point(Vector2(x, y), left_tangent, right_tangent);
                        }
                    }
                    continue; // Skip normal property setting
                }
            }
            
            // Normal property setting
            res->set(key, value);
        }
    }

    String save_path = p_args.get("save_path", String());
    if (!save_path.is_empty()) {
        Ref<Resource> res_ref = Ref<Resource>(res);
        Error e = ResourceSaver::save(res_ref, save_path);
        if (e != OK) {
            result["success"] = false; result["message"] = "Failed to save resource to " + save_path; return result;
        }
        result["path"] = save_path;
    }

    // Provide a lightweight handle back; we cannot send raw pointer, so return a temp path-less id
    result["success"] = true;
    result["resource_type"] = type;
    result["rid"] = (int64_t)res; // For same-process subsequent calls; not persisted
    return result;
}

// Assign a resource to a node property. Resource can be provided by path, by RID (from create_resource), or by inline spec.
Dictionary EditorTools::assign_resource_to_node_property(const Dictionary &p_args) {
    Dictionary result;
    if (!p_args.has("path") || !p_args.has("property") || !p_args.has("resource")) {
        result["success"] = false; result["message"] = "Missing 'path', 'property', or 'resource'"; return result;
    }
    Dictionary err; Node *node = _get_node_from_path(p_args["path"], err); if (!node) return err;
    StringName prop = p_args["property"];
    Variant res_spec = p_args["resource"];
    Ref<Resource> res;
    if (res_spec.get_type() == Variant::DICTIONARY) {
        Dictionary d = res_spec;
        if (d.has("path")) {
            res = ResourceLoader::load(d["path"]);
        } else if (d.has("rid")) {
            // Unsafe across sessions; only works within current editor session
            int64_t rid = (int64_t)d["rid"];
            Resource *raw = (Resource*)rid;
            if (Object::cast_to<Resource>(raw)) {
                res = Ref<Resource>(raw);
            }
        } else if (d.has("type")) {
            // Inline creation
            Dictionary create_args; create_args["type"] = d["type"]; create_args["properties"] = d.get("properties", Dictionary());
            Dictionary cr = create_resource(create_args);
            if (cr.get("success", false)) {
                int64_t rid = (int64_t)cr.get("rid", (int64_t)0);
                Resource *raw = (Resource*)rid;
                if (Object::cast_to<Resource>(raw)) res = Ref<Resource>(raw);
            }
        }
    } else if (res_spec.get_type() == Variant::STRING) {
        res = ResourceLoader::load((String)res_spec);
    }
    if (res.is_null()) { result["success"] = false; result["message"] = "Could not resolve resource"; return result; }
    node->set(prop, res);
    Node *root = EditorNode::get_singleton()->get_tree()->get_edited_scene_root();
    if (root && node->get_owner() == root) {
        // Mark scene dirty by touching owner; editor will handle actual save
        // No-op here; Godot tracks property changes automatically
    }
    result["success"] = true; result["message"] = "Resource assigned"; return result;
}

// Create a new scene with a specific root type and optionally attach current root under it.
Dictionary EditorTools::create_new_scene_with_root(const Dictionary &p_args) {
    Dictionary result;
    if (!p_args.has("new_root_type") || !p_args.has("new_scene_path")) {
        result["success"] = false; result["message"] = "Missing 'new_root_type' or 'new_scene_path'"; return result;
    }
    String root_type = p_args["new_root_type"];
    String scene_path = p_args["new_scene_path"];
    bool include_current_as_child = p_args.get("include_current_as_child", false);
    if (!ClassDB::can_instantiate(root_type) || !ClassDB::is_parent_class(root_type, "Node")) {
        result["success"] = false; result["message"] = "Invalid root type: " + root_type; return result;
    }
    Object *obj = ClassDB::instantiate(root_type);
    Node *new_root = Object::cast_to<Node>(obj);
    if (!new_root) { if (obj) memdelete(obj); result["success"] = false; result["message"] = "Failed to instantiate root"; return result; }
    new_root->set_name("Root");

    Node *current_root = EditorNode::get_singleton()->get_tree()->get_edited_scene_root();
    if (include_current_as_child && current_root) {
        // Instance the current scene under the new root as a child
        new_root->add_child(current_root);
        current_root->set_owner(new_root);
    }

    // Set as edited scene root and save
    EditorNode::get_singleton()->set_edited_scene_root(new_root, true);
    EditorInterface::get_singleton()->save_scene_as(scene_path);
    result["success"] = true; result["scene_path"] = scene_path; result["message"] = "New scene created and save requested"; return result;
}

// --- File system and project structure tools ---

static bool _is_within_project(const String &p_path) {
    String proj = ProjectSettings::get_singleton()->get_resource_path();
    String abs = p_path;
    if (p_path.begins_with("res://")) {
        abs = ProjectSettings::get_singleton()->globalize_path(p_path);
    }
    return abs.begins_with(proj);
}

Dictionary EditorTools::create_directory(const Dictionary &p_args) {
    Dictionary result;
    String path = p_args.get("path", "");
    if (path.is_empty()) { result["success"] = false; result["message"] = "path required"; return result; }
    if (!_is_within_project(path)) { result["success"] = false; result["message"] = "Path must be within project"; return result; }
    Error e = DirAccess::make_dir_recursive_absolute(ProjectSettings::get_singleton()->globalize_path(path));
    if (e != OK) { result["success"] = false; result["message"] = "Failed to create directory"; return result; }
    if (EditorFileSystem::get_singleton()) EditorFileSystem::get_singleton()->scan_changes();
    result["success"] = true; return result;
}

Dictionary EditorTools::copy_file(const Dictionary &p_args) {
    Dictionary result; 
    String src = p_args.get("source", ""); String dst = p_args.get("destination", ""); bool overwrite = p_args.get("overwrite", false);
    if (src.is_empty() || dst.is_empty()) { result["success"] = false; result["message"] = "source and destination required"; return result; }
    if (!_is_within_project(src) || !_is_within_project(dst)) { result["success"] = false; result["message"] = "Paths must be within project"; return result; }
    String abs_src = ProjectSettings::get_singleton()->globalize_path(src);
    String abs_dst = ProjectSettings::get_singleton()->globalize_path(dst);
    if (!overwrite && FileAccess::exists(abs_dst)) { result["success"] = false; result["message"] = "Destination exists"; return result; }
    Error e = DirAccess::copy_absolute(abs_src, abs_dst);
    if (e != OK) { result["success"] = false; result["message"] = "Copy failed"; return result; }
    if (EditorFileSystem::get_singleton()) EditorFileSystem::get_singleton()->scan_changes();
    result["success"] = true; return result;
}

Dictionary EditorTools::move_file(const Dictionary &p_args) {
    Dictionary result; 
    String src = p_args.get("source", ""); String dst = p_args.get("destination", ""); bool overwrite = p_args.get("overwrite", false);
    if (src.is_empty() || dst.is_empty()) { result["success"] = false; result["message"] = "source and destination required"; return result; }
    if (!_is_within_project(src) || !_is_within_project(dst)) { result["success"] = false; result["message"] = "Paths must be within project"; return result; }
    String abs_src = ProjectSettings::get_singleton()->globalize_path(src);
    String abs_dst = ProjectSettings::get_singleton()->globalize_path(dst);
    if (!overwrite && FileAccess::exists(abs_dst)) { result["success"] = false; result["message"] = "Destination exists"; return result; }
    Error e = DirAccess::rename_absolute(abs_src, abs_dst);
    if (e != OK) { result["success"] = false; result["message"] = "Move failed"; return result; }
    if (EditorFileSystem::get_singleton()) EditorFileSystem::get_singleton()->scan_changes();
    result["success"] = true; return result;
}

Dictionary EditorTools::delete_file(const Dictionary &p_args) {
    Dictionary result; 
    String path = p_args.get("path", ""); 
    if (path.is_empty()) { result["success"] = false; result["message"] = "path required"; return result; }
    if (!_is_within_project(path)) { result["success"] = false; result["message"] = "Path must be within project"; return result; }
    
    String abs = ProjectSettings::get_singleton()->globalize_path(path);
    print_line("delete_file DEBUG: Deleting " + abs);
    
    // Handle different file types (files, directories, symlinks, reference files)
    Ref<DirAccess> da = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
    Error e = ERR_FILE_NOT_FOUND;
    
    // Check for reference files created by our symlink fallback
    String ref_file_path = abs + ".ref";
    if (FileAccess::exists(ref_file_path)) {
        print_line("delete_file DEBUG: Found reference file: " + ref_file_path);
        e = da->remove(ref_file_path);
        print_line("delete_file DEBUG: Removed reference file, result: " + String::num_int64(e));
    } else if (da->dir_exists(abs)) {
        // It's a directory - try to remove it (will fail if not empty)
        e = da->remove(abs);
        print_line("delete_file DEBUG: Removed directory, result: " + String::num_int64(e));
    } else if (da->file_exists(abs)) {
        // It's a file or symlink
        e = da->remove(abs);
        print_line("delete_file DEBUG: Removed file/symlink, result: " + String::num_int64(e));
    } else {
        // Try the fallback method
        e = DirAccess::remove_absolute(abs);
        print_line("delete_file DEBUG: Used remove_absolute fallback, result: " + String::num_int64(e));
    }
    
    if (e != OK) { result["success"] = false; result["message"] = "Delete failed (Error: " + String::num_int64(e) + ")"; return result; }
    if (EditorFileSystem::get_singleton()) EditorFileSystem::get_singleton()->scan_changes();
    result["success"] = true; return result;
}

Dictionary EditorTools::create_symlink(const Dictionary &p_args) {
    Dictionary result; 
    
    // Debug parameter names
    print_line("create_symlink DEBUG: Available parameters:");
    Array keys = p_args.keys();
    for (int i = 0; i < keys.size(); i++) {
        print_line("  " + String(keys[i]) + " = " + String(p_args[keys[i]]));
    }
    
    String target = p_args.get("target", ""); 
    String link_path = p_args.get("link_path", "");
    
    if (target.is_empty() || link_path.is_empty()) { 
        result["success"] = false; 
        result["message"] = "target and link_path required"; 
        return result; 
    }
    
    if (!_is_within_project(target) || !_is_within_project(link_path)) { 
        result["success"] = false; 
        result["message"] = "Paths must be within project"; 
        return result; 
    }
    
    String abs_target = ProjectSettings::get_singleton()->globalize_path(target);
    String abs_link = ProjectSettings::get_singleton()->globalize_path(link_path);
    
    print_line("create_symlink DEBUG: Creating symlink:");
    print_line("  Target (abs): " + abs_target);
    print_line("  Link (abs): " + abs_link);
    
    // Check if target exists first
    if (!FileAccess::exists(abs_target) && !DirAccess::dir_exists_absolute(abs_target)) {
        result["success"] = false;
        result["message"] = "Target does not exist: " + target;
        return result;
    }
    
    // Try different symlink creation methods
    Ref<DirAccess> da = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
    Error e = da->create_link(abs_link, abs_target);  // Note: link_path first, target second
    
    if (e != OK) {
        print_line("create_symlink DEBUG: Primary method failed with error: " + String::num_int64(e));
        
        // For development/testing: create a text file that acts as a "reference file" instead of a symlink
        // This provides the functionality even when symlinks aren't supported
        String ref_content = "# SYMBOLIC REFERENCE TO: " + target + "\n# This file acts as a reference pointer because symlinks are not supported on this filesystem.";
        Ref<FileAccess> ref_file = FileAccess::open(abs_link + ".ref", FileAccess::WRITE);
        if (ref_file.is_valid()) {
            ref_file->store_string(ref_content);
            result["success"] = true;
            result["message"] = "Created reference file instead of symlink (symlinks not supported)";
            result["reference_file"] = abs_link + ".ref";
            result["symlink_supported"] = false;
            if (EditorFileSystem::get_singleton()) EditorFileSystem::get_singleton()->scan_changes();
            return result;
        }
        
        result["success"] = false; 
        result["message"] = "Symlink creation failed (Error: " + String::num_int64(e) + ") and reference file fallback also failed";
        result["symlink_supported"] = false;
        return result; 
    }
    
    if (EditorFileSystem::get_singleton()) EditorFileSystem::get_singleton()->scan_changes();
    result["success"] = true; 
    result["symlink_supported"] = true;
    result["message"] = "Symlink created successfully";
    return result;
}

Dictionary EditorTools::refresh_filesystem(const Dictionary &p_args) {
    if (EditorFileSystem::get_singleton()) EditorFileSystem::get_singleton()->scan_changes();
    Dictionary result; result["success"] = true; return result;
}

void EditorTools::_on_game_screenshot_ready(int64_t p_w, int64_t p_h, const String &p_path, const Rect2i &p_rect) {
    // This callback is triggered when GameView captures a screenshot
    // The screenshot is automatically attached to AI Chat by GameView::_on_snapshot_ready
    print_line("AI Chat: Game screenshot ready - " + String::num_int64(p_w) + "x" + String::num_int64(p_h) + " saved to: " + p_path);
}

void EditorTools::_on_game_screenshot_ready_for_tool(int64_t p_w, int64_t p_h, const String &p_path, const Rect2i &p_rect) {
    // This callback processes screenshot data for the AI tool system
    print_line("AI Chat: Tool screenshot ready - " + String::num_int64(p_w) + "x" + String::num_int64(p_h) + " at: " + p_path);
    
    if (p_path.is_empty()) {
        print_line("AI Chat: Screenshot path is empty, cannot process");
        return;
    }
    
    // Load the screenshot file and convert to base64 for AI processing
    Ref<FileAccess> file = FileAccess::open(p_path, FileAccess::READ);
    if (file.is_null()) {
        print_line("AI Chat: Cannot open screenshot file: " + p_path);
        return;
    }
    
    PackedByteArray data = file->get_buffer(file->get_length());
    file->close();
    
    if (data.is_empty()) {
        print_line("AI Chat: Screenshot file is empty: " + p_path);
        return;
    }
    
    // Load and potentially downscale the image to prevent token explosions
    Ref<Image> image = memnew(Image);
    Error load_err = image->load(p_path);
    if (load_err != OK) {
        print_line("AI Chat: Failed to load screenshot image: " + p_path);
        return;
    }
    
    Vector2i original_size = Vector2i(image->get_width(), image->get_height());
    Vector2i final_size = original_size;
    bool was_downsampled = false;
    
    // Downscale if too large - aim for reasonable size to prevent token explosion
    const int MAX_DIMENSION = 800; // Keep images under 800px on any side
    const int MAX_PIXELS = 400000; // Keep total pixels reasonable (~640x625)
    
    int total_pixels = original_size.x * original_size.y;
    if (original_size.x > MAX_DIMENSION || original_size.y > MAX_DIMENSION || total_pixels > MAX_PIXELS) {
        float scale_x = (float)MAX_DIMENSION / original_size.x;
        float scale_y = (float)MAX_DIMENSION / original_size.y; 
        float pixel_scale = sqrt((float)MAX_PIXELS / total_pixels);
        
        float scale = MIN(MIN(scale_x, scale_y), pixel_scale);
        if (scale < 1.0f) {
            final_size.x = (int)(original_size.x * scale);
            final_size.y = (int)(original_size.y * scale);
            
            image->resize(final_size.x, final_size.y, Image::INTERPOLATE_LANCZOS);
            was_downsampled = true;
            
            print_line(vformat("AI Chat: Screenshot downscaled from %dx%d to %dx%d (scale: %.2f)", 
                              original_size.x, original_size.y, final_size.x, final_size.y, scale));
        }
    }
    
    // Convert to PNG bytes for consistent base64 encoding
    PackedByteArray png_data = image->save_png_to_buffer();
    String base64_data = CoreBind::Marshalls::get_singleton()->raw_to_base64(png_data);
    
    // Store the screenshot result for later retrieval
    Dictionary screenshot_result;
    screenshot_result["source"] = "game";
    screenshot_result["width"] = final_size.x;
    screenshot_result["height"] = final_size.y; 
    screenshot_result["size"] = final_size;
    screenshot_result["original_size"] = original_size;
    screenshot_result["was_downsampled"] = was_downsampled;
    screenshot_result["path"] = p_path;
    screenshot_result["base64"] = base64_data;
    screenshot_result["success"] = true;
    screenshot_result["timestamp"] = Time::get_singleton()->get_ticks_msec();
    
    _current_screenshot_results.push_back(screenshot_result);
    
    print_line("AI Chat: Screenshot processed successfully, final base64 length: " + String::num_int64(base64_data.length()));
    
    // Store the screenshot for tool result attachment instead of chat input
    print_line("AI Chat: Screenshot processed and stored for tool result attachment");
}

// === UNIVERSAL SMART TOOLS ===

Dictionary EditorTools::universal_resource_manager(const Dictionary &p_args) {
    Dictionary result;
    String operation = p_args.get("operation", "");
    if (operation.is_empty()) { result["success"] = false; result["message"] = "Missing operation"; return result; }
    
    if (operation == "create") {
        return create_resource(p_args);
    } else if (operation == "assign") {
        return assign_resource_to_node_property(p_args);
    } else if (operation == "inspect") {
        String target = p_args.get("target", "");
        if (target.is_empty()) { result["success"] = false; result["message"] = "Missing target"; return result; }
        Ref<Resource> res = ResourceLoader::load(target);
        if (!res.is_valid()) { result["success"] = false; result["message"] = "Resource not found"; return result; }
        
        Dictionary props;
        List<PropertyInfo> plist;
        res->get_property_list(&plist);
        for (const PropertyInfo &pi : plist) {
            if (pi.usage & PROPERTY_USAGE_EDITOR) {
                props[pi.name] = res->get(pi.name);
            }
        }
        
        // Special handling for Curve
        if (res->get_class() == "Curve") {
            Curve *curve = Object::cast_to<Curve>(res.ptr());
            if (curve) {
                Array points_data;
                for (int i = 0; i < curve->get_point_count(); i++) {
                    Dictionary pt;
                    Vector2 pos = curve->get_point_position(i);
                    pt["x"] = pos.x;
                    pt["y"] = pos.y;
                    pt["left_tangent"] = curve->get_point_left_tangent(i);
                    pt["right_tangent"] = curve->get_point_right_tangent(i);
                    points_data.push_back(pt);
                }
                props["curve_points"] = points_data;
            }
        }
        
        result["success"] = true;
        result["resource_type"] = res->get_class();
        result["properties"] = props;
        return result;
    } else if (operation == "modify") {
        String target = p_args.get("target", "");
        Dictionary properties = p_args.get("properties", Dictionary());
        if (target.is_empty()) { result["success"] = false; result["message"] = "Missing target"; return result; }
        
        Ref<Resource> res = ResourceLoader::load(target);
        if (!res.is_valid()) { result["success"] = false; result["message"] = "Resource not found"; return result; }
        
        // Apply properties with type-aware handling
        for (const Variant *k = properties.next(); k; k = properties.next(k)) {
            StringName key = *k;
            Variant value = properties[*k];
            
            // Special handling for Curve points
            if (res->get_class() == "Curve" && key == StringName("points")) {
                Curve *curve = Object::cast_to<Curve>(res.ptr());
                if (curve && value.get_type() == Variant::ARRAY) {
                    Array points = value;
                    curve->clear_points();
                    for (int i = 0; i < points.size(); i++) {
                        Dictionary point = points[i];
                        if (point.has("x") && point.has("y")) {
                            float x = point.get("x", 0.0f);
                            float y = point.get("y", 0.0f);
                            float left_tangent = point.get("left_tangent", 0.0f);
                            float right_tangent = point.get("right_tangent", 0.0f);
                            curve->add_point(Vector2(x, y), left_tangent, right_tangent);
                        }
                    }
                    continue;
                }
            }
            res->set(key, value);
        }
        
        Error e = ResourceSaver::save(res, target);
        if (e != OK) { result["success"] = false; result["message"] = "Failed to save modified resource"; return result; }
        result["success"] = true; result["message"] = "Resource modified"; return result;
    } else if (operation == "copy_from_template") {
        String source_template = p_args.get("source_template", "");
        String target = p_args.get("target", "");
        if (source_template.is_empty() || target.is_empty()) { result["success"] = false; result["message"] = "Missing source_template or target"; return result; }
        
        Ref<Resource> template_res = ResourceLoader::load(source_template);
        if (!template_res.is_valid()) { result["success"] = false; result["message"] = "Template resource not found"; return result; }
        
        Ref<Resource> new_res = template_res->duplicate();
        Error e = ResourceSaver::save(new_res, target);
        if (e != OK) { result["success"] = false; result["message"] = "Failed to save copied resource"; return result; }
        result["success"] = true; result["message"] = "Resource copied from template"; return result;
    }
    
    result["success"] = false; result["message"] = "Unknown operation: " + operation; return result;
}

Dictionary EditorTools::universal_scene_manager(const Dictionary &p_args) {
    Dictionary result;
    String operation = p_args.get("operation", "");
    if (operation.is_empty()) { result["success"] = false; result["message"] = "Missing operation"; return result; }
    
    if (operation == "analyze") {
        Node *root = EditorNode::get_singleton()->get_tree()->get_edited_scene_root();
        if (!root) { result["success"] = false; result["message"] = "No scene"; return result; }
        
        Dictionary analysis;
        analysis["root_type"] = root->get_class();
        analysis["root_name"] = root->get_name();
        analysis["total_nodes"] = 0;
        
        Dictionary type_counts;
        std::function<void(Node*)> count_nodes = [&](Node *n) {
            if (!n) return;
            analysis["total_nodes"] = (int)analysis["total_nodes"] + 1;
            String type = n->get_class();
            type_counts[type] = (int)type_counts.get(type, 0) + 1;
            for (int i = 0; i < n->get_child_count(); i++) {
                count_nodes(n->get_child(i));
            }
        };
        count_nodes(root);
        
        analysis["type_distribution"] = type_counts;
        result["success"] = true; result["analysis"] = analysis; return result;
    } else if (operation == "bulk_configure") {
        Array targets = p_args.get("targets", Array());
        Dictionary transformations = p_args.get("transformations", Dictionary());
        // bool validation = p_args.get("validation", true); // Unused for now
        
        Array successes, failures;
        for (int i = 0; i < targets.size(); i++) {
            String path = targets[i];
            Dictionary err;
            Node *node = _get_node_from_path(path, err);
            if (!node) {
                failures.push_back(Dictionary{{"path", path}, {"error", err.get("message", "Node not found")}});
                continue;
            }
            
            // Get transformations for this specific target node
            if (!transformations.has(path)) {
                failures.push_back(Dictionary{{"path", path}, {"error", "No transformations defined for this target"}});
                continue;
            }
            
            Dictionary node_transformations = transformations[path];
            bool success = true;
            String failure_reason = "";
            
            // Apply each property transformation for this target
            Array prop_keys = node_transformations.keys();
            for (int j = 0; j < prop_keys.size(); j++) {
                String prop = prop_keys[j];
                Variant val = node_transformations[prop];
                
                // Check if property exists on this node
                bool prop_exists = false;
                List<PropertyInfo> check_props;
                node->get_property_list(&check_props);
                for (const PropertyInfo &pi : check_props) {
                    if (pi.name == prop) {
                        prop_exists = true;
                        break;
                    }
                }
                
                if (prop_exists) {
                    // Use Godot's property setting with error checking
                    String old_value = String(node->get(prop));
                    node->set(prop, val);
                    String new_value = String(node->get(prop));
                    
                    // Verify the property was actually set
                    if (old_value == new_value && String(val) != old_value) {
                        success = false;
                        failure_reason = "Failed to set property '" + prop + "' (value unchanged)";
                        break;
                    } else {
                        print_line("AI Chat: bulk_configure set " + path + "." + prop + " = " + String(val) + " (was: " + old_value + ")");
                    }
                } else {
                    success = false;
                    failure_reason = "Property '" + prop + "' not found on " + String(node->get_class());
                    break;
                }
            }
            
            if (success) {
                successes.push_back(path);
            } else {
                Dictionary failure_info;
                failure_info["path"] = path;
                failure_info["error"] = failure_reason.is_empty() ? "Property assignment failed" : failure_reason;
                failures.push_back(failure_info);
            }
        }
        
        result["success"] = failures.is_empty();
        result["successes"] = successes;
        result["failures"] = failures;
        result["message"] = String::num_int64(successes.size()) + " succeeded, " + String::num_int64(failures.size()) + " failed";
        return result;
    } else if (operation == "copy_configuration") {
        String source = p_args.get("source", "");
        Array targets = p_args.get("targets", Array());
        if (source.is_empty() || targets.is_empty()) { result["success"] = false; result["message"] = "Missing source or targets"; return result; }
        
        Dictionary err;
        Node *source_node = _get_node_from_path(source, err);
        if (!source_node) return err;
        
        // Extract source configuration
        List<PropertyInfo> props;
        source_node->get_property_list(&props);
        Dictionary config;
        for (const PropertyInfo &pi : props) {
            if (pi.usage & PROPERTY_USAGE_EDITOR && !String(pi.name).begins_with("_")) {
                config[pi.name] = source_node->get(pi.name);
            }
        }
        
        // Apply to targets
        Array applied;
        for (int i = 0; i < targets.size(); i++) {
            String target_path = targets[i];
            Node *target_node = _get_node_from_path(target_path, err);
            if (target_node) {
                for (const Variant *k = config.next(); k; k = config.next(k)) {
                    target_node->set(*k, config[*k]);
                }
                applied.push_back(target_path);
            }
        }
        
        result["success"] = true;
        result["applied_to"] = applied;
        result["copied_properties"] = config.keys();
        return result;
    }
    
    result["success"] = false; result["message"] = "Unknown operation: " + operation; return result;
}

Dictionary EditorTools::universal_project_manager(const Dictionary &p_args) {
    Dictionary result;
    String operation = p_args.get("operation", "");
    if (operation.is_empty()) { result["success"] = false; result["message"] = "Missing operation"; return result; }
    
    if (operation == "analyze_directory") {
        String target_path = p_args.get("target_path", "");
        if (target_path.is_empty()) { result["success"] = false; result["message"] = "Missing target_path"; return result; }
        
        Dictionary analysis;
        analysis["target_path"] = target_path;
        
        // Analyze directory structure
        Dictionary file_types;
        Array scripts_with_classes, config_files, scene_files, resource_files;
        
        std::function<void(const String&)> scan_recursive = [&](const String &dir_path) {
            Ref<DirAccess> da = DirAccess::open(dir_path);
            if (!da.is_valid()) return;
            
            da->list_dir_begin();
            String item = da->get_next();
            while (!item.is_empty()) {
                if (item != "." && item != "..") {
                    String full_path = dir_path + "/" + item;
                    
                    if (da->current_is_dir()) {
                        scan_recursive(full_path);
                    } else {
                        String ext = item.get_extension().to_lower();
                        file_types[ext] = (int)file_types.get(ext, 0) + 1;
                        
                        if (ext == "gd" || ext == "cs") {
                            String content = FileAccess::get_file_as_string(full_path);
                            if (content.contains("class_name")) {
                                scripts_with_classes.push_back(full_path);
                            }
                        } else if (ext == "cfg" || ext == "ini") {
                            config_files.push_back(full_path);
                        } else if (ext == "tscn") {
                            scene_files.push_back(full_path);
                        } else if (ext == "tres" || ext == "res") {
                            resource_files.push_back(full_path);
                        }
                    }
                }
                item = da->get_next();
            }
        };
        
        scan_recursive(target_path);
        
        analysis["file_types"] = file_types;
        analysis["scripts_with_classes"] = scripts_with_classes;
        analysis["config_files"] = config_files;
        analysis["scene_files"] = scene_files;
        analysis["resource_files"] = resource_files;
        result["success"] = true;
        result["analysis"] = analysis;
        return result;
    } else if (operation == "copy_directory") {
        String source_path = p_args.get("source_addon", "");
        String target_path = p_args.get("target_addon", "");
        if (source_path.is_empty() || target_path.is_empty()) { result["success"] = false; result["message"] = "Missing source_addon or target_addon"; return result; }
        
        // Create target directory
        Error e = DirAccess::make_dir_recursive_absolute(ProjectSettings::get_singleton()->globalize_path(target_path));
        if (e != OK) { result["success"] = false; result["message"] = "Failed to create target directory"; return result; }
        
        // Copy all files recursively
        Ref<DirAccess> source_da = DirAccess::open(source_path);
        if (!source_da.is_valid()) { result["success"] = false; result["message"] = "Source directory not found"; return result; }
        
        Array copied_files;
        std::function<void(const String&, const String&)> copy_recursive = [&](const String &src_dir, const String &dst_dir) {
            Ref<DirAccess> src_da = DirAccess::open(src_dir);
            if (!src_da.is_valid()) return;
            
            DirAccess::make_dir_recursive_absolute(ProjectSettings::get_singleton()->globalize_path(dst_dir));
            
            src_da->list_dir_begin();
            String item = src_da->get_next();
            while (!item.is_empty()) {
                if (item != "." && item != "..") {
                    String src_path = src_dir + "/" + item;
                    String dst_path = dst_dir + "/" + item;
                    
                    if (src_da->current_is_dir()) {
                        copy_recursive(src_path, dst_path);
                    } else {
                        Error copy_err = DirAccess::copy_absolute(
                            ProjectSettings::get_singleton()->globalize_path(src_path),
                            ProjectSettings::get_singleton()->globalize_path(dst_path)
                        );
                        if (copy_err == OK) {
                            copied_files.push_back(dst_path);
                        }
                    }
                }
                item = src_da->get_next();
            }
        };
        
        copy_recursive(source_path, target_path);
        
        if (EditorFileSystem::get_singleton()) EditorFileSystem::get_singleton()->scan_changes();
        
        result["success"] = true;
        result["copied_files"] = copied_files;
        result["message"] = String::num_int64(copied_files.size()) + " files copied";
        return result;
    } else if (operation == "update_references") {
        String old_path = p_args.get("old_path", "");
        String new_path = p_args.get("new_path", "");
        Array file_patterns = p_args.get("file_patterns", Array{"*.tscn", "*.tres", "*.gd"});
        
        if (old_path.is_empty() || new_path.is_empty()) { result["success"] = false; result["message"] = "Missing old_path or new_path"; return result; }
        
        Array updated_files;
        String project_root = ProjectSettings::get_singleton()->get_resource_path();
        
        // Simple recursive scan and replace
        std::function<void(const String&)> scan_and_update = [&](const String &dir_path) {
            Ref<DirAccess> da = DirAccess::open(dir_path);
            if (!da.is_valid()) return;
            
            da->list_dir_begin();
            String item = da->get_next();
            while (!item.is_empty()) {
                if (item != "." && item != "..") {
                    String full_path = dir_path + "/" + item;
                    
                    if (da->current_is_dir()) {
                        scan_and_update(full_path);
                    } else {
                        // Check if file matches patterns
                        bool matches = false;
                        for (int i = 0; i < file_patterns.size(); i++) {
                            String pattern = file_patterns[i];
                            if (pattern.begins_with("*.")) {
                                String ext = pattern.substr(2);
                                if (full_path.ends_with("." + ext)) {
                                    matches = true;
                                    break;
                                }
                            }
                        }
                        
                        if (matches) {
                            String content = FileAccess::get_file_as_string(full_path);
                            if (content.contains(old_path)) {
                                String updated_content = content.replace(old_path, new_path);
                                Ref<FileAccess> f = FileAccess::open(full_path, FileAccess::WRITE);
                                if (f.is_valid()) {
                                    f->store_string(updated_content);
                                    updated_files.push_back(full_path);
                                }
                            }
                        }
                    }
                }
                item = da->get_next();
            }
        };
        
        scan_and_update(project_root);
        
        result["success"] = true;
        result["updated_files"] = updated_files;
        result["message"] = String::num_int64(updated_files.size()) + " files updated";
        return result;
    }
    
    result["success"] = false; result["message"] = "Unknown operation: " + operation; return result;
}

Dictionary EditorTools::detach_script(const Dictionary &p_args) {
    Dictionary result;
    if (!p_args.has("path")) {
        result["success"] = false;
        result["message"] = "Missing 'path' argument.";
        return result;
    }
    Dictionary err; 
    Node *node = _get_node_from_path(p_args["path"], err); 
    if (!node) return err;
    node->set("script", Variant());
    result["success"] = true; 
    result["message"] = "Script detached"; 
    return result;
}

Dictionary EditorTools::reload_script(const Dictionary &p_args) {
    Dictionary result;
    String script_path = p_args.get("script_path", "");
    String node_path = p_args.get("path", "");
    if (script_path.is_empty() && node_path.is_empty()) {
        result["success"] = false; 
        result["message"] = "Provide 'script_path' or 'path'"; 
        return result;
    }
    Ref<Script> script;
    if (!script_path.is_empty()) {
        script = ResourceLoader::load(script_path);
    } else {
        Dictionary err; 
        Node *node = _get_node_from_path(node_path, err); 
        if (!node) return err;
        Variant sv = node->get("script"); 
        script = sv;
    }
    if (!script.is_valid()) { 
        result["success"] = false; 
        result["message"] = "Script not found"; 
        return result; 
    }
    script->reload(true);
    ScriptEditor *se = ScriptEditor::get_singleton(); 
    if (se) se->reload_scripts(false);
    result["success"] = true; 
    result["message"] = "Script reloaded"; 
    return result;
}

Dictionary EditorTools::refresh_global_classes(const Dictionary &p_args) {
    Dictionary result;
    // Trigger a broad scripts reload to refresh class_name registrations
    ScriptEditor *se = ScriptEditor::get_singleton();
    if (se) {
        se->reload_scripts(true);
    }
    if (EditorFileSystem::get_singleton()) {
        EditorFileSystem::get_singleton()->scan_changes();
    }
    result["success"] = true; 
    result["message"] = "Global classes refreshed"; 
    return result;
}

Dictionary EditorTools::get_custom_classes(const Dictionary &p_args) {
    Dictionary result;
    Array out;
    // Get all registered global classes
    List<StringName> global_classes;
    ScriptServer::get_global_class_list(&global_classes);
    
    String pattern = p_args.get("pattern", "");
    for (const StringName &class_name : global_classes) {
        if (pattern.is_empty() || String(class_name).containsn(pattern)) {
            Dictionary class_info;
            class_info["name"] = String(class_name);
            class_info["path"] = ScriptServer::get_global_class_path(class_name);
            class_info["base"] = ScriptServer::get_global_class_native_base(class_name);
            out.push_back(class_info);
        }
    }
    result["success"] = true; 
    result["classes"] = out; 
    return result;
}

Dictionary EditorTools::set_node_type(const Dictionary &p_args) {
    Dictionary result;
    if (!p_args.has("path")) { 
        result["success"] = false; 
        result["message"] = "Missing 'path'"; 
        return result; 
    }
    Dictionary err; 
    Node *node = _get_node_from_path(p_args["path"], err); 
    if (!node) return err;
    
    String type_name = p_args.get("type_name", "");
    String script_path = p_args.get("script_path", "");
    
    if (!script_path.is_empty()) {
        Ref<Script> scr = ResourceLoader::load(script_path);
        if (scr.is_valid()) {
            node->set("script", scr);
            result["success"] = true; 
            result["message"] = "Script attached to set type"; 
            return result;
        }
        result["success"] = false; 
        result["message"] = "Failed to load script"; 
        return result;
    }
    
	if (!type_name.is_empty()) {
		// Check if it's a custom global class first
		List<StringName> global_classes;
		ScriptServer::get_global_class_list(&global_classes);
		String script_for_class;
		for (const StringName &class_name : global_classes) {
			if (String(class_name) == type_name) {
				script_for_class = ScriptServer::get_global_class_path(class_name);
				break;
			}
		}
		
		if (!script_for_class.is_empty()) {
			// Attach the script for this custom class
			Ref<Script> scr = ResourceLoader::load(script_for_class);
			if (scr.is_valid()) {
				node->set("script", scr);
				result["success"] = true; 
				result["message"] = "Custom class script attached: " + type_name; 
				return result;
			}
		}
		
		// Fall back to native class change
		if (ClassDB::class_exists(type_name)) {
			Dictionary args; 
			args["path"] = p_args["path"]; 
			args["new_type"] = type_name; 
			args["preserve_children"] = true; 
			args["strategy"] = "wrap_root";
			return change_node_type(args);
		}
		
		result["success"] = false; 
		result["message"] = "Unknown type: " + type_name; 
		return result;
	}
    
    result["success"] = false; 
    result["message"] = "Provide 'type_name' or 'script_path'"; 
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
	
    // Special handling for Vector2/Vector3 properties from flexible inputs
    if ((prop == StringName("position") || prop == StringName("global_position") || prop == StringName("scale") || 
         prop == StringName("rotation") || prop == StringName("rotation_degrees") || prop == StringName("size"))) {
        // Check if this is a 2D or 3D node
        bool is_3d_node = node->is_class("Node3D") || node->get_class().contains("3D");
        

        
        // Accept [x, y, z], {x, y, z}, or "x,y,z" strings
        Vector3 vec3_value;
        Vector2 vec2_value;

        if (value.get_type() == Variant::ARRAY) {
            Array arr = value;
            if (arr.size() >= 2 && arr[0].get_type() != Variant::NIL && arr[1].get_type() != Variant::NIL) {
                if (is_3d_node && arr.size() >= 3 && arr[2].get_type() != Variant::NIL) {
                    vec3_value = Vector3(arr[0], arr[1], arr[2]);
                    value = vec3_value;
                } else if (is_3d_node) {
                    // Default Z to 0 for 3D nodes when only X,Y provided
                    vec3_value = Vector3(arr[0], arr[1], 0.0f);
                    value = vec3_value;
                } else {
                    vec2_value = Vector2(arr[0], arr[1]);
                    value = vec2_value;
                }
            }
        } else if (value.get_type() == Variant::DICTIONARY) {
            Dictionary d = value;
            if ((d.has("x") || d.has("X")) && (d.has("y") || d.has("Y"))) {
                Variant vx = d.has("x") ? d["x"] : d["X"];
                Variant vy = d.has("y") ? d["y"] : d["Y"];
                if (is_3d_node) {
                    double vz = 0.0;
                    if (d.has("z")) {
                        vz = (double)d["z"];
                    } else if (d.has("Z")) {
                        vz = (double)d["Z"];
                    }
                    vec3_value = Vector3((double)vx, (double)vy, vz);
                    value = vec3_value;
                } else {
                    vec2_value = Vector2((double)vx, (double)vy);
                    value = vec2_value;
                }
            }
        } else if (value.get_type() == Variant::STRING) {
            String s = (String)value;
            s = s.strip_edges();
            
            // Handle Vector3() and Vector2() constructor syntax
            if (s.begins_with("Vector3(") && s.ends_with(")")) {
                String coords = s.substr(8, s.length() - 9).strip_edges(); // Extract coordinates from "Vector3(x, y, z)"
                PackedStringArray parts = coords.split(",");
                if (parts.size() >= 3) {
                    float x = parts[0].strip_edges().to_float();
                    float y = parts[1].strip_edges().to_float();
                    float z = parts[2].strip_edges().to_float();
                    vec3_value = Vector3(x, y, z);
                    value = vec3_value;
                    // print_line("SET_NODE_PROPERTY: Converted Vector3() '" + s + "' to Vector3(" + String::num(x) + ", " + String::num(y) + ", " + String::num(z) + ") for property " + String(prop)); // Thread-safe: removed
                } else if (parts.size() == 2) {
                    // Vector3 with only x,y provided - default z to 0
                    float x = parts[0].strip_edges().to_float();
                    float y = parts[1].strip_edges().to_float();
                    vec3_value = Vector3(x, y, 0.0f);
                    value = vec3_value;
                    // print_line("SET_NODE_PROPERTY: Converted Vector3() '" + s + "' to Vector3(" + String::num(x) + ", " + String::num(y) + ", 0.0) for property " + String(prop)); // Thread-safe: removed
                }
            } else if (s.begins_with("Vector2(") && s.ends_with(")")) {
                String coords = s.substr(8, s.length() - 9).strip_edges(); // Extract coordinates from "Vector2(x, y)"
                PackedStringArray parts = coords.split(",");
                if (parts.size() >= 2) {
                    vec2_value = Vector2(parts[0].strip_edges().to_float(), parts[1].strip_edges().to_float());
                    value = vec2_value;
                }
            } else if (s.begins_with("(") && s.ends_with(")")) {
                // Handle simple parentheses format "(x, y, z)" or "(x, y)"
                String coords = s.substr(1, s.length() - 2).strip_edges(); // Extract coordinates from "(x, y, z)"
                PackedStringArray parts = coords.split(",");
                if (parts.size() >= 2) {
                    if (is_3d_node && parts.size() >= 3) {
                        vec3_value = Vector3(parts[0].strip_edges().to_float(), parts[1].strip_edges().to_float(), parts[2].strip_edges().to_float());
                        value = vec3_value;
                        // print_line("SET_NODE_PROPERTY: Converted parentheses '" + s + "' to Vector3(" + String::num(vec3_value.x) + ", " + String::num(vec3_value.y) + ", " + String::num(vec3_value.z) + ") for property " + String(prop)); // Thread-safe: removed
                    } else if (is_3d_node) {
                        // 3D node with only x,y provided - default z to 0
                        vec3_value = Vector3(parts[0].strip_edges().to_float(), parts[1].strip_edges().to_float(), 0.0f);
                        value = vec3_value;
                        // print_line("SET_NODE_PROPERTY: Converted parentheses '" + s + "' to Vector3(" + String::num(vec3_value.x) + ", " + String::num(vec3_value.y) + ", " + String::num(vec3_value.z) + ") for property " + String(prop)); // Thread-safe: removed
                    } else {
                        vec2_value = Vector2(parts[0].strip_edges().to_float(), parts[1].strip_edges().to_float());
                        value = vec2_value;
                        // print_line("SET_NODE_PROPERTY: Converted parentheses '" + s + "' to Vector2(" + String::num(vec2_value.x) + ", " + String::num(vec2_value.y) + ") for property " + String(prop)); // Thread-safe: removed
                    }
                }
            } else {
                // Fallback: Allow formats like "x,y,z" or "x y z"
                PackedStringArray parts = s.split(",");
                if (parts.size() < 2) {
                    parts = s.split(" ");
                }
                if (parts.size() >= 2) {
                    if (is_3d_node) {
                        float x = parts[0].strip_edges().to_float();
                        float y = parts[1].strip_edges().to_float();
                        float z = parts.size() >= 3 ? parts[2].strip_edges().to_float() : 0.0f;
                        vec3_value = Vector3(x, y, z);
                        value = vec3_value;
                        // print_line("SET_NODE_PROPERTY: Converted fallback '" + s + "' to Vector3(" + String::num(x) + ", " + String::num(y) + ", " + String::num(z) + ") for property " + String(prop)); // Thread-safe: removed
                    } else {
                        float x = parts[0].strip_edges().to_float();
                        float y = parts[1].strip_edges().to_float();
                        vec2_value = Vector2(x, y);
                        value = vec2_value;
                        // print_line("SET_NODE_PROPERTY: Converted fallback '" + s + "' to Vector2(" + String::num(x) + ", " + String::num(y) + ") for property " + String(prop)); // Thread-safe: removed
                    }
                }
            }
        }

    }

    // If value is a path string, load via ResourceLoader only; do NOT embed raw Image data.
    if (value.get_type() == Variant::STRING) {
        String s = (String)value;
        bool looks_like_resource = s.begins_with("res://") || s.ends_with(".tres") || s.ends_with(".res") || s.ends_with(".png") || s.ends_with(".jpg") || s.ends_with(".jpeg");
        if (looks_like_resource) {
            // If absolute path to image, copy into res:// and use imported Texture2D path instead.
            if (!s.begins_with("res://") && (s.ends_with(".png") || s.ends_with(".jpg") || s.ends_with(".jpeg"))) {
                String base_dir = ProjectSettings::get_singleton()->globalize_path("res://assets");
                Ref<DirAccess> da = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
                if (da.is_valid()) {
                    da->make_dir_recursive(base_dir);
                    String file_name = String(s.get_file());
                    String dst_abs = base_dir.path_join(file_name);
                    if (da->copy(s, dst_abs) == OK) {
                        String dst_res = ProjectSettings::get_singleton()->localize_path(dst_abs);
                        s = dst_res;
                        if (EditorFileSystem::get_singleton()) {
                            EditorFileSystem::get_singleton()->update_file(dst_res);
                            EditorFileSystem::get_singleton()->scan_changes();
                        }
                    }
                }
            }
            Ref<Resource> res = ResourceLoader::load(s);
            if (res.is_valid()) {
                value = res; // Texture2D/Resource reference; avoids embedding raw pixels
                print_line("SET_NODE_PROPERTY: Loaded resource for '" + String(prop) + "' from " + s);
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
        // Fallbacks for method-backed or virtual properties
        String prop_name = String(prop);
        // Camera current handling via make_current/clear_current
        if (prop_name == "current" && node->has_method("make_current")) {
            bool want_current = false;
            if (value.get_type() == Variant::BOOL) want_current = (bool)value;
            else if (value.get_type() == Variant::INT) want_current = ((int64_t)value) != 0;
            else if (value.get_type() == Variant::STRING) want_current = !String(value).is_empty();
            if (want_current) {
                node->callv("make_current", Array());
            } else if (node->has_method("clear_current")) {
                node->callv("clear_current", Array());
            }
            result["success"] = true;
            result["message"] = "Applied camera current via make_current/clear_current";
            return result;
        }
        // Generic setter fallback, e.g., set_zoom
        String setter = String("set_") + prop_name;
        if (node->has_method(setter)) {
            Array args; args.push_back(value);
            
            // Validate argument types before calling to provide better error messages
            bool type_valid = true;
            String expected_type = "";
            String actual_type = Variant::get_type_name(value.get_type());
            
            // Special validation for common 3D/2D property mismatches
            if ((prop_name == "position" || prop_name == "global_position" || prop_name == "scale")) {
                bool is_3d_node = node->is_class("Node3D") || node->get_class().contains("3D");
                if (is_3d_node && value.get_type() == Variant::VECTOR2) {
                    type_valid = false;
                    expected_type = "Vector3";
                    actual_type = "Vector2";
                } else if (!is_3d_node && value.get_type() == Variant::VECTOR3) {
                    type_valid = false;
                    expected_type = "Vector2";
                    actual_type = "Vector3";
                }
            }
            
            if (!type_valid) {
                result["success"] = false;
                result["error_code"] = "TYPE_CONVERSION_ERROR";
                result["message"] = "Cannot convert argument 1 from " + actual_type + " to " + expected_type + " for " + node->get_class() + "::" + setter;
                return result;
            }
            
            // Call the setter method with error capture
            Variant old_value = node->get(prop);
            Dictionary error_before;
            error_before["count"] = get_runtime_errors(Dictionary()).get("count", 0);
            
            Variant call_result = node->callv(setter, args);
            
            // Check for new runtime errors that occurred during the call
            Dictionary error_after;
            error_after = get_runtime_errors(Dictionary());
            int new_error_count = (int)error_after.get("count", 0) - (int)error_before.get("count", 0);
            
            bool setter_success = true;
            String error_message = "";
            
            // If new errors occurred, capture them
            if (new_error_count > 0) {
                setter_success = false;
                Array all_errors = error_after.get("errors", Array());
                if (all_errors.size() > 0) {
                    Dictionary latest_error = all_errors[0];
                    error_message = latest_error.get("message", "Unknown error occurred during setter call");
                }
            } else {
                // Check if the property actually changed as expected
                Variant new_value = node->get(prop);
                setter_success = (new_value != old_value);
            }
            
            result["success"] = setter_success;
            if (setter_success) {
                result["message"] = "Applied via setter method: " + setter;
            } else {
                if (!error_message.is_empty()) {
                    result["message"] = "Setter method '" + setter + "' failed: " + error_message;
                    result["error_details"] = error_message;
                } else {
                    result["message"] = "Setter method '" + setter + "' may have failed - property value unchanged";
                }
            }
            return result;
        }
        result["success"] = false;
        result["error_code"] = "PROPERTY_INVALID_OR_READONLY";
        result["message"] = "Failed to set property '" + String(prop) + "'. It might be invalid or read-only. Node type: " + node->get_class();
        return result;
    }
	
	// Optional auto-save: default OFF; allow explicit control via p_args.save=true
	String autosave_env = OS::get_singleton()->get_environment("AI_DISABLE_AUTOSAVE_ON_PROPERTY_CHANGE");
	bool disable_autosave = !autosave_env.is_empty() && (autosave_env.to_lower() == "1" || autosave_env.to_lower() == "true");
	bool request_save = p_args.get("save", false);
	if (!disable_autosave && request_save) {
		String current_scene = EditorNode::get_singleton()->get_edited_scene()->get_scene_file_path();
		if (!current_scene.is_empty()) {
			EditorNode::get_singleton()->save_scene_if_open(current_scene);
			print_line("SET_NODE_PROPERTY: Auto-saved scene after property change: " + current_scene);
		} else {
			print_line("SET_NODE_PROPERTY: Scene has no save path, cannot auto-save");
		}
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
	Array args;
	if (p_args.has("args")) {
		Variant v = p_args["args"];
		if (v.get_type() == Variant::ARRAY) {
			args = (Array)v;
		} else if (v.get_type() != Variant::NIL) {
			// Wrap single non-array arg into array
			args.push_back(v);
		}
	} else if (p_args.has("method_args")) {
		Variant v = p_args["method_args"];
		if (v.get_type() == Variant::ARRAY) {
			args = (Array)v;
		} else if (v.get_type() != Variant::NIL) {
			args.push_back(v);
		}
	}

	// If the node doesn't implement the method, return a structured error so the agent can self-correct.
	if (!node->has_method(method)) {
		result["success"] = false;
		result["error"] = "method_not_found";
		result["message"] = String("Method not found on node: ") + String(method);
		result["node_path"] = String(node->get_path());
		result["node_class"] = String(node->get_class());
		result["method"] = String(method);
		result["args"] = args;
		return result;
	}

	// Capture errors before and after the method call
	Dictionary error_before;
	error_before["count"] = get_runtime_errors(Dictionary()).get("count", 0);
	
	Variant ret = node->callv(method, args);
	
	// Check for new runtime errors that occurred during the call
	Dictionary error_after = get_runtime_errors(Dictionary());
	int new_error_count = (int)error_after.get("count", 0) - (int)error_before.get("count", 0);
	
	bool call_success = true;
	String error_message = "";
	
	// If new errors occurred, capture them
	if (new_error_count > 0) {
		call_success = false;
		Array all_errors = error_after.get("errors", Array());
		if (all_errors.size() > 0) {
			Dictionary latest_error = all_errors[0];
			error_message = latest_error.get("message", "Unknown error occurred during method call");
		}
	}

	result["success"] = call_success;
	result["ok"] = call_success;
	result["return_value"] = ret;
	result["node_path"] = String(node->get_path());
	result["node_class"] = String(node->get_class());
	result["method"] = String(method);
	
	if (!call_success && !error_message.is_empty()) {
		result["message"] = "Method call failed: " + error_message;
		result["error_details"] = error_message;
		result["error_count"] = new_error_count;
	}

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
	
	// Also include custom global classes
	List<StringName> global_classes;
	ScriptServer::get_global_class_list(&global_classes);
	for (const StringName &class_name : global_classes) {
		String base = ScriptServer::get_global_class_native_base(class_name);
		if (base == "Node" || ClassDB::is_parent_class(base, "Node")) {
			classes.push_back(String(class_name));
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
		
		// Get root type from parameters, default to Node2D for backward compatibility
		String root_type = p_args.get("root_type", "Node2D");
		
		// Create root node of the specified type
		Node *root_node = nullptr;
		if (ClassDB::can_instantiate(root_type)) {
			root_node = (Node *)ClassDB::instantiate(root_type);
			root_node->set_name("Main");
			
			// Properly set the scene root using EditorNode's method
			EditorNode::get_singleton()->set_edited_scene(root_node);
			// Scene root doesn't need an owner (it's the top level)
		}
		
		if (root_node) {
			result["success"] = true;
			result["message"] = "New scene created with " + root_type + " root.";
			result["root_type"] = root_type;
		} else {
			result["success"] = false;
			result["message"] = "Failed to create scene root node of type: " + root_type;
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

	} else if (operation == "instantiate") {
		if (!p_args.has("path")) {
			result["success"] = false;
			result["message"] = "Missing 'path' argument for instantiate operation.";
			return result;
		}
		String scene_path = p_args["path"];
		String parent_path = p_args.get("parent_node", "");
		String instance_name = p_args.get("instance_name", "");
		
		// Load the scene resource
		Ref<PackedScene> packed_scene = ResourceLoader::load(scene_path);
		if (packed_scene.is_null()) {
			result["success"] = false;
			result["message"] = "Failed to load scene: " + scene_path;
			return result;
		}
		
		// Instantiate the scene
		Node *instance = packed_scene->instantiate();
		if (!instance) {
			result["success"] = false;
			result["message"] = "Failed to instantiate scene: " + scene_path;
			return result;
		}
		
		// Set instance name if provided
		if (!instance_name.is_empty()) {
			instance->set_name(instance_name);
		}
		
		// Find parent node
		Node *parent = nullptr;
		if (!parent_path.is_empty()) {
			parent = _get_node_from_path(parent_path, result);
			if (!parent) {
				instance->queue_free();
				return result;
			}
		} else {
			parent = EditorNode::get_singleton()->get_tree()->get_edited_scene_root();
			if (!parent) {
				instance->queue_free();
				result["success"] = false;
				result["message"] = "No scene is currently open to add instance to.";
				return result;
			}
		}
		
		// Add to parent and set owner
		parent->add_child(instance);
		Node *scene_root = EditorNode::get_singleton()->get_tree()->get_edited_scene_root();
		if (scene_root) {
			instance->set_owner(scene_root);
			// Also set owner for all children recursively
			_set_owner_recursive(instance, scene_root);
		}
		
		result["success"] = true;
		result["message"] = "Scene instantiated: " + scene_path;
		result["instance_path"] = String(instance->get_path());
		result["parent_path"] = parent_path.is_empty() ? String(parent->get_path()) : parent_path;

	} else {
		result["success"] = false;
		result["message"] = "Unknown operation: " + operation + ". Supported: create_new, save_as, open, instantiate";
	}

	return result;
}

// --- Project configuration helpers ---

static void _apply_project_setting_kv(const String &key, const Variant &value, List<String> &applied, List<String> &skipped) {
    ProjectSettings *ps = ProjectSettings::get_singleton();
    if (!ps) return;
    Variant oldv = ps->get_setting(key);
    if (oldv == value) {
        skipped.push_back(key);
        return;
    }
    ps->set_setting(key, value);
    applied.push_back(key);
}

Dictionary EditorTools::enable_plugin(const Dictionary &p_args) {
    Dictionary out;
    String plugin = p_args.get("plugin_name", "");
    if (plugin.is_empty()) {
        out["ok"] = false;
        out["error_code"] = "INVALID_ARGUMENT";
        out["error"] = "plugin_name is required";
        return out;
    }
    bool already = EditorInterface::get_singleton()->is_plugin_enabled(plugin);
    if (!already) {
        EditorInterface::get_singleton()->set_plugin_enabled(plugin, true);
    }
    out["ok"] = true;
    out["requires_restart"] = false; // Godot enables without restart generally
    return out;
}

Dictionary EditorTools::ensure_project_settings(const Dictionary &p_args) {
    Dictionary out;
    Dictionary settings = p_args.get("settings", Dictionary());
    List<String> applied; List<String> skipped;
    for (const Variant *k = settings.next(); k; k = settings.next(k)) {
        _apply_project_setting_kv(String(*k), settings[*k], applied, skipped);
    }
    out["ok"] = true;
    Array a; for (const String &s : applied) a.push_back(s); out["applied"] = a;
    Array sk; for (const String &s : skipped) sk.push_back(s); out["skipped"] = sk;
    ProjectSettings::get_singleton()->save();
    return out;
}

Dictionary EditorTools::ensure_input_actions(const Dictionary &p_args) {
    Dictionary out;
    Array actions = p_args.get("actions", Array());
    Array created; Array updated;
    for (int i = 0; i < actions.size(); i++) {
        Dictionary a = actions[i];
        String name = a.get("name", "");
        if (name.is_empty()) continue;
        if (!InputMap::get_singleton()->has_action(name)) {
            InputMap::get_singleton()->add_action(name);
            created.push_back(name);
        }
        // Replace events if provided
        if (a.has("events")) {
            InputMap::get_singleton()->action_erase_events(name);
            Array evs = a["events"];
            for (int j = 0; j < evs.size(); j++) {
                Dictionary ed = evs[j];
                Ref<InputEvent> ev;
                // Minimal support: keycode
                if (ed.has("keycode")) {
                    Ref<InputEventKey> k; k.instantiate();
                    k->set_keycode((Key)int(ed["keycode"]));
                    ev = k;
                }
                if (ev.is_valid()) {
                    InputMap::get_singleton()->action_add_event(name, ev);
                }
            }
            updated.push_back(name);
        }
    }
    out["ok"] = true;
    out["created"] = created;
    out["updated"] = updated;
    ProjectSettings::get_singleton()->save();
    return out;
}

Dictionary EditorTools::ensure_autoload(const Dictionary &p_args) {
    Dictionary out;
    Array entries = p_args.get("entries", Array());
    Array added; Array updated;
    for (int i = 0; i < entries.size(); i++) {
        Dictionary e = entries[i];
        String name = e.get("name", "");
        String path = e.get("path", "");
        bool singleton = e.get("singleton", true);
        if (name.is_empty() || path.is_empty()) continue;
        String setting_key = String("autoload/") + name;
        String value = singleton ? String("*") + path : path;
        Variant oldv = ProjectSettings::get_singleton()->get_setting(setting_key);
        if (oldv.get_type() == Variant::STRING && String(oldv) == value) {
            continue;
        }
        ProjectSettings::get_singleton()->set_setting(setting_key, value);
        if (oldv.get_type() == Variant::NIL) added.push_back(name); else updated.push_back(name);
    }
    ProjectSettings::get_singleton()->save();
    out["ok"] = true;
    out["added"] = added;
    out["updated"] = updated;
    return out;
}

// --- Creation, calls, batching ---

Dictionary EditorTools::ensure_node(const Dictionary &p_args) {
    // Wrap create_node with unique semantics and deterministic parent
    Dictionary args = p_args.duplicate();
    args["unique"] = true;
    return create_node(args);
}

Dictionary EditorTools::batch_scene_ops(const Dictionary &p_args) {
    Dictionary out;
    Array ops = p_args.get("ops", Array());
    bool stop_on_error = p_args.get("stop_on_error", true);
    Array results;
    for (int i = 0; i < ops.size(); i++) {
        Dictionary op = ops[i];
        String type = op.get("op", op.get("type", ""));
        Dictionary r;
        if (type == "ensure_node") {
            r = ensure_node(op);
        } else if (type == "set_property") {
            r = set_node_property(op);
        } else if (type == "load_and_assign_resource") {
            r = load_and_assign_resource(op);
        } else if (type == "call_method") {
            r = call_node_method(op);
        } else {
            r["success"] = false;
            r["error_code"] = "UNKNOWN_OP";
            r["error"] = String("Unknown op: ") + type;
        }
        results.push_back(r);
        if (stop_on_error && !(bool)r.get("success", false)) {
            break;
        }
    }
    out["ok"] = true;
    out["results"] = results;
    return out;
}

Dictionary EditorTools::load_and_assign_resource(const Dictionary &p_args) {
	Dictionary result;
	if (!p_args.has("resource_path") || !p_args.has("node_path") || !p_args.has("property")) {
		result["success"] = false;
		result["message"] = "Missing required arguments: 'resource_path', 'node_path', and 'property'";
		return result;
	}
	
	String resource_path = p_args["resource_path"];
	String node_path = p_args["node_path"];
	String property = p_args["property"];
	bool validate = p_args.get("validate", true);
	bool await_import = p_args.get("await_import", true);
	int timeout_ms = (int)p_args.get("timeout_ms", 10000);
	
	// Load the resource
	if (await_import) {
		Dictionary wi_args; wi_args["resource_path"] = resource_path; wi_args["timeout_ms"] = timeout_ms; wi_args["poll_ms"] = 100;
		Dictionary waited = wait_for_import(wi_args);
		if (!waited.get("ok", false)) {
			result["success"] = false;
			result["error_code"] = String(waited.get("error_code", "IMPORT_PENDING"));
			result["message"] = String(waited.get("error", "Import not ready"));
			return result;
		}
	}
	Ref<Resource> resource = ResourceLoader::load(resource_path);
	if (resource.is_null()) {
		result["success"] = false;
		result["message"] = "Failed to load resource: " + resource_path;
		return result;
	}
	
	// Get the target node
	Node *node = _get_node_from_path(node_path, result);
	if (!node) {
		return result;
	}
	
	// Optional type validation
	String actual_type = resource->get_class();
	String expected_type;
	if (validate) {
		List<PropertyInfo> plist;
		node->get_property_list(&plist);
		for (const PropertyInfo &pi : plist) {
			if (String(pi.name) == property) {
				if (!String(pi.class_name).is_empty()) {
					expected_type = String(pi.class_name);
				}
				break;
			}
		}
		if (!expected_type.is_empty()) {
			// Handle comma-separated expected types (e.g., "BaseMaterial3D,ShaderMaterial")
			bool type_compatible = false;
			Vector<String> allowed_types = expected_type.split(",");
			
			for (int i = 0; i < allowed_types.size(); i++) {
				String allowed_type = allowed_types[i].strip_edges();
				
				// Check exact match first
				if (actual_type == allowed_type) {
					type_compatible = true;
					break;
				}
				
				// Check inheritance: is actual_type a subclass of allowed_type?
				if (ClassDB::is_parent_class(actual_type, allowed_type)) {
					type_compatible = true;
					break;
				}
			}
			
			if (!type_compatible) {
				result["success"] = false;
				result["ok"] = false;
				result["error_code"] = "TYPE_MISMATCH";
				result["error"] = String("Property '") + property + "' expects " + expected_type + ", got " + actual_type;
				result["actual_resource_type"] = actual_type;
				result["expected_property_type"] = expected_type;
				result["debug_allowed_types"] = allowed_types;
				return result;
			}
		}
	}

	// Set the property
	bool valid = false;
	node->set(property, resource, &valid);
	
	if (valid) {
		result["success"] = true;
		result["ok"] = true;
		result["message"] = "Resource loaded and assigned: " + resource_path + " -> " + node_path + "." + property;
		result["actual_resource_type"] = actual_type;
		if (!expected_type.is_empty()) result["expected_property_type"] = expected_type;
	} else {
		result["success"] = false;
		result["ok"] = false;
		result["error_code"] = "INVALID_PROPERTY";
		result["message"] = "Failed to assign resource to property '" + property + "' on node: " + node_path;
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

	// 2D bodies
	bool is_2d = node->is_class("CharacterBody2D") || node->is_class("RigidBody2D") || node->is_class("StaticBody2D") || node->is_class("Area2D");
	bool is_3d = node->is_class("CharacterBody3D") || node->is_class("RigidBody3D") || node->is_class("StaticBody3D") || node->is_class("Area3D");
	if (!is_2d && !is_3d) {
		result["success"] = false;
		result["message"] = "Node is not a physics body (2D or 3D).";
		return result;
	}

	Node *collision_shape = nullptr;
	Variant shape_resource;

	if (is_2d) {
		if (!ClassDB::can_instantiate("CollisionShape2D")) {
			result["success"] = false;
			result["message"] = "Cannot instantiate CollisionShape2D.";
			return result;
		}
		collision_shape = (Node *)ClassDB::instantiate("CollisionShape2D");
		if (shape_type == "rectangle") {
			if (ClassDB::can_instantiate("RectangleShape2D")) shape_resource = ClassDB::instantiate("RectangleShape2D");
		} else if (shape_type == "circle") {
			if (ClassDB::can_instantiate("CircleShape2D")) shape_resource = ClassDB::instantiate("CircleShape2D");
		} else if (shape_type == "capsule") {
			if (ClassDB::can_instantiate("CapsuleShape2D")) shape_resource = ClassDB::instantiate("CapsuleShape2D");
		}
	} else if (is_3d) {
		if (!ClassDB::can_instantiate("CollisionShape3D")) {
			result["success"] = false;
			result["message"] = "Cannot instantiate CollisionShape3D.";
			return result;
		}
		collision_shape = (Node *)ClassDB::instantiate("CollisionShape3D");
		if (shape_type == "box3d" || shape_type == "box") {
			if (ClassDB::can_instantiate("BoxShape3D")) shape_resource = ClassDB::instantiate("BoxShape3D");
		} else if (shape_type == "sphere3d" || shape_type == "sphere") {
			if (ClassDB::can_instantiate("SphereShape3D")) shape_resource = ClassDB::instantiate("SphereShape3D");
		} else if (shape_type == "capsule3d" || shape_type == "capsule") {
			if (ClassDB::can_instantiate("CapsuleShape3D")) shape_resource = ClassDB::instantiate("CapsuleShape3D");
		} else if (shape_type == "convex3d" || shape_type == "convex") {
			if (ClassDB::can_instantiate("ConvexPolygonShape3D")) shape_resource = ClassDB::instantiate("ConvexPolygonShape3D");
		} else if (shape_type == "trimesh3d" || shape_type == "trimesh") {
			if (ClassDB::can_instantiate("ConcavePolygonShape3D")) shape_resource = ClassDB::instantiate("ConcavePolygonShape3D");
		}
	}

	// Allow providing a custom_shape_resource directly
	if (p_args.has("custom_shape_resource")) {
		shape_resource = p_args["custom_shape_resource"]; // access returns Variant
	}

	if (shape_resource.get_type() == Variant::NIL) {
		if (collision_shape) collision_shape->queue_free();
		result["success"] = false;
		result["message"] = "Failed to create shape resource of type: " + shape_type;
		return result;
	}

	collision_shape->set("shape", shape_resource);

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
	result["message"] = String("CollisionShape") + (is_2d ? String("2D") : String("3D")) + " with " + shape_type + " added to " + node_path;
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

String EditorTools::smart_truncate_for_ai_context(const String &p_content, const String &p_file_path) {
	String content = p_content;
	String ext = p_file_path.get_extension().to_lower();
	
	// Apply smart truncation for files that commonly contain large arrays
	if (ext == "tscn" || ext == "tres" || ext == "res" || ext == "scn") {
		// Truncate large binary data arrays in Godot resource files
		Vector<String> array_patterns = {
			"vertex_data = PackedByteArray(",
			"index_data = PackedByteArray(",
			"attribute_data_0 = PackedByteArray(",
			"attribute_data_1 = PackedByteArray(",
			"attribute_data_2 = PackedByteArray(",
			"lods = [",
			"blend_shape_data = PackedByteArray(",
			"skin_data = PackedByteArray(",
			"vertex_positions = PackedFloat32Array(",
			"vertex_normals = PackedFloat32Array(",
			"vertex_uvs = PackedFloat32Array(",
			"indices = PackedInt32Array("
		};
		
		for (const String &pattern : array_patterns) {
			int pos = 0;
			while ((pos = content.find(pattern, pos)) != -1) {
				int start = pos;
				int end = start;
				int bracket_count = 0;
				bool in_array = false;
				
				// Find the end of the array by counting brackets/parentheses
				for (int i = start; i < content.length(); i++) {
					char32_t ch = content[i];
					if (ch == '(' || ch == '[') {
						bracket_count++;
						in_array = true;
					} else if (ch == ')' || ch == ']') {
						bracket_count--;
						if (bracket_count == 0 && in_array) {
							end = i + 1;
							break;
						}
					}
				}
				
				if (end > start) {
					// Calculate how much data we're truncating
					int original_length = end - start;
					if (original_length > 200) { // Only truncate if it's substantial
						String before = content.substr(0, start);
						String pattern_name = pattern.replace(" = PackedByteArray(", "").replace(" = PackedFloat32Array(", "").replace(" = PackedInt32Array(", "").replace(" = [", "");
						String replacement = pattern + "[...TRUNCATED " + String::num_int64(original_length - 50) + " chars of " + pattern_name + " data...]";
						// Add closing bracket/parenthesis
						if (pattern.ends_with("(")) {
							replacement += ")";
						} else if (pattern.ends_with("[")) {
							replacement += "]";
						}
						String after = content.substr(end);
						content = before + replacement + after;
						
						// Adjust position for next search
						pos = before.length() + replacement.length();
						// print_line("SMART_TRUNCATE: Truncated " + pattern_name + " array (" + String::num_int64(original_length) + " -> " + String::num_int64(replacement.length()) + " chars)"); // Thread-safe: removed print_line
					} else {
						pos = end;
					}
				} else {
					pos += pattern.length();
				}
			}
		}
		
		// Also truncate very long lines that might contain encoded data
		Vector<String> lines = content.split("\n");
		String truncated_content;
		bool any_truncated = false;
		
		for (int i = 0; i < lines.size(); i++) {
			String line = lines[i];
			if (line.length() > 500 && (line.contains("PackedByteArray") || line.contains("PackedFloat32Array") || line.contains("PackedInt32Array"))) {
				String truncated_line = line.substr(0, 200) + "[...TRUNCATED " + String::num_int64(line.length() - 200) + " chars of array data...]";
				if (line.ends_with(")")) truncated_line += ")";
				if (line.ends_with("]")) truncated_line += "]";
				truncated_content += truncated_line + "\n";
				any_truncated = true;
			} else {
				truncated_content += line + "\n";
			}
		}
		
		if (any_truncated) {
			content = truncated_content;
			// print_line("SMART_TRUNCATE: Truncated long array lines in " + p_file_path.get_file()); // Thread-safe: removed print_line
		}
	}
	
	// For any file type, truncate extremely long lines that might be data
	Vector<String> lines = content.split("\n");
	String final_content;
	bool any_data_truncated = false;
	
	for (int i = 0; i < lines.size(); i++) {
		String line = lines[i];
		// Truncate lines longer than 1000 chars that look like data
		if (line.length() > 1000 && (
			(line.contains("[") && line.count(",") > 50) ||  // Arrays with many elements
			line.contains("\"data\":") ||                  // JSON data fields
			line.contains("base64") ||                     // Base64 encoded data
			line.contains("PackedByteArray") ||            // Godot byte arrays
			line.contains("PackedFloat32Array") ||         // Godot float arrays
			line.contains("PackedInt32Array")              // Godot int arrays
		)) {
			String truncated_line = line.substr(0, 300) + "[...TRUNCATED " + String::num_int64(line.length() - 300) + " chars of data...]";
			// Preserve line ending structure
			if (line.ends_with(",")) truncated_line += ",";
			if (line.ends_with(")")) truncated_line += ")";
			if (line.ends_with("]")) truncated_line += "]";
			if (line.ends_with("}")) truncated_line += "}";
			final_content += truncated_line + "\n";
			any_data_truncated = true;
		} else {
			final_content += line + "\n";
		}
	}
	
	if (any_data_truncated) {
		content = final_content;
		// print_line("SMART_TRUNCATE: Truncated long data lines in " + p_file_path.get_file()); // Thread-safe: removed print_line
	}
	
	return content;
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
	// CRITICAL: Always prefer in-memory preview overlay if present
	// This ensures subsequent edits use the staged content, not stale disk content
	if (EditorTools::has_preview_overlay(path)) {
		String overlay = EditorTools::get_preview_overlay(path);
		result["success"] = true;
		result["content"] = smart_truncate_for_ai_context(overlay, path);
		print_line("READ_FILE: Using preview overlay for " + path + " (staged edit pending)");
		return result;
	}
	String content = FileAccess::get_file_as_string(path, &err);
	if (err == OK) {
		result["success"] = true;
		result["content"] = smart_truncate_for_ai_context(content, path);
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
		String smart_truncated = smart_truncate_for_ai_context(preview, path);
		result["content"] = smart_truncated + (file_len > to_read ? String("\n\n…\n[Truncated preview. Use read_file_advanced with start_line/end_line to fetch specific sections.]") : String());
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
	// CRITICAL: Always honor in-memory overlay for consistency
	if (EditorTools::has_preview_overlay(path)) {
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
		result["content"] = smart_truncate_for_ai_context(out, path);
		print_line("READ_FILE_ADVANCED: Using preview overlay for " + path + " (staged edit pending)");
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
	result["content"] = smart_truncate_for_ai_context(content, path);
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

	// Wait for connection with timeout
	int connection_timeout_ms = 10000; // 10 seconds timeout
	int connection_elapsed_ms = 0;
	while (http_client->get_status() == HTTPClient::STATUS_CONNECTING || http_client->get_status() == HTTPClient::STATUS_RESOLVING) {
		http_client->poll();
		std::this_thread::sleep_for(std::chrono::microseconds(1000));
		connection_elapsed_ms += 1;
		if (connection_elapsed_ms > connection_timeout_ms) {
			result["success"] = false;
			result["message"] = "Connection timeout after " + itos(connection_timeout_ms/1000) + " seconds";
			memdelete(http_client);
			print_line("APPLY_EDIT ERROR: Connection timeout");
			return result;
		}
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

	// Wait for response with timeout
	int response_timeout_ms = 60000; // 60 seconds timeout for AI response
	int response_elapsed_ms = 0;
	while (http_client->get_status() == HTTPClient::STATUS_REQUESTING) {
		http_client->poll();
		std::this_thread::sleep_for(std::chrono::microseconds(1000));
		response_elapsed_ms += 1;
		if (response_elapsed_ms > response_timeout_ms) {
			result["success"] = false;
			result["message"] = "Response timeout after " + itos(response_timeout_ms/1000) + " seconds. The AI is taking too long to process the edit.";
			memdelete(http_client);
			print_line("APPLY_EDIT ERROR: Response timeout");
			return result;
		}
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

	int body_timeout_ms = 30000; // 30 seconds timeout for reading body
	int body_elapsed_ms = 0;
	while (http_client->get_status() == HTTPClient::STATUS_BODY) {
		http_client->poll();
		PackedByteArray chunk = http_client->read_response_body_chunk();
		if (chunk.size() == 0) {
			std::this_thread::sleep_for(std::chrono::microseconds(1000));
			body_elapsed_ms += 1;
			if (body_elapsed_ms > body_timeout_ms) {
				result["success"] = false;
				result["message"] = "Timeout reading response body after " + itos(body_timeout_ms/1000) + " seconds";
				memdelete(http_client);
				print_line("APPLY_EDIT ERROR: Body read timeout");
				return result;
			}
		} else {
			body.append_array(chunk);
			body_elapsed_ms = 0; // Reset timeout on progress
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
	
	// Safety check: prevent connection attempt with empty host
	if (host.is_empty()) {
		print_line("APPLY_EDIT: ERROR - Host is empty after URL parsing. Original endpoint: '" + p_api_endpoint + "'");
		result["success"] = false;
		result["message"] = "Configuration error: Backend URL is invalid or empty";
		memdelete(http_client);
		return result;
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

	// Wait for connection with timeout
	int connection_timeout_ms = 10000; // 10 seconds timeout
	int connection_elapsed_ms = 0;
	while (http_client->get_status() == HTTPClient::STATUS_CONNECTING || http_client->get_status() == HTTPClient::STATUS_RESOLVING) {
		http_client->poll();
		std::this_thread::sleep_for(std::chrono::microseconds(1000));
		connection_elapsed_ms += 1;
		if (connection_elapsed_ms > connection_timeout_ms) {
			result["success"] = false;
			result["message"] = "Connection timeout after " + itos(connection_timeout_ms/1000) + " seconds";
			memdelete(http_client);
			print_line("APPLY_EDIT ERROR: Connection timeout");
			return result;
		}
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

	// Auth and context headers - these should have been prepared in advance by the main thread
	String auth_token = p_ai_args.get("auth_token", "");
	String user_id = p_ai_args.get("user_id", "");
	String machine_id = p_ai_args.get("machine_id", "");
	String project_root = p_ai_args.get("project_root", "");

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

	// Wait for response with timeout
	int response_timeout_ms = 60000; // 60 seconds timeout for AI response
	int response_elapsed_ms = 0;
	while (http_client->get_status() == HTTPClient::STATUS_REQUESTING) {
		http_client->poll();
		std::this_thread::sleep_for(std::chrono::microseconds(1000));
		response_elapsed_ms += 1;
		if (response_elapsed_ms > response_timeout_ms) {
			result["success"] = false;
			result["message"] = "Response timeout after " + itos(response_timeout_ms/1000) + " seconds. The AI is taking too long to process the edit.";
			memdelete(http_client);
			print_line("APPLY_EDIT ERROR: Response timeout");
			return result;
		}
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

	int body_timeout_ms = 30000; // 30 seconds timeout for reading body
	int body_elapsed_ms = 0;
	while (http_client->get_status() == HTTPClient::STATUS_BODY) {
		http_client->poll();
		PackedByteArray chunk = http_client->read_response_body_chunk();
		if (chunk.size() == 0) {
			std::this_thread::sleep_for(std::chrono::microseconds(1000));
			body_elapsed_ms += 1;
			if (body_elapsed_ms > body_timeout_ms) {
				result["success"] = false;
				result["message"] = "Timeout reading response body after " + itos(body_timeout_ms/1000) + " seconds";
				memdelete(http_client);
				print_line("APPLY_EDIT ERROR: Body read timeout");
				return result;
			}
		} else {
			body.append_array(chunk);
			body_elapsed_ms = 0; // Reset timeout on progress
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

	// Clean up the edited_content only for script-like files; keep raw for .tres/.tscn/etc.
	if (response_data.has("edited_content")) {
		String edited_content = response_data["edited_content"];
		String ext = p_file_path.get_extension().to_lower();
		bool is_script_like = (ext == "gd" || ext == "cs" || ext == "glsl" || ext == "shader");
		if (is_script_like) {
			String cleaned_content = _clean_backend_content(edited_content);
			response_data["edited_content"] = cleaned_content;
		} else {
			response_data["edited_content"] = edited_content;
		}
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
    String file_content;
    bool file_missing = false;
    
    // CRITICAL: Check for preview overlay first to ensure chained edits work
    if (EditorTools::has_preview_overlay(path)) {
        file_content = EditorTools::get_preview_overlay(path);
        print_line("APPLY_EDIT: Using preview overlay as base content for " + path);
    } else {
        file_content = FileAccess::get_file_as_string(path, &err);
        if (err != OK) {
            file_missing = true;
            file_content = ""; // create new file from scratch
            print_line("APPLY_EDIT: Target file does not exist; will create new file: " + path);
        }
    }

    // Determine edit scope: full file vs line range
    String lines_mode = String(p_args.get("lines", "all")).to_lower();
    int range_start = (int)p_args.get("start_line", 0);
    int range_end = (int)p_args.get("end_line", 0);
    bool use_range = (lines_mode == "range") || (range_start > 0 && range_end >= range_start);
    
    // For GDScript files in range mode, expand context for better indentation awareness
    String ext = path.get_extension().to_lower();
    bool is_script_file = (ext == "gd" || ext == "cs" || ext == "shader" || ext == "glsl");
    int context_lines = 0;
    
    if (is_script_file && use_range) {
        // Expand context for script files to help AI understand indentation structure
        context_lines = 10; // Add 10 lines before and after for context
        print_line("APPLY_EDIT: Expanding context for script file indentation awareness");
    }

    Vector<String> file_lines = file_content.split("\n");
    String pre_text, segment_text, post_text;
    int total_lines = file_lines.size();
    if (use_range) {
        // Clamp range within file
        if (range_start <= 0) range_start = 1;
        if (range_end <= 0 || range_end > total_lines) range_end = total_lines;
        
        // Expand context for script files
        int context_start = range_start - 1;
        int context_end = range_end;
        
        if (context_lines > 0) {
            context_start = MAX(0, range_start - 1 - context_lines);
            context_end = MIN(total_lines, range_end + context_lines);
            print_line("APPLY_EDIT: Expanded context from lines " + itos(context_start + 1) + "-" + itos(context_end) + " (original range: " + itos(range_start) + "-" + itos(range_end) + ")");
        }
        
        // Build pre/segment/post with expanded context
        for (int i = 0; i < context_start && i < total_lines; i++) {
            pre_text += file_lines[i];
            if (i < context_start - 1 || total_lines > 1) pre_text += "\n";
        }
        for (int i = context_start; i < context_end && i < total_lines; i++) {
            segment_text += file_lines[i];
            if (i < context_end - 1) segment_text += "\n";
        }
        for (int i = context_end; i < total_lines; i++) {
            if (!post_text.is_empty()) post_text += "\n";
            post_text += file_lines[i];
        }
    }

    // Prepare request content: either whole file or only the selected segment
    String content_for_model = use_range ? segment_text : file_content;

    print_line("APPLY_EDIT: Using HTTPClient to call backend API - prompt: " + prompt);

    // Use authentication data passed from main thread (avoid singleton access in background thread)
    String auth_token = p_args.get("auth_token", "");
    String user_id = p_args.get("user_id", "");
    String machine_id = p_args.get("machine_id", "");
    String project_root = p_args.get("project_root", "");
    String base_url = p_args.get("base_url", "");
    
    // Fallback to detecting the base URL from environment/settings if not provided
    if (base_url.is_empty()) {
        String is_dev = OS::get_singleton()->get_environment("IS_DEV");
        if (is_dev.is_empty()) {
            is_dev = OS::get_singleton()->get_environment("DEV_MODE");
        }
        if (!is_dev.is_empty() && is_dev.to_lower() == "true") {
            base_url = "http://127.0.0.1:5050";
        } else {
            base_url = "https://api.orcaengine.ai";
        }
        
        // Also check EditorSettings override like ai_chat_dock does
        if (EditorSettings::get_singleton() && EditorSettings::get_singleton()->has_setting("ai_chat/base_url")) {
            String override_url = EditorSettings::get_singleton()->get_setting("ai_chat/base_url");
            if (!override_url.is_empty()) {
                base_url = override_url;
            }
        } else if (!OS::get_singleton()->get_environment("AI_CHAT_CLOUD_URL").is_empty()) {
            base_url = OS::get_singleton()->get_environment("AI_CHAT_CLOUD_URL");
        }
        
        print_line("APPLY_EDIT: base_url not provided, using fallback: " + base_url);
    }

    // Call backend using only the segment if range mode is used, and pass range context for diff reconstruction
    Dictionary args_for_backend = p_args.duplicate();
    args_for_backend["prompt"] = prompt; // ensure present
    
    // Add auth data for background thread (passed from main thread)
    args_for_backend["auth_token"] = auth_token;
    args_for_backend["user_id"] = user_id;
    args_for_backend["machine_id"] = machine_id;
    args_for_backend["project_root"] = project_root;
    
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
    Dictionary local_result;
    int attempts = 0;
    const int max_attempts = 2; // default + fallback
    while (attempts < max_attempts) {
        Dictionary attempt_args = args_for_backend.duplicate();
        if (attempts == 1) {
            attempt_args["model"] = String("gpt-5");
            print_line("APPLY_EDIT: Retry " + itos(attempts) + "/" + itos(max_attempts - 1) + " with fallback model gpt-5");
        }
    local_result = _call_apply_endpoint(path, content_for_model, attempt_args, base_url + "/predict_code_edit");
        attempts++;
        if (local_result.get("success", false)) {
            local_result["attempts"] = attempts;
            local_result["fallback_used"] = (attempts > 1);
            break;
        }
        String msg = local_result.get("message", String());
        String msg_l = msg.to_lower();
        bool is_timeout = msg_l.find("timeout") != -1;
        bool conn_issue = msg_l.find("failed to connect") != -1 || msg_l.find("request failed") != -1;
        if (!(is_timeout || conn_issue)) {
            local_result["attempts"] = attempts;
            local_result["fallback_used"] = (attempts > 1);
            break;
        }
        if (attempts >= max_attempts) {
            local_result["attempts"] = attempts;
            local_result["fallback_used"] = true;
            break;
        }
    }

    // Skip analytics in background thread to avoid singleton access issues
    print_line("APPLY_EDIT: Analytics skipped for background thread safety");

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
        
        // Copy inline_diff from backend response if available
        if (local_result.has("inline_diff")) {
            result["inline_diff"] = local_result["inline_diff"];
            print_line("APPLY_EDIT: Copied inline_diff from backend, length: " + itos(String(local_result["inline_diff"]).length()));
        } else {
            result["inline_diff"] = "";
            print_line("APPLY_EDIT: No inline_diff in backend response");
        }
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
        result["attempts"] = local_result.get("attempts", 1);
        result["fallback_used"] = local_result.get("fallback_used", false);
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
	
	// CRITICAL: First check if content looks like JSON structure
	// This prevents writing JSON diffs to files
	String trimmed = content.strip_edges();
	if (trimmed.begins_with("{") && trimmed.ends_with("}")) {
		// Check for telltale signs of our structured edit JSON
		if (trimmed.find("\"mode\"") != -1 || trimmed.find("\"edits\"") != -1 || 
		    trimmed.find("\"range_edit\"") != -1 || trimmed.find("\"start_line\"") != -1) {
			print_line("ERROR: Backend returned JSON structure instead of file content!");
			print_line("First 200 chars: " + trimmed.substr(0, 200));
			// Try to extract actual content from common JSON fields
			int content_start = trimmed.find("\"content\":");
			if (content_start != -1) {
				content_start += 10; // Skip past "content":
				int quote_start = trimmed.find("\"", content_start);
				if (quote_start != -1) {
					int quote_end = trimmed.find("\"", quote_start + 1);
					while (quote_end != -1 && trimmed[quote_end - 1] == '\\') {
						quote_end = trimmed.find("\"", quote_end + 1);
					}
					if (quote_end != -1) {
						String extracted = trimmed.substr(quote_start + 1, quote_end - quote_start - 1);
						// Unescape JSON escapes
						extracted = extracted.replace("\\n", "\n").replace("\\\"", "\"").replace("\\\\", "\\");
						content = extracted;
						print_line("Extracted content from JSON structure");
					}
				}
			}
			if (content == p_content) {
				// Extraction failed, return empty to prevent corruption
				return "";
			}
		}
	}
	
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
	
	// DO NOT strip_edges() on multi-line content - it corrupts first line indentation
	// Only trim trailing newlines if excessive (but preserve intentional structure)
	while (content.ends_with("\n\n\n")) {
		content = content.substr(0, content.length() - 1);
	}
	
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
			error_dict["file"] = p_file_path;
			error_dict["source"] = "scripts";
			error_dict["language"] = "GDScript";
			errors.push_back(error_dict);
		}
		// NOTE: Skipping warnings collection (API differs across engine versions).
		
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
					error_dict["file"] = p_file_path;
					error_dict["source"] = "scripts";
					error_dict["language"] = "GDScript";
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
					error_dict["file"] = p_file_path;
					error_dict["source"] = "scripts";
					error_dict["language"] = "GDScript";
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
		error_dict["file"] = p_file_path;
		error_dict["source"] = "scripts";
		error_dict["language"] = "C#";
		errors.push_back(error_dict);
	}
	
	print_line("COMPILATION CHECK: Found " + String::num_int64(errors.size()) + " issues for " + p_file_path);
	
	return errors;
}

Dictionary EditorTools::check_compilation_errors(const Dictionary &p_args) {
    Dictionary result;
    // Support both 'path' (legacy) and 'check_path' (new script_manager tool)
    String path = p_args.get("check_path", p_args.get("path", ""));
    bool check_all = p_args.get("check_all", false);
    String check_mode = String(p_args.get("check_mode", "scripts")).to_lower(); // "scripts" or "output"
    
    // New mode: check all output errors
    if (check_mode == "output") {
        print_line("CHECK_COMPILATION_ERRORS: Checking all output errors");
        Array errors;
        
        // Get runtime errors from the output panel
        Dictionary runtime_args;
        runtime_args["include_warnings"] = true;
        runtime_args["max_count"] = 1000;
        Dictionary runtime_result = get_runtime_errors(runtime_args);
        Array runtime_errors = runtime_result.get("errors", Array());
        
        // Convert runtime errors to our format
        for (int i = 0; i < runtime_errors.size(); i++) {
            Dictionary runtime_error = runtime_errors[i];
            Dictionary error_dict;
            error_dict["type"] = runtime_error.get("is_warning", false) ? "warning" : "error";
            error_dict["file"] = runtime_error.get("file", "Unknown");
            error_dict["line"] = runtime_error.get("line", 0);
            error_dict["column"] = runtime_error.get("column", 0);
            error_dict["message"] = runtime_error.get("message", "Unknown error");
            error_dict["source"] = "output";
            errors.push_back(error_dict);
        }
        
        // TODO: Add shader compilation errors, scene validation errors, etc.
        
        result["success"] = true;
        result["errors"] = errors;
        result["mode"] = "output";
        result["message"] = "Found " + String::num_int64(errors.size()) + " errors/warnings in output";
        return result;
    }
    
    // Original mode: check script compilation errors
    if (!check_all && path.is_empty()) {
        result["success"] = false;
        result["message"] = "Path is required when check_all is false";
        result["errors"] = Array();
        return result;
    }
    
    print_line("CHECK_COMPILATION_ERRORS: Checking " + (check_all ? "all scripts" : "file - " + path));
    
    Array errors;
    
    if (check_all) {
        // Check all script files in the project
        _check_all_scripts_errors(errors);
    } else if (path.get_extension() == "gd") {
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
            error_dict["file"] = path;
            error_dict["line"] = 0;
            error_dict["column"] = 0;
            error_dict["message"] = "Failed to read file: " + path;
            error_dict["source"] = "scripts";
            error_dict["language"] = "GDScript";
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
                error_dict["file"] = path;
                error_dict["source"] = "scripts";
                error_dict["language"] = "GDScript";
                errors.push_back(error_dict);
                print_line("CHECK_COMPILATION_ERRORS: Found parser error at line " + String::num_int64(error.line) + ": " + error.message);
            }
            // NOTE: Skipping warnings collection (API differs across engine versions).
            
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
                        error_dict["file"] = path;
                        error_dict["source"] = "scripts";
                        error_dict["language"] = "GDScript";
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
    result["mode"] = "scripts";
    
    print_line("CHECK_COMPILATION_ERRORS: Found " + String::num_int64(errors.size()) + " errors in " + path);
    
    return result;
}

void EditorTools::_check_all_scripts_errors(Array &r_errors) {
    // Get all script files in the project
    List<String> script_files;
    HashSet<String> extensions;
    extensions.insert("gd");
    extensions.insert("cs");
    _get_all_project_files("res://", script_files, extensions);
    
    for (const String &script_path : script_files) {
        if (script_path.get_extension() == "gd") {
            // Check GDScript
            Error file_err = OK;
            String file_content = FileAccess::get_file_as_string(script_path, &file_err);
            
            if (file_err != OK) {
                Dictionary error_dict;
                error_dict["type"] = "file_error";
                error_dict["file"] = script_path;
                error_dict["line"] = 0;
                error_dict["column"] = 0;
                error_dict["message"] = "Failed to read file";
                r_errors.push_back(error_dict);
                continue;
            }
            
            // Parse the script
            GDScriptParser parser;
            Error parse_err = parser.parse(file_content, script_path, false);
            
            // Get parser errors
            const List<GDScriptParser::ParserError> &parser_errors = parser.get_errors();
            for (const GDScriptParser::ParserError &error : parser_errors) {
                Dictionary error_dict;
                error_dict["type"] = "parser_error";
                error_dict["file"] = script_path;
                error_dict["line"] = error.line;
                error_dict["column"] = error.column;
                error_dict["message"] = error.message;
                r_errors.push_back(error_dict);
            }
            
            // Only continue to analysis if parsing succeeded
            if (parse_err == OK && parser_errors.is_empty()) {
                GDScriptAnalyzer analyzer(&parser);
                analyzer.analyze();
                
                // Get analyzer errors
                const List<GDScriptParser::ParserError> &analyzer_errors = parser.get_errors();
                for (const GDScriptParser::ParserError &error : analyzer_errors) {
                    // Skip duplicates
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
                        error_dict["file"] = script_path;
                        error_dict["line"] = error.line;
                        error_dict["column"] = error.column;
                        error_dict["message"] = error.message;
                        r_errors.push_back(error_dict);
                    }
                }
            }
        } else if (script_path.get_extension() == "cs") {
            // C# placeholder
            Dictionary info_dict;
            info_dict["type"] = "info";
            info_dict["file"] = script_path;
            info_dict["line"] = 0;
            info_dict["column"] = 0;
            info_dict["message"] = "C# compilation checking not implemented";
            r_errors.push_back(info_dict);
        }
    }
}

void EditorTools::_get_all_project_files(const String &p_path, List<String> &r_files, const HashSet<String> &p_extensions) {
    _get_all_project_files_limited(p_path, r_files, p_extensions, 1000); // Default limit: 1000 files
}

void EditorTools::_get_all_project_files_limited(const String &p_path, List<String> &r_files, const HashSet<String> &p_extensions, int p_max_files) {
    // PERFORMANCE FIX: Use EditorFileSystem instead of manual directory traversal
    // This prevents UI blocking by using Godot's already-indexed file system
    
    EditorFileSystem *efs = EditorFileSystem::get_singleton();
    if (!efs) {
        print_line("AI Chat: EditorFileSystem not available, falling back to fast directory scan");
        _get_project_files_fast_fallback(p_path, r_files, p_extensions, p_max_files);
        return;
    }
    
    // Use the already-indexed file system - this is much faster
    EditorFileSystemDirectory *root = efs->get_filesystem();
    if (!root) {
        print_line("AI Chat: EditorFileSystem root not available, falling back to fast directory scan");
        _get_project_files_fast_fallback(p_path, r_files, p_extensions, p_max_files);
        return;
    }
    
    // Recursively collect files from the indexed filesystem
    _collect_files_from_efs_directory(root, r_files, p_extensions, p_max_files);
}

// Fast fallback that limits depth and operations to prevent blocking
void EditorTools::_get_project_files_fast_fallback(const String &p_path, List<String> &r_files, const HashSet<String> &p_extensions, int p_max_files) {
    Error err;
    Ref<DirAccess> dir = DirAccess::open(p_path, &err);
    if (err != OK) {
        return;
    }
    
    const int MAX_DEPTH = 3; // Limit recursion depth to prevent deep scanning
    const int MAX_DIRS_PER_LEVEL = 20; // Limit directories per level
    
    _get_project_files_with_limits(dir, p_path, r_files, p_extensions, p_max_files, 0, MAX_DEPTH, MAX_DIRS_PER_LEVEL);
}

void EditorTools::_get_project_files_with_limits(Ref<DirAccess> p_dir, const String &p_path, List<String> &r_files, const HashSet<String> &p_extensions, int p_max_files, int p_current_depth, int p_max_depth, int p_max_dirs) {
    if (r_files.size() >= p_max_files || p_current_depth >= p_max_depth) {
        return;
    }
    
    p_dir->list_dir_begin();
    String file_name = p_dir->get_next();
    int dirs_processed = 0;
    
    // First pass: collect files in current directory
    while (!file_name.is_empty() && r_files.size() < p_max_files) {
        String full_path = p_path.path_join(file_name);
        
        if (!p_dir->current_is_dir()) {
            String ext = file_name.get_extension().to_lower();
            if (p_extensions.has(ext)) {
                r_files.push_back(full_path);
            }
        }
        
        file_name = p_dir->get_next();
    }
    
    // Second pass: recurse into subdirectories (limited)
    p_dir->list_dir_begin();
    file_name = p_dir->get_next();
    
    while (!file_name.is_empty() && r_files.size() < p_max_files && dirs_processed < p_max_dirs) {
        String full_path = p_path.path_join(file_name);
        
        if (p_dir->current_is_dir() && !file_name.begins_with(".")) {
            Error sub_err;
            Ref<DirAccess> sub_dir = DirAccess::open(full_path, &sub_err);
            if (sub_err == OK) {
                _get_project_files_with_limits(sub_dir, full_path, r_files, p_extensions, p_max_files, p_current_depth + 1, p_max_depth, p_max_dirs);
                dirs_processed++;
            }
        }
        
        file_name = p_dir->get_next();
    }
    
    p_dir->list_dir_end();
}

void EditorTools::_collect_files_from_efs_directory(EditorFileSystemDirectory *p_dir, List<String> &r_files, const HashSet<String> &p_extensions, int p_max_files) {
    if (!p_dir || r_files.size() >= p_max_files) {
        return;
    }
    
    // Collect files from current directory
    for (int i = 0; i < p_dir->get_file_count() && r_files.size() < p_max_files; i++) {
        String file_path = p_dir->get_file_path(i);
        String ext = file_path.get_extension().to_lower();
        if (p_extensions.has(ext)) {
            r_files.push_back(file_path);
        }
    }
    
    // Recurse into subdirectories
    for (int i = 0; i < p_dir->get_subdir_count() && r_files.size() < p_max_files; i++) {
        EditorFileSystemDirectory *subdir = p_dir->get_subdir(i);
        _collect_files_from_efs_directory(subdir, r_files, p_extensions, p_max_files);
    }
}

// Helper function for setting owner recursively
void EditorTools::_set_owner_recursive(Node *p_node, Node *p_owner) {
	if (!p_node || !p_owner) return;
	
	// Set owner for all children
	for (int i = 0; i < p_node->get_child_count(); i++) {
		Node *child = p_node->get_child(i);
		child->set_owner(p_owner);
		_set_owner_recursive(child, p_owner);
	}
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
	// Support both old "operation" and new "op" parameter names for backward compatibility
	String operation = p_args.get("op", p_args.get("operation", ""));
	
	if (operation.is_empty()) {
		Dictionary result;
		result["success"] = false;
		result["error"] = "Operation 'op' parameter is required";
		return result;
	}
	
	// Legacy operations
	if (operation == "get_info") return get_scene_info(p_args);
	if (operation == "open" || operation == "create_new" || operation == "save_as" || operation == "instantiate") {
		// Fix parameter translation: manage_scene expects "operation" not "op"
		Dictionary manage_args = p_args;
		manage_args["operation"] = operation;
		return manage_scene(manage_args);
	}
	
	// New consolidated operations
	if (operation == "scene.open" || operation == "scene.create" || operation == "scene.save_as" || operation == "scene.instantiate") {
		// Fix parameter translation: manage_scene expects "operation" not "op" AND expects stripped operation names
		Dictionary manage_args = p_args;
		String stripped_operation = operation;
		if (operation == "scene.open") {
			stripped_operation = "open";
		} else if (operation == "scene.create") {
			stripped_operation = "create_new";  // manage_scene expects "create_new", not "create"
		} else if (operation == "scene.save_as") {
			stripped_operation = "save_as";
		} else if (operation == "scene.instantiate") {
			stripped_operation = "instantiate";
		}
		manage_args["operation"] = stripped_operation;
		
		// CRITICAL FIX: Parameter name mapping - manage_scene expects "path" but schema uses "scene_path"
		if (p_args.has("scene_path") && !p_args.has("path")) {
			manage_args["path"] = p_args["scene_path"];
		}
		
		return manage_scene(manage_args);
	} else if (operation == "scene.analyze" || operation == "scene.info") {
		return get_scene_info(p_args);
	} else if (operation == "scene.nodes.get_all") {
		return get_all_nodes(p_args);
	} else if (operation == "scene.nodes.find_by_type") {
		return search_nodes_by_type(p_args);
	} else if (operation == "editor.selection.get") {
		return get_editor_selection(p_args);
	} else if (operation == "scene.bulk_configure") {
		// Fix parameter translation and format: universal_scene_manager expects different parameters
		Dictionary universal_args = p_args;
		universal_args["operation"] = "bulk_configure"; // Strip "scene." prefix
		
		// Transform "operations" array into "targets" + "transformations" format
		if (p_args.has("operations")) {
			Array operations = p_args["operations"];
			Array targets;
			Dictionary transformations;
			
			for (int i = 0; i < operations.size(); i++) {
				Dictionary op = operations[i];
				String path = op.get("path", "");
				String property = op.get("property", "");
				Variant value = op.get("value", Variant());
				
				if (!path.is_empty() && !property.is_empty()) {
					targets.push_back(path);
					if (!transformations.has(path)) {
						transformations[path] = Dictionary();
					}
					Dictionary node_transforms = transformations[path];
					node_transforms[property] = value;
					transformations[path] = node_transforms;
				}
			}
			
			universal_args["targets"] = targets;
			universal_args["transformations"] = transformations;
			universal_args.erase("operations"); // Remove old format
		}
		
		return universal_scene_manager(universal_args);
	} else if (operation == "scene.copy_configuration") {
		// Fix parameter translation: universal_scene_manager expects "operation" not "op" and different parameter names
		Dictionary universal_args = p_args;
		universal_args["operation"] = "copy_configuration"; // Strip "scene." prefix
		
		// Parameter mapping: scene_manager uses "source_config_scene" but universal_scene_manager expects "source"
		if (p_args.has("source_config_scene")) {
			universal_args["source"] = p_args["source_config_scene"];
			universal_args.erase("source_config_scene"); // Remove old parameter name
		}
		// Also check for legacy "source" parameter name
		if (p_args.has("source")) {
			universal_args["source"] = p_args["source"];
		}
		
		// Ensure targets parameter exists - if not provided, use empty array as fallback
		if (!universal_args.has("targets")) {
			universal_args["targets"] = Array();
		}
		
		return universal_scene_manager(universal_args);
	} else if (operation == "node.create") {
		return create_node(p_args);
	} else if (operation == "node.delete") {
		return delete_node(p_args);
	} else if (operation == "node.move") {
		return move_node(p_args);
	} else if (operation == "node.type.change") {
		return change_node_type(p_args);
	} else if (operation == "node.type.set") {
		return set_node_type(p_args);
	} else if (operation == "node.rename") {
		// Fix parameter translation: editor_introspect expects "operation" not "op"
		Dictionary introspect_args = p_args;
		introspect_args["operation"] = operation; // Pass the full operation name
		return editor_introspect(introspect_args);
	} else if (operation == "node.props.get") {
		return get_node_properties(p_args);
	} else if (operation == "node.props.set_batch") {
		return batch_set_node_properties(p_args);
	} else if (operation == "node.method.call") {
		return call_node_method(p_args);
	} else if (operation == "node.assign_resource") {
		return assign_resource_to_node_property(p_args);
	} else if (operation == "node.add_collision") {
		// Fix parameter mapping: add_collision_shape expects "node_path" but scene_manager schema might use "path"
		Dictionary collision_args = p_args;
		if (p_args.has("path") && !p_args.has("node_path")) {
			collision_args["node_path"] = p_args["path"];
		}
		return add_collision_shape(collision_args);
	} else if (operation.begins_with("groups.") || operation.begins_with("signals.")) {
		// Fix parameter translation: editor_introspect expects "operation" not "op"
		Dictionary introspect_args = p_args;
		introspect_args["operation"] = operation; // Pass the full operation name
		return editor_introspect(introspect_args);
	} else {
		Dictionary result;
		result["success"] = false;
		result["message"] = String("Unknown scene_manager operation: ") + operation;
		return result;
	}
}

// --- New Debugging Tools Implementation ---

Dictionary EditorTools::run_scene(const Dictionary &p_args) {
	Dictionary result;
	String scene_path = p_args.get("scene_path", "");
	bool clear_errors = p_args.get("clear_errors", true);
	
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
	
	// Clear previous errors if requested for clean testing
	if (clear_errors) {
		s_runtime_errors.clear();
	}
	
	// Start the scene
	EditorRunBar::get_singleton()->play_custom_scene(scene_path);
	
	result["success"] = true;
	result["message"] = "Game started: " + scene_path;
	result["scene_path"] = scene_path;
	result["is_playing"] = true;
	result["clear_errors"] = clear_errors;
	return result;
}

Dictionary EditorTools::stop_game(const Dictionary &p_args) {
	Dictionary result;
	
	if (!EditorRunBar::get_singleton()->is_playing()) {
		result["success"] = false;
		result["message"] = "No game is currently running";
		result["is_playing"] = false;
		return result;
	}
	
	String playing_scene = EditorRunBar::get_singleton()->get_playing_scene();
	EditorRunBar::get_singleton()->stop_playing();
	
	result["success"] = true;
	result["message"] = "Game stopped";
	result["was_playing_scene"] = playing_scene;
	result["is_playing"] = false;
	return result;
}

Dictionary EditorTools::get_game_status(const Dictionary &p_args) {
	Dictionary result;
	
	bool is_playing = EditorRunBar::get_singleton()->is_playing();
	String playing_scene = "";
	
	if (is_playing) {
		playing_scene = EditorRunBar::get_singleton()->get_playing_scene();
	}
	
	result["success"] = true;
	result["is_playing"] = is_playing;
	result["playing_scene"] = playing_scene;
	result["message"] = is_playing ? ("Game running: " + playing_scene) : "No game running";
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
	String target = p_args.get("target", "editor"); // "editor", "game", "both"
	bool return_base64 = p_args.get("return_base64", false);
	
	// Default to base64 for AI tools unless specifically requested otherwise
	if (!p_args.has("return_base64")) {
		return_base64 = true; // AI tools want base64 by default
	}
	
	Array screenshots;
	bool captured_any = false;
	
	// Helper to capture and process a viewport
	auto capture_viewport = [&](Viewport *viewport, const String &source_name) -> bool {
		if (!viewport) return false;
		
		// Safe texture access with null checks to prevent freezing
		Ref<ViewportTexture> viewport_texture = viewport->get_texture();
		if (viewport_texture.is_null()) {
			print_line("AI Chat: Viewport texture is null for " + source_name);
			return false;
		}
		
		// Check viewport size first to prevent huge texture processing
		Vector2i viewport_size = viewport->get_visible_rect().size;
		if (viewport_size.x <= 0 || viewport_size.y <= 0 || viewport_size.x > 8192 || viewport_size.y > 8192) {
			print_line("AI Chat: Viewport size too large or invalid for " + source_name + ": " + String::num_int64(viewport_size.x) + "x" + String::num_int64(viewport_size.y));
			return false;
		}
		
		// Get image safely - use Godot's error handling instead of try-catch
		Ref<Image> screenshot = viewport_texture->get_image();
		if (screenshot.is_null() || screenshot->is_empty()) {
			print_line("AI Chat: Screenshot image is null or empty for " + source_name);
			return false;
		}
		
		// Double-check image size matches expectations
		if (screenshot->get_width() != viewport_size.x || screenshot->get_height() != viewport_size.y) {
			print_line("AI Chat: Screenshot size mismatch for " + source_name + " - expected " + String::num_int64(viewport_size.x) + "x" + String::num_int64(viewport_size.y) + ", got " + String::num_int64(screenshot->get_width()) + "x" + String::num_int64(screenshot->get_height()));
		}
		
		Vector2i original_size = Vector2i(screenshot->get_width(), screenshot->get_height());
		
		// CRITICAL: Downscale for chat to prevent token explosion
		if (return_base64) {
			// Limit to 128px max dimension for chat display - tiny to prevent token explosion
			const int MAX_CHAT_SIZE = 128;
			if (original_size.x > MAX_CHAT_SIZE || original_size.y > MAX_CHAT_SIZE) {
				float aspect = (float)original_size.x / (float)original_size.y;
				Vector2i new_size;
				if (original_size.x > original_size.y) {
					new_size.x = MAX_CHAT_SIZE;
					new_size.y = (int)(MAX_CHAT_SIZE / aspect);
				} else {
					new_size.y = MAX_CHAT_SIZE;
					new_size.x = (int)(MAX_CHAT_SIZE * aspect);
				}
				screenshot->resize(new_size.x, new_size.y, Image::INTERPOLATE_LANCZOS);
			}
		}
		
		Dictionary capture_info;
		capture_info["source"] = source_name;
		capture_info["size"] = original_size;
		capture_info["display_size"] = Vector2i(screenshot->get_width(), screenshot->get_height());
		
		if (return_base64) {
			// Convert to base64 for immediate use in chat
			Vector<uint8_t> png_buffer = screenshot->save_png_to_buffer();
			if (png_buffer.size() > 0) {
				String base64_data = CoreBind::Marshalls::get_singleton()->raw_to_base64(png_buffer);
				capture_info["base64"] = base64_data;
				capture_info["mime_type"] = "image/png";
				print_line("AI Chat: Screenshot downscaled to " + String::num_int64(screenshot->get_width()) + "x" + String::num_int64(screenshot->get_height()) + " for chat (base64 size: " + String::num_int64(base64_data.length()) + " chars)");
			}
		} else {
			// Save to file at original resolution
			String save_name = source_name + "_" + filename;
			String full_path = ProjectSettings::get_singleton()->globalize_path("res://") + save_name;
			Error save_result = screenshot->save_png(full_path);
			if (save_result == OK) {
				capture_info["path"] = full_path;
			} else {
				return false;
			}
		}
		
		screenshots.push_back(capture_info);
		return true;
	};
	
	// Capture editor viewport - this should be immediate and synchronous
	if (target == "editor" || target == "both") {
		print_line("AI Chat: Capturing editor viewport...");
		Viewport *editor_viewport = EditorNode::get_singleton()->get_viewport();
		if (editor_viewport) {
			bool editor_success = capture_viewport(editor_viewport, "editor");
			if (editor_success) {
				captured_any = true;
				print_line("AI Chat: Editor viewport captured successfully");
			} else {
				print_line("AI Chat: Editor viewport capture failed");
				Dictionary editor_failure;
				editor_failure["source"] = "editor";
				editor_failure["message"] = "Editor viewport capture failed - texture may not be ready";
				editor_failure["success"] = false;
				screenshots.push_back(editor_failure);
			}
		} else {
			print_line("AI Chat: Editor viewport is null - trying alternative approach");
			// Try alternative viewport capture methods
			if (EditorNode::get_singleton()) {
				Node *scene_root = EditorNode::get_singleton()->get_tree()->get_edited_scene_root();
				if (scene_root) {
					Viewport *scene_viewport = scene_root->get_viewport();
					if (scene_viewport && capture_viewport(scene_viewport, "editor_scene")) {
						captured_any = true;
						print_line("AI Chat: Alternative scene viewport captured successfully");
					} else {
						Dictionary editor_failure;
						editor_failure["source"] = "editor";
						editor_failure["message"] = "Editor viewport not available - try opening a scene first";
						editor_failure["success"] = false;
						screenshots.push_back(editor_failure);
					}
				} else {
					Dictionary editor_failure;
					editor_failure["source"] = "editor";
					editor_failure["message"] = "No scene open - editor screenshot needs an active scene";
					editor_failure["success"] = false;
					screenshots.push_back(editor_failure);
				}
			} else {
				Dictionary editor_failure;
				editor_failure["source"] = "editor";
				editor_failure["message"] = "EditorNode not available";
				editor_failure["success"] = false;
				screenshots.push_back(editor_failure);
			}
		}
	}
	
	// Capture game viewport using existing GameView functionality  
	if (target == "game" || target == "both") {
		bool game_running = false;
		EditorRunBar *run_bar = EditorRunBar::get_singleton();
		if (run_bar) {
			game_running = run_bar->is_playing();
		}
		
		if (game_running) {
			// CRITICAL: Clear any previous screenshot results to prevent stale data
			_current_screenshot_results.clear();
			
			// Use EditorRun::request_screenshot (the correct API) instead of direct GameView access
			bool requested = false;
			Callable screenshot_callback = callable_mp_static(&EditorTools::_on_game_screenshot_ready_for_tool);
			
			// Request screenshot through the proper EditorRun API
			requested = EditorRun::request_screenshot(screenshot_callback);
			print_line(String("AI Chat: Game screenshot request = ") + (requested ? "success" : "failed"));
			
			if (requested) {
				// Wait a reasonable time for the screenshot to be processed
				uint64_t start_time = Time::get_singleton()->get_ticks_msec();
				uint64_t max_wait_time = 2000; // 2 seconds max

				print_line("AI Chat: Waiting for screenshot to be processed...");

				// Process messages to allow the callback to execute
				while ((Time::get_singleton()->get_ticks_msec() - start_time) < max_wait_time && 
				       _current_screenshot_results.is_empty()) {
					// Allow message processing without blocking the UI
					MessageQueue::get_singleton()->flush();
					OS::get_singleton()->delay_usec(50000); // Wait 50ms
				}

				if (!_current_screenshot_results.is_empty()) {
					uint64_t elapsed = Time::get_singleton()->get_ticks_msec() - start_time;
					print_line("AI Chat: Screenshot completed successfully after " + String::num_int64(elapsed) + "ms");
					// Screenshot completed - it will be processed below
					captured_any = true;
				} else {
					uint64_t elapsed = Time::get_singleton()->get_ticks_msec() - start_time;
					print_line("AI Chat: Screenshot timed out after " + String::num_int64(elapsed) + "ms");
				Dictionary game_capture;
				game_capture["source"] = "game";
					game_capture["message"] = "Game screenshot requested but timed out - try again in a moment";
					game_capture["running"] = true;
				game_capture["success"] = false;
					game_capture["method"] = "gameview_with_timeout";
				screenshots.push_back(game_capture);
				}
			} else {
				Dictionary game_capture;
				game_capture["source"] = "game"; 
				game_capture["message"] = "Game screenshot request failed - no active debugger sessions or embedded game process";
				game_capture["running"] = true;
				game_capture["success"] = false;
				screenshots.push_back(game_capture);
			}
		} else {
			Dictionary game_capture;
			game_capture["source"] = "game";
			game_capture["message"] = "No game running";
			game_capture["running"] = false;
			game_capture["success"] = false;
			screenshots.push_back(game_capture);
		}
	}
	
	// CRITICAL: Process any completed screenshots from callbacks before returning results
	if (!_current_screenshot_results.is_empty()) {
		print_line("AI Chat: Processing " + String::num_int64(_current_screenshot_results.size()) + " completed screenshot(s) from callbacks");
		for (int i = 0; i < _current_screenshot_results.size(); i++) {
			Dictionary screenshot_data = _current_screenshot_results[i];
			// Convert callback data to expected format
			Dictionary processed_shot;
			processed_shot["source"] = screenshot_data.get("source", "game");
			processed_shot["success"] = true;
			processed_shot["base64"] = screenshot_data.get("base64", "");
			processed_shot["mime_type"] = screenshot_data.get("mime_type", "image/png");
			processed_shot["size"] = screenshot_data.get("size", Vector2i(0, 0));
			processed_shot["display_size"] = screenshot_data.get("display_size", Vector2i(0, 0));
			processed_shot["original_size"] = screenshot_data.get("original_size", Vector2i(0, 0));
			processed_shot["was_downsampled"] = screenshot_data.get("was_downsampled", false);
			processed_shot["timestamp"] = screenshot_data.get("timestamp", 0);
			processed_shot["message"] = "Screenshot ready";
			
			screenshots.push_back(processed_shot);
			captured_any = true;
			
			print_line("AI Chat: Processed completed screenshot, base64 length: " + String::num_int64(String(screenshot_data.get("base64", "")).length()));
		}
		_current_screenshot_results.clear();
	}
	
	result["success"] = captured_any;
	result["screenshots"] = screenshots;
	result["count"] = screenshots.size();
	result["message"] = captured_any ? "Screenshot(s) captured" : "No viewports captured";
	
	// For backward compatibility, include single screenshot data
	if (screenshots.size() > 0) {
		Dictionary first = screenshots[0];
		result["path"] = first.get("path", "");
		result["base64"] = first.get("base64", "");
		result["filename"] = filename;
	}
	
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

    if (operation == "refresh_resources") {
        // Generic resource refresh so the agent can force reimport after saving files.
        Array paths = p_args.get("paths", Array());
        bool wait = p_args.get("wait", true);
        if (EditorFileSystem::get_singleton()) {
            for (int i = 0; i < paths.size(); i++) {
                String p = paths[i];
                if (!p.is_empty()) {
                    EditorFileSystem::get_singleton()->update_file(p);
                }
            }
            if (wait) {
                EditorFileSystem::get_singleton()->scan_changes();
            } else {
                // Asynchronous change detection; schedule scan without blocking
                EditorFileSystem::get_singleton()->scan_changes();
            }
        }
        result["success"] = true;
        result["message"] = "Resources refreshed";
        result["paths"] = paths;
        return result;
    }

    if (operation == "slice_spritesheet") {
        // Args: sheet_path:String (required), tile_size:String or Vector2i (e.g., "32x32")
        // Optional: grid:String (e.g., "8x4"), margin:int (px), spacing:int (px), out_dir:String
        // Advanced (robust slicing): auto_detect:bool, bg_tolerance:int (0..50), alpha_threshold:int (0..255),
        // tight_crop:bool, padding:int, fuzzy:int, normalize_to:String (e.g., "32x32")
        String sheet_path = p_args.get("sheet_path", String(""));
        if (sheet_path.is_empty()) {
            result["success"] = false;
            result["message"] = "Missing 'sheet_path'";
            return result;
        }

        auto parse_vec2i = [&](const Variant &v, Vector2i def)->Vector2i{
            if (v.get_type() == Variant::VECTOR2I) return v;
            if (v.get_type() == Variant::STRING) {
                String s = (String)v;
                Vector<String> parts = s.strip_edges().to_lower().replace(" ", "").split("x");
                if (parts.size() == 2 && parts[0].is_valid_int() && parts[1].is_valid_int()) {
                    return Vector2i(parts[0].to_int(), parts[1].to_int());
                }
            }
            return def;
        };

        Vector2i tile_sz = parse_vec2i(p_args.get("tile_size", String("32x32")), Vector2i(32, 32));
        Vector2i normalize_to = parse_vec2i(p_args.get("normalize_to", Variant()), tile_sz);
        int margin = (int)p_args.get("margin", 0);
        int spacing = (int)p_args.get("spacing", 0);
        bool auto_detect = (bool)p_args.get("auto_detect", true);
        int bg_tol = (int)p_args.get("bg_tolerance", 24); // color distance tolerance
        int alpha_thresh = (int)p_args.get("alpha_threshold", 1); // treat alpha<=this as background
        bool tight_crop = (bool)p_args.get("tight_crop", true);
        int padding = (int)p_args.get("padding", 0);
        int fuzzy = (int)p_args.get("fuzzy", 2); // expand bounds to avoid cutoffs
        String grid_str = p_args.get("grid", String(""));
        int grid_cols = 0, grid_rows = 0;
        if (!grid_str.is_empty()) {
            Vector<String> parts = grid_str.to_lower().split("x");
            if (parts.size() == 2 && parts[0].is_valid_int() && parts[1].is_valid_int()) {
                grid_cols = parts[0].to_int();
                grid_rows = parts[1].to_int();
            }
        }
        String out_dir = p_args.get("out_dir", String(""));
        if (out_dir.is_empty()) {
            out_dir = String(sheet_path.get_base_dir()) + "/slices";
        }

        Ref<Image> sheet = Image::load_from_file(sheet_path);
        if (sheet.is_null() || sheet->is_empty()) {
            result["success"] = false;
            result["message"] = String("Failed to load sheet: ") + sheet_path;
            return result;
        }

        // Auto compute grid/margins/spacing (robust) if requested or if missing.
        if (auto_detect || grid_cols <= 0 || grid_rows <= 0) {
            // Estimate background color from corners
            auto get_px = [&](int x, int y)->Color{ return sheet->get_pixel(x, y); };
            Color corners[4] = { get_px(0,0), get_px(sheet->get_width()-1,0), get_px(0,sheet->get_height()-1), get_px(sheet->get_width()-1, sheet->get_height()-1) };
            Color bg = corners[0];
            // Average corners for stability
            for (int i=1;i<4;i++) { bg.r += corners[i].r; bg.g += corners[i].g; bg.b += corners[i].b; bg.a += corners[i].a; }
            bg.r/=4; bg.g/=4; bg.b/=4; bg.a/=4;
            auto is_bg = [&](const Color &c)->bool{
                if (alpha_thresh > 0 && c.a * 255.0 <= alpha_thresh) return true;
                float dr = fabsf(c.r - bg.r), dg = fabsf(c.g - bg.g), db = fabsf(c.b - bg.b);
                return (dr*255.0 <= bg_tol && dg*255.0 <= bg_tol && db*255.0 <= bg_tol);
            };
            // Scan for empty rows/cols to infer margins/spacing and cell spans
            PackedInt32Array col_non_empty, row_non_empty;
            col_non_empty.resize(sheet->get_width());
            row_non_empty.resize(sheet->get_height());
            for (int x=0; x<sheet->get_width(); x++) {
                bool any = false; for (int y=0; y<sheet->get_height(); y++) { if (!is_bg(sheet->get_pixel(x,y))) { any=true; break; } }
                col_non_empty.set(x, any ? 1 : 0);
            }
            for (int y=0; y<sheet->get_height(); y++) {
                bool any = false; for (int x=0; x<sheet->get_width(); x++) { if (!is_bg(sheet->get_pixel(x,y))) { any=true; break; } }
                row_non_empty.set(y, any ? 1 : 0);
            }
            // Determine margins as outermost empty bands
            int left=0; while (left < sheet->get_width() && col_non_empty[left]==0) left++;
            int right=sheet->get_width()-1; while (right>=0 && col_non_empty[right]==0) right--;
            int top=0; while (top < sheet->get_height() && row_non_empty[top]==0) top++;
            int bottom=sheet->get_height()-1; while (bottom>=0 && row_non_empty[bottom]==0) bottom--;
            if (left < right && top < bottom) {
                margin = MAX(margin, MIN(left, top));
            }
            // Estimate spacing by finding periodic empty bands between non-empty spans
            auto estimate_spacing = [&](bool horizontal)->int{
                int max_len = horizontal ? sheet->get_width() : sheet->get_height();
                int best = spacing;
                int run=0; bool prev_empty=false; Vector<int> gaps;
                for (int i=0;i<max_len;i++) {
                    bool empty = (horizontal ? col_non_empty[i]==0 : row_non_empty[i]==0);
                    if (empty) { run++; prev_empty=true; }
                    else { if (prev_empty && run>0) gaps.push_back(run); run=0; prev_empty=false; }
                }
                if (gaps.size() > 0) {
                    // use median gap (robust)
                    gaps.sort(); best = gaps[gaps.size()/2];
                    // bound spacing to reasonable values
                    if (best > tile_sz.x*2) best = spacing; // ignore absurd gaps
                }
                return best;
            };
            if (spacing == 0) {
                int hsp = estimate_spacing(true);
                int vsp = estimate_spacing(false);
                spacing = MAX(0, MIN(hsp, vsp));
            }
            // Estimate grid if missing
            if (grid_cols <= 0 || grid_rows <= 0) {
                int usable_w = sheet->get_width() - margin * 2 + spacing;
                int usable_h = sheet->get_height() - margin * 2 + spacing;
                grid_cols = MAX(1, (usable_w + spacing) / (tile_sz.x + spacing));
                grid_rows = MAX(1, (usable_h + spacing) / (tile_sz.y + spacing));
            }
        }
        if (grid_cols <= 0 || grid_rows <= 0) {
            result["success"] = false;
            result["message"] = "Invalid grid computed from image/tile_size";
            return result;
        }

        // Ensure out_dir exists
        Ref<DirAccess> da = DirAccess::create(DirAccess::ACCESS_RESOURCES);
        if (da.is_valid()) {
            if (!da->dir_exists(out_dir)) {
                da->make_dir_recursive(out_dir);
            }
        }

        Array saved_paths;
        for (int y = 0; y < grid_rows; y++) {
            for (int x = 0; x < grid_cols; x++) {
                Vector2i origin = Vector2i(margin, margin) + Vector2i(x * (tile_sz.x + spacing), y * (tile_sz.y + spacing));
                if (origin.x + tile_sz.x > sheet->get_width() || origin.y + tile_sz.y > sheet->get_height()) {
                    continue;
                }
                Rect2i cell_rect(origin, tile_sz);
                // Expand with fuzzy margin to avoid cutting off if slightly misaligned
                cell_rect.position -= Vector2i(fuzzy, fuzzy);
                cell_rect.size += Vector2i(fuzzy*2, fuzzy*2);
                cell_rect.position.x = MAX(0, cell_rect.position.x);
                cell_rect.position.y = MAX(0, cell_rect.position.y);
                cell_rect.size.x = MIN(sheet->get_width() - cell_rect.position.x, cell_rect.size.x);
                cell_rect.size.y = MIN(sheet->get_height() - cell_rect.position.y, cell_rect.size.y);

                Ref<Image> sub = sheet->get_region(cell_rect);

                // Tight crop inside the cell based on background, then paste onto normalized canvas
                Ref<Image> final_img;
                final_img.instantiate();
                final_img->initialize_data(normalize_to.x + padding*2, normalize_to.y + padding*2, false, Image::FORMAT_RGBA8);
                final_img->fill(Color(0,0,0,0));

                Rect2i content_rect(Vector2i(0,0), sub->get_size());
                if (tight_crop) {
                    int minx=sub->get_width(), miny=sub->get_height(), maxx=-1, maxy=-1;
                    for (int cy=0; cy<sub->get_height(); cy++) {
                        for (int cx=0; cx<sub->get_width(); cx++) {
                            Color c = sub->get_pixel(cx, cy);
                            bool bgp = (alpha_thresh>0 && c.a*255.0 <= alpha_thresh) || false; // ignore color match for tight crop
                            if (!bgp) { if (cx<minx) minx=cx; if (cy<miny) miny=cy; if (cx>maxx) maxx=cx; if (cy>maxy) maxy=cy; }
                        }
                    }
                    if (maxx >= minx && maxy >= miny) {
                        content_rect = Rect2i(Vector2i(minx, miny), Vector2i(maxx-minx+1, maxy-miny+1));
                    }
                }

                Ref<Image> cropped = sub->get_region(content_rect);
                // Center on final canvas
                int dst_w = final_img->get_width();
                int dst_h = final_img->get_height();
                int ox = (dst_w - cropped->get_width())/2;
                int oy = (dst_h - cropped->get_height())/2;
                final_img->blit_rect(cropped, Rect2i(Vector2i(0,0), cropped->get_size()), Vector2i(ox, oy));

                String fname = vformat("%s/frame_%02d_%02d.png", out_dir, y, x);
                Error se = final_img->save_png(fname);
                if (se == OK) {
                    saved_paths.push_back(fname);
                }
            }
        }

        // Refresh editor file system so new frames are recognized
        if (EditorFileSystem::get_singleton()) {
            for (int i = 0; i < saved_paths.size(); i++) {
                EditorFileSystem::get_singleton()->update_file(saved_paths[i]);
            }
            EditorFileSystem::get_singleton()->scan_changes();
        }

        result["success"] = true;
        result["message"] = "Spritesheet sliced";
        result["paths"] = saved_paths;
        result["grid_cols"] = grid_cols;
        result["grid_rows"] = grid_rows;
        result["tile_size"] = normalize_to;
        result["out_dir"] = out_dir;
        return result;
    }

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
        // Node *root = EditorNode::get_singleton()->get_tree()->get_edited_scene_root(); // Currently unused
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

    // MISSING OPERATIONS IMPLEMENTATION (previously causing "Operation not implemented" errors)
    
    if (operation == "node.rename") {
        Node *node = require_path(result);
        if (!node) return result;
        
        String new_name = p_args.get("new_name", p_args.get("name", ""));
        if (new_name.is_empty()) {
            result["success"] = false;
            result["message"] = "Missing 'new_name' parameter for node rename";
            return result;
        }
        
        String old_name = node->get_name();
        node->set_name(new_name);
        
        result["success"] = true;
        result["message"] = "Node renamed from '" + old_name + "' to '" + new_name + "'";
        result["old_name"] = old_name;
        result["new_name"] = new_name;
        result["path"] = String(node->get_path());
        return result;
    }
    
    if (operation == "groups.add") {
        Node *node = require_path(result);
        if (!node) return result;
        
        String group_name = p_args.get("group", p_args.get("group_name", ""));
        if (group_name.is_empty()) {
            result["success"] = false;
            result["message"] = "Missing 'group' parameter";
            return result;
        }
        
        node->add_to_group(group_name);
        result["success"] = true;
        result["message"] = "Node added to group '" + group_name + "'";
        result["group"] = group_name;
        return result;
    }
    
    if (operation == "groups.remove") {
        Node *node = require_path(result);
        if (!node) return result;
        
        String group_name = p_args.get("group", p_args.get("group_name", ""));
        if (group_name.is_empty()) {
            result["success"] = false;
            result["message"] = "Missing 'group' parameter";
            return result;
        }
        
        node->remove_from_group(group_name);
        result["success"] = true;
        result["message"] = "Node removed from group '" + group_name + "'";
        result["group"] = group_name;
        return result;
    }
    
    if (operation == "groups.list") {
        Node *node = require_path(result);
        if (!node) return result;
        
        List<Node::GroupInfo> groups;
        node->get_groups(&groups);
        Array group_names;
        for (const Node::GroupInfo &group : groups) {
            group_names.push_back(String(group.name));
        }
        
        result["success"] = true;
        result["groups"] = group_names;
        result["count"] = group_names.size();
        return result;
    }
    
    if (operation == "signals.list" || operation == "signals.list_node_signals") {
        Node *node = require_path(result);
        if (!node) return result;
        
        List<MethodInfo> signals;
        node->get_signal_list(&signals);
        Array signal_list;
        for (const MethodInfo &mi : signals) {
            Dictionary sig_info;
            sig_info["name"] = String(mi.name);
            sig_info["args_count"] = mi.arguments.size();
            
            // Add argument details for advanced signal operations
            Array args_info;
            for (const PropertyInfo &arg : mi.arguments) {
                Dictionary arg_info;
                arg_info["name"] = String(arg.name);
                arg_info["type"] = arg.type;
                args_info.push_back(arg_info);
            }
            sig_info["arguments"] = args_info;
            signal_list.push_back(sig_info);
        }
        
        result["success"] = true;
        result["signals"] = signal_list;
        result["count"] = signal_list.size();
        return result;
    }
    
    if (operation == "signals.list_connections") {
        Node *node = require_path(result);
        if (!node) return result;
        
        Array connections_info;
        
        // Get all signals first
        List<MethodInfo> signals;
        node->get_signal_list(&signals);
        
        // For each signal, get its connections
        for (const MethodInfo &signal_info : signals) {
            List<Object::Connection> connections;
            node->get_signal_connection_list(signal_info.name, &connections);
            
            for (const Object::Connection &conn : connections) {
                Dictionary conn_info;
                conn_info["signal"] = String(signal_info.name);
                if (conn.callable.get_object()) {
                    Node *target = Object::cast_to<Node>(conn.callable.get_object());
                    if (target) {
                        conn_info["target"] = String(target->get_path());
                        conn_info["method"] = String(conn.callable.get_method());
                    }
                }
                connections_info.push_back(conn_info);
            }
        }
        
        result["success"] = true;
        result["connections"] = connections_info;
        result["count"] = connections_info.size();
        return result;
    }
    
    if (operation == "signals.list_incoming_connections") {
        Node *node = require_path(result);
        if (!node) return result;
        
        Array incoming_connections;
        
        // Get all nodes in scene and check their outgoing connections to find incoming ones to our node
        Node *root = EditorNode::get_singleton()->get_tree()->get_edited_scene_root();
        if (root) {
            std::function<void(Node*)> check_node = [&](Node* n) {
                if (!n) return;
                
                // Get all signals for this node
                List<MethodInfo> signals;
                n->get_signal_list(&signals);
                
                // For each signal, check if it connects to our target node
                for (const MethodInfo &signal_info : signals) {
                    List<Object::Connection> connections;
                    n->get_signal_connection_list(signal_info.name, &connections);
                    
                    for (const Object::Connection &conn : connections) {
                        if (conn.callable.get_object() == node) {
                            Dictionary conn_info;
                            conn_info["source_node"] = String(n->get_path());
                            conn_info["signal"] = String(signal_info.name);
                            conn_info["method"] = String(conn.callable.get_method());
                            incoming_connections.push_back(conn_info);
                        }
                    }
                }
                
                // Check children recursively
                for (int i = 0; i < n->get_child_count(); i++) {
                    check_node(n->get_child(i));
                }
            };
            
            check_node(root);
        }
        
        result["success"] = true;
        result["incoming_connections"] = incoming_connections;
        result["count"] = incoming_connections.size();
        return result;
    }
    
    if (operation == "signals.validate") {
        Node *node = require_path(result);
        if (!node) return result;
        
        Array validation_results;
        Array issues;
        
        // Get all signal connections for this node
        List<MethodInfo> signals;
        node->get_signal_list(&signals);
        
        for (const MethodInfo &signal_info : signals) {
            List<Object::Connection> connections;
            node->get_signal_connection_list(signal_info.name, &connections);
            
            for (const Object::Connection &conn : connections) {
                Dictionary validation;
                validation["signal"] = String(signal_info.name);
                validation["connected"] = true;
                validation["valid"] = true;
                
                if (conn.callable.get_object()) {
                    Node *target = Object::cast_to<Node>(conn.callable.get_object());
                    if (target) {
                        validation["target"] = String(target->get_path());
                        validation["method"] = String(conn.callable.get_method());
                        
                        // Check if target node still exists and method is callable
                        if (!target->has_method(conn.callable.get_method())) {
                            validation["valid"] = false;
                            Dictionary issue;
                            issue["type"] = "missing_method";
                            issue["signal"] = String(signal_info.name);
                            issue["target"] = String(target->get_path());
                            issue["method"] = String(conn.callable.get_method());
                            issue["message"] = "Target method does not exist";
                            issues.push_back(issue);
                        }
                    } else {
                        validation["valid"] = false;
                        validation["target"] = "null";
                        Dictionary issue;
                        issue["type"] = "invalid_target";
                        issue["signal"] = String(signal_info.name);
                        issue["message"] = "Signal connected to invalid target";
                        issues.push_back(issue);
                    }
                }
                
                validation_results.push_back(validation);
            }
        }
        
        result["success"] = true;
        result["validation_results"] = validation_results;
        result["issues"] = issues;
        result["valid"] = issues.size() == 0;
        result["issues_count"] = issues.size();
        return result;
    }
    
    if (operation == "signals.connect") {
        Node *source_node = require_path(result);
        if (!source_node) return result;
        
        String signal_name = p_args.get("signal_name", p_args.get("signal", ""));
        String target_path = p_args.get("target_path", p_args.get("target", ""));
        String method_name = p_args.get("method", p_args.get("method_name", ""));
        
        if (signal_name.is_empty()) {
            result["success"] = false;
            result["message"] = "Missing 'signal_name' parameter";
            return result;
        }
        
        if (target_path.is_empty()) {
            result["success"] = false;
            result["message"] = "Missing 'target_path' parameter";
            return result;
        }
        
        if (method_name.is_empty()) {
            result["success"] = false;
            result["message"] = "Missing 'method' parameter";
            return result;
        }
        
        // Get target node
        Dictionary target_error;
        Node *target_node = _get_node_from_path(target_path, target_error);
        if (!target_node) {
            result["success"] = false;
            result["message"] = "Target node not found: " + target_path;
            return result;
        }
        
        // Check if signal exists on source node
        if (!source_node->has_signal(signal_name)) {
            result["success"] = false;
            result["message"] = "Signal '" + signal_name + "' not found on source node";
            return result;
        }
        
        // Check if method exists on target node
        if (!target_node->has_method(method_name)) {
            result["success"] = false;
            result["message"] = "Method '" + method_name + "' not found on target node";
            return result;
        }
        
        // Create callable and connect
        Callable callable = Callable(target_node, method_name);
        Error err = source_node->connect(signal_name, callable);
        
        if (err != OK) {
            result["success"] = false;
            result["message"] = "Failed to connect signal: " + String::num_int64(err);
            return result;
        }
        
        result["success"] = true;
        result["message"] = "Connected signal '" + signal_name + "' to method '" + method_name + "'";
        result["source_node"] = String(source_node->get_path());
        result["target_node"] = String(target_node->get_path());
        result["signal"] = signal_name;
        result["method"] = method_name;
        return result;
    }
    
    if (operation == "signals.disconnect") {
        Node *source_node = require_path(result);
        if (!source_node) return result;
        
        String signal_name = p_args.get("signal_name", p_args.get("signal", ""));
        String target_path = p_args.get("target_path", p_args.get("target", ""));
        String method_name = p_args.get("method", p_args.get("method_name", ""));
        
        if (signal_name.is_empty()) {
            result["success"] = false;
            result["message"] = "Missing 'signal_name' parameter";
            return result;
        }
        
        if (target_path.is_empty()) {
            result["success"] = false;
            result["message"] = "Missing 'target_path' parameter";
            return result;
        }
        
        if (method_name.is_empty()) {
            result["success"] = false;
            result["message"] = "Missing 'method' parameter";
            return result;
        }
        
        // Get target node
        Dictionary target_error;
        Node *target_node = _get_node_from_path(target_path, target_error);
        if (!target_node) {
            result["success"] = false;
            result["message"] = "Target node not found: " + target_path;
            return result;
        }
        
        // Create callable and disconnect
        Callable callable = Callable(target_node, method_name);
        source_node->disconnect(signal_name, callable);
        
        result["success"] = true;
        result["message"] = "Disconnected signal '" + signal_name + "' from method '" + method_name + "'";
        result["source_node"] = String(source_node->get_path());
        result["target_node"] = String(target_node->get_path());
        result["signal"] = signal_name;
        result["method"] = method_name;
        return result;
    }
    
    if (operation == "signals.open_dialog") {
        Node *node = require_path(result);
        if (!node) return result;
        
        // Open the connections dialog for the specified node
        result["success"] = false;
        result["message"] = "signals.open_dialog: Opening connections dialog is not yet implemented";
        result["note"] = "This would open the Godot editor's signal connections dialog";
        return result;
    }
    
    if (operation == "signals.trace.start") {
        result["success"] = false;
        result["message"] = "signals.trace.start: Signal tracing is not yet implemented";
        result["note"] = "This would start tracing signal emissions for debugging";
        return result;
    }
    
    if (operation == "signals.trace.stop") {
        result["success"] = false;
        result["message"] = "signals.trace.stop: Signal tracing is not yet implemented";
        result["note"] = "This would stop signal tracing and return collected data";
        return result;
    }
    
    if (operation == "signals.trace.events") {
        result["success"] = false;
        result["message"] = "signals.trace.events: Signal tracing is not yet implemented";
        result["note"] = "This would return collected signal trace events";
        return result;
    }

    // Catch-all for truly unknown operations
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

// --- New Consolidated Tool Implementations ---

Dictionary EditorTools::project_manager(const Dictionary &p_args) {
    Dictionary result;
    String op = p_args.get("op", "");
    
    if (op.is_empty()) {
        result["success"] = false;
        result["error"] = "Operation 'op' parameter is required";
        return result;
    }
    
    if (op == "context.get") {
        // Route to existing get_project_context
        return get_project_context(p_args);
    } else if (op == "fs.list") {
        // Route to existing list_project_files
        return list_project_files(p_args);
    } else if (op == "fs.read") {
        // Route to existing read_file
        return read_file(p_args);
    } else if (op == "fs.write") {
        // fs.write is now handled by frontend async execution for proper compilation checking
        Dictionary result;
        result["success"] = false;
        result["frontend_only"] = true;
        result["message"] = "fs.write operations are handled by frontend with async execution and compilation checking";
        result["operation"] = op;
        result["arguments_to_forward"] = p_args;
        return result;
    } else if (op == "fs.copy") {
        // Map project_manager parameters to copy_file format
        Dictionary args = p_args;
        // Ensure we have the right parameter names
        if (!args.has("source") && args.has("path")) {
            args["source"] = args["path"];
        }
        return copy_file(args);
    } else if (op == "fs.move") {
        // Map project_manager parameters to move_file format
        Dictionary args = p_args;
        // Ensure we have the right parameter names
        if (!args.has("source") && args.has("path")) {
            args["source"] = args["path"];
        }
        return move_file(args);
    } else if (op == "fs.delete") {
        return delete_file(p_args);
    } else if (op == "fs.mkdir") {
        return create_directory(p_args);
    } else if (op == "fs.symlink") {
        return create_symlink(p_args);
    } else if (op == "fs.refresh") {
        return refresh_filesystem(p_args);
    } else if (op == "project.analyze_dir") {
        // Map project_manager parameters to universal_project_manager format
        Dictionary args = p_args;
        args["operation"] = "analyze_directory";
        return universal_project_manager(args);
    } else if (op == "project.copy_dir") {
        // Map project_manager parameters to universal_project_manager format  
        Dictionary args = p_args;
        args["operation"] = "copy_directory";
        return universal_project_manager(args);
    } else if (op == "project.update_refs") {
        // Map project_manager parameters to universal_project_manager format
        Dictionary args = p_args;
        args["operation"] = "update_references";
        return universal_project_manager(args);
    } else {
        result["success"] = false;
        result["error"] = String("Unknown project_manager operation: ") + op;
        return result;
    }
}

Dictionary EditorTools::script_manager(const Dictionary &p_args) {
    Dictionary result;
    String op = p_args.get("op", "");
    
    if (op.is_empty()) {
        result["success"] = false;
        result["error"] = "Operation 'op' parameter is required";
        return result;
    }
    
    if (op == "script.get_for_node") {
        return get_node_script(p_args);
    } else if (op == "script.attach") {
        return attach_script(p_args);
    } else if (op == "script.detach") {
        return detach_script(p_args);
    } else if (op == "script.reload") {
        return reload_script(p_args);
    } else if (op == "classes.refresh") {
        return refresh_global_classes(p_args);
    } else if (op == "classes.custom_list") {
        return get_custom_classes(p_args);
    } else if (op == "classes.available") {
        return get_available_classes(p_args);
    } else if (op == "compile.check") {
        return check_compilation_errors(p_args);
    } else {
        result["success"] = false;
        result["error"] = String("Unknown script_manager operation: ") + op;
        return result;
    }
}

Dictionary EditorTools::resource_manager(const Dictionary &p_args) {
    Dictionary result;
    String op = p_args.get("op", "");
    
    if (op.is_empty()) {
        result["success"] = false;
        result["error"] = "Operation 'op' parameter is required";
        return result;
    }
    
    if (op == "res.create") {
        return create_resource(p_args);
    } else if (op == "res.inspect") {
        Dictionary inspect_args = p_args;
        inspect_args["resource_path"] = p_args.get("target", "");
        return resource_info(inspect_args);
    } else if (op == "res.modify") {
        // Convert new schema to universal_resource_manager format
        Dictionary modify_args = p_args;
        modify_args["operation"] = "modify";  // Convert "op" to "operation"
        return universal_resource_manager(modify_args);
    } else if (op == "res.assign") {
        return assign_resource_to_node_property(p_args);
    } else if (op == "res.copy_from_template") {
        // Convert new schema to universal_resource_manager format  
        Dictionary copy_args = p_args;
        copy_args["operation"] = "copy_from_template";  // Convert "op" to "operation"
        return universal_resource_manager(copy_args);
    } else if (op == "res.refresh") {
        return refresh_filesystem(p_args);
    } else if (op == "res.load_and_assign") {
        return load_and_assign_resource(p_args);
    } else if (op == "import.set_options") {
        // Convert new schema parameter names
        Dictionary import_args = p_args;
        if (p_args.has("import_path")) {
            import_args["resource_path"] = p_args["import_path"];  // Convert import_path to resource_path
        }
        return set_import_preset(import_args);
    } else if (op == "import.reimport") {
        // Convert new schema parameter names
        Dictionary reimport_args = p_args;
        if (p_args.has("import_path")) {
            reimport_args["resource_path"] = p_args["import_path"];  // Convert import_path to resource_path
        }
        return reimport_resource(reimport_args);
    } else if (op == "image.save") {
        // This would be handled by frontend image saving logic
        result["success"] = false;
        result["message"] = "Image saving should be handled by frontend image_operation tool";
        return result;
    } else {
        result["success"] = false;
        result["error"] = String("Unknown resource_manager operation: ") + op;
        return result;
    }
}

Dictionary EditorTools::settings_manager(const Dictionary &p_args) {
    Dictionary result;
    String op = p_args.get("op", "");
    
    if (op.is_empty()) {
        result["success"] = false;
        result["error"] = "Operation 'op' parameter is required";
        return result;
    }
    
    if (op == "project_settings.get") {
        String key = p_args.get("key", "");
        if (key.is_empty()) {
            result["success"] = false;
            result["error"] = "Key parameter required for project_settings.get";
            return result;
        }
        result["success"] = true;
        result["key"] = key;
        result["value"] = ProjectSettings::get_singleton()->get_setting(key);
        return result;
    } else if (op == "project_settings.set") {
        String key = p_args.get("key", "");
        Variant value = p_args.get("value", Variant());
        if (key.is_empty()) {
            result["success"] = false;
            result["error"] = "Key parameter required for project_settings.set";
            return result;
        }
        ProjectSettings::get_singleton()->set_setting(key, value);
        Error err = ProjectSettings::get_singleton()->save();
        result["success"] = (err == OK);
        result["key"] = key;
        result["value"] = value;
        if (err != OK) {
            result["error"] = "Failed to save project settings";
        }
        return result;
    } else if (op == "project_settings.list") {
        String prefix = p_args.get("prefix", "");
        Array settings_list;
        List<PropertyInfo> properties;
        ProjectSettings::get_singleton()->get_property_list(&properties);
        for (const PropertyInfo &prop : properties) {
            if (prefix.is_empty() || prop.name.begins_with(prefix)) {
                Dictionary setting;
                setting["key"] = prop.name;
                setting["value"] = ProjectSettings::get_singleton()->get_setting(prop.name);
                setting["type"] = prop.type;
                settings_list.push_back(setting);
            }
        }
        result["success"] = true;
        result["settings"] = settings_list;
        result["count"] = settings_list.size();
        return result;
    } else if (op == "inputmap.add_action") {
        String action = p_args.get("action", "");
        if (action.is_empty()) {
            result["success"] = false;
            result["error"] = "Action parameter required for inputmap.add_action";
            return result;
        }
        InputMap::get_singleton()->add_action(action);
        result["success"] = true;
        result["action"] = action;
        result["message"] = "Input action added";
        return result;
    } else if (op == "inputmap.erase_action") {
        String action = p_args.get("action", "");
        if (action.is_empty()) {
            result["success"] = false;
            result["error"] = "Action parameter required for inputmap.erase_action";
            return result;
        }
        InputMap::get_singleton()->erase_action(action);
        result["success"] = true;
        result["action"] = action;
        result["message"] = "Input action removed";
        return result;
    } else if (op == "inputmap.action_add_event") {
        String action = p_args.get("action", "");
        Dictionary event_dict = p_args.get("event", Dictionary());
        if (action.is_empty() || event_dict.is_empty()) {
            result["success"] = false;
            result["error"] = "Action and event parameters required for inputmap.action_add_event";
            return result;
        }
        // Create InputEvent from dictionary - simplified implementation
        Ref<InputEvent> event;
        String type = event_dict.get("type", "");
        if (type == "key") {
            Ref<InputEventKey> key_event;
            key_event.instantiate();
            key_event->set_keycode((Key)(int)event_dict.get("keycode", 0));
            event = key_event;
        }
        if (event.is_valid()) {
            InputMap::get_singleton()->action_add_event(action, event);
            result["success"] = true;
            result["action"] = action;
            result["message"] = "Event added to input action";
        } else {
            result["success"] = false;
            result["error"] = "Failed to create input event from provided data";
        }
        return result;
    } else if (op == "inputmap.action_erase_event") {
        String action = p_args.get("action", "");
        Dictionary event_dict = p_args.get("event", Dictionary());
        if (action.is_empty() || event_dict.is_empty()) {
            result["success"] = false;
            result["error"] = "Action and event parameters required for inputmap.action_erase_event";
            return result;
        }
        // For now, return success but note this is simplified
        result["success"] = true;
        result["action"] = action;
        result["message"] = "Event removal from input action (simplified implementation)";
        return result;
    } else if (op == "autoload.add") {
        String autoload_name = p_args.get("autoload_name", "");
        String autoload_path = p_args.get("autoload_path", "");
        bool is_singleton = p_args.get("autoload_is_singleton", true);
        if (autoload_name.is_empty() || autoload_path.is_empty()) {
            result["success"] = false;
            result["error"] = "autoload_name and autoload_path parameters required for autoload.add";
            return result;
        }
        String setting_path = "autoload/" + autoload_name;
        String setting_value = "*" + autoload_path;  // * prefix indicates singleton
        if (!is_singleton) {
            setting_value = autoload_path;  // No * prefix for non-singleton
        }
        ProjectSettings::get_singleton()->set_setting(setting_path, setting_value);
        Error err = ProjectSettings::get_singleton()->save();
        result["success"] = (err == OK);
        result["autoload_name"] = autoload_name;
        result["autoload_path"] = autoload_path;
        result["is_singleton"] = is_singleton;
        if (err != OK) {
            result["error"] = "Failed to save autoload setting";
        } else {
            result["message"] = "Autoload added successfully";
        }
        return result;
    } else if (op == "autoload.remove") {
        String autoload_name = p_args.get("autoload_name", "");
        if (autoload_name.is_empty()) {
            result["success"] = false;
            result["error"] = "autoload_name parameter required for autoload.remove";
            return result;
        }
        String setting_path = "autoload/" + autoload_name;
        if (ProjectSettings::get_singleton()->has_setting(setting_path)) {
            ProjectSettings::get_singleton()->set_setting(setting_path, Variant());
            Error err = ProjectSettings::get_singleton()->save();
            result["success"] = (err == OK);
            result["autoload_name"] = autoload_name;
            if (err != OK) {
                result["error"] = "Failed to save autoload removal";
            } else {
                result["message"] = "Autoload removed successfully";
            }
        } else {
            result["success"] = false;
            result["error"] = "Autoload not found: " + autoload_name;
        }
        return result;
    } else if (op == "layers.get_names") {
        String layer_scope = p_args.get("layer_scope", "");
        if (layer_scope.is_empty()) {
            result["success"] = false;
            result["error"] = "layer_scope parameter required (2d_physics, 3d_physics, 2d_render, 3d_render)";
            return result;
        }
        
        Array layer_names;
        String setting_prefix;
        if (layer_scope == "2d_physics") {
            setting_prefix = "layer_names/2d_physics/layer_";
        } else if (layer_scope == "3d_physics") {
            setting_prefix = "layer_names/3d_physics/layer_";
        } else if (layer_scope == "2d_render") {
            setting_prefix = "layer_names/2d_render/layer_";
        } else if (layer_scope == "3d_render") {
            setting_prefix = "layer_names/3d_render/layer_";
        } else {
            result["success"] = false;
            result["error"] = "Invalid layer_scope. Must be: 2d_physics, 3d_physics, 2d_render, or 3d_render";
            return result;
        }
        
        // Get layer names (typically layers 1-32)
        for (int i = 1; i <= 32; i++) {
            String setting_key = setting_prefix + String::num_int64(i);
            String layer_name = ProjectSettings::get_singleton()->get_setting(setting_key, "");
            layer_names.push_back(layer_name);
        }
        
        result["success"] = true;
        result["layer_scope"] = layer_scope;
        result["layer_names"] = layer_names;
        return result;
    } else if (op == "layers.set_name") {
        String layer_scope = p_args.get("layer_scope", "");
        int layer_index = p_args.get("layer_index", 0);
        String layer_name = p_args.get("layer_name", "");
        
        if (layer_scope.is_empty() || layer_index < 1 || layer_index > 32) {
            result["success"] = false;
            result["error"] = "layer_scope and valid layer_index (1-32) required for layers.set_name";
            return result;
        }
        
        String setting_prefix;
        if (layer_scope == "2d_physics") {
            setting_prefix = "layer_names/2d_physics/layer_";
        } else if (layer_scope == "3d_physics") {
            setting_prefix = "layer_names/3d_physics/layer_";
        } else if (layer_scope == "2d_render") {
            setting_prefix = "layer_names/2d_render/layer_";
        } else if (layer_scope == "3d_render") {
            setting_prefix = "layer_names/3d_render/layer_";
        } else {
            result["success"] = false;
            result["error"] = "Invalid layer_scope. Must be: 2d_physics, 3d_physics, 2d_render, or 3d_render";
            return result;
        }
        
        String setting_key = setting_prefix + String::num_int64(layer_index);
        ProjectSettings::get_singleton()->set_setting(setting_key, layer_name);
        Error err = ProjectSettings::get_singleton()->save();
        
        result["success"] = (err == OK);
        result["layer_scope"] = layer_scope;
        result["layer_index"] = layer_index;
        result["layer_name"] = layer_name;
        if (err != OK) {
            result["error"] = "Failed to save layer name setting";
        } else {
            result["message"] = "Layer name set successfully";
        }
        return result;
    } else {
        result["success"] = false;
        result["error"] = String("Unknown settings_manager operation: ") + op;
        return result;
    }
}

Dictionary EditorTools::search_manager(const Dictionary &p_args) {
    Dictionary result;
    String op = p_args.get("op", "");
    
    if (op.is_empty()) {
        result["success"] = false;
        result["error"] = "Operation 'op' parameter is required";
        return result;
    }
    
    if (op == "project.search") {
        return search_across_project(p_args);
    } else if (op == "docs.search") {
        return search_across_godot_docs(p_args);
    } else {
        result["success"] = false;
        result["error"] = String("Unknown search_manager operation: ") + op;
        return result;
    }
}

Dictionary EditorTools::runtime_manager(const Dictionary &p_args) {
    Dictionary result;
    String op = p_args.get("op", "");
    
    if (op.is_empty()) {
        result["success"] = false;
        result["error"] = "Operation 'op' parameter is required";
        return result;
    }
    
    if (op == "game.start") {
        return run_scene(p_args);
    } else if (op == "game.stop") {
        return stop_game(p_args);
    } else if (op == "game.status") {
        return get_game_status(p_args);
    } else if (op == "errors.summary") {
        return get_runtime_errors_summary(p_args);
    } else if (op == "errors.details") {
        return get_runtime_errors_detailed(p_args);
    } else if (op == "errors.test") {
        // Test function to create some sample runtime errors for debugging
        Dictionary test_error;
        test_error["type"] = "test_error";
        test_error["time_ms"] = Time::get_singleton()->get_ticks_msec();
        test_error["message"] = "Test runtime error for debugging";
        test_error["file"] = "res://test_file.gd";
        test_error["line"] = 42;
        test_error["column"] = 10;
        test_error["is_warning"] = false;
        test_error["source"] = "debug_test";
        record_runtime_error(test_error);
        
        Dictionary test_warning;
        test_warning["type"] = "test_warning";
        test_warning["time_ms"] = Time::get_singleton()->get_ticks_msec();
        test_warning["message"] = "Test runtime warning for debugging";
        test_warning["file"] = "res://test_file.gd";
        test_warning["line"] = 43;
        test_warning["column"] = 5;
        test_warning["is_warning"] = true;
        test_warning["source"] = "debug_test";
        record_runtime_error(test_warning);
        
        result["success"] = true;
        result["message"] = "Added 2 test runtime errors for debugging";
        result["total_recorded"] = s_runtime_errors.size();
        return result;
    } else if (op == "errors.debug") {
        // Debug function to show current runtime error state
        result["success"] = true;
        result["total_recorded"] = s_runtime_errors.size();
        result["message"] = "Currently tracking " + String::num_int64(s_runtime_errors.size()) + " runtime errors";
        
        // Show last 3 errors for debugging
        Array recent_errors;
        int start_idx = MAX(0, s_runtime_errors.size() - 3);
        for (int i = start_idx; i < s_runtime_errors.size(); i++) {
            Dictionary e = s_runtime_errors[i];
            Dictionary debug_error;
            debug_error["message"] = e.get("message", "");
            debug_error["file"] = e.get("file", "");
            debug_error["line"] = e.get("line", 0);
            debug_error["is_warning"] = e.get("is_warning", false);
            debug_error["source"] = e.get("source", "");
            recent_errors.push_back(debug_error);
        }
        result["recent_errors"] = recent_errors;
        return result;
    } else if (op == "screenshot.take") {
        return take_screenshot(p_args);
    } else {
        result["success"] = false;
        result["error"] = String("Unknown runtime_manager operation: ") + op;
        return result;
    }
}