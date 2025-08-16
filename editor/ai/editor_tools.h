/*
 * © 2025 Simplifine Corp.
 * This file is an original contribution to Orca Engine (based on Godot Engine).
 * Licensed for free personal/non-commercial use under the Company Non‑Commercial License.
 * See LICENSES/COMPANY-NONCOMMERCIAL.md. Commercial use requires a separate license from Simplifine.
 */
#pragma once

#include "core/object/class_db.h"
#include "core/object/object.h"
#include "core/variant/dictionary.h"

class Node;

class EditorTools : public Object {
	GDCLASS(EditorTools, Object);

private:
	static Dictionary _get_node_info(Node *p_node);
	static Node *_get_node_from_path(const String &p_path, Dictionary &r_error_result);
    // Trace support
    static EditorTools *tracer_instance;
    static Dictionary trace_registry; // trace_id -> { events:Array, connections:Array, include_args:bool, max_events:int, next_index:int }
    static Dictionary property_watch_registry; // watch_id -> { node_path:String, variables:Array, last_values:Dictionary, events:Array, next_index:int, max_events:int }
    static EditorTools *ensure_tracer();
    static void _record_trace_event(const String &p_trace_id, const String &p_source_path, const String &p_signal_name, const Array &p_args);
    // Signal trace callbacks for 0..4 signal-args; bound extras follow after signal args
    void _on_traced_signal_0(const String &p_trace_id, const String &p_source_path, const String &p_signal_name);
    void _on_traced_signal_1(const Variant &a0, const String &p_trace_id, const String &p_source_path, const String &p_signal_name);
    void _on_traced_signal_2(const Variant &a0, const Variant &a1, const String &p_trace_id, const String &p_source_path, const String &p_signal_name);
    void _on_traced_signal_3(const Variant &a0, const Variant &a1, const Variant &a2, const String &p_trace_id, const String &p_source_path, const String &p_signal_name);
    void _on_traced_signal_4(const Variant &a0, const Variant &a1, const Variant &a2, const Variant &a3, const String &p_trace_id, const String &p_source_path, const String &p_signal_name);

public:
    static Dictionary _predict_code_edit(const String &p_file_content, const String &p_prompt, const String &p_api_endpoint);
    	static Dictionary _call_apply_endpoint(const String &p_file_path, const String &p_file_content, const Dictionary &p_ai_args, const String &p_api_endpoint);
	static String _clean_backend_content(const String &p_content);
	static String _convert_javascript_to_gdscript(const String &p_content);
	static String _fix_malformed_content(const String &p_content);
	static String _generate_unified_diff(const String &p_original, const String &p_modified, const String &p_file_path);
	static Array _check_compilation_errors(const String &p_file_path, const String &p_content);
    static void set_api_endpoint(const String &p_endpoint);

    // Pending preview overlay management (in-memory edited content before save)
    static void set_preview_overlay(const String &p_path, const String &p_content);
    static void clear_preview_overlay(const String &p_path);
    static bool has_preview_overlay(const String &p_path);
    static String get_preview_overlay(const String &p_path);

    // File utilities
    static int get_file_line_count(const String &p_path, int p_max_bytes = 0);

	// Individual Tool Methods (used by universal tools)
	static Dictionary get_scene_info(const Dictionary &p_args);
	static Dictionary get_all_nodes(const Dictionary &p_args);
	static Dictionary search_nodes_by_type(const Dictionary &p_args);
	static Dictionary get_editor_selection(const Dictionary &p_args);
	static Dictionary get_node_properties(const Dictionary &p_args);
	static Dictionary create_node(const Dictionary &p_args);
	static Dictionary delete_node(const Dictionary &p_args);
	static Dictionary set_node_property(const Dictionary &p_args);
	static Dictionary move_node(const Dictionary &p_args);
	static Dictionary call_node_method(const Dictionary &p_args);
	static Dictionary get_available_classes(const Dictionary &p_args);
	static Dictionary get_node_script(const Dictionary &p_args);
	static Dictionary attach_script(const Dictionary &p_args);
	static Dictionary manage_scene(const Dictionary &p_args);
	static Dictionary add_collision_shape(const Dictionary &p_args);
	static Dictionary generalnodeeditor(const Dictionary &p_args);
	static Dictionary list_project_files(const Dictionary &p_args);
	static Dictionary search_project_files(const Dictionary &p_args);
	static Dictionary read_file(const Dictionary &p_args);
	static Dictionary read_file_content(const Dictionary &p_args);
	static Dictionary read_file_advanced(const Dictionary &p_args);
	static Dictionary apply_edit(const Dictionary &p_args);
	static Dictionary check_compilation_errors(const Dictionary &p_args);

	// New Debugging Tools
	static Dictionary run_scene(const Dictionary &p_args);
	static Dictionary get_scene_tree_hierarchy(const Dictionary &p_args);
	static Dictionary inspect_physics_body(const Dictionary &p_args);
	static Dictionary get_camera_info(const Dictionary &p_args);
	static Dictionary take_screenshot(const Dictionary &p_args);
	static Dictionary check_node_in_scene_tree(const Dictionary &p_args);
	static Dictionary inspect_animation_state(const Dictionary &p_args);
	static Dictionary get_layers_and_zindex(const Dictionary &p_args);
	static Dictionary search_across_project(const Dictionary &p_args);
	static Dictionary search_across_godot_docs(const Dictionary &p_args);

	// Universal Tools (New Consolidated API)
	static Dictionary universal_node_manager(const Dictionary &p_args);
	static Dictionary universal_file_manager(const Dictionary &p_args);
	static Dictionary scene_manager(const Dictionary &p_args);

	// Multiplexed introspection/debug tool
	static Dictionary editor_introspect(const Dictionary &p_args);
}; 