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
#include "core/io/dir_access.h"
#include "core/templates/list.h"
#include "core/templates/hash_set.h"

class Node;

class EditorTools : public Object {
	GDCLASS(EditorTools, Object);

private:
	static Dictionary _get_node_info(Node *p_node);
	static Node *_get_node_from_path(const String &p_path, Dictionary &r_error_result);
	static void _set_owner_recursive(Node *p_node, Node *p_owner);
	static void _clear_directory_recursive(Ref<DirAccess> p_dir, const String &p_path, int &r_cleared_count, Array &r_cleared_paths);
	static void _list_directory_files(Ref<DirAccess> p_dir, Array &r_files, bool p_recursive);
	static void _find_files_by_extension(const String &p_path, Array &r_files, const PackedStringArray &p_extensions);
	static Dictionary _grep_search_project(const String &p_query, const Dictionary &p_args);
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
	static void _check_all_scripts_errors(Array &r_errors);
	static void _get_all_project_files(const String &p_path, List<String> &r_files, const HashSet<String> &p_extensions);
	static void _get_all_project_files_limited(const String &p_path, List<String> &r_files, const HashSet<String> &p_extensions, int p_max_files);
    static void set_api_endpoint(const String &p_endpoint);

    // Pending preview overlay management (in-memory edited content before save)
    static void set_preview_overlay(const String &p_path, const String &p_content);
    static void clear_preview_overlay(const String &p_path);
    static void clear_all_preview_overlays();
    static bool has_preview_overlay(const String &p_path);
    static String get_preview_overlay(const String &p_path);

    // Runtime error collection (Debugger/Output → AI)
    static void record_runtime_error(const Dictionary &p_error);
    static Dictionary get_runtime_errors(const Dictionary &p_args);

    // File utilities
    static int get_file_line_count(const String &p_path, int p_max_bytes = 0);
    static String smart_truncate_for_ai_context(const String &p_content, const String &p_file_path);

	// Individual Tool Methods (used by universal tools)
	static Dictionary get_scene_info(const Dictionary &p_args);
	static Dictionary get_all_nodes(const Dictionary &p_args);
	static Dictionary search_nodes_by_type(const Dictionary &p_args);
	static Dictionary get_editor_selection(const Dictionary &p_args);
	static Dictionary get_node_properties(const Dictionary &p_args);
	// Enhanced: include_script_properties (bool)
	static Dictionary create_node(const Dictionary &p_args);
	static Dictionary delete_node(const Dictionary &p_args);
	static Dictionary set_node_property(const Dictionary &p_args);
	static Dictionary batch_set_node_properties(const Dictionary &p_args);
	static Dictionary move_node(const Dictionary &p_args);
	static Dictionary call_node_method(const Dictionary &p_args);
	static Dictionary get_available_classes(const Dictionary &p_args);
	static Dictionary get_node_script(const Dictionary &p_args);
	static Dictionary attach_script(const Dictionary &p_args);
	static Dictionary detach_script(const Dictionary &p_args);
	static Dictionary reload_script(const Dictionary &p_args);
	static Dictionary manage_scene(const Dictionary &p_args);
	static Dictionary load_and_assign_resource(const Dictionary &p_args);
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
	static Dictionary stop_game(const Dictionary &p_args);
	static Dictionary get_game_status(const Dictionary &p_args);
	static Dictionary get_runtime_errors_summary(const Dictionary &p_args);
	static Dictionary get_runtime_errors_detailed(const Dictionary &p_args);
	static Dictionary get_scene_tree_hierarchy(const Dictionary &p_args);
	static Dictionary inspect_physics_body(const Dictionary &p_args);
	static Dictionary get_camera_info(const Dictionary &p_args);
	static Dictionary take_screenshot(const Dictionary &p_args);
	static Dictionary get_console_output(const Dictionary &p_args);
	static Dictionary test_input_action(const Dictionary &p_args);
	static Dictionary test_input_key(const Dictionary &p_args);
	static Dictionary clear_shader_cache(const Dictionary &p_args);
	static Dictionary force_shader_recompile(const Dictionary &p_args);
	static Dictionary debug_shader_cache(const Dictionary &p_args);
	static Dictionary check_node_in_scene_tree(const Dictionary &p_args);
	static Dictionary inspect_animation_state(const Dictionary &p_args);
	static Dictionary get_layers_and_zindex(const Dictionary &p_args);
	static Dictionary search_across_project(const Dictionary &p_args);
	static Dictionary search_across_godot_docs(const Dictionary &p_args);

	// Universal Tools (New Consolidated API)
	static Dictionary universal_node_manager(const Dictionary &p_args);
	static Dictionary universal_file_manager(const Dictionary &p_args);
	static Dictionary scene_manager(const Dictionary &p_args);

	// New Consolidated Tool Methods
	static Dictionary project_manager(const Dictionary &p_args);
	static Dictionary script_manager(const Dictionary &p_args);
	static Dictionary resource_manager(const Dictionary &p_args);
	static Dictionary settings_manager(const Dictionary &p_args);
	static Dictionary search_manager(const Dictionary &p_args);
	static Dictionary runtime_manager(const Dictionary &p_args);
	static Dictionary runtime_inspector(const Dictionary &p_args);

	// Multiplexed introspection/debug tool
	static Dictionary editor_introspect(const Dictionary &p_args);

	// New capabilities for structural edits and resources
	static Dictionary change_node_type(const Dictionary &p_args); // { path:String, new_type:String, preserve_children:bool=true, strategy:String="wrap_root" }
	static Dictionary create_resource(const Dictionary &p_args); // { type:String, properties:Dictionary, save_path:String? }
	static Dictionary assign_resource_to_node_property(const Dictionary &p_args); // { path:String, property:String, resource:Dictionary{ resource_id|path|{type,properties} } }
	static Dictionary create_new_scene_with_root(const Dictionary &p_args); // { new_root_type:String, new_scene_path:String, include_current_as_child:bool=false }

	// Global class and custom type utilities
	static Dictionary refresh_global_classes(const Dictionary &p_args); // {}
	static Dictionary get_custom_classes(const Dictionary &p_args); // {pattern?:String}
	static Dictionary set_node_type(const Dictionary &p_args); // { path:String, type_name?:String, script_path?:String }

	// Universal smart tools (type-aware, bulk-capable)
	static Dictionary universal_resource_manager(const Dictionary &p_args); // { operation:String, type?:String, target?:String, properties?:Dict, source_template?:String }
	static Dictionary universal_scene_manager(const Dictionary &p_args); // { operation:String, scope?:String, targets?:Array, transformations?:Dict, validation?:bool }
	static Dictionary universal_project_manager(const Dictionary &p_args); // { operation:String, assets?:Array, dependencies?:Dict, validation_rules?:Dict }

	// File system and project structure tools (project-root constrained)
	static Dictionary create_directory(const Dictionary &p_args); // { path:String }
	static Dictionary copy_file(const Dictionary &p_args); // { source:String, destination:String, overwrite:bool=false }
	static Dictionary move_file(const Dictionary &p_args); // { source:String, destination:String, overwrite:bool=false }
	static Dictionary delete_file(const Dictionary &p_args); // { path:String }
	static Dictionary create_symlink(const Dictionary &p_args); // { target:String, link_path:String }
	static Dictionary refresh_filesystem(const Dictionary &p_args); // {}
	
	// Enhanced file editing methods
	static Dictionary fs_write_whole_file(const Dictionary &p_args); // { path:String, content:String }
	static Dictionary fs_write_lines_range(const Dictionary &p_args); // { path:String, lines_content:String, start_line:int, end_line:int }
	static Dictionary fs_replace_string_exact(const Dictionary &p_args); // { path:String, find_string:String, replace_string:String, replace_all:bool=false, case_sensitive:bool=true }

	// Introspection & readiness
	static Dictionary resource_info(const Dictionary &p_args); // { resource_path:String }
	static Dictionary script_info(const Dictionary &p_args);   // { script_path:String }

	// Import control
	static Dictionary set_import_preset(const Dictionary &p_args);   // { resource_path:String, importer?:String, options?:Dictionary }
	static Dictionary reimport_resource(const Dictionary &p_args);   // { resource_path:String, timeout_ms?:int }
	static Dictionary wait_for_import(const Dictionary &p_args);     // { resource_path:String, timeout_ms?:int, poll_ms?:int }

	// Project configuration helpers
	static Dictionary enable_plugin(const Dictionary &p_args);             // { plugin_name:String }
	static Dictionary ensure_project_settings(const Dictionary &p_args);   // { settings:Dictionary }
	static Dictionary ensure_input_actions(const Dictionary &p_args);      // { actions:Array<ActionSpec> }
	static Dictionary ensure_autoload(const Dictionary &p_args);           // { entries:Array<AutoloadSpec> }
	static Dictionary get_project_context(const Dictionary &p_args);       // { operation:String }

	// Creation, calls, batching
	static Dictionary ensure_node(const Dictionary &p_args);               // { type:String, name:String, parent?:String, unique?:bool }
	static Dictionary batch_scene_ops(const Dictionary &p_args);           // { ops:Array<Dict>, stop_on_error?:bool }
	
	// Image saving
	static Dictionary save_image_to_path(const Dictionary &p_args);        // { image_id:String, path:String, format?:String }
	
	// Batch operations
	static Dictionary delete_nodes_batch(const Dictionary &p_args);        // { node_paths:Array, ignore_missing?:bool, skip_scene_root?:bool }
	static Dictionary create_nodes_batch(const Dictionary &p_args);        // { nodes_to_create:Array, stop_on_error?:bool }
	static Dictionary set_node_mesh_properties(const Dictionary &p_args);  // { path:String, mesh_property:String, mesh_value:Variant }
	
	// Advanced batch operations
	static Dictionary create_and_configure_nodes_batch(const Dictionary &p_args);  // { templates:Array }
	static Dictionary assign_resources_batch(const Dictionary &p_args);           // { batch_resources:Array }
	static Dictionary set_transforms_batch(const Dictionary &p_args);             // { batch_transforms:Array }
	static Dictionary instantiate_scenes_batch(const Dictionary &p_args);         // { instantiate_batch:Array }
	
	// Pattern-based operations
	static Dictionary set_node_properties_pattern(const Dictionary &p_args);      // { node_pattern:String, property_pattern:String, value_pattern:Variant }
	static Dictionary delete_nodes_pattern(const Dictionary &p_args);             // { node_pattern:String }
	static Dictionary assign_resource_pattern(const Dictionary &p_args);          // { node_pattern:String, property_pattern:String, resource_path_pattern:String }
}; 