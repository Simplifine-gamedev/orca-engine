/*
 * © 2025 Simplifine Corp.
 * Personal Non‑Commercial License applies to this file as an original contribution to this Godot fork.
 * See LICENSES/COMPANY-NONCOMMERCIAL.md for terms. Commercial use requires a separate license from the Project Owner.
 */
#ifndef RUNTIME_INSPECTOR_H
#define RUNTIME_INSPECTOR_H

#include "core/object/object.h"
#include "core/variant/dictionary.h"
#include "core/variant/array.h"
#include "scene/main/node.h"

class RuntimeInspector : public Object {
	GDCLASS(RuntimeInspector, Object);

protected:
	static void _bind_methods();

public:
	// Game state helpers (public for EditorTools access)
	static bool _is_game_running();
	static Node* _get_running_scene_root();

private:
	// Helper to access remote scene tree during play
	static Node* _get_remote_node(const String &p_path);
	
	// Material/shader helpers
	static Dictionary _extract_shader_params(const Ref<Material> &p_material);
	static Dictionary _get_material_info(const Ref<Material> &p_material, bool p_include_code);
	
	// Mesh helpers  
	static Dictionary _extract_mesh_arrays(const Ref<Mesh> &p_mesh, int p_surface_idx, const String &p_array_type);
	static Dictionary _get_uv_info(const Ref<Mesh> &p_mesh, int p_surface_idx);

public:
	// Runtime node operations
	static Dictionary get_runtime_node_properties(const String &p_node_path);
	static Dictionary set_runtime_node_property(const String &p_node_path, const String &p_property, const Variant &p_value);
	static Dictionary get_runtime_scene_tree(int p_max_depth, bool p_include_internal);
	static Dictionary find_runtime_nodes_by_type(const String &p_type_filter);
	
	// Material/shader inspection
	static Dictionary get_runtime_material(const String &p_node_path, const String &p_material_property);
	static Dictionary set_runtime_shader_param(const String &p_node_path, const String &p_material_property, const String &p_param_name, const Variant &p_value);
	static Dictionary list_runtime_shader_params(const String &p_node_path, const String &p_material_property);
	static Dictionary get_runtime_shader_code(const String &p_node_path, const String &p_material_property);
	
	// Mesh inspection
	static Dictionary get_runtime_mesh_arrays(const String &p_node_path, int p_surface_idx, const String &p_array_type);
	static Dictionary get_runtime_mesh_uv_info(const String &p_node_path, int p_surface_idx);
	static Dictionary get_runtime_mesh_surface_count(const String &p_node_path);
	static Dictionary get_runtime_mesh_surface_material(const String &p_node_path, int p_surface_idx);
	
	// Environment/lighting
	static Dictionary get_runtime_environment(const String &p_property);
	static Dictionary set_runtime_environment(const String &p_property, const Variant &p_value);
	static Dictionary get_camera_exposure();
	
	// Screenshot capture
	static Dictionary capture_viewport_screenshot(const String &p_target, bool p_return_base64);
	
	// Watch/breakpoint system (simplified for now)
	static Dictionary add_watch(const String &p_id, const String &p_expression);
	static Dictionary remove_watch(const String &p_id);
	static Dictionary get_watch_values();
};

#endif // RUNTIME_INSPECTOR_H
