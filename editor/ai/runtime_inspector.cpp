/*
 * © 2025 Simplifine Corp.
 * Personal Non‑Commercial License applies to this file as an original contribution to this Godot fork.
 * See LICENSES/COMPANY-NONCOMMERCIAL.md for terms. Commercial use requires a separate license from the Project Owner.
 */
#include "runtime_inspector.h"
#include "core/io/image.h"
#include "core/io/marshalls.h"
#include "core/core_bind.h"
#include "editor/editor_node.h"
#include "editor/editor_interface.h"
#include "editor/run/editor_run_bar.h"
#include "editor/debugger/editor_debugger_inspector.h"
#include "editor/debugger/script_editor_debugger.h"
#include "scene/main/canvas_item.h"
#include "scene/3d/node_3d.h"
#include "scene/3d/mesh_instance_3d.h"
#include "scene/3d/camera_3d.h"
#include "scene/3d/world_environment.h"
#include "scene/resources/camera_attributes.h"
#include "scene/resources/material.h"
#include "scene/resources/shader.h"
#include "scene/resources/mesh.h"
#include "scene/resources/environment.h"
#include "scene/main/viewport.h"
#include "scene/main/window.h"

// Static member definitions
static HashMap<String, Variant> watch_values;
static HashMap<String, String> watch_expressions;

void RuntimeInspector::_bind_methods() {
	// Bind all the public static methods if needed
}

bool RuntimeInspector::_is_game_running() {
	EditorInterface *editor_interface = EditorInterface::get_singleton();
	if (!editor_interface) {
		return false;
	}
	// Check if a scene is being played
	return editor_interface->is_playing_scene();
}

Node* RuntimeInspector::_get_running_scene_root() {
	SceneTree *scene_tree = SceneTree::get_singleton();
	if (!scene_tree) {
		return nullptr;
	}
	
	// During gameplay, we need to access the RUNNING game scene, not the editor scene
	// The running scene is in a separate viewport/context
	EditorInterface *editor_interface = EditorInterface::get_singleton();
	if (editor_interface && editor_interface->is_playing_scene()) {
		
		// Method 1: Try SceneTree's current_scene (the running game scene)
		Node *running_scene = scene_tree->get_current_scene();
		if (running_scene && !running_scene->get_class().contains("Editor") && !running_scene->get_class().contains("Progress")) {
			return running_scene;
		}
		
		// Method 2: Access through the game viewport (separate from editor)
		// During play, there should be a game viewport containing the scene
		Window *root_window = scene_tree->get_root();
		if (root_window) {
			// Look for SubViewport that contains the game
			TypedArray<Node> subviewports = root_window->find_children("*", "SubViewport", true, false);
			for (int i = 0; i < subviewports.size(); i++) {
				SubViewport *viewport = Object::cast_to<SubViewport>(subviewports[i]);
				if (viewport && viewport->get_child_count() > 0) {
					Node *scene_in_viewport = viewport->get_child(0);
					if (scene_in_viewport && 
						!scene_in_viewport->get_class().contains("Editor") &&
						!scene_in_viewport->get_class().contains("Progress")) {
						return scene_in_viewport;
					}
				}
			}
			
			// Method 3: Look for any Window that might contain the game scene
			TypedArray<Node> windows = root_window->find_children("*", "Window", true, false);
			for (int i = 0; i < windows.size(); i++) {
				Window *win = Object::cast_to<Window>(windows[i]);
				if (win && win != root_window && win->get_child_count() > 0) {
					Node *scene_in_window = win->get_child(0);
					if (scene_in_window && 
						!scene_in_window->get_class().contains("Editor") &&
						!scene_in_window->get_class().contains("Progress")) {
						return scene_in_window;
					}
				}
			}
		}
		
		// Method 4: Direct search in the entire tree for nodes with 3D names
		Window *root = scene_tree->get_root();
		if (root) {
			// Look for nodes with names suggesting 3D scene content
			TypedArray<Node> candidates = root->find_children("*", "", true, false);
			for (int i = 0; i < candidates.size(); i++) {
				Node *candidate = Object::cast_to<Node>(candidates[i]);
				if (candidate) {
					String name = String(candidate->get_name()).to_lower();
					String class_name = candidate->get_class();
					
					// Look for scene-like nodes or 3D-specific nodes
					if ((name.contains("rocket") || name.contains("main") || name.contains("game") || 
						 class_name == "Node3D" || class_name == "RigidBody3D" || class_name == "CharacterBody3D") &&
						!class_name.contains("Editor") && !class_name.contains("Progress")) {
						
						// Found a promising node - check if it has reasonable children
						if (candidate->get_child_count() > 0) {
							return candidate;
						}
					}
				}
			}
		}
	}
	
	// Fallback: return whatever SceneTree thinks is current
	return scene_tree->get_current_scene();
}

Node* RuntimeInspector::_get_remote_node(const String &p_path) {
	if (!_is_game_running()) {
		return nullptr;
	}
	
	// This function would ideally access remote scene tree via debugger
	// For now, we return nullptr as proper remote access needs more implementation
	// The actual runtime inspection is done directly in other methods
	return nullptr;
}

Dictionary RuntimeInspector::get_runtime_node_properties(const String &p_node_path) {
	Dictionary result;
	
	if (!_is_game_running()) {
		result["success"] = false;
		result["error"] = "Game is not running. Start the game to inspect runtime properties.";
		result["debug_tip"] = "Use runtime_manager with op: 'game.start' first";
		return result;
	}
	
	// For now, we'll access the editor's running scene directly
	// In a full implementation, this would use the debugger's remote inspection
	EditorNode *editor = EditorNode::get_singleton();
	if (!editor) {
		result["success"] = false;
		result["error"] = "Editor not accessible";
		return result;
	}
	
	// Get the running scene root
	Node *scene_root = _get_running_scene_root();
	if (!scene_root) {
		result["success"] = false;
		result["error"] = "No running scene found";
		return result;
	}
	
	// Find the node
	Node *node = scene_root->get_node_or_null(NodePath(p_node_path));
	if (!node) {
		result["success"] = false;
		result["error"] = "Node not found: " + p_node_path;
		result["scene_root_path"] = String(scene_root->get_path());
		result["scene_root_type"] = scene_root->get_class();
		result["debug_info"] = "Try using relative paths from scene root, or use runtime.node.get_tree to see available nodes";
		return result;
	}
	
	// Get all properties
	Dictionary properties;
	List<PropertyInfo> prop_list;
	node->get_property_list(&prop_list);
	
	for (const PropertyInfo &prop : prop_list) {
		// Skip internal properties
		if (prop.name.begins_with("_")) {
			continue;
		}
		
		Variant value = node->get(prop.name);
		properties[prop.name] = value;
	}
	
	result["success"] = true;
	result["properties"] = properties;
	result["node_type"] = node->get_class();
	result["node_path"] = node->get_path();
	
	return result;
}

Dictionary RuntimeInspector::set_runtime_node_property(const String &p_node_path, const String &p_property, const Variant &p_value) {
	Dictionary result;
	
	if (!_is_game_running()) {
		result["success"] = false;
		result["error"] = "Game is not running";
		return result;
	}
	
	Node *scene_root = _get_running_scene_root();
	if (!scene_root) {
		result["success"] = false;
		result["error"] = "No running scene found";
		return result;
	}
	
	Node *node = scene_root->get_node_or_null(NodePath(p_node_path));
	
	if (!node) {
		result["success"] = false;
		result["error"] = "Node not found: " + p_node_path;
		return result;
	}
	
	// Store old value
	Variant old_value = node->get(p_property);
	
	// Set new value
	node->set(p_property, p_value);
	
	result["success"] = true;
	result["old_value"] = old_value;
	result["new_value"] = node->get(p_property);
	result["property"] = p_property;
	
	return result;
}

Dictionary RuntimeInspector::get_runtime_material(const String &p_node_path, const String &p_material_property) {
	Dictionary result;
	
	if (!_is_game_running()) {
		result["success"] = false;
		result["error"] = "Game is not running";
		return result;
	}
	
	Window *root_window = SceneTree::get_singleton()->get_root();
	Node *scene_root = root_window ? root_window->get_child(root_window->get_child_count() - 1) : nullptr;
	Node *node = scene_root ? scene_root->get_node_or_null(NodePath(p_node_path)) : nullptr;
	
	if (!node) {
		result["success"] = false;
		result["error"] = "Node not found: " + p_node_path;
		return result;
	}
	
	// Get material based on property path
	Ref<Material> material;
	
	if (p_material_property == "material_override") {
		if (MeshInstance3D *mesh_inst = Object::cast_to<MeshInstance3D>(node)) {
			material = mesh_inst->get_material_override();
		}
	} else if (p_material_property.begins_with("surface_material_override/")) {
		String idx_str = p_material_property.get_slice("/", 1);
		int surface_idx = idx_str.to_int();
		if (MeshInstance3D *mesh_inst = Object::cast_to<MeshInstance3D>(node)) {
			material = mesh_inst->get_surface_override_material(surface_idx);
		}
	} else {
		// Try generic property access
		Variant mat_var = node->get(p_material_property);
		if (mat_var.get_type() == Variant::OBJECT) {
			material = mat_var;
		}
	}
	
	if (material.is_null()) {
		result["success"] = false;
		result["error"] = "No material found at " + p_material_property;
		return result;
	}
	
	result["success"] = true;
	result["material_class"] = material->get_class();
	result["material_rid"] = material->get_rid().get_id();
	result["parameters"] = _extract_shader_params(material);
	
	return result;
}

Dictionary RuntimeInspector::_extract_shader_params(const Ref<Material> &p_material) {
	Dictionary params;
	
	if (p_material.is_null()) {
		return params;
	}
	
	// Get shader parameters
	if (ShaderMaterial *shader_mat = Object::cast_to<ShaderMaterial>(p_material.ptr())) {
		Ref<Shader> shader = shader_mat->get_shader();
		if (shader.is_valid()) {
			// Get all shader params
			List<PropertyInfo> prop_list;
			shader_mat->get_property_list(&prop_list);
			
			for (const PropertyInfo &prop : prop_list) {
				if (prop.name.begins_with("shader_parameter/")) {
					String param_name = prop.name.get_slice("/", 1);
					Variant value = shader_mat->get_shader_parameter(param_name);
					params[param_name] = value;
				}
			}
		}
	} else if (StandardMaterial3D *std_mat = Object::cast_to<StandardMaterial3D>(p_material.ptr())) {
		// For standard materials, get common properties
		params["albedo_color"] = std_mat->get_albedo();
		params["metallic"] = std_mat->get_metallic();
		params["roughness"] = std_mat->get_roughness();
		params["emission_enabled"] = std_mat->get_feature(BaseMaterial3D::FEATURE_EMISSION);
		if (std_mat->get_feature(BaseMaterial3D::FEATURE_EMISSION)) {
			params["emission_color"] = std_mat->get_emission();
			params["emission_energy_multiplier"] = std_mat->get_emission_energy_multiplier();
		}
	}
	
	return params;
}

Dictionary RuntimeInspector::set_runtime_shader_param(const String &p_node_path, const String &p_material_property, const String &p_param_name, const Variant &p_value) {
	Dictionary result;
	
	if (!_is_game_running()) {
		result["success"] = false;
		result["error"] = "Game is not running";
		return result;
	}
	
	Window *root_window = SceneTree::get_singleton()->get_root();
	Node *scene_root = root_window ? root_window->get_child(root_window->get_child_count() - 1) : nullptr;
	Node *node = scene_root ? scene_root->get_node_or_null(NodePath(p_node_path)) : nullptr;
	
	if (!node) {
		result["success"] = false;
		result["error"] = "Node not found: " + p_node_path;
		return result;
	}
	
	// Get material
	Ref<Material> material;
	if (p_material_property == "material_override") {
		if (MeshInstance3D *mesh_inst = Object::cast_to<MeshInstance3D>(node)) {
			material = mesh_inst->get_material_override();
		}
	}
	
	if (material.is_null()) {
		result["success"] = false;
		result["error"] = "Material not found";
		return result;
	}
	
	if (ShaderMaterial *shader_mat = Object::cast_to<ShaderMaterial>(material.ptr())) {
		Variant old_value = shader_mat->get_shader_parameter(p_param_name);
		shader_mat->set_shader_parameter(p_param_name, p_value);
		
		result["success"] = true;
		result["old_value"] = old_value;
		result["new_value"] = shader_mat->get_shader_parameter(p_param_name);
	} else {
		result["success"] = false;
		result["error"] = "Material is not a ShaderMaterial";
	}
	
	return result;
}

Dictionary RuntimeInspector::get_runtime_mesh_arrays(const String &p_node_path, int p_surface_idx, const String &p_array_type) {
	Dictionary result;
	
	if (!_is_game_running()) {
		result["success"] = false;
		result["error"] = "Game is not running";
		return result;
	}
	
	Window *root_window = SceneTree::get_singleton()->get_root();
	Node *scene_root = root_window ? root_window->get_child(root_window->get_child_count() - 1) : nullptr;
	Node *node = scene_root ? scene_root->get_node_or_null(NodePath(p_node_path)) : nullptr;
	
	if (!node) {
		result["success"] = false;
		result["error"] = "Node not found: " + p_node_path;
		return result;
	}
	
	MeshInstance3D *mesh_inst = Object::cast_to<MeshInstance3D>(node);
	if (!mesh_inst) {
		result["success"] = false;
		result["error"] = "Node is not a MeshInstance3D";
		return result;
	}
	
	Ref<Mesh> mesh = mesh_inst->get_mesh();
	if (mesh.is_null()) {
		result["success"] = false;
		result["error"] = "No mesh found";
		return result;
	}
	
	if (p_surface_idx < 0 || p_surface_idx >= mesh->get_surface_count()) {
		result["success"] = false;
		result["error"] = "Invalid surface index";
		return result;
	}
	
	// Get mesh arrays
	Array arrays = mesh->surface_get_arrays(p_surface_idx);
	
	// Map array type string to index
	int array_idx = -1;
	if (p_array_type == "vertex") array_idx = Mesh::ARRAY_VERTEX;
	else if (p_array_type == "normal") array_idx = Mesh::ARRAY_NORMAL;
	else if (p_array_type == "tangent") array_idx = Mesh::ARRAY_TANGENT;
	else if (p_array_type == "color") array_idx = Mesh::ARRAY_COLOR;
	else if (p_array_type == "tex_uv") array_idx = Mesh::ARRAY_TEX_UV;
	else if (p_array_type == "tex_uv2") array_idx = Mesh::ARRAY_TEX_UV2;
	else if (p_array_type == "index") array_idx = Mesh::ARRAY_INDEX;
	
	if (array_idx < 0 || array_idx >= arrays.size()) {
		result["success"] = false;
		result["error"] = "Invalid array type: " + p_array_type;
		return result;
	}
	
	result["success"] = true;
	result["array_data"] = arrays[array_idx];
	result["array_type"] = p_array_type;
	result["surface_index"] = p_surface_idx;
	
	// Add some stats
	if (array_idx == Mesh::ARRAY_VERTEX) {
		PackedVector3Array vertices = arrays[array_idx];
		result["vertex_count"] = vertices.size();
		
		// Calculate bounds
		if (vertices.size() > 0) {
			Vector3 min_v = vertices[0];
			Vector3 max_v = vertices[0];
			for (int i = 1; i < vertices.size(); i++) {
				min_v = min_v.min(vertices[i]);
				max_v = max_v.max(vertices[i]);
			}
			result["bounds_min"] = min_v;
			result["bounds_max"] = max_v;
		}
	} else if (array_idx == Mesh::ARRAY_TEX_UV || array_idx == Mesh::ARRAY_TEX_UV2) {
		PackedVector2Array uvs = arrays[array_idx];
		result["uv_count"] = uvs.size();
		
		// Calculate UV bounds
		if (uvs.size() > 0) {
			Vector2 min_uv = uvs[0];
			Vector2 max_uv = uvs[0];
			for (int i = 1; i < uvs.size(); i++) {
				min_uv = min_uv.min(uvs[i]);
				max_uv = max_uv.max(uvs[i]);
			}
			result["uv_bounds_min"] = min_uv;
			result["uv_bounds_max"] = max_uv;
		}
	}
	
	return result;
}

Dictionary RuntimeInspector::get_runtime_mesh_uv_info(const String &p_node_path, int p_surface_idx) {
	// Delegate to get_runtime_mesh_arrays with UV type
	return get_runtime_mesh_arrays(p_node_path, p_surface_idx, "tex_uv");
}

Dictionary RuntimeInspector::get_runtime_scene_tree(int p_max_depth, bool p_include_internal) {
	Dictionary result;
	
	if (!_is_game_running()) {
		result["success"] = false;
		result["error"] = "Game is not running";
		return result;
	}
	
	Window *root_window = SceneTree::get_singleton()->get_root();
	if (!root_window) {
		result["success"] = false;
		result["error"] = "No root window";
		return result;
	}
	
	Node *scene_root = root_window->get_child(root_window->get_child_count() - 1);
	if (!scene_root) {
		result["success"] = false;
		result["error"] = "No running scene";
		return result;
	}
	
	// Build tree structure
	Array tree;
	
	// Helper function to build tree recursively (using simple recursion instead of std::function)
	struct TreeBuilder {
		static void build_tree(Node* node, int depth, int max_depth, bool include_internal, Array& parent_array) {
			if (!node || depth > max_depth) {
				return;
			}
			
			if (!include_internal && node->is_class("EditorNode")) {
				return;
			}
			
			Dictionary node_info;
			node_info["name"] = node->get_name();
			node_info["type"] = node->get_class();
			node_info["path"] = String(node->get_path());
			// Check if node is visible (for CanvasItem or Node3D)
			if (CanvasItem *ci = Object::cast_to<CanvasItem>(node)) {
				node_info["visible"] = ci->is_visible();
			} else if (Node3D *n3d = Object::cast_to<Node3D>(node)) {
				node_info["visible"] = n3d->is_visible();
			} else {
				node_info["visible"] = true;
			}
			
			// Add some key properties for context
			if (MeshInstance3D *mesh = Object::cast_to<MeshInstance3D>(node)) {
				node_info["has_mesh"] = mesh->get_mesh().is_valid();
				node_info["has_material_override"] = mesh->get_material_override().is_valid();
			}
			
			Array children;
			for (int i = 0; i < node->get_child_count(); i++) {
				build_tree(node->get_child(i), depth + 1, max_depth, include_internal, children);
			}
			
			if (children.size() > 0) {
				node_info["children"] = children;
			}
			
			parent_array.push_back(node_info);
		}
	};
	
	TreeBuilder::build_tree(scene_root, 0, p_max_depth, p_include_internal, tree);
	
	result["success"] = true;
	result["tree"] = tree;
	result["root_path"] = String(scene_root->get_path());
	
	return result;
}

Dictionary RuntimeInspector::capture_viewport_screenshot(const String &p_target, bool p_return_base64) {
	Dictionary result;
	
	SceneTree *scene_tree = SceneTree::get_singleton();
	if (!scene_tree) {
		result["success"] = false;
		result["error"] = "SceneTree not available";
		return result;
	}
	
	// Get the viewport to capture - find the GAME viewport, not editor
	Viewport *viewport = nullptr;
	
	// Try to find the game window/viewport by looking for running game instances
	if (EditorRunBar::get_singleton() && EditorRunBar::get_singleton()->is_playing()) {
		// Look for game viewports in the scene tree
		Window *root_window = scene_tree->get_root();
		if (root_window) {
			// Search through all child windows to find the game window
			TypedArray<Node> windows = root_window->find_children("*", "Window", true, false);
			for (int i = 0; i < windows.size(); i++) {
				Window *child_window = Object::cast_to<Window>(windows[i]);
				if (child_window && child_window != root_window) {
					// This could be the game window - try to get its main viewport
					viewport = child_window;
					print_line("AI Chat: Found potential game window: " + child_window->get_name());
					break;
				}
			}
		}
	}
	
	// Fallback to root viewport if no game viewport found
	if (!viewport) {
		viewport = scene_tree->get_root();
		print_line("AI Chat: Using root viewport as fallback");
	}
	
	if (!viewport) {
		result["success"] = false;
		result["error"] = "Could not access viewport for target: " + p_target;
		return result;
	}
	
	// Get the viewport texture
	Ref<ViewportTexture> viewport_texture = viewport->get_texture();
	if (viewport_texture.is_null()) {
		result["success"] = false;
		result["error"] = "Viewport texture not available";
		return result;
	}
	
	// NON-BLOCKING: Get image from viewport with yield to prevent freeze
	print_line("AI Chat: Capturing viewport image...");
	
	// Brief yield before expensive image capture
	OS::get_singleton()->delay_usec(1000);
	
	Ref<Image> img = viewport_texture->get_image();
	
	if (img.is_null() || img->is_empty()) {
		result["success"] = false;
		result["error"] = "Failed to capture viewport image - viewport may be empty";
		result["debug_info"] = "Viewport size: " + String::num_int64(viewport->get_visible_rect().size.x) + "x" + String::num_int64(viewport->get_visible_rect().size.y);
		return result;
	}
	
	// Yield during image processing
	OS::get_singleton()->delay_usec(1000);
	
	result["success"] = true;
	result["width"] = img->get_width();
	result["height"] = img->get_height();
	result["format"] = img->get_format();
	
	if (p_return_base64) {
		// NON-BLOCKING: Convert to PNG and encode as base64 with yields
		print_line("AI Chat: Converting to PNG buffer...");
		Vector<uint8_t> png_buffer = img->save_png_to_buffer();
		
		// Yield during base64 encoding (expensive operation)
		OS::get_singleton()->delay_usec(2000); // 2ms yield
		
		print_line("AI Chat: Encoding to base64...");
		String base64 = CoreBind::Marshalls::get_singleton()->raw_to_base64(png_buffer);
		
		// Generate unique ID for screenshot
		String screenshot_id = "screenshot_" + p_target + "_" + String::num_int64(OS::get_singleton()->get_ticks_msec());
		
		// CRITICAL: Format result for UI lazy loader (same as image generation)
		result["image_data"] = base64;  // Key field that triggers lazy loading
		result["base64"] = base64;      // Backward compatibility
		result["data_uri"] = "data:image/png;base64," + base64;
		result["prompt"] = "Runtime Screenshot (" + p_target + ")";  // Title for lazy loader
		result["image_type"] = "screenshot";  // Tells UI this is a screenshot
		result["image_id"] = screenshot_id;
		result["image_name"] = screenshot_id;
		result["target"] = p_target;
		result["model"] = "Viewport Capture";  // For lazy loader display
		
		print_line("AI Chat: Screenshot ready - " + String::num_int64(img->get_width()) + "x" + String::num_int64(img->get_height()) + " pixels");
	}
	
	// DISABLED: Don't automatically save screenshots to disk - they're for AI analysis only
	// This prevents cluttering the project with screenshot files
	// String filename = p_target + "_screenshot_" + String::num_int64(OS::get_singleton()->get_unix_time()) + ".png";
	// String path = "res://" + filename;
	// Error err = img->save_png(path);
	// if (err == OK) {
	//     result["saved_path"] = path;
	//     result["filename"] = filename;
	// } else {
	//     result["save_error"] = "Failed to save screenshot to " + path;
	// }
	
	return result;
}

Dictionary RuntimeInspector::get_runtime_environment(const String &p_property) {
	Dictionary result;
	
	if (!_is_game_running()) {
		result["success"] = false;
		result["error"] = "Game is not running";
		return result;
	}
	
	// Find WorldEnvironment node
	Window *root = SceneTree::get_singleton()->get_root();
	if (!root) {
		result["success"] = false;
		result["error"] = "No root window";
		return result;
	}
	
	WorldEnvironment *world_env = nullptr;
	
	// Search for WorldEnvironment in scene
	Node *scene_root = root->get_child(root->get_child_count() - 1);
	if (scene_root) {
		// Find first WorldEnvironment node
		TypedArray<Node> nodes = scene_root->find_children("*", "WorldEnvironment", true, false);
		if (nodes.size() > 0) {
			world_env = Object::cast_to<WorldEnvironment>(nodes[0]);
		}
	}
	
	if (!world_env) {
		result["success"] = false;
		result["error"] = "No WorldEnvironment found in scene";
		return result;
	}
	
	Ref<Environment> env = world_env->get_environment();
	if (env.is_null()) {
		result["success"] = false;
		result["error"] = "WorldEnvironment has no Environment resource";
		return result;
	}
	
	// Get specific property or all properties
	if (!p_property.is_empty()) {
		if (p_property == "tonemap_mode") {
			result["value"] = env->get_tonemapper();
		} else if (p_property == "tonemap_exposure") {
			result["value"] = env->get_tonemap_exposure();
		} else if (p_property == "tonemap_white") {
			result["value"] = env->get_tonemap_white();
		} else {
			Variant value = env->get(p_property);
			result["value"] = value;
		}
		result["property"] = p_property;
	} else {
		// Return all key environment settings
		Dictionary env_data;
		env_data["tonemap_mode"] = env->get_tonemapper();
		env_data["tonemap_exposure"] = env->get_tonemap_exposure();
		env_data["tonemap_white"] = env->get_tonemap_white();
		env_data["background_mode"] = env->get_background();
		env_data["ambient_source"] = env->get_ambient_source();
		env_data["ambient_light_energy"] = env->get_ambient_light_energy();
		
		result["environment"] = env_data;
	}
	
	result["success"] = true;
	return result;
}

Dictionary RuntimeInspector::set_runtime_environment(const String &p_property, const Variant &p_value) {
	Dictionary result;
	
	if (!_is_game_running()) {
		result["success"] = false;
		result["error"] = "Game is not running";
		return result;
	}
	
	// Find WorldEnvironment (same as get method)
	Window *root = SceneTree::get_singleton()->get_root();
	Node *scene_root = root ? root->get_child(root->get_child_count() - 1) : nullptr;
	
	WorldEnvironment *world_env = nullptr;
	if (scene_root) {
		TypedArray<Node> nodes = scene_root->find_children("*", "WorldEnvironment", true, false);
		if (nodes.size() > 0) {
			world_env = Object::cast_to<WorldEnvironment>(nodes[0]);
		}
	}
	
	if (!world_env) {
		result["success"] = false;
		result["error"] = "No WorldEnvironment found";
		return result;
	}
	
	Ref<Environment> env = world_env->get_environment();
	if (env.is_null()) {
		result["success"] = false;
		result["error"] = "No Environment resource";
		return result;
	}
	
	// Set the property
	Variant old_value = env->get(p_property);
	env->set(p_property, p_value);
	
	result["success"] = true;
	result["old_value"] = old_value;
	result["new_value"] = env->get(p_property);
	result["property"] = p_property;
	
	return result;
}

Dictionary RuntimeInspector::find_runtime_nodes_by_type(const String &p_type_filter) {
	Dictionary result;
	
	if (!_is_game_running()) {
		result["success"] = false;
		result["error"] = "Game is not running";
		return result;
	}
	
	Node *scene_root = _get_running_scene_root();
	if (!scene_root) {
		result["success"] = false;
		result["error"] = "No running scene found";
		return result;
	}
	
	// Find all nodes of the specified type
	Array found_nodes;
	TypedArray<Node> nodes = scene_root->find_children("*", p_type_filter, true, false);
	
	for (int i = 0; i < nodes.size(); i++) {
		Node *node = Object::cast_to<Node>(nodes[i]);
		if (node) {
			Dictionary node_info;
			node_info["path"] = String(node->get_path());
			node_info["name"] = node->get_name();
			node_info["type"] = node->get_class();
			found_nodes.push_back(node_info);
		}
	}
	
	result["success"] = true;
	result["nodes"] = found_nodes;
	result["count"] = found_nodes.size();
	result["type_filter"] = p_type_filter;
	
	return result;
}

Dictionary RuntimeInspector::list_runtime_shader_params(const String &p_node_path, const String &p_material_property) {
	Dictionary result = get_runtime_material(p_node_path, p_material_property);
	
	if (result.get("success", false)) {
		// Parameters are already extracted in get_runtime_material
		result["shader_params"] = result.get("parameters", Dictionary());
		result.erase("parameters");
	}
	
	return result;
}

Dictionary RuntimeInspector::get_runtime_shader_code(const String &p_node_path, const String &p_material_property) {
	Dictionary result;
	
	if (!_is_game_running()) {
		result["success"] = false;
		result["error"] = "Game is not running";
		return result;
	}
	
	Window *root_window = SceneTree::get_singleton()->get_root();
	Node *scene_root = root_window ? root_window->get_child(root_window->get_child_count() - 1) : nullptr;
	Node *node = scene_root ? scene_root->get_node_or_null(NodePath(p_node_path)) : nullptr;
	
	if (!node) {
		result["success"] = false;
		result["error"] = "Node not found: " + p_node_path;
		return result;
	}
	
	// Get material
	Ref<Material> material;
	if (p_material_property == "material_override") {
		if (MeshInstance3D *mesh_inst = Object::cast_to<MeshInstance3D>(node)) {
			material = mesh_inst->get_material_override();
		}
	}
	
	if (material.is_null()) {
		result["success"] = false;
		result["error"] = "Material not found";
		return result;
	}
	
	if (ShaderMaterial *shader_mat = Object::cast_to<ShaderMaterial>(material.ptr())) {
		Ref<Shader> shader = shader_mat->get_shader();
		if (shader.is_valid()) {
			result["success"] = true;
			result["shader_code"] = shader->get_code();
			result["shader_type"] = "ShaderMaterial";
		} else {
			result["success"] = false;
			result["error"] = "Shader not found on material";
		}
	} else {
		result["success"] = false;
		result["error"] = "Material is not a ShaderMaterial";
		result["material_type"] = material->get_class();
	}
	
	return result;
}

Dictionary RuntimeInspector::get_runtime_mesh_surface_count(const String &p_node_path) {
	Dictionary result;
	
	if (!_is_game_running()) {
		result["success"] = false;
		result["error"] = "Game is not running";
		return result;
	}
	
	Window *root_window = SceneTree::get_singleton()->get_root();
	Node *scene_root = root_window ? root_window->get_child(root_window->get_child_count() - 1) : nullptr;
	Node *node = scene_root ? scene_root->get_node_or_null(NodePath(p_node_path)) : nullptr;
	
	if (!node) {
		result["success"] = false;
		result["error"] = "Node not found: " + p_node_path;
		return result;
	}
	
	MeshInstance3D *mesh_inst = Object::cast_to<MeshInstance3D>(node);
	if (!mesh_inst) {
		result["success"] = false;
		result["error"] = "Node is not a MeshInstance3D";
		return result;
	}
	
	Ref<Mesh> mesh = mesh_inst->get_mesh();
	if (mesh.is_null()) {
		result["success"] = false;
		result["error"] = "No mesh found";
		return result;
	}
	
	result["success"] = true;
	result["surface_count"] = mesh->get_surface_count();
	result["mesh_type"] = mesh->get_class();
	
	return result;
}

Dictionary RuntimeInspector::get_runtime_mesh_surface_material(const String &p_node_path, int p_surface_idx) {
	Dictionary result;
	
	if (!_is_game_running()) {
		result["success"] = false;
		result["error"] = "Game is not running";
		return result;
	}
	
	Window *root_window = SceneTree::get_singleton()->get_root();
	Node *scene_root = root_window ? root_window->get_child(root_window->get_child_count() - 1) : nullptr;
	Node *node = scene_root ? scene_root->get_node_or_null(NodePath(p_node_path)) : nullptr;
	
	if (!node) {
		result["success"] = false;
		result["error"] = "Node not found: " + p_node_path;
		return result;
	}
	
	MeshInstance3D *mesh_inst = Object::cast_to<MeshInstance3D>(node);
	if (!mesh_inst) {
		result["success"] = false;
		result["error"] = "Node is not a MeshInstance3D";
		return result;
	}
	
	Ref<Material> surface_material = mesh_inst->get_surface_override_material(p_surface_idx);
	
	if (surface_material.is_valid()) {
		result["success"] = true;
		result["has_surface_override"] = true;
		result["material_class"] = surface_material->get_class();
		result["parameters"] = _extract_shader_params(surface_material);
	} else {
		// Check mesh's built-in surface material
		Ref<Mesh> mesh = mesh_inst->get_mesh();
		if (mesh.is_valid() && p_surface_idx < mesh->get_surface_count()) {
			Ref<Material> mesh_material = mesh->surface_get_material(p_surface_idx);
			if (mesh_material.is_valid()) {
				result["success"] = true;
				result["has_surface_override"] = false;
				result["material_class"] = mesh_material->get_class();
				result["parameters"] = _extract_shader_params(mesh_material);
			} else {
				result["success"] = true;
				result["has_material"] = false;
				result["message"] = "No material on this surface";
			}
		} else {
			result["success"] = false;
			result["error"] = "Invalid surface index";
		}
	}
	
	result["surface_index"] = p_surface_idx;
	
	return result;
}

Dictionary RuntimeInspector::get_camera_exposure() {
	Dictionary result;
	
	if (!_is_game_running()) {
		result["success"] = false;
		result["error"] = "Game is not running";
		return result;
	}
	
	Node *scene_root = _get_running_scene_root();
	if (!scene_root) {
		result["success"] = false;
		result["error"] = "No running scene found";
		return result;
	}
	
	// Find active Camera3D
	Camera3D *camera = nullptr;
	TypedArray<Node> cameras = scene_root->find_children("*", "Camera3D", true, false);
	
	for (int i = 0; i < cameras.size(); i++) {
		Camera3D *cam = Object::cast_to<Camera3D>(cameras[i]);
		if (cam && cam->is_current()) {
			camera = cam;
			break;
		}
	}
	
	if (!camera) {
		result["success"] = false;
		result["error"] = "No active Camera3D found";
		return result;
	}
	
	// Get camera attributes
	Ref<CameraAttributes> attributes = camera->get_attributes();
	
	result["success"] = true;
	result["camera_path"] = String(camera->get_path());
	
	if (attributes.is_valid()) {
		result["has_attributes"] = true;
		
		// Check if it's CameraAttributesPractical (has exposure settings)
		if (CameraAttributesPractical *practical = Object::cast_to<CameraAttributesPractical>(attributes.ptr())) {
			result["exposure_sensitivity"] = practical->get_exposure_sensitivity();
			result["exposure_multiplier"] = practical->get_exposure_multiplier();
			
			result["auto_exposure_enabled"] = practical->is_auto_exposure_enabled();
			if (practical->is_auto_exposure_enabled()) {
				result["auto_exposure_min"] = practical->get_auto_exposure_min_sensitivity();
				result["auto_exposure_max"] = practical->get_auto_exposure_max_sensitivity();
				result["auto_exposure_speed"] = practical->get_auto_exposure_speed();
				result["auto_exposure_scale"] = practical->get_auto_exposure_scale();
			}
		} else {
			result["attributes_type"] = attributes->get_class();
		}
	} else {
		result["has_attributes"] = false;
		result["message"] = "Camera has no attributes set";
	}
	
	return result;
}

Dictionary RuntimeInspector::add_watch(const String &p_id, const String &p_expression) {
	Dictionary result;
	
	if (p_id.is_empty() || p_expression.is_empty()) {
		result["success"] = false;
		result["error"] = "Watch ID and expression are required";
		return result;
	}
	
	watch_expressions[p_id] = p_expression;
	
	result["success"] = true;
	result["watch_id"] = p_id;
	result["expression"] = p_expression;
	result["message"] = "Watch added (Note: Expression evaluation not yet implemented)";
	
	return result;
}

Dictionary RuntimeInspector::remove_watch(const String &p_id) {
	Dictionary result;
	
	if (watch_expressions.has(p_id)) {
		watch_expressions.erase(p_id);
		watch_values.erase(p_id);
		result["success"] = true;
		result["message"] = "Watch removed";
	} else {
		result["success"] = false;
		result["error"] = "Watch ID not found: " + p_id;
	}
	
	return result;
}

Dictionary RuntimeInspector::get_watch_values() {
	Dictionary result;
	Array watches;
	
	for (const KeyValue<String, String> &E : watch_expressions) {
		Dictionary watch_info;
		watch_info["id"] = E.key;
		watch_info["expression"] = E.value;
		
		if (watch_values.has(E.key)) {
			watch_info["value"] = watch_values[E.key];
		} else {
			watch_info["value"] = "<not evaluated>";
		}
		
		watches.push_back(watch_info);
	}
	
	result["success"] = true;
	result["watches"] = watches;
	result["count"] = watches.size();
	
	return result;
}

// ========== ADVANCED DIAGNOSTICS - "System Observatory" Implementation ==========

Dictionary RuntimeInspector::diagnose_node(const String &p_node_path, bool p_compare_to_editor) {
	Dictionary result;
	
	if (!_is_game_running()) {
		result["success"] = false;
		result["error"] = "Game is not running";
		return result;
	}
	
	Node *node = _get_remote_node(p_node_path);
	if (!node) {
		result["success"] = false;
		result["error"] = "Node not found: " + p_node_path;
		return result;
	}
	
	result["success"] = true;
	result["node_path"] = String(node->get_path());
	result["node_type"] = node->get_class();
	result["node_name"] = node->get_name();
	
	// Get current runtime properties
	Dictionary current_state;
	List<PropertyInfo> prop_list;
	node->get_property_list(&prop_list);
	
	for (const PropertyInfo &prop : prop_list) {
		if (prop.name.begins_with("_") || !(prop.usage & PROPERTY_USAGE_EDITOR)) {
			continue;
		}
		current_state[prop.name] = node->get(prop.name);
	}
	result["current_state"] = current_state;
	
	// List attached scripts and their effects
	Ref<Script> script = node->get_script();
	Array scripts_info;
	if (script.is_valid()) {
		Dictionary script_dict;
		script_dict["path"] = script->get_path();
		script_dict["class"] = script->get_class();
		
		// Detect if script has _process or _physics_process (property modifiers)
		Array potential_modifiers;
		if (script->has_method("_process")) {
			potential_modifiers.push_back("_process() - runs every frame");
		}
		if (script->has_method("_physics_process")) {
			potential_modifiers.push_back("_physics_process() - runs every physics frame");
		}
		if (script->has_method("_ready")) {
			potential_modifiers.push_back("_ready() - runs on scene start");
		}
		script_dict["potential_modifiers"] = potential_modifiers;
		
		scripts_info.push_back(script_dict);
	}
	result["attached_scripts"] = scripts_info;
	
	// Detect animation players affecting this node
	Array animations_affecting;
	TypedArray<Node> anim_players = node->find_children("*", "AnimationPlayer", true, false);
	for (int i = 0; i < anim_players.size(); i++) {
		Node *anim_node = Object::cast_to<Node>(anim_players[i]);
		if (anim_node) {
			Dictionary anim_dict;
			anim_dict["path"] = String(anim_node->get_path());
			anim_dict["name"] = anim_node->get_name();
			animations_affecting.push_back(anim_dict);
		}
	}
	result["animations_affecting"] = animations_affecting;
	
	// Compare to editor state if requested
	if (p_compare_to_editor) {
		Dictionary editor_state;
		Node *editor_scene = EditorNode::get_singleton()->get_edited_scene();
		if (editor_scene) {
			Node *editor_node = editor_scene->get_node_or_null(NodePath(p_node_path));
			if (editor_node) {
				List<PropertyInfo> editor_props;
				editor_node->get_property_list(&editor_props);
				for (const PropertyInfo &prop : editor_props) {
					if (prop.name.begins_with("_") || !(prop.usage & PROPERTY_USAGE_EDITOR)) {
						continue;
					}
					editor_state[prop.name] = editor_node->get(prop.name);
				}
			}
		}
		result["editor_state"] = editor_state;
		
		// Find differences
		Array differences;
		for (const Variant *key = current_state.next(); key; key = current_state.next(key)) {
			String prop_name = *key;
			if (editor_state.has(prop_name)) {
				Variant runtime_val = current_state[prop_name];
				Variant editor_val = editor_state[prop_name];
				if (runtime_val != editor_val) {
					Dictionary diff;
					diff["property"] = prop_name;
					diff["runtime_value"] = runtime_val;
					diff["editor_value"] = editor_val;
					diff["conclusion"] = "Property is being modified at runtime";
					differences.push_back(diff);
				}
			}
		}
		result["differences"] = differences;
		result["differences_count"] = differences.size();
	}
	
	// Generate diagnostic summary
	String summary = "Node: " + node->get_name() + " (" + node->get_class() + ")\n";
	summary += "Scripts attached: " + String::num_int64(scripts_info.size()) + "\n";
	if (scripts_info.size() > 0) {
		summary += "⚠ Scripts may be modifying properties in _process() or _physics_process()\n";
	}
	if (animations_affecting.size() > 0) {
		summary += "Animations affecting node: " + String::num_int64(animations_affecting.size()) + "\n";
	}
	if (p_compare_to_editor && result.has("differences_count")) {
		int diff_count = result.get("differences_count", 0);
		if (diff_count > 0) {
			summary += "⚠ " + String::num_int64(diff_count) + " properties differ between editor and runtime!\n";
			summary += "💡 Suggestion: Check scripts and animations for property modifications\n";
		}
	}
	result["diagnostic_summary"] = summary;
	
	return result;
}

Dictionary RuntimeInspector::trace_property_changes(const String &p_node_path, const String &p_property, float p_duration, bool p_include_callstack) {
	Dictionary result;
	
	if (!_is_game_running()) {
		result["success"] = false;
		result["error"] = "Game is not running";
		return result;
	}
	
	Node *node = _get_remote_node(p_node_path);
	if (!node) {
		result["success"] = false;
		result["error"] = "Node not found: " + p_node_path;
		return result;
	}
	
	// Record initial value
	Variant initial_value = node->get(p_property);
	
	// Wait for the trace duration (simplified - in production would use callbacks)
	OS::get_singleton()->delay_usec((int)(p_duration * 1000000)); // Convert seconds to microseconds
	
	// Record final value
	Variant final_value = node->get(p_property);
	
	result["success"] = true;
	result["property"] = p_property;
	result["initial_value"] = initial_value;
	result["final_value"] = final_value;
	result["changed"] = (initial_value != final_value);
	result["trace_duration"] = p_duration;
	
	if (initial_value != final_value) {
		result["conclusion"] = "Property '" + p_property + "' is being modified during runtime";
		result["suggestion"] = "Check scripts with _process() or _physics_process() methods, or active animations";
		
		// Try to identify likely modifier
		Ref<Script> script = node->get_script();
		if (script.is_valid()) {
			if (script->has_method("_process") || script->has_method("_physics_process")) {
				result["likely_modifier"] = script->get_path() + " (_process or _physics_process method)";
			}
		}
	} else {
		result["conclusion"] = "Property '" + p_property + "' remained stable over " + String::num(p_duration, 2) + " seconds";
	}
	
	return result;
}

Dictionary RuntimeInspector::analyze_script_effects(const String &p_script_path) {
	Dictionary result;
	
	if (!_is_game_running()) {
		result["success"] = false;
		result["error"] = "Game is not running";
		return result;
	}
	
	result["success"] = true;
	result["script_path"] = p_script_path;
	
	// Find all nodes using this script in the running scene
	Node *scene_root = _get_running_scene_root();
	if (!scene_root) {
		result["success"] = false;
		result["error"] = "No running scene found";
		return result;
	}
	
	Array affected_nodes;
	TypedArray<Node> all_nodes = scene_root->find_children("*", "", true, false);
	
	for (int i = 0; i < all_nodes.size(); i++) {
		Node *node = Object::cast_to<Node>(all_nodes[i]);
		if (node) {
			Ref<Script> node_script = node->get_script();
			if (node_script.is_valid() && node_script->get_path() == p_script_path) {
				Dictionary node_info;
				node_info["path"] = String(node->get_path());
				node_info["name"] = node->get_name();
				node_info["type"] = node->get_class();
				affected_nodes.push_back(node_info);
			}
		}
	}
	
	result["affected_nodes"] = affected_nodes;
	result["affected_count"] = affected_nodes.size();
	
	// Analyze script content to detect property modifications
	Ref<FileAccess> file = FileAccess::open(p_script_path, FileAccess::READ);
	if (file.is_valid()) {
		String script_content = file->get_as_text();
		file->close();
		
		// Simple pattern detection for property assignments
		Array detected_modifications;
		PackedStringArray lines = script_content.split("\n");
		
		for (int i = 0; i < lines.size(); i++) {
			String line = lines[i].strip_edges();
			
			// Detect property assignments (self.property = value or just property = value)
			if (line.contains(" = ") && !line.begins_with("#") && !line.begins_with("var ")) {
				// Check for common property patterns
				if (line.contains("position") || line.contains("rotation") || line.contains("scale") ||
					line.contains("emission") || line.contains("energy") || line.contains("color") ||
					line.contains("material") || line.contains("modulate")) {
					
					Dictionary mod_info;
					mod_info["line_number"] = i + 1;
					mod_info["code"] = line;
					
					// Extract property name (simplified)
					String prop_name = "";
					if (line.contains(".")) {
						prop_name = line.get_slice("=", 0).get_slice(".", 1).strip_edges();
					} else {
						prop_name = line.get_slice("=", 0).strip_edges();
					}
					mod_info["property"] = prop_name;
					
					detected_modifications.push_back(mod_info);
				}
			}
		}
		
		result["detected_modifications"] = detected_modifications;
		result["modification_count"] = detected_modifications.size();
		
		// Detect process methods
		Array process_methods;
		if (script_content.contains("func _process(")) {
			process_methods.push_back("_process() - runs every frame");
		}
		if (script_content.contains("func _physics_process(")) {
			process_methods.push_back("_physics_process() - runs every physics frame");
		}
		result["process_methods"] = process_methods;
		
		// Generate diagnostic conclusion
		if (detected_modifications.size() > 0 && process_methods.size() > 0) {
			result["conclusion"] = "⚠ Script modifies " + String::num_int64(detected_modifications.size()) + " properties in process methods - these will override manual changes!";
			result["suggestion"] = "To fix conflicts: Either modify the script parameters, disable the script temporarily, or set values in the script instead of inspector";
		}
	}
	
	return result;
}

Dictionary RuntimeInspector::list_node_scripts(const String &p_node_path) {
	Dictionary result;
	
	if (!_is_game_running()) {
		result["success"] = false;
		result["error"] = "Game is not running";
		return result;
	}
	
	Node *node = _get_remote_node(p_node_path);
	if (!node) {
		result["success"] = false;
		result["error"] = "Node not found: " + p_node_path;
		return result;
	}
	
	result["success"] = true;
	result["node_path"] = String(node->get_path());
	
	// Get direct script
	Array scripts;
	Ref<Script> script = node->get_script();
	if (script.is_valid()) {
		Dictionary script_dict;
		script_dict["path"] = script->get_path();
		script_dict["type"] = "attached";
		script_dict["enabled"] = true; // Can't easily detect if disabled
		scripts.push_back(script_dict);
	}
	
	result["scripts"] = scripts;
	result["script_count"] = scripts.size();
	
	return result;
}

Dictionary RuntimeInspector::get_node_full_state(const String &p_node_path) {
	// Combines diagnose_node with property trace results for comprehensive view
	Dictionary result = diagnose_node(p_node_path, true);
	
	if (result.get("success", false)) {
		// Add runtime-specific context
		Node *node = _get_remote_node(p_node_path);
		if (node) {
			// Add process mode info
			result["process_mode"] = node->get_process_mode();
			result["physics_process_enabled"] = node->can_process();
			
			// Add tree position context
			result["parent"] = node->get_parent() ? String(node->get_parent()->get_path()) : "";
			result["child_count"] = node->get_child_count();
			
			// Add visibility/active state
			if (CanvasItem *canvas_item = Object::cast_to<CanvasItem>(node)) {
				result["visible"] = canvas_item->is_visible_in_tree();
			} else if (Node3D *node_3d = Object::cast_to<Node3D>(node)) {
				result["visible"] = node_3d->is_visible_in_tree();
			}
		}
	}
	
	return result;
}

Dictionary RuntimeInspector::toggle_script(const String &p_node_path, const String &p_script_path, bool p_enabled) {
	Dictionary result;
	
	if (!_is_game_running()) {
		result["success"] = false;
		result["error"] = "Game is not running";
		return result;
	}
	
	Node *node = _get_remote_node(p_node_path);
	if (!node) {
		result["success"] = false;
		result["error"] = "Node not found: " + p_node_path;
		return result;
	}
	
	Ref<Script> current_script = node->get_script();
	
	if (p_enabled) {
		// Enable: Load and attach script
		if (current_script.is_valid() && current_script->get_path() == p_script_path) {
			result["success"] = true;
			result["message"] = "Script already attached and enabled";
			result["already_enabled"] = true;
			return result;
		}
		
		// Load and attach the script
		Ref<Script> script = ResourceLoader::load(p_script_path);
		if (script.is_null()) {
			result["success"] = false;
			result["error"] = "Failed to load script: " + p_script_path;
			return result;
		}
		
		node->set_script(script);
		result["success"] = true;
		result["message"] = "Script enabled: " + p_script_path;
		result["action"] = "enabled";
	} else {
		// Disable: Remove script temporarily
		if (current_script.is_null()) {
			result["success"] = true;
			result["message"] = "No script attached to disable";
			result["already_disabled"] = true;
			return result;
		}
		
		if (!p_script_path.is_empty() && current_script->get_path() != p_script_path) {
			result["success"] = false;
			result["error"] = "Script mismatch - node has different script: " + current_script->get_path();
			return result;
		}
		
		// Remove the script
		node->set_script(Ref<Script>());
		result["success"] = true;
		result["message"] = "Script disabled (removed temporarily)";
		result["action"] = "disabled";
		result["warning"] = "Script will be re-enabled if scene is reloaded";
	}
	
	return result;
}
