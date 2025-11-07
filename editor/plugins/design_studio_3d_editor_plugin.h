// --------------------------------------------------------------
// © 2025 Simplifine Corp. Original backend contribution for this Godot fork.
// Personal Non‑Commercial License applies. Commercial use requires a separate license from Simplifine.
// See LICENSES/COMPANY-NONCOMMERCIAL.md.
// --------------------------------------------------------------

#pragma once

#include "editor/plugins/editor_plugin.h"
#include "scene/gui/panel_container.h"
#include "scene/main/http_request.h"
#include "design_studio_texture_system.h"

class Button;
class Camera3D;
class CheckBox;
class DirectionalLight3D;
class AcceptDialog;
class EditorFileDialog;
class HTTPRequest;
class HSlider;
class ItemList;
class Label;
class LineEdit;
class MeshInstance3D;
class OptionButton;
class ScrollContainer;
class SubViewport;
class SubViewportContainer;
class TabContainer;
class TextureRect;
class Timer;
class VBoxContainer;

class DesignStudio3DEditor : public PanelContainer {
	GDCLASS(DesignStudio3DEditor, PanelContainer);

	// API Configuration
	const String API_URL = "https://gpu-proxy-awdwh5ovsa-uc.a.run.app";
	const String TEXTURE_API_URL = "https://gpu-proxy-976792908107.us-central1.run.app";
	const String REMESH_API_URL = "https://remesh.orcaengine.ai";
	
	// UI Elements - Left Panel
	TabContainer *mode_tabs = nullptr;
	
	// Generate tab (unified for text and image)
	OptionButton *generation_mode = nullptr; // Text or Image
	
	// Text mode UI
	VBoxContainer *text_mode_container = nullptr;
	LineEdit *prompt_input = nullptr;
	CheckBox *multiview_checkbox = nullptr;
	
	// Image mode UI  
	VBoxContainer *image_mode_container = nullptr;
	Button *select_image_button = nullptr;
	Label *image_path_label = nullptr;
	TextureRect *image_preview = nullptr;
	
	// Shared generate controls
	OptionButton *quality_selector = nullptr;
	LineEdit *target_faces_input = nullptr;
	Button *generate_button = nullptr;
	Label *status_label = nullptr;
	
	// Browse tab
	ItemList *models_list = nullptr; // Keep for compatibility during transition
	ScrollContainer *models_scroll = nullptr;
	VBoxContainer *models_container = nullptr;
	Button *load_selected_button = nullptr;
	Button *refresh_list_button = nullptr;
	Label *browse_status_label = nullptr;
	Dictionary model_rows; // Maps model_id -> row container
	Dictionary expanded_models; // Tracks which models are expanded
	
	// Current View tab (appears after model is loaded)
	VBoxContainer *current_view_tab = nullptr;
	Label *model_info_label = nullptr;
	Label *texture_status_label = nullptr;
	Button *add_texture_button = nullptr;
	Button *segment_button = nullptr;
	Button *remesh_button = nullptr;
	Button *cancel_operation_button = nullptr;
	
	// LOD UI Elements
	VBoxContainer *lod_container = nullptr;
	Button *generate_lods_button = nullptr;
	Label *lod_status_label = nullptr;
	OptionButton *lod_quality_selector = nullptr;
	CheckBox *auto_lod_checkbox = nullptr;
	Label *current_lod_label = nullptr;
	
	// Export button (shared)
	Button *export_button = nullptr;
	
	// File dialog
	EditorFileDialog *file_dialog = nullptr;
	
	// Model selection dialog
	AcceptDialog *model_selection_dialog = nullptr;
	OptionButton *model_version_selector = nullptr;
	Dictionary pending_base_model_data;
	Array pending_textured_models;
	
	// UI Elements - 3D Viewer
	SubViewportContainer *viewport_container = nullptr;
	SubViewport *viewport = nullptr;
	Camera3D *camera = nullptr;
	DirectionalLight3D *light = nullptr;
	MeshInstance3D *preview_mesh = nullptr;
	
	// LOD Viewer Controls
	VBoxContainer *viewer_controls_container = nullptr;
	class HSlider *lod_slider = nullptr;
	Label *lod_slider_label = nullptr;
	
	// HTTP Requests
	HTTPRequest *submit_request = nullptr;
	HTTPRequest *poll_request = nullptr;
	HTTPRequest *download_request = nullptr;
	HTTPRequest *browse_request = nullptr;
	HTTPRequest *textured_models_request = nullptr;
	Timer *poll_timer = nullptr;
	
	// New Texture System HTTP Requests
	HTTPRequest *texture_submit_request = nullptr;
	HTTPRequest *texture_poll_request = nullptr;
	HTTPRequest *texture_download_request = nullptr;
	Timer *texture_poll_timer = nullptr;
	
	// Texture System
	DesignStudioTextureSystem *texture_system = nullptr;

	// Remeshing
	HTTPRequest *remesh_request = nullptr;
	AcceptDialog *remesh_dialog = nullptr;
	LineEdit *remesh_faces_input = nullptr;
	
	// New Texture Generation Dialog
	AcceptDialog *texture_generation_dialog = nullptr;
	LineEdit *texture_prompt_input = nullptr;
	OptionButton *texture_type_selector = nullptr;
	OptionButton *texture_resolution_selector = nullptr;
	Button *texture_reference_button = nullptr;
	TextureRect *texture_reference_preview = nullptr;
	Label *texture_reference_label = nullptr;
	Label *texture_type_note = nullptr;
	EditorFileDialog *texture_file_dialog = nullptr;
	
	// State
	String current_job_id;
	String current_user_id; // Generated dynamically from machine ID
	bool is_generating = false;
	String current_model_path; // Path to loaded model (if not yet exported)
	Ref<Mesh> current_loaded_mesh; // Currently loaded mesh in viewer
	Dictionary current_model_data; // Data of currently loaded model
	String selected_image_path; // Path to selected image
	Ref<class ImporterMesh> current_importer_mesh; // For textured GLB models
	
	// Textured models tracking - SIMPLIFIED
	Dictionary textured_models_cache; // Maps base_model_id -> Array of textured models
	
	// New Texture Generation State
	String current_texture_job_id;
	bool is_generating_texture = false;
	String texture_prompt;
	String texture_reference_image; // Base64 encoded image
	String texture_type = "hybrid"; // text-to-texture, hybrid, pbr, single-view, image-to-texture
	
	// Model statistics
	int current_vertex_count = 0;
	int current_face_count = 0;
	int current_normal_count = 0;
	int current_texture_coord_count = 0;
	
	// LOD System
	struct LODLevel {
		Ref<Mesh> mesh;
		String model_path;
		int target_faces;
		int vertex_count;
		int face_count;
		float distance_threshold; // Distance at which this LOD becomes active
		Ref<class ImporterMesh> importer_mesh; // For textured GLB LODs
	};
	
	Vector<LODLevel> lod_levels;
	int current_lod_index = 0;
	bool auto_lod_enabled = true;
	bool is_generating_lods = false;
	int lods_generated_count = 0;
	int total_lods_to_generate = 0;
	float lod_distance_threshold_pending = 0.0f; // Temp storage for LOD generation
	
	// Download retry logic
	String download_url_to_retry;
	int download_retry_count = 0;
	Timer *download_retry_timer = nullptr;
	
	// 3D Viewer controls
	bool is_rotating = false;
	Vector2 last_mouse_pos;
	Vector3 orbit_center;
	float orbit_distance = 5.0f;
	float orbit_pitch = -20.0f; // degrees
	float orbit_yaw = 45.0f; // degrees
	
	void _setup_ui();
	void _setup_3d_viewer();
	void _setup_current_view_tab();
	void _show_current_view_tab();
	void _hide_current_view_tab();
	void _update_model_info();
	
	void _on_generate_pressed();
	void _on_job_submitted(int p_result, int p_code, const PackedStringArray &p_headers, const PackedByteArray &p_body);
	void _on_poll_timeout();
	void _on_job_status_received(int p_result, int p_code, const PackedStringArray &p_headers, const PackedByteArray &p_body);
	void _on_model_downloaded(int p_result, int p_code, const PackedStringArray &p_headers, const PackedByteArray &p_body);
	
	void _start_polling(const String &p_job_id);
	void _stop_polling();
	void _load_model_from_data(const PackedByteArray &p_data);
	Ref<ArrayMesh> _parse_obj_to_mesh(const String &p_obj_content);
	void _setup_camera_orbit();
	void _update_camera_from_orbit();
	void _on_viewport_input(const Ref<InputEvent> &p_event);
	void _on_download_retry_timeout();
	void _start_download_with_headers(const String &p_url);
	void _load_imported_mesh(const String &p_path);
	void _load_glb_directly(const String &p_path);
	MeshInstance3D *_find_mesh_instance_recursive(Node *p_node);
	class ImporterMeshInstance3D *_find_importer_mesh_instance_recursive(Node *p_node);
	void _print_node_hierarchy(Node *p_node, int p_depth);
	
	void _on_refresh_models_pressed();
	void _on_models_list_received(int p_result, int p_code, const PackedStringArray &p_headers, const PackedByteArray &p_body);
	void _on_load_selected_pressed();
	void _on_export_pressed();
	void _load_model_for_viewing(const Dictionary &p_model_data);
	
	// Textured models support - SIMPLIFIED
	void _fetch_all_user_texture_jobs();
	void _on_all_texture_jobs_received(int p_result, int p_code, const PackedStringArray &p_headers, const PackedByteArray &p_body);
	void _match_textures_with_models(const Array &p_texture_jobs);
	void _on_textured_model_selected(const String &p_textured_model_id);
	void _show_model_selection_dialog(const Dictionary &p_base_model, const Array &p_textured_models);
	void _on_model_selection_confirmed();
	
	// New expandable UI methods
	void _create_model_row(const Dictionary &p_model_data, int p_index);
	void _update_model_row_with_textures(const String &p_base_model_id, const Array &p_textured_models);
	void _on_model_row_pressed(const String &p_model_id);
	void _on_expand_button_pressed(const String &p_model_id);
	void _on_textured_option_pressed(const String &p_textured_model_id);
	
	void _on_select_image_pressed();
	void _on_image_file_selected(const String &p_path);
	void _on_generation_mode_changed(int p_index);
	String _image_to_base64(const String &p_image_path);
	String _get_or_create_persistent_user_id();
	
	// Current View tab callbacks
	void _on_add_texture_pressed();
	void _on_segment_pressed();
	void _on_remesh_pressed();
	void _on_cancel_operation_pressed();

	// Remeshing operations
	void _on_remesh_dialog_confirmed();
	void _start_remeshing(int p_target_faces);
	void _on_remesh_completed(int p_result, int p_code, const PackedStringArray &p_headers, const PackedByteArray &p_body);
	
	// Texture System Callbacks
	void _on_texture_started(const String &p_job_id);
	void _on_texture_progress(const String &p_status, const Dictionary &p_data = Dictionary());
	void _on_texture_completed(const PackedByteArray &p_model_data, const String &p_filename);
	void _on_texture_failed(const String &p_error_message);
	
	// New Texture Generation Methods
	void _setup_texture_dialog();
	void _show_texture_generation_dialog();
	void _on_texture_dialog_confirmed();
	void _on_texture_reference_pressed();
	void _on_texture_reference_selected(const String &p_path);
	void _on_texture_type_changed(int p_index);
	void _update_texture_type_tip();
	void _start_texture_generation(const String &p_prompt, const String &p_type, int p_resolution, const String &p_reference_image = "");
	void _on_texture_job_submitted(int p_result, int p_code, const PackedStringArray &p_headers, const PackedByteArray &p_body);
	void _on_texture_poll_timeout();
	void _on_texture_status_received(int p_result, int p_code, const PackedStringArray &p_headers, const PackedByteArray &p_body);
	void _on_textured_model_downloaded(int p_result, int p_code, const PackedStringArray &p_headers, const PackedByteArray &p_body);
	void _start_texture_polling(const String &p_texture_job_id);
	void _stop_texture_polling();
	void _cancel_texture_generation();
	
	// LOD operations
	void _setup_lod_ui();
	void _on_generate_lods_pressed();
	void _on_auto_lod_toggled(bool p_pressed);
	void _on_lod_quality_changed(int p_index);
	void _on_lod_slider_changed(float p_value);
	void _start_lod_generation();
	void _generate_next_lod();
	void _start_remeshing_for_lod(int p_target_faces, float p_distance_threshold);
	void _on_lod_generated(int p_result, int p_code, const PackedStringArray &p_headers, const PackedByteArray &p_body);
	void _update_lod_based_on_distance();
	void _switch_to_lod(int p_lod_index);
	void _clear_lod_levels();
	void _update_lod_info();
	void _update_lod_slider();
	
	// HTTP Request Management
	void _cancel_all_requests();

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	DesignStudio3DEditor();
};

class DesignStudio3DEditorPlugin : public EditorPlugin {
	GDCLASS(DesignStudio3DEditorPlugin, EditorPlugin);

	DesignStudio3DEditor *design_studio_editor = nullptr;

protected:
	void _notification(int p_what);

public:
	virtual String get_plugin_name() const override { return TTRC("3D Design Studio"); }
	bool has_main_screen() const override { return true; }
	virtual void make_visible(bool p_visible) override;

	DesignStudio3DEditor *get_design_studio_editor() { return design_studio_editor; }

	DesignStudio3DEditorPlugin();
	~DesignStudio3DEditorPlugin();
};

