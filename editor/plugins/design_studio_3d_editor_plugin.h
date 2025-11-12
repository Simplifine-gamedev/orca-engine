// --------------------------------------------------------------
// © 2025 Simplifine Corp. Original backend contribution for this Godot fork.
// Personal Non‑Commercial License applies. Commercial use requires a separate license from Simplifine.
// See LICENSES/COMPANY-NONCOMMERCIAL.md.
// --------------------------------------------------------------

#pragma once

#include "editor/plugins/editor_plugin.h"
#include "scene/gui/panel_container.h"
#include "scene/main/http_request.h"

class Button;
class Camera3D;
class CheckBox;
class DirectionalLight3D;
class EditorFileDialog;
class HTTPRequest;
class ItemList;
class Tree;
class TreeItem;
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
	const String SHAPE_GEN_URL = "https://shapegen.orcaengine.ai";
	const String DATABASE_SERVER_URL = "https://godot-database-server-awdwh5ovsa-uc.a.run.app";
	const String TEXTURE_API_URL = "https://texture.orcaengine.ai";
	const String REMESH_API_URL = "https://remesh.orcaengine.ai";
	
	// UI Elements - Left Panel
	TabContainer *tabs = nullptr;
	
	// Generate Tab
	OptionButton *generation_mode = nullptr;
	VBoxContainer *text_container = nullptr;
	VBoxContainer *image_container = nullptr;
	LineEdit *prompt_input = nullptr;
	CheckBox *multiview_check = nullptr;
	Button *select_image_btn = nullptr;
	Label *image_path_label = nullptr;
	TextureRect *image_preview = nullptr;
	LineEdit *image_prompt_input = nullptr;
	CheckBox *auto_multiview_check = nullptr;
	OptionButton *quality_selector = nullptr;
	Button *generate_btn = nullptr;
	Label *status_label = nullptr;
	
	// Browse Tab
	Tree *models_tree = nullptr;
	Label *browse_status_label = nullptr;
	
	// Viewer Tab (appears when model is loaded)
	VBoxContainer *viewer_tab = nullptr;
	Label *viewer_model_info = nullptr;
	Button *remesh_btn = nullptr;
	Button *texture_placeholder_btn = nullptr;
	Button *lod_placeholder_btn = nullptr;
	
	// Remesh Dialog
	class AcceptDialog *remesh_dialog = nullptr;
	LineEdit *remesh_target_faces_input = nullptr;
	
	// Export
	Button *export_button = nullptr;
	
	// File Dialog
	EditorFileDialog *file_dialog = nullptr;
	
	// Texture Generation Dialog
	class AcceptDialog *texture_dialog = nullptr;
	OptionButton *texture_type_selector = nullptr;
	LineEdit *texture_prompt_input = nullptr;
	Button *texture_image_btn = nullptr;
	Label *texture_image_label = nullptr;
	TextureRect *texture_image_preview = nullptr;
	OptionButton *texture_resolution_selector = nullptr;
	EditorFileDialog *texture_file_dialog = nullptr;
	
	// UI Elements - 3D Viewer
	SubViewportContainer *viewport_container = nullptr;
	SubViewport *viewport = nullptr;
	Camera3D *camera = nullptr;
	DirectionalLight3D *light = nullptr;
	MeshInstance3D *mesh_instance = nullptr;
	Label *model_info_label = nullptr;
	
	// HTTP Requests
	HTTPRequest *generate_request = nullptr;
	HTTPRequest *poll_request = nullptr;
	HTTPRequest *download_request = nullptr;
	HTTPRequest *browse_request = nullptr;
	Timer *poll_timer = nullptr;
	Timer *chunk_timer = nullptr; // For non-blocking OBJ processing
	
	// Texture Generation HTTP Requests
	HTTPRequest *texture_submit_request = nullptr;
	HTTPRequest *texture_poll_request = nullptr;
	HTTPRequest *texture_download_request = nullptr;
	Timer *texture_poll_timer = nullptr;
	
	// Remesh HTTP Request
	HTTPRequest *remesh_request = nullptr;
	
	// State
	String current_user_id;
	String current_job_id;
	String current_prompt; // Store prompt for export folder naming
	bool is_generating = false;
	String selected_image_path;
	String current_model_path;
	Ref<Mesh> current_loaded_mesh;
	PackedByteArray pending_model_data; // Temporary storage for async processing
	
	// Chunked processing state
	PackedStringArray obj_lines; // All lines from OBJ file
	int current_line_index = 0;
	PackedVector3Array temp_vertices;
	PackedVector2Array temp_uvs; // UV coordinates for texture mapping
	PackedVector3Array temp_normals;
	PackedInt32Array temp_indices;
	PackedVector2Array final_uvs; // Final UV array matching vertex order
	bool is_processing_chunks = false;
	
	// Texture generation state
	String current_texture_job_id;
	bool is_generating_texture = false;
	String texture_reference_image; // Base64 encoded image
	int remeshed_target_faces = 0; // Store remeshed face count for texture generation
	
	// Textured model texture data (loaded from ZIP)
	PackedByteArray albedo_texture_data;
	PackedByteArray metallic_texture_data;
	PackedByteArray roughness_texture_data;
	
	// Model statistics
	int current_vertex_count = 0;
	int current_face_count = 0;
	int current_normal_count = 0;
	
	// 3D Viewer controls
	bool is_rotating = false;
	Vector2 last_mouse_pos;
	float orbit_distance = 5.0f;
	float orbit_pitch = -20.0f;
	float orbit_yaw = 45.0f;
	
	// UI Setup Methods
	void _setup_ui();
	void _setup_generate_tab();
	void _setup_browse_tab();
	void _setup_viewer_tab();
	void _setup_3d_panel(class HSplitContainer *main_split);
	void _setup_3d_viewer();
	void _show_viewer_tab();
	void _hide_viewer_tab();
	void _update_viewer_info();
	
	// Generation Methods
	void _on_mode_changed(int index);
	void _on_select_image();
	void _on_image_selected(const String &path);
	void _on_generate();
	void _on_generate_completed(int result, int code, const PackedStringArray &headers, const PackedByteArray &body);
	void _poll_job_status();
	void _on_poll_timeout();
	void _on_status_received(int result, int code, const PackedStringArray &headers, const PackedByteArray &body);
	void _on_download_delay_finished(class Timer *delay_timer);
	void _download_model(const String &url);
	void _on_model_downloaded(int result, int code, const PackedStringArray &headers, const PackedByteArray &body);
	void _process_browse_model_delayed();
	void _process_generated_model_delayed();
	void _start_chunked_processing(const String &content, bool is_generated_model);
	void _process_obj_chunk();
	void _finish_chunked_processing(bool is_generated_model);
	
	// Browse Methods
	void _on_refresh_models();
	void _on_models_loaded(int result, int code, const PackedStringArray &headers, const PackedByteArray &body);
	void _on_model_selected();
	void _on_browse_model_downloaded(int result, int code, const PackedStringArray &headers, const PackedByteArray &body);
	void _on_textured_model_selected(const String &user_id, const String &texture_job_id);
	void _on_textured_package_downloaded(int result, int code, const PackedStringArray &headers, const PackedByteArray &body);
	
	// Export Methods
	void _on_export_pressed();
	
	// Viewer Tab Methods
	void _setup_remesh_dialog();
	void _show_remesh_dialog();
	void _on_remesh_dialog_confirmed();
	void _start_remesh_textured(int target_faces);
	void _start_remesh_regular(int target_faces);
	void _on_remesh_completed(int result, int code, const PackedStringArray &headers, const PackedByteArray &body);
	void _on_lod_placeholder();
	
	// Texture Generation Methods
	void _setup_texture_dialog();
	void _show_texture_dialog();
	void _on_texture_dialog_confirmed();
	void _on_texture_image_button();
	void _on_texture_image_selected(const String &path);
	void _on_texture_type_changed(int index);
	void _start_texture_generation(const String &prompt, const String &job_type, const String &reference_image, int resolution);
	void _on_texture_submitted(int result, int code, const PackedStringArray &headers, const PackedByteArray &body);
	void _poll_texture_status();
	void _on_texture_poll_timeout();
	void _on_texture_status_received(int result, int code, const PackedStringArray &headers, const PackedByteArray &body);
	void _on_textured_model_downloaded(int result, int code, const PackedStringArray &headers, const PackedByteArray &body);
	void _process_textured_model(const PackedByteArray &body);
	
	// 3D Viewer Methods
	void _on_viewport_input(const Ref<InputEvent> &event);
	void _setup_camera_for_model();
	void _update_camera_orbit();
	void _zoom_camera(float factor);
	
	// Utility Methods
	Ref<ArrayMesh> _parse_obj_to_mesh(const String &obj_content);
	void _calculate_model_stats(const String &obj_content);
	void _update_model_info();
	String _image_to_base64(const String &image_path);
	String _get_persistent_user_id();

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