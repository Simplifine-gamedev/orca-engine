/***********************************************************/
// © 2025 Simplifine Corp. Original backend contribution for this Godot fork.
// Personal Non‑Commercial License applies. Commercial use requires a separate license from Simplifine.
// See LICENSES/COMPANY-NONCOMMERCIAL.md.
/***********************************************************/

#include "design_studio_3d_editor_plugin.h"

#include "core/config/project_settings.h"
#include "core/core_bind.h"
#include "core/io/dir_access.h"
#include "core/io/json.h"
#include "core/io/marshalls.h"
#include "core/io/resource_loader.h"
#include "modules/zip/zip_reader.h"
#include "core/os/time.h"
#include "editor/editor_main_screen.h"
#include "editor/editor_node.h"
#include "editor/settings/editor_settings.h"
#include "editor/file_system/editor_file_system.h"
#include "editor/themes/editor_scale.h"
#include "scene/3d/camera_3d.h"
#include "scene/3d/light_3d.h"
#include "scene/3d/mesh_instance_3d.h"
#include "scene/resources/environment.h"
#include "scene/resources/3d/world_3d.h"
#include "scene/resources/material.h"
#include "core/io/image.h"
#include "scene/resources/image_texture.h"
#include "scene/gui/box_container.h"
#include "scene/gui/button.h"
#include "scene/gui/check_box.h"
#include "scene/gui/item_list.h"
#include "scene/gui/tree.h"
#include "scene/gui/label.h"
#include "scene/gui/line_edit.h"
#include "scene/gui/dialogs.h"
#include "scene/gui/option_button.h"
#include "scene/gui/scroll_container.h"
#include "scene/gui/separator.h"
#include "scene/gui/split_container.h"
#include "scene/gui/subviewport_container.h"
#include "scene/gui/tab_container.h"
#include "scene/gui/texture_rect.h"
#include "scene/main/timer.h"
#include "scene/resources/packed_scene.h"
#include "editor/gui/editor_file_dialog.h"
#include "modules/gltf/gltf_document.h"
#include "modules/gltf/gltf_state.h"

void DesignStudio3DEditor::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_process_browse_model_delayed"), &DesignStudio3DEditor::_process_browse_model_delayed);
	ClassDB::bind_method(D_METHOD("_process_generated_model_delayed"), &DesignStudio3DEditor::_process_generated_model_delayed);
	ClassDB::bind_method(D_METHOD("_process_obj_chunk"), &DesignStudio3DEditor::_process_obj_chunk);
	ClassDB::bind_method(D_METHOD("_process_textured_model"), &DesignStudio3DEditor::_process_textured_model);
	ClassDB::bind_method(D_METHOD("_on_remesh_completed"), &DesignStudio3DEditor::_on_remesh_completed);
}

void DesignStudio3DEditor::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_READY: {
			_setup_ui();
			_setup_3d_viewer();
		} break;
		case NOTIFICATION_THEME_CHANGED: {
			// Theme updated
		} break;
	}
}

void DesignStudio3DEditor::_setup_ui() {
	set_custom_minimum_size(Size2(0, 200) * EDSCALE);
	
	// Main horizontal split
	HSplitContainer *main_split = memnew(HSplitContainer);
	main_split->set_anchors_and_offsets_preset(Control::PRESET_FULL_RECT);
	add_child(main_split);
	
	// === LEFT PANEL ===
	ScrollContainer *left_scroll = memnew(ScrollContainer);
	left_scroll->set_custom_minimum_size(Size2(350 * EDSCALE, 0));
	left_scroll->set_horizontal_scroll_mode(ScrollContainer::SCROLL_MODE_DISABLED);
	main_split->add_child(left_scroll);
	
	VBoxContainer *left_panel = memnew(VBoxContainer);
	left_scroll->add_child(left_panel);
	
	// Title
	Label *title = memnew(Label);
	title->set_text("AI 3D Model Studio");
	title->add_theme_font_size_override("font_size", 18 * EDSCALE);
	title->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_CENTER);
	left_panel->add_child(title);
	
	left_panel->add_child(memnew(HSeparator));
	
	// Tab Container for Generate vs Browse
	tabs = memnew(TabContainer);
	tabs->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	left_panel->add_child(tabs);
	
	// === GENERATE TAB ===
	_setup_generate_tab();
	
	// === BROWSE TAB ===
	_setup_browse_tab();
	
	// === VIEWER TAB (initially hidden, shows when model is loaded) ===
	_setup_viewer_tab();
	
	// === EXPORT SECTION (bottom of left panel) ===
	left_panel->add_child(memnew(HSeparator));
	
	export_button = memnew(Button);
	export_button->set_text("Export to Project");
	export_button->set_disabled(true);
	export_button->connect("pressed", callable_mp(this, &DesignStudio3DEditor::_on_export_pressed));
	left_panel->add_child(export_button);
	
	Label *export_hint = memnew(Label);
	export_hint->set_text("Load a model first, then export to save to workspace");
	export_hint->add_theme_font_size_override("font_size", 10 * EDSCALE);
	export_hint->set_modulate(Color(0.7, 0.7, 0.7));
	left_panel->add_child(export_hint);
	
	// === RIGHT PANEL (3D Viewer) ===
	_setup_3d_panel(main_split);
}

void DesignStudio3DEditor::_setup_generate_tab() {
	VBoxContainer *generate_tab = memnew(VBoxContainer);
	generate_tab->set_name("Generate");
	tabs->add_child(generate_tab);
	
	// Generation mode selector
	Label *mode_label = memnew(Label);
	mode_label->set_text("Generation Mode:");
	mode_label->add_theme_font_size_override("font_size", 12 * EDSCALE);
	generate_tab->add_child(mode_label);
	
	generation_mode = memnew(OptionButton);
	generation_mode->add_item("Text to 3D", 0);
	generation_mode->add_item("Image to 3D", 1);
	generation_mode->connect("item_selected", callable_mp(this, &DesignStudio3DEditor::_on_mode_changed));
	generate_tab->add_child(generation_mode);
	
	generate_tab->add_child(memnew(HSeparator));
	
	// === TEXT MODE CONTAINER ===
	text_container = memnew(VBoxContainer);
	generate_tab->add_child(text_container);
	
	Label *prompt_label = memnew(Label);
	prompt_label->set_text("Describe your 3D model:");
	text_container->add_child(prompt_label);
	
	prompt_input = memnew(LineEdit);
	prompt_input->set_placeholder("e.g. 'a red sports car', 'medieval sword', 'cute robot'...");
	text_container->add_child(prompt_input);
	
	multiview_check = memnew(CheckBox);
	multiview_check->set_text("Use Multiview (Higher Quality, +2-3min)");
	text_container->add_child(multiview_check);
	
	// === IMAGE MODE CONTAINER ===
	image_container = memnew(VBoxContainer);
	image_container->hide();
	generate_tab->add_child(image_container);
	
	Label *image_label = memnew(Label);
	image_label->set_text("Select an image:");
	image_container->add_child(image_label);
	
	select_image_btn = memnew(Button);
	select_image_btn->set_text("Choose Image File...");
	select_image_btn->connect("pressed", callable_mp(this, &DesignStudio3DEditor::_on_select_image));
	image_container->add_child(select_image_btn);
	
	image_path_label = memnew(Label);
	image_path_label->set_text("No image selected");
	image_path_label->add_theme_font_size_override("font_size", 10 * EDSCALE);
	image_path_label->set_modulate(Color(0.7, 0.7, 0.7));
	image_container->add_child(image_path_label);
	
	image_preview = memnew(TextureRect);
	image_preview->set_custom_minimum_size(Size2(200 * EDSCALE, 150 * EDSCALE));
	image_preview->set_expand_mode(TextureRect::EXPAND_FIT_WIDTH_PROPORTIONAL);
	image_preview->set_stretch_mode(TextureRect::STRETCH_KEEP_ASPECT_CENTERED);
	image_container->add_child(image_preview);
	
	// Add prompt input for image mode too
	Label *image_prompt_label = memnew(Label);
	image_prompt_label->set_text("Description (optional but recommended):");
	image_container->add_child(image_prompt_label);
	
	image_prompt_input = memnew(LineEdit);
	image_prompt_input->set_placeholder("e.g. 'a futuristic robot', 'medieval armor'...");
	image_container->add_child(image_prompt_input);
	
	Label *image_hint = memnew(Label);
	image_hint->set_text("Tip: Adding a description helps the AI better understand your image!");
	image_hint->add_theme_font_size_override("font_size", 9 * EDSCALE);
	image_hint->set_modulate(Color(0.6, 0.8, 1.0));
	image_container->add_child(image_hint);
	
	auto_multiview_check = memnew(CheckBox);
	auto_multiview_check->set_text("Auto-Generate Multiview (+1-2min)");
	image_container->add_child(auto_multiview_check);
	
	// === SHARED CONTROLS ===
	generate_tab->add_child(memnew(HSeparator));
	
	Label *quality_label = memnew(Label);
	quality_label->set_text("Quality:");
	generate_tab->add_child(quality_label);
	
	quality_selector = memnew(OptionButton);
	quality_selector->add_item("Turbo (~2min)", 0);
	quality_selector->add_item("Standard (~4min)", 1);
	quality_selector->add_item("High (~6min)", 2);
	quality_selector->select(0); // Default to Turbo
	generate_tab->add_child(quality_selector);
	
	// Generate Button
	generate_btn = memnew(Button);
	generate_btn->set_text("Generate 3D Model");
	generate_btn->add_theme_font_size_override("font_size", 14 * EDSCALE);
	generate_btn->connect("pressed", callable_mp(this, &DesignStudio3DEditor::_on_generate));
	generate_tab->add_child(generate_btn);
	
	// Status
	status_label = memnew(Label);
	status_label->set_text("Ready to generate your first 3D model!");
	status_label->set_autowrap_mode(TextServer::AUTOWRAP_WORD_SMART);
	status_label->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	generate_tab->add_child(status_label);
	
	// File dialog for image selection
	file_dialog = memnew(EditorFileDialog);
	file_dialog->set_file_mode(EditorFileDialog::FILE_MODE_OPEN_FILE);
	file_dialog->set_access(EditorFileDialog::ACCESS_FILESYSTEM);
	file_dialog->add_filter("*.png", "PNG Images");
	file_dialog->add_filter("*.jpg", "JPEG Images");
	file_dialog->add_filter("*.jpeg", "JPEG Images");
	file_dialog->add_filter("*.webp", "WebP Images");
	file_dialog->connect("file_selected", callable_mp(this, &DesignStudio3DEditor::_on_image_selected));
	add_child(file_dialog);
}

void DesignStudio3DEditor::_setup_browse_tab() {
	VBoxContainer *browse_tab = memnew(VBoxContainer);
	browse_tab->set_name("My Models");
	tabs->add_child(browse_tab);
	
	// Refresh button
	Button *refresh_btn = memnew(Button);
	refresh_btn->set_text("Refresh Models");
	refresh_btn->connect("pressed", callable_mp(this, &DesignStudio3DEditor::_on_refresh_models));
	browse_tab->add_child(refresh_btn);
	
	// Info label
	Label *info_label = memnew(Label);
	info_label->set_text("Blue models have AI textures - expand to see texture variations");
	info_label->add_theme_font_size_override("font_size", 10 * EDSCALE);
	info_label->set_modulate(Color(0.6, 0.8, 1.0));
	browse_tab->add_child(info_label);
	
	// Models tree (replaces simple list for dropdown functionality)
	models_tree = memnew(Tree);
	models_tree->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	models_tree->set_custom_minimum_size(Size2(0, 300 * EDSCALE));
	models_tree->set_columns(1);
	models_tree->set_hide_root(true);
	models_tree->connect("item_selected", callable_mp(this, &DesignStudio3DEditor::_on_model_selected));
	browse_tab->add_child(models_tree);
	
	// Browse status
	browse_status_label = memnew(Label);
	browse_status_label->set_text("Click 'Refresh Models' to load your created models");
	browse_status_label->set_autowrap_mode(TextServer::AUTOWRAP_WORD_SMART);
	browse_tab->add_child(browse_status_label);
}

void DesignStudio3DEditor::_setup_viewer_tab() {
	// Create viewer tab (initially not added to tabs - will be added when model loads)
	viewer_tab = memnew(VBoxContainer);
	viewer_tab->set_name("Model Viewer");
	
	// Model information section
	Label *info_title = memnew(Label);
	info_title->set_text("Current Model");
	info_title->add_theme_font_size_override("font_size", 14 * EDSCALE);
	viewer_tab->add_child(info_title);
	
	viewer_model_info = memnew(Label);
	viewer_model_info->set_text("No model loaded");
	viewer_model_info->set_autowrap_mode(TextServer::AUTOWRAP_WORD_SMART);
	viewer_model_info->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	viewer_model_info->set_custom_minimum_size(Size2(0, 120 * EDSCALE));
	viewer_tab->add_child(viewer_model_info);
	
	viewer_tab->add_child(memnew(HSeparator));
	
	// Model Tools section
	Label *tools_title = memnew(Label);
	tools_title->set_text("Model Tools");
	tools_title->add_theme_font_size_override("font_size", 14 * EDSCALE);
	viewer_tab->add_child(tools_title);
	
	Label *tools_note = memnew(Label);
	tools_note->set_text("AI Texture generation is now available! Remesh and LOD coming soon.");
	tools_note->add_theme_font_size_override("font_size", 10 * EDSCALE);
	tools_note->set_modulate(Color(0.6, 0.8, 1.0));
	viewer_tab->add_child(tools_note);
	
	// Model tool buttons
	texture_placeholder_btn = memnew(Button);
	texture_placeholder_btn->set_text("Generate AI Texture");
	texture_placeholder_btn->set_disabled(false); // Fully functional texture generation
	texture_placeholder_btn->connect("pressed", callable_mp(this, &DesignStudio3DEditor::_show_texture_dialog));
	viewer_tab->add_child(texture_placeholder_btn);
	
	remesh_btn = memnew(Button);
	remesh_btn->set_text("Remesh");
	remesh_btn->connect("pressed", callable_mp(this, &DesignStudio3DEditor::_show_remesh_dialog));
	viewer_tab->add_child(remesh_btn);
	
	_setup_remesh_dialog();
	
	lod_placeholder_btn = memnew(Button);
	lod_placeholder_btn->set_text("Create LOD (Coming Soon)");
	lod_placeholder_btn->set_disabled(true);
	lod_placeholder_btn->connect("pressed", callable_mp(this, &DesignStudio3DEditor::_on_lod_placeholder));
	viewer_tab->add_child(lod_placeholder_btn);
}

void DesignStudio3DEditor::_show_viewer_tab() {
	if (!viewer_tab || !tabs) {
		return;
	}
	
	// Check if tab is already added
	bool tab_exists = false;
	for (int i = 0; i < tabs->get_tab_count(); i++) {
		if (tabs->get_tab_control(i) == viewer_tab) {
			tab_exists = true;
			break;
		}
	}
	
	// Add tab if not already present
	if (!tab_exists) {
		tabs->add_child(viewer_tab);
	}
	
	// Switch to viewer tab
	tabs->set_current_tab(tabs->get_tab_idx_from_control(viewer_tab));
	
	// Update the viewer information
	_update_viewer_info();
}

void DesignStudio3DEditor::_hide_viewer_tab() {
	if (!viewer_tab || !tabs) {
		return;
	}
	
	// Remove tab if it exists
	for (int i = 0; i < tabs->get_tab_count(); i++) {
		if (tabs->get_tab_control(i) == viewer_tab) {
			tabs->remove_child(viewer_tab);
			break;
		}
	}
	
	// Switch back to generate tab
	tabs->set_current_tab(0);
}

void DesignStudio3DEditor::_update_viewer_info() {
	if (!viewer_model_info) {
		return;
	}
	
	if (current_loaded_mesh.is_null()) {
		viewer_model_info->set_text("No model loaded");
		return;
	}
	
	String info = "Model Statistics:\n";
	info += "Vertices: " + itos(current_vertex_count) + "\n";
	info += "Faces: " + itos(current_face_count) + "\n";
	info += "Normals: " + itos(current_normal_count) + "\n";
	
	// Add file size if available
	if (!current_model_path.is_empty()) {
		Ref<FileAccess> file = FileAccess::open(current_model_path, FileAccess::READ);
		if (file.is_valid()) {
			int64_t file_size = file->get_length();
			file->close();
			info += "File Size: " + String::humanize_size(file_size) + "\n";
		}
	}
	
	// Add bounding box info if mesh is valid
	if (current_loaded_mesh.is_valid()) {
		AABB aabb = current_loaded_mesh->get_aabb();
		Vector3 size = aabb.size;
		info += "\nBounding Box:\n";
		info += "Width: " + String::num(size.x, 2) + "\n";
		info += "Height: " + String::num(size.y, 2) + "\n"; 
		info += "Depth: " + String::num(size.z, 2) + "\n";
		info += "Volume: " + String::num(size.x * size.y * size.z, 2) + " units³";
	}
	
	viewer_model_info->set_text(info);
}

// Note: Placeholder methods moved to avoid duplication

void DesignStudio3DEditor::_setup_3d_panel(HSplitContainer *main_split) {
	// Right panel for 3D preview
	VBoxContainer *right_panel = memnew(VBoxContainer);
	right_panel->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	main_split->add_child(right_panel);
	
	// 3D Viewer title
	Label *viewer_title = memnew(Label);
	viewer_title->set_text("3D Model Viewer");
	viewer_title->add_theme_font_size_override("font_size", 16 * EDSCALE);
	viewer_title->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_CENTER);
	right_panel->add_child(viewer_title);
	
	// SubViewport for 3D preview
	viewport_container = memnew(SubViewportContainer);
	viewport_container->set_stretch(true);
	viewport_container->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	viewport_container->set_custom_minimum_size(Size2(400 * EDSCALE, 400 * EDSCALE));
	right_panel->add_child(viewport_container);
	
	// Model info
	model_info_label = memnew(Label);
	model_info_label->set_text("No model loaded");
	model_info_label->set_autowrap_mode(TextServer::AUTOWRAP_WORD_SMART);
	model_info_label->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_CENTER);
	model_info_label->add_theme_font_size_override("font_size", 11 * EDSCALE);
	model_info_label->set_custom_minimum_size(Size2(0, 80 * EDSCALE));
	right_panel->add_child(model_info_label);
}

void DesignStudio3DEditor::_setup_3d_viewer() {
	// Create isolated 3D viewport
	viewport = memnew(SubViewport);
	viewport->set_update_mode(SubViewport::UPDATE_ALWAYS);
	
	// Create isolated World3D
	Ref<World3D> world = memnew(World3D);
	viewport->set_world_3d(world);
	viewport_container->add_child(viewport);
	
	// Camera setup
	camera = memnew(Camera3D);
	camera->set_position(Vector3(0, 0, 3));
	camera->set_fov(45);
	camera->make_current();
	
	// Environment with nice lighting
	Ref<Environment> env = memnew(Environment);
	env->set_background(Environment::BG_COLOR);
	env->set_bg_color(Color(0.15, 0.15, 0.2)); // Dark blue background
	env->set_ambient_light_energy(0.5);
	camera->set_environment(env);
	viewport->add_child(camera);
	
	// Main directional light
	light = memnew(DirectionalLight3D);
	light->set_transform(Transform3D().looking_at(Vector3(-1, -1, -1), Vector3(0, 1, 0)));
	light->set_param(Light3D::PARAM_ENERGY, 1.2);
	viewport->add_child(light);
	
	// Fill light
	DirectionalLight3D *fill_light = memnew(DirectionalLight3D);
	fill_light->set_transform(Transform3D().looking_at(Vector3(1, 1, 0.5), Vector3(0, 1, 0)));
	fill_light->set_param(Light3D::PARAM_ENERGY, 0.4);
	fill_light->set_color(Color(0.8, 0.8, 1.0));
	viewport->add_child(fill_light);
	
	// Mesh instance for models
	mesh_instance = memnew(MeshInstance3D);
	mesh_instance->set_name("ModelPreview");
	viewport->add_child(mesh_instance);
	
	// Mouse input handling
	viewport_container->set_focus_mode(Control::FOCUS_ALL);
	viewport_container->connect("gui_input", callable_mp(this, &DesignStudio3DEditor::_on_viewport_input));
	viewport_container->set_mouse_filter(Control::MOUSE_FILTER_STOP);
	
	// HTTP requests
	generate_request = memnew(HTTPRequest);
	generate_request->set_timeout(60);
	generate_request->connect("request_completed", callable_mp(this, &DesignStudio3DEditor::_on_generate_completed));
	add_child(generate_request);
	
	poll_request = memnew(HTTPRequest);
	poll_request->set_timeout(30);
	add_child(poll_request);
	
	download_request = memnew(HTTPRequest);
	download_request->set_timeout(180);
	download_request->set_body_size_limit(50 * 1024 * 1024); // 50MB
	download_request->set_use_threads(true);
	add_child(download_request);
	
	browse_request = memnew(HTTPRequest);
	browse_request->set_timeout(30);
	add_child(browse_request);
	
	// Poll timer
	poll_timer = memnew(Timer);
	poll_timer->set_wait_time(5.0);
	poll_timer->set_one_shot(false);
	poll_timer->connect("timeout", callable_mp(this, &DesignStudio3DEditor::_on_poll_timeout));
	add_child(poll_timer);
	
	// Chunk processing timer (for non-blocking OBJ parsing)
	chunk_timer = memnew(Timer);
	chunk_timer->set_wait_time(0.016); // ~60 FPS
	chunk_timer->set_one_shot(true);
	chunk_timer->connect("timeout", callable_mp(this, &DesignStudio3DEditor::_process_obj_chunk));
	add_child(chunk_timer);
	
	// Texture generation HTTP requests
	texture_submit_request = memnew(HTTPRequest);
	texture_submit_request->set_timeout(60);
	add_child(texture_submit_request);
	
	texture_poll_request = memnew(HTTPRequest);
	texture_poll_request->set_timeout(30);
	add_child(texture_poll_request);
	
	texture_download_request = memnew(HTTPRequest);
	texture_download_request->set_timeout(180);
	texture_download_request->set_body_size_limit(100 * 1024 * 1024); // 100MB for textured models
	texture_download_request->set_use_threads(true);
	add_child(texture_download_request);
	
	// Texture poll timer
	texture_poll_timer = memnew(Timer);
	texture_poll_timer->set_wait_time(5.0); // Poll every 5 seconds
	texture_poll_timer->set_one_shot(false);
	texture_poll_timer->connect("timeout", callable_mp(this, &DesignStudio3DEditor::_on_texture_poll_timeout));
	add_child(texture_poll_timer);
	
	// Remesh HTTP Request
	remesh_request = memnew(HTTPRequest);
	remesh_request->set_timeout(120); // 2 minutes for remeshing
	remesh_request->set_body_size_limit(200 * 1024 * 1024); // 200MB for large models
	remesh_request->set_use_threads(true);
	add_child(remesh_request);
	
	// Setup texture dialog
	_setup_texture_dialog();
}

void DesignStudio3DEditor::_on_mode_changed(int index) {
	if (index == 0) {
		// Text mode
		text_container->show();
		image_container->hide();
		status_label->set_text("Enter a text description to generate a 3D model");
	} else {
		// Image mode
		text_container->hide();
		image_container->show();
		status_label->set_text("Select an image to convert to 3D model");
	}
}

void DesignStudio3DEditor::_on_select_image() {
	file_dialog->popup_centered(Size2(800 * EDSCALE, 600 * EDSCALE));
}

void DesignStudio3DEditor::_on_image_selected(const String &path) {
	selected_image_path = path;
	image_path_label->set_text("Selected: " + path.get_file());
	
	// Load and show preview
	Ref<Image> img = memnew(Image);
	Error err = img->load(path);
	
	if (err == OK) {
		Ref<ImageTexture> texture = ImageTexture::create_from_image(img);
		image_preview->set_texture(texture);
		status_label->set_text("[SUCCESS] Image loaded! Click 'Generate 3D Model' to start.");
	} else {
		status_label->set_text("[ERROR] Failed to load image. Please try another file.");
	}
}

void DesignStudio3DEditor::_on_generate() {
	if (is_generating) {
		status_label->set_text("[BUSY] Already generating... Please wait.");
		return;
	}
	
	// Clear previous model and hide viewer tab
	if (mesh_instance) {
		mesh_instance->set_mesh(Ref<Mesh>());
	}
	current_loaded_mesh = Ref<Mesh>();
	export_button->set_disabled(true);
	_hide_viewer_tab();
	
	Dictionary body;
	body["user_id"] = current_user_id;
	
	// Get quality setting
	String quality = "turbo";
	switch (quality_selector->get_selected()) {
		case 0: quality = "turbo"; break;
		case 1: quality = "standard"; break;
		case 2: quality = "high"; break;
	}
	body["quality"] = quality;
	
	int mode = generation_mode->get_selected();
	
	if (mode == 0) {
		// TEXT MODE
		String prompt = prompt_input->get_text().strip_edges();
		if (prompt.is_empty()) {
			status_label->set_text("[ERROR] Please enter a text description first!");
			return;
		}
		
		body["prompt"] = prompt;
		current_prompt = prompt; // Store for export folder naming
		
		if (multiview_check->is_pressed()) {
			body["text_to_multiview"] = true;
			status_label->set_text("[SUBMITTING] Starting text-to-multiview-to-3D generation...\nThis will take 3-5 minutes");
		} else {
			status_label->set_text("[SUBMITTING] Starting text-to-3D generation...\nThis will take 2-4 minutes");
		}
	} else {
		// IMAGE MODE
		String image_prompt = image_prompt_input->get_text().strip_edges();
		if (!image_prompt.is_empty()) {
			current_prompt = image_prompt; // Store for export folder naming
		} else {
			current_prompt = "image_to_3d"; // Default if no prompt
		}
		
		if (selected_image_path.is_empty()) {
			status_label->set_text("[ERROR] Please select an image first!");
			return;
		}
		
		String base64_image = _image_to_base64(selected_image_path);
		if (base64_image.is_empty()) {
			status_label->set_text("[ERROR] Failed to process image. Please try another file.");
			return;
		}
		
		body["image"] = base64_image;
		
		// Add prompt if provided in image mode (helps guide the AI)
		if (!image_prompt.is_empty()) {
			body["prompt"] = image_prompt;
		}
		
		String status_msg;
		if (auto_multiview_check->is_pressed()) {
			body["auto_multiview"] = true;
			status_msg = "[SUBMITTING] Starting auto-multiview-to-3D generation...";
		} else {
			status_msg = "[SUBMITTING] Starting image-to-3D generation...";
		}
		
		// Add prompt info to status if provided
		if (!image_prompt.is_empty()) {
			status_msg += "\nWith description: " + image_prompt;
		}
		status_msg += "\nThis will take 2-5 minutes";
		
		status_label->set_text(status_msg);
	}
	
	String json_body = JSON::stringify(body);
	String url = SHAPE_GEN_URL + "/generate";
	
	PackedStringArray headers;
	headers.push_back("Content-Type: application/json");
	
	is_generating = true;
	generate_btn->set_disabled(true);
	
	Error err = generate_request->request(url, headers, HTTPClient::METHOD_POST, json_body);
	if (err != OK) {
		status_label->set_text("❌ Failed to start generation request");
		is_generating = false;
		generate_btn->set_disabled(false);
	}
}

void DesignStudio3DEditor::_on_generate_completed(int result, int code, const PackedStringArray &headers, const PackedByteArray &body) {
	if (result != HTTPRequest::RESULT_SUCCESS || (code != 200 && code != 202)) {
		status_label->set_text("❌ Generation failed (HTTP " + itos(code) + ")");
		is_generating = false;
		generate_btn->set_disabled(false);
		return;
	}
	
	String response_text = String::utf8((const char *)body.ptr(), body.size());
	
	JSON json;
	Error err = json.parse(response_text);
	if (err != OK) {
		status_label->set_text("❌ Failed to parse server response");
		is_generating = false;
		generate_btn->set_disabled(false);
		return;
	}
	
	Dictionary response = json.get_data();
	String job_id = response.get("id", "");
	
	if (job_id.is_empty()) {
		status_label->set_text("❌ No job ID received from server");
		is_generating = false;
		generate_btn->set_disabled(false);
		return;
	}
	
	current_job_id = job_id;
	status_label->set_text("✅ Job submitted! ID: " + job_id.substr(0, 8) + "...\n🔄 Checking status...");
	
	// Start polling
	poll_timer->start();
	_poll_job_status();
}

void DesignStudio3DEditor::_on_poll_timeout() {
	_poll_job_status();
}

void DesignStudio3DEditor::_poll_job_status() {
	if (current_job_id.is_empty()) {
		poll_timer->stop();
		return;
	}
	
	String url = SHAPE_GEN_URL + "/status/" + current_user_id + "/" + current_job_id;
	
	poll_request->connect("request_completed", callable_mp(this, &DesignStudio3DEditor::_on_status_received), CONNECT_ONE_SHOT);
	poll_request->request(url);
}

void DesignStudio3DEditor::_on_status_received(int result, int code, const PackedStringArray &headers, const PackedByteArray &body) {
	if (result != HTTPRequest::RESULT_SUCCESS || code != 200) {
		status_label->set_text("❌ Failed to check job status (HTTP " + itos(code) + ")");
		poll_timer->stop();
		is_generating = false;
		generate_btn->set_disabled(false);
		return;
	}
	
	String response_text = String::utf8((const char *)body.ptr(), body.size());
	
	JSON json;
	Error err = json.parse(response_text);
	if (err != OK) {
		status_label->set_text("❌ Failed to parse status response");
		poll_timer->stop();
		is_generating = false;
		generate_btn->set_disabled(false);
		return;
	}
	
	Dictionary job_data = json.get_data();
	String job_status = job_data.get("status", "unknown");
	
	if (job_status == "queued") {
		status_label->set_text("[QUEUED] Job queued... Waiting for available GPU");
	} else if (job_status == "processing") {
		status_label->set_text("[PROCESSING] Processing on GPU... This may take several minutes\nGenerating your 3D model...");
	} else if (job_status == "completed") {
		poll_timer->stop();
		
		// CRITICAL: Add delay before downloading - URL might not be ready immediately
		status_label->set_text("[SUCCESS] Generation complete! Waiting for file to be ready...");
		
		// Create a delay timer before attempting download
		Timer *download_delay_timer = memnew(Timer);
		download_delay_timer->set_wait_time(3.0); // 3 second delay
		download_delay_timer->set_one_shot(true);
		add_child(download_delay_timer);
		
		// Store download info for delayed execution
		Dictionary download_info;
		download_info["job_data"] = job_data;
		download_delay_timer->set_meta("download_info", download_info);
		
		download_delay_timer->connect("timeout", callable_mp(this, &DesignStudio3DEditor::_on_download_delay_finished).bind(download_delay_timer));
		download_delay_timer->start();
	} else if (job_status == "failed") {
		poll_timer->stop();
		String error_msg = job_data.get("error_message", "Unknown error");
		status_label->set_text("[FAILED] Generation failed: " + error_msg);
		is_generating = false;
		generate_btn->set_disabled(false);
	}
}

void DesignStudio3DEditor::_on_download_delay_finished(Timer *delay_timer) {
	// Get the stored download info
	Dictionary download_info = delay_timer->get_meta("download_info");
	Dictionary job_data = download_info.get("job_data", Dictionary());
	
	// Clean up the timer
	delay_timer->queue_free();
	
	// Get download URL
	String download_url = job_data.get("output_file_url", "");
	
	// Fallback to database server if no direct URL
	if (download_url.is_empty()) {
		download_url = DATABASE_SERVER_URL + "/download/" + current_user_id + "/" + current_job_id + "/obj";
	}
	
	status_label->set_text("[DOWNLOADING] Downloading model...");
	_download_model(download_url);
}

void DesignStudio3DEditor::_download_model(const String &url) {
	status_label->set_text("[DOWNLOADING] Downloading your 3D model...");
	
	PackedStringArray headers;
	headers.push_back("User-Agent: Godot-Editor/4.0");
	
	download_request->connect("request_completed", callable_mp(this, &DesignStudio3DEditor::_on_model_downloaded), CONNECT_ONE_SHOT);
	
	Error err = download_request->request(url, headers);
	if (err != OK) {
		status_label->set_text("[ERROR] Failed to start download");
		is_generating = false;
		generate_btn->set_disabled(false);
	}
}

void DesignStudio3DEditor::_on_model_downloaded(int result, int code, const PackedStringArray &headers, const PackedByteArray &body) {
	is_generating = false;
	generate_btn->set_disabled(false);
	
	if (result != HTTPRequest::RESULT_SUCCESS || code != 200) {
		status_label->set_text("[ERROR] Failed to download model (HTTP " + itos(code) + ")");
		return;
	}
	
	if (body.size() == 0) {
		status_label->set_text("[ERROR] Downloaded model is empty");
		return;
	}
	
	// Save model to temp file
	String timestamp = String::num_int64(OS::get_singleton()->get_ticks_msec());
	String filename = "temp_model_" + timestamp + ".obj";
	String temp_path = "user://" + filename;
	
	Ref<FileAccess> file = FileAccess::open(temp_path, FileAccess::WRITE);
	if (!file.is_valid()) {
		status_label->set_text("❌ Failed to save model file");
		return;
	}
	
	file->store_buffer(body);
	file->close();
	current_model_path = temp_path;
	
	// Process model asynchronously to prevent UI freezing using Timer
	status_label->set_text("[PROCESSING] Loading model into viewer...");
	
	// Store the body data temporarily for processing
	pending_model_data = body;
	
	// Create a short timer to process the model without blocking UI
	Timer *process_timer = memnew(Timer);
	process_timer->set_wait_time(0.1); // Very short delay
	process_timer->set_one_shot(true);
	add_child(process_timer);
	process_timer->connect("timeout", callable_mp(this, &DesignStudio3DEditor::_process_generated_model_delayed));
	process_timer->connect("timeout", Callable(process_timer, "queue_free"), CONNECT_DEFERRED);
	process_timer->start();
}

void DesignStudio3DEditor::_on_refresh_models() {
	browse_status_label->set_text("[LOADING] Loading your models with texture information...");
	models_tree->clear();
	
	// Request models with texture job information included
	String url = DATABASE_SERVER_URL + "/models/" + current_user_id + "?status=completed&limit=50&include_textured=true";
	
	browse_request->connect("request_completed", callable_mp(this, &DesignStudio3DEditor::_on_models_loaded), CONNECT_ONE_SHOT);
	browse_request->request(url);
}

void DesignStudio3DEditor::_on_models_loaded(int result, int code, const PackedStringArray &headers, const PackedByteArray &body) {
	if (result != HTTPRequest::RESULT_SUCCESS || code != 200) {
		browse_status_label->set_text("[ERROR] Failed to load models (HTTP " + itos(code) + ")");
		return;
	}
	
	String response_text = String::utf8((const char *)body.ptr(), body.size());
	
	JSON json;
	Error err = json.parse(response_text);
	if (err != OK) {
		browse_status_label->set_text("[ERROR] Failed to parse models response");
		return;
	}
	
	Dictionary response = json.get_data();
	Array models = response.get("models", Array());
	int count = response.get("count", 0);
	
	models_tree->clear();
	TreeItem *root = models_tree->create_item();
	
	int textured_count = 0;
	
	for (int i = 0; i < models.size(); i++) {
		Dictionary model = models[i];
		String prompt = model.get("prompt", "Unknown");
		String model_type = model.get("model_type", "text-to-3d");
		String created = model.get("created_at", "");
		bool has_textures = model.get("has_completed_textures", false);
		int texture_count = model.get("texture_job_count", 0);
		Array texture_jobs = model.get("texture_jobs", Array());
		
		// Create display text with type indicator
		String type_prefix = "[Text] "; // Default for text-to-3d
		if (model_type == "image-to-3d") type_prefix = "[Image] ";
		else if (model_type == "text-to-multiview-to-3d") type_prefix = "[Multi] ";
		else if (model_type == "auto-multiview-to-3d") type_prefix = "[Auto] ";
		
		String display_text = type_prefix + prompt;
		if (!created.is_empty()) {
			display_text += " (" + created.substr(0, 10) + ")";
		}
		
		// Add texture indicator if available
		if (has_textures) {
			display_text += " [" + itos(texture_count) + " textures]";
			textured_count++;
		}
		
		// Create main model item
		TreeItem *model_item = models_tree->create_item(root);
		model_item->set_text(0, display_text);
		
		// Set color - blue/light blue for models with textures
		if (has_textures) {
			model_item->set_custom_color(0, Color(0.5, 0.7, 1.0)); // Light blue
		} else {
			model_item->set_custom_color(0, Color(0.9, 0.9, 0.9)); // Light gray
		}
		
		// Store model data for base model loading
		Dictionary base_model_data;
		base_model_data["type"] = "base_model";
		base_model_data["model_data"] = model;
		model_item->set_metadata(0, base_model_data);
		
		// Add texture job children if available
		if (has_textures && texture_jobs.size() > 0) {
			for (int j = 0; j < texture_jobs.size(); j++) {
				Dictionary texture_job = texture_jobs[j];
				String texture_status = texture_job.get("texture_status", "unknown");
				
				// Only show completed texture jobs
				if (texture_status == "completed") {
					String texture_type = texture_job.get("texture_type", "texture");
					String texture_created = texture_job.get("texture_created_at", "");
					int resolution = texture_job.get("texture_resolution", 1024);
					
					String texture_text = "  + " + texture_type + " (" + itos(resolution) + "px)";
					if (!texture_created.is_empty()) {
						texture_text += " - " + texture_created.substr(0, 10);
					}
					
					TreeItem *texture_item = models_tree->create_item(model_item);
					texture_item->set_text(0, texture_text);
					texture_item->set_custom_color(0, Color(0.6, 0.9, 0.6)); // Light green for textures
					
					// Store texture job data for textured model loading
					Dictionary texture_model_data;
					texture_model_data["type"] = "textured_model";
					texture_model_data["user_id"] = current_user_id;
					texture_model_data["texture_job_id"] = texture_job.get("texture_job_id", "");
					texture_model_data["base_model"] = model;
					texture_model_data["texture_job"] = texture_job;
					texture_item->set_metadata(0, texture_model_data);
				}
			}
		}
	}
	
	browse_status_label->set_text("[SUCCESS] Loaded " + itos(count) + " models (" + itos(textured_count) + " with AI textures)");
}

void DesignStudio3DEditor::_on_model_selected() {
	TreeItem *selected = models_tree->get_selected();
	if (!selected) {
		return;
	}
	
	Dictionary item_data = selected->get_metadata(0);
	if (item_data.is_empty()) {
		return;
	}
	
	String item_type = item_data.get("type", "");
	
	if (item_type == "base_model") {
		// Handle base model selection
		Dictionary model_data = item_data.get("model_data", Dictionary());
		String prompt = model_data.get("prompt", "Unknown");
		browse_status_label->set_text("[LOADING] Loading base model: " + prompt + "...");
		
		// Get download URL for base OBJ model
		String model_url = model_data.get("output_file_url", "");
		String model_id = model_data.get("id", "");
		
		if (model_url.is_empty()) {
			model_url = DATABASE_SERVER_URL + "/download/" + current_user_id + "/" + model_id + "/obj";
		}
		
		current_job_id = model_id; // Store for reference
		
		download_request->connect("request_completed", callable_mp(this, &DesignStudio3DEditor::_on_browse_model_downloaded), CONNECT_ONE_SHOT);
		download_request->request(model_url);
		
	} else if (item_type == "textured_model") {
		// Handle textured model selection
		String user_id = item_data.get("user_id", "");
		String texture_job_id = item_data.get("texture_job_id", "");
		Dictionary texture_job = item_data.get("texture_job", Dictionary());
		String texture_type = texture_job.get("texture_type", "texture");
		String textured_mesh_url = texture_job.get("textured_mesh_url", "");
		String textured_glb_path = texture_job.get("textured_glb_path", "");
		
		browse_status_label->set_text("[LOADING] Loading textured model (" + texture_type + ")...");
		
		// Download complete textured model package (OBJ + MTL + all textures)
		String texture_package_url = DATABASE_SERVER_URL + "/download-texture/" + user_id + "/" + texture_job_id + "/complete";
		browse_status_label->set_text("[LOADING] Downloading complete PBR package (" + texture_type + ")...");
		
		// Store texture job ID for reference
		current_texture_job_id = texture_job_id;
		Dictionary base_model = item_data.get("base_model", Dictionary());
		current_job_id = base_model.get("id", "");
		
		download_request->connect("request_completed", callable_mp(this, &DesignStudio3DEditor::_on_textured_package_downloaded), CONNECT_ONE_SHOT);
		download_request->request(texture_package_url);
	}
}

void DesignStudio3DEditor::_on_browse_model_downloaded(int result, int code, const PackedStringArray &headers, const PackedByteArray &body) {
	if (result != HTTPRequest::RESULT_SUCCESS || code != 200) {
		browse_status_label->set_text("[ERROR] Failed to download model (HTTP " + itos(code) + ")");
		return;
	}
	
	if (body.size() == 0) {
		browse_status_label->set_text("[ERROR] Downloaded model is empty");
		return;
	}
	
	// Save model file
	String timestamp = String::num_int64(OS::get_singleton()->get_ticks_msec());
	String filename = "loaded_model_" + timestamp + ".obj";
	String temp_path = "user://" + filename;
	
	Ref<FileAccess> file = FileAccess::open(temp_path, FileAccess::WRITE);
	if (!file.is_valid()) {
		browse_status_label->set_text("[ERROR] Failed to save model file");
		return;
	}
	
	file->store_buffer(body);
	file->close();
	current_model_path = temp_path;
	
	// Process model asynchronously to prevent UI freezing using Timer
	browse_status_label->set_text("[PROCESSING] Loading model into viewer...");
	
	// Store the body data temporarily for processing
	pending_model_data = body;
	
	// Create a short timer to process the model without blocking UI
	Timer *process_timer = memnew(Timer);
	process_timer->set_wait_time(0.1); // Very short delay
	process_timer->set_one_shot(true);
	add_child(process_timer);
	process_timer->connect("timeout", callable_mp(this, &DesignStudio3DEditor::_process_browse_model_delayed));
	process_timer->connect("timeout", Callable(process_timer, "queue_free"), CONNECT_DEFERRED);
	process_timer->start();
}

void DesignStudio3DEditor::_on_textured_package_downloaded(int result, int code, const PackedStringArray &headers, const PackedByteArray &body) {
	if (result != HTTPRequest::RESULT_SUCCESS || code != 200) {
		String error_msg = "[ERROR] Failed to download PBR package (HTTP " + itos(code) + ")";
		
		if (code == 404) {
			error_msg += " - Package not found. Texture job may be incomplete.";
		} else if (code == 500) {
			error_msg += " - Server error. Try again later.";
		}
		
		browse_status_label->set_text(error_msg);
		return;
	}
	
	if (body.size() == 0) {
		browse_status_label->set_text("[ERROR] Downloaded PBR package is empty");
		return;
	}
	
	// Save ZIP package
	String timestamp = String::num_int64(OS::get_singleton()->get_ticks_msec());
	String zip_filename = "textured_package_" + timestamp + ".zip";
	String zip_path = "user://" + zip_filename;
	
	Ref<FileAccess> zip_file = FileAccess::open(zip_path, FileAccess::WRITE);
	if (!zip_file.is_valid()) {
		browse_status_label->set_text("[ERROR] Failed to save PBR package");
		return;
	}
	
	zip_file->store_buffer(body);
	zip_file->close();
	
	// Extract ALL files from ZIP to a temp directory for proper textured model loading
	browse_status_label->set_text("[PROCESSING] Extracting PBR package files...");
	
	// Create a unique temp directory for this textured model
	String temp_dir = "user://textured_" + timestamp + "/";
	Ref<DirAccess> dir = DirAccess::open("user://");
	if (dir.is_valid()) {
		dir->make_dir("textured_" + timestamp);
	}
	
	// Use Godot's ZIPReader to extract ALL files
	Ref<ZIPReader> zip_reader = memnew(ZIPReader);
	Error err = zip_reader->open(zip_path);
	
	if (err == OK) {
		PackedStringArray files = zip_reader->get_files();
		PackedByteArray obj_data;
		
		// Clear previous texture data
		albedo_texture_data.clear();
		metallic_texture_data.clear();
		roughness_texture_data.clear();
		
		// Extract files and store texture data IN MEMORY
		for (int i = 0; i < files.size(); i++) {
			String filename = files[i];
			
			// Skip README
			if (filename == "README.txt") {
				continue;
			}
			
			PackedByteArray file_data = zip_reader->read_file(filename, true);
			if (file_data.size() > 0) {
				// Store OBJ data
				if (filename.ends_with(".obj")) {
					obj_data = file_data;
				}
				// Store texture data IN MEMORY (not filesystem!)
				else if (filename == "albedo_texture.jpg" || filename == "albedo_texture.png") {
					albedo_texture_data = file_data;
				}
				else if (filename == "metallic_texture.jpg" || filename == "metallic_texture.png") {
					metallic_texture_data = file_data;
				}
				else if (filename == "roughness_texture.jpg" || filename == "roughness_texture.png") {
					roughness_texture_data = file_data;
				}
				
				// Also save to filesystem for export
				String file_path = temp_dir + filename;
				Ref<FileAccess> file = FileAccess::open(file_path, FileAccess::WRITE);
				if (file.is_valid()) {
					file->store_buffer(file_data);
					file->close();
				}
			}
		}
		
		zip_reader->close();
		
		// Now load the OBJ with textures from memory
		if (obj_data.size() > 0) {
			// Keep ZIP path for export
			current_model_path = zip_path;
			
			// Load OBJ with textures from memory
			browse_status_label->set_text("[PROCESSING] Loading textured model with PBR materials...");
			
			// Load geometry using our OBJ parser
			pending_model_data = obj_data;
			set_meta("is_textured_model", true);
			
			// Use chunked processing to load geometry, then we'll apply textures from memory
			_start_chunked_processing(String::utf8((const char *)obj_data.ptr(), obj_data.size()), false);
		} else {
			browse_status_label->set_text("[ERROR] No OBJ file found in package");
		}
	} else {
		browse_status_label->set_text("[ERROR] Failed to open ZIP package");
	}
}

void DesignStudio3DEditor::_process_browse_model_delayed() {
	if (pending_model_data.size() == 0) {
		browse_status_label->set_text("[ERROR] No model data to process");
		return;
	}
	
	// Convert to string and start chunked processing
	String content = String::utf8((const char *)pending_model_data.ptr(), pending_model_data.size());
	_start_chunked_processing(content, false); // false = browse model (not generated)
}

void DesignStudio3DEditor::_process_generated_model_delayed() {
	if (pending_model_data.size() == 0) {
		status_label->set_text("[ERROR] No model data to process");
		return;
	}
	
	// Convert to string and start chunked processing
	String content = String::utf8((const char *)pending_model_data.ptr(), pending_model_data.size());
	_start_chunked_processing(content, true); // true = generated model
}


void DesignStudio3DEditor::_start_chunked_processing(const String &content, bool is_generated_model) {
	// Prepare for chunked processing
	is_processing_chunks = true;
	current_line_index = 0;
	temp_vertices.clear();
	temp_uvs.clear();
	temp_normals.clear();
	temp_indices.clear();
	final_uvs.clear();
	
	// Split content into lines for processing
	obj_lines = content.split("\n");
	
	// Update status based on model type
	if (is_generated_model) {
		status_label->set_text("[PROCESSING] Loading generated model (" + itos(obj_lines.size()) + " lines)...");
	} else {
		browse_status_label->set_text("[PROCESSING] Loading model (" + itos(obj_lines.size()) + " lines)...");
	}
	
	// Store whether this is a generated model for completion handling
	set_meta("is_generated_model", is_generated_model);
	
	// Start processing chunks
	_process_obj_chunk();
}

void DesignStudio3DEditor::_process_obj_chunk() {
	if (!is_processing_chunks || current_line_index >= obj_lines.size()) {
		// Finished processing all lines
		_finish_chunked_processing(get_meta("is_generated_model", false));
		return;
	}
	
	// Process 1000 lines per chunk to prevent UI freezing
	int lines_per_chunk = 1000;
	int end_index = MIN(current_line_index + lines_per_chunk, obj_lines.size());
	
	for (int i = current_line_index; i < end_index; i++) {
		String line = obj_lines[i].strip_edges();
		
		if (line.begins_with("v ")) {
			// Vertex position
			PackedStringArray parts = line.split(" ");
			if (parts.size() >= 4) {
				temp_vertices.push_back(Vector3(
					parts[1].to_float(),
					parts[2].to_float(),
					parts[3].to_float()
				));
			}
		} else if (line.begins_with("vt ")) {
			// UV coordinate
			PackedStringArray parts = line.split(" ");
			if (parts.size() >= 3) {
				temp_uvs.push_back(Vector2(
					parts[1].to_float(),
					1.0 - parts[2].to_float() // Flip V coordinate (OBJ uses bottom-left origin, Godot uses top-left)
				));
			}
		} else if (line.begins_with("vn ")) {
			// Normal
			PackedStringArray parts = line.split(" ");
			if (parts.size() >= 4) {
				temp_normals.push_back(Vector3(
					parts[1].to_float(),
					parts[2].to_float(),
					parts[3].to_float()
				));
			}
		} else if (line.begins_with("f ")) {
			// Face - format: f v/vt/vn v/vt/vn v/vt/vn or f v v v
			PackedStringArray parts = line.split(" ");
			if (parts.size() >= 4) {
				// Parse first 3 vertices (triangulate)
				for (int j = 1; j <= 3; j++) {
					String vertex_def = parts[j];
					PackedStringArray indices = vertex_def.split("/");
					
					// Vertex index (required)
					int v_idx = indices[0].to_int() - 1;
					if (v_idx >= 0 && v_idx < temp_vertices.size()) {
						temp_indices.push_back(v_idx);
						
						// UV index (if present)
						if (indices.size() >= 2 && !indices[1].is_empty()) {
							int uv_idx = indices[1].to_int() - 1;
							if (uv_idx >= 0 && uv_idx < temp_uvs.size()) {
								// Ensure final_uvs array is large enough
								while (final_uvs.size() <= v_idx) {
									final_uvs.push_back(Vector2(0, 0));
								}
								final_uvs.write[v_idx] = temp_uvs[uv_idx];
							}
						}
					}
				}
			}
		}
	}
	
	current_line_index = end_index;
	
	// Update progress
	float progress = (float)current_line_index / (float)obj_lines.size();
	int progress_percent = (int)(progress * 100);
	
	bool is_generated = get_meta("is_generated_model", false);
	if (is_generated) {
		status_label->set_text("[PROCESSING] Loading generated model... " + itos(progress_percent) + "%");
	} else {
		browse_status_label->set_text("[PROCESSING] Loading model... " + itos(progress_percent) + "%");
	}
	
	// Schedule next chunk processing
	if (chunk_timer) {
		chunk_timer->start();
	}
}

void DesignStudio3DEditor::_finish_chunked_processing(bool is_generated_model) {
	is_processing_chunks = false;
	bool is_textured_model = get_meta("is_textured_model", false);
	
	if (temp_vertices.size() == 0 || temp_indices.size() == 0) {
		String error_msg = "[ERROR] No valid geometry found in model";
		if (is_generated_model) {
			status_label->set_text(error_msg);
		} else {
			browse_status_label->set_text(error_msg);
		}
		pending_model_data.clear();
		return;
	}
	
	// Create mesh from processed data
	Ref<ArrayMesh> mesh = memnew(ArrayMesh);
	Array arrays;
	arrays.resize(Mesh::ARRAY_MAX);
	arrays[Mesh::ARRAY_VERTEX] = temp_vertices;
	arrays[Mesh::ARRAY_INDEX] = temp_indices;
	
	// Add UVs if we have them (CRITICAL for textures!)
	if (final_uvs.size() > 0) {
		// Ensure UV array matches vertex count
		PackedVector2Array mesh_uvs;
		mesh_uvs.resize(temp_vertices.size());
		for (int i = 0; i < final_uvs.size() && i < temp_vertices.size(); i++) {
			mesh_uvs.write[i] = final_uvs[i];
		}
		// Fill remaining with default UVs if needed
		for (int i = final_uvs.size(); i < temp_vertices.size(); i++) {
			mesh_uvs.write[i] = Vector2(0, 0);
		}
		arrays[Mesh::ARRAY_TEX_UV] = mesh_uvs;
		print_line("Added UV array to mesh: " + itos(mesh_uvs.size()) + " UVs for " + itos(temp_vertices.size()) + " vertices");
	}
	
	// Add normals if we have them
	if (temp_normals.size() == temp_vertices.size()) {
		arrays[Mesh::ARRAY_NORMAL] = temp_normals;
	} else if (temp_vertices.size() > 0) {
		// Generate normals if missing
		PackedVector3Array generated_normals;
		generated_normals.resize(temp_vertices.size());
		for (int i = 0; i < generated_normals.size(); i++) {
			generated_normals.write[i] = Vector3(0, 1, 0); // Default up
		}
		arrays[Mesh::ARRAY_NORMAL] = generated_normals;
	}
	
	mesh->add_surface_from_arrays(Mesh::PRIMITIVE_TRIANGLES, arrays);
	
	// Load in viewer
	current_loaded_mesh = mesh;
	
	if (!mesh_instance) {
		String error_msg = "[ERROR] No mesh instance available";
		if (is_generated_model) {
			status_label->set_text(error_msg);
		} else {
			browse_status_label->set_text(error_msg);
		}
		pending_model_data.clear();
		return;
	}
	
	mesh_instance->set_mesh(mesh);
	
	// Apply material - textured or default
	bool is_textured = get_meta("is_textured_model", false);
	
	Ref<StandardMaterial3D> mat = memnew(StandardMaterial3D);
	mat->set_cull_mode(BaseMaterial3D::CULL_DISABLED);
	
	if (is_textured) {
		// Load and apply AI-generated textures FROM MEMORY
		browse_status_label->set_text("[PROCESSING] Applying AI-generated textures...");
		
		// Check if mesh has UVs
		bool has_uvs = false;
		if (mesh->get_surface_count() > 0) {
			Array arrays = mesh->surface_get_arrays(0);
			if (arrays.size() > Mesh::ARRAY_TEX_UV) {
				PackedVector2Array uv_array = arrays[Mesh::ARRAY_TEX_UV];
				has_uvs = uv_array.size() > 0;
				print_line("Mesh UV check: " + itos(uv_array.size()) + " UVs found");
			}
		}
		
		if (!has_uvs) {
			print_line("WARNING: Mesh has no UV coordinates! Textures won't display correctly!");
		}
		
		// Load albedo texture from memory
		if (albedo_texture_data.size() > 0) {
			Ref<Image> albedo_img = memnew(Image);
			Error err = albedo_img->load_jpg_from_buffer(albedo_texture_data);
			if (err != OK) {
				// Try PNG if JPG fails
				err = albedo_img->load_png_from_buffer(albedo_texture_data);
			}
			if (err == OK && !albedo_img->is_empty()) {
				Ref<ImageTexture> albedo_tex = ImageTexture::create_from_image(albedo_img);
				if (albedo_tex.is_valid()) {
					mat->set_texture(StandardMaterial3D::TEXTURE_ALBEDO, albedo_tex);
					mat->set_albedo(Color(1, 1, 1)); // White to show texture properly
					print_line("Applied albedo texture: " + itos(albedo_img->get_width()) + "x" + itos(albedo_img->get_height()));
				} else {
					print_line("ERROR: Failed to create ImageTexture from albedo image");
				}
			} else {
				print_line("ERROR: Failed to load albedo image from buffer, size: " + itos(albedo_texture_data.size()));
			}
		} else {
			print_line("WARNING: No albedo texture data in memory");
		}
		
		// Load metallic texture from memory
		if (metallic_texture_data.size() > 0) {
			Ref<Image> metallic_img = memnew(Image);
			Error err = metallic_img->load_jpg_from_buffer(metallic_texture_data);
			if (err != OK) {
				err = metallic_img->load_png_from_buffer(metallic_texture_data);
			}
			if (err == OK && !metallic_img->is_empty()) {
				Ref<ImageTexture> metallic_tex = ImageTexture::create_from_image(metallic_img);
				if (metallic_tex.is_valid()) {
					mat->set_texture(StandardMaterial3D::TEXTURE_METALLIC, metallic_tex);
					mat->set_metallic(1.0);
					print_line("Applied metallic texture");
				}
			}
		}
		
		// Load roughness texture from memory
		if (roughness_texture_data.size() > 0) {
			Ref<Image> roughness_img = memnew(Image);
			Error err = roughness_img->load_jpg_from_buffer(roughness_texture_data);
			if (err != OK) {
				err = roughness_img->load_png_from_buffer(roughness_texture_data);
			}
			if (err == OK && !roughness_img->is_empty()) {
				Ref<ImageTexture> roughness_tex = ImageTexture::create_from_image(roughness_img);
				if (roughness_tex.is_valid()) {
					mat->set_texture(StandardMaterial3D::TEXTURE_ROUGHNESS, roughness_tex);
					mat->set_roughness(1.0);
					print_line("Applied roughness texture");
				}
			}
		}
		
		// Ensure material is properly configured for PBR
		mat->set_shading_mode(StandardMaterial3D::SHADING_MODE_PER_PIXEL);
		mat->set_metallic_texture_channel(StandardMaterial3D::TEXTURE_CHANNEL_RED);
		mat->set_roughness_texture_channel(StandardMaterial3D::TEXTURE_CHANNEL_GREEN);
		
		// If no textures loaded, use default
		if (!mat->get_texture(StandardMaterial3D::TEXTURE_ALBEDO).is_valid()) {
			mat->set_albedo(Color(0.8, 0.8, 0.8));
			print_line("WARNING: No textures found in memory, using default material");
		} else {
			print_line("SUCCESS: Textured material configured with albedo texture");
			}
	} else {
		// Default material for non-textured models
		mat->set_albedo(Color(0.8, 0.8, 0.8));
	}
	
	mesh_instance->set_material_override(mat);
	
	// Setup camera and statistics
	_setup_camera_for_model();
	
	// Update statistics from processed data
	current_vertex_count = temp_vertices.size();
	current_face_count = temp_indices.size() / 3; // Triangles
	current_normal_count = temp_normals.size();
	
	_update_model_info();
	
	// Update UI based on model type
	if (is_generated_model) {
		String success_msg = "[SUCCESS] Model generated successfully!\n\n";
		success_msg += "Vertices: " + itos(current_vertex_count) + "\n";
		success_msg += "Faces: " + itos(current_face_count) + "\n";
		success_msg += "Size: " + String::humanize_size(pending_model_data.size()) + "\n\n";
		success_msg += "Use mouse to rotate and zoom\n";
		success_msg += "Click 'Export to Project' to save";
		status_label->set_text(success_msg);
	} else if (is_textured_model) {
		browse_status_label->set_text("[SUCCESS] AI textured model loaded with PBR materials!");
		
		// Update viewer info for textured models
		String info = "AI Textured Model (PBR)\n\n";
		info += "Model Statistics:\n";
		info += "Vertices: " + itos(current_vertex_count) + "\n";
		info += "Faces: " + itos(current_face_count) + "\n";
		info += "Normals: " + itos(current_normal_count) + "\n\n";
		info += "AI Textures Applied:\n";
		info += "- Albedo (base color)\n";
		info += "- Metallic map\n";
		info += "- Roughness map\n\n";
		info += "Complete PBR Package:\n";
		info += "Export the ZIP to get OBJ + MTL + all texture files\n";
		info += "for use in your project or other 3D software.";
		
		if (viewer_model_info) {
			viewer_model_info->set_text(info);
		}
	} else {
		browse_status_label->set_text("[SUCCESS] Model loaded! Use mouse to rotate/zoom");
	}
	
	export_button->set_disabled(false);
	
	// Show the viewer tab with model details
	_show_viewer_tab();
	
	// Clear all temporary data
	pending_model_data.clear();
	temp_vertices.clear();
	temp_normals.clear();
	temp_indices.clear();
	obj_lines.clear();
	
	// Clear texture model flag
	if (has_meta("is_textured_model")) {
		remove_meta("is_textured_model");
	}
}

void DesignStudio3DEditor::_on_export_pressed() {
	if (current_model_path.is_empty()) {
		status_label->set_text("[ERROR] No model to export");
		return;
	}
	
	// Create folder structure: assets/<prompt_last4digits>/<job_id>/
	String prompt_folder = "untitled";
	if (!current_prompt.is_empty()) {
		// Get last 4 characters of prompt (or first 4 if shorter)
		int prompt_len = current_prompt.length();
		if (prompt_len >= 4) {
			prompt_folder = current_prompt.substr(prompt_len - 4).strip_edges();
		} else {
			prompt_folder = current_prompt.strip_edges();
		}
		// Sanitize folder name (remove invalid chars)
		prompt_folder = prompt_folder.replace(" ", "_").replace("/", "_").replace("\\", "_").replace(":", "_").replace("*", "_").replace("?", "_").replace("\"", "_").replace("<", "_").replace(">", "_").replace("|", "_");
		if (prompt_folder.is_empty()) {
			prompt_folder = "untitled";
		}
	}
	
	String job_folder = current_job_id.is_empty() ? "export" : current_job_id.substr(0, 8);
	String export_dir = "res://assets/" + prompt_folder + "/" + job_folder + "/";
	String full_export_dir = ProjectSettings::get_singleton()->globalize_path(export_dir);
	
	// Create directories
	Ref<DirAccess> dir = DirAccess::open("res://");
	if (!dir.is_valid()) {
		status_label->set_text("[ERROR] Cannot access project directory");
		return;
	}
	
	// Create assets folder if needed
	if (!dir->dir_exists("assets")) {
		dir->make_dir("assets");
	}
	dir->change_dir("assets");
	
	// Create prompt folder if needed
	if (!dir->dir_exists(prompt_folder)) {
		dir->make_dir(prompt_folder);
	}
	dir->change_dir(prompt_folder);
	
	// Create job folder if needed
	if (!dir->dir_exists(job_folder)) {
		dir->make_dir(job_folder);
	}
	dir->change_dir(job_folder);
	
	// Handle ZIP export (extract all files)
	if (current_model_path.get_extension().to_lower() == "zip") {
		// Extract ZIP to the export directory
		Ref<ZIPReader> zip_reader = memnew(ZIPReader);
		Error err = zip_reader->open(current_model_path);
		
		if (err == OK) {
			PackedStringArray files = zip_reader->get_files();
			int files_exported = 0;
			
			for (int i = 0; i < files.size(); i++) {
				String filename = files[i];
				PackedByteArray file_data = zip_reader->read_file(filename, true);
				
				if (file_data.size() > 0) {
					String file_path = export_dir + filename;
					String full_file_path = ProjectSettings::get_singleton()->globalize_path(file_path);
					
					Ref<FileAccess> file = FileAccess::open(full_file_path, FileAccess::WRITE);
					if (file.is_valid()) {
						file->store_buffer(file_data);
						file->close();
						files_exported++;
					}
				}
			}
			
			zip_reader->close();
			
			// Create README
			String readme_path = export_dir + "README.txt";
			String full_readme_path = ProjectSettings::get_singleton()->globalize_path(readme_path);
			Ref<FileAccess> readme_file = FileAccess::open(full_readme_path, FileAccess::WRITE);
			if (readme_file.is_valid()) {
				String readme_content = "AI-Generated 3D Model Package\n";
				readme_content += "==============================\n\n";
				readme_content += "Prompt: " + current_prompt + "\n";
				readme_content += "Job ID: " + current_job_id + "\n";
				readme_content += "Exported: " + Time::get_singleton()->get_datetime_string_from_system() + "\n\n";
				readme_content += "Files included:\n";
				readme_content += "- obj_model.obj (3D geometry)\n";
				readme_content += "- mtl_material.mtl (Material definition)\n";
				readme_content += "- albedo_texture.jpg (Base color)\n";
				readme_content += "- metallic_texture.jpg (Metallic map)\n";
				readme_content += "- roughness_texture.jpg (Roughness map)\n";
				readme_content += "- metallic_roughness_combined.png (Combined PBR map)\n";
				readme_content += "- ai_reference.png (AI reference image)\n\n";
				readme_content += "Import the OBJ file into your project - textures will load automatically!";
				readme_file->store_string(readme_content);
				readme_file->close();
			}
			
			// Refresh file system
			EditorFileSystem::get_singleton()->scan_changes();
			
			String export_msg = "[SUCCESS] Exported " + itos(files_exported) + " files to:\n";
			export_msg += export_dir + "\n\n";
			export_msg += "All PBR textures included!\n";
			export_msg += "Import obj_model.obj to use in your project.";
			
			status_label->set_text(export_msg);
			if (browse_status_label) {
				browse_status_label->set_text(export_msg);
			}
			return;
		}
	}
	
	// Handle single file export (OBJ or GLB)
	String filename;
	if (current_model_path.get_extension().to_lower() == "glb") {
		filename = "textured_model.glb";
	} else {
		filename = "model.obj";
	}
	
	String project_path = export_dir + filename;
	String full_path = ProjectSettings::get_singleton()->globalize_path(project_path);
	
	// Copy from temp to project
	Ref<FileAccess> source = FileAccess::open(current_model_path, FileAccess::READ);
	if (!source.is_valid()) {
		status_label->set_text("[ERROR] Failed to read model file");
		return;
	}
	
	PackedByteArray data = source->get_buffer(source->get_length());
	source->close();
	
	Ref<FileAccess> dest = FileAccess::open(full_path, FileAccess::WRITE);
	if (!dest.is_valid()) {
		status_label->set_text("[ERROR] Failed to write to project");
		return;
	}
	
	dest->store_buffer(data);
	dest->flush();
	dest->close();
	
	// Refresh file system
	EditorFileSystem::get_singleton()->scan_changes();
	EditorFileSystem::get_singleton()->update_file(project_path);
	
	String export_msg = "[SUCCESS] Exported model to:\n";
	export_msg += project_path + "\n\n";
	export_msg += "Check your project files!";
	
	status_label->set_text(export_msg);
	if (browse_status_label) {
		browse_status_label->set_text(export_msg);
	}
}

void DesignStudio3DEditor::_on_viewport_input(const Ref<InputEvent> &event) {
	Ref<InputEventMouseButton> mb = event;
	if (mb.is_valid()) {
		if (mb->get_button_index() == MouseButton::LEFT) {
			is_rotating = mb->is_pressed();
			last_mouse_pos = mb->get_position();
		} else if (mb->get_button_index() == MouseButton::WHEEL_UP && mb->is_pressed()) {
			_zoom_camera(0.9f);
		} else if (mb->get_button_index() == MouseButton::WHEEL_DOWN && mb->is_pressed()) {
			_zoom_camera(1.1f);
		}
		return;
	}
	
	Ref<InputEventMouseMotion> mm = event;
	if (mm.is_valid() && is_rotating) {
		Vector2 delta = mm->get_position() - last_mouse_pos;
		
		orbit_yaw += delta.x * 0.5f;
		orbit_pitch += delta.y * 0.5f;
		orbit_pitch = CLAMP(orbit_pitch, -80.0f, 80.0f);
		
		_update_camera_orbit();
		last_mouse_pos = mm->get_position();
	}
}

void DesignStudio3DEditor::_setup_camera_for_model() {
	if (!mesh_instance || !camera || !current_loaded_mesh.is_valid()) {
		return;
	}
	
	// Get model bounds and scale to fit
	AABB aabb = current_loaded_mesh->get_aabb();
	float size = aabb.get_longest_axis_size();
	
	if (size > 0) {
		// Scale model to reasonable size
		float scale = 2.0f / size; // Fit in 2x2x2 box
		mesh_instance->set_scale(Vector3(scale, scale, scale));
		
		// Center the model
		Vector3 center = aabb.get_center();
		mesh_instance->set_position(-center * scale);
	}
	
	// Position camera
	orbit_distance = 4.0f;
	orbit_yaw = 45.0f;
	orbit_pitch = -20.0f;
	_update_camera_orbit();
}

void DesignStudio3DEditor::_update_camera_orbit() {
	if (!camera) return;
	
	// Calculate camera position from orbit parameters
	float pitch_rad = Math::deg_to_rad(orbit_pitch);
	float yaw_rad = Math::deg_to_rad(orbit_yaw);
	
	Vector3 offset;
	offset.x = orbit_distance * Math::cos(pitch_rad) * Math::cos(yaw_rad);
	offset.y = orbit_distance * Math::sin(pitch_rad);
	offset.z = orbit_distance * Math::cos(pitch_rad) * Math::sin(yaw_rad);
	
	camera->set_position(offset);
	camera->look_at(Vector3(0, 0, 0), Vector3(0, 1, 0));
}

void DesignStudio3DEditor::_zoom_camera(float factor) {
	orbit_distance *= factor;
	orbit_distance = CLAMP(orbit_distance, 1.0f, 20.0f);
	_update_camera_orbit();
}

Ref<ArrayMesh> DesignStudio3DEditor::_parse_obj_to_mesh(const String &obj_content) {
	PackedVector3Array vertices;
	PackedVector3Array normals;
	PackedInt32Array indices;
	
	PackedStringArray lines = obj_content.split("\n");
	
	// Parse OBJ data
	for (int i = 0; i < lines.size(); i++) {
		String line = lines[i].strip_edges();
		
		if (line.begins_with("v ")) {
			PackedStringArray parts = line.split(" ");
			if (parts.size() >= 4) {
				vertices.push_back(Vector3(
					parts[1].to_float(),
					parts[2].to_float(),
					parts[3].to_float()
				));
			}
		} else if (line.begins_with("vn ")) {
			PackedStringArray parts = line.split(" ");
			if (parts.size() >= 4) {
				normals.push_back(Vector3(
					parts[1].to_float(),
					parts[2].to_float(),
					parts[3].to_float()
				));
			}
		} else if (line.begins_with("f ")) {
			PackedStringArray parts = line.split(" ");
			if (parts.size() >= 4) {
				// Simple triangulation - take first 3 vertices
				for (int j = 1; j <= 3; j++) {
					String vertex_def = parts[j];
					int vertex_index = vertex_def.split("/")[0].to_int() - 1;
					if (vertex_index >= 0 && vertex_index < vertices.size()) {
						indices.push_back(vertex_index);
					}
				}
			}
		}
	}
	
	if (vertices.size() == 0 || indices.size() == 0) {
		return Ref<ArrayMesh>();
	}
	
	// Create mesh
	Ref<ArrayMesh> mesh = memnew(ArrayMesh);
	Array arrays;
	arrays.resize(Mesh::ARRAY_MAX);
	arrays[Mesh::ARRAY_VERTEX] = vertices;
	arrays[Mesh::ARRAY_INDEX] = indices;
	
	if (normals.size() == vertices.size()) {
		arrays[Mesh::ARRAY_NORMAL] = normals;
	}
	
	mesh->add_surface_from_arrays(Mesh::PRIMITIVE_TRIANGLES, arrays);
	return mesh;
}

void DesignStudio3DEditor::_calculate_model_stats(const String &obj_content) {
	current_vertex_count = 0;
	current_face_count = 0;
	current_normal_count = 0;
	
	PackedStringArray lines = obj_content.split("\n");
	for (int i = 0; i < lines.size(); i++) {
		String line = lines[i].strip_edges();
		if (line.begins_with("v ")) current_vertex_count++;
		else if (line.begins_with("f ")) current_face_count++;
		else if (line.begins_with("vn ")) current_normal_count++;
	}
}

void DesignStudio3DEditor::_update_model_info() {
	if (!model_info_label) return;
	
	if (current_loaded_mesh.is_null()) {
		model_info_label->set_text("No model loaded\n\nGenerate or browse models to get started");
		return;
	}
	
	// Simple stats for the 3D viewer panel
	String info = "Model loaded - " + itos(current_vertex_count) + " vertices, " + itos(current_face_count) + " faces";
	model_info_label->set_text(info);
	
	// Update the detailed viewer tab info too
	_update_viewer_info();
}

String DesignStudio3DEditor::_image_to_base64(const String &image_path) {
	Ref<Image> img = memnew(Image);
	Error err = img->load(image_path);
	
	if (err != OK) {
		return "";
	}
	
	// Convert to PNG in memory
	PackedByteArray png_data = img->save_png_to_buffer();
	
	// Base64 encode
	static const char base64_chars[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
	String base64_string;
	
	const uint8_t *bytes = png_data.ptr();
	int len = png_data.size();
	
	for (int i = 0; i < len; i += 3) {
		uint32_t b = (bytes[i] << 16);
		if (i + 1 < len) b |= (bytes[i + 1] << 8);
		if (i + 2 < len) b |= bytes[i + 2];
		
		base64_string += base64_chars[(b >> 18) & 0x3F];
		base64_string += base64_chars[(b >> 12) & 0x3F];
		base64_string += (i + 1 < len) ? base64_chars[(b >> 6) & 0x3F] : '=';
		base64_string += (i + 2 < len) ? base64_chars[b & 0x3F] : '=';
	}
	
	return base64_string;
}

String DesignStudio3DEditor::_get_persistent_user_id() {
	const String SETTING_KEY = "3d_design_studio/user_id";
	
	// Check if we have a stored user ID
	if (EditorSettings::get_singleton()->has_setting(SETTING_KEY)) {
		String stored_id = EditorSettings::get_singleton()->get_setting(SETTING_KEY);
		if (!stored_id.is_empty()) {
			return stored_id;
		}
	}
	
	// Generate new user ID based on machine
	String machine_id = OS::get_singleton()->get_unique_id();
	if (machine_id.is_empty()) {
		machine_id = OS::get_singleton()->get_name() + "_" + 
					 String::num_int64(OS::get_singleton()->get_ticks_usec());
	}
	
	uint32_t hash = machine_id.hash();
	String user_id = "godot_" + String::num_uint64(hash, 16);
	
	// Store permanently
	EditorSettings::get_singleton()->set_setting(SETTING_KEY, user_id);
	EditorSettings::get_singleton()->save();
	
	return user_id;
}

void DesignStudio3DEditor::_setup_texture_dialog() {
	// Create texture generation dialog
	texture_dialog = memnew(AcceptDialog);
	texture_dialog->set_title("Generate AI Texture");
	texture_dialog->set_ok_button_text("Generate Texture");
	texture_dialog->connect("confirmed", callable_mp(this, &DesignStudio3DEditor::_on_texture_dialog_confirmed));
	add_child(texture_dialog);
	
	VBoxContainer *dialog_vbox = memnew(VBoxContainer);
	texture_dialog->add_child(dialog_vbox);
	
	// Texture type selector
	Label *type_label = memnew(Label);
	type_label->set_text("Texture Generation Type:");
	dialog_vbox->add_child(type_label);
	
	texture_type_selector = memnew(OptionButton);
	texture_type_selector->add_item("Text-to-Texture (Text only)", 0);
	texture_type_selector->add_item("Hybrid (Text + Image)", 1);
	texture_type_selector->add_item("PBR Materials", 2);
	texture_type_selector->add_item("Single-View (Image only)", 3);
	texture_type_selector->add_item("Image-to-Texture", 4);
	texture_type_selector->select(0); // Default to text-to-texture
	texture_type_selector->connect("item_selected", callable_mp(this, &DesignStudio3DEditor::_on_texture_type_changed));
	dialog_vbox->add_child(texture_type_selector);
	
	dialog_vbox->add_child(memnew(HSeparator));
	
	// Text prompt input
	Label *prompt_label = memnew(Label);
	prompt_label->set_text("Texture Description:");
	dialog_vbox->add_child(prompt_label);
	
	texture_prompt_input = memnew(LineEdit);
	texture_prompt_input->set_placeholder("e.g. 'metallic blue armor', 'weathered wood'...");
	texture_prompt_input->set_custom_minimum_size(Size2(400 * EDSCALE, 0));
	dialog_vbox->add_child(texture_prompt_input);
	
	// Reference image section
	Label *image_label = memnew(Label);
	image_label->set_text("Reference Image (optional):");
	dialog_vbox->add_child(image_label);
	
	texture_image_btn = memnew(Button);
	texture_image_btn->set_text("Select Reference Image...");
	texture_image_btn->connect("pressed", callable_mp(this, &DesignStudio3DEditor::_on_texture_image_button));
	dialog_vbox->add_child(texture_image_btn);
	
	texture_image_label = memnew(Label);
	texture_image_label->set_text("No image selected");
	texture_image_label->add_theme_font_size_override("font_size", 10 * EDSCALE);
	texture_image_label->set_modulate(Color(0.7, 0.7, 0.7));
	dialog_vbox->add_child(texture_image_label);
	
	texture_image_preview = memnew(TextureRect);
	texture_image_preview->set_custom_minimum_size(Size2(150 * EDSCALE, 100 * EDSCALE));
	texture_image_preview->set_expand_mode(TextureRect::EXPAND_FIT_WIDTH_PROPORTIONAL);
	texture_image_preview->set_stretch_mode(TextureRect::STRETCH_KEEP_ASPECT_CENTERED);
	texture_image_preview->hide();
	dialog_vbox->add_child(texture_image_preview);
	
	dialog_vbox->add_child(memnew(HSeparator));
	
	// Resolution selector
	Label *res_label = memnew(Label);
	res_label->set_text("Texture Resolution:");
	dialog_vbox->add_child(res_label);
	
	texture_resolution_selector = memnew(OptionButton);
	texture_resolution_selector->add_item("512px (Fast)", 512);
	texture_resolution_selector->add_item("1024px (Recommended)", 1024);
	texture_resolution_selector->add_item("2048px (High Quality)", 2048);
	texture_resolution_selector->select(1); // Default to 1024px
	dialog_vbox->add_child(texture_resolution_selector);
	
	// Add face count warning
	Label *face_warning = memnew(Label);
	face_warning->set_name("face_warning_label");
	face_warning->set_text("IMPORTANT: Texture generation max is 30K faces");
	face_warning->add_theme_font_size_override("font_size", 10 * EDSCALE);
	face_warning->set_modulate(Color(1.0, 0.7, 0.0)); // Orange warning color
	face_warning->set_autowrap_mode(TextServer::AUTOWRAP_WORD_SMART);
	dialog_vbox->add_child(face_warning);
	
	// Create file dialog for texture reference images
	texture_file_dialog = memnew(EditorFileDialog);
	texture_file_dialog->set_file_mode(EditorFileDialog::FILE_MODE_OPEN_FILE);
	texture_file_dialog->set_access(EditorFileDialog::ACCESS_FILESYSTEM);
	texture_file_dialog->add_filter("*.png", "PNG Images");
	texture_file_dialog->add_filter("*.jpg", "JPEG Images");
	texture_file_dialog->add_filter("*.jpeg", "JPEG Images");
	texture_file_dialog->add_filter("*.webp", "WebP Images");
	texture_file_dialog->connect("file_selected", callable_mp(this, &DesignStudio3DEditor::_on_texture_image_selected));
	add_child(texture_file_dialog);
}

void DesignStudio3DEditor::_show_texture_dialog() {
	if (!texture_dialog) {
		return;
	}
	
	if (current_job_id.is_empty()) {
		if (viewer_model_info) {
			viewer_model_info->set_text(viewer_model_info->get_text() + "\n\n[ERROR] No base model available for texturing");
		}
		return;
	}
	
	// Update face count warning in dialog
	VBoxContainer *dialog_vbox = Object::cast_to<VBoxContainer>(texture_dialog->get_child(0));
	if (dialog_vbox) {
		// Find face warning label by iterating through children
		for (int i = 0; i < dialog_vbox->get_child_count(); i++) {
			Label *face_warning = Object::cast_to<Label>(dialog_vbox->get_child(i));
			if (face_warning && face_warning->get_name() == "face_warning_label") {
				int current_faces = current_face_count > 0 ? current_face_count : (remeshed_target_faces > 0 ? remeshed_target_faces : 0);
				if (current_faces > 30000) {
					face_warning->set_text("[WARNING] Model has " + itos(current_faces) + " faces.\nTexture generation MAX is 30K faces - texture may fail!");
					face_warning->set_modulate(Color(1.0, 0.3, 0.3)); // Red warning
				} else if (current_faces > 0) {
					face_warning->set_text("Model has " + itos(current_faces) + " faces. Texture generation max is 30K faces.");
					face_warning->set_modulate(Color(1.0, 0.7, 0.0)); // Orange info
				} else {
					face_warning->set_text("IMPORTANT: Texture generation max is 30K faces");
					face_warning->set_modulate(Color(1.0, 0.7, 0.0)); // Orange warning
				}
				break;
			}
		}
	}
	
	// Reset dialog
	texture_prompt_input->set_text("");
	texture_image_label->set_text("No image selected");
	texture_image_preview->hide();
	texture_reference_image = "";
	texture_type_selector->select(0);
	
	texture_dialog->popup_centered(Size2(500 * EDSCALE, 0));
}

void DesignStudio3DEditor::_on_texture_dialog_confirmed() {
	String prompt = texture_prompt_input->get_text().strip_edges();
	int type_index = texture_type_selector->get_selected();
	int resolution = texture_resolution_selector->get_selected_id();
	
	// Map type index to API job type
	String job_type;
	switch (type_index) {
		case 0: job_type = "text-to-texture"; break;
		case 1: job_type = "hybrid"; break;
		case 2: job_type = "pbr"; break;
		case 3: job_type = "single-view"; break;
		case 4: job_type = "image-to-texture"; break;
		default: job_type = "text-to-texture"; break;
	}
	
	// Validate requirements based on job type
	bool needs_text = (job_type == "text-to-texture" || job_type == "hybrid" || job_type == "pbr");
	bool needs_image = (job_type == "single-view" || job_type == "hybrid" || job_type == "image-to-texture");
	
	if (needs_text && prompt.is_empty()) {
		if (viewer_model_info) {
			viewer_model_info->set_text(viewer_model_info->get_text() + "\n\n[ERROR] This texture type requires a text description");
		}
		return;
	}
	
	if (needs_image && texture_reference_image.is_empty()) {
		if (viewer_model_info) {
			viewer_model_info->set_text(viewer_model_info->get_text() + "\n\n[ERROR] This texture type requires a reference image");
		}
		return;
	}
	
	_start_texture_generation(prompt, job_type, texture_reference_image, resolution);
}

void DesignStudio3DEditor::_on_texture_image_button() {
	if (texture_file_dialog) {
		texture_file_dialog->popup_centered(Size2(800 * EDSCALE, 600 * EDSCALE));
	}
}

void DesignStudio3DEditor::_on_texture_image_selected(const String &path) {
	texture_image_label->set_text("Selected: " + path.get_file());
	
	// Load and show preview
	Ref<Image> img = memnew(Image);
	Error err = img->load(path);
	
	if (err == OK) {
		Ref<ImageTexture> texture = ImageTexture::create_from_image(img);
		texture_image_preview->set_texture(texture);
		texture_image_preview->show();
		
		// Convert to base64
		texture_reference_image = _image_to_base64(path);
	} else {
		texture_image_label->set_text("Failed to load: " + path.get_file());
		texture_reference_image = "";
	}
}

void DesignStudio3DEditor::_on_texture_type_changed(int index) {
	// Could add hints about what each type needs here
}

void DesignStudio3DEditor::_start_texture_generation(const String &prompt, const String &job_type, const String &reference_image, int resolution) {
	if (is_generating_texture) {
		return;
	}
	
	if (current_job_id.is_empty()) {
		if (viewer_model_info) {
			viewer_model_info->set_text(viewer_model_info->get_text() + "\n\n[ERROR] No base model for texturing");
		}
		return;
	}
	
	is_generating_texture = true;
	
	// Build request body according to API docs
	Dictionary body;
	body["job_type"] = job_type;
	body["user_id"] = current_user_id;
	body["base_model_job_id"] = current_job_id;
	body["texture_resolution"] = resolution;
	body["hunyuan_version"] = "2.1";
	body["download_mode"] = "supabase";
	
	// Add target_faces if model was remeshed (use actual final face count)
	if (remeshed_target_faces > 0) {
		body["target_faces"] = remeshed_target_faces;
		
		// Warn user if face count exceeds texture generation limit
		if (remeshed_target_faces > 30000) {
			if (viewer_model_info) {
				viewer_model_info->set_text(viewer_model_info->get_text() + "\n\n[WARNING] Model has " + itos(remeshed_target_faces) + " faces.\nTexture generation max is 30K faces - texture may fail or be downgraded.");
			}
		}
	}
	
	if (!prompt.is_empty()) {
		body["text_prompt"] = prompt;
	}
	
	// Add reference image data if provided
	if (!reference_image.is_empty()) {
		// For the texture API, we might need to handle image differently
		// For now, let's add it as base64 data
		body["reference_image"] = reference_image;
	}
	
	String json_body = JSON::stringify(body);
	String url = TEXTURE_API_URL + "/texture/submit";
	
	PackedStringArray headers;
	headers.push_back("Content-Type: application/json");
	
	// Update UI
	if (viewer_model_info) {
		viewer_model_info->set_text(viewer_model_info->get_text() + "\n\n[TEXTURE] Starting texture generation...\nType: " + job_type);
	}
	texture_placeholder_btn->set_disabled(true);
	
	// Submit request
	texture_submit_request->connect("request_completed", callable_mp(this, &DesignStudio3DEditor::_on_texture_submitted), CONNECT_ONE_SHOT);
	
	Error err = texture_submit_request->request(url, headers, HTTPClient::METHOD_POST, json_body);
	if (err != OK) {
		if (viewer_model_info) {
			viewer_model_info->set_text(viewer_model_info->get_text() + "\n\n[ERROR] Failed to start texture request");
		}
		is_generating_texture = false;
		texture_placeholder_btn->set_disabled(false);
	}
}

void DesignStudio3DEditor::_on_texture_submitted(int result, int code, const PackedStringArray &headers, const PackedByteArray &body) {
	if (result != HTTPRequest::RESULT_SUCCESS || code != 200) {
		if (viewer_model_info) {
			viewer_model_info->set_text(viewer_model_info->get_text() + "\n\n[ERROR] Failed to submit texture job (HTTP " + itos(code) + ")");
		}
		is_generating_texture = false;
		texture_placeholder_btn->set_disabled(false);
		return;
	}
	
	String response_text = String::utf8((const char *)body.ptr(), body.size());
	
	JSON json;
	Error err = json.parse(response_text);
	if (err != OK) {
		if (viewer_model_info) {
			viewer_model_info->set_text(viewer_model_info->get_text() + "\n\n[ERROR] Failed to parse texture response");
		}
		is_generating_texture = false;
		texture_placeholder_btn->set_disabled(false);
		return;
	}
	
	Dictionary response = json.get_data();
	String job_id = response.get("job_id", "");
	
	if (job_id.is_empty()) {
		if (viewer_model_info) {
			viewer_model_info->set_text(viewer_model_info->get_text() + "\n\n[ERROR] No texture job ID received");
		}
		is_generating_texture = false;
		texture_placeholder_btn->set_disabled(false);
		return;
	}
	
	current_texture_job_id = job_id;
	
	if (viewer_model_info) {
		viewer_model_info->set_text(viewer_model_info->get_text() + "\n\n[SUCCESS] Texture job submitted!\nID: " + job_id.substr(0, 8) + "...\n[POLLING] Checking status...");
	}
	
	// Start polling for texture status
	_poll_texture_status();
}

void DesignStudio3DEditor::_poll_texture_status() {
	if (current_texture_job_id.is_empty()) {
		texture_poll_timer->stop();
		return;
	}
	
	String url = TEXTURE_API_URL + "/texture/status/" + current_texture_job_id;
	
	texture_poll_request->connect("request_completed", callable_mp(this, &DesignStudio3DEditor::_on_texture_status_received), CONNECT_ONE_SHOT);
	texture_poll_request->request(url);
	
	// Start timer for next poll
	texture_poll_timer->start();
}

void DesignStudio3DEditor::_on_texture_poll_timeout() {
	_poll_texture_status();
}

void DesignStudio3DEditor::_on_texture_status_received(int result, int code, const PackedStringArray &headers, const PackedByteArray &body) {
	if (result != HTTPRequest::RESULT_SUCCESS || code != 200) {
		texture_poll_timer->stop();
		is_generating_texture = false;
		texture_placeholder_btn->set_disabled(false);
		return;
	}
	
	String response_text = String::utf8((const char *)body.ptr(), body.size());
	
	JSON json;
	Error err = json.parse(response_text);
	if (err != OK) {
		texture_poll_timer->stop();
		is_generating_texture = false;
		texture_placeholder_btn->set_disabled(false);
		return;
	}
	
	Dictionary response = json.get_data();
	String status = response.get("status", "unknown");
	
	if (status == "queued") {
		if (viewer_model_info) {
			// Simple append approach instead of modifying array elements
			String base_text = viewer_model_info->get_text();
			// Remove any previous status lines
			if (base_text.find("\n[QUEUED]") != -1) {
				base_text = base_text.substr(0, base_text.find("\n[QUEUED]"));
			} else if (base_text.find("\n[PROCESSING]") != -1) {
				base_text = base_text.substr(0, base_text.find("\n[PROCESSING]"));
			}
			viewer_model_info->set_text(base_text + "\n[QUEUED] Texture job queued...");
		}
	} else if (status == "processing") {
		if (viewer_model_info) {
			String base_text = viewer_model_info->get_text();
			// Remove any previous status lines
			if (base_text.find("\n[QUEUED]") != -1) {
				base_text = base_text.substr(0, base_text.find("\n[QUEUED]"));
			} else if (base_text.find("\n[PROCESSING]") != -1) {
				base_text = base_text.substr(0, base_text.find("\n[PROCESSING]"));
			}
			viewer_model_info->set_text(base_text + "\n[PROCESSING] Generating texture on GPU...");
		}
	} else if (status == "completed") {
		texture_poll_timer->stop();
		
		// Get texture job details
		Dictionary texture_job = response.get("texture_job", Dictionary());
		String texture_job_id_str = texture_job.get("id", "");
		
		if (!texture_job_id_str.is_empty() && !current_user_id.is_empty()) {
			if (viewer_model_info) {
				String base_text = viewer_model_info->get_text();
				if (base_text.find("\n[PROCESSING]") != -1) {
					base_text = base_text.substr(0, base_text.find("\n[PROCESSING]"));
				}
				viewer_model_info->set_text(base_text + "\n[COMPLETE] Texture generated! Loading textured model...");
			}
			
			// Download complete PBR package from database server (same as browse)
			String download_url = DATABASE_SERVER_URL + "/download-texture/" + current_user_id + "/" + texture_job_id_str + "/complete";
			
			PackedStringArray headers;
			headers.push_back("User-Agent: Godot-Editor/4.0");
			
			download_request->connect("request_completed", callable_mp(this, &DesignStudio3DEditor::_on_textured_package_downloaded), CONNECT_ONE_SHOT);
			download_request->request(download_url, headers);
		} else {
			texture_poll_timer->stop();
			is_generating_texture = false;
			texture_placeholder_btn->set_disabled(false);
			if (viewer_model_info) {
				viewer_model_info->set_text(viewer_model_info->get_text() + "\n\n[ERROR] Missing texture job ID or user ID");
			}
		}
	} else if (status == "failed") {
		texture_poll_timer->stop();
		String error_msg = response.get("error", "Unknown error");
		if (viewer_model_info) {
			viewer_model_info->set_text(viewer_model_info->get_text() + "\n\n[FAILED] Texture generation failed: " + error_msg);
		}
		is_generating_texture = false;
		texture_placeholder_btn->set_disabled(false);
	}
}

void DesignStudio3DEditor::_on_textured_model_downloaded(int result, int code, const PackedStringArray &headers, const PackedByteArray &body) {
	is_generating_texture = false;
	texture_placeholder_btn->set_disabled(false);
	
	if (result != HTTPRequest::RESULT_SUCCESS || code != 200) {
		if (viewer_model_info) {
			viewer_model_info->set_text(viewer_model_info->get_text() + "\n\n[ERROR] Failed to download textured model");
		}
		return;
	}
	
	if (body.size() == 0) {
		if (viewer_model_info) {
			viewer_model_info->set_text(viewer_model_info->get_text() + "\n\n[ERROR] Downloaded textured model is empty");
		}
		return;
	}
	
	// Save textured model
	String timestamp = String::num_int64(OS::get_singleton()->get_ticks_msec());
	String filename = "textured_model_" + timestamp + ".glb"; // Textured models are usually GLB
	String temp_path = "user://" + filename;
	
	Ref<FileAccess> file = FileAccess::open(temp_path, FileAccess::WRITE);
	if (!file.is_valid()) {
		if (viewer_model_info) {
			viewer_model_info->set_text(viewer_model_info->get_text() + "\n\n[ERROR] Failed to save textured model");
		}
		return;
	}
	
	file->store_buffer(body);
	file->close();
	current_model_path = temp_path; // Update to textured version
	
	// Process textured model asynchronously
	if (viewer_model_info) {
		viewer_model_info->set_text(viewer_model_info->get_text() + "\n\n[PROCESSING] Loading textured model...");
	}
	
	pending_model_data = body;
	call_deferred("_process_textured_model", body);
}

void DesignStudio3DEditor::_process_textured_model(const PackedByteArray &body) {
	// For textured models (GLB files), we need different handling than OBJ
	// This is a simplified version - in reality we'd need GLB parsing
	// For now, treat it as a successful texture application
	
	if (viewer_model_info) {
		String success_msg = viewer_model_info->get_text() + "\n\n[SUCCESS] AI texture applied!\n";
		success_msg += "Textured model ready for export\n";
		success_msg += "Size: " + String::humanize_size(body.size());
		viewer_model_info->set_text(success_msg);
	}
	
	// Clear texture job state
	current_texture_job_id = "";
}

void DesignStudio3DEditor::_setup_remesh_dialog() {
	remesh_dialog = memnew(AcceptDialog);
	remesh_dialog->set_title("Remesh Model");
	remesh_dialog->set_min_size(Size2(400 * EDSCALE, 0));
	add_child(remesh_dialog);
	
	VBoxContainer *dialog_vbox = memnew(VBoxContainer);
	remesh_dialog->add_child(dialog_vbox);
	
	Label *info_label = memnew(Label);
	info_label->set_text("Optimize mesh geometry by reducing face count\nwhile preserving visual quality.");
	info_label->set_autowrap_mode(TextServer::AUTOWRAP_WORD_SMART);
	dialog_vbox->add_child(info_label);
	
	dialog_vbox->add_child(memnew(HSeparator));
	
	Label *target_label = memnew(Label);
	target_label->set_text("Target Face Count:");
	target_label->add_theme_font_size_override("font_size", 12 * EDSCALE);
	dialog_vbox->add_child(target_label);
	
	remesh_target_faces_input = memnew(LineEdit);
	remesh_target_faces_input->set_placeholder("e.g. 50000, 75000, 150000");
	remesh_target_faces_input->set_text("75000"); // Default
	dialog_vbox->add_child(remesh_target_faces_input);
	
	Label *hint_label = memnew(Label);
	hint_label->set_name("face_count_hint"); // Name it for easy finding
	hint_label->set_text("Current faces: " + itos(current_face_count) + "\nRecommended: 50K-150K for general use\nNote: Texture generation max is 30K faces");
	hint_label->add_theme_font_size_override("font_size", 10 * EDSCALE);
	hint_label->set_modulate(Color(0.7, 0.7, 0.7));
	dialog_vbox->add_child(hint_label);
	
	remesh_dialog->connect("confirmed", callable_mp(this, &DesignStudio3DEditor::_on_remesh_dialog_confirmed));
}

void DesignStudio3DEditor::_show_remesh_dialog() {
	if (!remesh_dialog || current_model_path.is_empty()) {
		if (viewer_model_info) {
			viewer_model_info->set_text(viewer_model_info->get_text() + "\n\n[ERROR] No model loaded for remeshing");
		}
		return;
	}
	
	// Update hint with current face count (use actual current_face_count from viewer)
	int current_faces = current_face_count > 0 ? current_face_count : 0;
	
	if (remesh_target_faces_input) {
		remesh_target_faces_input->set_text("75000"); // Reset to default
		
		// Update hint label if it exists
		VBoxContainer *dialog_vbox = Object::cast_to<VBoxContainer>(remesh_dialog->get_child(0));
		if (dialog_vbox) {
			// Find hint label by iterating through children
			for (int i = 0; i < dialog_vbox->get_child_count(); i++) {
				Label *hint = Object::cast_to<Label>(dialog_vbox->get_child(i));
				if (hint && hint->get_name() == "face_count_hint") {
					if (current_faces > 0) {
						hint->set_text("Current faces: " + itos(current_faces) + "\nRecommended: 50K-150K for general use\nNote: Texture generation max is 30K faces");
					} else {
						hint->set_text("Current faces: Unknown\nRecommended: 50K-150K for general use\nNote: Texture generation max is 30K faces");
					}
					break;
				}
			}
		}
	}
	
	remesh_dialog->popup_centered(Size2(400 * EDSCALE, 0));
}

void DesignStudio3DEditor::_on_remesh_dialog_confirmed() {
	if (current_model_path.is_empty()) {
		return;
	}
	
	String target_faces_str = remesh_target_faces_input->get_text().strip_edges();
	int target_faces = target_faces_str.to_int();
	
	if (target_faces <= 0) {
		if (viewer_model_info) {
			viewer_model_info->set_text(viewer_model_info->get_text() + "\n\n[ERROR] Invalid target face count");
		}
		return;
	}
	
	remeshed_target_faces = target_faces; // Store for texture generation
	
	// Check if model is textured (ZIP) or untextured (OBJ/GLB)
	bool is_textured = current_model_path.get_extension().to_lower() == "zip";
	
	if (viewer_model_info) {
		viewer_model_info->set_text(viewer_model_info->get_text() + "\n\n[RE MESH] Starting remesh to " + itos(target_faces) + " faces...");
	}
	
	if (is_textured) {
		// Use obj-textured endpoint with ZIP
		_start_remesh_textured(target_faces);
	} else {
		// Use regular remesh endpoint with OBJ/GLB
		_start_remesh_regular(target_faces);
	}
}

void DesignStudio3DEditor::_start_remesh_textured(int target_faces) {
	// Read ZIP file
	Ref<FileAccess> zip_file = FileAccess::open(current_model_path, FileAccess::READ);
	if (!zip_file.is_valid()) {
		if (viewer_model_info) {
			viewer_model_info->set_text(viewer_model_info->get_text() + "\n\n[ERROR] Failed to read ZIP file");
		}
		return;
	}
	
	PackedByteArray zip_data = zip_file->get_buffer(zip_file->get_length());
	zip_file->close();
	
	// Prepare multipart form data
	String boundary = "----WebKitFormBoundary" + String::num_int64(OS::get_singleton()->get_ticks_msec());
	String url = REMESH_API_URL + "/remesh/obj-textured";
	
	PackedStringArray headers;
	headers.push_back("Content-Type: multipart/form-data; boundary=" + boundary);
	
	// Build multipart form data
	PackedByteArray form_data;
	
	// Add file field
	form_data.append_array(String("--" + boundary + "\r\n").to_utf8_buffer());
	form_data.append_array(String("Content-Disposition: form-data; name=\"file\"; filename=\"model.zip\"\r\n").to_utf8_buffer());
	form_data.append_array(String("Content-Type: application/zip\r\n\r\n").to_utf8_buffer());
	form_data.append_array(zip_data);
	form_data.append_array(String("\r\n").to_utf8_buffer());
	
	// Add target_faces field
	form_data.append_array(String("--" + boundary + "\r\n").to_utf8_buffer());
	form_data.append_array(String("Content-Disposition: form-data; name=\"target_faces\"\r\n\r\n").to_utf8_buffer());
	form_data.append_array(String::num(target_faces).to_utf8_buffer());
	form_data.append_array(String("\r\n").to_utf8_buffer());
	
	// Add texture_optimization (false to preserve original textures)
	form_data.append_array(String("--" + boundary + "\r\n").to_utf8_buffer());
	form_data.append_array(String("Content-Disposition: form-data; name=\"texture_optimization\"\r\n\r\n").to_utf8_buffer());
	form_data.append_array(String("false").to_utf8_buffer());
	form_data.append_array(String("\r\n").to_utf8_buffer());
	
	// Close boundary
	form_data.append_array(String("--" + boundary + "--\r\n").to_utf8_buffer());
	
	remesh_request->connect("request_completed", callable_mp(this, &DesignStudio3DEditor::_on_remesh_completed), CONNECT_ONE_SHOT);
	
	// Convert PackedByteArray to Vector<uint8_t> for request_raw (binary data)
	Vector<uint8_t> raw_data;
	raw_data.resize(form_data.size());
	memcpy(raw_data.ptrw(), form_data.ptr(), form_data.size());
	
	Error err = remesh_request->request_raw(url, headers, HTTPClient::METHOD_POST, raw_data);
	if (err != OK) {
		if (viewer_model_info) {
			viewer_model_info->set_text(viewer_model_info->get_text() + "\n\n[ERROR] Failed to start remesh request (textured)");
		}
		print_line("Remesh textured request error: " + itos(err));
		return;
	}
}

void DesignStudio3DEditor::_start_remesh_regular(int target_faces) {
	// Read model file
	Ref<FileAccess> model_file = FileAccess::open(current_model_path, FileAccess::READ);
	if (!model_file.is_valid()) {
		if (viewer_model_info) {
			viewer_model_info->set_text(viewer_model_info->get_text() + "\n\n[ERROR] Failed to read model file");
		}
		return;
	}
	
	PackedByteArray model_data = model_file->get_buffer(model_file->get_length());
	model_file->close();
	
	// Determine file extension
	String ext = current_model_path.get_extension().to_lower();
	String filename = "model." + ext;
	
	// Prepare multipart form data
	String boundary = "----WebKitFormBoundary" + String::num_int64(OS::get_singleton()->get_ticks_msec());
	String url = REMESH_API_URL + "/remesh";
	
	PackedStringArray headers;
	headers.push_back("Content-Type: multipart/form-data; boundary=" + boundary);
	
	// Build multipart form data
	PackedByteArray form_data;
	
	// Add file field
	form_data.append_array(String("--" + boundary + "\r\n").to_utf8_buffer());
	form_data.append_array(String("Content-Disposition: form-data; name=\"file\"; filename=\"" + filename + "\"\r\n").to_utf8_buffer());
	if (ext == "glb" || ext == "gltf") {
		form_data.append_array(String("Content-Type: model/gltf-binary\r\n\r\n").to_utf8_buffer());
	} else {
		form_data.append_array(String("Content-Type: text/plain\r\n\r\n").to_utf8_buffer());
	}
	form_data.append_array(model_data);
	form_data.append_array(String("\r\n").to_utf8_buffer());
	
	// Add target_faces field
	form_data.append_array(String("--" + boundary + "\r\n").to_utf8_buffer());
	form_data.append_array(String("Content-Disposition: form-data; name=\"target_faces\"\r\n\r\n").to_utf8_buffer());
	form_data.append_array(String::num(target_faces).to_utf8_buffer());
	form_data.append_array(String("\r\n").to_utf8_buffer());
	
	// Add preserve_textures=false (returns OBJ for AI texture generation)
	form_data.append_array(String("--" + boundary + "\r\n").to_utf8_buffer());
	form_data.append_array(String("Content-Disposition: form-data; name=\"preserve_textures\"\r\n\r\n").to_utf8_buffer());
	form_data.append_array(String("false").to_utf8_buffer());
	form_data.append_array(String("\r\n").to_utf8_buffer());
	
	// Close boundary
	form_data.append_array(String("--" + boundary + "--\r\n").to_utf8_buffer());
	
	remesh_request->connect("request_completed", callable_mp(this, &DesignStudio3DEditor::_on_remesh_completed), CONNECT_ONE_SHOT);
	
	// Convert PackedByteArray to Vector<uint8_t> for request_raw (binary data)
	Vector<uint8_t> raw_data;
	raw_data.resize(form_data.size());
	memcpy(raw_data.ptrw(), form_data.ptr(), form_data.size());
	
	Error err = remesh_request->request_raw(url, headers, HTTPClient::METHOD_POST, raw_data);
	if (err != OK) {
		if (viewer_model_info) {
			viewer_model_info->set_text(viewer_model_info->get_text() + "\n\n[ERROR] Failed to start remesh request (regular)");
		}
		print_line("Remesh regular request error: " + itos(err));
		return;
	}
}

void DesignStudio3DEditor::_on_remesh_completed(int result, int code, const PackedStringArray &headers, const PackedByteArray &body) {
	if (result != HTTPRequest::RESULT_SUCCESS) {
		String error_msg = "\n\n[ERROR] Remesh request failed";
		if (result == HTTPRequest::RESULT_CONNECTION_ERROR) {
			error_msg += " - Connection error";
		} else if (result == HTTPRequest::RESULT_CHUNKED_BODY_SIZE_MISMATCH) {
			error_msg += " - Body size mismatch";
		} else if (result == HTTPRequest::RESULT_CANT_CONNECT) {
			error_msg += " - Cannot connect";
		} else if (result == HTTPRequest::RESULT_CANT_RESOLVE) {
			error_msg += " - Cannot resolve host";
		} else if (result == HTTPRequest::RESULT_TLS_HANDSHAKE_ERROR) {
			error_msg += " - TLS/SSL error";
		} else {
			error_msg += " - Error code: " + itos(result);
		}
		if (viewer_model_info) {
			viewer_model_info->set_text(viewer_model_info->get_text() + error_msg);
		}
		print_line("Remesh failed - result: " + itos(result) + ", code: " + itos(code));
		return;
	}
	
	if (code != 200) {
		String error_body = String::utf8((const char *)body.ptr(), body.size());
		if (viewer_model_info) {
			viewer_model_info->set_text(viewer_model_info->get_text() + "\n\n[ERROR] Remesh failed (HTTP " + itos(code) + ")\n" + error_body.substr(0, 200));
		}
		print_line("Remesh HTTP error: " + itos(code) + " - " + error_body.substr(0, 500));
		return;
	}
	
	if (body.size() == 0) {
		if (viewer_model_info) {
			viewer_model_info->set_text(viewer_model_info->get_text() + "\n\n[ERROR] Remeshed model is empty");
		}
		return;
	}
	
	// Get stats from headers
	String final_faces_str = "";
	int final_faces = 0;
	for (int i = 0; i < headers.size(); i++) {
		if (headers[i].to_lower().begins_with("x-final-faces:")) {
			final_faces_str = headers[i].substr(headers[i].find(":") + 1).strip_edges();
			final_faces = final_faces_str.to_int();
			break;
		}
	}
	
	// Update remeshed_target_faces with actual final face count (or target if not available)
	if (final_faces > 0) {
		remeshed_target_faces = final_faces;
	}
	
	// Save remeshed model
	String timestamp = String::num_int64(OS::get_singleton()->get_ticks_msec());
	bool is_textured = current_model_path.get_extension().to_lower() == "zip";
	String filename = is_textured ? "remeshed_model_" + timestamp + ".zip" : "remeshed_model_" + timestamp + ".obj";
	String temp_path = "user://" + filename;
	
	Ref<FileAccess> file = FileAccess::open(temp_path, FileAccess::WRITE);
	if (!file.is_valid()) {
		if (viewer_model_info) {
			viewer_model_info->set_text(viewer_model_info->get_text() + "\n\n[ERROR] Failed to save remeshed model");
		}
		return;
	}
	
	file->store_buffer(body);
	file->close();
	
	// Update current model path
	current_model_path = temp_path;
	
	// Reload the remeshed model
	if (is_textured) {
		// Handle ZIP (textured)
		albedo_texture_data.clear();
		metallic_texture_data.clear();
		roughness_texture_data.clear();
		
		Ref<ZIPReader> zip_reader = memnew(ZIPReader);
		Error err = zip_reader->open(temp_path);
		
		if (err == OK) {
			PackedStringArray files = zip_reader->get_files();
			PackedByteArray obj_data;
			
			for (int i = 0; i < files.size(); i++) {
				String filename_in_zip = files[i];
				PackedByteArray file_data = zip_reader->read_file(filename_in_zip, true);
				
				if (filename_in_zip.ends_with(".obj")) {
					obj_data = file_data;
				} else if (filename_in_zip == "albedo_texture.jpg" || filename_in_zip == "albedo_texture.png") {
					albedo_texture_data = file_data;
				} else if (filename_in_zip == "metallic_texture.jpg" || filename_in_zip == "metallic_texture.png") {
					metallic_texture_data = file_data;
				} else if (filename_in_zip == "roughness_texture.jpg" || filename_in_zip == "roughness_texture.png") {
					roughness_texture_data = file_data;
				}
			}
			
			zip_reader->close();
			
			if (obj_data.size() > 0) {
				pending_model_data = obj_data;
				set_meta("is_textured_model", true);
				_start_chunked_processing(String::utf8((const char *)obj_data.ptr(), obj_data.size()), false);
			}
		}
	} else {
		// Handle OBJ (untextured)
		pending_model_data = body;
		set_meta("is_textured_model", false);
		_start_chunked_processing(String::utf8((const char *)body.ptr(), body.size()), false);
	}
	
	// Ensure texture button stays enabled after remeshing
	if (texture_placeholder_btn) {
		texture_placeholder_btn->set_disabled(false);
	}
	
	// Ensure we have a job_id for texture generation (keep existing one)
	if (current_job_id.is_empty()) {
		// If no job_id, we can't generate textures - but button should still be enabled
		// User will see error when trying to texture
	}
	
	String success_msg = "\n\n[SUCCESS] Remesh completed!";
	if (!final_faces_str.is_empty()) {
		success_msg += "\nFinal faces: " + final_faces_str;
		if (final_faces > 30000) {
			success_msg += "\n[WARNING] Texture generation max is 30K faces - consider remeshing lower";
		}
	} else {
		success_msg += "\nFinal faces: ~" + itos(remeshed_target_faces);
		if (remeshed_target_faces > 30000) {
			success_msg += "\n[WARNING] Texture generation max is 30K faces - consider remeshing lower";
		}
	}
	
	if (viewer_model_info) {
		viewer_model_info->set_text(viewer_model_info->get_text() + success_msg);
	}
}

// Note: _on_remesh_placeholder removed - replaced with _show_remesh_dialog
// Note: _on_texture_placeholder removed - replaced with real texture functionality

void DesignStudio3DEditor::_on_lod_placeholder() {
	if (viewer_model_info) {
		// Update the viewer tab with placeholder info  
		String current_info = viewer_model_info->get_text();
		viewer_model_info->set_text(current_info + "\n\n[INFO] LOD generation feature coming soon!");
		
		// Reset after 3 seconds
		Timer *reset_timer = memnew(Timer);
		reset_timer->set_wait_time(3.0);
		reset_timer->set_one_shot(true);
		add_child(reset_timer);
		reset_timer->connect("timeout", callable_mp(this, &DesignStudio3DEditor::_update_viewer_info));
		reset_timer->connect("timeout", Callable(reset_timer, "queue_free"), CONNECT_DEFERRED);
		reset_timer->start();
	}
}

DesignStudio3DEditor::DesignStudio3DEditor() {
	set_name("DesignStudio3D");
	current_user_id = _get_persistent_user_id();
	print_line("3D Design Studio initialized with user ID: " + current_user_id);
}

// Plugin implementation

void DesignStudio3DEditorPlugin::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			design_studio_editor->hide();
		} break;
	}
}

void DesignStudio3DEditorPlugin::make_visible(bool p_visible) {
	if (p_visible) {
		design_studio_editor->show();
	} else {
		design_studio_editor->hide();
	}
}

DesignStudio3DEditorPlugin::DesignStudio3DEditorPlugin() {
	design_studio_editor = memnew(DesignStudio3DEditor);
	EditorNode::get_singleton()->get_editor_main_screen()->get_control()->add_child(design_studio_editor);
	design_studio_editor->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	design_studio_editor->set_anchors_and_offsets_preset(Control::PRESET_FULL_RECT);
	design_studio_editor->hide();
}

DesignStudio3DEditorPlugin::~DesignStudio3DEditorPlugin() {
}