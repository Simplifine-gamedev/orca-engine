/***********************************************************/
// © 2025 Simplifine Corp. Original backend contribution for this Godot fork.
// Personal Non‑Commercial License applies. Commercial use requires a separate license from Simplifine.
// See LICENSES/COMPANY-NONCOMMERCIAL.md.
/***********************************************************/


#include "design_studio_3d_editor_plugin.h"

#include "core/config/project_settings.h"
#include "core/core_bind.h"
#include "core/io/json.h"
#include "core/io/marshalls.h"
#include "core/io/resource_loader.h"
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
#include "scene/resources/camera_attributes.h"
#include "scene/resources/image_texture.h"
#include "scene/gui/box_container.h"
#include "scene/gui/button.h"
#include "scene/gui/check_box.h"
#include "scene/gui/item_list.h"
#include "scene/gui/label.h"
#include "scene/gui/line_edit.h"
#include "scene/gui/dialogs.h"
#include "scene/gui/option_button.h"
#include "scene/gui/scroll_container.h"
#include "scene/gui/separator.h"
#include "scene/gui/slider.h"
#include "scene/gui/split_container.h"
#include "scene/gui/subviewport_container.h"
#include "scene/gui/tab_container.h"
#include "scene/gui/texture_rect.h"
#include "scene/main/timer.h"
#include "scene/resources/3d/primitive_meshes.h"
#include "scene/resources/packed_scene.h"
#include "scene/resources/style_box_flat.h"
#include "editor/gui/editor_file_dialog.h"

void DesignStudio3DEditor::_bind_methods() {
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
	HSplitContainer *hsplit = memnew(HSplitContainer);
	hsplit->set_anchors_and_offsets_preset(Control::PRESET_FULL_RECT);
	add_child(hsplit);
	
	// Left panel with scroll container
	ScrollContainer *scroll_container = memnew(ScrollContainer);
	scroll_container->set_custom_minimum_size(Size2(320 * EDSCALE, 0));
	scroll_container->set_horizontal_scroll_mode(ScrollContainer::SCROLL_MODE_DISABLED);
	scroll_container->set_vertical_scroll_mode(ScrollContainer::SCROLL_MODE_AUTO);
	hsplit->add_child(scroll_container);
	
	VBoxContainer *left_panel = memnew(VBoxContainer);
	left_panel->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	scroll_container->add_child(left_panel);
	
	// Title
	Label *title = memnew(Label);
	title->set_text("3D Model Studio");
	title->add_theme_font_size_override("font_size", 16 * EDSCALE);
	left_panel->add_child(title);
	
	// Tab container for Generate vs Browse
	mode_tabs = memnew(TabContainer);
	mode_tabs->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	left_panel->add_child(mode_tabs);
	
	// === GENERATE TAB (Unified Text/Image) ===
	VBoxContainer *generate_tab = memnew(VBoxContainer);
	generate_tab->set_name("Generate");
	mode_tabs->add_child(generate_tab);
	
	// Mode selector
	generation_mode = memnew(OptionButton);
	generation_mode->add_item("From Text", 0);
	generation_mode->add_item("From Image", 1);
	generation_mode->connect("item_selected", callable_mp(this, &DesignStudio3DEditor::_on_generation_mode_changed));
	generate_tab->add_child(generation_mode);
	
	generate_tab->add_child(memnew(HSeparator));
	
	// TEXT MODE CONTAINER
	text_mode_container = memnew(VBoxContainer);
	generate_tab->add_child(text_mode_container);
	
	prompt_input = memnew(LineEdit);
	prompt_input->set_placeholder("Describe your 3D model...");
	text_mode_container->add_child(prompt_input);
	
	multiview_checkbox = memnew(CheckBox);
	multiview_checkbox->set_text("Use Multiview (Higher Quality, +1-2min)");
	text_mode_container->add_child(multiview_checkbox);
	
	// IMAGE MODE CONTAINER (initially hidden)
	image_mode_container = memnew(VBoxContainer);
	image_mode_container->hide();
	generate_tab->add_child(image_mode_container);
	
	select_image_button = memnew(Button);
	select_image_button->set_text("Select Image File...");
	select_image_button->connect("pressed", callable_mp(this, &DesignStudio3DEditor::_on_select_image_pressed));
	image_mode_container->add_child(select_image_button);
	
	image_path_label = memnew(Label);
	image_path_label->set_text("No image selected");
	image_mode_container->add_child(image_path_label);
	
	image_preview = memnew(TextureRect);
	image_preview->set_custom_minimum_size(Size2(200 * EDSCALE, 150 * EDSCALE));
	image_preview->set_expand_mode(TextureRect::EXPAND_FIT_WIDTH_PROPORTIONAL);
	image_preview->set_stretch_mode(TextureRect::STRETCH_KEEP_ASPECT_CENTERED);
	image_mode_container->add_child(image_preview);
	
	// SHARED CONTROLS
	generate_tab->add_child(memnew(HSeparator));
	
	quality_selector = memnew(OptionButton);
	quality_selector->add_item("Turbo (~2min)", 0);
	quality_selector->add_item("Standard (~3min)", 1);
	quality_selector->add_item("High (~5min)", 2);
	quality_selector->select(0);
	generate_tab->add_child(quality_selector);

	// Target faces input
	Label *target_faces_label = memnew(Label);
	target_faces_label->set_text("Target Faces (optional):");
	generate_tab->add_child(target_faces_label);
	
	target_faces_input = memnew(LineEdit);
	target_faces_input->set_placeholder("e.g. 20000 (leave empty for default)");
	target_faces_input->set_custom_minimum_size(Size2(200 * EDSCALE, 0));
	generate_tab->add_child(target_faces_input);

	generate_button = memnew(Button);
	generate_button->set_text("Generate 3D Model");
	generate_button->connect("pressed", callable_mp(this, &DesignStudio3DEditor::_on_generate_pressed));
	generate_tab->add_child(generate_button);
	
	status_label = memnew(Label);
	status_label->set_text("Ready to generate");
	status_label->set_autowrap_mode(TextServer::AUTOWRAP_WORD_SMART);
	status_label->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	generate_tab->add_child(status_label);
	
	// === BROWSE TAB ===
	VBoxContainer *browse_tab = memnew(VBoxContainer);
	browse_tab->set_name("My Models");
	mode_tabs->add_child(browse_tab);
	
	refresh_list_button = memnew(Button);
	refresh_list_button->set_text("Refresh List");
	refresh_list_button->connect("pressed", callable_mp(this, &DesignStudio3DEditor::_on_refresh_models_pressed));
	browse_tab->add_child(refresh_list_button);
	
	// New expandable models UI
	models_scroll = memnew(ScrollContainer);
	models_scroll->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	models_scroll->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	models_scroll->set_custom_minimum_size(Size2(280 * EDSCALE, 200 * EDSCALE));
	models_scroll->set_horizontal_scroll_mode(ScrollContainer::SCROLL_MODE_DISABLED);
	models_scroll->set_vertical_scroll_mode(ScrollContainer::SCROLL_MODE_AUTO);
	browse_tab->add_child(models_scroll);
	
	models_container = memnew(VBoxContainer);
	models_container->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	models_container->set_v_size_flags(Control::SIZE_SHRINK_CENTER);
	models_scroll->add_child(models_container);
	
	// Hide old ItemList - use new expandable UI instead
	models_list = memnew(ItemList);
	models_list->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	models_list->set_custom_minimum_size(Size2(0, 150 * EDSCALE));
	models_list->hide();
	browse_tab->add_child(models_list);
	
	load_selected_button = memnew(Button);
	load_selected_button->set_text("Load Selected");
	load_selected_button->connect("pressed", callable_mp(this, &DesignStudio3DEditor::_on_load_selected_pressed));
	load_selected_button->hide(); // Hide - we use direct clicking in new UI
	browse_tab->add_child(load_selected_button);
	
	browse_status_label = memnew(Label);
	browse_status_label->set_text("Click Refresh to load your models");
	browse_status_label->set_autowrap_mode(TextServer::AUTOWRAP_WORD_SMART);
	browse_tab->add_child(browse_status_label);
	
	// === EXPORT BUTTON (at bottom, shared between tabs) ===
	left_panel->add_child(memnew(HSeparator));
	
	export_button = memnew(Button);
	export_button->set_text("Export to Workspace");
	export_button->set_disabled(true);
	export_button->connect("pressed", callable_mp(this, &DesignStudio3DEditor::_on_export_pressed));
	left_panel->add_child(export_button);
	
	Label *export_hint = memnew(Label);
	export_hint->set_text("View models first, then export to save");
	export_hint->add_theme_font_size_override("font_size", 9 * EDSCALE);
	left_panel->add_child(export_hint);
	
	// Create file dialog for image selection
	file_dialog = memnew(EditorFileDialog);
	file_dialog->set_file_mode(EditorFileDialog::FILE_MODE_OPEN_FILE);
	file_dialog->set_access(EditorFileDialog::ACCESS_FILESYSTEM);
	file_dialog->add_filter("*.png", "PNG Images");
	file_dialog->add_filter("*.jpg", "JPEG Images");
	file_dialog->add_filter("*.jpeg", "JPEG Images");
	file_dialog->add_filter("*.bmp", "BMP Images");
	file_dialog->add_filter("*.tga", "TGA Images");
	file_dialog->add_filter("*.webp", "WebP Images");
	file_dialog->connect("file_selected", callable_mp(this, &DesignStudio3DEditor::_on_image_file_selected));
	add_child(file_dialog);
	
	// Create model selection dialog
	model_selection_dialog = memnew(AcceptDialog);
	model_selection_dialog->set_title("Select Model Version");
	model_selection_dialog->set_ok_button_text("Load Selected");
	model_selection_dialog->connect("confirmed", callable_mp(this, &DesignStudio3DEditor::_on_model_selection_confirmed));
	add_child(model_selection_dialog);
	
	VBoxContainer *selection_vbox = memnew(VBoxContainer);
	model_selection_dialog->add_child(selection_vbox);
	
	Label *selection_label = memnew(Label);
	selection_label->set_text("Choose which version to load:");
	selection_vbox->add_child(selection_label);
	
	model_version_selector = memnew(OptionButton);
	model_version_selector->set_custom_minimum_size(Size2(400 * EDSCALE, 0));
	selection_vbox->add_child(model_version_selector);
	
	// Setup Current View tab (initially hidden)
	_setup_current_view_tab();
}

void DesignStudio3DEditor::_setup_3d_viewer() {
	// Right panel for 3D preview
	PanelContainer *right_panel = memnew(PanelContainer);
	right_panel->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	
	// Find the HSplitContainer we created
	HSplitContainer *hsplit = Object::cast_to<HSplitContainer>(get_child(0));
	if (hsplit) {
		hsplit->add_child(right_panel);
	}
	
	viewer_controls_container = memnew(VBoxContainer);
	right_panel->add_child(viewer_controls_container);
	
	Label *viewer_title = memnew(Label);
	viewer_title->set_text("3D Preview (Isolated)");
	viewer_title->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_CENTER);
	viewer_controls_container->add_child(viewer_title);
	
	// SubViewport for 3D preview
	viewport_container = memnew(SubViewportContainer);
	viewport_container->set_stretch(true);
	viewport_container->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	viewport_container->set_custom_minimum_size(Size2(400 * EDSCALE, 400 * EDSCALE));
	viewer_controls_container->add_child(viewport_container);
	
	// LOD Controls at bottom of viewer
	VBoxContainer *lod_controls = memnew(VBoxContainer);
	lod_controls->set_custom_minimum_size(Size2(0, 60 * EDSCALE));
	viewer_controls_container->add_child(lod_controls);
	
	// LOD Slider Label
	lod_slider_label = memnew(Label);
	lod_slider_label->set_text("LOD Level: Not Available");
	lod_slider_label->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_CENTER);
	lod_slider_label->add_theme_font_size_override("font_size", 11 * EDSCALE);
	lod_controls->add_child(lod_slider_label);
	
	// LOD Slider
	lod_slider = memnew(HSlider);
	lod_slider->set_min(0);
	lod_slider->set_max(0);
	lod_slider->set_step(1);
	lod_slider->set_value(0);
	lod_slider->set_custom_minimum_size(Size2(200 * EDSCALE, 24 * EDSCALE));
	lod_slider->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	lod_slider->connect("value_changed", callable_mp(this, &DesignStudio3DEditor::_on_lod_slider_changed));
	lod_slider->set_visible(false); // Hidden until LODs are generated
	lod_controls->add_child(lod_slider);
	
	// PROPER ISOLATED 3D VIEWER
	viewport = memnew(SubViewport);
	viewport->set_update_mode(SubViewport::UPDATE_ALWAYS);
	
	// CRITICAL: Create NEW World3D to ensure complete isolation
	Ref<World3D> new_world = memnew(World3D);
	viewport->set_world_3d(new_world);
	
	viewport_container->add_child(viewport);
	
	// Simple camera with proper environment
	camera = memnew(Camera3D);
	camera->set_position(Vector3(0, 0, 3));
	camera->set_fov(45);
	camera->make_current();
	
	// Setup environment with ambient light
	Ref<Environment> env = memnew(Environment);
	env->set_background(Environment::BG_COLOR);
	env->set_bg_color(Color(0.2, 0.2, 0.25)); // Dark blue-gray background
	env->set_ambient_light_energy(0.4); // Add ambient light to see dark areas
	camera->set_environment(env);
	
	viewport->add_child(camera);
	
	// Proper lighting for detail visibility
	light = memnew(DirectionalLight3D);
	light->set_transform(Transform3D().looking_at(Vector3(-1, -1, -1), Vector3(0, 1, 0)));
	light->set_param(Light3D::PARAM_ENERGY, 1.0); // Normal energy
	viewport->add_child(light);
	
	DirectionalLight3D *light2 = memnew(DirectionalLight3D);
	light2->set_transform(Transform3D().looking_at(Vector3(0, 1, 0), Vector3(0, 0, 1)));
	light2->set_color(Color(0.7, 0.7, 0.7));
	light2->set_param(Light3D::PARAM_ENERGY, 0.5); // Fill light
	viewport->add_child(light2);
	
	// Simple mesh instance
	preview_mesh = memnew(MeshInstance3D);
	preview_mesh->set_name("IsolatedModelViewer");
	viewport->add_child(preview_mesh);
	
	// Input handling
	set_process_input(true);
	viewport_container->set_focus_mode(Control::FOCUS_ALL);
	viewport_container->connect("gui_input", callable_mp(this, &DesignStudio3DEditor::_on_viewport_input));
	viewport_container->set_mouse_filter(Control::MOUSE_FILTER_STOP); // STOP events from passing through
	
	// HTTP Request nodes - separate instances to avoid conflicts
	submit_request = memnew(HTTPRequest);
	submit_request->set_name("SubmitRequest");
	submit_request->set_timeout(60); // 60 seconds timeout
	add_child(submit_request);
	
	poll_request = memnew(HTTPRequest);
	poll_request->set_name("PollRequest");
	poll_request->set_timeout(30); // 30 seconds timeout
	add_child(poll_request);
	
	download_request = memnew(HTTPRequest);
	download_request->set_name("DownloadRequest");
	download_request->set_timeout(180); // 3 minutes timeout for large files
	download_request->set_body_size_limit(50 * 1024 * 1024); // 50 MB limit
	download_request->set_use_threads(true); // Use threads for large downloads
	download_request->set_download_chunk_size(65536); // 64KB chunks
	add_child(download_request);
	
	browse_request = memnew(HTTPRequest);
	browse_request->set_name("BrowseRequest");
	browse_request->set_timeout(30);
	add_child(browse_request);
	
	textured_models_request = memnew(HTTPRequest);
	textured_models_request->set_name("TexturedModelsRequest");
	textured_models_request->set_timeout(30);
	add_child(textured_models_request);
	
	// Poll timer
	poll_timer = memnew(Timer);
	poll_timer->set_wait_time(5.0); // Poll every 5 seconds
	poll_timer->set_one_shot(false);
	poll_timer->connect("timeout", callable_mp(this, &DesignStudio3DEditor::_on_poll_timeout));
	add_child(poll_timer);
	
	// Download retry timer
	download_retry_timer = memnew(Timer);
	download_retry_timer->set_wait_time(1.0); // Retry after 1 second
	download_retry_timer->set_one_shot(true);
	download_retry_timer->connect("timeout", callable_mp(this, &DesignStudio3DEditor::_on_download_retry_timeout));
	add_child(download_retry_timer);
	
	// Initialize Texture System
	if (texture_system) {
		texture_system->initialize_texture_system(this, current_user_id);
		
		// Set up texture system callbacks
		texture_system->set_texture_started_callback(callable_mp(this, &DesignStudio3DEditor::_on_texture_started));
		texture_system->set_texture_progress_callback(callable_mp(this, &DesignStudio3DEditor::_on_texture_progress));
		texture_system->set_texture_completed_callback(callable_mp(this, &DesignStudio3DEditor::_on_texture_completed));
		texture_system->set_texture_failed_callback(callable_mp(this, &DesignStudio3DEditor::_on_texture_failed));
	}
	
	// Remeshing HTTP request
	remesh_request = memnew(HTTPRequest);
	remesh_request->set_name("RemeshRequest");
	remesh_request->set_timeout(180); // allow up to 3 minutes
	remesh_request->set_body_size_limit(200 * 1024 * 1024); // 200 MB limit
	remesh_request->set_use_threads(true);
	add_child(remesh_request);
}

void DesignStudio3DEditor::_setup_current_view_tab() {
	// === CURRENT VIEW TAB (initially hidden, shows after model is loaded) ===
	current_view_tab = memnew(VBoxContainer);
	current_view_tab->set_name("Current View");
	
	// Model information section
	Label *info_title = memnew(Label);
	info_title->set_text("Model Information");
	info_title->add_theme_font_size_override("font_size", 14 * EDSCALE);
	current_view_tab->add_child(info_title);
	
	model_info_label = memnew(Label);
	model_info_label->set_text("No model loaded");
	model_info_label->set_autowrap_mode(TextServer::AUTOWRAP_WORD_SMART);
	model_info_label->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	current_view_tab->add_child(model_info_label);
	
	current_view_tab->add_child(memnew(HSeparator));
	
	// LOD Section
	_setup_lod_ui();
	
	current_view_tab->add_child(memnew(HSeparator));
	
	// Action buttons section
	Label *actions_title = memnew(Label);
	actions_title->set_text("Model Actions");
	actions_title->add_theme_font_size_override("font_size", 14 * EDSCALE);
	current_view_tab->add_child(actions_title);
	
	add_texture_button = memnew(Button);
	add_texture_button->set_text("Generate AI Texture...");
	add_texture_button->connect("pressed", callable_mp(this, &DesignStudio3DEditor::_on_add_texture_pressed));
	current_view_tab->add_child(add_texture_button);
	
	segment_button = memnew(Button);
	segment_button->set_text("Segment Model");
	segment_button->connect("pressed", callable_mp(this, &DesignStudio3DEditor::_on_segment_pressed));
	current_view_tab->add_child(segment_button);
	
	remesh_button = memnew(Button);
	remesh_button->set_text("Re-mesh");
	remesh_button->connect("pressed", callable_mp(this, &DesignStudio3DEditor::_on_remesh_pressed));
	current_view_tab->add_child(remesh_button);
	
	// Cancel button for long-running operations
	cancel_operation_button = memnew(Button);
	cancel_operation_button->set_text("Cancel Operation");
	cancel_operation_button->set_visible(false); // Hidden by default
	cancel_operation_button->connect("pressed", callable_mp(this, &DesignStudio3DEditor::_on_cancel_operation_pressed));
	current_view_tab->add_child(cancel_operation_button);
	
	current_view_tab->add_child(memnew(HSeparator));
	
	// Status area for texture/segmentation operations
	Label *texture_status_title = memnew(Label);
	texture_status_title->set_text("Operation Status");
	texture_status_title->add_theme_font_size_override("font_size", 12 * EDSCALE);
	current_view_tab->add_child(texture_status_title);
	
	texture_status_label = memnew(Label);
	texture_status_label->set_text("Ready for texture/segmentation operations");
	texture_status_label->set_autowrap_mode(TextServer::AUTOWRAP_WORD_SMART);
	texture_status_label->set_custom_minimum_size(Size2(0, 60 * EDSCALE));
	current_view_tab->add_child(texture_status_label);
	
	// Remesh dialog (prompt for target faces)
	remesh_dialog = memnew(AcceptDialog);
	remesh_dialog->set_title("Remesh Model");
	remesh_dialog->set_ok_button_text("Remesh");
	remesh_dialog->connect("confirmed", callable_mp(this, &DesignStudio3DEditor::_on_remesh_dialog_confirmed));
	add_child(remesh_dialog);
	
	VBoxContainer *remesh_vbox = memnew(VBoxContainer);
	remesh_dialog->add_child(remesh_vbox);
	
	Label *faces_label = memnew(Label);
	faces_label->set_text("Target Faces:");
	remesh_vbox->add_child(faces_label);
	
	remesh_faces_input = memnew(LineEdit);
	remesh_faces_input->set_placeholder("e.g. 75000");
	remesh_vbox->add_child(remesh_faces_input);
	
	// Don't add to mode_tabs yet - will be added when model is loaded
}

void DesignStudio3DEditor::_show_current_view_tab() {
	if (current_view_tab && mode_tabs) {
		// Check if tab is already added
		bool tab_exists = false;
		for (int i = 0; i < mode_tabs->get_tab_count(); i++) {
			if (mode_tabs->get_tab_control(i) == current_view_tab) {
				tab_exists = true;
				break;
			}
		}
		
		if (!tab_exists) {
			mode_tabs->add_child(current_view_tab);
		}
		
		// Switch to the Current View tab
		mode_tabs->set_current_tab(mode_tabs->get_tab_idx_from_control(current_view_tab));
		_update_model_info();
		
		// Reset texture operation status when showing the tab
		if (texture_status_label) {
			texture_status_label->set_text("Ready for texture/segmentation operations");
		}
	}
}

void DesignStudio3DEditor::_hide_current_view_tab() {
	if (current_view_tab && mode_tabs) {
		// Check if tab exists and remove it
		for (int i = 0; i < mode_tabs->get_tab_count(); i++) {
			if (mode_tabs->get_tab_control(i) == current_view_tab) {
				mode_tabs->remove_child(current_view_tab);
				break;
			}
		}
		
		// Switch back to Generate tab
		mode_tabs->set_current_tab(0);
	}
}

void DesignStudio3DEditor::_update_model_info() {
	if (!model_info_label) {
		return;
	}
	
	String info_text = "Model Statistics:\n";
	info_text += "  Vertices: " + String::num_int64(current_vertex_count) + "\n";
	info_text += "  Faces: " + String::num_int64(current_face_count) + "\n";
	info_text += "  Normals: " + String::num_int64(current_normal_count) + "\n";
	
	if (current_texture_coord_count > 0) {
		info_text += "  Texture Coords: " + String::num_int64(current_texture_coord_count) + "\n";
	}
	
	if (current_loaded_mesh.is_valid()) {
		AABB aabb = current_loaded_mesh->get_aabb();
		Vector3 size = aabb.size;
		info_text += "\nBounding Box:\n";
		info_text += "  Size: " + String::num(size.x, 2) + " x " + String::num(size.y, 2) + " x " + String::num(size.z, 2) + "\n";
		info_text += "  Volume: " + String::num(size.x * size.y * size.z, 2) + " units^3\n";
	}
	
	if (!current_model_path.is_empty()) {
		Ref<FileAccess> file = FileAccess::open(current_model_path, FileAccess::READ);
		if (file.is_valid()) {
			int64_t file_size = file->get_length();
			file->close();
			info_text += "\nFile Size: " + String::humanize_size(file_size);
		}
	}
	
	// Add LOD information
	if (lod_levels.size() > 0) {
		info_text += "\n\nLOD System:";
		info_text += "\n  Total LOD Levels: " + String::num_int64(lod_levels.size());
		info_text += "\n  Current LOD: " + String::num_int64(current_lod_index);
		info_text += String("\n  Auto LOD: ") + (auto_lod_enabled ? "Enabled" : "Disabled");
		
		// Show all LOD levels with their face counts
		for (int i = 0; i < lod_levels.size(); i++) {
			String prefix = (i == current_lod_index) ? "→ " : "  ";
			info_text += "\n" + prefix + "LOD " + String::num_int64(i) + 
						": " + String::num_int64(lod_levels[i].face_count) + " faces" + 
						" (≥" + String::num(lod_levels[i].distance_threshold, 1) + " units)";
		}
	} else {
		info_text += "\n\nLOD System: Not generated";
		info_text += "\n  Click 'Generate LOD Levels' to enable automatic detail switching";
	}
	
	model_info_label->set_text(info_text);
}

void DesignStudio3DEditor::_on_generate_pressed() {
	if (is_generating) {
		status_label->set_text("[BUSY] Already generating...");
		return;
	}
	
	// Cancel any ongoing operations before starting new generation
	_cancel_all_requests();
	
	// Clear the 3D viewer and hide Current View tab
	if (preview_mesh) {
		preview_mesh->set_mesh(Ref<Mesh>());
		preview_mesh->set_transform(Transform3D());
	}
	
	// Hide Current View tab since we're generating a new model
	_hide_current_view_tab();
	
	// Clear model statistics
	current_vertex_count = 0;
	current_face_count = 0;
	current_normal_count = 0;
	current_texture_coord_count = 0;
	
	// Clear LOD system for new generation
	_clear_lod_levels();
	
	// Clear job ID so new generation gets fresh parent ID
	current_job_id = "";
	
	int mode = generation_mode->get_selected();
	String url;
	Dictionary body_dict;
	
	// Get quality
	String quality = "turbo";
	switch (quality_selector->get_selected_id()) {
		case 0: quality = "turbo"; break;
		case 1: quality = "standard"; break;
		case 2: quality = "high"; break;
	}
	
	body_dict["user_id"] = current_user_id;
	body_dict["quality"] = quality;
	
	// Add target_faces and use_retopology if specified
	String target_faces_text = target_faces_input->get_text().strip_edges();
	if (!target_faces_text.is_empty()) {
		int target_faces = target_faces_text.to_int();
		if (target_faces > 0) {
			body_dict["target_faces"] = target_faces;
			body_dict["use_retopology"] = true; // Required for target_faces to work
		}
	}
	
	if (mode == 0) {
		// TEXT MODE
		String prompt = prompt_input->get_text().strip_edges();
		if (prompt.is_empty()) {
			status_label->set_text("[ERROR] Please enter a prompt");
			return;
		}
		
		body_dict["prompt"] = prompt;
		
		// Check if multiview is enabled
		if (multiview_checkbox->is_pressed()) {
			url = API_URL + "/api/jobs/text-to-multiview-3d";
			status_label->set_text("[SUBMITTING] Starting multiview generation...");
		} else {
			url = API_URL + "/api/jobs/text-to-3d";
			status_label->set_text("[SUBMITTING] Starting text-to-3D...");
		}
	} else {
		// IMAGE MODE
		if (selected_image_path.is_empty()) {
			status_label->set_text("[ERROR] Please select an image");
			return;
		}
		
		String base64_image = _image_to_base64(selected_image_path);
		if (base64_image.is_empty()) {
			status_label->set_text("[ERROR] Failed to convert image");
			return;
		}
		
		body_dict["image"] = base64_image;
		url = API_URL + "/api/jobs/image-to-3d";
		status_label->set_text("[SUBMITTING] Starting image-to-3D...");
	}
	
	String json_body = JSON::stringify(body_dict);
	
	// Debug: Print the request body to see what we're sending
	print_line("=== 3D Generation Request ===");
	print_line("URL: " + url);
	print_line("Body: " + json_body);
	print_line("=============================");
	
	PackedStringArray headers;
	headers.push_back("Content-Type: application/json");
	
	submit_request->connect("request_completed", callable_mp(this, &DesignStudio3DEditor::_on_job_submitted), CONNECT_ONE_SHOT);
	
	Error err = submit_request->request(url, headers, HTTPClient::METHOD_POST, json_body);
	
	if (err == OK) {
		is_generating = true;
		generate_button->set_disabled(true);
	} else {
		status_label->set_text("[ERROR] Failed to start request");
	}
}

void DesignStudio3DEditor::_on_job_submitted(int p_result, int p_code, const PackedStringArray &p_headers, const PackedByteArray &p_body) {
	if (p_result != HTTPRequest::RESULT_SUCCESS || p_code != 200) {
		status_label->set_text("[ERROR] Failed to submit job (HTTP " + itos(p_code) + ")");
		is_generating = false;
		generate_button->set_disabled(false);
		return;
	}
	
	String response_text;
	if (p_body.size() > 0) {
		const uint8_t *r = p_body.ptr();
		response_text = String::utf8((const char *)r, p_body.size());
	}
	
	Variant json_variant;
	JSON json;
	Error err = json.parse(response_text);
	
	if (err != OK) {
		status_label->set_text("[ERROR] Failed to parse response");
		is_generating = false;
		generate_button->set_disabled(false);
		return;
	}
	
	Dictionary response = json.get_data();
	
	if (response.has("record_id")) {
		current_job_id = response["record_id"];
		status_label->set_text("[SUCCESS] Job submitted! ID: " + current_job_id.substr(0, 8) + "...\n[POLLING] Checking status...");
		_start_polling(current_job_id);
	} else {
		status_label->set_text("[ERROR] No job ID in response");
		is_generating = false;
		generate_button->set_disabled(false);
	}
}

void DesignStudio3DEditor::_start_polling(const String &p_job_id) {
	current_job_id = p_job_id;
	poll_timer->start();
	// Immediately poll once
	_on_poll_timeout();
}

void DesignStudio3DEditor::_stop_polling() {
	poll_timer->stop();
}

void DesignStudio3DEditor::_on_poll_timeout() {
	if (current_job_id.is_empty()) {
		_stop_polling();
		return;
	}
	
	String url = API_URL + "/api/jobs/" + current_job_id;
	
	// Use dedicated poll_request instance
	poll_request->connect("request_completed", callable_mp(this, &DesignStudio3DEditor::_on_job_status_received), CONNECT_ONE_SHOT);
	poll_request->request(url);
}

void DesignStudio3DEditor::_on_job_status_received(int p_result, int p_code, const PackedStringArray &p_headers, const PackedByteArray &p_body) {
	if (p_result != HTTPRequest::RESULT_SUCCESS || p_code != 200) {
		status_label->set_text("[ERROR] Failed to get job status (HTTP " + itos(p_code) + ")");
		_stop_polling();
		is_generating = false;
		generate_button->set_disabled(false);
		return;
	}
	
	String response_text;
	if (p_body.size() > 0) {
		const uint8_t *r = p_body.ptr();
		response_text = String::utf8((const char *)r, p_body.size());
	}
	
	Variant json_variant;
	JSON json;
	Error err = json.parse(response_text);
	
	if (err != OK) {
		status_label->set_text("[ERROR] Failed to parse status response");
		_stop_polling();
		is_generating = false;
		generate_button->set_disabled(false);
		return;
	}
	
	Dictionary job_data = json.get_data();
	
	if (!job_data.has("status")) {
		status_label->set_text("[ERROR] Invalid status response");
		_stop_polling();
		is_generating = false;
		generate_button->set_disabled(false);
		return;
	}
	
	String status = job_data["status"];
	
	if (status == "queued") {
		status_label->set_text("[QUEUED] Job queued... waiting for GPU");
	} else if (status == "processing") {
		status_label->set_text("[PROCESSING] Processing on GPU... (this may take a few minutes)");
	} else if (status == "completed") {
		_stop_polling();
		
		// Use direct Supabase URL for faster downloads
		String model_url = "";
		if (job_data.has("output_file_url")) {
			Variant url_variant = job_data["output_file_url"];
			if (url_variant.get_type() == Variant::STRING) {
				model_url = url_variant;
			}
		}
		
		// Fallback to proxy endpoint if no direct URL
		if (model_url.is_empty()) {
			String record_id = "";
			if (job_data.has("id")) {
				record_id = job_data["id"];
			}
			
			if (record_id.is_empty()) {
				status_label->set_text("[ERROR] No job ID or URL found in completed response");
				is_generating = false;
				generate_button->set_disabled(false);
				return;
			}
			
			model_url = API_URL + "/api/download/" + record_id + "?user_id=" + current_user_id;
		}
		status_label->set_text("[COMPLETE] Downloading model (fast)...");
		
		// Reset retry count and start download with proper headers
		download_retry_count = 0;
		download_url_to_retry = model_url;
		status_label->set_text("[DOWNLOADING] Downloading model... please wait\nThis may take 10-30 seconds for large files");
		_start_download_with_headers(model_url);
	} else if (status == "failed") {
		_stop_polling();
		String error_msg = job_data.get("error_message", "Unknown error");
		status_label->set_text("[FAILED] Generation failed: " + error_msg);
		is_generating = false;
		generate_button->set_disabled(false);
	} else {
		status_label->set_text("Status: " + status);
	}
}

void DesignStudio3DEditor::_on_model_downloaded(int p_result, int p_code, const PackedStringArray &p_headers, const PackedByteArray &p_body) {
	print_line("=== _on_model_downloaded CALLBACK TRIGGERED ===");
	print_line("p_result: " + itos(p_result) + " (RESULT_SUCCESS=" + itos(HTTPRequest::RESULT_SUCCESS) + ")");
	print_line("p_code: " + itos(p_code));
	print_line("p_body.size(): " + itos(p_body.size()));
	print_line("Headers count: " + itos(p_headers.size()));
	
	if (p_result != HTTPRequest::RESULT_SUCCESS) {
		String error_msg = "Unknown error";
		switch (p_result) {
			case HTTPRequest::RESULT_CHUNKED_BODY_SIZE_MISMATCH: error_msg = "Chunked body size mismatch"; break;
			case HTTPRequest::RESULT_CANT_CONNECT: error_msg = "Can't connect"; break;
			case HTTPRequest::RESULT_CANT_RESOLVE: error_msg = "Can't resolve hostname"; break;
			case HTTPRequest::RESULT_CONNECTION_ERROR: error_msg = "Connection error"; break;
			case HTTPRequest::RESULT_TLS_HANDSHAKE_ERROR: error_msg = "TLS handshake error"; break;
			case HTTPRequest::RESULT_NO_RESPONSE: error_msg = "No response"; break;
			case HTTPRequest::RESULT_BODY_SIZE_LIMIT_EXCEEDED: error_msg = "Body size limit exceeded"; break;
			case HTTPRequest::RESULT_REQUEST_FAILED: error_msg = "Request failed"; break;
			case HTTPRequest::RESULT_DOWNLOAD_FILE_CANT_OPEN: error_msg = "Can't open download file"; break;
			case HTTPRequest::RESULT_DOWNLOAD_FILE_WRITE_ERROR: error_msg = "Download file write error"; break;
			case HTTPRequest::RESULT_REDIRECT_LIMIT_REACHED: error_msg = "Redirect limit reached"; break;
			case HTTPRequest::RESULT_TIMEOUT: error_msg = "Timeout"; break;
		}
		status_label->set_text("[ERROR] Download failed: " + error_msg + "\nResult code: " + itos(p_result));
		is_generating = false;
		generate_button->set_disabled(false);
		return;
	}
	
	if (p_code != 200 && p_code != 302 && p_code != 307) {
		// Handle retry logic for 404 errors (might be timing issue)
		if (p_code == 404 && download_retry_count < 2) {
			status_label->set_text("[404 ERROR] Retrying in 1 second... (Attempt " + itos(download_retry_count + 1) + "/3)");
			download_retry_timer->start();
			return;
		}
		
		// All retries exhausted or different error
		String error_details = "[ERROR] Failed to download model (HTTP " + itos(p_code) + ")\n\n";
		
		// Show the error response if available
		if (p_body.size() > 0) {
			String response_text;
			const uint8_t *r = p_body.ptr();
			response_text = String::utf8((const char *)r, p_body.size());
			error_details += "Server response: " + response_text + "\n\n";
		}
		
		if (p_code == 404) {
			error_details += "RETRIED " + itos(download_retry_count) + " TIMES - BACKEND ISSUE:\n";
			error_details += "1. Job completed successfully\n";
			error_details += "2. But file upload to storage failed\n";
			error_details += "3. Check server logs for upload errors\n";
			error_details += "4. Verify Supabase storage permissions\n\n";
			error_details += "The 3D generation worked, but file storage failed.";
		}
		
		status_label->set_text(error_details);
		is_generating = false;
		generate_button->set_disabled(false);
		download_retry_count = 0; // Reset for next time
		return;
	}
	
	// Handle redirects
	if (p_code == 302 || p_code == 307) {
		// Look for Location header
		for (int i = 0; i < p_headers.size(); i++) {
			String header = p_headers[i];
			if (header.begins_with("Location: ") || header.begins_with("location: ")) {
				String redirect_url = header.substr(header.find(":") + 2);
				download_request->connect("request_completed", callable_mp(this, &DesignStudio3DEditor::_on_model_downloaded), CONNECT_ONE_SHOT);
				download_request->request(redirect_url);
				return;
			}
		}
	}
	
	if (p_body.size() > 0) {
		_load_model_from_data(p_body);
	} else {
		status_label->set_text("[ERROR] Downloaded file is empty");
		is_generating = false;
		generate_button->set_disabled(false);
	}
	
	// Reset state
	if (is_generating) {
		// Only clear job ID for newly generated models, not for loaded existing models
		current_job_id = "";
	}
	is_generating = false;
	generate_button->set_disabled(false);
	download_retry_count = 0;
}

void DesignStudio3DEditor::_load_model_from_data(const PackedByteArray &p_data) {
	
	// Save to TEMP directory (NOT workspace yet - user must export)
	String timestamp = String::num_int64(OS::get_singleton()->get_ticks_msec());
	String filename = "temp_model_" + timestamp + ".obj";
	String temp_path = "user://" + filename;
	current_model_path = temp_path; // Store for later export
	
	Ref<FileAccess> file = FileAccess::open(temp_path, FileAccess::WRITE);
	if (file.is_valid()) {
		file->store_buffer(p_data);
		file->close();
		
		// Show preview info
		String preview = "[SUCCESS] Model loaded for viewing!\n\n";
		preview += "Size: " + String::humanize_size(p_data.size()) + "\n";
		
		// Parse the entire OBJ file to get accurate statistics
		String content;
		if (p_data.size() > 0) {
			const uint8_t *r = p_data.ptr();
			content = String::utf8((const char *)r, p_data.size());
		}
		
		int vertex_count = 0;
		int normal_count = 0;
		int texture_count = 0;
		int face_count = 0;
		
		PackedStringArray lines = content.split("\n");
		for (int i = 0; i < lines.size(); i++) {
			String line = lines[i].strip_edges();
			if (line.begins_with("v ")) vertex_count++;
			else if (line.begins_with("vn ")) normal_count++;
			else if (line.begins_with("vt ")) texture_count++;
			else if (line.begins_with("f ")) face_count++;
		}
		
		// Store model statistics for Current View tab
		current_vertex_count = vertex_count;
		current_face_count = face_count;
		current_normal_count = normal_count;
		current_texture_coord_count = texture_count;
		
		// Clear any existing LODs since this is a new model
		_clear_lod_levels();
		
		if (vertex_count > 0 || face_count > 0) {
			preview += "Model Statistics:\n";
			preview += "  Vertices: " + String::num_int64(vertex_count) + "\n";
			preview += "  Faces: " + String::num_int64(face_count) + "\n";
			preview += "  Normals: " + String::num_int64(normal_count) + "\n";
			if (texture_count > 0) {
				preview += "  Texture Coords: " + String::num_int64(texture_count) + "\n";
			}
		}
		
		// Parse OBJ directly and load (works without import)
		Ref<ArrayMesh> mesh = _parse_obj_to_mesh(content);
		if (mesh.is_valid()) {
			current_loaded_mesh = mesh;
			
			if (preview_mesh) {
				preview_mesh->set_mesh(mesh);
				
				// Add double-sided material to prevent see-through issues
				Ref<StandardMaterial3D> mat = memnew(StandardMaterial3D);
				mat->set_cull_mode(BaseMaterial3D::CULL_DISABLED); // DOUBLE-SIDED rendering
				mat->set_albedo(Color(0.8, 0.8, 0.8)); // Neutral gray
				mat->set_shading_mode(StandardMaterial3D::SHADING_MODE_PER_VERTEX); // Better shading
				preview_mesh->set_material_override(mat);
				
				_setup_camera_orbit();
				
				preview += "\n[3D VIEWER] Model loaded! Use mouse to rotate/zoom.";
			}
			
			// Enable export button
			export_button->set_disabled(false);
			
			// Show Current View tab
			_show_current_view_tab();
		}
		
		status_label->set_text(preview);
		
	} else {
		status_label->set_text("[ERROR] Failed to save temp file");
	}
}

Ref<ArrayMesh> DesignStudio3DEditor::_parse_obj_to_mesh(const String &p_obj_content) {
	PackedVector3Array vertices;
	PackedVector3Array normals;
	PackedInt32Array indices;
	
	PackedStringArray lines = p_obj_content.split("\n");
	
	// Parse vertices AND normals
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
			// Parse normals too!
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
				for (int j = 1; j <= 3 && j < parts.size(); j++) {
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
	
	// Create mesh WITH normals for proper shading
	Ref<ArrayMesh> mesh = memnew(ArrayMesh);
	Array arrays;
	arrays.resize(Mesh::ARRAY_MAX);
	arrays[Mesh::ARRAY_VERTEX] = vertices;
	arrays[Mesh::ARRAY_INDEX] = indices;
	
	// Add normals for proper shading detail
	if (normals.size() == vertices.size()) {
		arrays[Mesh::ARRAY_NORMAL] = normals;
	}
	
	mesh->add_surface_from_arrays(Mesh::PRIMITIVE_TRIANGLES, arrays);
	return mesh;
}

void DesignStudio3DEditor::_setup_camera_orbit() {
	if (!preview_mesh || !camera) {
		return;
	}
	
	Ref<Mesh> mesh = preview_mesh->get_mesh();
	if (!mesh.is_valid()) {
		return;
	}
	
	// Simple approach: just scale the model to fit in view
	AABB aabb = mesh->get_aabb();
	float size = aabb.get_longest_axis_size();
	
	if (size > 0) {
		// Scale model to fit in a 1x1x1 box
		float scale = 1.0f / size;
		preview_mesh->set_scale(Vector3(scale, scale, scale));
		
		// Center the model
		Vector3 center = aabb.get_center();
		preview_mesh->set_position(-center * scale);
	}
	
	// Position camera to see the 1x1x1 scaled model
	camera->set_position(Vector3(1.5, 1.0, 2.0));
	camera->look_at(Vector3(0, 0, 0), Vector3(0, 1, 0));
}

void DesignStudio3DEditor::_update_camera_from_orbit() {
	// SIMPLE: Just rotate the mesh directly, no complex nodes
	if (!preview_mesh) {
		return;
	}
	
	Vector3 current_rotation = preview_mesh->get_rotation_degrees();
	current_rotation.x = orbit_pitch;
	current_rotation.y = orbit_yaw;
	preview_mesh->set_rotation_degrees(current_rotation);
}

void DesignStudio3DEditor::_on_viewport_input(const Ref<InputEvent> &p_event) {
	Ref<InputEventMouseButton> mb = p_event;
	if (mb.is_valid()) {
		if (mb->get_button_index() == MouseButton::LEFT) {
			is_rotating = mb->is_pressed();
			last_mouse_pos = mb->get_position();
			viewport_container->accept_event(); // Consume the event
		} else if (mb->get_button_index() == MouseButton::WHEEL_UP && mb->is_pressed()) {
			// Zoom in toward center (origin)
			if (camera) {
				Vector3 pos = camera->get_position();
				Vector3 center = Vector3(0, 0, 0); // Model is centered at origin
				Vector3 direction = (center - pos).normalized();
				
				// Move camera toward center
				Vector3 new_pos = pos + direction * 0.2f;
				
				// Don't get too close
				if (new_pos.length() > 0.3f) {
					camera->set_position(new_pos);
					camera->look_at(center, Vector3(0, 1, 0));
					// Update LOD based on new camera distance
					_update_lod_based_on_distance();
				}
				viewport_container->accept_event();
			}
		} else if (mb->get_button_index() == MouseButton::WHEEL_DOWN && mb->is_pressed()) {
			// Zoom out away from center
			if (camera) {
				Vector3 pos = camera->get_position();
				Vector3 center = Vector3(0, 0, 0);
				Vector3 direction = (pos - center).normalized();
				
				// Move camera away from center
				Vector3 new_pos = pos + direction * 0.2f;
				
				// Don't get too far
				if (new_pos.length() < 15.0f) {
					camera->set_position(new_pos);
					camera->look_at(center, Vector3(0, 1, 0));
					// Update LOD based on new camera distance
					_update_lod_based_on_distance();
				}
				viewport_container->accept_event();
			}
		}
		return;
	}
	
	Ref<InputEventMouseMotion> mm = p_event;
	if (mm.is_valid() && is_rotating) {
		Vector2 delta = mm->get_position() - last_mouse_pos;
		
		// Rotate the model
		orbit_yaw += delta.x * 0.5f;
		orbit_pitch += delta.y * 0.5f;
		orbit_pitch = CLAMP(orbit_pitch, -90.0f, 90.0f);
		
		_update_camera_from_orbit();
		last_mouse_pos = mm->get_position();
		viewport_container->accept_event(); // Consume the event
		
		// Update LOD based on camera distance if auto LOD is enabled
		_update_lod_based_on_distance();
	}
}

void DesignStudio3DEditor::_start_download_with_headers(const String &p_url) {
	// Check if request is busy
	if (download_request->get_http_client_status() != HTTPClient::STATUS_DISCONNECTED) {
		status_label->set_text("[ERROR] Download request is busy. Please wait...");
		return;
	}
	
	// Disconnect any existing connections first
	if (download_request->is_connected("request_completed", callable_mp(this, &DesignStudio3DEditor::_on_model_downloaded))) {
		download_request->disconnect("request_completed", callable_mp(this, &DesignStudio3DEditor::_on_model_downloaded));
	}
	
	// Connect callback
	download_request->connect("request_completed", callable_mp(this, &DesignStudio3DEditor::_on_model_downloaded), CONNECT_ONE_SHOT);
	
	// Add headers to match curl behavior
	PackedStringArray headers;
	headers.push_back("User-Agent: Godot-Editor/4.0");
	headers.push_back("Accept: */*");
	headers.push_back("Accept-Encoding: identity");
	
	Error download_err = download_request->request(p_url, headers, HTTPClient::METHOD_GET, "");
	
	if (download_err != OK) {
		status_label->set_text("[ERROR] Failed to start download request. Error code: " + itos(download_err));
		is_generating = false;
		generate_button->set_disabled(false);
	}
}

void DesignStudio3DEditor::_on_download_retry_timeout() {
	download_retry_count++;
	status_label->set_text("[RETRYING] Download attempt " + itos(download_retry_count + 1) + "/3");
	
	_start_download_with_headers(download_url_to_retry);
}

void DesignStudio3DEditor::_on_refresh_models_pressed() {
	// Cancel any ongoing operations before browsing models
	_cancel_all_requests();
	
	browse_status_label->set_text("Loading models...");
	models_list->clear();
	
	String url = API_URL + "/api/users/" + current_user_id + "/models?status=completed";
	
	browse_request->connect("request_completed", callable_mp(this, &DesignStudio3DEditor::_on_models_list_received), CONNECT_ONE_SHOT);
	browse_request->request(url);
}

void DesignStudio3DEditor::_on_models_list_received(int p_result, int p_code, const PackedStringArray &p_headers, const PackedByteArray &p_body) {
	if (p_result != HTTPRequest::RESULT_SUCCESS || p_code != 200) {
		browse_status_label->set_text("[ERROR] Failed to load models (HTTP " + itos(p_code) + ")");
		return;
	}
	
	String response_text;
	if (p_body.size() > 0) {
		const uint8_t *r = p_body.ptr();
		response_text = String::utf8((const char *)r, p_body.size());
	}
	
	JSON json;
	Error err = json.parse(response_text);
	if (err != OK) {
		browse_status_label->set_text("[ERROR] Failed to parse models list");
		return;
	}
	
	Dictionary response = json.get_data();
	if (!response.has("models")) {
		browse_status_label->set_text("[ERROR] Invalid response");
		return;
	}
	
	Array models = response["models"];
	int count = response.get("count", 0);
	
	// Clear both old and new UI
	models_list->clear();
	for (int i = models_container->get_child_count() - 1; i >= 0; i--) {
		models_container->get_child(i)->queue_free();
	}
	
	textured_models_cache.clear();
	pending_textured_requests.clear();
	textured_request_queue.clear();
	is_processing_textured_requests = false;
	model_rows.clear();
	expanded_models.clear();
	
	// Limit the number of models we process to prevent UI overflow
	int max_models_to_show = 8; // Very strict limit for expandable UI
	int models_to_process = MIN(models.size(), max_models_to_show);
	
	for (int i = 0; i < models_to_process; i++) {
		Dictionary model = models[i];
		String prompt = model.get("prompt", "Unknown");
		String id = model.get("id", "");
		String created = model.get("created_at", "");
		
		// Create new expandable row
		_create_model_row(model, i);
		
		// Also keep old ItemList for compatibility
		String display_text = prompt + " (" + created.substr(0, 10) + ")";
		models_list->add_item(display_text);
		models_list->set_item_metadata(i, model);
		
		// Queue textured models request for this base model
		_fetch_textured_models_for_base_model(id, i);
	}
	
	// Show info if we truncated the list
	if (models.size() > max_models_to_show) {
		Label *info_label = memnew(Label);
		info_label->set_text("... and " + itos(models.size() - max_models_to_show) + " more models (showing first " + itos(max_models_to_show) + ")");
		info_label->add_theme_font_size_override("font_size", 10 * EDSCALE);
		info_label->set_modulate(Color(0.7, 0.7, 0.7));
		models_container->add_child(info_label);
	}
	
	// Start processing the queue
	_process_textured_request_queue();
	
	browse_status_label->set_text("Loaded " + itos(count) + " completed models (loading textured versions...)");
}

void DesignStudio3DEditor::_on_load_selected_pressed() {
	PackedInt32Array selected = models_list->get_selected_items();
	if (selected.is_empty()) {
		browse_status_label->set_text("[ERROR] No model selected");
		return;
	}
	
	int selected_idx = selected[0];
	
	Dictionary model_data = models_list->get_item_metadata(selected_idx);
	String model_id = model_data.get("id", "");
	String prompt = model_data.get("prompt", "Unknown");
	
	if (model_id.is_empty()) {
		browse_status_label->set_text("[ERROR] Invalid model data");
		return;
	}
	
	// Check if this model has textured versions
	if (textured_models_cache.has(model_id)) {
		Array textured_models = textured_models_cache[model_id];
		if (textured_models.size() > 0) {
			_show_model_selection_dialog(model_data, textured_models);
			return;
		}
	}
	
	// No textured models, load the base model directly
	browse_status_label->set_text("Loading: " + prompt + "...");
	_load_model_for_viewing(model_data);
}

void DesignStudio3DEditor::_load_model_for_viewing(const Dictionary &p_model_data) {
	current_model_data = p_model_data;
	
	// IMPORTANT: Set current_job_id from the loaded model data
	String job_id = p_model_data.get("id", "");
	if (job_id.is_empty()) {
		job_id = p_model_data.get("job_id", "");
	}
	
	if (!job_id.is_empty()) {
		current_job_id = job_id;
	}
	
	String model_url = p_model_data.get("output_file_url", "");
	
	// Check if this is a textured model (uses different URL field)
	bool is_textured = p_model_data.get("is_textured", false) || p_model_data.has("textured_mesh_url");
	
	// Debug output to track URL issues  
	String model_id = p_model_data.get("id", "unknown");
	print_line("Loading model ID: " + model_id + ", original URL: '" + model_url + "', is_textured: " + (is_textured ? "true" : "false"));
	
	if (model_url.is_empty() && p_model_data.has("textured_mesh_url")) {
		Variant url_variant = p_model_data.get("textured_mesh_url", "");
		print_line("textured_mesh_url variant type: " + itos(url_variant.get_type()) + ", is_null: " + (url_variant.is_null() ? "true" : "false"));
		
		// Handle both NIL (type 0) and STRING types
		if (url_variant.get_type() == Variant::STRING && !url_variant.is_null()) {
			String temp_url = url_variant;
			print_line("textured_mesh_url string value: '" + temp_url + "'");
			// Check if it's not null/empty string
			if (!temp_url.is_empty() && temp_url != "null" && temp_url.begins_with("http")) {
				model_url = temp_url;
				print_line("Using direct textured_mesh_url: " + model_url);
			}
		} else if (url_variant.get_type() == Variant::NIL || url_variant.is_null()) {
			print_line("textured_mesh_url is null/nil - will use API download endpoint");
		}
	}
	
	// For textured models with null/empty URLs, ALWAYS use the texture API download endpoint
	if (is_textured && (model_url.is_empty() || model_url == "null" || model_url == "<null>")) {
		model_url = "https://gpu-proxy-976792908107.us-central1.run.app/api/texture-jobs/" + model_id + "/download?user_id=" + current_user_id;
		print_line("Using texture API download URL for textured model: " + model_url);
	}
	
	print_line("Final model_url after all processing: '" + model_url + "'");
	
	// Validate URL is not empty and has a valid scheme
	if (model_url.is_empty() || model_url.strip_edges().is_empty()) {
		browse_status_label->set_text("[ERROR] No URL for model: " + model_id);
		print_line("ERROR: Empty URL for model " + model_id);
		return;
	}
	
	// Check for valid URL scheme
	if (!model_url.begins_with("http://") && !model_url.begins_with("https://")) {
		browse_status_label->set_text("[ERROR] Invalid URL format: " + model_url);
		print_line("ERROR: Invalid URL scheme for model " + model_id + ": " + model_url);
		return;
	}
	
	print_line("URL validation passed, proceeding with download: " + model_url);
	
	// Show different loading message for textured models
	if (is_textured) {
		browse_status_label->set_text("Downloading textured model...");
	} else {
		browse_status_label->set_text("Downloading...");
	}
	
	// Check if request is busy
	if (download_request->get_http_client_status() != HTTPClient::STATUS_DISCONNECTED) {
		browse_status_label->set_text("[ERROR] Download request is busy. Please wait or try again...");
		return;
	}
	
	// Disconnect any existing connections first
	if (download_request->is_connected("request_completed", callable_mp(this, &DesignStudio3DEditor::_on_model_downloaded))) {
		download_request->disconnect("request_completed", callable_mp(this, &DesignStudio3DEditor::_on_model_downloaded));
	}
	
	// Download but DON'T save to workspace yet
	download_request->connect("request_completed", callable_mp(this, &DesignStudio3DEditor::_on_model_downloaded), CONNECT_ONE_SHOT);
	
	print_line("About to call download_request->request() with URL: '" + model_url + "'");
	print_line("URL length: " + itos(model_url.length()) + ", is_empty: " + (model_url.is_empty() ? "true" : "false"));
	
	// Try adding headers as we do in other successful requests
	PackedStringArray headers;
	headers.push_back("User-Agent: Godot-Editor/4.0");
	headers.push_back("Accept: */*");
	
	Error err = download_request->request(model_url, headers);
	
	print_line("Request result: " + itos(err) + " (OK=0)");
	
	if (err != OK) {
		browse_status_label->set_text("[ERROR] Failed to start download. Error: " + itos(err));
		print_line("ERROR: HTTPRequest.request() failed with error " + itos(err));
	} else {
		print_line("HTTPRequest.request() call successful");
	}
}

void DesignStudio3DEditor::_on_export_pressed() {
	if (current_loaded_mesh.is_null()) {
		status_label->set_text("[ERROR] No model loaded to export");
		return;
	}
	
	if (current_model_path.is_empty()) {
		status_label->set_text("[ERROR] No model data to export");
		return;
	}
	
	String timestamp = String::num_int64(OS::get_singleton()->get_ticks_msec());
	int exported_files = 0;
	String export_summary = "";
	
	// Export LOD levels if they exist
	if (lod_levels.size() > 1) {
		// Export all LOD levels
		for (int i = 0; i < lod_levels.size(); i++) {
			const LODLevel &lod = lod_levels[i];
			
			if (lod.model_path.is_empty()) {
				continue; // Skip if no file path
			}
			
			String ext = lod.model_path.get_extension();
			if (ext.is_empty()) {
				ext = "obj";
			}
			
			String filename = "exported_model_lod" + itos(i) + "_" + timestamp + "." + ext;
			String save_path = "res://" + filename;
			String project_path = ProjectSettings::get_singleton()->globalize_path(save_path);
			
			// Read the LOD file and save to workspace
			Ref<FileAccess> source = FileAccess::open(lod.model_path, FileAccess::READ);
			if (source.is_valid()) {
				PackedByteArray data = source->get_buffer(source->get_length());
				source->close();
				
				Ref<FileAccess> dest = FileAccess::open(project_path, FileAccess::WRITE);
				if (dest.is_valid()) {
					dest->store_buffer(data);
					dest->flush();
					dest->close();
					
					exported_files++;
					export_summary += "LOD " + itos(i) + ": " + filename + " (" + itos(lod.face_count) + " faces)\n";
				}
			}
		}
		
		if (exported_files > 0) {
			// IMMEDIATE scan to detect new files
			EditorFileSystem::get_singleton()->scan_changes();
			EditorFileSystem::get_singleton()->call_deferred("scan_changes");
			
			status_label->set_text("[SUCCESS] Exported " + itos(exported_files) + " LOD levels:\n" + export_summary + "Importing...");
			if (browse_status_label) {
				browse_status_label->set_text("[SUCCESS] Exported " + itos(exported_files) + " LOD files!");
			}
			
			if (lod_status_label) {
				lod_status_label->set_text(String("[EXPORT SUCCESS] All LOD levels exported to workspace!\n") + 
										  "Files: " + itos(exported_files) + " LOD levels");
			}
		} else {
			status_label->set_text("[ERROR] Failed to export LOD levels");
		}
	} else {
		// Export single model (original behavior)
		String ext = current_model_path.get_extension();
		if (ext.is_empty()) {
			ext = "obj";
		}
		String filename = "exported_model_" + timestamp + "." + ext;
		String save_path = "res://" + filename;
		String project_path = ProjectSettings::get_singleton()->globalize_path(save_path);
		
		// Read the temp file and save to workspace
		Ref<FileAccess> source = FileAccess::open(current_model_path, FileAccess::READ);
		if (source.is_valid()) {
			PackedByteArray data = source->get_buffer(source->get_length());
			source->close();
			
			Ref<FileAccess> dest = FileAccess::open(project_path, FileAccess::WRITE);
			if (dest.is_valid()) {
				dest->store_buffer(data);
				dest->flush(); // FORCE flush to disk immediately
				dest->close();
				
				// IMMEDIATE scan to detect new file
				EditorFileSystem::get_singleton()->scan_changes();
				
				// Also update the specific file
				EditorFileSystem::get_singleton()->update_file(save_path);
				
				// Call scan again after short delay to ensure it's picked up
				EditorFileSystem::get_singleton()->call_deferred("scan_changes");
				
				status_label->set_text("[SUCCESS] Exported to: " + filename + "\nImporting...");
				if (browse_status_label) {
					browse_status_label->set_text("[SUCCESS] Exported and importing!");
				}
				
				if (lod_status_label) {
					lod_status_label->set_text("[EXPORT SUCCESS] Model exported (no LODs generated)");
				}
			} else {
				status_label->set_text("[ERROR] Failed to write to workspace");
			}
		} else {
			status_label->set_text("[ERROR] Failed to read temp model");
		}
	}
}

void DesignStudio3DEditor::_load_imported_mesh(const String &p_path) {
	if (!preview_mesh) {
		return;
	}
	
	// Try loading the resource - Godot should have imported it by now
	Ref<Resource> resource = ResourceLoader::load(p_path);
	
	if (resource.is_valid()) {
		// Check if it's a mesh directly
		Ref<Mesh> mesh = resource;
		if (mesh.is_valid()) {
			current_loaded_mesh = mesh; // Store for export
			preview_mesh->set_mesh(mesh);
			
			_setup_camera_orbit();
			
			// Update all status labels
			String success_msg = "[3D VIEWER] Model loaded! Use mouse to rotate/zoom.";
			if (status_label) status_label->set_text(status_label->get_text() + "\n" + success_msg);
			if (browse_status_label) browse_status_label->set_text("Model loaded! Export to save.");
			
			export_button->set_disabled(false);
			
			// Show Current View tab
			_show_current_view_tab();
			
			return;
		}
		
		// Check if it's a scene with MeshInstance3D
		Ref<PackedScene> scene = resource;
		if (scene.is_valid()) {
			print_line("Got PackedScene resource");
			Node *root = scene->instantiate();
			if (root) {
				MeshInstance3D *mesh_instance = Object::cast_to<MeshInstance3D>(root);
				if (!mesh_instance) {
					// Look for MeshInstance3D in children
					for (int i = 0; i < root->get_child_count(); i++) {
						mesh_instance = Object::cast_to<MeshInstance3D>(root->get_child(i));
						if (mesh_instance) break;
					}
				}
				
				if (mesh_instance && mesh_instance->get_mesh().is_valid()) {
					print_line("Found MeshInstance3D in scene with valid mesh");
					Ref<Mesh> scene_mesh = mesh_instance->get_mesh();
					
					current_loaded_mesh = scene_mesh; // Store for export
					preview_mesh->set_mesh(scene_mesh);
					_setup_camera_orbit();
					
					// Update status labels
					String success_msg = "[3D VIEWER] Model loaded! Use mouse to rotate/zoom.";
					if (status_label) status_label->set_text(status_label->get_text() + "\n" + success_msg);
					if (browse_status_label) browse_status_label->set_text("Model loaded! Export to save.");
					
					export_button->set_disabled(false);
					
					// Show Current View tab
					_show_current_view_tab();
					
					root->queue_free();
					return;
				}
				
				root->queue_free();
			}
		}
		
		print_line("Resource loaded but couldn't extract mesh from it");
		status_label->set_text(status_label->get_text() + "\n[3D VIEWER] Loaded resource but no mesh found");
	} else {
		print_line("Failed to load resource - import might not be complete yet");
		status_label->set_text(status_label->get_text() + "\n[3D VIEWER] Import still in progress...");
		
		// Try again in 3 more seconds
		Timer *retry_timer = memnew(Timer);
		retry_timer->set_wait_time(3.0);
		retry_timer->set_one_shot(true);
		add_child(retry_timer);
		retry_timer->connect("timeout", callable_mp(this, &DesignStudio3DEditor::_load_imported_mesh).bind(p_path));
		retry_timer->start();
		
		// Clean up timer after it fires
		retry_timer->connect("timeout", Callable(retry_timer, "queue_free"), CONNECT_DEFERRED);
	}
}

void DesignStudio3DEditor::_on_generation_mode_changed(int p_index) {
	if (p_index == 0) {
		// Text mode
		text_mode_container->show();
		image_mode_container->hide();
		status_label->set_text("Ready to generate from text");
	} else {
		// Image mode
		text_mode_container->hide();
		image_mode_container->show();
		status_label->set_text("Select an image to generate");
	}
}

void DesignStudio3DEditor::_on_select_image_pressed() {
	if (file_dialog) {
		file_dialog->popup_centered(Size2(800 * EDSCALE, 600 * EDSCALE));
	}
}

void DesignStudio3DEditor::_on_image_file_selected(const String &p_path) {
	selected_image_path = p_path;
	image_path_label->set_text("Selected: " + p_path.get_file());
	
	// Load and display image preview
	Ref<Image> img = memnew(Image);
	Error err = img->load(p_path);
	
	if (err == OK) {
		Ref<ImageTexture> texture = ImageTexture::create_from_image(img);
		image_preview->set_texture(texture);
		status_label->set_text("Image loaded! Click Generate.");
	} else {
		status_label->set_text("[ERROR] Failed to load image");
	}
}

String DesignStudio3DEditor::_image_to_base64(const String &p_image_path) {
	Ref<Image> img = memnew(Image);
	Error err = img->load(p_image_path);
	
	if (err != OK) {
		return "";
	}
	
	// Convert to PNG format in memory
	PackedByteArray png_data = img->save_png_to_buffer();
	
	// Simple base64 encoding
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

void DesignStudio3DEditor::_setup_lod_ui() {
	// LOD Section
	Label *lod_title = memnew(Label);
	lod_title->set_text("Level of Detail (LOD)");
	lod_title->add_theme_font_size_override("font_size", 14 * EDSCALE);
	current_view_tab->add_child(lod_title);
	
	lod_container = memnew(VBoxContainer);
	current_view_tab->add_child(lod_container);
	
	// Auto LOD toggle
	auto_lod_checkbox = memnew(CheckBox);
	auto_lod_checkbox->set_text("Auto LOD (Distance-Based)");
	auto_lod_checkbox->set_pressed(true);
	auto_lod_checkbox->connect("toggled", callable_mp(this, &DesignStudio3DEditor::_on_auto_lod_toggled));
	lod_container->add_child(auto_lod_checkbox);
	
	// LOD Quality Preset
	Label *quality_label = memnew(Label);
	quality_label->set_text("LOD Quality Preset:");
	lod_container->add_child(quality_label);
	
	lod_quality_selector = memnew(OptionButton);
	lod_quality_selector->add_item("Conservative (3 LODs)", 0);
	lod_quality_selector->add_item("Balanced (4 LODs)", 1);
	lod_quality_selector->add_item("Aggressive (5 LODs)", 2);
	lod_quality_selector->select(1); // Default to Balanced
	lod_quality_selector->connect("item_selected", callable_mp(this, &DesignStudio3DEditor::_on_lod_quality_changed));
	lod_container->add_child(lod_quality_selector);
	
	// Generate LODs button
	generate_lods_button = memnew(Button);
	generate_lods_button->set_text("Generate LOD Levels");
	generate_lods_button->connect("pressed", callable_mp(this, &DesignStudio3DEditor::_on_generate_lods_pressed));
	lod_container->add_child(generate_lods_button);
	
	// Current LOD display
	current_lod_label = memnew(Label);
	current_lod_label->set_text("LOD: Not generated");
	current_lod_label->add_theme_font_size_override("font_size", 11 * EDSCALE);
	lod_container->add_child(current_lod_label);
	
	// LOD Status
	lod_status_label = memnew(Label);
	lod_status_label->set_text("Generate LOD levels for automatic detail switching");
	lod_status_label->set_autowrap_mode(TextServer::AUTOWRAP_WORD_SMART);
	lod_status_label->set_custom_minimum_size(Size2(0, 40 * EDSCALE));
	lod_container->add_child(lod_status_label);
}

void DesignStudio3DEditor::_on_generate_lods_pressed() {
	if (is_generating_lods) {
		lod_status_label->set_text("[BUSY] Already generating LODs...");
		return;
	}
	
	if (current_loaded_mesh.is_null() || current_model_path.is_empty()) {
		lod_status_label->set_text("[ERROR] No model loaded to generate LODs from");
		return;
	}
	
	_start_lod_generation();
}

void DesignStudio3DEditor::_on_auto_lod_toggled(bool p_pressed) {
	auto_lod_enabled = p_pressed;
	if (auto_lod_enabled && lod_levels.size() > 0) {
		lod_status_label->set_text("[AUTO LOD] Enabled - will switch based on camera distance");
		// Update based on current distance
		_update_lod_based_on_distance();
	} else if (!auto_lod_enabled) {
		lod_status_label->set_text("[MANUAL LOD] Disabled auto switching - use slider to control");
	}
	_update_lod_slider();
}

void DesignStudio3DEditor::_on_lod_slider_changed(float p_value) {
	if (auto_lod_enabled) {
		return; // Don't allow manual changes when auto LOD is enabled
	}
	
	int target_lod = (int)p_value;
	if (target_lod != current_lod_index && target_lod < lod_levels.size()) {
		_switch_to_lod(target_lod);
	}
}

void DesignStudio3DEditor::_on_lod_quality_changed(int p_index) {
	// Quality preset changed - user can regenerate LODs with new settings
	if (lod_levels.size() > 0) {
		lod_status_label->set_text("[PRESET CHANGED] Click 'Generate LOD Levels' to apply new quality preset");
	}
}

void DesignStudio3DEditor::_start_lod_generation() {
	_clear_lod_levels();
	
	int quality_preset = lod_quality_selector->get_selected();
	
	// Define LOD configurations based on quality preset
	Vector<float> lod_face_ratios;
	Vector<float> lod_distances;
	
	switch (quality_preset) {
		case 0: // Conservative (3 LODs)
			lod_face_ratios.push_back(1.0f);    // LOD 0: 100%
			lod_face_ratios.push_back(0.5f);    // LOD 1: 50%
			lod_face_ratios.push_back(0.25f);   // LOD 2: 25%
			
			lod_distances.push_back(0.0f);      // LOD 0: 0-3 units
			lod_distances.push_back(3.0f);      // LOD 1: 3-8 units
			lod_distances.push_back(8.0f);      // LOD 2: 8+ units
			break;
			
		case 1: // Balanced (4 LODs) - DEFAULT
			lod_face_ratios.push_back(1.0f);    // LOD 0: 100%
			lod_face_ratios.push_back(0.65f);   // LOD 1: 65%
			lod_face_ratios.push_back(0.35f);   // LOD 2: 35%
			lod_face_ratios.push_back(0.15f);   // LOD 3: 15%
			
			lod_distances.push_back(0.0f);      // LOD 0: 0-2.5 units
			lod_distances.push_back(2.5f);      // LOD 1: 2.5-5 units
			lod_distances.push_back(5.0f);      // LOD 2: 5-10 units
			lod_distances.push_back(10.0f);     // LOD 3: 10+ units
			break;
			
		case 2: // Aggressive (5 LODs)
			lod_face_ratios.push_back(1.0f);    // LOD 0: 100%
			lod_face_ratios.push_back(0.7f);    // LOD 1: 70%
			lod_face_ratios.push_back(0.45f);   // LOD 2: 45%
			lod_face_ratios.push_back(0.25f);   // LOD 3: 25%
			lod_face_ratios.push_back(0.1f);    // LOD 4: 10%
			
			lod_distances.push_back(0.0f);      // LOD 0: 0-2 units
			lod_distances.push_back(2.0f);      // LOD 1: 2-4 units
			lod_distances.push_back(4.0f);      // LOD 2: 4-7 units
			lod_distances.push_back(7.0f);      // LOD 3: 7-12 units
			lod_distances.push_back(12.0f);     // LOD 4: 12+ units
			break;
	}
	
	// Create LOD 0 (original model)
	LODLevel lod0;
	lod0.mesh = current_loaded_mesh;
	lod0.model_path = current_model_path;
	lod0.target_faces = current_face_count;
	lod0.vertex_count = current_vertex_count;
	lod0.face_count = current_face_count;
	lod0.distance_threshold = lod_distances[0];
	lod_levels.push_back(lod0);
	
	// Setup for generating remaining LODs
	is_generating_lods = true;
	lods_generated_count = 1; // LOD 0 is already "generated"
	total_lods_to_generate = lod_face_ratios.size();
	current_lod_index = 0;
	
	generate_lods_button->set_disabled(true);
	
	lod_status_label->set_text("[GENERATING] Creating LOD 0 (Original): " + 
							   itos(current_face_count) + " faces\n" +
							   "Generating " + itos(total_lods_to_generate - 1) + " additional LOD levels...");
	
	// Start generating LOD 1
	if (lod_face_ratios.size() > 1) {
		_generate_next_lod();
	} else {
		// Only one LOD level requested
		is_generating_lods = false;
		generate_lods_button->set_disabled(false);
		_update_lod_info();
	}
}

void DesignStudio3DEditor::_generate_next_lod() {
	if (lods_generated_count >= total_lods_to_generate) {
		// All LODs generated
		is_generating_lods = false;
		generate_lods_button->set_disabled(false);
		_update_lod_info();
		_update_lod_slider();
		lod_status_label->set_text("[SUCCESS] Generated " + itos(lod_levels.size()) + " LOD levels!\n" +
								   "Auto LOD switching is " + (auto_lod_enabled ? "ENABLED" : "DISABLED") + "\n" +
								   (auto_lod_enabled ? "Zoom in/out to see LOD switching" : "Use slider below 3D viewer to change LOD"));
		return;
	}
	
	int quality_preset = lod_quality_selector->get_selected();
	Vector<float> lod_face_ratios;
	Vector<float> lod_distances;
	
	// Recreate ratios and distances (same as in _start_lod_generation)
	switch (quality_preset) {
		case 0: // Conservative
			lod_face_ratios.append_array({1.0f, 0.5f, 0.25f});
			lod_distances.append_array({0.0f, 3.0f, 8.0f});
			break;
		case 1: // Balanced
			lod_face_ratios.append_array({1.0f, 0.65f, 0.35f, 0.15f});
			lod_distances.append_array({0.0f, 2.5f, 5.0f, 10.0f});
			break;
		case 2: // Aggressive
			lod_face_ratios.append_array({1.0f, 0.7f, 0.45f, 0.25f, 0.1f});
			lod_distances.append_array({0.0f, 2.0f, 4.0f, 7.0f, 12.0f});
			break;
	}
	
	// Calculate target faces for current LOD
	int target_faces = (int)(current_face_count * lod_face_ratios[lods_generated_count]);
	if (target_faces < 1) target_faces = 1;
	
	// Update status
	lod_status_label->set_text("[GENERATING] LOD " + itos(lods_generated_count) + " (" + itos(lods_generated_count + 1) + "/" + itos(total_lods_to_generate) + ")\n" +
							   "Target: " + itos(target_faces) + " faces (" + itos((int)(lod_face_ratios[lods_generated_count] * 100)) + "% of original)");
	
	// Start remeshing for this LOD level
	_start_remeshing_for_lod(target_faces, lod_distances[lods_generated_count]);
}

void DesignStudio3DEditor::_start_remeshing_for_lod(int p_target_faces, float p_distance_threshold) {
	if (current_model_path.is_empty()) {
		lod_status_label->set_text("[ERROR] No model data available for LOD generation");
		return;
	}

	Ref<FileAccess> source = FileAccess::open(current_model_path, FileAccess::READ);
	if (source.is_null()) {
		lod_status_label->set_text("[ERROR] Failed to read model file for LOD generation");
		return;
	}
	PackedByteArray file_bytes = source->get_buffer(source->get_length());
	source->close();

	String boundary = "----WebKitFormBoundary" + String::num_int64(OS::get_singleton()->get_ticks_msec());
	PackedByteArray body;

	String filename = current_model_path.get_file();
	if (filename.is_empty()) {
		filename = "model.glb";
	}

	// File part
	String part1 = "--" + boundary + "\r\n";
	part1 += "Content-Disposition: form-data; name=\"file\"; filename=\"" + filename + "\"\r\n";
	part1 += "Content-Type: application/octet-stream\r\n\r\n";
	body.append_array(part1.to_utf8_buffer());
	body.append_array(file_bytes);
	body.append_array(String("\r\n").to_utf8_buffer());

	// target_faces part
	String part2 = "--" + boundary + "\r\n";
	part2 += "Content-Disposition: form-data; name=\"target_faces\"\r\n\r\n";
	part2 += itos(p_target_faces) + "\r\n";
	body.append_array(part2.to_utf8_buffer());

	String closing = "--" + boundary + "--\r\n";
	body.append_array(closing.to_utf8_buffer());

	PackedStringArray headers;
	headers.push_back("Content-Type: multipart/form-data; boundary=" + boundary);
	headers.push_back("User-Agent: Godot-Editor/4.0");
	headers.push_back("Accept: */*");

	String url = REMESH_API_URL + "/remesh";

	// Store the distance threshold for when the LOD is completed
	lod_distance_threshold_pending = p_distance_threshold;

	// Check if remesh request is busy
	if (remesh_request->get_http_client_status() != HTTPClient::STATUS_DISCONNECTED) {
		lod_status_label->set_text("[ERROR] Remesh request is busy. Please wait...");
		is_generating_lods = false;
		generate_lods_button->set_disabled(false);
		return;
	}
	
	// Disconnect any existing connections
	if (remesh_request->is_connected("request_completed", callable_mp(this, &DesignStudio3DEditor::_on_lod_generated))) {
		remesh_request->disconnect("request_completed", callable_mp(this, &DesignStudio3DEditor::_on_lod_generated));
	}
	if (remesh_request->is_connected("request_completed", callable_mp(this, &DesignStudio3DEditor::_on_remesh_completed))) {
		remesh_request->disconnect("request_completed", callable_mp(this, &DesignStudio3DEditor::_on_remesh_completed));
	}
	
	remesh_request->connect("request_completed", callable_mp(this, &DesignStudio3DEditor::_on_lod_generated), CONNECT_ONE_SHOT);

	Error err = remesh_request->request_raw(url, headers, HTTPClient::METHOD_POST, body);
	if (err != OK) {
		lod_status_label->set_text("[ERROR] Failed to start LOD remesh request. Error: " + itos(err));
		is_generating_lods = false;
		generate_lods_button->set_disabled(false);
	}
}

void DesignStudio3DEditor::_on_lod_generated(int p_result, int p_code, const PackedStringArray &p_headers, const PackedByteArray &p_body) {
	if (p_result != HTTPRequest::RESULT_SUCCESS) {
		lod_status_label->set_text("[ERROR] LOD generation failed. Result: " + itos(p_result));
		is_generating_lods = false;
		generate_lods_button->set_disabled(false);
		return;
	}

	if (p_code != 200) {
		lod_status_label->set_text("[ERROR] LOD generation failed (HTTP " + itos(p_code) + ")");
		is_generating_lods = false;
		generate_lods_button->set_disabled(false);
		return;
	}

	if (p_body.size() == 0) {
		lod_status_label->set_text("[ERROR] LOD generation response is empty");
		is_generating_lods = false;
		generate_lods_button->set_disabled(false);
		return;
	}

	// Create new LOD level from the response
	LODLevel new_lod;
	
	// Save LOD data to temp file
	String timestamp = String::num_int64(OS::get_singleton()->get_ticks_msec());
	String filename = "lod_" + itos(lods_generated_count) + "_" + timestamp + ".obj";
	String temp_path = "user://" + filename;
	
	Ref<FileAccess> file = FileAccess::open(temp_path, FileAccess::WRITE);
	if (file.is_valid()) {
		file->store_buffer(p_body);
		file->close();
		
		// Parse OBJ to get mesh and statistics
		String content = String::utf8((const char *)p_body.ptr(), p_body.size());
		Ref<ArrayMesh> lod_mesh = _parse_obj_to_mesh(content);
		
		if (lod_mesh.is_valid()) {
			// Count faces and vertices from the content
			int vertex_count = 0;
			int face_count = 0;
			PackedStringArray lines = content.split("\n");
			for (int i = 0; i < lines.size(); i++) {
				String line = lines[i].strip_edges();
				if (line.begins_with("v ")) vertex_count++;
				else if (line.begins_with("f ")) face_count++;
			}
			
			new_lod.mesh = lod_mesh;
			new_lod.model_path = temp_path;
			new_lod.vertex_count = vertex_count;
			new_lod.face_count = face_count;
			new_lod.distance_threshold = lod_distance_threshold_pending;
			
			// Calculate target faces based on current_face_count
			int quality_preset = lod_quality_selector->get_selected();
			Vector<float> lod_face_ratios;
			switch (quality_preset) {
				case 0: lod_face_ratios.append_array({1.0f, 0.5f, 0.25f}); break;
				case 1: lod_face_ratios.append_array({1.0f, 0.65f, 0.35f, 0.15f}); break;
				case 2: lod_face_ratios.append_array({1.0f, 0.7f, 0.45f, 0.25f, 0.1f}); break;
			}
			
			if (lods_generated_count < lod_face_ratios.size()) {
				new_lod.target_faces = (int)(current_face_count * lod_face_ratios[lods_generated_count]);
			} else {
				new_lod.target_faces = face_count;
			}
			
			lod_levels.push_back(new_lod);
			lods_generated_count++;
			
			lod_status_label->set_text("[SUCCESS] LOD " + itos(lods_generated_count - 1) + " created: " + 
									   itos(face_count) + " faces\n" +
									   "Progress: " + itos(lods_generated_count) + "/" + itos(total_lods_to_generate));
			
			// Generate next LOD or finish
			_generate_next_lod();
		} else {
			lod_status_label->set_text("[ERROR] Failed to parse LOD mesh data");
			is_generating_lods = false;
			generate_lods_button->set_disabled(false);
		}
	} else {
		lod_status_label->set_text("[ERROR] Failed to save LOD file");
		is_generating_lods = false;
		generate_lods_button->set_disabled(false);
	}
}

void DesignStudio3DEditor::_update_lod_based_on_distance() {
	if (!auto_lod_enabled || lod_levels.size() <= 1 || !camera || !preview_mesh) {
		return;
	}
	
	// Calculate distance from camera to model center
	Vector3 camera_pos = camera->get_global_position();
	Vector3 model_pos = preview_mesh->get_global_position();
	float distance = camera_pos.distance_to(model_pos);
	
	// Find appropriate LOD level based on distance
	int target_lod = current_lod_index;
	
	for (int i = lod_levels.size() - 1; i >= 0; i--) {
		if (distance >= lod_levels[i].distance_threshold) {
			target_lod = i;
			break;
		}
	}
	
	// Switch if needed
	if (target_lod != current_lod_index) {
		_switch_to_lod(target_lod);
	}
}

void DesignStudio3DEditor::_switch_to_lod(int p_lod_index) {
	if (p_lod_index < 0 || p_lod_index >= lod_levels.size()) {
		return;
	}
	
	if (p_lod_index == current_lod_index) {
		return; // Already using this LOD
	}
	
	current_lod_index = p_lod_index;
	
	// Switch the mesh in the viewer
	if (preview_mesh && lod_levels[current_lod_index].mesh.is_valid()) {
		preview_mesh->set_mesh(lod_levels[current_lod_index].mesh);
		
		// Update current loaded mesh reference
		current_loaded_mesh = lod_levels[current_lod_index].mesh;
		
		_update_lod_info();
		_update_lod_slider();
		
		print_line("LOD switched to level " + itos(current_lod_index) + " (" + itos(lod_levels[current_lod_index].face_count) + " faces)");
	}
}

void DesignStudio3DEditor::_clear_lod_levels() {
	lod_levels.clear();
	current_lod_index = 0;
	is_generating_lods = false;
	lods_generated_count = 0;
	total_lods_to_generate = 0;
	
	if (current_lod_label) {
		current_lod_label->set_text("LOD: Not generated");
	}
	
	_update_lod_slider();
}

void DesignStudio3DEditor::_update_lod_info() {
	if (!current_lod_label) {
		return;
	}
	
	if (lod_levels.size() == 0) {
		current_lod_label->set_text("LOD: Not generated");
		return;
	}
	
	String lod_info = "LOD " + itos(current_lod_index) + "/" + itos(lod_levels.size() - 1);
	lod_info += " (" + itos(lod_levels[current_lod_index].face_count) + " faces)";
	
	if (auto_lod_enabled) {
		lod_info += " [AUTO]";
	} else {
		lod_info += " [MANUAL]";
	}
	
	current_lod_label->set_text(lod_info);
}

void DesignStudio3DEditor::_update_lod_slider() {
	if (!lod_slider || !lod_slider_label) {
		return;
	}
	
	if (lod_levels.size() <= 1) {
		// No LODs or only one LOD - hide slider
		lod_slider->set_visible(false);
		lod_slider_label->set_text("LOD Level: Not Available");
		return;
	}
	
	// Show and configure slider
	lod_slider->set_visible(true);
	lod_slider->set_max(lod_levels.size() - 1);
	lod_slider->set_value(current_lod_index);
	
	// Update label
	String slider_text = "LOD " + itos(current_lod_index) + "/" + itos(lod_levels.size() - 1);
	slider_text += " (" + itos(lod_levels[current_lod_index].face_count) + " faces)";
	
	if (auto_lod_enabled) {
		slider_text += " - AUTO MODE";
		lod_slider->set_editable(false); // Disable slider in auto mode
		lod_slider->set_modulate(Color(0.7, 0.7, 0.7, 1.0)); // Dim it
	} else {
		slider_text += " - MANUAL";
		lod_slider->set_editable(true); // Enable slider in manual mode
		lod_slider->set_modulate(Color(1.0, 1.0, 1.0, 1.0)); // Full color
	}
	
	lod_slider_label->set_text(slider_text);
}

void DesignStudio3DEditor::_cancel_all_requests() {
	// Cancel and disconnect all HTTP requests to prevent conflicts
	
	if (submit_request) {
		submit_request->cancel_request();
		if (submit_request->is_connected("request_completed", callable_mp(this, &DesignStudio3DEditor::_on_job_submitted))) {
			submit_request->disconnect("request_completed", callable_mp(this, &DesignStudio3DEditor::_on_job_submitted));
		}
	}
	
	if (poll_request) {
		poll_request->cancel_request();
		if (poll_request->is_connected("request_completed", callable_mp(this, &DesignStudio3DEditor::_on_job_status_received))) {
			poll_request->disconnect("request_completed", callable_mp(this, &DesignStudio3DEditor::_on_job_status_received));
		}
	}
	
	if (download_request) {
		download_request->cancel_request();
		if (download_request->is_connected("request_completed", callable_mp(this, &DesignStudio3DEditor::_on_model_downloaded))) {
			download_request->disconnect("request_completed", callable_mp(this, &DesignStudio3DEditor::_on_model_downloaded));
		}
	}
	
	if (browse_request) {
		browse_request->cancel_request();
		if (browse_request->is_connected("request_completed", callable_mp(this, &DesignStudio3DEditor::_on_models_list_received))) {
			browse_request->disconnect("request_completed", callable_mp(this, &DesignStudio3DEditor::_on_models_list_received));
		}
	}
	
	if (textured_models_request) {
		textured_models_request->cancel_request();
		if (textured_models_request->is_connected("request_completed", callable_mp(this, &DesignStudio3DEditor::_on_textured_models_received))) {
			textured_models_request->disconnect("request_completed", callable_mp(this, &DesignStudio3DEditor::_on_textured_models_received));
		}
	}
	
	if (remesh_request) {
		remesh_request->cancel_request();
		if (remesh_request->is_connected("request_completed", callable_mp(this, &DesignStudio3DEditor::_on_remesh_completed))) {
			remesh_request->disconnect("request_completed", callable_mp(this, &DesignStudio3DEditor::_on_remesh_completed));
		}
		if (remesh_request->is_connected("request_completed", callable_mp(this, &DesignStudio3DEditor::_on_lod_generated))) {
			remesh_request->disconnect("request_completed", callable_mp(this, &DesignStudio3DEditor::_on_lod_generated));
		}
	}
	
	// Cancel texture system operations
	if (texture_system) {
		texture_system->cancel_texture_generation();
	}
	
	// Stop timers
	if (poll_timer && poll_timer->is_connected("timeout", callable_mp(this, &DesignStudio3DEditor::_on_poll_timeout))) {
		poll_timer->stop();
	}
	
	if (download_retry_timer && download_retry_timer->is_connected("timeout", callable_mp(this, &DesignStudio3DEditor::_on_download_retry_timeout))) {
		download_retry_timer->stop();
	}
	
	print_line("All HTTP requests cancelled and disconnected");
}

String DesignStudio3DEditor::_get_or_create_persistent_user_id() {
	const String SETTING_KEY = "3d_design_studio/user_id";
	
	// Check if we already have a stored user ID
	if (EditorSettings::get_singleton()->has_setting(SETTING_KEY)) {
		String stored_id = EditorSettings::get_singleton()->get_setting(SETTING_KEY);
		if (!stored_id.is_empty()) {
			return stored_id;
		}
	}
	
	// Generate new persistent user ID based on machine
	String machine_id = OS::get_singleton()->get_unique_id();
	
	// If machine ID is empty, create from system info
	if (machine_id.is_empty()) {
		machine_id = OS::get_singleton()->get_name() + "_" + 
					 OS::get_singleton()->get_processor_name() + "_" + 
					 String::num_int64(OS::get_singleton()->get_ticks_usec());
	}
	
	// Create hash for privacy and consistent length
	uint32_t hash = machine_id.hash();
	String user_id = "godot_" + String::num_uint64(hash, 16); // Hex format
	
	// Store permanently in editor settings
	EditorSettings::get_singleton()->set_setting(SETTING_KEY, user_id);
	EditorSettings::get_singleton()->save();
	
	return user_id;
}

void DesignStudio3DEditor::_on_add_texture_pressed() {
	if (!texture_system) {
		if (texture_status_label) {
			texture_status_label->set_text("[ERROR] Texture system not initialized");
		}
		return;
	}
	
	if (texture_system->is_texture_generation_active()) {
	if (texture_status_label) {
			texture_status_label->set_text("[BUSY] Already generating texture...");
		}
		return;
	}
	
	if (current_loaded_mesh.is_null()) {
		if (texture_status_label) {
			texture_status_label->set_text("[ERROR] No model loaded for texturing");
		}
		return;
	}
	
	// Check if we have a valid job ID for the base model
	if (current_job_id.is_empty()) {
		if (texture_status_label) {
			texture_status_label->set_text("[ERROR] Cannot generate texture without a valid base model");
		}
		return;
	}
	
	// Show texture generation dialog
	texture_system->show_texture_generation_dialog(current_job_id);
	}
	
void DesignStudio3DEditor::_on_segment_pressed() {
		if (texture_status_label) {
		texture_status_label->set_text("[INFO] Segmentation feature coming soon...");
	}
	
	// TODO: Implement segmentation using the new texture system
	// This will be added in a future update
}

void DesignStudio3DEditor::_on_remesh_pressed() {
	if (current_loaded_mesh.is_null() && current_model_path.is_empty()) {
		if (texture_status_label) {
			texture_status_label->set_text("[ERROR] No model loaded for remeshing");
		}
		return;
	}
	
	int default_faces = current_face_count > 0 ? current_face_count / 2 : 75000;
	if (default_faces < 1) {
		default_faces = 1;
	}
	if (remesh_faces_input) {
		remesh_faces_input->set_text(itos(default_faces));
	}
	if (remesh_dialog) {
		remesh_dialog->popup_centered(Size2(420 * EDSCALE, 0));
	}
}

void DesignStudio3DEditor::_on_remesh_dialog_confirmed() {
	int target_faces = remesh_faces_input ? remesh_faces_input->get_text().strip_edges().to_int() : 0;
	if (target_faces <= 0) {
		if (texture_status_label) {
			texture_status_label->set_text("[ERROR] Please enter a valid target face count");
		}
		return;
	}
	_start_remeshing(target_faces);
}

void DesignStudio3DEditor::_start_remeshing(int p_target_faces) {
	if (current_model_path.is_empty()) {
		if (texture_status_label) {
			texture_status_label->set_text("[ERROR] No model data available to upload");
		}
		return;
	}

	Ref<FileAccess> source = FileAccess::open(current_model_path, FileAccess::READ);
	if (source.is_null()) {
		if (texture_status_label) {
			texture_status_label->set_text("[ERROR] Failed to open current model file for remeshing");
		}
		return;
	}
	PackedByteArray file_bytes = source->get_buffer(source->get_length());
	source->close();

	String boundary = "----WebKitFormBoundary" + String::num_int64(OS::get_singleton()->get_ticks_msec());
	PackedByteArray body;

	String filename = current_model_path.get_file();
	if (filename.is_empty()) {
		filename = "model.glb";
	}

	// File part
	String part1 = "--" + boundary + "\r\n";
	part1 += "Content-Disposition: form-data; name=\"file\"; filename=\"" + filename + "\"\r\n";
	part1 += "Content-Type: application/octet-stream\r\n\r\n";
	body.append_array(part1.to_utf8_buffer());
	body.append_array(file_bytes);
	body.append_array(String("\r\n").to_utf8_buffer());

	// target_faces part
	String part2 = "--" + boundary + "\r\n";
	part2 += "Content-Disposition: form-data; name=\"target_faces\"\r\n\r\n";
	part2 += itos(p_target_faces) + "\r\n";
	body.append_array(part2.to_utf8_buffer());

	String closing = "--" + boundary + "--\r\n";
	body.append_array(closing.to_utf8_buffer());

	PackedStringArray headers;
	headers.push_back("Content-Type: multipart/form-data; boundary=" + boundary);
	headers.push_back("User-Agent: Godot-Editor/4.0");
	headers.push_back("Accept: */*");

	String url = REMESH_API_URL + "/remesh";

	if (texture_status_label) {
		texture_status_label->set_text("[REMESH] Uploading model for remeshing to " + itos(p_target_faces) + " faces...");
	}
	if (remesh_button) {
		remesh_button->set_disabled(true);
	}

	// Check if remesh request is busy
	if (remesh_request->get_http_client_status() != HTTPClient::STATUS_DISCONNECTED) {
		if (texture_status_label) {
			texture_status_label->set_text("[ERROR] Remesh request is busy. Please wait...");
		}
		if (remesh_button) {
			remesh_button->set_disabled(false);
		}
		return;
	}
	
	// Disconnect any existing connections
	if (remesh_request->is_connected("request_completed", callable_mp(this, &DesignStudio3DEditor::_on_remesh_completed))) {
		remesh_request->disconnect("request_completed", callable_mp(this, &DesignStudio3DEditor::_on_remesh_completed));
	}
	if (remesh_request->is_connected("request_completed", callable_mp(this, &DesignStudio3DEditor::_on_lod_generated))) {
		remesh_request->disconnect("request_completed", callable_mp(this, &DesignStudio3DEditor::_on_lod_generated));
	}
	
	remesh_request->connect("request_completed", callable_mp(this, &DesignStudio3DEditor::_on_remesh_completed), CONNECT_ONE_SHOT);

	Error err = remesh_request->request_raw(url, headers, HTTPClient::METHOD_POST, body);
	if (err != OK) {
		if (texture_status_label) {
			texture_status_label->set_text("[ERROR] Failed to start remesh request. Error: " + itos(err));
		}
		if (remesh_button) {
			remesh_button->set_disabled(false);
		}
	}
}

void DesignStudio3DEditor::_on_remesh_completed(int p_result, int p_code, const PackedStringArray &p_headers, const PackedByteArray &p_body) {
	if (p_result != HTTPRequest::RESULT_SUCCESS) {
		if (texture_status_label) {
			texture_status_label->set_text("[ERROR] Remesh request failed. Result: " + itos(p_result));
		}
		if (remesh_button) {
			remesh_button->set_disabled(false);
		}
		return;
	}

	if (p_code != 200) {
		if (texture_status_label) {
			texture_status_label->set_text("[ERROR] Remesh failed (HTTP " + itos(p_code) + ")");
		}
		if (remesh_button) {
			remesh_button->set_disabled(false);
		}
		return;
	}

	if (p_body.size() == 0) {
		if (texture_status_label) {
			texture_status_label->set_text("[ERROR] Remesh response is empty");
		}
		if (remesh_button) {
			remesh_button->set_disabled(false);
		}
		return;
	}

	// Try to determine file type from headers and content
	String content_type;
	String header_filename;
	for (int i = 0; i < p_headers.size(); i++) {
		String h = p_headers[i];
		if (h.begins_with("Content-Type:") || h.begins_with("content-type:")) {
			content_type = h.substr(h.find(":") + 1).strip_edges();
		}
		if (h.begins_with("Content-Disposition:") || h.begins_with("content-disposition:")) {
			int fn_pos = h.find("filename=");
			if (fn_pos != -1) {
				String fn = h.substr(fn_pos + 9).strip_edges();
				fn = fn.trim_prefix("\"").trim_suffix("\"");
				header_filename = fn;
			}
		}
	}

	bool looks_glb = false;
	bool looks_obj = false;
	if (p_body.size() >= 4) {
		const uint8_t *b = p_body.ptr();
		// GLB magic 'glTF'
		if (b[0] == 'g' && b[1] == 'l' && b[2] == 'T' && b[3] == 'F') {
			looks_glb = true;
		}
	}
	// Heuristic for OBJ: text-based starting tokens
	int probe_len = MIN(128, p_body.size());
	String probe = String::utf8((const char *)p_body.ptr(), probe_len);
	if (probe.begins_with("#") || probe.begins_with("v ") || probe.begins_with("o ") || probe.begins_with("mtllib") || probe.begins_with("g ")) {
		looks_obj = true;
	}

	// If server provided a filename, use its extension preference
	if (!header_filename.is_empty()) {
		String ext = header_filename.get_extension().to_lower();
		if (ext == "glb") looks_glb = true;
		if (ext == "obj") looks_obj = true;
	}

	if (looks_obj && !looks_glb) {
		// Directly load OBJ into viewer and set path via existing helper
		_load_model_from_data(p_body);
		if (texture_status_label) {
			texture_status_label->set_text("[SUCCESS] Remeshed OBJ received and loaded in viewer. Export to save.");
		}
	} else if (looks_glb) {
		String timestamp = String::num_int64(OS::get_singleton()->get_ticks_msec());
		String filename = header_filename.is_empty() ? (String("remeshed_model_") + timestamp + ".glb") : header_filename;
		String temp_path = "user://" + filename;
		Ref<FileAccess> file = FileAccess::open(temp_path, FileAccess::WRITE);
		if (file.is_valid()) {
			file->store_buffer(p_body);
			file->close();
			current_model_path = temp_path;
			if (texture_status_label) {
				texture_status_label->set_text("[SUCCESS] Remeshed GLB downloaded. Export to import into project.");
			}
		} else {
			if (texture_status_label) {
				texture_status_label->set_text("[ERROR] Failed to save remeshed GLB file");
			}
		}
	} else {
		// Unknown content; show brief snippet for debugging
		String snippet = probe;
		if (texture_status_label) {
			texture_status_label->set_text("[ERROR] Unknown remesh response format. First bytes: \n" + snippet);
		}
	}

	if (remesh_button) {
		remesh_button->set_disabled(false);
	}
}

void DesignStudio3DEditor::_on_cancel_operation_pressed() {
	// Cancel texture generation if active
	if (texture_system && texture_system->is_texture_generation_active()) {
		texture_system->cancel_texture_generation();
		
		// Hide cancel button
		if (cancel_operation_button) {
			cancel_operation_button->set_visible(false);
		}
		
		// Update status
		if (texture_status_label) {
			texture_status_label->set_text("[CANCELLED] Texture generation cancelled by user. Ready for new operations.");
		}
	}
}

// ============================================================================
// Texture System Callbacks
// ============================================================================

void DesignStudio3DEditor::_on_texture_started(const String &p_job_id) {
			if (texture_status_label) {
		String job_id_short = p_job_id.is_empty() ? "..." : p_job_id.substr(0, 8) + "...";
		texture_status_label->set_text("[TEXTURE] Started texture generation job: " + job_id_short);
	}
	
	// Disable texture button and show cancel button
	if (add_texture_button) {
		add_texture_button->set_disabled(true);
	}
	
		if (cancel_operation_button) {
			cancel_operation_button->set_visible(true);
	}
}

void DesignStudio3DEditor::_on_texture_progress(const String &p_status, const Dictionary &p_data) {
		if (texture_status_label) {
		String status_text = "[TEXTURE] " + p_status;
		
		// Add any additional progress information from p_data
		if (p_data.has("progress_percent")) {
			int progress = p_data["progress_percent"];
			status_text += " (" + itos(progress) + "%)";
		}
		
		texture_status_label->set_text(status_text);
	}
}

void DesignStudio3DEditor::_on_texture_completed(const PackedByteArray &p_model_data, const String &p_filename) {
	// Save textured model to temp directory
	String timestamp = String::num_int64(OS::get_singleton()->get_ticks_msec());
	String filename = p_filename.is_empty() ? ("textured_model_" + timestamp + ".glb") : p_filename;
	String temp_path = "user://" + filename;
	
	Ref<FileAccess> file = FileAccess::open(temp_path, FileAccess::WRITE);
	if (file.is_valid()) {
		file->store_buffer(p_model_data);
		file->close();
		
		// Update current model path to the textured version
		current_model_path = temp_path;
		
		if (texture_status_label) {
			String status_text = "[SUCCESS] Textured model completed!\n";
			status_text += "Size: " + String::humanize_size(p_model_data.size()) + "\n";
			status_text += "File: " + filename + "\n\n";
			status_text += "[INFO] Textured model ready for export to workspace!";
			texture_status_label->set_text(status_text);
		}
		
		print_line("Texture generation completed: " + temp_path);
	} else {
		if (texture_status_label) {
			texture_status_label->set_text("[ERROR] Failed to save textured model file");
		}
	}
	
	// Re-enable texture button and hide cancel button
	if (add_texture_button) {
		add_texture_button->set_disabled(false);
	}
	
		if (cancel_operation_button) {
			cancel_operation_button->set_visible(false);
		}
		}
	
void DesignStudio3DEditor::_on_texture_failed(const String &p_error_message) {
		if (texture_status_label) {
		texture_status_label->set_text("[ERROR] Texture generation failed:\n" + p_error_message);
	}
	
	// Re-enable texture button and hide cancel button
	if (add_texture_button) {
			add_texture_button->set_disabled(false);
	}
	
		if (cancel_operation_button) {
			cancel_operation_button->set_visible(false);
	}
	
	print_line("Texture generation failed: " + p_error_message);
}

// ============================================================================
// Textured Models Support
// ============================================================================

void DesignStudio3DEditor::_fetch_textured_models_for_base_model(const String &p_base_model_id, int p_item_index) {
	if (p_base_model_id.is_empty()) {
		return;
	}
	
	// Store the request mapping
	pending_textured_requests[p_base_model_id] = p_item_index;
	
	// Add to queue instead of making request immediately
	Dictionary request_data;
	request_data["base_model_id"] = p_base_model_id;
	request_data["item_index"] = p_item_index;
	request_data["url"] = "https://gpu-proxy-976792908107.us-central1.run.app/api/users/" + current_user_id + "/models/" + p_base_model_id + "/textures";
	
	textured_request_queue.push_back(request_data);
}

void DesignStudio3DEditor::_process_textured_request_queue() {
	if (is_processing_textured_requests || textured_request_queue.is_empty()) {
		return;
	}
	
	// Check if the HTTP request is available
	if (textured_models_request->get_http_client_status() != HTTPClient::STATUS_DISCONNECTED) {
		// Request is busy, try again later
		Timer *retry_timer = memnew(Timer);
		retry_timer->set_wait_time(0.1); // 100ms delay
		retry_timer->set_one_shot(true);
		add_child(retry_timer);
		retry_timer->connect("timeout", callable_mp(this, &DesignStudio3DEditor::_process_textured_request_queue));
		retry_timer->connect("timeout", Callable(retry_timer, "queue_free"), CONNECT_DEFERRED);
		retry_timer->start();
		return;
	}
	
	is_processing_textured_requests = true;
	
	// Get the next request from the queue
	Dictionary request_data = textured_request_queue[0];
	textured_request_queue.remove_at(0);
	
	String base_model_id = request_data["base_model_id"];
	String url = request_data["url"];
	
	// Disconnect any existing connections
	if (textured_models_request->is_connected("request_completed", callable_mp(this, &DesignStudio3DEditor::_on_textured_models_received))) {
		textured_models_request->disconnect("request_completed", callable_mp(this, &DesignStudio3DEditor::_on_textured_models_received));
	}
	
	textured_models_request->connect("request_completed", callable_mp(this, &DesignStudio3DEditor::_on_textured_models_received), CONNECT_ONE_SHOT);
	
	PackedStringArray headers;
	headers.push_back("User-Agent: Godot-Editor/4.0");
	
	textured_models_request->request(url, headers);
	
	print_line("Processing textured models request for: " + base_model_id);
}

void DesignStudio3DEditor::_on_textured_models_received(int p_result, int p_code, const PackedStringArray &p_headers, const PackedByteArray &p_body) {
	// Mark this request as complete
	is_processing_textured_requests = false;
	
	if (p_result != HTTPRequest::RESULT_SUCCESS || p_code != 200) {
		// Silently fail - not all models have textured versions
		print_line("No textured models found (HTTP " + itos(p_code) + ")");
		
		// Continue processing the queue
		_process_textured_request_queue();
		return;
	}
	
	String response_text;
	if (p_body.size() > 0) {
		const uint8_t *r = p_body.ptr();
		response_text = String::utf8((const char *)r, p_body.size());
	}
	
	JSON json;
	Error err = json.parse(response_text);
	if (err != OK) {
		print_line("Failed to parse textured models response");
		return;
	}
	
	Dictionary response = json.get_data();
	if (!response.has("textures")) {
		return;
	}
	
	Array textures = response["textures"];
	if (textures.size() == 0) {
		return; // No textured models for this base model
	}
	
	// Filter out textured models with invalid URLs - but show completed ones even if URL is null
	Array valid_textures;
	for (int i = 0; i < textures.size(); i++) {
		Dictionary texture_data = textures[i];
		String status = texture_data.get("status", "");
		Variant url_variant = texture_data.get("textured_mesh_url", "");
		
		print_line("Checking texture: status=" + status + ", url_variant type=" + itos(url_variant.get_type()));
		
		// Include all completed textures - we'll handle URL issues during loading
		if (status == "completed") {
			valid_textures.push_back(texture_data);
			
			String url_debug = "";
			if (url_variant.get_type() == Variant::STRING) {
				url_debug = url_variant;
	} else {
				url_debug = "null/" + itos(url_variant.get_type());
			}
			print_line("Added texture with URL: " + url_debug);
		}
	}
	
	if (valid_textures.size() == 0) {
		print_line("No completed textured models found for base model");
		return; // No valid textured models for this base model
	}
	
	textures = valid_textures; // Use only valid textures
	print_line("Found " + itos(textures.size()) + " completed textured models");
	
	// Get the base model info
	String base_model_id = "";
	if (response.has("base_model") && response["base_model"].get_type() == Variant::DICTIONARY) {
		Dictionary base_model = response["base_model"];
		base_model_id = base_model.get("id", "");
	}
	
	if (base_model_id.is_empty()) {
		return;
	}
	
	// Store textured models in cache
	textured_models_cache[base_model_id] = textures;
	
	print_line("About to update UI for base model: " + base_model_id + " with " + itos(textures.size()) + " textures");
	
	// Update the UI to show that textured models are available
	_update_model_row_with_textures(base_model_id, textures);
	
	// Don't modify the ItemList text - keep it clean
	if (pending_textured_requests.has(base_model_id)) {
		pending_textured_requests.erase(base_model_id);
	}
	
	print_line("Found " + itos(textures.size()) + " textured models for base model: " + base_model_id);
	
	// Continue processing the queue
	_process_textured_request_queue();
}

void DesignStudio3DEditor::_on_textured_model_selected(const String &p_textured_model_id) {
	if (p_textured_model_id.is_empty()) {
		return;
	}
	
	// Create a dictionary with textured model data
	Dictionary textured_model_data;
	textured_model_data["id"] = p_textured_model_id;
	textured_model_data["is_textured"] = true;
	
	// Find the textured model data in our cache
	for (const Variant &base_id_var : textured_models_cache.keys()) {
		String base_id = base_id_var;
		Array textures = textured_models_cache[base_id];
		
		for (int i = 0; i < textures.size(); i++) {
			Dictionary texture_data = textures[i];
			if (texture_data.get("id", "") == p_textured_model_id) {
				textured_model_data = texture_data;
				break;
			}
		}
	}
	
	// Load the textured model
	_load_model_for_viewing(textured_model_data);
	
	print_line("Loading textured model: " + p_textured_model_id);
}

void DesignStudio3DEditor::_show_model_selection_dialog(const Dictionary &p_base_model, const Array &p_textured_models) {
	if (!model_selection_dialog || !model_version_selector) {
		return;
	}
	
	// Store the data for later use
	pending_base_model_data = p_base_model;
	pending_textured_models = p_textured_models;
	
	// Clear and populate the option button
	model_version_selector->clear();
	
	// Add the base model as the first option
	String base_prompt = p_base_model.get("prompt", "Unknown");
	model_version_selector->add_item("Original: " + base_prompt, 0);
	model_version_selector->set_item_metadata(0, p_base_model);
	
	// Add textured models
	for (int i = 0; i < p_textured_models.size(); i++) {
		Dictionary textured_model = p_textured_models[i];
		String texture_prompt = textured_model.get("prompt", "Textured version");
		String created_at = textured_model.get("created_at", "");
		String display_text = "Textured: " + texture_prompt;
		if (!created_at.is_empty()) {
			display_text += " (" + created_at.substr(0, 10) + ")";
		}
		
		model_version_selector->add_item(display_text, i + 1);
		model_version_selector->set_item_metadata(i + 1, textured_model);
	}
	
	// Select the first item (base model) by default
	model_version_selector->select(0);
	
	// Show the dialog
	model_selection_dialog->popup_centered(Size2(500 * EDSCALE, 0));
}

void DesignStudio3DEditor::_on_model_selection_confirmed() {
	if (!model_version_selector) {
		return;
	}
	
	int selected_idx = model_version_selector->get_selected();
	if (selected_idx < 0) {
		return;
	}
	
	Dictionary selected_model_data = model_version_selector->get_item_metadata(selected_idx);
	if (selected_model_data.is_empty()) {
		return;
	}
	
	String selected_text = model_version_selector->get_item_text(selected_idx);
	browse_status_label->set_text("Loading: " + selected_text + "...");
	
	_load_model_for_viewing(selected_model_data);
}

// ============================================================================
// New Expandable UI Methods
// ============================================================================

void DesignStudio3DEditor::_create_model_row(const Dictionary &p_model_data, int p_index) {
	String model_id = p_model_data.get("id", "");
	String prompt = p_model_data.get("prompt", "Unknown");
	String created = p_model_data.get("created_at", "");
	
	// Add minimal spacing between rows
	if (p_index > 0) { // Don't add spacing before first item
		Control *spacer = memnew(Control);
		spacer->set_custom_minimum_size(Size2(0, 1 * EDSCALE)); // Minimal spacing
		models_container->add_child(spacer);
	}
	
	// Create main row container
	VBoxContainer *row_container = memnew(VBoxContainer);
	row_container->set_v_size_flags(Control::SIZE_SHRINK_CENTER);
	models_container->add_child(row_container);
	
	// Store reference to this row
	model_rows[model_id] = row_container;
	
	// Create main model button
	HBoxContainer *main_row = memnew(HBoxContainer);
	row_container->add_child(main_row);
	
	// Main model button
	Button *model_button = memnew(Button);
	// Truncate long prompts to fit in panel
	String truncated_prompt = prompt;
	if (truncated_prompt.length() > 20) {
		truncated_prompt = truncated_prompt.substr(0, 20) + "...";
	}
	model_button->set_text(truncated_prompt + " (" + created.substr(0, 10) + ")");
	model_button->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	model_button->set_text_alignment(HORIZONTAL_ALIGNMENT_LEFT);
	model_button->set_custom_minimum_size(Size2(0, 20 * EDSCALE)); // Very compact height
	model_button->connect("pressed", callable_mp(this, &DesignStudio3DEditor::_on_model_row_pressed).bind(model_id));
	main_row->add_child(model_button);
	
	// Expand button (initially hidden)
	Button *expand_button = memnew(Button);
	expand_button->set_text(">");
	expand_button->set_custom_minimum_size(Size2(20 * EDSCALE, 20 * EDSCALE));
	expand_button->set_flat(true);
	expand_button->connect("pressed", callable_mp(this, &DesignStudio3DEditor::_on_expand_button_pressed).bind(model_id));
	expand_button->hide(); // Will be shown when textured models are found
	main_row->add_child(expand_button);
	
	// Container for textured options (initially hidden)
	VBoxContainer *textured_container = memnew(VBoxContainer);
	textured_container->set_custom_minimum_size(Size2(0, 0));
	textured_container->set_v_size_flags(Control::SIZE_SHRINK_CENTER);
	textured_container->hide();
	row_container->add_child(textured_container);
	
	// Store references in button metadata for easy access
	model_button->set_meta("model_data", p_model_data);
	expand_button->set_meta("model_id", model_id);
	expand_button->set_meta("textured_container", textured_container);
	row_container->set_meta("expand_button", expand_button);
	row_container->set_meta("textured_container", textured_container);
	row_container->set_meta("model_button", model_button);
}

void DesignStudio3DEditor::_update_model_row_with_textures(const String &p_base_model_id, const Array &p_textured_models) {
	print_line("_update_model_row_with_textures called for " + p_base_model_id + " with " + itos(p_textured_models.size()) + " textures");
	
	if (!model_rows.has(p_base_model_id)) {
		print_line("ERROR: No model row found for " + p_base_model_id);
		return;
	}
	
	VBoxContainer *row_container = Object::cast_to<VBoxContainer>(model_rows[p_base_model_id]);
	if (!row_container) {
		print_line("ERROR: Could not cast row_container for " + p_base_model_id);
		return;
	}
	
	Button *expand_button = Object::cast_to<Button>(row_container->get_meta("expand_button"));
	VBoxContainer *textured_container = Object::cast_to<VBoxContainer>(row_container->get_meta("textured_container"));
	Button *model_button = Object::cast_to<Button>(row_container->get_meta("model_button"));
	
	if (!expand_button || !textured_container || !model_button) {
		String debug_msg = "ERROR: Missing UI elements - expand_button: ";
		debug_msg += expand_button ? "OK" : "NULL";
		debug_msg += ", textured_container: ";
		debug_msg += textured_container ? "OK" : "NULL";
		debug_msg += ", model_button: ";
		debug_msg += model_button ? "OK" : "NULL";
		print_line(debug_msg);
		return;
	}
	
	print_line("UI elements found, updating row for " + p_base_model_id);
	
	// Change the row color to indicate textured models are available
	Color textured_color = Color(0.4, 0.6, 1.0, 1.0); // More visible blue tint
	model_button->set_modulate(textured_color);
	
	// Force show expand button and make it visible
	expand_button->show();
	expand_button->set_visible(true);
	
	// Force update the layout
	row_container->queue_redraw();
	models_container->queue_redraw();
	
	print_line("Expand button shown and row updated for " + p_base_model_id);
	
	// Create textured model options (limit to prevent UI overflow)
	int max_textured_to_show = 2; // Limit to 2 textured models per base model to prevent UI explosion
	int textured_count = MIN(p_textured_models.size(), max_textured_to_show);
	
	for (int i = 0; i < textured_count; i++) {
		Dictionary textured_model = p_textured_models[i];
		String texture_id = textured_model.get("id", "");
		String texture_prompt = textured_model.get("prompt", "Textured version");
		String created_at = textured_model.get("created_at", "");
		
		print_line("Creating textured button for: " + texture_id + " with prompt: " + texture_prompt);
		
		// Truncate long prompts to prevent UI overflow
		if (texture_prompt.length() > 20) {
			texture_prompt = texture_prompt.substr(0, 20) + "...";
		}
		
		Button *textured_button = memnew(Button);
		String display_text = "  [T] " + texture_prompt;
		textured_button->set_text(display_text);
		textured_button->set_text_alignment(HORIZONTAL_ALIGNMENT_LEFT);
		textured_button->set_h_size_flags(Control::SIZE_EXPAND_FILL);
		textured_button->set_custom_minimum_size(Size2(0, 14 * EDSCALE)); // Super compact
		textured_button->set_flat(true);
		textured_button->connect("pressed", callable_mp(this, &DesignStudio3DEditor::_on_textured_option_pressed).bind(texture_id));
		textured_button->set_meta("textured_model_data", textured_model);
		
		// Simple color modulation for textured options
		Color textured_option_color = Color(0.8, 0.9, 1.0, 1.0);
		textured_button->set_modulate(textured_option_color);
		
		textured_container->add_child(textured_button);
		print_line("Added textured button to container");
	}
	
	print_line("Created " + itos(textured_count) + " textured buttons");
	
	// Show "..." button if there are more textured models than we're showing
	if (p_textured_models.size() > max_textured_to_show) {
		Button *more_button = memnew(Button);
		more_button->set_text("  +" + itos(p_textured_models.size() - max_textured_to_show) + " more");
		more_button->set_text_alignment(HORIZONTAL_ALIGNMENT_LEFT);
		more_button->set_h_size_flags(Control::SIZE_EXPAND_FILL);
		more_button->set_custom_minimum_size(Size2(0, 12 * EDSCALE));
		more_button->set_flat(true);
		more_button->set_disabled(true); // Just for info, not clickable
		textured_container->add_child(more_button);
	}
}

void DesignStudio3DEditor::_on_model_row_pressed(const String &p_model_id) {
	if (!model_rows.has(p_model_id)) {
		return;
	}
	
	VBoxContainer *row_container = Object::cast_to<VBoxContainer>(model_rows[p_model_id]);
	Button *model_button = Object::cast_to<Button>(row_container->get_meta("model_button"));
	
	if (!model_button) {
		return;
	}
	
	Dictionary model_data = model_button->get_meta("model_data");
	String prompt = model_data.get("prompt", "Unknown");
	
	browse_status_label->set_text("Loading: " + prompt + "...");
	_load_model_for_viewing(model_data);
}

void DesignStudio3DEditor::_on_expand_button_pressed(const String &p_model_id) {
	print_line("Expand button pressed for model: " + p_model_id);
	
	if (!model_rows.has(p_model_id)) {
		print_line("ERROR: Model row not found for: " + p_model_id);
		return;
	}
	
	VBoxContainer *row_container = Object::cast_to<VBoxContainer>(model_rows[p_model_id]);
	Button *expand_button = Object::cast_to<Button>(row_container->get_meta("expand_button"));
	VBoxContainer *textured_container = Object::cast_to<VBoxContainer>(row_container->get_meta("textured_container"));
	
	if (!expand_button || !textured_container) {
		print_line("ERROR: Missing expand button or textured container");
		return;
	}
	
	bool is_expanded = expanded_models.get(p_model_id, false);
	expanded_models[p_model_id] = !is_expanded;
	
	String debug_msg = "Toggle expand state: was ";
	debug_msg += is_expanded ? "expanded" : "collapsed";
	debug_msg += ", now ";
	debug_msg += (!is_expanded) ? "expanded" : "collapsed";
	print_line(debug_msg);
	
	if (is_expanded) {
		// Collapse
		textured_container->hide();
		expand_button->set_text(">");
		print_line("Collapsed textured options for " + p_model_id);
	} else {
		// Expand
		textured_container->show();
		expand_button->set_text("v");
		String debug_msg = "Expanded textured options for " + p_model_id + " - container has " + itos(textured_container->get_child_count()) + " children";
		print_line(debug_msg);
	}
}

void DesignStudio3DEditor::_on_textured_option_pressed(const String &p_textured_model_id) {
	if (p_textured_model_id.is_empty()) {
		return;
	}
	
	// Find the textured model data
	Dictionary textured_model_data;
	for (const Variant &base_id_var : textured_models_cache.keys()) {
		String base_id = base_id_var;
		Array textures = textured_models_cache[base_id];
		
		for (int i = 0; i < textures.size(); i++) {
			Dictionary texture_data = textures[i];
			if (texture_data.get("id", "") == p_textured_model_id) {
				textured_model_data = texture_data;
				break;
			}
		}
		
		if (!textured_model_data.is_empty()) {
			break;
		}
	}
	
	if (textured_model_data.is_empty()) {
		browse_status_label->set_text("[ERROR] Textured model data not found");
		return;
	}
	
	String texture_prompt = textured_model_data.get("prompt", "Textured model");
	browse_status_label->set_text("Loading textured model: " + texture_prompt + "...");
	
	// Force mark this as textured for proper URL handling
	textured_model_data["is_textured"] = true;
	
	print_line("About to load textured model with data: " + JSON::stringify(textured_model_data));
	
	_load_model_for_viewing(textured_model_data);
}

DesignStudio3DEditor::DesignStudio3DEditor() {
	set_name("DesignStudio3D");
	
	// Generate persistent user ID on first creation
	current_user_id = _get_or_create_persistent_user_id();
	
	// Initialize texture system
	texture_system = memnew(DesignStudioTextureSystem);
	
	// Confirm user ID is working
	print_line("3D Design Studio initialized with persistent user ID: " + current_user_id);
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