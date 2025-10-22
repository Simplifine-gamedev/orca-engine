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
#include "scene/gui/separator.h"
#include "scene/gui/split_container.h"
#include "scene/gui/subviewport_container.h"
#include "scene/gui/tab_container.h"
#include "scene/gui/texture_rect.h"
#include "scene/main/timer.h"
#include "scene/resources/3d/primitive_meshes.h"
#include "scene/resources/packed_scene.h"
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
	
	// Left panel with tabs
	VBoxContainer *left_panel = memnew(VBoxContainer);
	left_panel->set_custom_minimum_size(Size2(320 * EDSCALE, 0));
	hsplit->add_child(left_panel);
	
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
	
	models_list = memnew(ItemList);
	models_list->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	models_list->set_custom_minimum_size(Size2(0, 200 * EDSCALE));
	browse_tab->add_child(models_list);
	
	load_selected_button = memnew(Button);
	load_selected_button->set_text("Load Selected");
	load_selected_button->connect("pressed", callable_mp(this, &DesignStudio3DEditor::_on_load_selected_pressed));
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
	
	VBoxContainer *viewer_vbox = memnew(VBoxContainer);
	right_panel->add_child(viewer_vbox);
	
	Label *viewer_title = memnew(Label);
	viewer_title->set_text("3D Preview (Isolated)");
	viewer_title->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_CENTER);
	viewer_vbox->add_child(viewer_title);
	
	// SubViewport for 3D preview
	viewport_container = memnew(SubViewportContainer);
	viewport_container->set_stretch(true);
	viewport_container->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	viewport_container->set_custom_minimum_size(Size2(400 * EDSCALE, 400 * EDSCALE));
	viewer_vbox->add_child(viewport_container);
	
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
	
	// Texture/Segmentation HTTP Requests
	texture_request = memnew(HTTPRequest);
	texture_request->set_name("TextureRequest");
	texture_request->set_timeout(60); // 60 seconds timeout
	add_child(texture_request);
	
	segment_request = memnew(HTTPRequest);
	segment_request->set_name("SegmentRequest");
	segment_request->set_timeout(60); // 60 seconds timeout
	add_child(segment_request);
	
	texture_poll_request = memnew(HTTPRequest);
	texture_poll_request->set_name("TexturePollRequest");
	texture_poll_request->set_timeout(30); // 30 seconds timeout
	add_child(texture_poll_request);
	
	texture_download_request = memnew(HTTPRequest);
	texture_download_request->set_name("TextureDownloadRequest");
	texture_download_request->set_timeout(180); // 3 minutes timeout for large files
	texture_download_request->set_body_size_limit(50 * 1024 * 1024); // 50 MB limit
	texture_download_request->set_use_threads(true);
	add_child(texture_download_request);
	
	// Remeshing HTTP request
	remesh_request = memnew(HTTPRequest);
	remesh_request->set_name("RemeshRequest");
	remesh_request->set_timeout(180); // allow up to 3 minutes
	remesh_request->set_body_size_limit(200 * 1024 * 1024); // 200 MB limit
	remesh_request->set_use_threads(true);
	add_child(remesh_request);
	
	// Texture poll timer
	texture_poll_timer = memnew(Timer);
	texture_poll_timer->set_wait_time(5.0); // Poll every 5 seconds
	texture_poll_timer->set_one_shot(false);
	texture_poll_timer->connect("timeout", callable_mp(this, &DesignStudio3DEditor::_on_texture_poll_timeout));
	add_child(texture_poll_timer);
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
	
	// Action buttons section
	Label *actions_title = memnew(Label);
	actions_title->set_text("Model Actions");
	actions_title->add_theme_font_size_override("font_size", 14 * EDSCALE);
	current_view_tab->add_child(actions_title);
	
	add_texture_button = memnew(Button);
	add_texture_button->set_text("Add Texture (Demo: Red Sports Car)");
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
		
		// Clear parent job ID so it gets refreshed with current model's job_id
		current_parent_job_id = "";
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
	
	model_info_label->set_text(info_text);
}

void DesignStudio3DEditor::_on_generate_pressed() {
	if (is_generating) {
		status_label->set_text("[BUSY] Already generating...");
		return;
	}
	
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
	
	// Clear texture state
	current_parent_job_id = "";
	current_texture_job_id = "";
	is_texturing = false;
	is_segmenting = false;
	
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
	}
}

void DesignStudio3DEditor::_start_download_with_headers(const String &p_url) {
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
	
	models_list->clear();
	for (int i = 0; i < models.size(); i++) {
		Dictionary model = models[i];
		String prompt = model.get("prompt", "Unknown");
		String id = model.get("id", "");
		String created = model.get("created_at", "");
		
		String display_text = prompt + " (" + created.substr(0, 10) + ")";
		models_list->add_item(display_text);
		models_list->set_item_metadata(i, model);
	}
	
	browse_status_label->set_text("Loaded " + itos(count) + " completed models");
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
	
	if (model_url.is_empty()) {
		browse_status_label->set_text("[ERROR] No URL for this model");
		return;
	}
	
	browse_status_label->set_text("Downloading...");
	
	// Download but DON'T save to workspace yet
	download_request->connect("request_completed", callable_mp(this, &DesignStudio3DEditor::_on_model_downloaded), CONNECT_ONE_SHOT);
	download_request->request(model_url);
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
	
	// Save the cached model data to workspace (preserve extension)
	String timestamp = String::num_int64(OS::get_singleton()->get_ticks_msec());
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
		} else {
			status_label->set_text("[ERROR] Failed to write to workspace");
		}
	} else {
		status_label->set_text("[ERROR] Failed to read temp model");
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
	if (is_texturing) {
		status_label->set_text("[BUSY] Already generating texture...");
		return;
	}
	
	if (current_loaded_mesh.is_null()) {
		status_label->set_text("[ERROR] No model loaded for texturing");
		return;
	}
	
	// Simple prompt for texture description - replace with proper dialog later
	String prompt = "red sports car"; // TODO: Replace with actual user input dialog
	if (texture_status_label) {
		texture_status_label->set_text("[TEXTURE] Starting texture generation with prompt: " + prompt);
	}
	
	_create_parent_job_if_needed();
	
	// Check if we have a valid parent job ID
	if (current_parent_job_id.is_empty()) {
		if (texture_status_label) {
			texture_status_label->set_text("[ERROR] Cannot generate texture without a valid parent model");
		}
		return;
	}
	
	_start_texture_generation(prompt);
}

void DesignStudio3DEditor::_on_segment_pressed() {
	if (is_segmenting) {
		if (texture_status_label) {
			texture_status_label->set_text("[BUSY] Already segmenting...");
		}
		return;
	}
	
	if (current_loaded_mesh.is_null()) {
		if (texture_status_label) {
			texture_status_label->set_text("[ERROR] No model loaded for segmentation");
		}
		return;
	}
	
	if (texture_status_label) {
		texture_status_label->set_text("[SEGMENT] Starting model segmentation...");
	}
	
	_create_parent_job_if_needed();
	
	// Check if we have a valid parent job ID
	if (current_parent_job_id.is_empty()) {
		if (texture_status_label) {
			texture_status_label->set_text("[ERROR] Cannot segment model without a valid parent model");
		}
		return;
	}
	
	_start_segmentation();
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

	if (remesh_request->is_connected("request_completed", callable_mp(this, &DesignStudio3DEditor::_on_remesh_completed))) {
		remesh_request->disconnect("request_completed", callable_mp(this, &DesignStudio3DEditor::_on_remesh_completed));
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
	// Stop any active texture/segmentation operations
	if (is_texturing || is_segmenting) {
		texture_poll_timer->stop();
		
		// Reset states
		is_texturing = false;
		is_segmenting = false;
		add_texture_button->set_disabled(false);
		segment_button->set_disabled(false);
		
		// Hide cancel button
		if (cancel_operation_button) {
			cancel_operation_button->set_visible(false);
		}
		
		// Update status
		if (texture_status_label) {
			texture_status_label->set_text("[CANCELLED] Operation cancelled by user. Ready for new operations.");
		}
		
		// Clear job ID
		current_texture_job_id = "";
	}
}

void DesignStudio3DEditor::_create_parent_job_if_needed() {
	// Use current_job_id as parent_job_id if we have one from successful generation
	if (current_parent_job_id.is_empty()) {
		if (!current_job_id.is_empty()) {
			current_parent_job_id = current_job_id;
			if (texture_status_label) {
				texture_status_label->set_text("[TEXTURE] Using job ID: " + current_parent_job_id.substr(0, 12) + "... as parent");
			}
		} else {
			// No parent model available - this shouldn't happen if we only show Current View after model loading
			if (texture_status_label) {
				texture_status_label->set_text("[ERROR] No parent model available for texture generation");
			}
			return;
		}
	}
}

void DesignStudio3DEditor::_start_texture_generation(const String &p_prompt) {
	// Create multipart form data (NO FILE UPLOAD - backend downloads from storage)
	String boundary = "----WebKitFormBoundary" + String::num_int64(OS::get_singleton()->get_ticks_msec());
	String form_data;
	
	// Add user_id
	form_data += "--" + boundary + "\r\n";
	form_data += "Content-Disposition: form-data; name=\"user_id\"\r\n\r\n";
	form_data += current_user_id + "\r\n";
	
	// Add parent_job_id
	form_data += "--" + boundary + "\r\n";
	form_data += "Content-Disposition: form-data; name=\"parent_job_id\"\r\n\r\n";
	form_data += current_parent_job_id + "\r\n";
	
	// Add mesh_filename (GLB format as expected by server)
	form_data += "--" + boundary + "\r\n";
	form_data += "Content-Disposition: form-data; name=\"mesh_filename\"\r\n\r\n";
	form_data += "model.glb\r\n";
	
	// Add text_prompt
	form_data += "--" + boundary + "\r\n";
	form_data += "Content-Disposition: form-data; name=\"text_prompt\"\r\n\r\n";
	form_data += p_prompt + "\r\n";
	
	// Close form
	form_data += "--" + boundary + "--\r\n";
	
	PackedByteArray form_bytes = form_data.to_utf8_buffer();
	PackedStringArray headers;
	headers.push_back("Content-Type: multipart/form-data; boundary=" + boundary);
	
	String url = TEXTURE_API_URL + "/texture/text-to-texture-single";
	
	// Disconnect any existing connections first
	if (texture_request->is_connected("request_completed", callable_mp(this, &DesignStudio3DEditor::_on_texture_job_submitted))) {
		texture_request->disconnect("request_completed", callable_mp(this, &DesignStudio3DEditor::_on_texture_job_submitted));
	}
	
	texture_request->connect("request_completed", callable_mp(this, &DesignStudio3DEditor::_on_texture_job_submitted), CONNECT_ONE_SHOT);
	
	Error err = texture_request->request_raw(url, headers, HTTPClient::METHOD_POST, form_bytes);
	
	if (err == OK) {
		is_texturing = true;
		add_texture_button->set_disabled(true);
		if (cancel_operation_button) {
			cancel_operation_button->set_visible(true);
		}
		if (texture_status_label) {
			texture_status_label->set_text("[TEXTURE] Submitting texture generation job...");
		}
	} else {
		if (texture_status_label) {
			texture_status_label->set_text("[ERROR] Failed to start texture request. Error: " + itos(err));
		}
	}
}

void DesignStudio3DEditor::_start_segmentation() {
	// Create multipart form data (NO FILE UPLOAD - backend downloads from storage)
	String boundary = "----WebKitFormBoundary" + String::num_int64(OS::get_singleton()->get_ticks_msec());
	String form_data;
	
	// Add user_id
	form_data += "--" + boundary + "\r\n";
	form_data += "Content-Disposition: form-data; name=\"user_id\"\r\n\r\n";
	form_data += current_user_id + "\r\n";
	
	// Add parent_job_id
	form_data += "--" + boundary + "\r\n";
	form_data += "Content-Disposition: form-data; name=\"parent_job_id\"\r\n\r\n";
	form_data += current_parent_job_id + "\r\n";
	
	// Add mesh_filename (GLB format as expected by server)
	form_data += "--" + boundary + "\r\n";
	form_data += "Content-Disposition: form-data; name=\"mesh_filename\"\r\n\r\n";
	form_data += "model.glb\r\n";
	
	// Close form
	form_data += "--" + boundary + "--\r\n";
	
	PackedByteArray form_bytes = form_data.to_utf8_buffer();
	PackedStringArray headers;
	headers.push_back("Content-Type: multipart/form-data; boundary=" + boundary);
	
	String url = TEXTURE_API_URL + "/segment";
	
	// Disconnect any existing connections first
	if (segment_request->is_connected("request_completed", callable_mp(this, &DesignStudio3DEditor::_on_segment_job_submitted))) {
		segment_request->disconnect("request_completed", callable_mp(this, &DesignStudio3DEditor::_on_segment_job_submitted));
	}
	
	segment_request->connect("request_completed", callable_mp(this, &DesignStudio3DEditor::_on_segment_job_submitted), CONNECT_ONE_SHOT);
	
	Error err = segment_request->request_raw(url, headers, HTTPClient::METHOD_POST, form_bytes);
	
	if (err == OK) {
		is_segmenting = true;
		segment_button->set_disabled(true);
		if (cancel_operation_button) {
			cancel_operation_button->set_visible(true);
		}
		if (texture_status_label) {
			texture_status_label->set_text("[SEGMENT] Submitting segmentation job...");
		}
	} else {
		if (texture_status_label) {
			texture_status_label->set_text("[ERROR] Failed to start segmentation request. Error: " + itos(err));
		}
	}
}

void DesignStudio3DEditor::_on_texture_job_submitted(int p_result, int p_code, const PackedStringArray &p_headers, const PackedByteArray &p_body) {
	String job_type = is_texturing ? "texture" : "segmentation";
	
	if (p_result != HTTPRequest::RESULT_SUCCESS || p_code != 200) {
		if (texture_status_label) {
			texture_status_label->set_text("[ERROR] Failed to submit " + job_type + " job (HTTP " + itos(p_code) + ")");
		}
		is_texturing = false;
		is_segmenting = false;
		add_texture_button->set_disabled(false);
		segment_button->set_disabled(false);
		if (cancel_operation_button) {
			cancel_operation_button->set_visible(false);
		}
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
		if (texture_status_label) {
			texture_status_label->set_text("[ERROR] Failed to parse " + job_type + " response");
		}
		is_texturing = false;
		is_segmenting = false;
		add_texture_button->set_disabled(false);
		segment_button->set_disabled(false);
		if (cancel_operation_button) {
			cancel_operation_button->set_visible(false);
		}
		return;
	}
	
	Dictionary response = json.get_data();
	
	if (response.has("job_id")) {
		current_texture_job_id = response["job_id"];
		String job_type_caps = job_type.capitalize();
		if (texture_status_label) {
			texture_status_label->set_text("[SUCCESS] " + job_type_caps + " job submitted! ID: " + current_texture_job_id.substr(0, 8) + "...\n[POLLING] Checking status...");
		}
		_start_texture_polling(current_texture_job_id);
	} else {
		if (texture_status_label) {
			texture_status_label->set_text("[ERROR] No job ID in " + job_type + " response");
		}
		is_texturing = false;
		is_segmenting = false;
		add_texture_button->set_disabled(false);
		segment_button->set_disabled(false);
		if (cancel_operation_button) {
			cancel_operation_button->set_visible(false);
		}
	}
}

void DesignStudio3DEditor::_on_segment_job_submitted(int p_result, int p_code, const PackedStringArray &p_headers, const PackedByteArray &p_body) {
	if (p_result != HTTPRequest::RESULT_SUCCESS || p_code != 200) {
		if (texture_status_label) {
			texture_status_label->set_text("[ERROR] Failed to submit segmentation job (HTTP " + itos(p_code) + ")");
		}
		is_segmenting = false;
		segment_button->set_disabled(false);
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
		if (texture_status_label) {
			texture_status_label->set_text("[ERROR] Failed to parse segmentation response");
		}
		is_segmenting = false;
		segment_button->set_disabled(false);
		return;
	}
	
	Dictionary response = json.get_data();
	
	if (response.has("job_id")) {
		current_texture_job_id = response["job_id"];
		if (texture_status_label) {
			texture_status_label->set_text("[SUCCESS] Segmentation job submitted! ID: " + current_texture_job_id.substr(0, 8) + "...\n[POLLING] Checking status...");
		}
		_start_texture_polling(current_texture_job_id);
	} else {
		if (texture_status_label) {
			texture_status_label->set_text("[ERROR] No job ID in segmentation response");
		}
		is_segmenting = false;
		segment_button->set_disabled(false);
	}
}

void DesignStudio3DEditor::_start_texture_polling(const String &p_job_id) {
	current_texture_job_id = p_job_id;
	texture_poll_timer->start();
	// Immediately poll once
	_on_texture_poll_timeout();
}

void DesignStudio3DEditor::_on_texture_poll_timeout() {
	if (current_texture_job_id.is_empty()) {
		texture_poll_timer->stop();
		return;
	}
	
	String url = TEXTURE_API_URL + "/jobs/" + current_texture_job_id;
	
	// Disconnect any existing connections first
	if (texture_poll_request->is_connected("request_completed", callable_mp(this, &DesignStudio3DEditor::_on_texture_status_received))) {
		texture_poll_request->disconnect("request_completed", callable_mp(this, &DesignStudio3DEditor::_on_texture_status_received));
	}
	
	texture_poll_request->connect("request_completed", callable_mp(this, &DesignStudio3DEditor::_on_texture_status_received), CONNECT_ONE_SHOT);
	texture_poll_request->request(url);
}

void DesignStudio3DEditor::_on_texture_status_received(int p_result, int p_code, const PackedStringArray &p_headers, const PackedByteArray &p_body) {
	String job_type = is_texturing ? "texture" : "segmentation";
	
	if (p_result != HTTPRequest::RESULT_SUCCESS || p_code != 200) {
		if (texture_status_label) {
			texture_status_label->set_text("[ERROR] Failed to get " + job_type + " job status (HTTP " + itos(p_code) + ")");
		}
		texture_poll_timer->stop();
		is_texturing = false;
		is_segmenting = false;
		add_texture_button->set_disabled(false);
		segment_button->set_disabled(false);
		if (cancel_operation_button) {
			cancel_operation_button->set_visible(false);
		}
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
		if (texture_status_label) {
			texture_status_label->set_text("[ERROR] Failed to parse " + job_type + " status response");
		}
		texture_poll_timer->stop();
		is_texturing = false;
		is_segmenting = false;
		add_texture_button->set_disabled(false);
		segment_button->set_disabled(false);
		if (cancel_operation_button) {
			cancel_operation_button->set_visible(false);
		}
		return;
	}
	
	Dictionary job_data = json.get_data();
	
	if (!job_data.has("status")) {
		if (texture_status_label) {
			texture_status_label->set_text("[ERROR] Invalid " + job_type + " status response");
		}
		texture_poll_timer->stop();
		is_texturing = false;
		is_segmenting = false;
		add_texture_button->set_disabled(false);
		segment_button->set_disabled(false);
		return;
	}
	
	String status = job_data["status"];
	
	if (status == "queued") {
		String queue_info = "";
		if (job_data.has("queue_position")) {
			int position = job_data["queue_position"];
			queue_info = " (Position: " + itos(position) + " in queue)";
		}
		if (texture_status_label) {
			texture_status_label->set_text("[QUEUED] " + job_type.capitalize() + " job queued... waiting for GPU" + queue_info);
		}
	} else if (status == "processing") {
		String timing = is_texturing ? "(this may take 60-90 seconds)" : "(this may take 20-30 seconds)";
		String progress_info = "";
		if (job_data.has("progress_percent")) {
			int progress = job_data["progress_percent"];
			progress_info = " - " + itos(progress) + "% complete";
		}
		if (texture_status_label) {
			texture_status_label->set_text("[PROCESSING] Generating " + job_type + " on GPU... " + timing + progress_info);
		}
	} else if (status == "completed") {
		texture_poll_timer->stop();
		
		String file_url = "";
		if (job_data.has("texture_file_url")) {
			file_url = job_data["texture_file_url"];
		} else if (job_data.has("segmented_model_url")) {
			file_url = job_data["segmented_model_url"];
		} else if (job_data.has("output_file_url")) {
			file_url = job_data["output_file_url"];
		}
		
		if (file_url.is_empty()) {
			if (texture_status_label) {
				texture_status_label->set_text("[ERROR] No file URL in completed " + job_type + " response");
			}
			is_texturing = false;
			is_segmenting = false;
			add_texture_button->set_disabled(false);
			segment_button->set_disabled(false);
			return;
		}
		
		if (texture_status_label) {
			texture_status_label->set_text("[COMPLETE] Downloading " + job_type + " result...");
		}
		
		// Disconnect any existing connections first
		if (texture_download_request->is_connected("request_completed", callable_mp(this, &DesignStudio3DEditor::_on_textured_model_downloaded))) {
			texture_download_request->disconnect("request_completed", callable_mp(this, &DesignStudio3DEditor::_on_textured_model_downloaded));
		}
		
		texture_download_request->connect("request_completed", callable_mp(this, &DesignStudio3DEditor::_on_textured_model_downloaded), CONNECT_ONE_SHOT);
		texture_download_request->request(file_url);
	} else if (status == "failed") {
		texture_poll_timer->stop();
		String error_msg = job_data.get("error_message", "Unknown error");
		if (texture_status_label) {
			texture_status_label->set_text("[FAILED] " + job_type.capitalize() + " generation failed: " + error_msg);
		}
		is_texturing = false;
		is_segmenting = false;
		add_texture_button->set_disabled(false);
		segment_button->set_disabled(false);
		if (cancel_operation_button) {
			cancel_operation_button->set_visible(false);
		}
	} else if (status == "initializing" || status == "starting") {
		if (texture_status_label) {
			texture_status_label->set_text("[STARTING] Initializing " + job_type + " generation...");
		}
	} else {
		if (texture_status_label) {
			texture_status_label->set_text(job_type.capitalize() + " Status: " + status);
		}
	}
}

void DesignStudio3DEditor::_on_textured_model_downloaded(int p_result, int p_code, const PackedStringArray &p_headers, const PackedByteArray &p_body) {
	String job_type = is_texturing ? "texture" : "segmentation";
	
	if (p_result != HTTPRequest::RESULT_SUCCESS) {
		if (texture_status_label) {
			texture_status_label->set_text("[ERROR] Failed to download " + job_type + " result. Result: " + itos(p_result));
		}
		is_texturing = false;
		is_segmenting = false;
		add_texture_button->set_disabled(false);
		segment_button->set_disabled(false);
		return;
	}
	
	if (p_code != 200) {
		if (texture_status_label) {
			texture_status_label->set_text("[ERROR] Failed to download " + job_type + " result (HTTP " + itos(p_code) + ")");
		}
		is_texturing = false;
		is_segmenting = false;
		add_texture_button->set_disabled(false);
		segment_button->set_disabled(false);
		return;
	}
	
	if (p_body.size() > 0) {
		// Save result to temp directory
		String timestamp = String::num_int64(OS::get_singleton()->get_ticks_msec());
		String extension = is_texturing ? ".glb" : ".obj";
		String filename = job_type + "_model_" + timestamp + extension;
		String temp_path = "user://" + filename;
		
		Ref<FileAccess> file = FileAccess::open(temp_path, FileAccess::WRITE);
		if (file.is_valid()) {
			file->store_buffer(p_body);
			file->close();
			
			if (texture_status_label) {
				texture_status_label->set_text("[SUCCESS] " + job_type.capitalize() + " result downloaded!\nSize: " + String::humanize_size(p_body.size()) + "\nSaved as: " + filename + "\n\n[INFO] " + job_type.capitalize() + " result ready for viewing and export!");
			}
			
			// Update current model path to the processed version
			current_model_path = temp_path;
			
			// Try to load the processed model in viewer
			// For now, just show success message - GLB/OBJ loading might need additional work
		} else {
			if (texture_status_label) {
				texture_status_label->set_text("[ERROR] Failed to save " + job_type + " result file");
			}
		}
	} else {
		if (texture_status_label) {
			texture_status_label->set_text("[ERROR] Downloaded " + job_type + " result is empty");
		}
	}
	
	// Reset state
	is_texturing = false;
	is_segmenting = false;
	add_texture_button->set_disabled(false);
	segment_button->set_disabled(false);
	current_texture_job_id = "";
	
	// Hide cancel button
	if (cancel_operation_button) {
		cancel_operation_button->set_visible(false);
	}
}

DesignStudio3DEditor::DesignStudio3DEditor() {
	set_name("DesignStudio3D");
	
	// Generate persistent user ID on first creation
	current_user_id = _get_or_create_persistent_user_id();
	
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