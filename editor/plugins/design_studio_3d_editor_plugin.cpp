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
#include "scene/gui/item_list.h"
#include "scene/gui/label.h"
#include "scene/gui/line_edit.h"
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
	
	// === GENERATE TAB ===
	VBoxContainer *generate_tab = memnew(VBoxContainer);
	generate_tab->set_name("Generate New");
	mode_tabs->add_child(generate_tab);
	
	prompt_input = memnew(LineEdit);
	prompt_input->set_placeholder("Describe model (e.g., 'a robot')");
	generate_tab->add_child(prompt_input);
	
	quality_selector = memnew(OptionButton);
	quality_selector->add_item("Turbo (~20s)", 0);
	quality_selector->add_item("Standard (~2min)", 1);
	quality_selector->add_item("High (~5min)", 2);
	quality_selector->select(0);
	generate_tab->add_child(quality_selector);
	
	generate_button = memnew(Button);
	generate_button->set_text("Generate");
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
	
	// === IMAGE TO 3D TAB ===
	VBoxContainer *image_tab = memnew(VBoxContainer);
	image_tab->set_name("Image to 3D");
	mode_tabs->add_child(image_tab);
	
	select_image_button = memnew(Button);
	select_image_button->set_text("Select Image File");
	select_image_button->connect("pressed", callable_mp(this, &DesignStudio3DEditor::_on_select_image_pressed));
	image_tab->add_child(select_image_button);
	
	image_path_label = memnew(Label);
	image_path_label->set_text("No image selected");
	image_path_label->set_autowrap_mode(TextServer::AUTOWRAP_WORD_SMART);
	image_tab->add_child(image_path_label);
	
	image_preview = memnew(TextureRect);
	image_preview->set_custom_minimum_size(Size2(200 * EDSCALE, 150 * EDSCALE));
	image_preview->set_expand_mode(TextureRect::EXPAND_FIT_WIDTH_PROPORTIONAL);
	image_preview->set_stretch_mode(TextureRect::STRETCH_KEEP_ASPECT_CENTERED);
	image_tab->add_child(image_preview);
	
	image_quality_selector = memnew(OptionButton);
	image_quality_selector->add_item("Turbo (~20s)", 0);
	image_quality_selector->add_item("Standard (~2min)", 1);
	image_quality_selector->add_item("High (~5min)", 2);
	image_quality_selector->select(0);
	image_tab->add_child(image_quality_selector);
	
	generate_from_image_button = memnew(Button);
	generate_from_image_button->set_text("Generate 3D from Image");
	generate_from_image_button->set_disabled(true);
	generate_from_image_button->connect("pressed", callable_mp(this, &DesignStudio3DEditor::_on_generate_from_image_pressed));
	image_tab->add_child(generate_from_image_button);
	
	image_status_label = memnew(Label);
	image_status_label->set_text("Select an image to generate 3D model");
	image_status_label->set_autowrap_mode(TextServer::AUTOWRAP_WORD_SMART);
	image_status_label->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	image_tab->add_child(image_status_label);
	
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
}

void DesignStudio3DEditor::_on_generate_pressed() {
	if (is_generating) {
		status_label->set_text("[BUSY] Already generating...");
		return;
	}
	
	String prompt = prompt_input->get_text().strip_edges();
	if (prompt.is_empty()) {
		status_label->set_text("[ERROR] Please enter a prompt");
		return;
	}
	
	// Clear the 3D viewer - SIMPLE
	if (preview_mesh) {
		preview_mesh->set_mesh(Ref<Mesh>());
		preview_mesh->set_transform(Transform3D()); // Reset everything
	}
	
	// Get quality setting
	String quality = "turbo";
	switch (quality_selector->get_selected_id()) {
		case 0: quality = "turbo"; break;
		case 1: quality = "standard"; break;
		case 2: quality = "high"; break;
	}
	
	// Prepare JSON body
	Dictionary body_dict;
	body_dict["user_id"] = current_user_id;
	body_dict["prompt"] = prompt;
	body_dict["quality"] = quality;
	
	String json_body = JSON::stringify(body_dict);
	
	// Setup HTTP request
	PackedStringArray headers;
	headers.push_back("Content-Type: application/json");
	
	String url = API_URL + "/api/jobs/text-to-3d";
	
	// Use dedicated submit_request instance
	submit_request->connect("request_completed", callable_mp(this, &DesignStudio3DEditor::_on_job_submitted), CONNECT_ONE_SHOT);
	
	Error err = submit_request->request(url, headers, HTTPClient::METHOD_POST, json_body);
	
	if (err == OK) {
		is_generating = true;
		generate_button->set_disabled(true);
		generate_from_image_button->set_disabled(true); // Disable image generation too
		status_label->set_text("[SUBMITTING] Sending job to GPU server...");
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
		
		// Debug: Print the response to see what we got
		print_line("Job completed. Full response: " + JSON::stringify(job_data));
		
		// OPTIMIZATION: Use direct Supabase URL instead of proxy for faster downloads
		String model_url = "";
		if (job_data.has("output_file_url")) {
			Variant url_variant = job_data["output_file_url"];
			if (url_variant.get_type() == Variant::STRING) {
				model_url = url_variant;
				print_line("Using direct Supabase URL for faster download: " + model_url);
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
			print_line("Using proxy download URL: " + model_url);
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
		status_label->set_text("⏳ Status: " + status);
	}
}

void DesignStudio3DEditor::_on_model_downloaded(int p_result, int p_code, const PackedStringArray &p_headers, const PackedByteArray &p_body) {
	print_line("========================================");
	print_line("=== MODEL DOWNLOAD CALLBACK TRIGGERED ===");
	print_line("========================================");
	print_line("Result: " + itos(p_result) + " (0=SUCCESS)");
	print_line("HTTP Code: " + itos(p_code));
	print_line("Body size: " + itos(p_body.size()) + " bytes (" + String::humanize_size(p_body.size()) + ")");
	print_line("Headers count: " + itos(p_headers.size()));
	for (int i = 0; i < p_headers.size(); i++) {
		print_line("  Header[" + itos(i) + "]: " + p_headers[i]);
	}
	print_line("========================================");
	
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
		status_label->set_text("❌ Download failed: " + error_msg + "\nResult code: " + itos(p_result));
		is_generating = false;
		generate_button->set_disabled(false);
		return;
	}
	
	if (p_code != 200 && p_code != 302 && p_code != 307) {
		// Handle retry logic for 404 errors (might be timing issue)
		if (p_code == 404 && download_retry_count < 2) {
			print_line("Got 404, scheduling retry " + itos(download_retry_count + 1) + "/2 in 1 second...");
			status_label->set_text("[404 ERROR] Download failed, retrying in 1 second...\n(Attempt " + itos(download_retry_count + 1) + "/3)");
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
		print_line("Download successful after " + itos(download_retry_count) + " retries");
		_load_model_from_data(p_body);
	} else {
		status_label->set_text("[ERROR] Downloaded file is empty");
		is_generating = false;
		generate_button->set_disabled(false);
	}
	
	// Reset state - enable all generation buttons
	is_generating = false;
	generate_button->set_disabled(false);
	if (generate_from_image_button && !selected_image_path.is_empty()) {
		generate_from_image_button->set_disabled(false);
	}
	current_job_id = "";
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
	
	// Save the cached model data to workspace
	String timestamp = String::num_int64(OS::get_singleton()->get_ticks_msec());
	String filename = "exported_model_" + timestamp + ".obj";
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
		generate_from_image_button->set_disabled(false);
		image_status_label->set_text("Image loaded! Click Generate to create 3D model.");
	} else {
		image_status_label->set_text("[ERROR] Failed to load image");
		generate_from_image_button->set_disabled(true);
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

void DesignStudio3DEditor::_on_generate_from_image_pressed() {
	if (is_generating) {
		image_status_label->set_text("[BUSY] Already generating...");
		return;
	}
	
	if (selected_image_path.is_empty()) {
		image_status_label->set_text("[ERROR] No image selected");
		return;
	}
	
	// Convert image to base64
	String base64_image = _image_to_base64(selected_image_path);
	if (base64_image.is_empty()) {
		image_status_label->set_text("[ERROR] Failed to convert image");
		return;
	}
	
	// Get quality setting
	String quality = "turbo";
	switch (image_quality_selector->get_selected_id()) {
		case 0: quality = "turbo"; break;
		case 1: quality = "standard"; break;
		case 2: quality = "high"; break;
	}
	
	// Prepare JSON body for image-to-3D
	Dictionary body_dict;
	body_dict["user_id"] = current_user_id;
	body_dict["image"] = base64_image;
	body_dict["quality"] = quality;
	
	String json_body = JSON::stringify(body_dict);
	
	// Setup HTTP request
	PackedStringArray headers;
	headers.push_back("Content-Type: application/json");
	
	String url = API_URL + "/api/jobs/image-to-3d";
	
	// Use submit_request for consistency
	submit_request->connect("request_completed", callable_mp(this, &DesignStudio3DEditor::_on_job_submitted), CONNECT_ONE_SHOT);
	
	Error err = submit_request->request(url, headers, HTTPClient::METHOD_POST, json_body);
	
	if (err == OK) {
		is_generating = true;
		generate_from_image_button->set_disabled(true);
		generate_button->set_disabled(true); // Disable text generation too
		image_status_label->set_text("[SUBMITTING] Sending image to GPU server...");
		status_label->set_text("[SUBMITTING] Processing image-to-3D...");
	} else {
		image_status_label->set_text("[ERROR] Failed to start request");
	}
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