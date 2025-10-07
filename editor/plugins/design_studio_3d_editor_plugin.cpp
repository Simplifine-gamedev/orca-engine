/**************************************************************************/
/*  design_studio_3d_editor_plugin.cpp                                    */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "design_studio_3d_editor_plugin.h"

#include "core/config/project_settings.h"
#include "core/io/json.h"
#include "core/io/resource_loader.h"
#include "editor/editor_main_screen.h"
#include "editor/editor_node.h"
#include "editor/file_system/editor_file_system.h"
#include "editor/themes/editor_scale.h"
#include "scene/3d/camera_3d.h"
#include "scene/3d/light_3d.h"
#include "scene/3d/mesh_instance_3d.h"
#include "scene/resources/environment.h"
#include "scene/resources/3d/world_3d.h"
#include "scene/resources/material.h"
#include "scene/resources/camera_attributes.h"
#include "scene/gui/box_container.h"
#include "scene/gui/button.h"
#include "scene/gui/label.h"
#include "scene/gui/line_edit.h"
#include "scene/gui/option_button.h"
#include "scene/gui/separator.h"
#include "scene/gui/split_container.h"
#include "scene/gui/subviewport_container.h"
#include "scene/main/timer.h"
#include "scene/resources/3d/primitive_meshes.h"
#include "scene/resources/packed_scene.h"

void DesignStudio3DEditor::_bind_methods() {
}

void DesignStudio3DEditor::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_READY: {
			_setup_ui();
			_setup_3d_viewer();
		} break;
		case NOTIFICATION_THEME_CHANGED: {
			if (generate_button) {
				generate_button->set_button_icon(get_editor_theme_icon(SNAME("Add")));
			}
		} break;
	}
}

void DesignStudio3DEditor::_setup_ui() {
	set_custom_minimum_size(Size2(0, 200) * EDSCALE);
	
	// Main horizontal split
	HSplitContainer *hsplit = memnew(HSplitContainer);
	hsplit->set_anchors_and_offsets_preset(Control::PRESET_FULL_RECT);
	add_child(hsplit);
	
	// Left panel for controls
	VBoxContainer *left_panel = memnew(VBoxContainer);
	left_panel->set_custom_minimum_size(Size2(300 * EDSCALE, 0));
	hsplit->add_child(left_panel);
	
	// Title
	Label *title = memnew(Label);
	title->set_text("3D Model Generator");
	title->add_theme_font_size_override("font_size", 18 * EDSCALE);
	left_panel->add_child(title);
	
	left_panel->add_child(memnew(HSeparator));
	
	// Prompt input
	Label *prompt_label = memnew(Label);
	prompt_label->set_text("Prompt:");
	left_panel->add_child(prompt_label);
	
	prompt_input = memnew(LineEdit);
	prompt_input->set_placeholder("Describe your 3D model (e.g., 'A cute robot')");
	prompt_input->set_custom_minimum_size(Size2(0, 32 * EDSCALE));
	left_panel->add_child(prompt_input);
	
	// Quality selector
	Label *quality_label = memnew(Label);
	quality_label->set_text("Quality:");
	left_panel->add_child(quality_label);
	
	quality_selector = memnew(OptionButton);
	quality_selector->add_item("Turbo (~20s)", 0);
	quality_selector->add_item("Standard (~110s)", 1);
	quality_selector->add_item("High (~5min)", 2);
	quality_selector->select(0);
	left_panel->add_child(quality_selector);
	
	// Generate button
	generate_button = memnew(Button);
	generate_button->set_text("Generate 3D Model");
	generate_button->set_custom_minimum_size(Size2(0, 40 * EDSCALE));
	generate_button->connect("pressed", callable_mp(this, &DesignStudio3DEditor::_on_generate_pressed));
	left_panel->add_child(generate_button);
	
	// Status label
	status_label = memnew(Label);
	status_label->set_text("Ready");
	status_label->set_autowrap_mode(TextServer::AUTOWRAP_WORD_SMART);
	left_panel->add_child(status_label);
	
	// Spacer
	Control *spacer = memnew(Control);
	spacer->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	left_panel->add_child(spacer);
	
	// Info label
	Label *info_label = memnew(Label);
	info_label->set_text("Generated models appear in the isolated 3D viewer on the right.\n\n3D VIEWER CONTROLS:\n- Left click + drag: Rotate\n- Mouse wheel: Zoom in/out\n\nViewer starts empty until you generate a model.\n\nAPI: " + API_URL);
	info_label->set_autowrap_mode(TextServer::AUTOWRAP_WORD_SMART);
	info_label->add_theme_font_size_override("font_size", 10 * EDSCALE);
	left_panel->add_child(info_label);
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
	print_line("Creating properly isolated 3D viewer...");
	
	viewport = memnew(SubViewport);
	viewport->set_update_mode(SubViewport::UPDATE_ALWAYS);
	
	// CRITICAL: Create NEW World3D to ensure complete isolation
	Ref<World3D> new_world = memnew(World3D);
	viewport->set_world_3d(new_world);
	print_line("Created isolated World3D: " + itos(new_world->get_scenario().get_id()));
	
	viewport_container->add_child(viewport);
	
	// Simple camera
	camera = memnew(Camera3D);
	camera->set_position(Vector3(0, 0, 3));
	camera->set_fov(45);
	camera->make_current(); // Make it the active camera in this viewport
	viewport->add_child(camera);
	
	// Two lights like MeshEditor
	light = memnew(DirectionalLight3D);
	light->set_transform(Transform3D().looking_at(Vector3(-1, -1, -1), Vector3(0, 1, 0)));
	viewport->add_child(light);
	
	DirectionalLight3D *light2 = memnew(DirectionalLight3D);
	light2->set_transform(Transform3D().looking_at(Vector3(0, 1, 0), Vector3(0, 0, 1)));
	light2->set_color(Color(0.7, 0.7, 0.7));
	viewport->add_child(light2);
	
	// Simple mesh instance
	preview_mesh = memnew(MeshInstance3D);
	preview_mesh->set_name("IsolatedModelViewer");
	viewport->add_child(preview_mesh);
	
	print_line("Isolated 3D viewer created with new World3D");
	
	// OVERRIDE gui_input on THIS control for proper mouse handling
	set_process_input(true);
	viewport_container->set_focus_mode(Control::FOCUS_ALL);
	viewport_container->connect("gui_input", callable_mp(this, &DesignStudio3DEditor::_on_viewport_input));
	viewport_container->set_mouse_filter(Control::MOUSE_FILTER_STOP); // STOP events from passing through
	
	print_line("Input handling configured");
	
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
	
	print_line("3D Design Studio: HTTP request nodes initialized");
	
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
		print_line("Cleared simple 3D viewer");
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
	
	// Reset state
	is_generating = false;
	generate_button->set_disabled(false);
	current_job_id = "";
	download_retry_count = 0;
}

void DesignStudio3DEditor::_load_model_from_data(const PackedByteArray &p_data) {
	print_line("Loading model from data, size: " + itos(p_data.size()) + " bytes");
	
	// Save to project root with timestamp
	String timestamp = String::num_int64(OS::get_singleton()->get_ticks_msec());
	String filename = "generated_model_" + timestamp + ".obj";
	String save_path = "res://" + filename;
	
	// For saving during runtime, use absolute path
	String project_path = ProjectSettings::get_singleton()->globalize_path(save_path);
	
	print_line("Attempting to save to: " + project_path);
	
	Ref<FileAccess> file = FileAccess::open(project_path, FileAccess::WRITE);
	if (file.is_valid()) {
		file->store_buffer(p_data);
		file->close();
		
		print_line("✅ File saved successfully!");
		
		// Show preview of first few lines
		String preview = "[SUCCESS] Model saved successfully!\n\n";
		preview += "Location: " + filename + "\n";
		preview += "Size: " + String::humanize_size(p_data.size()) + "\n\n";
		preview += "To use this model:\n";
		preview += "1. The file is in your project root\n";
		preview += "2. Drag it into your scene\n";
		preview += "3. Godot will auto-import it\n\n";
		
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
		
		status_label->set_text(preview);
		
		// Try to load the imported mesh after a delay
		print_line("Scheduling 3D viewer update after import...");
		
		// Use a timer to wait for import to complete, then try loading
		Timer *import_wait_timer = memnew(Timer);
		import_wait_timer->set_wait_time(2.0); // Wait 2 seconds for import
		import_wait_timer->set_one_shot(true);
		add_child(import_wait_timer);
		
		// Connect lambda to load the imported mesh
		import_wait_timer->connect("timeout", callable_mp(this, &DesignStudio3DEditor::_load_imported_mesh).bind(save_path));
		import_wait_timer->start();
		
		preview += "\n[3D VIEWER] Waiting for import, then loading...";
		status_label->set_text(preview);
		
		// FORCE IMMEDIATE REIMPORT to fix the "cross" issue
		print_line("Forcing immediate reimport of generated file...");
		
		Vector<String> reimport_files;
		reimport_files.push_back(save_path);
		
		// Force reimport immediately 
		EditorFileSystem::get_singleton()->reimport_files(reimport_files);
		
		// Also trigger general scan for good measure
		EditorFileSystem::get_singleton()->scan_changes();
		
		print_line("Forced reimport completed. File should appear correctly in FileSystem dock.");
		
	} else {
		status_label->set_text("[ERROR] Failed to save model file to: " + project_path + "\n\nCheck write permissions.");
		print_line("ERROR: Could not open file for writing: " + project_path);
	}
}

Ref<ArrayMesh> DesignStudio3DEditor::_parse_obj_to_mesh(const String &p_obj_content) {
	print_line("SIMPLE OBJ parsing...");
	
	PackedVector3Array vertices;
	PackedInt32Array indices;
	
	PackedStringArray lines = p_obj_content.split("\n");
	
	// SIMPLE parsing - just vertices and triangular faces
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
		} else if (line.begins_with("f ")) {
			PackedStringArray parts = line.split(" ");
			if (parts.size() >= 4) {
				// Only handle triangular faces for simplicity
				for (int j = 1; j <= 3 && j < parts.size(); j++) {
					String vertex_def = parts[j];
					int vertex_index = vertex_def.split("/")[0].to_int() - 1; // OBJ is 1-based
					if (vertex_index >= 0 && vertex_index < vertices.size()) {
						indices.push_back(vertex_index);
					}
				}
			}
		}
	}
	
	print_line("SIMPLE parsed: " + itos(vertices.size()) + " vertices, " + itos(indices.size()) + " indices");
	
	if (vertices.size() == 0 || indices.size() == 0) {
		return Ref<ArrayMesh>();
	}
	
	// Create simplest possible mesh
	Ref<ArrayMesh> mesh = memnew(ArrayMesh);
	Array arrays;
	arrays.resize(Mesh::ARRAY_MAX);
	arrays[Mesh::ARRAY_VERTEX] = vertices;
	arrays[Mesh::ARRAY_INDEX] = indices;
	
	mesh->add_surface_from_arrays(Mesh::PRIMITIVE_TRIANGLES, arrays);
	print_line("SIMPLE ArrayMesh created");
	
	return mesh;
}

void DesignStudio3DEditor::_setup_camera_orbit() {
	print_line("Setting up SIMPLE camera for model");
	
	if (!preview_mesh || !camera) {
		print_line("preview_mesh or camera is null!");
		return;
	}
	
	Ref<Mesh> mesh = preview_mesh->get_mesh();
	if (!mesh.is_valid()) {
		print_line("Mesh is not valid!");
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
		
		print_line("Scaled model by " + String::num(scale) + ", centered at " + (-center * scale).operator String());
	}
	
	// Position camera to see the 1x1x1 scaled model
	camera->set_position(Vector3(1.5, 1.0, 2.0));
	camera->look_at(Vector3(0, 0, 0), Vector3(0, 1, 0));
	
	print_line("Camera positioned to view scaled model");
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
			print_line("Left mouse button: " + String(is_rotating ? "PRESSED" : "RELEASED"));
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
					print_line("Zoom in - distance from center: " + String::num(new_pos.length()));
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
					print_line("Zoom out - distance from center: " + String::num(new_pos.length()));
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
	print_line("Starting download with proper headers: " + p_url);
	
	// Connect callback
	download_request->connect("request_completed", callable_mp(this, &DesignStudio3DEditor::_on_model_downloaded), CONNECT_ONE_SHOT);
	
	// Add headers to match curl behavior
	PackedStringArray headers;
	headers.push_back("User-Agent: Godot-Editor/4.0"); // Identify as Godot
	headers.push_back("Accept: */*"); // Accept any content type
	headers.push_back("Accept-Encoding: identity"); // Don't request compression to avoid issues
	
	print_line("Headers: " + String(" | ").join(headers));
	
	Error download_err = download_request->request(p_url, headers, HTTPClient::METHOD_GET, "");
	print_line("Download request result: " + itos(download_err) + " (0=OK)");
	
	if (download_err != OK) {
		status_label->set_text("[ERROR] Failed to start download request. Error code: " + itos(download_err));
		is_generating = false;
		generate_button->set_disabled(false);
	}
}

void DesignStudio3DEditor::_on_download_retry_timeout() {
	print_line("Download retry timeout - attempting retry " + itos(download_retry_count + 1) + "/2");
	
	download_retry_count++;
	status_label->set_text("[RETRYING] Download attempt " + itos(download_retry_count + 1) + "/3");
	
	_start_download_with_headers(download_url_to_retry);
}

void DesignStudio3DEditor::_load_imported_mesh(const String &p_path) {
	print_line("======================================");
	print_line("Attempting to load imported mesh from: " + p_path);
	
	if (!preview_mesh) {
		print_line("ERROR: preview_mesh is NULL!");
		return;
	}
	
	print_line("preview_mesh parent: " + (preview_mesh->get_parent() ? preview_mesh->get_parent()->get_name() : "NULL"));
	print_line("preview_mesh is in viewport: " + String(preview_mesh->is_inside_tree() ? "YES" : "NO"));
	
	// Try loading the resource - Godot should have imported it by now
	Ref<Resource> resource = ResourceLoader::load(p_path);
	
	if (resource.is_valid()) {
		print_line("Resource loaded successfully, type: " + resource->get_class());
		
		// Check if it's a mesh directly
		Ref<Mesh> mesh = resource;
		if (mesh.is_valid()) {
			print_line("Got Mesh resource directly - setting it on isolated preview_mesh");
			print_line("Mesh has " + itos(mesh->get_surface_count()) + " surfaces");
			
			preview_mesh->set_mesh(mesh);
			print_line("Mesh set successfully");
			
			_setup_camera_orbit();
			status_label->set_text(status_label->get_text() + "\n[3D VIEWER] Imported mesh loaded!");
			print_line("======================================");
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
					preview_mesh->set_mesh(scene_mesh);
					_setup_camera_orbit();
					status_label->set_text(status_label->get_text() + "\n[3D VIEWER] Scene mesh loaded!");
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

DesignStudio3DEditor::DesignStudio3DEditor() {
	set_name("DesignStudio3D");
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