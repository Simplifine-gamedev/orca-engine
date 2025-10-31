/***********************************************************/
// © 2025 Simplifine Corp. Original backend contribution for this Godot fork.
// Personal Non‑Commercial License applies. Commercial use requires a separate license from Simplifine.
// See LICENSES/COMPANY-NONCOMMERCIAL.md.
/***********************************************************/

#include "design_studio_texture_system.h"

#include "core/io/json.h"
#include "core/os/os.h"
#include "editor/themes/editor_scale.h"
#include "scene/gui/box_container.h"
#include "scene/gui/button.h"
#include "scene/gui/separator.h"

const String DesignStudioTextureSystem::TEXTURE_API_URL = "https://gpu-proxy-976792908107.us-central1.run.app";

void DesignStudioTextureSystem::_bind_methods() {
    // Bind methods for GDScript access if needed
    ClassDB::bind_method(D_METHOD("is_texture_generation_active"), &DesignStudioTextureSystem::is_texture_generation_active);
    ClassDB::bind_method(D_METHOD("get_current_texture_job_id"), &DesignStudioTextureSystem::get_current_texture_job_id);
    ClassDB::bind_method(D_METHOD("cancel_texture_generation"), &DesignStudioTextureSystem::cancel_texture_generation);
}

DesignStudioTextureSystem::DesignStudioTextureSystem() {
    is_texturing = false;
}

DesignStudioTextureSystem::~DesignStudioTextureSystem() {
    cleanup_texture_system();
}

void DesignStudioTextureSystem::initialize_texture_system(Node *p_parent, const String &p_user_id) {
    current_user_id = p_user_id;
    
    // Create HTTP request nodes
    texture_submit_request = memnew(HTTPRequest);
    texture_submit_request->set_name("TextureSubmitRequest");
    texture_submit_request->set_timeout(480); // 8 minutes for texture submission
    p_parent->add_child(texture_submit_request);
    
    texture_poll_request = memnew(HTTPRequest);
    texture_poll_request->set_name("TexturePollRequest");
    texture_poll_request->set_timeout(60); // 1 minute for polling requests
    p_parent->add_child(texture_poll_request);
    
    texture_download_request = memnew(HTTPRequest);
    texture_download_request->set_name("TextureDownloadRequest");
    texture_download_request->set_timeout(480); // 8 minutes timeout for large files
    texture_download_request->set_body_size_limit(200 * 1024 * 1024); // 200 MB limit
    texture_download_request->set_use_threads(true);
    p_parent->add_child(texture_download_request);
    
    // Create polling timer
    texture_poll_timer = memnew(Timer);
    texture_poll_timer->set_wait_time(5.0); // Poll every 5 seconds
    texture_poll_timer->set_one_shot(false);
    texture_poll_timer->connect("timeout", callable_mp(this, &DesignStudioTextureSystem::_on_texture_poll_timeout));
    p_parent->add_child(texture_poll_timer);
    
    // Setup texture dialog
    _setup_texture_dialog(p_parent);
    
    print_line("Texture system initialized for user: " + current_user_id);
}

void DesignStudioTextureSystem::cleanup_texture_system() {
    // Cancel any active operations
    cancel_texture_generation();
    
    // Clean up HTTP requests
    if (texture_submit_request) {
        texture_submit_request->queue_free();
        texture_submit_request = nullptr;
    }
    
    if (texture_poll_request) {
        texture_poll_request->queue_free();
        texture_poll_request = nullptr;
    }
    
    if (texture_download_request) {
        texture_download_request->queue_free();
        texture_download_request = nullptr;
    }
    
    // Clean up timer
    if (texture_poll_timer) {
        texture_poll_timer->queue_free();
        texture_poll_timer = nullptr;
    }
    
    // Clean up dialog
    if (texture_dialog) {
        texture_dialog->queue_free();
        texture_dialog = nullptr;
    }
}

void DesignStudioTextureSystem::_setup_texture_dialog(Node *p_parent) {
    texture_dialog = memnew(AcceptDialog);
    texture_dialog->set_title("Generate AI Texture");
    texture_dialog->set_ok_button_text("Generate Texture");
    texture_dialog->connect("confirmed", callable_mp(this, &DesignStudioTextureSystem::_on_texture_dialog_confirmed));
    texture_dialog->connect("canceled", callable_mp(this, &DesignStudioTextureSystem::_on_texture_dialog_cancelled));
    p_parent->add_child(texture_dialog);
    
    VBoxContainer *dialog_vbox = memnew(VBoxContainer);
    texture_dialog->add_child(dialog_vbox);
    
    // Texture prompt input
    Label *prompt_label = memnew(Label);
    prompt_label->set_text("Texture Description:");
    dialog_vbox->add_child(prompt_label);
    
    texture_prompt_input = memnew(LineEdit);
    texture_prompt_input->set_placeholder("e.g. Bronze armor with rust and battle damage");
    texture_prompt_input->set_custom_minimum_size(Size2(400 * EDSCALE, 0));
    dialog_vbox->add_child(texture_prompt_input);
    
    dialog_vbox->add_child(memnew(HSeparator));
    
    // Hunyuan version selector
    Label *version_label = memnew(Label);
    version_label->set_text("AI Model Version:");
    dialog_vbox->add_child(version_label);
    
    hunyuan_version_selector = memnew(OptionButton);
    hunyuan_version_selector->add_item("Hunyuan3D 2.0 (Faster)", 0);
    hunyuan_version_selector->add_item("Hunyuan3D 2.1 (Higher Quality)", 1);
    hunyuan_version_selector->select(0); // Default to 2.0
    dialog_vbox->add_child(hunyuan_version_selector);
    
    // Resolution selector
    Label *resolution_label = memnew(Label);
    resolution_label->set_text("Texture Resolution:");
    dialog_vbox->add_child(resolution_label);
    
    resolution_selector = memnew(OptionButton);
    resolution_selector->add_item("512x512 (Fast)", 512);
    resolution_selector->add_item("1024x1024 (High Quality)", 1024);
    resolution_selector->add_item("2048x2048 (Ultra)", 2048);
    resolution_selector->select(0); // Default to 512
    dialog_vbox->add_child(resolution_selector);
    
    dialog_vbox->add_child(memnew(HSeparator));
    
    // Status label
    texture_dialog_status = memnew(Label);
    texture_dialog_status->set_text("Enter a texture description and click Generate Texture");
    texture_dialog_status->set_autowrap_mode(TextServer::AUTOWRAP_WORD_SMART);
    texture_dialog_status->set_custom_minimum_size(Size2(0, 40 * EDSCALE));
    dialog_vbox->add_child(texture_dialog_status);
}

void DesignStudioTextureSystem::show_texture_generation_dialog(const String &p_base_model_id) {
    if (is_texturing) {
        if (texture_dialog_status) {
            texture_dialog_status->set_text("[ERROR] Texture generation already in progress");
        }
        return;
    }
    
    current_base_model_id = p_base_model_id;
    
    if (texture_dialog) {
        // Reset dialog state
        if (texture_prompt_input) {
            texture_prompt_input->set_text("");
        }
        if (hunyuan_version_selector) {
            hunyuan_version_selector->select(0);
        }
        if (resolution_selector) {
            resolution_selector->select(0);
        }
        if (texture_dialog_status) {
            texture_dialog_status->set_text("Enter a texture description and click Generate Texture");
        }
        
        texture_dialog->popup_centered(Size2(500 * EDSCALE, 0));
        
        // Focus on prompt input
        if (texture_prompt_input) {
            texture_prompt_input->grab_focus();
        }
    }
}

void DesignStudioTextureSystem::_on_texture_dialog_confirmed() {
    if (!texture_prompt_input || !hunyuan_version_selector || !resolution_selector) {
        return;
    }
    
    String prompt = texture_prompt_input->get_text().strip_edges();
    if (prompt.is_empty()) {
        if (texture_dialog_status) {
            texture_dialog_status->set_text("[ERROR] Please enter a texture description");
        }
        return;
    }
    
	// Get selected options
	String version = hunyuan_version_selector->get_selected_id() == 0 ? "2.0" : "2.1";
	int resolution = 512; // Default resolution
	int selected_idx = resolution_selector->get_selected();
	if (selected_idx == 0) resolution = 512;
	else if (selected_idx == 1) resolution = 1024;
	else if (selected_idx == 2) resolution = 2048;
    
    // Close dialog and start generation
    texture_dialog->hide();
    start_texture_generation(current_base_model_id, prompt, version, resolution);
}

void DesignStudioTextureSystem::_on_texture_dialog_cancelled() {
    current_base_model_id = "";
}

void DesignStudioTextureSystem::start_texture_generation(const String &p_base_model_id, const String &p_prompt, const String &p_version, int p_resolution) {
    if (is_texturing) {
        _notify_texture_failed("Texture generation already in progress");
        return;
    }
    
    if (p_base_model_id.is_empty()) {
        _notify_texture_failed("No base model ID provided");
        return;
    }
    
    if (p_prompt.is_empty()) {
        _notify_texture_failed("No texture prompt provided");
        return;
    }
    
    current_base_model_id = p_base_model_id;
    
    // Create request body according to API documentation
    Dictionary body_dict;
    body_dict["user_id"] = current_user_id;
    body_dict["job_id"] = p_base_model_id; // Base model ID
    body_dict["prompt"] = p_prompt;
    body_dict["hunyuan_version"] = p_version;
    body_dict["resolution"] = p_resolution;
    body_dict["max_views"] = 6; // Default value from API docs
    
    String json_body = JSON::stringify(body_dict);
    String url = TEXTURE_API_URL + "/api/jobs/texture-generation";
    
    // Debug output
    print_line("=== Texture Generation Request ===");
    print_line("URL: " + url);
    print_line("Body: " + json_body);
    print_line("==================================");
    
    PackedStringArray headers;
    headers.push_back("Content-Type: application/json");
    headers.push_back("User-Agent: Godot-Editor/4.0");
    
    // Disconnect any existing connections
    if (texture_submit_request->is_connected("request_completed", callable_mp(this, &DesignStudioTextureSystem::_on_texture_job_submitted))) {
        texture_submit_request->disconnect("request_completed", callable_mp(this, &DesignStudioTextureSystem::_on_texture_job_submitted));
    }
    
    texture_submit_request->connect("request_completed", callable_mp(this, &DesignStudioTextureSystem::_on_texture_job_submitted), CONNECT_ONE_SHOT);
    
    Error err = texture_submit_request->request(url, headers, HTTPClient::METHOD_POST, json_body);
    
    if (err == OK) {
        is_texturing = true;
        _notify_texture_started("");
        _notify_texture_progress("Submitting texture generation job...");
    } else {
        _notify_texture_failed("Failed to start texture request. Error: " + itos(err));
    }
}

void DesignStudioTextureSystem::_on_texture_job_submitted(int p_result, int p_code, const PackedStringArray &p_headers, const PackedByteArray &p_body) {
    if (p_result != HTTPRequest::RESULT_SUCCESS || p_code != 200) {
        _notify_texture_failed("Failed to submit texture job (HTTP " + itos(p_code) + ")");
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
        _notify_texture_failed("Failed to parse texture job response");
        return;
    }
    
    Dictionary response = json.get_data();
    
    if (response.has("texture_record_id")) {
        current_texture_job_id = response["texture_record_id"];
        String job_id_short = current_texture_job_id.substr(0, 8) + "...";
        _notify_texture_progress("Texture job submitted! ID: " + job_id_short + "\nChecking status...");
        _start_texture_polling();
    } else {
        _notify_texture_failed("No texture job ID in response");
    }
}

void DesignStudioTextureSystem::_start_texture_polling() {
    if (texture_poll_timer) {
        texture_poll_timer->start();
        // Poll immediately
        _on_texture_poll_timeout();
    }
}

void DesignStudioTextureSystem::_stop_texture_polling() {
    if (texture_poll_timer) {
        texture_poll_timer->stop();
    }
}

void DesignStudioTextureSystem::_on_texture_poll_timeout() {
    if (current_texture_job_id.is_empty()) {
        _stop_texture_polling();
        return;
    }
    
    String url = TEXTURE_API_URL + "/api/texture-jobs/" + current_texture_job_id;
    
    // Disconnect any existing connections
    if (texture_poll_request->is_connected("request_completed", callable_mp(this, &DesignStudioTextureSystem::_on_texture_status_received))) {
        texture_poll_request->disconnect("request_completed", callable_mp(this, &DesignStudioTextureSystem::_on_texture_status_received));
    }
    
    texture_poll_request->connect("request_completed", callable_mp(this, &DesignStudioTextureSystem::_on_texture_status_received), CONNECT_ONE_SHOT);
    texture_poll_request->request(url);
}

void DesignStudioTextureSystem::_on_texture_status_received(int p_result, int p_code, const PackedStringArray &p_headers, const PackedByteArray &p_body) {
    if (p_result != HTTPRequest::RESULT_SUCCESS || p_code != 200) {
        _notify_texture_failed("Failed to get texture job status (HTTP " + itos(p_code) + ")");
        _stop_texture_polling();
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
        _notify_texture_failed("Failed to parse texture status response");
        _stop_texture_polling();
        return;
    }
    
    Dictionary job_data = json.get_data();
    
    if (!job_data.has("status")) {
        _notify_texture_failed("Invalid texture status response");
        _stop_texture_polling();
        return;
    }
    
    String status = job_data["status"];
    
    if (status == "queued") {
        _notify_texture_progress("Texture job queued... waiting for GPU processing");
    } else if (status == "processing") {
        String progress_info = "Generating AI texture... (this may take 60-90 seconds)";
        if (job_data.has("generation_time")) {
            float elapsed = job_data["generation_time"];
            progress_info += "\nElapsed: " + String::num(elapsed, 1) + "s";
        }
        _notify_texture_progress(progress_info);
    } else if (status == "completed") {
        _stop_texture_polling();
        
        String texture_url = "";
        if (job_data.has("textured_mesh_url")) {
            texture_url = job_data["textured_mesh_url"];
        }
        
        if (texture_url.is_empty()) {
            // Use download endpoint instead
            texture_url = TEXTURE_API_URL + "/api/texture-jobs/" + current_texture_job_id + "/download?user_id=" + current_user_id;
        }
        
        _notify_texture_progress("Texture completed! Downloading textured model...");
        
        // Start download
        // Disconnect any existing connections
        if (texture_download_request->is_connected("request_completed", callable_mp(this, &DesignStudioTextureSystem::_on_textured_model_downloaded))) {
            texture_download_request->disconnect("request_completed", callable_mp(this, &DesignStudioTextureSystem::_on_textured_model_downloaded));
        }
        
        texture_download_request->connect("request_completed", callable_mp(this, &DesignStudioTextureSystem::_on_textured_model_downloaded), CONNECT_ONE_SHOT);
        
        PackedStringArray download_headers;
        download_headers.push_back("User-Agent: Godot-Editor/4.0");
        download_headers.push_back("Accept: */*");
        
        texture_download_request->request(texture_url, download_headers);
        
    } else if (status == "failed") {
        _stop_texture_polling();
        String error_msg = job_data.get("error_message", "Unknown error");
        _notify_texture_failed("Texture generation failed: " + error_msg);
    } else {
        _notify_texture_progress("Texture Status: " + status);
    }
}

void DesignStudioTextureSystem::_on_textured_model_downloaded(int p_result, int p_code, const PackedStringArray &p_headers, const PackedByteArray &p_body) {
    if (p_result != HTTPRequest::RESULT_SUCCESS) {
        String error_msg = _get_http_error_message(p_result);
        _notify_texture_failed("Download failed: " + error_msg);
        return;
    }
    
    if (p_code != 200) {
        _notify_texture_failed("Failed to download textured model (HTTP " + itos(p_code) + ")");
        return;
    }
    
    if (p_body.size() == 0) {
        _notify_texture_failed("Downloaded textured model is empty");
        return;
    }
    
    // Extract filename from headers or create default
    String filename = "textured_model_" + String::num_int64(OS::get_singleton()->get_ticks_msec()) + ".glb";
    
    for (int i = 0; i < p_headers.size(); i++) {
        String header = p_headers[i];
        if (header.begins_with("Content-Disposition:") || header.begins_with("content-disposition:")) {
            int filename_pos = header.find("filename=");
            if (filename_pos != -1) {
                String extracted_filename = header.substr(filename_pos + 9).strip_edges();
                extracted_filename = extracted_filename.trim_prefix("\"").trim_suffix("\"");
                if (!extracted_filename.is_empty()) {
                    filename = extracted_filename;
                }
            }
        }
    }
    
    _notify_texture_completed(p_body, filename);
}

void DesignStudioTextureSystem::cancel_texture_generation() {
    if (!is_texturing) {
        return;
    }
    
    _stop_texture_polling();
    
    // Cancel HTTP requests
    if (texture_submit_request) {
        texture_submit_request->cancel_request();
    }
    
    if (texture_poll_request) {
        texture_poll_request->cancel_request();
    }
    
    if (texture_download_request) {
        texture_download_request->cancel_request();
    }
    
    _notify_texture_failed("Texture generation cancelled by user");
}

void DesignStudioTextureSystem::_notify_texture_started(const String &p_job_id) {
    if (texture_started_callback.is_valid()) {
        texture_started_callback.call(p_job_id);
    }
}

void DesignStudioTextureSystem::_notify_texture_progress(const String &p_status, const Dictionary &p_data) {
    if (texture_progress_callback.is_valid()) {
        texture_progress_callback.call(p_status, p_data);
    }
}

void DesignStudioTextureSystem::_notify_texture_completed(const PackedByteArray &p_model_data, const String &p_filename) {
    _reset_texture_state();
    
    if (texture_completed_callback.is_valid()) {
        texture_completed_callback.call(p_model_data, p_filename);
    }
}

void DesignStudioTextureSystem::_notify_texture_failed(const String &p_error_message) {
    _reset_texture_state();
    
    if (texture_failed_callback.is_valid()) {
        texture_failed_callback.call(p_error_message);
    }
}

void DesignStudioTextureSystem::_reset_texture_state() {
    is_texturing = false;
    current_texture_job_id = "";
    current_base_model_id = "";
}

String DesignStudioTextureSystem::_get_http_error_message(int p_result) const {
    switch (p_result) {
        case HTTPRequest::RESULT_CHUNKED_BODY_SIZE_MISMATCH: return "Chunked body size mismatch";
        case HTTPRequest::RESULT_CANT_CONNECT: return "Can't connect";
        case HTTPRequest::RESULT_CANT_RESOLVE: return "Can't resolve hostname";
        case HTTPRequest::RESULT_CONNECTION_ERROR: return "Connection error";
        case HTTPRequest::RESULT_TLS_HANDSHAKE_ERROR: return "TLS handshake error";
        case HTTPRequest::RESULT_NO_RESPONSE: return "No response";
        case HTTPRequest::RESULT_BODY_SIZE_LIMIT_EXCEEDED: return "Body size limit exceeded";
        case HTTPRequest::RESULT_REQUEST_FAILED: return "Request failed";
        case HTTPRequest::RESULT_DOWNLOAD_FILE_CANT_OPEN: return "Can't open download file";
        case HTTPRequest::RESULT_DOWNLOAD_FILE_WRITE_ERROR: return "Download file write error";
        case HTTPRequest::RESULT_REDIRECT_LIMIT_REACHED: return "Redirect limit reached";
        case HTTPRequest::RESULT_TIMEOUT: return "Timeout";
        default: return "Unknown error (Code: " + itos(p_result) + ")";
    }
}
