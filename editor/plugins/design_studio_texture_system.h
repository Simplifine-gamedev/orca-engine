/***********************************************************/
// © 2025 Simplifine Corp. Original backend contribution for this Godot fork.
// Personal Non‑Commercial License applies. Commercial use requires a separate license from Simplifine.
// See LICENSES/COMPANY-NONCOMMERCIAL.md.
/***********************************************************/

#ifndef DESIGN_STUDIO_TEXTURE_SYSTEM_H
#define DESIGN_STUDIO_TEXTURE_SYSTEM_H

#include "core/object/class_db.h"
#include "core/object/object.h"
#include "scene/main/http_request.h"
#include "scene/main/timer.h"
#include "scene/gui/dialogs.h"
#include "scene/gui/line_edit.h"
#include "scene/gui/label.h"
#include "scene/gui/option_button.h"

class DesignStudioTextureSystem : public Object {
    GDCLASS(DesignStudioTextureSystem, Object);

private:
    // API Configuration
    static const String TEXTURE_API_URL;
    
    // HTTP Request nodes
    HTTPRequest *texture_submit_request = nullptr;
    HTTPRequest *texture_poll_request = nullptr;
    HTTPRequest *texture_download_request = nullptr;
    
    // Polling timer
    Timer *texture_poll_timer = nullptr;
    
    // Current operation state
    String current_user_id;
    String current_base_model_id;
    String current_texture_job_id;
    bool is_texturing = false;
    
    // UI elements for texture prompt dialog
    AcceptDialog *texture_dialog = nullptr;
    LineEdit *texture_prompt_input = nullptr;
    OptionButton *hunyuan_version_selector = nullptr;
    OptionButton *resolution_selector = nullptr;
    Label *texture_dialog_status = nullptr;
    
    // Callbacks for parent
    Callable texture_started_callback;
    Callable texture_progress_callback;
    Callable texture_completed_callback;
    Callable texture_failed_callback;

protected:
    static void _bind_methods();

public:
    // Initialization
    void initialize_texture_system(Node *p_parent, const String &p_user_id);
    void cleanup_texture_system();
    
    // Main texture generation workflow
    void show_texture_generation_dialog(const String &p_base_model_id);
    void start_texture_generation(const String &p_base_model_id, const String &p_prompt, const String &p_version = "2.0", int p_resolution = 512);
    void cancel_texture_generation();
    
    // Status and polling
    bool is_texture_generation_active() const { return is_texturing; }
    String get_current_texture_job_id() const { return current_texture_job_id; }
    
    // Callback setters
    void set_texture_started_callback(const Callable &p_callback) { texture_started_callback = p_callback; }
    void set_texture_progress_callback(const Callable &p_callback) { texture_progress_callback = p_callback; }
    void set_texture_completed_callback(const Callable &p_callback) { texture_completed_callback = p_callback; }
    void set_texture_failed_callback(const Callable &p_callback) { texture_failed_callback = p_callback; }

private:
    // Dialog management
    void _setup_texture_dialog(Node *p_parent);
    void _on_texture_dialog_confirmed();
    void _on_texture_dialog_cancelled();
    
    // HTTP request handlers
    void _on_texture_job_submitted(int p_result, int p_code, const PackedStringArray &p_headers, const PackedByteArray &p_body);
    void _on_texture_status_received(int p_result, int p_code, const PackedStringArray &p_headers, const PackedByteArray &p_body);
    void _on_textured_model_downloaded(int p_result, int p_code, const PackedStringArray &p_headers, const PackedByteArray &p_body);
    
    // Polling management
    void _start_texture_polling();
    void _stop_texture_polling();
    void _on_texture_poll_timeout();
    
    // Utility functions
    void _notify_texture_started(const String &p_job_id);
    void _notify_texture_progress(const String &p_status, const Dictionary &p_data = Dictionary());
    void _notify_texture_completed(const PackedByteArray &p_model_data, const String &p_filename);
    void _notify_texture_failed(const String &p_error_message);
    void _reset_texture_state();
    
    // Error handling
    String _get_http_error_message(int p_result) const;

public:
    DesignStudioTextureSystem();
    ~DesignStudioTextureSystem();
};

#endif // DESIGN_STUDIO_TEXTURE_SYSTEM_H

