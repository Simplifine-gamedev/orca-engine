/**************************************************************************/
/*  update_notification_popup.cpp                                        */
/**************************************************************************/

#include "update_notification_popup.h"

#include "core/io/json.h"
#include "core/os/os.h"
#include "core/orca_version.h"  // Orca version embedded at build time
#include "editor/editor_node.h"
#include "editor/file_system/editor_paths.h"
#include "editor/themes/editor_scale.h"
#include "editor/update/editor_updater.h"
#include "scene/gui/margin_container.h"
#include "scene/resources/style_box_flat.h"

void UpdateNotificationPopup::_bind_methods() {
    ClassDB::bind_method(D_METHOD("show_update_notification", "version", "download_url"), &UpdateNotificationPopup::show_update_notification, DEFVAL(""));
    ClassDB::bind_method(D_METHOD("hide_notification"), &UpdateNotificationPopup::hide_notification);
    ClassDB::bind_method(D_METHOD("set_auto_check_enabled", "enabled"), &UpdateNotificationPopup::set_auto_check_enabled);
    ClassDB::bind_method(D_METHOD("set_check_interval", "interval"), &UpdateNotificationPopup::set_check_interval);
    ClassDB::bind_method(D_METHOD("trigger_update_check"), &UpdateNotificationPopup::trigger_update_check);
    
    ClassDB::bind_method(D_METHOD("_on_update_response"), &UpdateNotificationPopup::_on_update_response);
    ClassDB::bind_method(D_METHOD("_on_update_button_pressed"), &UpdateNotificationPopup::_on_update_button_pressed);
    ClassDB::bind_method(D_METHOD("_on_dismiss_button_pressed"), &UpdateNotificationPopup::_on_dismiss_button_pressed);
}

UpdateNotificationPopup::UpdateNotificationPopup() {
    set_name("UpdateNotificationPopup");
    set_mouse_filter(Control::MOUSE_FILTER_IGNORE);
    set_anchors_and_offsets_preset(Control::PRESET_FULL_RECT);
    
    _setup_ui();
    _get_current_version();
    
    // Create HTTP request for update checks
    http_request = memnew(HTTPRequest);
    add_child(http_request);
    http_request->connect("request_completed", callable_mp(this, &UpdateNotificationPopup::_on_update_response));
    
    // Start auto-checking if enabled
    if (auto_check_enabled) {
        // Check on startup (delayed)
        callable_mp(this, &UpdateNotificationPopup::trigger_update_check).call_deferred();
    }
}

UpdateNotificationPopup::~UpdateNotificationPopup() {
}

void UpdateNotificationPopup::_setup_ui() {
    // Create popup panel positioned at CENTER TOP for maximum visibility
    popup_panel = memnew(PanelContainer);
    popup_panel->set_mouse_filter(Control::MOUSE_FILTER_STOP);
    add_child(popup_panel);
    
    // Position at TOP CENTER - will be properly positioned when shown
    popup_panel->set_anchors_and_offsets_preset(Control::PRESET_TOP_LEFT);
    popup_panel->set_custom_minimum_size(Vector2(500 * EDSCALE, 110 * EDSCALE));  // Bigger size
    
    // Add some styling
    popup_panel->add_theme_style_override("panel", EditorNode::get_singleton()->get_gui_base()->get_theme_stylebox("panel", "PopupPanel"));
    
    // Content container
    content_container = memnew(HBoxContainer);
    popup_panel->add_child(content_container);
    
    MarginContainer *margin = memnew(MarginContainer);
    margin->add_theme_constant_override("margin_left", 12 * EDSCALE);
    margin->add_theme_constant_override("margin_right", 12 * EDSCALE);
    margin->add_theme_constant_override("margin_top", 8 * EDSCALE);
    margin->add_theme_constant_override("margin_bottom", 8 * EDSCALE);
    content_container->add_child(margin);
    
    VBoxContainer *vbox = memnew(VBoxContainer);
    margin->add_child(vbox);
    
    // Title label - bold and prominent
    Label *title_label = memnew(Label);
    title_label->set_text("Orca Engine Update Available!");
    title_label->set_h_size_flags(Control::SIZE_EXPAND_FILL);
    title_label->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_CENTER);
    title_label->add_theme_font_size_override("font_size", 16);
    vbox->add_child(title_label);
    
    // Message label (will show version comparison)
    message_label = memnew(Label);
    message_label->set_text("A new version is ready to download");
    message_label->set_h_size_flags(Control::SIZE_EXPAND_FILL);
    message_label->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_CENTER);
    message_label->add_theme_font_size_override("font_size", 12);
    vbox->add_child(message_label);
    
    // Button container - centered
    HBoxContainer *button_container = memnew(HBoxContainer);
    button_container->set_alignment(BoxContainer::ALIGNMENT_CENTER);
    vbox->add_child(button_container);
    
    // Update button - make it prominent
    update_button = memnew(Button);
    update_button->set_text("Download & Install Update");
    update_button->set_custom_minimum_size(Vector2(200 * EDSCALE, 36 * EDSCALE));
    update_button->connect("pressed", callable_mp(this, &UpdateNotificationPopup::_on_update_button_pressed));
    button_container->add_child(update_button);
    
    // Add spacing
    Control *spacer = memnew(Control);
    spacer->set_custom_minimum_size(Vector2(16 * EDSCALE, 0));
    button_container->add_child(spacer);
    
    // Dismiss button
    dismiss_button = memnew(Button);
    dismiss_button->set_text("Remind Me Later");
    dismiss_button->set_flat(true);
    dismiss_button->connect("pressed", callable_mp(this, &UpdateNotificationPopup::_on_dismiss_button_pressed));
    button_container->add_child(dismiss_button);
    
    // Initially hidden
    popup_panel->set_visible(false);
    is_visible_state = false;
}

void UpdateNotificationPopup::_notification(int p_what) {
    switch (p_what) {
        case NOTIFICATION_READY: {
            set_process(true);
        } break;
        
        case NOTIFICATION_PROCESS: {
            if (auto_check_enabled) {
                float current_time = OS::get_singleton()->get_ticks_msec() / 1000.0f;
                if (current_time - last_check_time > check_interval) {
                    trigger_update_check();
                }
            }
        } break;
    }
}

void UpdateNotificationPopup::_get_current_version() {
    // PRODUCTION: Use version baked into binary at build time by SCons
    // This comes from core/orca_version.gen.cpp (auto-generated before compilation)
    current_version = String(ORCA_VERSION_STRING);
    
    // Allow environment override for testing
    String env_version = OS::get_singleton()->get_environment("ORCA_VERSION");
    if (!env_version.is_empty()) {
        current_version = env_version;
    }
}

void UpdateNotificationPopup::trigger_update_check() {
    if (!auto_check_enabled) {
        return;
    }
    
    last_check_time = OS::get_singleton()->get_ticks_msec() / 1000.0f;
    _check_for_updates();
}

void UpdateNotificationPopup::_check_for_updates() {
    // Use GitHub API to check for latest release
    String api_url = "https://api.github.com/repos/Simplifine-gamedev/orca-engine/releases/latest";
    
    PackedStringArray headers;
    headers.push_back("User-Agent: OrcaEngine-UpdateChecker/1.0");
    headers.push_back("Accept: application/vnd.github+json");
    
    http_request->request(api_url, headers);
}

void UpdateNotificationPopup::_on_update_response(int p_result, int p_code, const PackedStringArray &p_headers, const PackedByteArray &p_body) {
    if (p_result != HTTPRequest::RESULT_SUCCESS || p_code != 200) {
        return;
    }
    
    String response_text = String::utf8((const char *)p_body.ptr(), p_body.size());
    Variant json_var = JSON::parse_string(response_text);
    
    if (json_var.get_type() != Variant::DICTIONARY) {
        return;
    }
    
    Dictionary release_data = json_var;
    String tag_name = release_data.get("tag_name", "");
    String remote_version = tag_name.lstrip("v");
    
    // SIMPLE RULE: Only show update if versions are DIFFERENT
    if (remote_version == current_version) {
        return;
    }
    
    if (remote_version.is_empty()) {
        return;
    }
    
    // Find appropriate download URL for user's platform
    Array assets = release_data.get("assets", Array());
    String platform_download_url;
    
    for (int i = 0; i < assets.size(); i++) {
        Dictionary asset = assets[i];
        String asset_name = String(asset.get("name", "")).to_lower();
        String asset_url = asset.get("browser_download_url", "");
        
        // Platform-specific asset selection
        bool matches = false;
        if (OS::get_singleton()->has_feature("macos")) {
            matches = asset_name.ends_with(".dmg") || asset_name.contains("mac");
        } else if (OS::get_singleton()->has_feature("windows")) {
            matches = asset_name.ends_with(".exe") || asset_name.contains("windows");
        } else if (OS::get_singleton()->has_feature("linux")) {
            matches = asset_name.ends_with(".tar.gz") || asset_name.contains("linux") || asset_name.contains("appimage");
        }
        
        if (matches) {
            platform_download_url = asset_url;
            break;
        }
    }
    
    if (!platform_download_url.is_empty()) {
        latest_version = remote_version;
        download_url = platform_download_url;
        show_update_notification(remote_version, platform_download_url);
    } else {
    }
}

void UpdateNotificationPopup::show_update_notification(const String &p_version, const String &p_download_url) {
    if (is_visible_state) {
        return; // Already showing
    }
    
    latest_version = p_version;
    if (!p_download_url.is_empty()) {
        download_url = p_download_url;
    }
    
    // Show version comparison clearly
    String message = "Current: " + current_version + "  ->  New: " + p_version;
    message_label->set_text(message);
    
    _show_popup();
}

void UpdateNotificationPopup::_show_popup() {
    if (is_visible_state) {
        return;
    }
    
    // Calculate center position (NOW we have a valid viewport)
    if (get_viewport()) {
        Rect2 viewport_rect = get_viewport()->get_visible_rect();
        float popup_width = 500 * EDSCALE;
        float center_x = (viewport_rect.size.x - popup_width) / 2.0f;
        
        // Position at center horizontally, animate from above vertically
        popup_panel->set_position(Vector2(center_x, -120 * EDSCALE));  // Start above screen
    } else {
        // Fallback: just position near left if no viewport
        popup_panel->set_position(Vector2(16 * EDSCALE, -120 * EDSCALE));
    }
    
    is_visible_state = true;
    popup_panel->set_visible(true);
    
    // Animate DOWN from top (slides into view)
    tween = create_tween();
    tween->set_ease(Tween::EASE_OUT);
    tween->set_trans(Tween::TRANS_CUBIC);
    
    Vector2 current_pos = popup_panel->get_position();
    Vector2 target_pos = Vector2(current_pos.x, 16 * EDSCALE);  // 16px from top
    
    tween->tween_property(popup_panel, NodePath("position"), target_pos, 0.6);
}

void UpdateNotificationPopup::_hide_popup() {
    if (!is_visible_state) {
        return;
    }
    
    // Animate UP out of view (slides back above screen)
    tween = create_tween();
    tween->set_ease(Tween::EASE_IN);
    tween->set_trans(Tween::TRANS_CUBIC);
    
    Vector2 current_pos = popup_panel->get_position();
    Vector2 hide_pos = Vector2(current_pos.x, -120 * EDSCALE);  // Above screen
    
    // Animate back up above screen
    tween->tween_property(popup_panel, NodePath("position"), hide_pos, 0.4);
    tween->tween_callback(callable_mp(this, &UpdateNotificationPopup::_hide_panel_callback));
    
    is_visible_state = false;
}

void UpdateNotificationPopup::_hide_panel_callback() {
    popup_panel->set_visible(false);
}

void UpdateNotificationPopup::hide_notification() {
    _hide_popup();
}

void UpdateNotificationPopup::_on_update_button_pressed() {
    // Hide the popup
    _hide_popup();
    
    // Launch the full EditorUpdater dialog for download and installation
    EditorUpdater *updater = memnew(EditorUpdater);
    
    // Configure with the same URLs as the main system
    String default_feed = OS::get_singleton()->has_feature("windows")
            ? String("https://simplifine-gamedev.github.io/orca-engine/appcast-windows.xml")
            : String("https://simplifine-gamedev.github.io/orca-engine/appcast.xml");
    
    updater->set_feed_url(default_feed);
    updater->set_owner_repo("Simplifine-gamedev/orca-engine");
    
    EditorNode::get_singleton()->get_gui_base()->add_child(updater);
    updater->popup_centered();
    updater->start_check();
}

void UpdateNotificationPopup::_on_dismiss_button_pressed() {
    _hide_popup();
}

void UpdateNotificationPopup::set_auto_check_enabled(bool p_enabled) {
    auto_check_enabled = p_enabled;
}

void UpdateNotificationPopup::set_check_interval(float p_interval) {
    check_interval = p_interval;
}
