/**************************************************************************/
/*  update_notification_popup.cpp                                        */
/**************************************************************************/

#include "update_notification_popup.h"

#include "core/io/json.h"
#include "core/os/os.h"
#include "editor/editor_node.h"
#include "editor/editor_paths.h"
#include "editor/themes/editor_scale.h"
#include "scene/gui/margin_container.h"

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
    // Create popup panel positioned at bottom-left
    popup_panel = memnew(PanelContainer);
    popup_panel->set_mouse_filter(Control::MOUSE_FILTER_STOP);
    add_child(popup_panel);
    
    // Position at bottom-left with some margin
    popup_panel->set_anchors_and_offsets_preset(Control::PRESET_BOTTOM_LEFT);
    popup_panel->set_position(Vector2(16 * EDSCALE, -80 * EDSCALE));
    popup_panel->set_size(Vector2(320 * EDSCALE, 64 * EDSCALE));
    
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
    
    // Message label
    message_label = memnew(Label);
    message_label->set_text("Update available!");
    message_label->set_h_size_flags(Control::SIZE_EXPAND_FILL);
    vbox->add_child(message_label);
    
    // Button container
    HBoxContainer *button_container = memnew(HBoxContainer);
    vbox->add_child(button_container);
    
    // Update button
    update_button = memnew(Button);
    update_button->set_text("Update Now");
    update_button->connect("pressed", callable_mp(this, &UpdateNotificationPopup::_on_update_button_pressed));
    button_container->add_child(update_button);
    
    // Dismiss button
    dismiss_button = memnew(Button);
    dismiss_button->set_text("Later");
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
                float current_time = Time::get_singleton()->get_ticks_msec() / 1000.0f;
                if (current_time - last_check_time > check_interval) {
                    trigger_update_check();
                }
            }
        } break;
    }
}

void UpdateNotificationPopup::_get_current_version() {
    // Try to get version from git or version file
    current_version = "0.01.unknown";
    
    // Try git first
    List<String> args;
    args.push_back("rev-parse");
    args.push_back("--short=8");
    args.push_back("HEAD");
    
    String output;
    int exit_code;
    Error err = OS::get_singleton()->execute("git", args, &output, &exit_code, true);
    if (err == OK && exit_code == 0) {
        String sha = output.strip_edges();
        if (!sha.is_empty()) {
            current_version = "0.01." + sha;
        }
    }
    
    print_line("UpdateNotificationPopup: Current version: " + current_version);
}

void UpdateNotificationPopup::trigger_update_check() {
    if (!auto_check_enabled) {
        return;
    }
    
    last_check_time = Time::get_singleton()->get_ticks_msec() / 1000.0f;
    _check_for_updates();
}

void UpdateNotificationPopup::_check_for_updates() {
    print_line("UpdateNotificationPopup: Checking for updates...");
    
    // Use GitHub API to check for latest release
    String api_url = "https://api.github.com/repos/Simplifine-gamedev/orca-engine/releases/latest";
    
    PackedStringArray headers;
    headers.push_back("User-Agent: OrcaEngine-UpdateChecker/1.0");
    headers.push_back("Accept: application/vnd.github+json");
    
    http_request->request(api_url, headers);
}

void UpdateNotificationPopup::_on_update_response(int p_result, int p_code, const PackedStringArray &p_headers, const PackedByteArray &p_body) {
    if (p_result != HTTPRequest::RESULT_SUCCESS || p_code != 200) {
        print_line("UpdateNotificationPopup: Failed to check for updates - " + itos(p_code));
        return;
    }
    
    String response_text = String::utf8((const char *)p_body.ptr(), p_body.size());
    Variant json_var = JSON::parse_string(response_text);
    
    if (json_var.get_type() != Variant::DICTIONARY) {
        print_line("UpdateNotificationPopup: Invalid JSON response");
        return;
    }
    
    Dictionary release_data = json_var;
    String tag_name = release_data.get("tag_name", "");
    String version = tag_name.lstrip("v");
    
    print_line("UpdateNotificationPopup: Latest version: " + version + ", Current: " + current_version);
    
    // Check if this is a newer version
    if (version != current_version && !version.is_empty()) {
        // Find appropriate download URL
        Array assets = release_data.get("assets", Array());
        String platform_download_url;
        
        for (int i = 0; i < assets.size(); i++) {
            Dictionary asset = assets[i];
            String asset_name = asset.get("name", "").to_lower();
            String asset_url = asset.get("browser_download_url", "");
            
            // Platform-specific asset selection
            bool matches = false;
            if (OS::get_singleton()->has_feature("macos")) {
                matches = asset_name.ends_with(".dmg");
            } else if (OS::get_singleton()->has_feature("windows")) {
                matches = asset_name.ends_with(".exe");
            } else if (OS::get_singleton()->has_feature("linux")) {
                matches = asset_name.ends_with(".tar.gz") || asset_name.contains("linux");
            }
            
            if (matches) {
                platform_download_url = asset_url;
                break;
            }
        }
        
        if (!platform_download_url.is_empty()) {
            latest_version = version;
            download_url = platform_download_url;
            show_update_notification(version, platform_download_url);
        }
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
    
    message_label->set_text("Orca Engine " + p_version + " available!");
    
    _show_popup();
}

void UpdateNotificationPopup::_show_popup() {
    if (is_visible_state) {
        return;
    }
    
    is_visible_state = true;
    popup_panel->set_visible(true);
    
    // Animate in from bottom
    tween = create_tween();
    tween->set_ease(Tween::EASE_OUT);
    tween->set_trans(Tween::TRANS_BACK);
    
    popup_panel->set_position(Vector2(16 * EDSCALE, 20 * EDSCALE)); // Start below screen
    tween->tween_property(popup_panel, "position", Vector2(16 * EDSCALE, -80 * EDSCALE), 0.5);
}

void UpdateNotificationPopup::_hide_popup() {
    if (!is_visible_state) {
        return;
    }
    
    // Animate out to bottom
    tween = create_tween();
    tween->set_ease(Tween::EASE_IN);
    tween->set_trans(Tween::TRANS_BACK);
    
    tween->tween_property(popup_panel, "position", Vector2(16 * EDSCALE, 20 * EDSCALE), 0.3);
    tween->tween_callback(callable_mp(popup_panel, &Control::set_visible).bind(false));
    
    is_visible_state = false;
}

void UpdateNotificationPopup::hide_notification() {
    _hide_popup();
}

void UpdateNotificationPopup::_on_update_button_pressed() {
    print_line("UpdateNotificationPopup: User clicked Update Now - launching EditorUpdater");
    
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
    print_line("UpdateNotificationPopup: User dismissed update notification");
    _hide_popup();
}

void UpdateNotificationPopup::set_auto_check_enabled(bool p_enabled) {
    auto_check_enabled = p_enabled;
}

void UpdateNotificationPopup::set_check_interval(float p_interval) {
    check_interval = p_interval;
}
