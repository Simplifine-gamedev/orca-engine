/**************************************************************************/
/*  update_notification_popup.h                                          */
/**************************************************************************/

#pragma once

#include "core/object/ref_counted.h"
#include "scene/gui/control.h"
#include "scene/gui/button.h"
#include "scene/gui/label.h"
#include "scene/gui/panel_container.h"
#include "scene/gui/box_container.h"
#include "scene/main/http_request.h"
#include "scene/animation/tween.h"

class UpdateNotificationPopup : public Control {
    GDCLASS(UpdateNotificationPopup, Control);

private:
    PanelContainer *popup_panel = nullptr;
    HBoxContainer *content_container = nullptr;
    Label *message_label = nullptr;
    Button *update_button = nullptr;
    Button *dismiss_button = nullptr;
    Ref<Tween> tween;
    
    bool is_visible_state = false;
    bool auto_check_enabled = true;
    float check_interval = 3600.0f; // 1 hour
    float last_check_time = 0.0f;
    
    HTTPRequest *http_request = nullptr;
    String current_version;
    String latest_version;
    String download_url;
    
    void _setup_ui();
    void _check_for_updates();
    void _on_update_response(int p_result, int p_code, const PackedStringArray &p_headers, const PackedByteArray &p_body);
    void _on_update_button_pressed();
    void _on_dismiss_button_pressed();
    void _show_popup();
    void _hide_popup();
    void _hide_panel_callback();
    void _get_current_version();

protected:
    static void _bind_methods();
    void _notification(int p_what);

public:
    void show_update_notification(const String &p_version, const String &p_download_url = "");
    void hide_notification();
    void set_auto_check_enabled(bool p_enabled);
    void set_check_interval(float p_interval);
    void trigger_update_check();
    
    UpdateNotificationPopup();
    ~UpdateNotificationPopup();
};

