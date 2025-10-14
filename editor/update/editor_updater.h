/**************************************************************************/
/*  editor_updater.h                                                       */
/**************************************************************************/

#pragma once

#include "core/object/ref_counted.h"
#include "core/object/object.h"
#include "core/io/xml_parser.h"
#include "scene/gui/dialogs.h"
#include "scene/gui/button.h"
#include "scene/gui/label.h"
#include "scene/gui/progress_bar.h"  // Re-added for download progress display
#include "scene/gui/box_container.h"
#include "scene/main/http_request.h"

class EditorUpdater : public AcceptDialog {
    GDCLASS(EditorUpdater, AcceptDialog);

    Label *status_label = nullptr;
    ProgressBar *progress = nullptr;  // Re-added for visual download progress
    Button *action_button = nullptr;
    Button *close_button = nullptr;
    HTTPRequest *http = nullptr;
    HTTPRequest *http_release = nullptr;

    String feed_url;
    String download_url;
    String latest_version;
    String current_version;  // Current installed version
    String owner_repo;
    String downloaded_file_path;
    
    // Manual download tracking for progress
    Ref<FileAccess> download_file;
    int64_t total_download_size = 0;
    int64_t current_download_size = 0;

    enum Stage {
        STAGE_IDLE,
        STAGE_CHECKING,
        STAGE_AVAILABLE,
        STAGE_DOWNLOADING,
        STAGE_DOWNLOADED,
        STAGE_ERROR,
    } stage = STAGE_IDLE;

    void _on_request_completed(int p_result, int p_code, const PackedStringArray &p_headers, const PackedByteArray &p_body);
    void _on_release_completed(int p_result, int p_code, const PackedStringArray &p_headers, const PackedByteArray &p_body);
    void _on_pressed();
    void _install_and_restart();
    void _start_github_check();
    void _process(double p_delta);
    void _get_current_version();
    void _notify_backend_update_installed();

protected:
    static void _bind_methods();

public:
    void set_feed_url(const String &p_url) { feed_url = p_url; }
    void set_owner_repo(const String &p_repo) { owner_repo = p_repo; }
    void start_check();

    EditorUpdater();
};


