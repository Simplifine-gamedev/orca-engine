/**************************************************************************/
/*  editor_updater.cpp                                                     */
/**************************************************************************/

#include "editor_updater.h"

#include "core/io/dir_access.h"
#include "core/os/os.h"
#include "core/io/json.h"
#include "core/string/translation_server.h"
#include "core/string/ustring.h"  // For TTR translation macro
#include "core/version.h"  // For VERSION_FULL_CONFIG and VERSION_HASH
#include "core/orca_version.h"  // Orca version embedded at build time
#include "editor/file_system/editor_paths.h"
#include "editor/editor_node.h"
#include "editor/themes/editor_scale.h"
#include "scene/gui/box_container.h"

void EditorUpdater::_bind_methods() {
}

EditorUpdater::EditorUpdater() {
    set_title("Update Orca");
    set_min_size(Size2(520, 180) * EDSCALE);  // Increased height for progress bar
    
    // Get current version using improved detection (consistent with backend)
    _get_current_version();

    VBoxContainer *vb = memnew(VBoxContainer);
    add_child(vb);

    // Version display labels
    Label *version_info = memnew(Label);
    version_info->set_text("");
    version_info->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_CENTER);
    version_info->add_theme_font_size_override("font_size", 11);
    vb->add_child(version_info);
    version_info->set_name("version_info_label");  // For easy lookup
    
    status_label = memnew(Label);
    status_label->set_text("Checking for updates...");
    vb->add_child(status_label);

    // Re-added progress bar for visual download progress
    progress = memnew(ProgressBar);
    progress->set_min(0);
    progress->set_max(100);
    progress->set_step(0.01);  // Finer granularity for smoother updates
    progress->set_value(0);
    progress->set_h_size_flags(Control::SIZE_EXPAND_FILL);
    progress->set_visible(false);  // Hidden until download starts
    vb->add_child(progress);

    action_button = memnew(Button);
    action_button->set_text("Install and Restart");
    action_button->connect("pressed", callable_mp(this, &EditorUpdater::_on_pressed));
    action_button->set_visible(false); // Hide until download is complete
    vb->add_child(action_button);

    http = memnew(HTTPRequest);
    add_child(http);
    http->connect("request_completed", callable_mp(this, &EditorUpdater::_on_request_completed));
    
    // CRITICAL: Set download chunk size to ensure we get frequent progress updates
    http->set_download_chunk_size(32768);  // 32KB chunks for smooth progress

    set_process(true);  // Enable _process() for progress updates
}

void EditorUpdater::_get_current_version() {
    // PRODUCTION: Use version baked into binary at build time by SCons
    // This comes from core/orca_version.gen.cpp (auto-generated during compilation)
    current_version = String(ORCA_VERSION_STRING);
    
    print_line("EditorUpdater: ✅ Version from compiled binary: " + current_version);
    
    // Allow environment override for testing
    String env_version = OS::get_singleton()->get_environment("ORCA_VERSION");
    if (!env_version.is_empty()) {
        current_version = env_version;
        print_line("EditorUpdater: 🧪 Version overridden for testing: " + current_version);
    }
}

void EditorUpdater::_notify_backend_update_installed() {
    // PRODUCTION: Update notifications work without backend server
    // This function is kept for future local development integration
    
    // Only try to notify backend in development mode (when AI backend might be running)
    String dev_mode = OS::get_singleton()->get_environment("IS_DEV");
    if (dev_mode.to_lower() != "true") {
        print_line("EditorUpdater: Skipping backend notification (production mode - not needed)");
        return;
    }
    
    print_line("EditorUpdater: Development mode - attempting backend notification");
    
    // Try to notify local backend that this version was installed (dev only)
    HTTPRequest *notify_http = memnew(HTTPRequest);
    add_child(notify_http);
    
    // Create JSON payload
    Dictionary payload;
    payload["version"] = latest_version.is_empty() ? "unknown" : latest_version;
    
    Ref<JSON> json;
    json.instantiate();
    String json_string = json->stringify(payload);
    
    // Try local backend only
    PackedStringArray headers;
    headers.push_back("Content-Type: application/json");
    
    String backend_url = "http://localhost:8080/update/mark_installed";
    Error notify_err = notify_http->request(backend_url, headers, HTTPClient::METHOD_POST, json_string);
    
    if (notify_err == OK) {
        print_line("EditorUpdater: Notified local backend that version " + latest_version + " was installed");
    } else {
        print_line("EditorUpdater: Local backend notification failed (expected in production): " + itos(notify_err));
    }
}

void EditorUpdater::start_check() {
    stage = STAGE_CHECKING;
    status_label->set_text("Checking for updates...");

    if (feed_url.is_empty()) {
        status_label->set_text("No update feed configured.");
        stage = STAGE_ERROR;
        action_button->set_text("Close");
        action_button->set_visible(true);
        return;
    }

    Error err = http->request(feed_url);
    if (err != OK) {
        status_label->set_text("Failed to request update feed.");
        // Try GitHub releases if configured
        if (!owner_repo.is_empty()) {
            status_label->set_text("Checking GitHub releases...");
            if (!http_release) {
                http_release = memnew(HTTPRequest);
                add_child(http_release);
                http_release->connect("request_completed", callable_mp(this, &EditorUpdater::_on_release_completed));
            }
            // Use /releases instead of /releases/latest to get draft releases too
            String api = "https://api.github.com/repos/" + owner_repo + "/releases";
            PackedStringArray headers;
            headers.push_back("User-Agent: OrcaEditorUpdater/1.0");
            headers.push_back("Accept: application/vnd.github+json");
            Error e2 = http_release->request(api, headers);
            if (e2 == OK) {
                return;
            }
        }
        stage = STAGE_ERROR;
        return;
    }
}

void EditorUpdater::_on_request_completed(int p_result, int p_code, const PackedStringArray &p_headers, const PackedByteArray &p_body) {
    if (stage == STAGE_CHECKING) {
        if (p_result != HTTPRequest::RESULT_SUCCESS || p_code != 200) {
            // Try GitHub releases if configured
            if (!owner_repo.is_empty()) {
                status_label->set_text("Checking GitHub releases...");
                if (!http_release) {
                    http_release = memnew(HTTPRequest);
                    add_child(http_release);
                http_release->connect("request_completed", callable_mp(this, &EditorUpdater::_on_release_completed));
            }
            // Use /releases instead of /releases/latest to get draft releases too
            String api = "https://api.github.com/repos/" + owner_repo + "/releases";
            PackedStringArray headers;
            headers.push_back("User-Agent: OrcaEditorUpdater/1.0");
            headers.push_back("Accept: application/vnd.github+json");
            Error e2 = http_release->request(api, headers);
                if (e2 == OK) {
                    return;
                }
            }
            status_label->set_text("Failed to check updates.");
            stage = STAGE_ERROR;
            action_button->set_text("Close");
            action_button->set_visible(true);
            return;
        }
        Ref<XMLParser> parser;
        parser.instantiate();
        if (parser->open_buffer(p_body) != OK) {
            status_label->set_text("Invalid appcast feed.");
            stage = STAGE_ERROR;
            return;
        }
        String found_url;
        String found_version;
        while (parser->read() == OK) {
            if (parser->get_node_type() == XMLParser::NODE_ELEMENT) {
                String name = parser->get_node_name();
                if (name == "enclosure") {
                    if (parser->has_attribute("url")) {
                        found_url = parser->get_named_attribute_value("url");
                    }
                    if (parser->has_attribute("sparkle:version")) {
                        found_version = parser->get_named_attribute_value("sparkle:version");
                    }
                }
            }
            if (!found_url.is_empty() && !found_version.is_empty()) {
                break;
            }
        }
        if (found_url.is_empty()) {
            status_label->set_text("No update available.");
            stage = STAGE_IDLE;
            action_button->set_text("Close");
            action_button->set_visible(true);
            return;
        }
        download_url = found_url;
        latest_version = found_version;
        
        // Update version info display (show user what versions we're comparing)
        Label *version_info_label = Object::cast_to<Label>(find_child("version_info_label", true, false));
        if (version_info_label) {
            version_info_label->set_text("Current ver: " + current_version + "  |  New ver: " + latest_version);
        }
        
        // Check if we're already on the latest version
        if (latest_version == current_version) {
            status_label->set_text("Already up to date! You have the latest version (" + current_version + ")");
            stage = STAGE_IDLE;
            action_button->set_text("Close");
            action_button->set_visible(true);
            if (progress) {
                progress->set_visible(false);
            }
            return;
        }
        
        status_label->set_text("Update " + latest_version + " available. Downloading...");
        stage = STAGE_DOWNLOADING;

        // Save using the asset's filename so macOS opens it with the right app (zip/dmg)
        String file_name = download_url.get_file();
        int qpos = file_name.find("?");
        if (qpos != -1) {
            file_name = file_name.substr(0, qpos);
        }
        if (file_name.is_empty()) {
            file_name = "orca_update";
        }
        downloaded_file_path = EditorPaths::get_singleton()->get_cache_dir().path_join(file_name);
        
        // Reset download tracking
        current_download_size = 0;
        total_download_size = 0;
        
        // DON'T use set_download_file() - we'll write manually for progress
        // http->set_download_file(downloaded_file_path);  
        
        // Use threading for non-blocking download
        http->set_use_threads(true);
        http->set_body_size_limit(-1);  // No limit, get full body
        http->set_timeout(300);  // 5 minutes
        
        // Show progress bar
        if (progress) {
            progress->set_visible(true);
            progress->set_value(0);
        }
        
        // Start download - we'll get body in request_completed
        Error download_err = http->request(download_url);
        if (download_err != OK) {
            status_label->set_text("Failed to start download.");
            stage = STAGE_ERROR;
            if (progress) {
                progress->set_visible(false);
            }
            action_button->set_text("Close");
            action_button->set_visible(true);
        } else {
            status_label->set_text("Connecting...");
            set_process(true);
        }
        return;
    }

    if (stage == STAGE_DOWNLOADING) {
        if (p_result != HTTPRequest::RESULT_SUCCESS || p_code != 200) {
            status_label->set_text("Download failed. Code: " + itos(p_code));
            stage = STAGE_ERROR;
            action_button->set_text("Close");
            action_button->set_visible(true);
            return;
        }
        
        // Write downloaded body to file
        Error write_err;
        Ref<FileAccess> file = FileAccess::open(downloaded_file_path, FileAccess::WRITE, &write_err);
        if (write_err != OK || file.is_null()) {
            status_label->set_text("Failed to save file.");
            stage = STAGE_ERROR;
            return;
        }
        
        file->store_buffer(p_body);
        file->close();
        
        int64_t file_size = p_body.size();
        
        stage = STAGE_DOWNLOADED;
        
        // Show 100% completion
        if (progress) {
            progress->set_value(100);
            progress->set_visible(true);
        }
        
        status_label->set_text("Download complete! (" + String::humanize_size(file_size) + ") Ready to install.");
        action_button->set_text("Install and Restart");
        action_button->set_visible(true);
    }
}

void EditorUpdater::_on_release_completed(int p_result, int p_code, const PackedStringArray &p_headers, const PackedByteArray &p_body) {
    if (p_result != HTTPRequest::RESULT_SUCCESS || p_code != 200) {
        status_label->set_text("Failed to check GitHub releases.");
        stage = STAGE_ERROR;
        action_button->set_text("Close");
        action_button->set_visible(true);
        return;
    }
    String s = String::utf8((const char *)p_body.ptr(), p_body.size());
    Variant json_v = JSON::parse_string(s);
    
    Dictionary d;
    
    if (json_v.get_type() == Variant::ARRAY) {
        // We got an array of releases (from /releases endpoint)
        Array releases = json_v;
        print_line("EditorUpdater: Received array of " + itos(releases.size()) + " releases");
        
        if (releases.size() == 0) {
            status_label->set_text("No releases found.");
            stage = STAGE_ERROR;
            return;
        }
        
        // Pick the first release (most recent)
        Variant first_release = releases[0];
        if (first_release.get_type() != Variant::DICTIONARY) {
            status_label->set_text("Invalid release data.");
            stage = STAGE_ERROR;
            return;
        }
        d = first_release;
        
    } else if (json_v.get_type() == Variant::DICTIONARY) {
        // We got a single release (from /releases/latest endpoint)
        print_line("EditorUpdater: Received single release object");
        d = json_v;
    } else {
        status_label->set_text("Invalid releases JSON.");
        stage = STAGE_ERROR;
        return;
    }
    String tag = d.get("tag_name", String());
    bool is_draft = d.get("draft", false);
    bool is_prerelease = d.get("prerelease", false);
    Array assets = d.get("assets", Array());
    
    // Debug: Print detailed release information
    String platform = "unknown";
    if (OS::get_singleton()->has_feature("windows")) {
        platform = "windows";
    } else if (OS::get_singleton()->has_feature("macos")) {
        platform = "macos";
    } else {
        platform = "linux";
    }
    
    print_line("EditorUpdater: Platform detected: " + platform);
    print_line("EditorUpdater: Found release " + tag + " (draft: " + (is_draft ? "yes" : "no") + ", prerelease: " + (is_prerelease ? "yes" : "no") + ")");
    print_line("EditorUpdater: API returned " + itos(assets.size()) + " assets for this release");
    
    // Debug: Print the raw JSON response (first 500 chars)
    String json_str = s.substr(0, 500);
    print_line("EditorUpdater: GitHub API Response (truncated): " + json_str);
    
    String best_url;
    for (int i = 0; i < assets.size(); i++) {
        Dictionary a = assets[i];
        String name = a.get("name", String());
        String url = a.get("browser_download_url", String());
        
        // Debug: Print each asset
        print_line("EditorUpdater: Asset " + itos(i) + ": " + name);
        
        if (name.is_empty() || url.is_empty()) {
            print_line("EditorUpdater: Skipping asset with empty name or URL");
            continue;
        }
        
        bool matches = false;
        String name_lower = name.to_lower();
        
        if (OS::get_singleton()->has_feature("windows")) {
            // Windows: Look for .exe, .msi, or .zip files
            // Also check for common Windows patterns like "win", "windows"
            if (name_lower.ends_with(".exe") || 
                name_lower.ends_with(".msi") ||
                (name_lower.ends_with(".zip") && (name_lower.contains("win") || name_lower.contains("windows")))) {
                matches = true;
            }
        } else if (OS::get_singleton()->has_feature("macos")) {
            // macOS: Look for .dmg, .zip files, or common macOS patterns
            // Also check for "mac", "macos", "osx", "darwin" in filename
            if (name_lower.ends_with(".dmg") || 
                name_lower.ends_with(".zip") || 
                name_lower.ends_with(".pkg") ||
                name_lower.contains("mac") || 
                name_lower.contains("macos") ||
                name_lower.contains("osx") ||
                name_lower.contains("darwin")) {
                matches = true;
            }
        } else {
            // Linux: Look for AppImage, tar.gz, deb, rpm, or common Linux patterns
            if (name_lower.contains("appimage") || 
                name_lower.ends_with(".tar.gz") || 
                name_lower.ends_with(".tar.xz") || 
                name_lower.ends_with(".deb") || 
                name_lower.ends_with(".rpm") ||
                (name_lower.ends_with(".zip") && (name_lower.contains("linux") || name_lower.contains("unix")))) {
                matches = true;
            }
        }
        
        if (matches) {
            print_line("EditorUpdater: Found matching asset: " + name);
            best_url = url;
            break;
        }
    }
    
    // If no platform-specific match found, try smart fallback patterns
    if (best_url.is_empty()) {
        print_line("EditorUpdater: No platform-specific asset found, trying smart fallback patterns...");
        
        // Fallback 1: Look for any .zip files (most universal)
        for (int i = 0; i < assets.size(); i++) {
            Dictionary a = assets[i];
            String name = a.get("name", String());
            String url = a.get("browser_download_url", String());
            if (!name.is_empty() && !url.is_empty() && name.to_lower().ends_with(".zip")) {
                print_line("EditorUpdater: Using fallback ZIP asset: " + name);
                best_url = url;
                break;
            }
        }
        
        // Fallback 2: For macOS, try to avoid Windows-specific files
        if (best_url.is_empty() && OS::get_singleton()->has_feature("macos")) {
            for (int i = 0; i < assets.size(); i++) {
                Dictionary a = assets[i];
                String name = a.get("name", String());
                String url = a.get("browser_download_url", String());
                String name_lower = name.to_lower();
                
                // Skip obviously Windows-only files
                if (name_lower.ends_with(".exe") || name_lower.ends_with(".msi") || name_lower.contains("windows")) {
                    print_line("EditorUpdater: Skipping Windows-only asset on macOS: " + name);
                    continue;
                }
                
                if (!name.is_empty() && !url.is_empty()) {
                    print_line("EditorUpdater: Using non-Windows asset as macOS fallback: " + name);
                    best_url = url;
                    break;
                }
            }
        }
        
        // Fallback 3: Similar logic for other platforms
        if (best_url.is_empty() && !OS::get_singleton()->has_feature("windows")) {
            for (int i = 0; i < assets.size(); i++) {
                Dictionary a = assets[i];
                String name = a.get("name", String());
                String url = a.get("browser_download_url", String());
                String name_lower = name.to_lower();
                
                // On non-Windows, avoid .exe and .msi files
                if (name_lower.ends_with(".exe") || name_lower.ends_with(".msi")) {
                    continue;
                }
                
                if (!name.is_empty() && !url.is_empty()) {
                    print_line("EditorUpdater: Using compatible asset as fallback: " + name);
                    best_url = url;
                    break;
                }
            }
        }
        
        // Last resort: Only use first asset if it's at least potentially compatible
        if (best_url.is_empty() && assets.size() > 0) {
            Dictionary a = assets[0];
            String name = a.get("name", String());
            String url = a.get("browser_download_url", String());
            String name_lower = name.to_lower();
            
            // Final compatibility check
            bool compatible = true;
            if (OS::get_singleton()->has_feature("macos") && (name_lower.ends_with(".exe") || name_lower.ends_with(".msi"))) {
                compatible = false;
                print_line("EditorUpdater: First asset (" + name + ") is not compatible with macOS");
            } else if (OS::get_singleton()->has_feature("linux") && (name_lower.ends_with(".exe") || name_lower.ends_with(".msi") || name_lower.ends_with(".dmg"))) {
                compatible = false;
                print_line("EditorUpdater: First asset (" + name + ") is not compatible with Linux");
            }
            
            if (compatible && !name.is_empty() && !url.is_empty()) {
                print_line("EditorUpdater: Using first available compatible asset as last resort: " + name);
                best_url = url;
            }
        }
    }
    
    if (best_url.is_empty()) {
        String error_msg = "No compatible update found for " + platform + ".\n";
        error_msg += "Release: " + tag + " has " + itos(assets.size()) + " assets\n";
        
        if (assets.size() > 0) {
            error_msg += "Available assets:\n";
            for (int i = 0; i < assets.size(); i++) {
                Dictionary a = assets[i];
                String name = a.get("name", String());
                if (!name.is_empty()) {
                    error_msg += "- " + name;
                    // Add compatibility note
                    String name_lower = name.to_lower();
                    if (OS::get_singleton()->has_feature("macos")) {
                        if (name_lower.ends_with(".exe") || name_lower.ends_with(".msi")) {
                            error_msg += " (Windows only)";
                        } else if (name_lower.ends_with(".dmg") || name_lower.contains("mac")) {
                            error_msg += " (Compatible)";
                        }
                    }
                    error_msg += "\n";
                }
            }
            error_msg += "\nThis release doesn't have assets compatible with " + platform + ".\n";
            error_msg += "Please check the project's GitHub releases page.";
        }
        
        print_line("EditorUpdater: " + error_msg);
        status_label->set_text(error_msg);
        stage = STAGE_ERROR;
        action_button->set_text("Close");
        action_button->set_visible(true);
        return;
    }
    download_url = best_url;
    latest_version = tag;
    
    // Update version info display (show user what versions we're comparing)
    Label *version_info_label = Object::cast_to<Label>(find_child("version_info_label", true, false));
    if (version_info_label) {
        version_info_label->set_text("Current ver: " + current_version + "  |  New ver: " + latest_version);
    }
    
    // Check if we're already on the latest version
    if (latest_version == current_version) {
        print_line("EditorUpdater: Already up to date! Version: " + current_version);
        status_label->set_text("Already up to date! You have the latest version (" + current_version + ")");
        stage = STAGE_IDLE;
        action_button->set_text("Close");
        action_button->set_visible(true);
        if (progress) {
            progress->set_visible(false);
        }
        return;
    }
    
    status_label->set_text("Update " + latest_version + " available. Downloading...");
    stage = STAGE_DOWNLOADING;
    
    // Show progress bar for download
    if (progress) {
        progress->set_visible(true);
        progress->set_value(0);
    }
    
    // Save using the asset's filename so the OS knows how to open it.
    String file_name = download_url.get_file();
    int qpos = file_name.find("?");
    if (qpos != -1) {
        file_name = file_name.substr(0, qpos);
    }
    if (file_name.is_empty()) {
        file_name = "orca_update";
    }
    downloaded_file_path = EditorPaths::get_singleton()->get_cache_dir().path_join(file_name);
    
    // Reset download tracking
    current_download_size = 0;
    total_download_size = 0;
    
    // DON'T use set_download_file() - we'll write manually for progress
    // http->set_download_file(downloaded_file_path);
    
    // Use threading for non-blocking download
    http->set_use_threads(true);
    http->set_body_size_limit(-1);  // No limit
    http->set_timeout(300);  // 5 minutes
    
    // Show progress bar
    if (progress) {
        progress->set_visible(true);
        progress->set_value(0);
    }
    
    // Start download
    print_line("EditorUpdater: Starting download from GitHub: " + download_url);
    print_line("EditorUpdater: Will save to: " + downloaded_file_path);
    Error download_err = http->request(download_url);
    if (download_err != OK) {
        print_line("EditorUpdater: Failed to start download, error: " + itos(download_err));
        status_label->set_text("Failed to start download.");
        stage = STAGE_ERROR;
        if (progress) {
            progress->set_visible(false);
        }
        action_button->set_text("Close");
        action_button->set_visible(true);
    } else {
        status_label->set_text("Connecting...");
        set_process(true);
    }
}

void EditorUpdater::_process(double p_delta) {
    if (stage == STAGE_DOWNLOADING) {
        // Get real-time download progress
        int64_t downloaded = http->get_downloaded_bytes();
        int64_t total = http->get_body_size();
        
        // Update our tracking
        current_download_size = downloaded;
        if (total > 0 && total_download_size == 0) {
            total_download_size = total;
            print_line("EditorUpdater: Total download size: " + String::humanize_size(total));
        }
        
        HTTPClient::Status status = http->get_http_client_status();
        switch (status) {
            case HTTPClient::STATUS_RESOLVING: 
                status_label->set_text("Resolving server...");
                break;
            case HTTPClient::STATUS_CONNECTING: 
                status_label->set_text("Connecting to server...");
                break;
            case HTTPClient::STATUS_REQUESTING: 
                status_label->set_text("Requesting download...");
                break;
            case HTTPClient::STATUS_CONNECTED: 
                status_label->set_text("Connected, starting download...");
                break;
            case HTTPClient::STATUS_BODY: {
                // Show real-time download progress
                progress->set_visible(true);
                
                if (downloaded > 0 && total > 0) {
                    // Calculate percentage
                    double percent = ((double)downloaded / (double)total) * 100.0;
                    if (percent < 0.0) percent = 0.0;
                    if (percent > 100.0) percent = 100.0;
                    
                    // Update progress bar (use our percentage calculation)
                    progress->set_value(percent);
                    
                    // Show progress text: "X MB / Y MB (Z%)"
                    String progress_text = String::humanize_size(downloaded) + " / " + String::humanize_size(total) + " (" + String::num(percent, 1) + "%)";
                    status_label->set_text(progress_text);
                    
                    // Log significant progress milestones
                    static int last_percent_logged = -1;
                    int current_percent = (int)percent;
                    if (current_percent % 10 == 0 && current_percent != last_percent_logged) {
                        print_line("EditorUpdater: Download progress: " + progress_text);
                        last_percent_logged = current_percent;
                    }
                } else if (downloaded > 0) {
                    // Unknown total - show indeterminate progress
                    progress->set_value(50);
                    status_label->set_text("Downloaded " + String::humanize_size(downloaded) + "...");
                } else {
                    // Just started
                    progress->set_value(5);
                    status_label->set_text("Downloading...");
                }
            } break;
            case HTTPClient::STATUS_DISCONNECTED:
                if (downloaded > 0) {
                    status_label->set_text("Download finishing...");
                }
                break;
            default: 
                status_label->set_text("Downloading... (Status: " + itos((int)status) + ")");
                break;
        }
    }
}

void EditorUpdater::_on_pressed() {
    if (stage == STAGE_DOWNLOADED) {
        _install_and_restart();
    } else {
        hide();
    }
}

void EditorUpdater::_install_and_restart() {
    if (downloaded_file_path.is_empty() || !FileAccess::exists(downloaded_file_path)) {
        hide();
        return;
    }

#ifdef WINDOWS_ENABLED
    {
        print_line("EditorUpdater: Windows installation starting...");
        print_line("EditorUpdater: Downloaded file: " + downloaded_file_path);
        
        // CRITICAL FIX: The installer should launch the new Godot with --editor flag
        // to prevent "Couldn't detect whether to run editor" error
        List<String> args;
        
        String file_lower = downloaded_file_path.to_lower();
        String file_name = downloaded_file_path.get_file().to_lower();
        
        // Improved installer/executable detection
        bool is_installer = file_lower.contains("setup") || 
                           file_lower.contains("install") ||
                           file_name.begins_with("orca-engine-") ||
                           file_name.contains("installer");
        
        // Check file size - installers are typically larger than 50MB
        Error size_err;
        Ref<FileAccess> size_check = FileAccess::open(downloaded_file_path, FileAccess::READ, &size_err);
        bool likely_installer = false;
        if (size_err == OK && size_check.is_valid()) {
            int64_t file_size = size_check->get_length();
            likely_installer = (file_size > 50 * 1024 * 1024);  // > 50MB
            size_check->close();
            print_line("EditorUpdater: File size: " + String::humanize_size(file_size) + (likely_installer ? " (likely installer)" : " (likely executable)"));
        }
        
        is_installer = is_installer || likely_installer;
        
        if (is_installer) {
            // For installers: Launch installer and let it handle installation
            print_line("EditorUpdater: Detected installer, launching with appropriate flags");
            
            // Try different installer flags in order of preference
            List<String> installer_flags[] = {
                {"/S", "/LAUNCH_EDITOR"},     // NSIS with custom editor launch flag
                {"/S", "/EDITOR"},            // NSIS with editor flag
                {"/S"},                       // Standard silent install
                {"/VERYSILENT", "/NORESTART", "/TASKS=desktopicon"}, // Inno Setup
                {}                            // No flags as last resort
            };
            
            bool launch_success = false;
            for (auto& flag_set : installer_flags) {
                List<String> current_args;
                for (const String& flag : flag_set) {
                    current_args.push_back(flag);
                }
                
                Error launch_err = OS::get_singleton()->create_process(downloaded_file_path, current_args);
                if (launch_err == OK) {
                    launch_success = true;
                    print_line("EditorUpdater: Installer launched with flags: " + String(", ").join(PackedStringArray()));
                    break;
                }
            }
            
            if (!launch_success) {
                print_line("EditorUpdater: All installer launch attempts failed");
                status_label->set_text("Failed to launch installer. Please run manually: " + downloaded_file_path);
                return;
            }
        } else {
            // WINDOWS AUTOMATED UPDATE: Enhanced launch for users
            print_line("EditorUpdater: WINDOWS: AUTO-LAUNCHING executable with explicit editor mode");
            print_line("EditorUpdater: WINDOWS: Downloaded file: " + downloaded_file_path);
            
            // ENHANCED APPROACH: Multiple launch strategies to ensure editor mode
            bool launch_success = false;
            
            // Strategy 1: Full arguments with project manager fallback
            List<String> full_args;
            full_args.push_back("--editor");           // Primary: Force editor mode
            full_args.push_back("--project-manager");  // Fallback: Project manager if no project
            
            print_line("EditorUpdater: WINDOWS: Trying Strategy 1 - Editor with project manager fallback");
            Error full_launch = OS::get_singleton()->create_process(downloaded_file_path, full_args);
            if (full_launch == OK) {
                launch_success = true;
                print_line("EditorUpdater: WINDOWS: SUCCESS - Strategy 1 worked");
            }
            
            // Strategy 2: Just --project-manager (most reliable for Windows)
            if (!launch_success) {
                List<String> pm_args;
                pm_args.push_back("--project-manager");
                
                print_line("EditorUpdater: WINDOWS: Trying Strategy 2 - Project manager only");
                Error pm_launch = OS::get_singleton()->create_process(downloaded_file_path, pm_args);
                if (pm_launch == OK) {
                    launch_success = true;
                    print_line("EditorUpdater: WINDOWS: SUCCESS - Strategy 2 worked (project manager)");
                }
            }
            
            // Strategy 3: Just --editor (simple approach)
            if (!launch_success) {
                List<String> editor_args;
                editor_args.push_back("--editor");
                
                print_line("EditorUpdater: WINDOWS: Trying Strategy 3 - Editor only");
                Error editor_launch = OS::get_singleton()->create_process(downloaded_file_path, editor_args);
                if (editor_launch == OK) {
                    launch_success = true;
                    print_line("EditorUpdater: WINDOWS: SUCCESS - Strategy 3 worked (editor only)");
                }
            }
            
            // Strategy 4: No arguments (let Godot decide)
            if (!launch_success) {
                List<String> no_args;
                
                print_line("EditorUpdater: WINDOWS: Trying Strategy 4 - No arguments (auto-detect)");
                Error auto_launch = OS::get_singleton()->create_process(downloaded_file_path, no_args);
                if (auto_launch == OK) {
                    launch_success = true;
                    print_line("EditorUpdater: WINDOWS: SUCCESS - Strategy 4 worked (auto-detect)");
                }
            }
            
            // Final result
            if (!launch_success) {
                print_line("EditorUpdater: WINDOWS: ERROR - All launch strategies failed");
                status_label->set_text("Update downloaded but failed to launch automatically.\nPlease run manually: " + downloaded_file_path + " --project-manager");
                return;
            } else {
                print_line("EditorUpdater: WINDOWS: SUCCESS - New version launched successfully");
            }
        }
        
        // CRITICAL FIX: Notify backend that update was installed before quitting
        _notify_backend_update_installed();
        
        // Give installer/new version time to start before quitting
        print_line("EditorUpdater: Waiting 2 seconds before quitting current instance...");
        OS::get_singleton()->delay_usec(2000 * 1000);
        
        print_line("EditorUpdater: Quitting current instance to complete update");
        get_tree()->quit();
        return;
    }
#endif

#ifdef MACOS_ENABLED
    {
        int exit_code = 0;
        String lower = downloaded_file_path.to_lower();

        auto open_app_and_quit = [&](const String &app_path) {
            // PRODUCTION AUTOMATED INSTALLATION - One-click for users
            _notify_backend_update_installed();
            
            print_line("EditorUpdater: 🚀 AUTO-LAUNCHING new Orca Engine version");
            print_line("EditorUpdater: New app path: " + app_path);
            
            // SAVE current project path from engine globals for preservation
            String current_project_path;
            
            // Get currently loaded project path from EditorNode if available
            if (EditorNode::get_singleton()) {
                // Try to get current project root - this should work reliably
                String project_root = OS::get_singleton()->get_executable_path().get_base_dir();
                
                // Navigate up to find project.godot
                String search_path = project_root;
                for (int i = 0; i < 5; i++) {  // Search up to 5 levels
                    String project_file = search_path.path_join("project.godot");
                    if (FileAccess::exists(project_file)) {
                        current_project_path = search_path;
                        break;
                    }
                    search_path = search_path.get_base_dir();
                }
            }
            
            // AUTOMATED LAUNCH: Use macOS 'open' command with arguments
            List<String> open_args;
            open_args.push_back("-a");  // Launch application
            open_args.push_back(app_path);  // Path to new Orca.app
            open_args.push_back("--args");  // Everything after this goes to the app
            open_args.push_back("--editor");  // Ensure editor mode
            
            if (!current_project_path.is_empty()) {
                open_args.push_back("--path");
                open_args.push_back(current_project_path);
                print_line("EditorUpdater: 📂 Preserving project: " + current_project_path);
            } else {
                print_line("EditorUpdater: 📋 Opening project manager (no specific project)");
            }
            
            // Execute the launch command
            Error launch_result = OS::get_singleton()->execute("/usr/bin/open", open_args, nullptr, &exit_code, true);
            
            if (launch_result == OK) {
                print_line("EditorUpdater: ✅ New version launched successfully!");
                print_line("EditorUpdater: User's project state will be preserved");
            } else {
                print_line("EditorUpdater: ⚠️ Launch command failed, trying fallback");
                
                // Simple fallback - just open the app
                List<String> simple_args;
                simple_args.push_back("-a");
                simple_args.push_back(app_path);
                OS::get_singleton()->execute("/usr/bin/open", simple_args, nullptr, &exit_code, true);
            }
            
            print_line("EditorUpdater: 🔄 Quitting current version to complete update");
            get_tree()->quit();
        };

        auto copy_to_applications = [&](const String &src_app) -> String {
            String dest_app = "/Applications/" + src_app.get_file();
            List<String> ditto_args;
            ditto_args.push_back(src_app);
            ditto_args.push_back(dest_app);
            OS::get_singleton()->execute("/usr/bin/ditto", ditto_args, nullptr, &exit_code, true);
            return dest_app;
        };

        if (lower.ends_with(".dmg")) {
            String mount_point = EditorPaths::get_singleton()->get_cache_dir().path_join("orca_update_mount");
            Ref<DirAccess> da = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
            if (da.is_valid()) {
                da->make_dir_recursive(mount_point);
            }
            List<String> args;
            args.push_back("attach");
            args.push_back("-nobrowse");
            args.push_back(downloaded_file_path);
            args.push_back("-mountpoint");
            args.push_back(mount_point);
            OS::get_singleton()->execute("/usr/bin/hdiutil", args, nullptr, &exit_code, true);

            String app_in_dmg;
            Ref<DirAccess> da2 = DirAccess::open(mount_point);
            if (da2.is_valid()) {
                da2->list_dir_begin();
                while (true) {
                    String f = da2->get_next();
                    if (f.is_empty()) break;
                    if (f.ends_with(".app")) {
                        app_in_dmg = mount_point.path_join(f);
                        break;
                    }
                }
                da2->list_dir_end();
            }
            if (!app_in_dmg.is_empty()) {
                String installed = copy_to_applications(app_in_dmg);
                List<String> detach_args; detach_args.push_back("detach"); detach_args.push_back(mount_point);
                OS::get_singleton()->execute("/usr/bin/hdiutil", detach_args, nullptr, &exit_code, true);
                open_app_and_quit(installed);
                return;
            }
            // Fallback: just open the DMG so user can drag-drop.
            _notify_backend_update_installed();
            OS::get_singleton()->shell_open("file://" + downloaded_file_path);
            get_tree()->quit();
            return;
        }

        if (lower.ends_with(".zip")) {
            String extract_dir = EditorPaths::get_singleton()->get_cache_dir().path_join("orca_update_extract");
            Ref<DirAccess> da3 = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
            if (da3.is_valid()) {
                da3->make_dir_recursive(extract_dir);
            }
            List<String> ex_args;
            ex_args.push_back("-x");
            ex_args.push_back("-k");
            ex_args.push_back(downloaded_file_path);
            ex_args.push_back(extract_dir);
            OS::get_singleton()->execute("/usr/bin/ditto", ex_args, nullptr, &exit_code, true);

            String app_found;
            Ref<DirAccess> da4 = DirAccess::open(extract_dir);
            if (da4.is_valid()) {
                da4->list_dir_begin();
                while (true) {
                    String f = da4->get_next();
                    if (f.is_empty()) break;
                    if (f.ends_with(".app")) { app_found = extract_dir.path_join(f); break; }
                }
                da4->list_dir_end();
            }
            if (!app_found.is_empty()) {
                String installed = copy_to_applications(app_found);
                open_app_and_quit(installed);
                return;
            }
            // Fallback: open the zip location.
            _notify_backend_update_installed();
            OS::get_singleton()->shell_open("file://" + downloaded_file_path);
            get_tree()->quit();
            return;
        }

        // If we downloaded an .app directly
        if (lower.ends_with(".app")) {
            String installed = copy_to_applications(downloaded_file_path);
            open_app_and_quit(installed);
            return;
        }

        // Unknown type: open file location as fallback.
        _notify_backend_update_installed();
        OS::get_singleton()->shell_open("file://" + downloaded_file_path);
        get_tree()->quit();
        return;
    }
#endif
}


