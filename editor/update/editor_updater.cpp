/**************************************************************************/
/*  editor_updater.cpp                                                     */
/**************************************************************************/

#include "editor_updater.h"

#include "core/io/dir_access.h"
#include "core/os/os.h"
#include "core/io/json.h"
#include "editor/file_system/editor_paths.h"
#include "editor/editor_node.h"
#include "editor/themes/editor_scale.h"
#include "scene/gui/box_container.h"

void EditorUpdater::_bind_methods() {
}

EditorUpdater::EditorUpdater() {
    set_title(TTR("Update Orca"));
    set_min_size(Size2(520, 140) * EDSCALE);

    VBoxContainer *vb = memnew(VBoxContainer);
    add_child(vb);

    status_label = memnew(Label);
    status_label->set_text(TTR("Checking for updates..."));
    vb->add_child(status_label);

    progress = memnew(ProgressBar);
    progress->set_min(0);
    progress->set_max(100);
    progress->set_step(0.1);
    progress->set_value(0);
    progress->set_h_size_flags(Control::SIZE_EXPAND_FILL);
    vb->add_child(progress);

    action_button = memnew(Button);
    action_button->set_text(TTR("Download"));
    action_button->connect("pressed", callable_mp(this, &EditorUpdater::_on_pressed));
    vb->add_child(action_button);

    http = memnew(HTTPRequest);
    add_child(http);
    http->connect("request_completed", callable_mp(this, &EditorUpdater::_on_request_completed));

    set_process(true);
}

void EditorUpdater::start_check() {
    stage = STAGE_CHECKING;
    status_label->set_text(TTR("Checking for updates..."));
    progress->set_indeterminate(true);

    if (feed_url.is_empty()) {
        status_label->set_text(TTR("No update feed configured."));
        stage = STAGE_ERROR;
        return;
    }

    Error err = http->request(feed_url);
    if (err != OK) {
        status_label->set_text(TTR("Failed to request update feed."));
        // Try GitHub releases if configured
        if (!owner_repo.is_empty()) {
            status_label->set_text(TTR("Checking GitHub releases..."));
            if (!http_release) {
                http_release = memnew(HTTPRequest);
                add_child(http_release);
                http_release->connect("request_completed", callable_mp(this, &EditorUpdater::_on_release_completed));
            }
            String api = "https://api.github.com/repos/" + owner_repo + "/releases/latest";
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
                status_label->set_text(TTR("Checking GitHub releases..."));
                if (!http_release) {
                    http_release = memnew(HTTPRequest);
                    add_child(http_release);
                    http_release->connect("request_completed", callable_mp(this, &EditorUpdater::_on_release_completed));
                }
                String api = "https://api.github.com/repos/" + owner_repo + "/releases/latest";
                PackedStringArray headers;
                headers.push_back("User-Agent: OrcaEditorUpdater/1.0");
                headers.push_back("Accept: application/vnd.github+json");
                Error e2 = http_release->request(api, headers);
                if (e2 == OK) {
                    return;
                }
            }
            status_label->set_text(TTR("Failed to check updates."));
            stage = STAGE_ERROR;
            return;
        }
        Ref<XMLParser> parser;
        parser.instantiate();
        if (parser->open_buffer(p_body) != OK) {
            status_label->set_text(TTR("Invalid appcast feed."));
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
            status_label->set_text(TTR("No update available."));
            progress->set_indeterminate(false);
            progress->set_value(0);
            stage = STAGE_IDLE;
            action_button->set_text(TTR("Close"));
            return;
        }
        download_url = found_url;
        latest_version = found_version;
        status_label->set_text(vformat(TTR("Update %s available. Downloading..."), latest_version));
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
        http->set_download_file(downloaded_file_path);
        http->set_use_threads(true);
        progress->set_indeterminate(false);
        progress->set_value(0);
        // Start actual download.
        http->request(download_url);
        return;
    }

    if (stage == STAGE_DOWNLOADING) {
        if (p_result != HTTPRequest::RESULT_SUCCESS || p_code != 200) {
            status_label->set_text(TTR("Download failed."));
            stage = STAGE_ERROR;
            return;
        }
        stage = STAGE_DOWNLOADED;
        status_label->set_text(TTR("Download complete. Ready to install."));
        action_button->set_text(TTR("Install and Restart"));
    }
}

void EditorUpdater::_on_release_completed(int p_result, int p_code, const PackedStringArray &p_headers, const PackedByteArray &p_body) {
    if (p_result != HTTPRequest::RESULT_SUCCESS || p_code != 200) {
        status_label->set_text(TTR("Failed to check GitHub releases."));
        stage = STAGE_ERROR;
        return;
    }
    String s = String::utf8((const char *)p_body.ptr(), p_body.size());
    Variant json_v = JSON::parse_string(s);
    if (json_v.get_type() != Variant::DICTIONARY) {
        status_label->set_text(TTR("Invalid releases JSON."));
        stage = STAGE_ERROR;
        return;
    }
    Dictionary d = json_v;
    String tag = d.get("tag_name", String());
    Array assets = d.get("assets", Array());
    String best_url;
    for (int i = 0; i < assets.size(); i++) {
        Dictionary a = assets[i];
        String name = a.get("name", String());
        String url = a.get("browser_download_url", String());
        if (name.is_empty() || url.is_empty()) {
            continue;
        }
        if (OS::get_singleton()->has_feature("windows")) {
            if (name.ends_with(".exe") || name.ends_with(".msi")) {
                best_url = url;
                break;
            }
        } else if (OS::get_singleton()->has_feature("macos")) {
            if (name.ends_with(".dmg") || name.ends_with(".zip")) {
                best_url = url;
                break;
            }
        } else {
            // Linux: pick .AppImage/.tar.* if present
            if (name.contains("AppImage") || name.ends_with(".tar.gz")) {
                best_url = url;
                break;
            }
        }
    }
    if (best_url.is_empty()) {
        status_label->set_text(TTR("No suitable asset found on latest release."));
        stage = STAGE_ERROR;
        return;
    }
    download_url = best_url;
    latest_version = tag;
    status_label->set_text(vformat(TTR("Update %s available. Downloading..."), latest_version));
    stage = STAGE_DOWNLOADING;
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
    http->set_download_file(downloaded_file_path);
    http->set_use_threads(true);
    progress->set_indeterminate(false);
    progress->set_value(0);
    http->request(download_url);
}

void EditorUpdater::_process(double p_delta) {
    if (stage == STAGE_DOWNLOADING) {
        int downloaded = http->get_downloaded_bytes();
        int total = http->get_body_size();

        switch (http->get_http_client_status()) {
            case HTTPClient::STATUS_RESOLVING: status_label->set_text(TTR("Resolving...")); break;
            case HTTPClient::STATUS_CONNECTING: status_label->set_text(TTR("Connecting...")); break;
            case HTTPClient::STATUS_REQUESTING: status_label->set_text(TTR("Requesting...")); break;
            case HTTPClient::STATUS_CONNECTED: status_label->set_text(TTR("Connected")); break;
            case HTTPClient::STATUS_BODY: {
                if (total > 0) {
                    double percent = (double)downloaded * 100.0 / (double)total;
                    if (percent < 0.0) percent = 0.0;
                    if (percent > 100.0) percent = 100.0;
                    progress->set_indeterminate(false);
                    progress->set_value(percent);
                    status_label->set_text(TTR("Downloading ") + String::humanize_size(downloaded) + "/" + String::humanize_size(total));
                } else if (downloaded >= 0) {
                    progress->set_indeterminate(true);
                    status_label->set_text(TTR("Downloading ") + String::humanize_size(downloaded));
                }
            } break;
            default: break;
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
        List<String> args;
        // Try common silent flags; if unsupported, installer will show UI.
        if (downloaded_file_path.to_lower().ends_with(".exe")) {
            args.push_back("/S");
        }
        OS::get_singleton()->create_process(downloaded_file_path, args);
        get_tree()->quit();
        return;
    }
#endif

#ifdef MACOS_ENABLED
    {
        int exit_code = 0;
        String lower = downloaded_file_path.to_lower();

        auto open_app_and_quit = [&](const String &app_path) {
            List<String> open_args;
            open_args.push_back("-a");
            open_args.push_back(app_path);
            OS::get_singleton()->execute("/usr/bin/open", open_args, nullptr, &exit_code, true);
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
        OS::get_singleton()->shell_open("file://" + downloaded_file_path);
        get_tree()->quit();
        return;
    }
#endif
}


