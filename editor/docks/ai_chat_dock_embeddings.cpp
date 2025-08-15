/**************************************************************************/
/*  ai_chat_dock_embeddings.cpp                                           */
/**************************************************************************/

#include "ai_chat_dock.h"

#include "core/config/project_settings.h"
#include "core/io/json.h"
#include "editor/file_system/editor_file_system.h"
#include "editor/settings/editor_settings.h"

// Helper: derive backend base URL from chat endpoint
static String _get_backend_base_url(const String &p_chat_endpoint) {
    // Expect .../chat; strip trailing "/chat" if present
    if (p_chat_endpoint.ends_with("/chat")) {
        return p_chat_endpoint.substr(0, p_chat_endpoint.length() - 5);
    }
    return p_chat_endpoint;
}

String AIChatDock::_get_embed_base_url() {
    // Resolve base URL with priority: EditorSettings -> env AI_CHAT_CLOUD_URL -> localhost
    String base_url;
    if (EditorSettings::get_singleton() && EditorSettings::get_singleton()->has_setting("ai_chat/base_url")) {
        base_url = EditorSettings::get_singleton()->get_setting("ai_chat/base_url");
    }
    if (base_url.is_empty()) {
        base_url = OS::get_singleton()->get_environment("AI_CHAT_CLOUD_URL");
    }
    if (base_url.is_empty()) {
        base_url = "http://127.0.0.1:8000";
    }
    // Normalize: strip trailing slash
    if (base_url.ends_with("/")) {
        base_url = base_url.substr(0, base_url.length() - 1);
    }
    return base_url;
}

void AIChatDock::_initialize_embedding_system() {
    if (embedding_system_initialized) {
        return;
    }

    // Create HTTPRequest for embedding ops
    if (!embedding_request) {
        embedding_request = memnew(HTTPRequest);
        add_child(embedding_request);
        embedding_request->connect("request_completed", callable_mp(this, &AIChatDock::_on_embedding_request_completed));
    }

    // Connect to editor file system signals for automatic reindexing
    if (EditorFileSystem::get_singleton()) {
        EditorFileSystem::get_singleton()->connect("filesystem_changed", callable_mp(this, &AIChatDock::_on_filesystem_changed));
        EditorFileSystem::get_singleton()->connect("sources_changed", callable_mp(this, &AIChatDock::_on_sources_changed));
        print_line("AI Chat: 🔗 Connected to EditorFileSystem change signals (filesystem_changed, sources_changed)");
    }

    embedding_system_initialized = true;
    // Kick off initial indexing
    call_deferred("_perform_initial_indexing");
}

void AIChatDock::_perform_initial_indexing() {
    if (initial_indexing_done || embedding_request_busy) {
        return;
    }
    Dictionary payload;
    payload["project_root"] = _get_project_root_path();
    payload["force_reindex"] = false;
    // Optional: pass project_id if you have a stable id; backend can derive one otherwise
    // payload["project_id"] = String();
    _set_embedding_status("Indexing project", true);
    _send_embedding_request("index_project", payload);
}

void AIChatDock::_on_filesystem_changed() {
    print_line("AI Chat: 📁 filesystem_changed signal received");
    // Debounced: simply request a lightweight re-index of project
    if (!embedding_system_initialized) {
        print_line("AI Chat: 🚫 Skipping indexing - system not initialized");
        return;
    }
    if (!_is_user_authenticated()) {
        print_line("AI Chat: 🚫 Skipping indexing - user not authenticated");
        return;
    }
    if (embedding_request_busy) {
        print_line("AI Chat: ⏳ Skipping indexing - previous embedding request still in flight");
        return;
    }

    Dictionary payload;
    payload["project_root"] = _get_project_root_path();
    payload["force_reindex"] = false; // backend is incremental
    _set_embedding_status("Indexing changes", true);
    _send_embedding_request("index_project", payload);
}

void AIChatDock::_on_sources_changed(bool p_exist) {
    print_line("AI Chat: 📝 sources_changed signal received (exist=" + String(p_exist ? "true" : "false") + ")");
    // Sources changed; treat same as filesystem_changed for indexing purposes
    _on_filesystem_changed();
}

void AIChatDock::_update_file_embedding(const String &p_file_path) {
    if (!embedding_system_initialized || !_is_user_authenticated() || embedding_request_busy) {
        return;
    }
    Dictionary payload;
    payload["project_root"] = _get_project_root_path();
    payload["file_path"] = p_file_path;
    _set_embedding_status("Updating file", true);
    _send_embedding_request("update_file", payload);
}

void AIChatDock::_remove_file_embedding(const String &p_file_path) {
    if (!embedding_system_initialized || !_is_user_authenticated() || embedding_request_busy) {
        return;
    }
    Dictionary payload;
    payload["project_root"] = _get_project_root_path();
    payload["file_path"] = p_file_path;
    _set_embedding_status("Removing file", true);
    _send_embedding_request("remove_file", payload);
}

void AIChatDock::_send_embedding_request(const String &p_action, const Dictionary &p_data) {
    if (!embedding_request) {
        return;
    }

    String base = _get_embed_base_url();
    // Always use /embed; action is provided in body
    String url = base + "/embed";

    PackedStringArray headers;
    headers.push_back("Content-Type: application/json");
    String machine_id = get_machine_id();
    if (machine_id != String()) {
        headers.push_back("X-Machine-ID: " + machine_id);
    }
    if (auth_token != String()) {
        headers.push_back("Authorization: Bearer " + auth_token);
    }
    if (current_user_id != String()) {
        headers.push_back("X-User-ID: " + current_user_id);
    }

    // Include action and machine/project context in body
    Dictionary body = p_data;
    body["action"] = p_action;
    if (!body.has("project_root")) {
        body["project_root"] = _get_project_root_path();
    }
    if (!body.has("machine_id")) {
        body["machine_id"] = machine_id;
    }

    String json = JSON::stringify(body);
    print_line(vformat("AI Chat: 📤 Embedding request %s -> %s", p_action, url));
    print_line(vformat("AI Chat: 📁 project_root=%s", body.get("project_root", "")));
    embedding_request_busy = true;
    Error err = embedding_request->request(url, headers, HTTPClient::METHOD_POST, json);
    if (err != OK) {
        print_line("AI Chat: ❌ Embedding request failed to start: " + itos(err));
        embedding_request_busy = false;
    }
}

void AIChatDock::_on_embedding_request_completed(int p_result, int p_code, const PackedStringArray &p_headers, const PackedByteArray &p_body) {
    String body_str = String::utf8((const char *)p_body.ptr(), p_body.size());

    if (p_code < 200 || p_code >= 300) {
        print_line("AI Chat: ❌ Embedding API error: HTTP " + itos(p_code) + " body: " + body_str);
        _set_embedding_status("Indexing failed", false);
        embedding_request_busy = false;
        return;
    }

    JSON json;
    Error perr = json.parse(body_str);
    if (perr != OK || json.get_data().get_type() != Variant::DICTIONARY) {
        print_line("AI Chat: ⚠️ Unexpected embedding response: " + body_str);
        return;
    }

    Dictionary resp = json.get_data();
    bool success = resp.get("success", false);
    if (!success) {
        print_line("AI Chat: ⚠️ Embedding operation unsuccessful: " + body_str);
        _set_embedding_status("Indexing failed", false);
        embedding_request_busy = false;
        return;
    }

    String action = resp.get("action", String());
    if (action == "index_project") {
        initial_indexing_done = true;
        int indexed = 0;
        int skipped = 0;
        if (resp.has("stats")) {
            Dictionary stats = resp["stats"];
            indexed = int(stats.get("indexed", 0));
            skipped = int(stats.get("skipped", 0));
        }
        print_line(vformat("AI Chat: ✅ Project indexed. indexed=%d skipped=%d", indexed, skipped));
        _set_embedding_status(vformat("Indexed (new %d, skipped %d)", indexed, skipped), false);
    } else if (action == "update_file") {
        print_line("AI Chat: ✅ File embedding updated");
        _set_embedding_status("File updated", false);
    } else if (action == "remove_file") {
        print_line("AI Chat: ✅ File embedding removed");
        _set_embedding_status("File removed", false);
    } else if (action == "search" || action == "search_with_graph") {
        // Suggestion results
        // This path is used by _suggest_relevant_files when querying suggestions
        Array similar = Array();
        if (resp.has("results")) {
            Dictionary results = resp["results"];
            if (results.has("similar_files")) {
                similar = results["similar_files"];
            }
        }
        int count = similar.size();
        print_line(vformat("AI Chat: 🔎 Suggested relevant files: %d", count));
    }
    embedding_request_busy = false;
}

bool AIChatDock::_should_index_file(const String &p_file_path) {
    // Basic client-side filter; backend has authoritative filter
    String ext = p_file_path.get_extension().to_lower();
    static const HashSet<String> allowed_ext = {
        "gd", "cs", "js", "tscn", "scn", "tres", "res", "json", "cfg", "md", "txt",
        "png", "jpg", "jpeg", "gif", "bmp", "webp", "svg", "tga", "exr",
        "wav", "ogg", "mp3", "aac", "flac",
        "mp4", "mov", "avi", "webm"
    };
    return allowed_ext.has(ext);
}

String AIChatDock::_get_project_root_path() {
    String res = "res://";
    return ProjectSettings::get_singleton()->globalize_path(res);
}

void AIChatDock::_set_embedding_status(const String &p_text, bool p_busy) {
    if (!embedding_status_label) return;
    embedding_status_base = p_text;
    embedding_status_label->set_text(embedding_status_base);
    embedding_in_progress = p_busy;
    if (p_busy) {
        if (!embedding_status_timer) {
            embedding_status_timer = memnew(Timer);
            add_child(embedding_status_timer);
            embedding_status_timer->connect("timeout", callable_mp(this, &AIChatDock::_on_embedding_status_tick));
            embedding_status_timer->set_wait_time(0.5);
            embedding_status_timer->set_one_shot(false);
        }
        embedding_status_dots = 0;
        if (embedding_status_timer && !embedding_status_timer->is_stopped()) embedding_status_timer->stop();
        embedding_status_timer->start();
    } else {
        if (embedding_status_timer) embedding_status_timer->stop();
    }
}

void AIChatDock::_on_embedding_status_tick() {
    if (!embedding_in_progress || !embedding_status_label) return;
    embedding_status_dots = (embedding_status_dots + 1) % 4;
    String dots;
    for (int i = 0; i < embedding_status_dots; i++) dots += ".";
    embedding_status_label->set_text(embedding_status_base + dots);
}

void AIChatDock::_suggest_relevant_files(const String &p_query) {
    if (!_is_user_authenticated()) {
        return;
    }
    Dictionary payload;
    payload["project_root"] = _get_project_root_path();
    payload["query"] = p_query;
    payload["k"] = 5;
    payload["include_graph"] = true;
    _send_embedding_request("search", payload);
}

void AIChatDock::_auto_attach_relevant_context() {
    // Heuristic: suggest based on last user message
    Vector<ChatMessage> &msgs = _get_current_chat_history();
    for (int i = msgs.size() - 1; i >= 0; i--) {
        if (msgs[i].role == "user" && !msgs[i].content.is_empty()) {
            _suggest_relevant_files(msgs[i].content);
            break;
        }
    }
}


