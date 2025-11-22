/**************************************************************************/
/*  template_selector_dialog.cpp                                         */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             ORCA ENGINE                                */
/**************************************************************************/

#include "template_selector_dialog.h"

#include "core/io/file_access.h"
#include "core/io/json.h"
#include "core/os/os.h"
#include "scene/gui/box_container.h"
#include "scene/gui/item_list.h"
#include "scene/gui/line_edit.h"
#include "scene/gui/scroll_container.h"
#include "scene/gui/separator.h"
#include "scene/gui/rich_text_label.h"
#include "scene/gui/texture_rect.h"
#include "scene/main/http_request.h"
#include "scene/resources/image_texture.h"
#include "servers/text_server.h"
#include "core/io/image.h"
#include "editor/themes/editor_scale.h"
#include "editor/settings/editor_settings.h"

void TemplateSelectorDialog::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_on_search_changed", "text"), &TemplateSelectorDialog::_on_search_changed);
	ClassDB::bind_method(D_METHOD("_on_template_selected", "index"), &TemplateSelectorDialog::_on_template_selected);
	ClassDB::bind_method(D_METHOD("_on_template_double_clicked", "index"), &TemplateSelectorDialog::_on_template_double_clicked);
	ClassDB::bind_method(D_METHOD("_on_http_request_completed", "result", "response_code", "headers", "body"), &TemplateSelectorDialog::_on_http_request_completed);
	ClassDB::bind_method(D_METHOD("_on_readme_request_completed", "result", "response_code", "headers", "body"), &TemplateSelectorDialog::_on_readme_request_completed);
	ClassDB::bind_method(D_METHOD("_on_image_request_completed", "result", "response_code", "headers", "body"), &TemplateSelectorDialog::_on_image_request_completed);
}

TemplateSelectorDialog::TemplateSelectorDialog() {
	set_title(TTR("Start from Template"));
	set_min_size(Size2(900, 700) * EDSCALE);
	
	main_vbox = memnew(VBoxContainer);
	main_vbox->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	add_child(main_vbox);
	
	// Header section
	VBoxContainer *header_vbox = memnew(VBoxContainer);
	main_vbox->add_child(header_vbox);
	
	Label *title_label = memnew(Label);
	title_label->set_text(TTR("Choose a Template"));
	title_label->add_theme_font_size_override("font_size", 20);
	header_vbox->add_child(title_label);
	
	Label *subtitle_label = memnew(Label);
	subtitle_label->set_text(TTR("Select a template to start your project"));
	subtitle_label->add_theme_color_override("font_color", Color(0.7, 0.7, 0.7));
	header_vbox->add_child(subtitle_label);
	
	main_vbox->add_child(memnew(HSeparator));
	
	// Search box
	HBoxContainer *search_hbox = memnew(HBoxContainer);
	search_hbox->add_theme_constant_override("separation", 8);
	main_vbox->add_child(search_hbox);
	
	Label *search_label = memnew(Label);
	search_label->set_text(TTR("Search:"));
	search_hbox->add_child(search_label);
	
	search_box = memnew(LineEdit);
	search_box->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	search_box->set_placeholder(TTR("Search templates by name..."));
	search_box->set_clear_button_enabled(true);
	search_box->connect(SceneStringName(text_changed), callable_mp(this, &TemplateSelectorDialog::_on_search_changed));
	search_hbox->add_child(search_box);
	
	main_vbox->add_child(memnew(HSeparator));
	
	// Content area with list and description
	HBoxContainer *content_hbox = memnew(HBoxContainer);
	content_hbox->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	content_hbox->add_theme_constant_override("separation", 12);
	main_vbox->add_child(content_hbox);
	
	// Template list (left side)
	VBoxContainer *list_vbox = memnew(VBoxContainer);
	list_vbox->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	content_hbox->add_child(list_vbox);
	
	Label *list_label = memnew(Label);
	list_label->set_text(TTR("Available Templates:"));
	list_vbox->add_child(list_label);
	
	template_item_list = memnew(ItemList);
	template_item_list->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	template_item_list->set_select_mode(ItemList::SELECT_SINGLE);
	template_item_list->connect(SceneStringName(item_selected), callable_mp(this, &TemplateSelectorDialog::_on_template_selected));
	template_item_list->connect("item_activated", callable_mp(this, &TemplateSelectorDialog::_on_template_double_clicked));
	list_vbox->add_child(template_item_list);
	
	// Description panel (right side)
	VBoxContainer *desc_vbox = memnew(VBoxContainer);
	desc_vbox->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	desc_vbox->set_custom_minimum_size(Size2(360, 0) * EDSCALE);
	content_hbox->add_child(desc_vbox);
	
	Label *desc_label = memnew(Label);
	desc_label->set_text(TTR("Template Details:"));
	desc_vbox->add_child(desc_label);
	
	// Scroll container for description
	description_scroll = memnew(ScrollContainer);
	description_scroll->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	description_scroll->set_custom_minimum_size(Size2(0, 240) * EDSCALE);
	desc_vbox->add_child(description_scroll);
	
	VBoxContainer *desc_content = memnew(VBoxContainer);
	desc_content->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	description_scroll->add_child(desc_content);
	description_scroll->set_horizontal_scroll_mode(ScrollContainer::SCROLL_MODE_DISABLED);
	
	description_label = memnew(RichTextLabel);
	description_label->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	description_label->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	description_label->set_custom_minimum_size(Size2(0, 220) * EDSCALE);
	description_label->set_autowrap_mode(TextServer::AUTOWRAP_WORD);
	description_label->set_use_bbcode(false);
	description_label->set_fit_content(false);
	description_label->set_scroll_active(true);
	description_label->set_scroll_follow(false);
	description_label->set_selection_enabled(true);
	description_label->add_text(TTR("Select a template to view details"));
	desc_content->add_child(description_label);
	
	// Image container
	image_container = memnew(VBoxContainer);
	image_container->set_visible(false);
	desc_content->add_child(image_container);
	
	Label *images_label = memnew(Label);
	images_label->set_text(TTR("Preview Images:"));
	images_label->set_visible(false);
	image_container->add_child(images_label);
	
	// Status label
	status_label = memnew(Label);
	status_label->set_text(TTR("Loading templates..."));
	status_label->add_theme_color_override("font_color", Color(0.7, 0.7, 0.7));
	main_vbox->add_child(status_label);
	
	// HTTP request for fetching templates
	http_request = memnew(HTTPRequest);
	add_child(http_request);
	http_request->connect("request_completed", callable_mp(this, &TemplateSelectorDialog::_on_http_request_completed));
	
	// HTTP request for fetching README
	readme_request = memnew(HTTPRequest);
	add_child(readme_request);
	readme_request->connect("request_completed", callable_mp(this, &TemplateSelectorDialog::_on_readme_request_completed));
	
	// Start loading templates
	_load_templates();
}

TemplateSelectorDialog::~TemplateSelectorDialog() {
}

String TemplateSelectorDialog::_get_backend_url() {
	String base_url;
	String is_dev = OS::get_singleton()->get_environment("IS_DEV");
	if (is_dev.is_empty()) {
		is_dev = OS::get_singleton()->get_environment("DEV_MODE");
	}
	if (!is_dev.is_empty() && is_dev.to_lower() == "true") {
		base_url = "http://127.0.0.1:5050";
	} else {
		base_url = "https://api.orcaengine.ai";
	}
	
	// Allow override via editor settings or environment variable
	if (EditorSettings::get_singleton() && EditorSettings::get_singleton()->has_setting("ai_chat/base_url")) {
		String override_url = EditorSettings::get_singleton()->get_setting("ai_chat/base_url");
		if (!override_url.is_empty()) {
			base_url = override_url;
		}
	} else if (!OS::get_singleton()->get_environment("AI_CHAT_CLOUD_URL").is_empty()) {
		base_url = OS::get_singleton()->get_environment("AI_CHAT_CLOUD_URL");
	}
	
	if (base_url.ends_with("/")) {
		base_url = base_url.substr(0, base_url.length() - 1);
	}
	
	return base_url;
}

void TemplateSelectorDialog::_load_templates() {
	is_loading = true;
	status_label->set_text(TTR("Loading templates from server..."));
	
	// Try to fetch from backend first
	_load_templates_from_backend();
}

void TemplateSelectorDialog::_load_templates_from_backend() {
	String url = _get_backend_url() + "/templates";
	
	Error err = http_request->request(url);
	if (err != OK) {
		print_line("Failed to start HTTP request for templates, falling back to local file");
		_load_templates_from_local();
	}
}

void TemplateSelectorDialog::_on_http_request_completed(int p_result, int p_response_code, const PackedStringArray &p_headers, const PackedByteArray &p_body) {
	is_loading = false;
	
	if (p_result == HTTPRequest::RESULT_SUCCESS && p_response_code == 200) {
		String response_text = String::utf8((const char *)p_body.ptr(), p_body.size());
		_parse_templates_json(response_text);
		status_label->set_text(vformat(TTR("Loaded %d templates"), templates.size()));
	} else {
		// Fallback to local file if backend request fails
		print_line("Failed to fetch templates from backend (code: " + itos(p_response_code) + "), falling back to local file");
		_load_templates_from_local();
	}
}

void TemplateSelectorDialog::_parse_templates_json(const String &p_json_text) {
	JSON json;
	Error err = json.parse(p_json_text);
	if (err != OK) {
		ERR_PRINT("Failed to parse templates JSON: " + json.get_error_message());
		_load_templates_from_local();
		return;
	}
	
	Variant data = json.get_data();
	if (data.get_type() == Variant::ARRAY) {
		templates = data;
		filtered_templates = templates;
		_populate_template_list();
	} else if (data.get_type() == Variant::DICTIONARY) {
		Dictionary dict = data;
		if (dict.has("templates")) {
			templates = dict["templates"];
			filtered_templates = templates;
			_populate_template_list();
		} else {
			_load_templates_from_local();
		}
	} else {
		_load_templates_from_local();
	}
}

void TemplateSelectorDialog::_load_templates_from_local() {
	is_loading = false;
	String json_path = OS::get_singleton()->get_executable_path().get_base_dir().path_join("backend").path_join("base_game_v2.json");
	
	// Try alternative paths
	if (!FileAccess::exists(json_path)) {
		json_path = "res://backend/base_game_v2.json";
	}
	if (!FileAccess::exists(json_path)) {
		json_path = "../backend/base_game_v2.json";
	}
	
	Ref<FileAccess> file = FileAccess::open(json_path, FileAccess::READ);
	if (file.is_null()) {
		ERR_PRINT("Could not open base_game_v2.json at: " + json_path);
		status_label->set_text(TTR("Failed to load templates. Please check your connection or try again later."));
		status_label->add_theme_color_override("font_color", Color(1.0, 0.5, 0.5));
		return;
	}
	
	String json_text = file->get_as_text();
	file->close();
	
	_parse_templates_json(json_text);
	status_label->set_text(vformat(TTR("Loaded %d templates from local file"), templates.size()));
}

void TemplateSelectorDialog::_on_search_changed(const String &p_text) {
	filtered_templates.clear();
	
	String search_term = p_text.to_lower();
	
	for (int i = 0; i < templates.size(); i++) {
		Dictionary template_info = templates[i];
		String name = template_info.get("name", "");
		
		if (search_term.is_empty() || name.to_lower().contains(search_term)) {
			filtered_templates.append(template_info);
		}
	}
	
	_populate_template_list();
}

void TemplateSelectorDialog::_populate_template_list() {
	template_item_list->clear();
	
	for (int i = 0; i < filtered_templates.size(); i++) {
		Dictionary template_info = filtered_templates[i];
		String name = template_info.get("name", "");
		
		// Display just the name - cleaner UI
		template_item_list->add_item(name);
	}
	
	if (filtered_templates.size() == 0) {
		template_item_list->add_item(TTR("No templates found"));
		template_item_list->set_item_disabled(0, true);
	}
}

void TemplateSelectorDialog::_on_template_selected(int p_index) {
	if (p_index < 0 || p_index >= filtered_templates.size()) {
		return;
	}
	
	Dictionary template_info = filtered_templates[p_index];
	selected_template_id = template_info.get("id", "");
	_update_description();
}

void TemplateSelectorDialog::_update_description() {
	if (selected_template_id.is_empty()) {
		description_label->clear();
		description_label->add_text(TTR("Select a template to view details"));
		description_label->queue_redraw();
		image_container->set_visible(false);
		return;
	}
	
	Dictionary template_info;
	for (int i = 0; i < templates.size(); i++) {
		Dictionary t = templates[i];
		if (t.get("id", "") == selected_template_id) {
			template_info = t;
			break;
		}
	}
	
	if (template_info.is_empty()) {
		description_label->clear();
		description_label->add_text(TTR("Template details not available"));
		description_label->queue_redraw();
		image_container->set_visible(false);
		return;
	}
	
	String name = template_info.get("name", "");
	String license = template_info.get("license", "");
	String repo_url = template_info.get("repo_url", "");
	Variant subdir_variant = template_info.get("subdir", "");
	String subdir = "";
	bool has_subdir = false;
	if (subdir_variant.get_type() != Variant::NIL) {
		String subdir_str = String(subdir_variant);
		if (!subdir_str.is_empty() && subdir_str != "null" && subdir_str != "<null>" && subdir_str != "None") {
			subdir = subdir_str;
			has_subdir = true;
		}
	}
	
	String description = name + "\n\n";
	
	if (!license.is_empty()) {
		description += "License: " + license + "\n\n";
	}
	
	if (!repo_url.is_empty()) {
		description += "Repository: " + repo_url + "\n";
	}
	
	if (has_subdir) {
		description += "Subdirectory: " + subdir + "\n\n";
	} else {
		description += "\n";
	}
	
	// Store base description
	base_description = description;
	description_label->clear();
	description_label->add_text(description);
	description_label->queue_redraw();
	
	// Cancel any pending image requests
	for (int i = image_requests.size() - 1; i >= 0; i--) {
		HTTPRequest *req = Object::cast_to<HTTPRequest>(image_requests[i]);
		if (req) {
			req->cancel_request();
			image_request_map.erase(req);
			image_requests.remove_at(i);
			req->queue_free();
		}
	}
	
	// Clear previous images
	for (int i = image_container->get_child_count() - 1; i >= 0; i--) {
		Node *child = image_container->get_child(i);
		if (child->get_class() == "TextureRect") {
			child->queue_free();
		}
	}
	image_urls.clear();
	image_container->set_visible(false);
	
	// Fetch README if repo URL is available
	if (!repo_url.is_empty()) {
		_fetch_readme(repo_url);
	}
}

String TemplateSelectorDialog::_extract_github_info(const String &p_repo_url, String &r_owner, String &r_repo) {
	// Extract owner/repo from GitHub URL
	// Supports: https://github.com/owner/repo.git, https://github.com/owner/repo, git@github.com:owner/repo.git
	String url = p_repo_url.strip_edges();
	
	if (url.contains("github.com")) {
		String path;
		if (url.begins_with("https://github.com/")) {
			path = url.substr(19); // Remove "https://github.com/"
		} else if (url.begins_with("http://github.com/")) {
			path = url.substr(18); // Remove "http://github.com/"
		} else if (url.begins_with("git@github.com:")) {
			path = url.substr(15); // Remove "git@github.com:"
		} else {
			return "";
		}
		
		// Remove .git suffix if present
		if (path.ends_with(".git")) {
			path = path.substr(0, path.length() - 4);
		}
		
		// Extract owner and repo
		PackedStringArray parts = path.split("/");
		if (parts.size() >= 2) {
			r_owner = parts[0];
			r_repo = parts[1];
			return r_owner + "/" + r_repo;
		}
	}
	
	return "";
}

void TemplateSelectorDialog::_fetch_readme(const String &p_repo_url) {
	String owner, repo;
	String github_path = _extract_github_info(p_repo_url, owner, repo);
	
	if (github_path.is_empty()) {
		print_line("DEBUG: Not a GitHub repo, skipping README fetch");
		return; // Not a GitHub repo or couldn't parse
	}
	
	// Get subdir if available
	Dictionary template_info;
	for (int i = 0; i < templates.size(); i++) {
		Dictionary t = templates[i];
		if (t.get("id", "") == selected_template_id) {
			template_info = t;
			break;
		}
	}
	
	Variant subdir_variant = template_info.get("subdir", "");
	String base_path = "";
	if (subdir_variant.get_type() != Variant::NIL) {
		String subdir = String(subdir_variant);
		if (!subdir.is_empty() && subdir != "null" && subdir != "<null>" && subdir != "None" && subdir != "nil") {
			base_path = subdir + "/";
		}
	}
	
	print_line("DEBUG: GitHub path: " + github_path);
	print_line("DEBUG: Base path (subdir): " + base_path);
	
	// Try to fetch README from main branch
	String readme_url = "https://raw.githubusercontent.com/" + github_path + "/main/" + base_path + "README.md";
	current_readme_url = readme_url;
	
	print_line("DEBUG: Fetching README from: " + readme_url);
	
	Error err = readme_request->request(readme_url);
	if (err != OK) {
		print_line("DEBUG: Failed to start README request, trying master branch");
		// Try with master branch
		readme_url = "https://raw.githubusercontent.com/" + github_path + "/master/" + base_path + "README.md";
		current_readme_url = readme_url;
		readme_request->request(readme_url);
	}
}

void TemplateSelectorDialog::_on_readme_request_completed(int p_result, int p_response_code, const PackedStringArray &p_headers, const PackedByteArray &p_body) {
	print_line("DEBUG: README request completed - Result: " + itos(p_result) + ", Code: " + itos(p_response_code));
	
	// Check if this README is still for the currently selected template
	// (user might have selected a different template while this was loading)
	String expected_repo_url;
	Dictionary template_info;
	for (int i = 0; i < templates.size(); i++) {
		Dictionary t = templates[i];
		if (t.get("id", "") == selected_template_id) {
			template_info = t;
			expected_repo_url = t.get("repo_url", "");
			break;
		}
	}
	
	// Verify this README matches the current selection
	if (template_info.is_empty()) {
		print_line("DEBUG: Template info is empty, ignoring README");
		return; // User selected a different template, ignore this result
	}
	
	if (p_result == HTTPRequest::RESULT_SUCCESS && p_response_code == 200) {
		print_line("DEBUG: README fetched successfully, size: " + itos(p_body.size()));
		String readme_content = String::utf8((const char *)p_body.ptr(), p_body.size());
		
		// Rebuild description from base + README (don't append to existing)
		String full_description = base_description;
		full_description += "\n--- README ---\n\n";
		
		// Convert markdown to plain text and fix line endings
		String processed_readme = readme_content;
		
		// Normalize line endings (convert \r\n to \n, \r to \n)
		processed_readme = processed_readme.replace("\r\n", "\n");
		processed_readme = processed_readme.replace("\r", "\n");
		
		// Remove markdown headers (keep the text, just remove the # symbols)
		processed_readme = processed_readme.replace("### ", "");
		processed_readme = processed_readme.replace("## ", "");
		processed_readme = processed_readme.replace("# ", "");
		
		// Remove markdown bold/italic (keep the text)
		processed_readme = processed_readme.replace("**", "");
		processed_readme = processed_readme.replace("*", "");
		
		// Remove markdown links but keep text: [text](url) -> text
		int link_start = 0;
		while ((link_start = processed_readme.find("[", link_start)) != -1) {
			int link_end = processed_readme.find("]", link_start);
			int url_start = processed_readme.find("(", link_end);
			int url_end = processed_readme.find(")", url_start);
			if (link_end != -1 && url_start != -1 && url_end != -1 && url_start == link_end + 1) {
				String link_text = processed_readme.substr(link_start + 1, link_end - link_start - 1);
				processed_readme = processed_readme.substr(0, link_start) + link_text + processed_readme.substr(url_end + 1);
				link_start += link_text.length();
			} else {
				link_start++;
			}
		}
		
		// Remove markdown image syntax: ![alt](url) -> (remove entirely for now)
		int img_start = 0;
		while ((img_start = processed_readme.find("![", img_start)) != -1) {
			int img_end = processed_readme.find(")", img_start);
			if (img_end != -1) {
				processed_readme = processed_readme.substr(0, img_start) + processed_readme.substr(img_end + 1);
			} else {
				break;
			}
		}
		
		// Limit README length to avoid overwhelming the UI
		if (processed_readme.length() > 2000) {
			processed_readme = processed_readme.substr(0, 2000) + "\n\n... (truncated)";
		}
		
		// Ensure proper text formatting for RichTextLabel
		full_description += processed_readme;
		
		print_line("DEBUG: Setting README content, length: " + itos(full_description.length()));
		
		// Set text with BBCode disabled to avoid formatting issues with README content
		description_label->clear();
		description_label->push_paragraph(HORIZONTAL_ALIGNMENT_LEFT);
		description_label->add_text(full_description);
		description_label->pop();
		description_label->queue_redraw();
		
		// Parse and fetch images
		String repo_url = template_info.get("repo_url", "");
		_parse_readme_for_images(readme_content, repo_url);
	} else {
		// Try alternative README paths or branches
		String owner, repo;
		String github_path = _extract_github_info(current_readme_url, owner, repo);
		if (!github_path.is_empty() && current_readme_url.contains("/main/")) {
			// Try master branch
			String master_url = current_readme_url.replace("/main/", "/master/");
			current_readme_url = master_url;
			readme_request->request(master_url);
		}
	}
}

void TemplateSelectorDialog::_parse_readme_for_images(const String &p_readme_content, const String &p_repo_url) {
	String owner, repo;
	String github_path = _extract_github_info(p_repo_url, owner, repo);
	if (github_path.is_empty()) {
		return;
	}
	
	// Find markdown image syntax: ![alt](path) using simple string parsing
	// More reliable than regex for this use case
	int pos = 0;
	int image_count = 0;
	
	while (pos < p_readme_content.length() && image_count < 3) {
		// Find ![ pattern
		int img_start = p_readme_content.find("![", pos);
		if (img_start == -1) {
			break;
		}
		
		// Find opening parenthesis after !
		int paren_start = p_readme_content.find("(", img_start);
		if (paren_start == -1 || paren_start - img_start > 100) {
			pos = img_start + 1;
			continue;
		}
		
		// Find closing parenthesis
		int paren_end = p_readme_content.find(")", paren_start);
		if (paren_end == -1) {
			pos = paren_start + 1;
			continue;
		}
		
		// Extract image path
		String image_path = p_readme_content.substr(paren_start + 1, paren_end - paren_start - 1);
		
		// Convert relative paths to GitHub raw URLs
		String image_url;
		if (image_path.begins_with("http")) {
			image_url = image_path; // Already absolute
		} else {
				// Relative path - construct GitHub raw URL
				Variant subdir_variant;
				Dictionary template_info;
				for (int j = 0; j < templates.size(); j++) {
					Dictionary t = templates[j];
					if (t.get("id", "") == selected_template_id) {
						template_info = t;
						break;
					}
				}
				subdir_variant = template_info.get("subdir", "");
				String base_path = "";
				if (subdir_variant.get_type() != Variant::NIL) {
					String subdir = String(subdir_variant);
					if (!subdir.is_empty() && subdir != "null" && subdir != "<null>" && subdir != "None") {
						base_path = subdir + "/";
					}
				}
			
			// Remove leading ./ if present
			if (image_path.begins_with("./")) {
				image_path = image_path.substr(2);
			}
			
			image_url = "https://raw.githubusercontent.com/" + github_path + "/main/" + base_path + image_path;
		}
		
		image_urls.push_back(image_url);
		_load_image(image_url, image_count);
		image_count++;
		
		pos = paren_end + 1;
	}
	
	if (image_count > 0) {
		image_container->set_visible(true);
		if (image_container->get_child_count() > 0) {
			Control *label = Object::cast_to<Control>(image_container->get_child(0));
			if (label) {
				label->set_visible(true); // Show "Preview Images:" label
			}
		}
	}
}

void TemplateSelectorDialog::_load_image(const String &p_image_url, int p_index) {
	// Create HTTPRequest for this image
	HTTPRequest *img_request = memnew(HTTPRequest);
	img_request->set_meta("image_index", p_index);
	img_request->set_meta("image_url", p_image_url);
	add_child(img_request);
	image_request_map[img_request] = p_index;
	image_requests.push_back(img_request);
	img_request->connect("request_completed", callable_mp(this, &TemplateSelectorDialog::_on_image_request_completed));
	
	Error err = img_request->request(p_image_url);
	if (err != OK) {
		image_request_map.erase(img_request);
		// Remove from image_requests array
		for (int i = 0; i < image_requests.size(); i++) {
			Variant v = image_requests[i];
			if (v.get_type() == Variant::OBJECT) {
				HTTPRequest *req = Object::cast_to<HTTPRequest>(v);
				if (req && req == img_request) {
					image_requests.remove_at(i);
					break;
				}
			}
		}
		img_request->queue_free();
	}
}

void TemplateSelectorDialog::_on_image_request_completed(int p_result, int p_response_code, const PackedStringArray &p_headers, const PackedByteArray &p_body) {
	// Find which request completed by checking all image requests
	HTTPRequest *completed_request = nullptr;
	for (int i = 0; i < image_requests.size(); i++) {
		HTTPRequest *req = Object::cast_to<HTTPRequest>(image_requests[i]);
		if (req && image_request_map.has(req)) {
			// Check if this request is in a completed state
			HTTPClient::Status status = req->get_http_client_status();
			if (status == HTTPClient::STATUS_DISCONNECTED || status == HTTPClient::STATUS_CONNECTION_ERROR) {
				completed_request = req;
				break;
			}
		}
	}
	
	// Fallback: use first request in map if we can't determine
	if (!completed_request && image_requests.size() > 0) {
		completed_request = Object::cast_to<HTTPRequest>(image_requests[0]);
	}
	
	if (!completed_request || !image_request_map.has(completed_request)) {
		return;
	}
	
	image_request_map.erase(completed_request);
	// Remove from image_requests array
	for (int i = 0; i < image_requests.size(); i++) {
		Variant v = image_requests[i];
		if (v.get_type() == Variant::OBJECT) {
			HTTPRequest *req = Object::cast_to<HTTPRequest>(v);
			if (req && req == completed_request) {
				image_requests.remove_at(i);
				break;
			}
		}
	}
	
	if (p_result == HTTPRequest::RESULT_SUCCESS && p_response_code == 200) {
		// Load image from bytes - try different formats
		Ref<Image> img = memnew(Image);
		Error err = img->load_png_from_buffer(p_body);
		if (err != OK) {
			err = img->load_jpg_from_buffer(p_body);
		}
		if (err != OK) {
			err = img->load_webp_from_buffer(p_body);
		}
		if (err != OK) {
			// Try SVG (though Godot doesn't natively support SVG, so this will likely fail)
			// For now, just skip unsupported formats
			completed_request->queue_free();
			return;
		}
		
		if (err == OK && img->get_width() > 0 && img->get_height() > 0) {
			Ref<ImageTexture> texture = ImageTexture::create_from_image(img);
			
			TextureRect *texture_rect = memnew(TextureRect);
			texture_rect->set_texture(texture);
			texture_rect->set_expand_mode(TextureRect::EXPAND_FIT_WIDTH_PROPORTIONAL);
			texture_rect->set_stretch_mode(TextureRect::STRETCH_KEEP_ASPECT_CENTERED);
			texture_rect->set_custom_minimum_size(Size2(0, 150) * EDSCALE);
			
			image_container->add_child(texture_rect);
			
			// Show image container if it has images
			if (image_container->get_child_count() > 1) { // More than just the label
				image_container->set_visible(true);
				Control *label = Object::cast_to<Control>(image_container->get_child(0));
				if (label) {
					label->set_visible(true);
				}
			}
		}
	}
	
	completed_request->queue_free();
}

void TemplateSelectorDialog::_on_template_double_clicked(int p_index) {
	_on_template_selected(p_index);
	if (!selected_template_id.is_empty()) {
		ok_pressed();
	}
}

Dictionary TemplateSelectorDialog::get_selected_template() const {
	for (int i = 0; i < templates.size(); i++) {
		Dictionary template_info = templates[i];
		if (template_info.get("id", "") == selected_template_id) {
			return template_info;
		}
	}
	return Dictionary();
}

