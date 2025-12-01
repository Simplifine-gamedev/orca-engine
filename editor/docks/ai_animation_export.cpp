/* AI Animation Export Implementation */

#include "ai_animation_export.h"
#include "core/io/json.h"
#include "core/io/file_access.h"
#include "core/io/dir_access.h"
#include "core/core_bind.h"
#include "core/config/project_settings.h"
#include "scene/gui/separator.h"
#include "scene/gui/check_box.h"
#include "scene/resources/style_box_flat.h"
#include "editor/settings/editor_settings.h"

void AIAnimationExport::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_on_mode_changed", "index"), &AIAnimationExport::_on_mode_changed);
	ClassDB::bind_method(D_METHOD("_on_template_checkbox_toggled", "pressed"), &AIAnimationExport::_on_template_checkbox_toggled);
	ClassDB::bind_method(D_METHOD("_on_dialog_confirmed"), &AIAnimationExport::_on_dialog_confirmed);
	ClassDB::bind_method(D_METHOD("_on_file_selected", "path"), &AIAnimationExport::_on_file_selected);
	ClassDB::bind_method(D_METHOD("_on_folder_selected", "path"), &AIAnimationExport::_on_folder_selected);
	ClassDB::bind_method(D_METHOD("_on_http_request_completed", "result", "response_code", "headers", "body"), &AIAnimationExport::_on_http_request_completed);
}

AIAnimationExport::AIAnimationExport() {
}

AIAnimationExport::~AIAnimationExport() {
	// Dialogs are children of parent_node, they'll be freed automatically
}

void AIAnimationExport::initialize(Control *p_parent) {
	parent_node = p_parent;
	_create_dialog();
}

String AIAnimationExport::_get_api_base_url() {
	// Use the Flask proxy URL (same as AIAnimationUI and AIChatDock)
	// This ensures consistent routing through the main backend
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
	
	// Allow override via environment variable
	String env_url = OS::get_singleton()->get_environment("AI_CHAT_CLOUD_URL");
	if (!env_url.is_empty()) {
		base_url = env_url;
	}
	
	return base_url;
}

void AIAnimationExport::_create_dialog() {
	if (!parent_node || export_dialog) {
		return;
	}
	
	// Main dialog
	export_dialog = memnew(ConfirmationDialog);
	export_dialog->set_title("Export Animation");
	export_dialog->set_ok_button_text("Export");
	export_dialog->set_min_size(Size2(450, 350));
	parent_node->add_child(export_dialog);
	export_dialog->connect("confirmed", callable_mp(this, &AIAnimationExport::_on_dialog_confirmed));
	
	VBoxContainer *main_vbox = memnew(VBoxContainer);
	main_vbox->add_theme_constant_override("separation", 10);
	export_dialog->add_child(main_vbox);
	
	// Export as Godot Template checkbox (default: checked)
	use_template_check = memnew(CheckBox);
	use_template_check->set_text("Export as Godot Template (recommended)");
	use_template_check->set_pressed(true);  // Default to template mode
	use_template_check->set_tooltip_text("Creates sprite sheet + .tres resource + .tscn scene + .gd script - ready to use in your game!");
	use_template_check->connect("toggled", callable_mp(this, &AIAnimationExport::_on_template_checkbox_toggled));
	main_vbox->add_child(use_template_check);
	
	// Hidden mode option (kept for backwards compatibility but controlled by checkbox)
	mode_option = memnew(OptionButton);
	mode_option->add_item("Simple (PNG/GIF)", EXPORT_SIMPLE);
	mode_option->add_item("Godot Template", EXPORT_GODOT_TEMPLATE);
	mode_option->select(EXPORT_GODOT_TEMPLATE);  // Default to template
	mode_option->set_visible(false);  // Hidden - controlled by checkbox
	mode_option->connect("item_selected", callable_mp(this, &AIAnimationExport::_on_mode_changed));
	main_vbox->add_child(mode_option);
	
	// Resolution
	HBoxContainer *res_row = memnew(HBoxContainer);
	main_vbox->add_child(res_row);
	
	Label *res_label = memnew(Label);
	res_label->set_text("Resolution:");
	res_label->set_custom_minimum_size(Size2(120, 0));
	res_row->add_child(res_label);
	
	resolution_spin = memnew(SpinBox);
	resolution_spin->set_min(32);
	resolution_spin->set_max(512);
	resolution_spin->set_step(32);
	resolution_spin->set_value(128);
	resolution_spin->set_suffix("px");
	resolution_spin->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	res_row->add_child(resolution_spin);
	
	// Simple export format (shown only in simple mode)
	HBoxContainer *format_row = memnew(HBoxContainer);
	format_row->set_name("format_row");
	main_vbox->add_child(format_row);
	
	Label *format_label = memnew(Label);
	format_label->set_text("Format:");
	format_label->set_custom_minimum_size(Size2(120, 0));
	format_row->add_child(format_label);
	
	format_option = memnew(OptionButton);
	format_option->add_item("Sprite Sheet (PNG)", 0);
	format_option->add_item("Animated GIF", 1);
	format_option->add_item("Individual Frames", 2);
	format_option->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	format_row->add_child(format_option);
	
	// Separator
	main_vbox->add_child(memnew(HSeparator));
	
	// Template options container (shown only in template mode)
	template_options_container = memnew(VBoxContainer);
	template_options_container->set_name("template_options");
	template_options_container->add_theme_constant_override("separation", 8);
	template_options_container->set_visible(false);
	main_vbox->add_child(template_options_container);
	
	// Template Type
	HBoxContainer *template_row = memnew(HBoxContainer);
	template_options_container->add_child(template_row);
	
	Label *template_label = memnew(Label);
	template_label->set_text("Template:");
	template_label->set_custom_minimum_size(Size2(120, 0));
	template_row->add_child(template_label);
	
	template_type_option = memnew(OptionButton);
	template_type_option->add_item("Character (with movement)", TEMPLATE_CHARACTER);
	template_type_option->add_item("RPG Character (8-dir top-down)", TEMPLATE_RPG_CHARACTER);
	template_type_option->add_item("Effect (one-shot)", TEMPLATE_EFFECT);
	template_type_option->add_item("Prop (animated object)", TEMPLATE_PROP);
	template_type_option->add_item("Simple (minimal)", TEMPLATE_SIMPLE);
	template_type_option->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	template_row->add_child(template_type_option);
	
	// Resource Name
	HBoxContainer *name_row = memnew(HBoxContainer);
	template_options_container->add_child(name_row);
	
	Label *name_label = memnew(Label);
	name_label->set_text("Resource Name:");
	name_label->set_custom_minimum_size(Size2(120, 0));
	name_row->add_child(name_label);
	
	resource_name_edit = memnew(LineEdit);
	resource_name_edit->set_placeholder("player_sprite");
	resource_name_edit->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	name_row->add_child(resource_name_edit);
	
	// FPS
	HBoxContainer *fps_row = memnew(HBoxContainer);
	template_options_container->add_child(fps_row);
	
	Label *fps_label = memnew(Label);
	fps_label->set_text("Animation FPS:");
	fps_label->set_custom_minimum_size(Size2(120, 0));
	fps_row->add_child(fps_label);
	
	fps_spin = memnew(SpinBox);
	fps_spin->set_min(1);
	fps_spin->set_max(60);
	fps_spin->set_step(1);
	fps_spin->set_value(10);
	fps_spin->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	fps_row->add_child(fps_spin);
	
	// Template description
	Label *desc_label = memnew(Label);
	desc_label->set_text("Creates: sprite_sheet.png, .tres, .tscn, .gd");
	desc_label->add_theme_color_override("font_color", Color(0.6, 0.8, 0.6));
	desc_label->add_theme_font_size_override("font_size", 12);
	template_options_container->add_child(desc_label);
	
	// Info label
	main_vbox->add_child(memnew(HSeparator));
	
	info_label = memnew(Label);
	info_label->set_text("Animation: ");
	info_label->add_theme_color_override("font_color", Color(0.7, 0.7, 0.7));
	main_vbox->add_child(info_label);
	
	// File dialog for simple export
	file_dialog = memnew(EditorFileDialog);
	file_dialog->set_file_mode(EditorFileDialog::FILE_MODE_SAVE_FILE);
	file_dialog->set_access(EditorFileDialog::ACCESS_RESOURCES);
	file_dialog->connect("file_selected", callable_mp(this, &AIAnimationExport::_on_file_selected));
	file_dialog->connect("dir_selected", callable_mp(this, &AIAnimationExport::_on_folder_selected));
	parent_node->add_child(file_dialog);
	
	// HTTP request
	http_request = memnew(HTTPRequest);
	http_request->set_use_threads(true);
	http_request->connect("request_completed", callable_mp(this, &AIAnimationExport::_on_http_request_completed));
	parent_node->add_child(http_request);
}

void AIAnimationExport::_on_mode_changed(int p_index) {
	current_mode = (ExportMode)p_index;
	
	// Toggle visibility of format vs template options
	if (format_option && format_option->get_parent()) {
		Control *parent_ctrl = Object::cast_to<Control>(format_option->get_parent());
		if (parent_ctrl) {
			parent_ctrl->set_visible(current_mode == EXPORT_SIMPLE);
		}
	}
	if (template_options_container) {
		template_options_container->set_visible(current_mode == EXPORT_GODOT_TEMPLATE);
	}
	
	// Resize dialog
	if (export_dialog) {
		export_dialog->reset_size();
	}
}

void AIAnimationExport::_on_template_checkbox_toggled(bool p_pressed) {
	// Toggle between template and simple mode based on checkbox
	ExportMode new_mode = p_pressed ? EXPORT_GODOT_TEMPLATE : EXPORT_SIMPLE;
	
	if (mode_option) {
		mode_option->select((int)new_mode);
	}
	_on_mode_changed((int)new_mode);
}

void AIAnimationExport::show_export_dialog(const String &p_project_id, const String &p_animation_id) {
	if (!export_dialog) {
		return;
	}
	
	pending_project_id = p_project_id;
	pending_animation_id = p_animation_id;
	
	// Update info label
	if (info_label) {
		info_label->set_text("Animation: " + p_animation_id);
	}
	
	// Set default resource name from animation ID
	if (resource_name_edit) {
		String clean_name = p_animation_id.to_lower().replace(" ", "_").replace("-", "_");
		resource_name_edit->set_text(clean_name);
	}
	
	// Default to Godot Template mode (checkbox checked)
	if (use_template_check) {
		use_template_check->set_pressed(true);
	}
	if (mode_option) {
		mode_option->select(EXPORT_GODOT_TEMPLATE);
		_on_mode_changed(EXPORT_GODOT_TEMPLATE);
	}
	
	export_dialog->popup_centered();
}

void AIAnimationExport::show_project_export_dialog(const String &p_project_id, const Array &p_animation_ids) {
	if (!export_dialog) {
		return;
	}
	
	pending_project_id = p_project_id;
	pending_animation_id = "";  // Empty means all animations
	
	// Update info label
	if (info_label) {
		info_label->set_text("Project: " + itos(p_animation_ids.size()) + " animations");
	}
	
	// Set default resource name
	if (resource_name_edit) {
		resource_name_edit->set_text("sprite");
	}
	
	// Default to template mode for project export
	if (mode_option) {
		mode_option->select(EXPORT_GODOT_TEMPLATE);
		_on_mode_changed(EXPORT_GODOT_TEMPLATE);
	}
	
	export_dialog->popup_centered();
}

void AIAnimationExport::_on_dialog_confirmed() {
	if (current_mode == EXPORT_SIMPLE) {
		// Simple export - show file dialog
		file_dialog->clear_filters();
		int format_idx = format_option ? format_option->get_selected() : 0;
		
		String ext = "png";
		String filter_name = "PNG Image";
		if (format_idx == 1) {
			ext = "gif";
			filter_name = "Animated GIF";
		}
		
		file_dialog->add_filter("*." + ext, filter_name);
		file_dialog->set_file_mode(EditorFileDialog::FILE_MODE_SAVE_FILE);
		
		String filename = pending_animation_id.is_empty() ? "sprite" : pending_animation_id;
		int res = resolution_spin ? (int)resolution_spin->get_value() : 128;
		file_dialog->set_current_file(filename + "_" + itos(res) + "px." + ext);
		file_dialog->popup_centered(Size2(800, 600));
	} else {
		// Template export - show folder dialog
		file_dialog->set_file_mode(EditorFileDialog::FILE_MODE_OPEN_DIR);
		file_dialog->set_current_dir("res://");
		file_dialog->popup_centered(Size2(800, 600));
	}
}

void AIAnimationExport::_on_file_selected(const String &p_path) {
	pending_export_path = p_path;
	
	// Build request for simple export
	Dictionary body;
	body["project_id"] = pending_project_id;
	body["animation_id"] = pending_animation_id;
	body["resolution"] = resolution_spin ? (int)resolution_spin->get_value() : 128;
	
	int format_idx = format_option ? format_option->get_selected() : 0;
	String format = "sprite_sheet";
	if (format_idx == 1) format = "gif";
	else if (format_idx == 2) format = "frames";
	body["format"] = format;
	
	String json_body = JSON::stringify(body);
	
	PackedStringArray headers;
	headers.push_back("Content-Type: application/json");
	
	http_request->set_meta("export_type", "simple");
	http_request->set_meta("save_path", p_path);
	http_request->set_meta("format", format);
	
	// Use /animation/export - Flask proxy forwards /animation/* to animation server
	String url = _get_api_base_url() + "/animation/export";
	Error err = http_request->request(url, headers, HTTPClient::METHOD_POST, json_body);
	if (err != OK) {
		print_line("ANIM_EXPORT: Failed to start request: " + itos(err));
	}
}

void AIAnimationExport::_on_folder_selected(const String &p_path) {
	pending_export_path = p_path;
	
	// Build request for template export
	Dictionary body;
	body["project_id"] = pending_project_id;
	
	// If specific animation, wrap in array
	if (!pending_animation_id.is_empty()) {
		Array anim_ids;
		anim_ids.push_back(pending_animation_id);
		body["animation_ids"] = anim_ids;
	}
	// else: exports all animations
	
	body["resolution"] = resolution_spin ? (int)resolution_spin->get_value() : 128;
	body["fps"] = fps_spin ? (int)fps_spin->get_value() : 10;
	body["resource_name"] = resource_name_edit ? resource_name_edit->get_text() : "sprite";
	
	int template_idx = template_type_option ? template_type_option->get_selected() : 0;
	String template_type = "character";
	if (template_idx == TEMPLATE_RPG_CHARACTER) template_type = "rpg_character";
	else if (template_idx == TEMPLATE_EFFECT) template_type = "effect";
	else if (template_idx == TEMPLATE_PROP) template_type = "prop";
	else if (template_idx == TEMPLATE_SIMPLE) template_type = "simple";
	body["template_type"] = template_type;
	
	String json_body = JSON::stringify(body);
	
	PackedStringArray headers;
	headers.push_back("Content-Type: application/json");
	
	http_request->set_meta("export_type", "template");
	http_request->set_meta("save_folder", p_path);
	http_request->set_meta("resource_name", body["resource_name"]);
	
	// Use /animation/export/godot_template - Flask proxy forwards /animation/* to animation server
	String url = _get_api_base_url() + "/animation/export/godot_template";
	Error err = http_request->request(url, headers, HTTPClient::METHOD_POST, json_body);
	if (err != OK) {
		print_line("ANIM_EXPORT: Failed to start template request: " + itos(err));
	} else {
		print_line("ANIM_EXPORT: Template export request sent to " + url);
	}
}

void AIAnimationExport::_on_http_request_completed(int p_result, int p_response_code, const PackedStringArray &p_headers, const PackedByteArray &p_body) {
	String export_type = http_request->get_meta("export_type", "simple");
	
	if (p_result != HTTPRequest::RESULT_SUCCESS || p_response_code != 200) {
		print_line("ANIM_EXPORT: HTTP Error " + itos(p_response_code));
		if (on_export_error.is_valid()) {
			on_export_error.call("Export failed: HTTP " + itos(p_response_code));
		}
		return;
	}
	
	String response_text = String::utf8((const char *)p_body.ptr(), p_body.size());
	Ref<JSON> json;
	json.instantiate();
	Error parse_err = json->parse(response_text);
	if (parse_err != OK) {
		print_line("ANIM_EXPORT: Failed to parse response");
		return;
	}
	
	Dictionary result = json->get_data();
	
	if (export_type == "simple") {
		// Simple export - save single file
		String save_path = http_request->get_meta("save_path", "");
		String format = http_request->get_meta("format", "sprite_sheet");
		
		String base64_data;
		if (format == "sprite_sheet") {
			base64_data = result.get("sprite_sheet_base64", "");
		} else if (format == "gif") {
			base64_data = result.get("gif_base64", "");
		}
		
		if (!base64_data.is_empty() && !save_path.is_empty()) {
			Vector<uint8_t> data = CoreBind::Marshalls::get_singleton()->base64_to_raw(base64_data);
			String global_path = ProjectSettings::get_singleton()->globalize_path(save_path);
			
			Ref<FileAccess> file = FileAccess::open(global_path, FileAccess::WRITE);
			if (file.is_valid()) {
				file->store_buffer(data);
				file->close();
				print_line("ANIM_EXPORT: Saved to " + save_path);
				
				if (on_export_complete.is_valid()) {
					on_export_complete.call(save_path);
				}
			}
		}
	} else {
		// Template export - save multiple files
		String save_folder = http_request->get_meta("save_folder", "");
		_save_template_files(result, save_folder);
	}
}

void AIAnimationExport::_save_template_files(const Dictionary &p_data, const String &p_folder) {
	Dictionary files = p_data.get("files", Dictionary());
	
	if (files.is_empty()) {
		print_line("ANIM_EXPORT: No files in template response");
		return;
	}
	
	// Ensure directory exists
	String global_folder = ProjectSettings::get_singleton()->globalize_path(p_folder);
	DirAccess::make_dir_recursive_absolute(global_folder);
	
	Array keys = files.keys();
	int saved_count = 0;
	
	for (int i = 0; i < keys.size(); i++) {
		String filename = keys[i];
		String content = files[filename];
		String filepath = p_folder.path_join(filename);
		String global_path = ProjectSettings::get_singleton()->globalize_path(filepath);
		
		// Check if it's base64 (PNG) or text (.tres, .tscn, .gd)
		bool is_binary = filename.ends_with(".png");
		
		Ref<FileAccess> file = FileAccess::open(global_path, FileAccess::WRITE);
		if (file.is_valid()) {
			if (is_binary) {
				Vector<uint8_t> data = CoreBind::Marshalls::get_singleton()->base64_to_raw(content);
				file->store_buffer(data);
			} else {
				file->store_string(content);
			}
			file->close();
			saved_count++;
			print_line("ANIM_EXPORT: Saved " + filepath);
		} else {
			print_line("ANIM_EXPORT: Failed to save " + filepath);
		}
	}
	
	print_line("ANIM_EXPORT: Template export complete - " + itos(saved_count) + " files saved to " + p_folder);
	
	if (on_export_complete.is_valid()) {
		on_export_complete.call(p_folder);
	}
}

void AIAnimationExport::set_on_export_complete(const Callable &p_callback) {
	on_export_complete = p_callback;
}

void AIAnimationExport::set_on_export_error(const Callable &p_callback) {
	on_export_error = p_callback;
}

void AIAnimationExport::quick_export_template(
	Control *p_parent,
	const String &p_project_id,
	const Array &p_animation_ids,
	const String &p_folder_path,
	const String &p_resource_name,
	int p_template_type,
	int p_resolution,
	int p_fps
) {
	// Quick export is implemented via the instance methods
	// Create a temporary exporter for this operation
	Ref<AIAnimationExport> exporter;
	exporter.instantiate();
	exporter->initialize(p_parent);
	
	// Set up the export
	exporter->pending_project_id = p_project_id;
	exporter->pending_animation_id = "";  // All animations
	
	// Directly trigger folder export
	exporter->_on_folder_selected(p_folder_path);
	
	print_line("QUICK_EXPORT: Started template export to " + p_folder_path);
}

