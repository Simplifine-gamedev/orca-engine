/*
 * © 2025 Simplifine Corp.
 * Explorer Panel - Deep codebase investigation UI component
 * Personal Non-Commercial License applies.
 */

#include "explorer_panel.h"

#include "core/io/json.h"
#include "core/os/os.h"
#include "editor/settings/editor_settings.h"
#include "editor/themes/editor_scale.h"
#include "scene/gui/separator.h"
#include "scene/resources/style_box_flat.h"
#include "scene/main/http_request.h"

ExplorerPanel::ExplorerPanel() {
	set_custom_minimum_size(Size2(400, 300) * EDSCALE);
	_setup_ui();
	// Don't call _style_panel() here - wait for NOTIFICATION_READY
}

ExplorerPanel::~ExplorerPanel() {
	stop_exploration();
}

void ExplorerPanel::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_READY: {
			// Now safe to access theme
			_style_panel();
		} break;
		case NOTIFICATION_THEME_CHANGED: {
			if (is_inside_tree()) {
				_style_panel();
			}
		} break;
	}
}

void ExplorerPanel::_bind_methods() {
	ClassDB::bind_method(D_METHOD("start_exploration", "question", "depth", "focus_areas"), &ExplorerPanel::start_exploration, DEFVAL("normal"), DEFVAL(Array()));
	ClassDB::bind_method(D_METHOD("stop_exploration"), &ExplorerPanel::stop_exploration);
	ClassDB::bind_method(D_METHOD("set_backend_url", "url"), &ExplorerPanel::set_backend_url);
	ClassDB::bind_method(D_METHOD("is_currently_exploring"), &ExplorerPanel::is_currently_exploring);
	ClassDB::bind_method(D_METHOD("set_project_context", "context"), &ExplorerPanel::set_project_context);
	
	// Internal bindings for signals
	ClassDB::bind_method(D_METHOD("_on_close_pressed"), &ExplorerPanel::_on_close_pressed);
	ClassDB::bind_method(D_METHOD("_on_http_request_completed", "result", "response_code", "headers", "body"), &ExplorerPanel::_on_http_request_completed);
	
	ADD_SIGNAL(MethodInfo("exploration_started", PropertyInfo(Variant::STRING, "exploration_id")));
	ADD_SIGNAL(MethodInfo("exploration_completed", PropertyInfo(Variant::STRING, "exploration_id"), PropertyInfo(Variant::DICTIONARY, "report")));
	ADD_SIGNAL(MethodInfo("exploration_error", PropertyInfo(Variant::STRING, "error")));
}

void ExplorerPanel::_setup_ui() {
	main_vbox = memnew(VBoxContainer);
	add_child(main_vbox);
	
	// === Header Section ===
	header_hbox = memnew(HBoxContainer);
	main_vbox->add_child(header_hbox);
	
	title_label = memnew(Label);
	title_label->set_text("🔍 Explorer Agent");
	title_label->add_theme_font_size_override("font_size", 16 * EDSCALE);
	header_hbox->add_child(title_label);
	
	header_hbox->add_spacer();
	
	status_label = memnew(Label);
	status_label->set_text("Ready");
	status_label->add_theme_color_override("font_color", Color(0.6, 0.6, 0.6));
	header_hbox->add_child(status_label);
	
	close_button = memnew(Button);
	close_button->set_text("✕");
	close_button->set_flat(true);
	close_button->connect("pressed", callable_mp(this, &ExplorerPanel::_on_close_pressed));
	header_hbox->add_child(close_button);
	
	// === Progress Section ===
	progress_hbox = memnew(HBoxContainer);
	progress_hbox->set_visible(false);
	main_vbox->add_child(progress_hbox);
	
	progress_bar = memnew(ProgressBar);
	progress_bar->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	progress_bar->set_min(0);
	progress_bar->set_max(100);
	progress_bar->set_value(0);
	progress_hbox->add_child(progress_bar);
	
	turn_label = memnew(Label);
	turn_label->set_text("Turn 0/10");
	turn_label->set_custom_minimum_size(Size2(80, 0) * EDSCALE);
	progress_hbox->add_child(turn_label);
	
	main_vbox->add_child(memnew(HSeparator));
	
	// === Question Display ===
	question_panel = memnew(PanelContainer);
	question_panel->set_visible(false);
	main_vbox->add_child(question_panel);
	
	question_label = memnew(Label);
	question_label->set_autowrap_mode(TextServer::AUTOWRAP_WORD_SMART);
	question_panel->add_child(question_label);
	
	// === Output Scroll Area ===
	output_scroll = memnew(ScrollContainer);
	output_scroll->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	output_scroll->set_horizontal_scroll_mode(ScrollContainer::SCROLL_MODE_DISABLED);
	main_vbox->add_child(output_scroll);
	
	output_vbox = memnew(VBoxContainer);
	output_vbox->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	output_scroll->add_child(output_vbox);
	
	// Thinking/streaming text
	thinking_label = memnew(RichTextLabel);
	thinking_label->set_use_bbcode(true);
	thinking_label->set_fit_content(true);
	thinking_label->set_scroll_active(false);
	thinking_label->set_selection_enabled(true);
	thinking_label->set_visible(false);
	output_vbox->add_child(thinking_label);
	
	// Tools section (tool calls will be added here)
	tools_section = memnew(VBoxContainer);
	output_vbox->add_child(tools_section);
	
	// === Report Section (initially hidden) ===
	report_panel = memnew(PanelContainer);
	report_panel->set_visible(false);
	main_vbox->add_child(report_panel);
	
	report_vbox = memnew(VBoxContainer);
	report_panel->add_child(report_vbox);
	
	Label *report_title = memnew(Label);
	report_title->set_text("📋 Exploration Report");
	report_title->add_theme_font_size_override("font_size", 14 * EDSCALE);
	report_vbox->add_child(report_title);
	
	report_metadata_hbox = memnew(HBoxContainer);
	report_vbox->add_child(report_metadata_hbox);
	
	confidence_label = memnew(Label);
	confidence_label->set_text("Confidence: -");
	report_metadata_hbox->add_child(confidence_label);
	
	report_metadata_hbox->add_spacer();
	
	files_explored_label = memnew(Label);
	files_explored_label->set_text("Files: 0");
	report_metadata_hbox->add_child(files_explored_label);
	
	report_metadata_hbox->add_spacer();
	
	queries_used_label = memnew(Label);
	queries_used_label->set_text("Queries: 0");
	report_metadata_hbox->add_child(queries_used_label);
	
	report_vbox->add_child(memnew(HSeparator));
	
	report_content = memnew(RichTextLabel);
	report_content->set_use_bbcode(true);
	report_content->set_fit_content(true);
	report_content->set_scroll_active(false);
	report_content->set_selection_enabled(true);
	report_content->set_custom_minimum_size(Size2(0, 100) * EDSCALE);
	report_vbox->add_child(report_content);
	
	// Setup HTTP request node
	http_request = memnew(HTTPRequest);
	add_child(http_request);
	http_request->connect("request_completed", callable_mp(this, &ExplorerPanel::_on_http_request_completed));
}

void ExplorerPanel::_style_panel() {
	// Safety check - don't access theme if not in tree
	if (!is_inside_tree()) {
		return;
	}
	
	// Use fallback colors if theme colors aren't available
	Color base_color = Color(0.2, 0.2, 0.22);
	Color dark_color_1 = Color(0.15, 0.15, 0.17);
	Color dark_color_2 = Color(0.12, 0.12, 0.14);
	
	// Try to get theme colors
	if (has_theme_color(SNAME("base_color"), SNAME("Editor"))) {
		base_color = get_theme_color(SNAME("base_color"), SNAME("Editor"));
	}
	if (has_theme_color(SNAME("dark_color_1"), SNAME("Editor"))) {
		dark_color_1 = get_theme_color(SNAME("dark_color_1"), SNAME("Editor"));
	}
	if (has_theme_color(SNAME("dark_color_2"), SNAME("Editor"))) {
		dark_color_2 = get_theme_color(SNAME("dark_color_2"), SNAME("Editor"));
	}
	
	// Main panel style
	Ref<StyleBoxFlat> panel_style = memnew(StyleBoxFlat);
	panel_style->set_bg_color(base_color);
	panel_style->set_border_width_all(1);
	panel_style->set_border_color(dark_color_2);
	panel_style->set_content_margin_all(12 * EDSCALE);
	panel_style->set_corner_radius_all(4 * EDSCALE);
	add_theme_style_override("panel", panel_style);
	
	// Question panel style
	if (question_panel) {
		Ref<StyleBoxFlat> question_style = memnew(StyleBoxFlat);
		question_style->set_bg_color(dark_color_1);
		question_style->set_content_margin_all(8 * EDSCALE);
		question_style->set_corner_radius_all(4 * EDSCALE);
		question_panel->add_theme_style_override("panel", question_style);
	}
	
	// Report panel style
	if (report_panel) {
		Ref<StyleBoxFlat> report_style = memnew(StyleBoxFlat);
		report_style->set_bg_color(dark_color_1);
		report_style->set_border_width_all(1);
		report_style->set_border_color(Color(0.3, 0.6, 0.4, 0.5)); // Subtle green tint for reports
		report_style->set_content_margin_all(10 * EDSCALE);
		report_style->set_corner_radius_all(4 * EDSCALE);
		report_panel->add_theme_style_override("panel", report_style);
	}
}

void ExplorerPanel::_clear_output() {
	// Clear thinking text
	accumulated_thinking_text = "";
	if (thinking_label) {
		thinking_label->clear();
		thinking_label->set_visible(false);
	}
	
	// Clear tool placeholders
	if (tools_section) {
		for (int i = tools_section->get_child_count() - 1; i >= 0; i--) {
			Node *child = tools_section->get_child(i);
			tools_section->remove_child(child);
			child->queue_free();
		}
	}
	tool_placeholders.clear();
	
	// Hide report
	if (report_panel) {
		report_panel->set_visible(false);
	}
	if (report_content) {
		report_content->clear();
	}
}

void ExplorerPanel::start_exploration(const String &p_question, const String &p_depth, const Array &p_focus_areas) {
	print_line("EXPLORER: start_exploration called with question: " + p_question);
	
	if (is_exploring) {
		print_line("EXPLORER: Stopping previous exploration");
		stop_exploration();
	}
	
	current_question = p_question;
	is_exploring = true;
	current_turn = 0;
	
	// Determine max turns based on depth
	if (p_depth == "quick") {
		max_turns = 5;
	} else if (p_depth == "thorough") {
		max_turns = 15;
	} else {
		max_turns = 10;
	}
	
	// Safety check for UI elements
	if (!question_label || !question_panel || !progress_hbox || !progress_bar || !turn_label || !status_label) {
		print_line("EXPLORER: ERROR - UI elements not initialized!");
		is_exploring = false;
		emit_signal("exploration_error", "UI not initialized");
		return;
	}
	
	// Update UI
	_clear_output();
	question_label->set_text("❓ " + p_question);
	question_panel->set_visible(true);
	progress_hbox->set_visible(true);
	progress_bar->set_value(0);
	turn_label->set_text("Turn 0/" + String::num_int64(max_turns));
	status_label->set_text("Starting...");
	status_label->add_theme_color_override("font_color", Color(0.4, 0.7, 1.0));
	
	// Build request JSON
	Dictionary request_data;
	request_data["question"] = p_question;
	request_data["depth"] = p_depth;
	if (!p_focus_areas.is_empty()) {
		request_data["focus_areas"] = p_focus_areas;
	}
	
	// Make HTTP request to /explore endpoint
	if (backend_url.is_empty()) {
		print_line("EXPLORER: No backend URL configured!");
		status_label->set_text("Error: No backend");
		status_label->add_theme_color_override("font_color", Color(1.0, 0.4, 0.4));
		is_exploring = false;
		emit_signal("exploration_error", "No backend URL configured");
		return;
	}
	
	// Validate HTTP request node
	if (!http_request) {
		print_line("EXPLORER: ERROR - HTTPRequest not initialized!");
		status_label->set_text("Error: HTTP not ready");
		status_label->add_theme_color_override("font_color", Color(1.0, 0.4, 0.4));
		is_exploring = false;
		emit_signal("exploration_error", "HTTP request not initialized");
		return;
	}
	
	// Make sure HTTP request is in the tree
	if (!http_request->is_inside_tree()) {
		print_line("EXPLORER: ERROR - HTTPRequest not in tree!");
		status_label->set_text("Error: HTTP not ready");
		status_label->add_theme_color_override("font_color", Color(1.0, 0.4, 0.4));
		is_exploring = false;
		emit_signal("exploration_error", "HTTP request not in scene tree");
		return;
	}
	
	String explore_url = backend_url;
	if (!explore_url.ends_with("/")) {
		explore_url += "/";
	}
	explore_url += "explore";
	
	PackedStringArray headers;
	headers.push_back("Content-Type: application/json");
	
	String json_body = JSON::stringify(request_data);
	
	print_line("EXPLORER: Starting exploration - URL: " + explore_url);
	print_line("EXPLORER: Question: " + p_question);
	print_line("EXPLORER: Request body length: " + itos(json_body.length()));
	
	Error err = http_request->request(explore_url, headers, HTTPClient::METHOD_POST, json_body);
	if (err != OK) {
		print_line("EXPLORER: Failed to start HTTP request: " + itos(err));
		status_label->set_text("Connection error: " + itos(err));
		status_label->add_theme_color_override("font_color", Color(1.0, 0.4, 0.4));
		is_exploring = false;
		emit_signal("exploration_error", "Failed to connect to backend: error " + itos(err));
	} else {
		print_line("EXPLORER: HTTP request started successfully");
	}
}

void ExplorerPanel::stop_exploration() {
	if (!is_exploring) {
		return;
	}
	
	is_exploring = false;
	if (http_request) {
		http_request->cancel_request();
	}
	
	status_label->set_text("Stopped");
	status_label->add_theme_color_override("font_color", Color(0.6, 0.6, 0.6));
	progress_hbox->set_visible(false);
	
	print_line("EXPLORER: Exploration stopped");
}

void ExplorerPanel::set_backend_url(const String &p_url) {
	backend_url = p_url;
}

void ExplorerPanel::set_project_context(const Dictionary &p_context) {
	// Store project context to send with exploration requests
	// This is handled in start_exploration via request headers
}

void ExplorerPanel::_on_close_pressed() {
	stop_exploration();
	set_visible(false);
}

void ExplorerPanel::_on_http_request_completed(int p_result, int p_response_code, const PackedStringArray &p_headers, const PackedByteArray &p_body) {
	// Process any remaining buffered data
	if (!response_buffer.is_empty()) {
		_process_ndjson_line(response_buffer);
		response_buffer = "";
	}
	
	// Handle the full response body for non-streaming responses
	if (p_body.size() > 0) {
		String body_str = String::utf8((const char *)p_body.ptr(), p_body.size());
		
		// Split by newlines and process each line
		Vector<String> lines = body_str.split("\n", false);
		for (int i = 0; i < lines.size(); i++) {
			String line = lines[i].strip_edges();
			if (!line.is_empty()) {
				_process_ndjson_line(line);
			}
		}
	}
	
	// Check for HTTP errors
	if (p_result != HTTPRequest::RESULT_SUCCESS) {
		print_line("EXPLORER: HTTP request failed with result: " + itos(p_result));
		status_label->set_text("Connection failed");
		status_label->add_theme_color_override("font_color", Color(1.0, 0.4, 0.4));
		is_exploring = false;
		emit_signal("exploration_error", "Connection failed");
		return;
	}
	
	if (p_response_code >= 400) {
		print_line("EXPLORER: HTTP error " + itos(p_response_code));
		status_label->set_text("Server error: " + itos(p_response_code));
		status_label->add_theme_color_override("font_color", Color(1.0, 0.4, 0.4));
		is_exploring = false;
		emit_signal("exploration_error", "Server error: " + itos(p_response_code));
		return;
	}
	
	// Exploration completed successfully
	is_exploring = false;
}

void ExplorerPanel::_process_ndjson_line(const String &p_line) {
	Ref<JSON> json;
	json.instantiate();
	Error err = json->parse(p_line);
	if (err != OK) {
		print_line("EXPLORER: Failed to parse JSON: " + p_line.left(100));
		return;
	}
	
	Dictionary data = json->get_data();
	
	// Handle different event types
	String event_type = data.get("type", "");
	String status = data.get("status", "");
	
	if (!event_type.is_empty()) {
		// Event-based messages
		if (event_type == "started") {
			current_exploration_id = data.get("exploration_id", "");
			print_line("EXPLORER: Started exploration: " + current_exploration_id);
			status_label->set_text("Exploring...");
			emit_signal("exploration_started", current_exploration_id);
		}
		else if (event_type == "explorer_text") {
			String delta = data.get("content_delta", "");
			_append_thinking_text(delta);
		}
		else if (event_type == "exploration_report") {
			_display_report(data);
		}
	}
	
	if (!status.is_empty()) {
		// Status-based messages
		if (status == "exploring") {
			current_turn = data.get("turn", 0);
			int mt = data.get("max_turns", max_turns);
			float progress = (float(current_turn) / float(mt)) * 100.0f;
			progress_bar->set_value(progress);
			turn_label->set_text("Turn " + String::num_int64(current_turn) + "/" + String::num_int64(mt));
			status_label->set_text("Thinking...");
		}
		else if (status == "explorer_tool_call") {
			String tool_id = data.get("tool_id", "");
			String tool_name = data.get("tool_name", "");
			_add_tool_call_placeholder(tool_id, tool_name);
			status_label->set_text("Calling: " + tool_name);
		}
		else if (status == "explorer_tool_result") {
			String tool_id = data.get("tool_id", "");
			String tool_name = data.get("tool_name", "");
			Dictionary result = data.get("result", Dictionary());
			_update_tool_result(tool_id, tool_name, result);
		}
		else if (status == "exploration_complete") {
			_display_report(data);
		}
		else if (status == "completed") {
			status_label->set_text("Complete ✓");
			status_label->add_theme_color_override("font_color", Color(0.4, 0.8, 0.4));
			progress_hbox->set_visible(false);
			is_exploring = false;
			
			Dictionary report_data;
			report_data["exploration_id"] = data.get("exploration_id", current_exploration_id);
			report_data["total_turns"] = data.get("total_turns", current_turn);
			emit_signal("exploration_completed", current_exploration_id, report_data);
		}
		else if (status == "error") {
			String error_msg = data.get("error", "Unknown error");
			status_label->set_text("Error: " + error_msg.left(30));
			status_label->add_theme_color_override("font_color", Color(1.0, 0.4, 0.4));
			is_exploring = false;
			emit_signal("exploration_error", error_msg);
		}
	}
}

void ExplorerPanel::_add_tool_call_placeholder(const String &p_tool_id, const String &p_tool_name) {
	if (tool_placeholders.has(p_tool_id)) {
		return; // Already exists
	}
	
	PanelContainer *placeholder = memnew(PanelContainer);
	placeholder->set_name("tool_" + p_tool_id);
	
	// Style
	Ref<StyleBoxFlat> style = memnew(StyleBoxFlat);
	style->set_bg_color(get_theme_color(SNAME("dark_color_2"), SNAME("Editor")));
	style->set_content_margin_all(6 * EDSCALE);
	style->set_corner_radius_all(3 * EDSCALE);
	placeholder->add_theme_style_override("panel", style);
	
	HBoxContainer *hbox = memnew(HBoxContainer);
	placeholder->add_child(hbox);
	
	// Loading indicator (animated dots would be nice, but simple label for now)
	Label *status_icon = memnew(Label);
	status_icon->set_text("⏳");
	status_icon->set_name("status_icon");
	hbox->add_child(status_icon);
	
	Label *name_label = memnew(Label);
	name_label->set_text(p_tool_name);
	name_label->add_theme_color_override("font_color", Color(0.7, 0.8, 1.0));
	hbox->add_child(name_label);
	
	hbox->add_spacer();
	
	Label *result_label = memnew(Label);
	result_label->set_text("Running...");
	result_label->set_name("result_label");
	result_label->add_theme_color_override("font_color", Color(0.5, 0.5, 0.5));
	hbox->add_child(result_label);
	
	tools_section->add_child(placeholder);
	tool_placeholders[p_tool_id] = placeholder;
	
	_scroll_to_bottom();
}

void ExplorerPanel::_update_tool_result(const String &p_tool_id, const String &p_tool_name, const Dictionary &p_result) {
	if (!tool_placeholders.has(p_tool_id)) {
		return;
	}
	
	PanelContainer *placeholder = tool_placeholders[p_tool_id];
	
	// Update status icon
	Label *status_icon = Object::cast_to<Label>(placeholder->find_child("status_icon", true, false));
	Label *result_label = Object::cast_to<Label>(placeholder->find_child("result_label", true, false));
	
	bool success = p_result.get("success", true);
	
	if (status_icon) {
		status_icon->set_text(success ? "✓" : "✗");
	}
	
	if (result_label) {
		if (success) {
			// Show brief success info
			if (p_result.has("message")) {
				String msg = p_result["message"];
				result_label->set_text(msg.left(50) + (msg.length() > 50 ? "..." : ""));
			} else if (p_result.has("preview")) {
				result_label->set_text("[Result received]");
			} else {
				result_label->set_text("Done");
			}
			result_label->add_theme_color_override("font_color", Color(0.4, 0.7, 0.4));
		} else {
			String error = p_result.get("error", "Failed");
			result_label->set_text(error.left(50));
			result_label->add_theme_color_override("font_color", Color(0.8, 0.4, 0.4));
		}
	}
	
	// Update panel style based on success
	Ref<StyleBoxFlat> style = memnew(StyleBoxFlat);
	if (success) {
		style->set_bg_color(Color(0.2, 0.3, 0.2, 0.3));
		style->set_border_width_all(1);
		style->set_border_color(Color(0.3, 0.5, 0.3, 0.5));
	} else {
		style->set_bg_color(Color(0.3, 0.2, 0.2, 0.3));
		style->set_border_width_all(1);
		style->set_border_color(Color(0.5, 0.3, 0.3, 0.5));
	}
	style->set_content_margin_all(6 * EDSCALE);
	style->set_corner_radius_all(3 * EDSCALE);
	placeholder->add_theme_style_override("panel", style);
}

void ExplorerPanel::_append_thinking_text(const String &p_text) {
	accumulated_thinking_text += p_text;
	
	if (!thinking_label->is_visible()) {
		thinking_label->set_visible(true);
	}
	
	// Use BBCode for basic formatting
	thinking_label->clear();
	thinking_label->append_text("[color=#888888]" + accumulated_thinking_text + "[/color]");
	
	_scroll_to_bottom();
}

void ExplorerPanel::_display_report(const Dictionary &p_report_data) {
	String report_text = p_report_data.get("report", "");
	String confidence = p_report_data.get("confidence", "medium");
	Array files_explored = p_report_data.get("files_explored", Array());
	Array queries_used = p_report_data.get("search_queries_used", Array());
	bool incomplete = p_report_data.get("incomplete", false);
	
	// Update metadata labels
	String conf_color = "#88CC88";  // Green for high
	if (confidence == "medium") {
		conf_color = "#CCCC88";  // Yellow
	} else if (confidence == "low") {
		conf_color = "#CC8888";  // Red
	}
	confidence_label->set_text("Confidence: " + confidence.capitalize());
	files_explored_label->set_text("Files: " + String::num_int64(files_explored.size()));
	queries_used_label->set_text("Queries: " + String::num_int64(queries_used.size()));
	
	// Display report content with BBCode formatting
	report_content->clear();
	
	if (incomplete) {
		report_content->append_text("[color=#CC8888][b]⚠️ Exploration Incomplete[/b][/color]\n\n");
	}
	
	// Convert markdown-style headers and code blocks to BBCode
	String formatted_report = report_text;
	
	// Headers
	formatted_report = formatted_report.replace("## ", "[b]");
	formatted_report = formatted_report.replace("### ", "[b]");
	
	// Code blocks (simplified)
	formatted_report = formatted_report.replace("```gdscript", "[code]");
	formatted_report = formatted_report.replace("```", "[/code]");
	
	// Bold
	formatted_report = formatted_report.replace("**", "[b]");
	
	// File paths (make them stand out)
	// This is a simplified approach - ideally we'd use regex
	
	report_content->append_text(formatted_report);
	
	// Show report panel
	report_panel->set_visible(true);
	
	_scroll_to_bottom();
	
	print_line("EXPLORER: Report displayed - " + String::num_int64(report_text.length()) + " chars");
}

void ExplorerPanel::_scroll_to_bottom() {
	if (output_scroll) {
		// Defer to ensure layout is complete
		callable_mp(output_scroll, &ScrollContainer::set_v_scroll).call_deferred(9999999);
	}
}

