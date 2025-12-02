/*
 * © 2025 Simplifine Corp.
 * Explorer Panel - Deep codebase investigation UI component
 * Personal Non-Commercial License applies.
 */
#pragma once

#include "scene/gui/panel_container.h"
#include "scene/gui/box_container.h"
#include "scene/gui/label.h"
#include "scene/gui/button.h"
#include "scene/gui/rich_text_label.h"
#include "scene/gui/scroll_container.h"
#include "scene/gui/progress_bar.h"
#include "scene/gui/text_edit.h"
#include "core/io/http_client.h"
#include "core/templates/hash_map.h"

class HTTPRequest;

/**
 * ExplorerPanel - UI component for the Explorer agent
 * 
 * This panel displays the progress and results of deep codebase exploration.
 * It shows:
 * - Current exploration status
 * - Tool calls being made by the explorer
 * - Streaming text output
 * - Final exploration report with citations
 */
class ExplorerPanel : public PanelContainer {
	GDCLASS(ExplorerPanel, PanelContainer);

private:
	// Main layout
	VBoxContainer *main_vbox = nullptr;
	
	// Header section
	HBoxContainer *header_hbox = nullptr;
	Label *title_label = nullptr;
	Label *status_label = nullptr;
	Button *close_button = nullptr;
	
	// Progress section
	HBoxContainer *progress_hbox = nullptr;
	ProgressBar *progress_bar = nullptr;
	Label *turn_label = nullptr;
	
	// Question display
	PanelContainer *question_panel = nullptr;
	Label *question_label = nullptr;
	
	// Explorer output area (scrollable)
	ScrollContainer *output_scroll = nullptr;
	VBoxContainer *output_vbox = nullptr;
	
	// Streaming text accumulator
	RichTextLabel *thinking_label = nullptr;
	String accumulated_thinking_text;
	
	// Tool calls display
	VBoxContainer *tools_section = nullptr;
	HashMap<String, PanelContainer *> tool_placeholders;
	
	// Final report section
	PanelContainer *report_panel = nullptr;
	VBoxContainer *report_vbox = nullptr;
	RichTextLabel *report_content = nullptr;
	HBoxContainer *report_metadata_hbox = nullptr;
	Label *confidence_label = nullptr;
	Label *files_explored_label = nullptr;
	Label *queries_used_label = nullptr;
	
	// Exploration state
	String current_exploration_id;
	String current_question;
	bool is_exploring = false;
	int current_turn = 0;
	int max_turns = 10;
	
	// HTTP connection for streaming
	HTTPRequest *http_request = nullptr;
	String response_buffer;
	String backend_url;
	
	// Internal methods
	void _setup_ui();
	void _style_panel();
	void _clear_output();
	void _add_tool_call_placeholder(const String &p_tool_id, const String &p_tool_name);
	void _update_tool_result(const String &p_tool_id, const String &p_tool_name, const Dictionary &p_result);
	void _append_thinking_text(const String &p_text);
	void _display_report(const Dictionary &p_report_data);
	void _process_ndjson_line(const String &p_line);
	void _scroll_to_bottom();
	
	// Signal handlers
	void _on_close_pressed();
	void _on_http_request_completed(int p_result, int p_response_code, const PackedStringArray &p_headers, const PackedByteArray &p_body);
	void _on_data_received(const PackedByteArray &p_chunk);

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	ExplorerPanel();
	~ExplorerPanel();
	
	// Public API
	void start_exploration(const String &p_question, const String &p_depth = "normal", const Array &p_focus_areas = Array());
	void stop_exploration();
	void set_backend_url(const String &p_url);
	bool is_currently_exploring() const { return is_exploring; }
	String get_current_question() const { return current_question; }
	
	// Project context
	void set_project_context(const Dictionary &p_context);
	
	// Signals
	// exploration_started(exploration_id: String)
	// exploration_completed(exploration_id: String, report: Dictionary)
	// exploration_error(error: String)
};

