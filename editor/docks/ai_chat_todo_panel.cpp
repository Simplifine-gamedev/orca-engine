#include "ai_chat_todo_panel.h"

#include "core/io/json.h"
#include "core/string/string_name.h"
#include "scene/gui/box_container.h"
#include "scene/gui/button.h"
#include "scene/gui/label.h"
#include "scene/gui/line_edit.h"
#include "scene/gui/scroll_container.h"
#include "scene/gui/panel_container.h"
#include "scene/main/http_request.h"
#include "core/config/project_settings.h"

void AIChatTodoPanel::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_on_add_todo_pressed"), &AIChatTodoPanel::_on_add_todo_pressed);
	ClassDB::bind_method(D_METHOD("_on_clear_button_pressed"), &AIChatTodoPanel::_on_clear_button_pressed);
	ClassDB::bind_method(D_METHOD("_on_status_button_pressed", "todo_id", "status"), &AIChatTodoPanel::_on_status_button_pressed);
	ClassDB::bind_method(D_METHOD("_on_remove_todo_pressed", "todo_id"), &AIChatTodoPanel::_on_remove_todo_pressed);
	ClassDB::bind_method(D_METHOD("_on_request_completed"), &AIChatTodoPanel::_on_request_completed);
}

AIChatTodoPanel::AIChatTodoPanel() {
	set_title("AI TODOs");
	set_size(Size2i(520, 520));

	VBoxContainer *root = memnew(VBoxContainer);
	root->set_anchors_preset(Control::PRESET_FULL_RECT);
	root->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	root->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	add_child(root);

	HBoxContainer *header = memnew(HBoxContainer);
	root->add_child(header);

	refresh_button = memnew(Button);
	refresh_button->set_text("Refresh");
	refresh_button->connect("pressed", callable_mp(this, &AIChatTodoPanel::refresh_todos));
	header->add_child(refresh_button);

	clear_button = memnew(Button);
	clear_button->set_text("Clear");
	clear_button->set_tooltip_text("Clear all todos for this project");
	clear_button->connect("pressed", callable_mp(this, &AIChatTodoPanel::_on_clear_button_pressed));
	header->add_child(clear_button);

	status_label = memnew(Label);
	status_label->set_text("Plan after semantic search + graph walk.");
	root->add_child(status_label);

	HBoxContainer *add_row = memnew(HBoxContainer);
	root->add_child(add_row);

	input_field = memnew(LineEdit);
	input_field->set_placeholder("Describe the next coding step…");
	input_field->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	add_row->add_child(input_field);

	add_button = memnew(Button);
	add_button->set_text("Add");
	add_button->connect("pressed", callable_mp(this, &AIChatTodoPanel::_on_add_todo_pressed));
	add_row->add_child(add_button);

	ScrollContainer *scroll = memnew(ScrollContainer);
	scroll->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	scroll->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	scroll->set_horizontal_scroll_mode(ScrollContainer::SCROLL_MODE_DISABLED);
	root->add_child(scroll);

	list_container = memnew(VBoxContainer);
	list_container->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	scroll->add_child(list_container);

	request = memnew(HTTPRequest);
	add_child(request);
	request->connect("request_completed", callable_mp(this, &AIChatTodoPanel::_on_request_completed));
}

void AIChatTodoPanel::set_api_base(const String &p_base) {
	if (p_base == api_base) {
		return;
	}
	api_base = p_base;
	_update_endpoint();
}

void AIChatTodoPanel::set_auth_token(const String &p_token) {
	auth_token = p_token;
}

void AIChatTodoPanel::set_project_root(const String &p_root) {
	project_root = p_root;
}

void AIChatTodoPanel::_update_endpoint() {
	if (api_base.is_empty()) {
		todo_endpoint = "";
		return;
	}
	if (api_base.ends_with("/chat")) {
		todo_endpoint = api_base.trim_suffix("/chat") + "/todo_manager";
	} else {
		todo_endpoint = api_base + "/todo_manager";
	}
}

void AIChatTodoPanel::open_panel() {
	popup_centered(Size2i(520, 520));
	refresh_todos();
	if (input_field && input_field->is_inside_tree()) {
		input_field->grab_focus();
	}
}

void AIChatTodoPanel::refresh_todos() {
	Dictionary payload;
	payload["op"] = "todo.list";
	_send_request(payload);
}

void AIChatTodoPanel::_set_ui_enabled(bool p_enabled) {
	if (input_field) {
		input_field->set_editable(p_enabled);
	}
	if (add_button) {
		add_button->set_disabled(!p_enabled);
	}
	if (refresh_button) {
		refresh_button->set_disabled(!p_enabled);
	}
	if (clear_button) {
		clear_button->set_disabled(!p_enabled);
	}
}

void AIChatTodoPanel::_send_request(const Dictionary &p_payload) {
	if (!request || todo_endpoint.is_empty()) {
		status_label->set_text("Todo endpoint unavailable.");
		return;
	}
	if (request_in_flight) {
		return;
	}

	Dictionary payload = p_payload.duplicate();
	if (!project_root.is_empty()) {
		payload["project_root"] = project_root;
	}

	pending_op = String(payload.get("op", ""));

	Ref<JSON> json;
	json.instantiate();
	String body = json->stringify(payload);

	PackedStringArray headers;
	headers.push_back("Content-Type: application/json");
	if (!auth_token.is_empty()) {
		headers.push_back("Authorization: Bearer " + auth_token);
	}
	if (!project_root.is_empty()) {
		headers.push_back("X-Project-Root: " + project_root);
	}

	Error err = request->request(todo_endpoint, headers, HTTPClient::METHOD_POST, body);
	if (err != OK) {
		status_label->set_text("Failed to contact todo manager.");
		return;
	}

	request_in_flight = true;
	_set_ui_enabled(false);
}

void AIChatTodoPanel::_on_request_completed(int p_result, int p_code, const PackedStringArray &p_headers, const PackedByteArray &p_body) {
	request_in_flight = false;
	_set_ui_enabled(true);

	if (p_code != 200) {
		status_label->set_text("Todo manager error: " + itos(p_code));
		return;
	}

	String text = String::utf8((const char *)p_body.ptr(), p_body.size());
	Ref<JSON> json;
	json.instantiate();
	if (json->parse(text) != OK) {
		status_label->set_text("Failed to parse todo response.");
		return;
	}
	Dictionary data = json->get_data();
	if (!data.get("success", false)) {
		status_label->set_text(data.get("error", "Todo operation failed"));
		return;
	}

	status_label->set_text("Todos synced.");
	if (pending_op == "todo.list") {
		Array todos = data.get("todos", Array());
		_render_todos(todos);
	} else {
		// Re-fetch list after mutations
		refresh_todos();
	}
}

void AIChatTodoPanel::_render_todos(const Array &p_todos) {
	if (!list_container) {
		return;
	}
	for (int i = list_container->get_child_count() - 1; i >= 0; i--) {
		Node *child = list_container->get_child(i);
		child->queue_free();
	}

	if (p_todos.is_empty()) {
		Label *empty = memnew(Label);
		empty->set_text("No todos yet. Gather context first, then add your plan.");
		list_container->add_child(empty);
		return;
	}

	for (int i = 0; i < p_todos.size(); i++) {
		Dictionary todo = p_todos[i];
		String todo_id = todo.get("id", "");
		String content = todo.get("content", "");
		String status = todo.get("status", "pending");

		PanelContainer *row_panel = memnew(PanelContainer);
		row_panel->add_theme_style_override("panel", get_theme_stylebox(SNAME("panel"), SNAME("Panel")));
		list_container->add_child(row_panel);

		HBoxContainer *row = memnew(HBoxContainer);
		row_panel->add_child(row);
		row->set_alignment(BoxContainer::ALIGNMENT_BEGIN);

		Label *status_badge = memnew(Label);
		status_badge->set_text(status.capitalize());
		status_badge->add_theme_color_override("font_color", get_theme_color(SNAME("accent_color"), SNAME("Editor")));
		row->add_child(status_badge);

		Label *content_label = memnew(Label);
		content_label->set_text(content);
		content_label->set_h_size_flags(Control::SIZE_EXPAND_FILL);
		row->add_child(content_label);

		HBoxContainer *actions = memnew(HBoxContainer);
		row->add_child(actions);

		actions->add_child(_create_status_button("Pending", "pending", todo_id));
		actions->add_child(_create_status_button("Start", "in_progress", todo_id));
		actions->add_child(_create_status_button("Complete", "completed", todo_id));

		Button *remove_button = memnew(Button);
		remove_button->set_text("Remove");
		remove_button->connect("pressed", callable_mp(this, &AIChatTodoPanel::_on_remove_todo_pressed).bind(todo_id));
		actions->add_child(remove_button);
	}
}

Button *AIChatTodoPanel::_create_status_button(const String &p_label, const String &p_status, const String &p_todo_id) {
	Button *btn = memnew(Button);
	btn->set_text(p_label);
	btn->set_tooltip_text("Set status to " + p_status);
	btn->connect("pressed", callable_mp(this, &AIChatTodoPanel::_on_status_button_pressed).bind(p_todo_id, p_status));
	return btn;
}

void AIChatTodoPanel::_on_add_todo_pressed() {
	if (!input_field) {
		return;
	}
	String content = input_field->get_text().strip_edges();
	if (content.is_empty()) {
		status_label->set_text("Enter a todo before adding.");
		return;
	}

	Dictionary payload;
	payload["op"] = "todo.add";
	payload["content"] = content;
	_send_request(payload);
	input_field->set_text("");
}

void AIChatTodoPanel::_on_clear_button_pressed() {
	Dictionary payload;
	payload["op"] = "todo.clear";
	_send_request(payload);
}

void AIChatTodoPanel::_on_status_button_pressed(const String &p_todo_id, const String &p_status) {
	Dictionary payload;
	payload["op"] = "todo.update";
	payload["todo_id"] = p_todo_id;
	payload["status"] = p_status;
	_send_request(payload);
}

void AIChatTodoPanel::_on_remove_todo_pressed(const String &p_todo_id) {
	Dictionary payload;
	payload["op"] = "todo.remove";
	payload["todo_id"] = p_todo_id;
	_send_request(payload);
}

