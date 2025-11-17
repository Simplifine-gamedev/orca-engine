#pragma once

#include "scene/gui/popup.h"
#include "core/object/object.h"

class Button;
class LineEdit;
class Label;
class VBoxContainer;
class HTTPRequest;

class AIChatTodoPanel : public Popup {
	GDCLASS(AIChatTodoPanel, Popup);

public:
	AIChatTodoPanel();

	void set_api_base(const String &p_base);
	void set_auth_token(const String &p_token);
	void set_project_root(const String &p_root);

	void open_panel();
	void refresh_todos();

protected:
	static void _bind_methods();

private:
	HTTPRequest *request = nullptr;
	LineEdit *input_field = nullptr;
	Button *add_button = nullptr;
	Button *refresh_button = nullptr;
	Button *clear_button = nullptr;
	Label *status_label = nullptr;
	VBoxContainer *list_container = nullptr;

	String api_base;
	String todo_endpoint;
	String auth_token;
	String project_root;
	String pending_op;
	bool request_in_flight = false;

	void _update_endpoint();
	void _set_ui_enabled(bool p_enabled);
	void _send_request(const Dictionary &p_payload);
	void _on_request_completed(int p_result, int p_code, const PackedStringArray &p_headers, const PackedByteArray &p_body);

	void _on_add_todo_pressed();
	void _on_clear_button_pressed();
	void _on_status_button_pressed(const String &p_todo_id, const String &p_status);
	void _on_remove_todo_pressed(const String &p_todo_id);

	void _render_todos(const Array &p_todos);
	Button *_create_status_button(const String &p_label, const String &p_status, const String &p_todo_id);
};

