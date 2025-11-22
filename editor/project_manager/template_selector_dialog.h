/**************************************************************************/
/*  template_selector_dialog.h                                           */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             ORCA ENGINE                                */
/**************************************************************************/

#ifndef TEMPLATE_SELECTOR_DIALOG_H
#define TEMPLATE_SELECTOR_DIALOG_H

#include "scene/gui/dialogs.h"
#include "core/io/json.h"

class Button;
class ScrollContainer;
class VBoxContainer;
class HBoxContainer;
class Label;
class LineEdit;
class ItemList;
class HSeparator;
class HTTPRequest;
class RichTextLabel;
class TextureRect;
class ScrollContainer;

struct TemplateInfo {
	String id;
	String name;
	String repo_url;
	String subdir;
	String license;
};

class TemplateSelectorDialog : public AcceptDialog {
	GDCLASS(TemplateSelectorDialog, AcceptDialog);

private:
	VBoxContainer *main_vbox = nullptr;
	VBoxContainer *content_vbox = nullptr;
	LineEdit *search_box = nullptr;
	ItemList *template_item_list = nullptr;
	HTTPRequest *http_request = nullptr;
	HTTPRequest *readme_request = nullptr;
	Label *status_label = nullptr;
	RichTextLabel *description_label = nullptr;
	VBoxContainer *image_container = nullptr;
	ScrollContainer *description_scroll = nullptr;
	
	Array templates;
	Array filtered_templates;
	
	String selected_template_id;
	bool is_loading = false;
	String current_readme_url;
	String base_description;
	Array image_urls;
	HashMap<HTTPRequest *, int> image_request_map;
	Array image_requests;
	
	String _get_backend_url();
	void _load_templates();
	void _load_templates_from_backend();
	void _load_templates_from_local();
	void _on_http_request_completed(int p_result, int p_response_code, const PackedStringArray &p_headers, const PackedByteArray &p_body);
	void _parse_templates_json(const String &p_json_text);
	void _on_search_changed(const String &p_text);
	void _on_template_selected(int p_index);
	void _on_template_double_clicked(int p_index);
	void _populate_template_list();
	void _update_description();
	void _fetch_readme(const String &p_repo_url);
	void _on_readme_request_completed(int p_result, int p_response_code, const PackedStringArray &p_headers, const PackedByteArray &p_body);
	String _extract_github_info(const String &p_repo_url, String &r_owner, String &r_repo);
	void _parse_readme_for_images(const String &p_readme_content, const String &p_repo_url);
	void _load_image(const String &p_image_url, int p_index);
	void _on_image_request_completed(int p_result, int p_response_code, const PackedStringArray &p_headers, const PackedByteArray &p_body);
	
	static void _bind_methods();

public:
	String get_selected_template_id() const { return selected_template_id; }
	Dictionary get_selected_template() const;
	
	TemplateSelectorDialog();
	~TemplateSelectorDialog();
};

#endif // TEMPLATE_SELECTOR_DIALOG_H

