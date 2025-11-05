/*
 * © 2025 Simplifine Corp.
 * Personal Non‑Commercial License applies to this file as an original contribution to this Godot fork.
 * See LICENSES/COMPANY-NONCOMMERCIAL.md for terms. Commercial use requires a separate license from Simplifine.
 */

#include "ai_chat_attachment_ui.h"
#include "ai_chat_dock.h"
#include "editor/editor_string_names.h"

void AIChatAttachmentUI::create_file_attachment_ui(
	VBoxContainer *p_parent,
	const Array &p_attached_files,
	AIChatDock *p_chat_dock
) {
	if (!p_parent || !p_chat_dock || p_attached_files.is_empty()) {
		return;
	}

	VBoxContainer *files_container = memnew(VBoxContainer);
	p_parent->add_child(files_container);
	
	Label *files_header = memnew(Label);
	files_header->set_text("Attached Files:");
	files_header->add_theme_font_override("font", p_chat_dock->get_theme_font(SNAME("bold"), SNAME("EditorFonts")));
	files_header->add_theme_color_override("font_color", p_chat_dock->get_theme_color(SNAME("accent_color"), SNAME("Editor")));
	files_container->add_child(files_header);
	
	HFlowContainer *files_flow = memnew(HFlowContainer);
	files_container->add_child(files_flow);
	
	// Create file chips for each attached file
	for (int i = 0; i < p_attached_files.size(); i++) {
		Dictionary file_data = p_attached_files[i];
		String file_path = file_data.get("path", "");
		String file_name = file_data.get("name", "");
		
		if (!file_path.is_empty() && !file_name.is_empty()) {
			create_file_chip(files_flow, file_path, file_name, p_chat_dock);
		}
	}
}

void AIChatAttachmentUI::create_file_chip(
	HFlowContainer *p_container,
	const String &p_file_path,
	const String &p_file_name,
	AIChatDock *p_chat_dock
) {
	if (!p_container || !p_chat_dock) {
		return;
	}

	HBoxContainer *file_row = _create_file_row(p_file_path, p_file_name, p_chat_dock);
	p_container->add_child(file_row);
}

HBoxContainer *AIChatAttachmentUI::_create_file_row(const String &p_file_path, const String &p_file_name, AIChatDock *p_chat_dock) {
	HBoxContainer *file_row = memnew(HBoxContainer);
	
	Label *file_icon = _create_file_icon(p_chat_dock);
	file_row->add_child(file_icon);
	
	Button *file_link = _create_file_link(p_file_path, p_file_name, p_chat_dock);
	file_row->add_child(file_link);
	
	return file_row;
}

Label *AIChatAttachmentUI::_create_file_icon(AIChatDock *p_chat_dock) {
	Label *file_icon = memnew(Label);
	file_icon->add_theme_icon_override("icon", p_chat_dock->get_theme_icon(SNAME("File"), SNAME("EditorIcons")));
	return file_icon;
}

Button *AIChatAttachmentUI::_create_file_link(const String &p_file_path, const String &p_file_name, AIChatDock *p_chat_dock) {
	Button *file_link = memnew(Button);
	file_link->set_text(p_file_name);
	file_link->set_flat(true);
	file_link->set_text_alignment(HORIZONTAL_ALIGNMENT_LEFT);
	file_link->set_tooltip_text("Click to open: " + p_file_path);
	file_link->connect("pressed", callable_mp(p_chat_dock, &AIChatDock::_on_tool_file_link_pressed).bind(p_file_path));
	return file_link;
}
