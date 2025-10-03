/*
 * © 2025 Simplifine Corp.
 * Personal Non‑Commercial License applies to this file as an original contribution to this Godot fork.
 * See LICENSES/COMPANY-NONCOMMERCIAL.md for terms. Commercial use requires a separate license from Simplifine.
 */

#include "ai_chat_input_box.h"

#include "ai_chat_dock.h"
#include "scene/gui/button.h"
#include "scene/gui/margin_container.h"
#include "scene/gui/panel_container.h"
#include "scene/gui/text_edit.h"
#include "scene/resources/style_box_flat.h"
#include "editor/settings/editor_settings.h"

void AIChatInputBox::create_input_ui(AIChatDock *p_chat_dock, VBoxContainer *p_parent_container) {
	// Wrapper for input panel (adds margin from edges)
	MarginContainer *input_wrapper = memnew(MarginContainer);
	input_wrapper->add_theme_constant_override("margin_left", 8);
	input_wrapper->add_theme_constant_override("margin_right", 8);
	input_wrapper->add_theme_constant_override("margin_top", 8);
	input_wrapper->add_theme_constant_override("margin_bottom", 8);
	p_parent_container->add_child(input_wrapper);

	// Panel container for the input field + buttons
	PanelContainer *input_panel = memnew(PanelContainer);
	input_wrapper->add_child(input_panel);
	
	// Always visible border style for the chatbox
	Ref<StyleBoxFlat> panel_style = memnew(StyleBoxFlat);
	panel_style->set_bg_color(p_chat_dock->get_theme_color(SNAME("dark_color_1"), SNAME("Editor")));
	panel_style->set_border_width_all(2);
	panel_style->set_border_color(p_chat_dock->get_theme_color(SNAME("font_color"), SNAME("Editor")) * Color(1, 1, 1, 0.8));
	panel_style->set_corner_radius_all(8);
	input_panel->add_theme_style_override("panel", panel_style);

	// Container inside the panel for text field and send button
	VBoxContainer *input_content = memnew(VBoxContainer);
	input_panel->add_child(input_content);

	p_chat_dock->input_field = memnew(TextEdit);
			
	// Enable word wrapping and disable auto-height expansion
	p_chat_dock->input_field->set_line_wrapping_mode(TextEdit::LINE_WRAPPING_BOUNDARY);
	p_chat_dock->input_field->set_fit_content_height_enabled(false);
	
	// Remove border from input field since panel provides it
	Ref<StyleBoxFlat> input_style = memnew(StyleBoxFlat);
	input_style->set_bg_color(Color(0, 0, 0, 0)); // Transparent background
	input_style->set_border_width_all(0);
	input_style->set_corner_radius_all(0);
	input_style->set_content_margin_all(8);
	p_chat_dock->input_field->add_theme_style_override("normal", input_style);
	p_chat_dock->input_field->add_theme_style_override("focus", input_style);
	
	p_chat_dock->input_field->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	p_chat_dock->input_field->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	p_chat_dock->input_field->set_placeholder("Ask me anything about Orca...");
	p_chat_dock->input_field->set_custom_minimum_size(Size2(0, 150));
	p_chat_dock->input_field->connect("text_changed", callable_mp(p_chat_dock, &AIChatDock::_on_input_text_changed));
	p_chat_dock->input_field->connect("gui_input", callable_mp(p_chat_dock, &AIChatDock::_on_input_field_gui_input));
	input_content->add_child(p_chat_dock->input_field);

	// Send button container positioned at bottom right with margin
	MarginContainer *button_margin = memnew(MarginContainer);
	button_margin->add_theme_constant_override("margin_right", 8);
	button_margin->add_theme_constant_override("margin_bottom", 8);
	input_content->add_child(button_margin);

	HBoxContainer *button_row = memnew(HBoxContainer);
	button_row->set_alignment(BoxContainer::ALIGNMENT_END);
	button_margin->add_child(button_row);

	// Send button with Enter symbol
	p_chat_dock->send_button = memnew(Button);
	p_chat_dock->send_button->set_text("↵");
	p_chat_dock->send_button->set_disabled(true);
	p_chat_dock->send_button->set_custom_minimum_size(Size2(28, 28));
	p_chat_dock->send_button->set_tooltip_text("Send (Enter)");
	
	style_send_button(p_chat_dock->send_button, p_chat_dock);
	
	p_chat_dock->send_button->connect("pressed", callable_mp(p_chat_dock, &AIChatDock::_on_send_button_pressed));
	button_row->add_child(p_chat_dock->send_button);

	// Stop button (initially hidden)
	p_chat_dock->stop_button = memnew(Button);
	p_chat_dock->stop_button->set_text("");
	p_chat_dock->stop_button->set_visible(false);
	p_chat_dock->stop_button->add_theme_icon_override("icon", p_chat_dock->get_theme_icon(SNAME("Stop"), SNAME("EditorIcons")));
	p_chat_dock->stop_button->set_custom_minimum_size(Size2(24, 24));
	p_chat_dock->stop_button->set_tooltip_text("Stop");
	
	style_stop_button(p_chat_dock->stop_button, p_chat_dock, false);

	p_chat_dock->stop_button->connect("pressed", callable_mp(p_chat_dock, &AIChatDock::_on_stop_button_pressed));
	button_row->add_child(p_chat_dock->stop_button);
}

void AIChatInputBox::style_send_button(Button *p_send_button, AIChatDock *p_chat_dock) {
	// Normal style - sleek, rounded
	Ref<StyleBoxFlat> button_style = memnew(StyleBoxFlat);
	button_style->set_bg_color(p_chat_dock->get_theme_color(SNAME("accent_color"), SNAME("Editor")));
	button_style->set_corner_radius_all(8); // Softer corners
	button_style->set_content_margin_all(6);
	p_send_button->add_theme_style_override("normal", button_style);
	
	// Hover style
	Ref<StyleBoxFlat> button_hover_style = memnew(StyleBoxFlat);
	button_hover_style->set_bg_color(p_chat_dock->get_theme_color(SNAME("accent_color"), SNAME("Editor")) * Color(1.15, 1.15, 1.15));
	button_hover_style->set_corner_radius_all(8);
	button_hover_style->set_content_margin_all(6);
	p_send_button->add_theme_style_override("hover", button_hover_style);
	
	// Disabled style
	Ref<StyleBoxFlat> button_disabled_style = memnew(StyleBoxFlat);
	button_disabled_style->set_bg_color(p_chat_dock->get_theme_color(SNAME("disabled_bg_color"), SNAME("Editor")));
	button_disabled_style->set_corner_radius_all(8);
	button_disabled_style->set_content_margin_all(6);
	p_send_button->add_theme_style_override("disabled", button_disabled_style);
}

void AIChatInputBox::style_stop_button(Button *p_stop_button, AIChatDock *p_chat_dock, bool p_enabled) {
	// White background when can use (enabled)
	Ref<StyleBoxFlat> stop_button_style = memnew(StyleBoxFlat);
	stop_button_style->set_bg_color(Color(1.0, 1.0, 1.0)); // White
	stop_button_style->set_corner_radius_all(8); // Softer corners
	stop_button_style->set_content_margin_all(6);
	p_stop_button->add_theme_style_override("normal", stop_button_style);
	
	// Hover style - slightly darker white
	Ref<StyleBoxFlat> stop_button_hover_style = memnew(StyleBoxFlat);
	stop_button_hover_style->set_bg_color(Color(0.95, 0.95, 0.95));
	stop_button_hover_style->set_corner_radius_all(8);
	stop_button_hover_style->set_content_margin_all(6);
	p_stop_button->add_theme_style_override("hover", stop_button_hover_style);
	
	// Dark gray when can't use (disabled)
	Ref<StyleBoxFlat> stop_button_disabled_style = memnew(StyleBoxFlat);
	stop_button_disabled_style->set_bg_color(Color(0.3, 0.3, 0.3)); // Dark gray
	stop_button_disabled_style->set_corner_radius_all(8);
	stop_button_disabled_style->set_content_margin_all(6);
	p_stop_button->add_theme_style_override("disabled", stop_button_disabled_style);
}

